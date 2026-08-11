"""Train vanilla diffusion policy on demos (PLAN Stage 2, step 6)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from data.dataset import DemoDataset
from models.temporal_unet import TemporalUNet
from models.diffusion_policy import DiffusionConfig, DiffusionPolicy
from eval.rollout import aggregate, rollout_episode
from trainers.utils import TSVLogger, results_dir, set_global_seed


def maybe_eval(policy_model: DiffusionPolicy, env_cfg: str, n_seeds: int, base_seed: int):
    env = UAVEnv(config=env_cfg)
    rollout_policy = policy_model.make_rollout_policy()
    results = [rollout_episode(env, rollout_policy, seed=base_seed + i) for i in range(n_seeds)]
    return aggregate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_diffusion.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-during-train", action="store_true",
                        help="Run mid-training rollouts (slower).")
    parser.add_argument("--eval-n-seeds", type=int, default=20)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or cfg["run_name"]
    epochs = int(args.epochs or cfg["epochs"])
    seed = int(cfg["seed"])
    set_global_seed(seed)

    demos_path = cfg["demos_path"]
    if not Path(demos_path).is_absolute():
        demos_path = str((ROOT / demos_path).resolve())
    env_cfg_path = cfg["env_config"]
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_float32_matmul_precision("high")

    H = int(cfg["action_horizon"])
    frame_stack = int(cfg.get("frame_stack", 1))
    dset = DemoDataset(demos_path, horizon=H, frame_stack=frame_stack)
    loader = DataLoader(
        dset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=True,
        persistent_workers=False,
    )

    state_dim = dset.s.shape[1]
    action_dim = dset.u.shape[1]
    latent_dim = int(cfg.get("latent_dim", 0))
    effective_state_dim = state_dim * frame_stack

    # Tile per-frame normalization stats to match the stacked conditioning vector.
    tiled_mean = np.tile(dset.state_mean, frame_stack).astype(np.float32)
    tiled_std = np.tile(dset.state_std, frame_stack).astype(np.float32)

    denoiser = TemporalUNet(
        action_dim=action_dim,
        cond_input_dim=effective_state_dim,
        latent_dim=latent_dim,
        hidden=int(cfg["hidden_dim"]),
    )
    dcfg = DiffusionConfig(
        action_dim=action_dim,
        horizon=H,
        cond_dim=effective_state_dim,
        latent_dim=latent_dim,
        hidden=int(cfg["hidden_dim"]),
        T_train=int(cfg["diffusion_steps_train"]),
        T_sample=int(cfg["diffusion_steps_sample"]),
        exec_k=int(cfg.get("exec_k", 4)),
        frame_stack=frame_stack,
    )
    policy = DiffusionPolicy(denoiser, dcfg, state_mean=tiled_mean, state_std=tiled_std, device=device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=float(cfg["lr"]), weight_decay=1e-6)

    out_dir = results_dir(run_name, root=ROOT / "results")
    logger = TSVLogger(
        path=out_dir / "train_log.tsv",
        columns=["epoch", "step", "loss", "eval_success", "eval_ep_len", "eval_delta_u2"],
    )

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"[train_diffusion] run={run_name} device={device} "
          f"H={H} T_train={dcfg.T_train} T_sample={dcfg.T_sample} params={n_params/1e6:.2f}M")
    print(f"[train_diffusion] dataset: {len(dset)} windows, batches/epoch={len(loader)}")

    t_start = time.time()
    global_step = 0
    eval_every = max(1, int(cfg.get("eval_every_epochs", 0)))
    eval_seeds = int(args.eval_n_seeds)

    for epoch in range(epochs):
        denoiser.train()
        losses = []
        for batch in loader:
            state = batch["state"].to(device, non_blocking=True)
            chunk = batch["action_chunk"].to(device, non_blocking=True)
            loss = policy.training_loss(chunk, state)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            global_step += 1
        ep_loss = float(np.mean(losses))
        row = {"epoch": epoch, "step": global_step, "loss": ep_loss,
               "eval_success": "", "eval_ep_len": "", "eval_delta_u2": ""}

        do_eval = args.eval_during_train and eval_every > 0 and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        if do_eval:
            denoiser.eval()
            m = maybe_eval(policy, env_cfg_path, n_seeds=eval_seeds, base_seed=500_000 + epoch)
            row.update({
                "eval_success": f"{m['success_rate']:.3f}",
                "eval_ep_len": f"{m['ep_length_mean']:.1f}",
                "eval_delta_u2": f"{m['delta_u2_mean']:.4f}",
            })
            print(f"  epoch={epoch:03d} loss={ep_loss:.4f} "
                  f"eval_succ={m['success_rate']:.3f} len={m['ep_length_mean']:.1f}")
        else:
            print(f"  epoch={epoch:03d} loss={ep_loss:.4f}")

        logger.log(row)

    elapsed = time.time() - t_start

    # Save checkpoint (weights + config + norm stats)
    ckpt_path = out_dir / "diffusion_policy.pt"
    torch.save({
        "model_state": denoiser.state_dict(),
        "state_mean": tiled_mean,
        "state_std": tiled_std,
        "config": {
            "action_dim": action_dim,
            "state_dim": effective_state_dim,     # total cond input dim (state_dim * frame_stack)
            "per_frame_state_dim": state_dim,
            "hidden": int(cfg["hidden_dim"]),
            "latent_dim": latent_dim,
            "horizon": H,
            "T_train": dcfg.T_train,
            "T_sample": dcfg.T_sample,
            "exec_k": dcfg.exec_k,
            "frame_stack": frame_stack,
        },
    }, ckpt_path)

    # Final eval (always, 30 seeds)
    denoiser.eval()
    final_metrics = maybe_eval(policy, env_cfg_path, n_seeds=30, base_seed=900_000)
    summary = {
        "run_name": run_name,
        "epochs": epochs,
        "wall_time_sec": elapsed,
        "n_params_M": n_params / 1e6,
        "final_eval_n": final_metrics["n"],
        "final_eval_success_rate": final_metrics["success_rate"],
        "final_eval_collision_rate": final_metrics["collision_rate"],
        "final_eval_final_dist_mean": final_metrics["final_dist_mean"],
        "final_eval_ep_length_mean": final_metrics["ep_length_mean"],
        "final_eval_delta_u2_mean": final_metrics["delta_u2_mean"],
        "final_eval_return_mean": final_metrics["return_mean"],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ls = []
        with open(out_dir / "train_log.tsv", "r", newline="") as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for row in r:
                if row.get("loss"):
                    ls.append(float(row["loss"]))
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ls); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(f"{run_name} — diffusion epsilon-MSE")
        fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"[train_diffusion] plot skipped: {e}")

    print(f"[train_diffusion] done in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
