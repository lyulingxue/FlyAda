"""Train FlyAda: diffusion policy + adaptation head on mixed nominal + perturbed demos.

The FlyAda reformulation uses real perturbed demos (collected by PPO teacher under
sampled dynamics) rather than synthetic replay of nominal actions under perturbed
dynamics. This ensures every training pair (s_t, u_t) is consistent with SOME
dynamics, giving f_phi a well-posed signal to extract the perturbation latent.
"""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from data.episodes import global_state_stats, load_episodes
from models.adaptation_head import AdaptationHead, roll_latent
from models.diffusion_policy import DiffusionConfig, DiffusionPolicy
from models.temporal_unet import TemporalUNet
from eval.rollout import aggregate, rollout_episode
from trainers.utils import TSVLogger, results_dir, set_global_seed


def sample_episode_batch(episodes, rng: np.random.Generator, batch_size: int, max_len: int):
    """Draw B episodes, clip each to max_len, right-pad by repeating the last action/state.

    Returns:
        states [B, max_len+1, state_dim]  (full states, velocity present)
        actions [B, max_len, action_dim]
        lens [B]
    """
    idx = rng.integers(0, len(episodes), size=batch_size)
    state_dim = episodes[0]["s"].shape[1]
    action_dim = episodes[0]["u"].shape[1]
    S = np.zeros((batch_size, max_len + 1, state_dim), dtype=np.float32)
    U = np.zeros((batch_size, max_len, action_dim), dtype=np.float32)
    lens = np.zeros(batch_size, dtype=np.int64)
    for b, i in enumerate(idx):
        ep = episodes[i]
        T = min(ep["T"], max_len)
        S[b, :T] = ep["s"][:T]
        S[b, T:] = ep["s"][T - 1]
        U[b, :T] = ep["u"][:T]
        if T < max_len:
            U[b, T:] = ep["u"][T - 1]
        lens[b] = T
    return S, U, lens


def mask_velocity(states: torch.Tensor) -> torch.Tensor:
    """Return states with velocity dims [3:6] zeroed (partial-obs masking)."""
    out = states.clone()
    out[..., 3:6] = 0.0
    return out


class FlyAdaRolloutPolicy:
    """Stateful wrapper used by eval/rollout.py: maintains z_t across env steps."""

    def __init__(
        self,
        diffusion: DiffusionPolicy,
        f_phi: AdaptationHead,
        alpha: float = 0.1,
        freeze_after_steps: int = -1,
        update_mode: str = "ema",
    ):
        self.dp = diffusion
        self.f_phi = f_phi.to(diffusion.device).eval()
        self.alpha = float(alpha)
        self.freeze_after_steps = int(freeze_after_steps)
        self.update_mode = str(update_mode)
        self.reset()

    def reset(self):
        self._z = torch.zeros(1, self.f_phi.latent_dim, device=self.dp.device)
        self._last_state = None
        self._last_action = None
        self._chunk = None
        self._pos = 0
        self._env_step = 0

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.dp.device).unsqueeze(0)

        if self._last_state is not None:
            update_ok = (
                self.alpha > 0.0
                and (self.freeze_after_steps < 0 or self._env_step < self.freeze_after_steps)
            )
            if update_ok:
                dz = self.f_phi(self._last_state, self._last_action, state)
                if self.update_mode == "ema":
                    self._z = (1.0 - self.alpha) * self._z + self.alpha * dz
                else:
                    self._z = self._z + self.alpha * dz

        if self._chunk is None or self._pos >= self.dp.cfg.exec_k:
            chunk = self.dp.sample(state, latent=self._z)
            self._chunk = chunk[0].cpu().numpy()
            self._pos = 0

        action = self._chunk[self._pos]
        self._pos += 1
        self._env_step += 1
        self._last_state = state
        self._last_action = torch.as_tensor(action, dtype=torch.float32, device=self.dp.device).unsqueeze(0)
        return action.astype(np.float32), None


def nominal_eval(policy_factory, env_cfg: str, n_seeds: int, base_seed: int):
    env = UAVEnv(config=env_cfg)
    results = [rollout_episode(env, policy_factory(), seed=base_seed + i) for i in range(n_seeds)]
    return aggregate(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_flyada.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--eval-during-train", action="store_true")
    parser.add_argument("--eval-n-seeds", type=int, default=12)
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

    nominal_path = cfg["demos_path"]
    perturbed_path = cfg["demos_perturbed_path"]
    if not Path(nominal_path).is_absolute():
        nominal_path = str((ROOT / nominal_path).resolve())
    if not Path(perturbed_path).is_absolute():
        perturbed_path = str((ROOT / perturbed_path).resolve())
    env_cfg_path = cfg["env_config"]
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_float32_matmul_precision("high")

    H = int(cfg["action_horizon"])
    latent_dim = int(cfg["latent_dim"])
    alpha = float(cfg["alpha"])
    update_mode = str(cfg.get("update_mode", "ema"))
    z_dropout = float(cfg.get("z_dropout", 0.0))
    hidden = int(cfg["hidden_dim"])
    adaptation_hidden = int(cfg["adaptation_hidden"])
    partial_obs = bool(cfg.get("partial_obs", False))
    vel_aux_coeff = float(cfg.get("vel_aux_coeff", 0.0))

    nominal_eps = load_episodes(nominal_path)
    perturbed_eps = load_episodes(perturbed_path)
    episodes = nominal_eps + perturbed_eps

    state_dim = episodes[0]["s"].shape[1]
    action_dim = episodes[0]["u"].shape[1]
    # Use stats across the merged pool for consistency with test-time distribution.
    all_states = np.concatenate([ep["s"] for ep in episodes], axis=0)
    mean = all_states.mean(axis=0).astype(np.float32)
    std = all_states.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-3, 1.0, std).astype(np.float32)

    max_ep_len = max(ep["T"] for ep in episodes)
    print(f"[train_flyada] episodes: nominal={len(nominal_eps)} perturbed={len(perturbed_eps)} "
          f"total={len(episodes)} max_len={max_ep_len}")

    denoiser = TemporalUNet(
        action_dim=action_dim,
        cond_input_dim=state_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    )
    f_phi = AdaptationHead(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden=adaptation_hidden,
    ).to(device)

    dcfg = DiffusionConfig(
        action_dim=action_dim,
        horizon=H,
        cond_dim=state_dim,
        latent_dim=latent_dim,
        hidden=hidden,
        T_train=int(cfg["diffusion_steps_train"]),
        T_sample=int(cfg["diffusion_steps_sample"]),
        exec_k=int(cfg.get("exec_k", 4)),
    )
    policy = DiffusionPolicy(denoiser, dcfg, state_mean=mean, state_std=std, device=device)

    params = list(denoiser.parameters()) + list(f_phi.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=1e-6)
    n_params = sum(p.numel() for p in params)

    out_dir = results_dir(run_name, root=ROOT / "results")
    logger = TSVLogger(
        path=out_dir / "train_log.tsv",
        columns=["epoch", "step", "loss", "z_norm_mean", "eval_success", "eval_ep_len"],
    )

    B = int(cfg["batch_size"])
    train_max_len = int(cfg.get("train_max_len", min(max_ep_len, 80)))
    steps_per_epoch = int(cfg.get("steps_per_epoch", 400))
    rng = np.random.default_rng(seed)

    print(f"[train_flyada] run={run_name} device={device} B={B} max_len={train_max_len} "
          f"steps/epoch={steps_per_epoch} alpha={alpha} z_dropout={z_dropout} "
          f"T_train={dcfg.T_train} params={n_params/1e6:.2f}M")

    t_start = time.time()
    global_step = 0
    eval_every = int(cfg.get("eval_every_epochs", 0))
    eval_seeds = int(args.eval_n_seeds)

    for epoch in range(epochs):
        denoiser.train(); f_phi.train()
        losses = []
        z_norms = []

        for _ in range(steps_per_epoch):
            S_np, U_np, lens_np = sample_episode_batch(episodes, rng, B, train_max_len)
            full_states = torch.from_numpy(S_np).to(device)   # full obs (vel present)
            actions = torch.from_numpy(U_np).to(device)
            lens = torch.from_numpy(lens_np).to(device)

            # Under partial_obs, f_phi sees velocity-masked states and the policy
            # conditions on velocity-masked states. True velocity is used only as
            # an auxiliary supervision target on z.
            obs_states = mask_velocity(full_states) if partial_obs else full_states
            true_vel = full_states[..., 3:6]                  # [B, L+1, 3]

            zs = roll_latent(f_phi, obs_states, actions, alpha=alpha, mode=update_mode)

            max_start = (lens - H).clamp(min=0)
            u = torch.rand(B, device=device)
            t_idx = (u * (max_start.float() + 1.0)).long().clamp(max=train_max_len - H)

            batch_arange = torch.arange(B, device=device)
            state_at_t = obs_states[batch_arange, t_idx]
            z_at_t = zs[batch_arange, t_idx]
            chunk_idx = t_idx.unsqueeze(1) + torch.arange(H, device=device).unsqueeze(0)
            action_chunk = actions.gather(1, chunk_idx.unsqueeze(-1).expand(-1, -1, action_dim))

            if z_dropout > 0.0:
                drop_mask = (torch.rand(B, 1, device=device) < z_dropout).float()
                z_at_t = z_at_t * (1.0 - drop_mask)

            diff_loss = policy.training_loss(action_chunk, state_at_t, latent=z_at_t)

            if vel_aux_coeff > 0.0:
                z_flat = zs[:, 1:].reshape(-1, latent_dim)
                vel_flat = true_vel[:, 1:].reshape(-1, 3)
                vel_pred = f_phi.decode_velocity(z_flat)
                vel_loss = torch.mean((vel_pred - vel_flat) ** 2)
                loss = diff_loss + vel_aux_coeff * vel_loss
                vel_loss_val = float(vel_loss.item())
            else:
                vel_loss_val = 0.0
                loss = diff_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses.append(float(diff_loss.item()))
            vel_losses_epoch = locals().setdefault("_vel_losses_epoch", [])
            vel_losses_epoch.append(vel_loss_val)
            with torch.no_grad():
                z_norms.append(float(zs.norm(dim=-1).mean().item()))
            global_step += 1

        ep_loss = float(np.mean(losses))
        z_norm_mean = float(np.mean(z_norms))
        vel_losses_epoch = locals().pop("_vel_losses_epoch", [])
        ep_vel_loss = float(np.mean(vel_losses_epoch)) if vel_losses_epoch else 0.0
        row = {"epoch": epoch, "step": global_step, "loss": ep_loss, "z_norm_mean": z_norm_mean,
               "eval_success": "", "eval_ep_len": ""}

        do_eval = args.eval_during_train and eval_every > 0 and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
        if do_eval:
            denoiser.eval(); f_phi.eval()
            m = nominal_eval(lambda: FlyAdaRolloutPolicy(policy, f_phi, alpha=alpha, update_mode=update_mode),
                             env_cfg_path, n_seeds=eval_seeds, base_seed=700_000 + epoch)
            row.update({
                "eval_success": f"{m['success_rate']:.3f}",
                "eval_ep_len": f"{m['ep_length_mean']:.1f}",
            })
            print(f"  epoch={epoch:03d} loss={ep_loss:.4f} vel_loss={ep_vel_loss:.4f} |z|={z_norm_mean:.3f} "
                  f"nom_succ={m['success_rate']:.3f} len={m['ep_length_mean']:.1f}")
        else:
            print(f"  epoch={epoch:03d} loss={ep_loss:.4f} vel_loss={ep_vel_loss:.4f} |z|={z_norm_mean:.3f}")
        logger.log(row)

    elapsed = time.time() - t_start

    ckpt_path = out_dir / "flyada_policy.pt"
    torch.save({
        "model_state": denoiser.state_dict(),
        "f_phi_state": f_phi.state_dict(),
        "state_mean": mean,
        "state_std": std,
        "config": {
            "action_dim": action_dim,
            "state_dim": state_dim,
            "hidden": hidden,
            "adaptation_hidden": adaptation_hidden,
            "latent_dim": latent_dim,
            "alpha": alpha,
            "horizon": H,
            "T_train": dcfg.T_train,
            "T_sample": dcfg.T_sample,
            "exec_k": dcfg.exec_k,
            "update_mode": update_mode,
        },
    }, ckpt_path)

    denoiser.eval(); f_phi.eval()
    final_metrics = nominal_eval(lambda: FlyAdaRolloutPolicy(policy, f_phi, alpha=alpha, update_mode=update_mode),
                                  env_cfg_path, n_seeds=30, base_seed=910_000)
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

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ls = []; zs = []
        with open(out_dir / "train_log.tsv", "r", newline="") as fh:
            r = csv.DictReader(fh, delimiter="\t")
            for row in r:
                if row.get("loss"):
                    ls.append(float(row["loss"]))
                if row.get("z_norm_mean"):
                    zs.append(float(row["z_norm_mean"]))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].plot(ls); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title(f"{run_name} — epsilon MSE")
        ax[1].plot(zs); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean ||z||"); ax[1].set_title("latent magnitude")
        fig.tight_layout(); fig.savefig(out_dir / "loss_curve.png", dpi=120); plt.close(fig)
    except Exception as e:
        print(f"[train_flyada] plot skipped: {e}")

    print(f"[train_flyada] done in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
