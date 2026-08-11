"""Adaptation ablation (PLAN §8 Experiment 3).

Four variants on a canonical mismatch condition (mass+30%, drag+100%, wind, delay=2):
  1. vanilla diffusion (no z)
  2. FlyAda alpha=0 at test (z stays 0)
  3. FlyAda frozen after warmup (z updates for the first k steps, then frozen)
  4. FlyAda full alpha=0.1

Also saves a z_t time-series for one representative episode (Experiment 3 figure).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from eval.rollout import aggregate, rollout_episode
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


# Canonical "hard" condition for the ablation
HARD_COND = {"mass": 1.3, "drag": 0.2, "wind": [1.0, 0.0, 0.0], "control_delay": 2}


def run_variant(name: str, policy_factory, env_cfg, n_seeds, base_seed, cond):
    env = UAVEnv(config=env_cfg, dynamics_overrides=cond)
    results = []
    for i in range(n_seeds):
        pol = policy_factory()
        results.append(rollout_episode(env, pol, seed=base_seed + i))
    m = aggregate(results)
    m["name"] = name
    return m, results


def collect_z_timeseries(flyada_policy_factory, env_cfg, cond, seed):
    """Rollout one episode with alpha=0.1 and record z_t at every env step."""
    env = UAVEnv(config=env_cfg, dynamics_overrides=cond)
    pol = flyada_policy_factory()
    obs, info = env.reset(seed=seed)
    zs = [pol._z.detach().cpu().numpy().reshape(-1).copy()]
    while True:
        action, _ = pol.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        zs.append(pol._z.detach().cpu().numpy().reshape(-1).copy())
        if term or trunc:
            break
    return np.stack(zs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-ckpt", type=str, required=True)
    parser.add_argument("--flyada-ckpt", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--n-seeds", type=int, default=40)
    parser.add_argument("--run-name", type=str, default="ablation_v1")
    parser.add_argument("--freeze-after", type=int, default=10,
                        help="For the 'frozen-after-warmup' variant.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env_cfg_path = args.env_config
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vanilla_policy = load_diffusion_checkpoint(args.vanilla_ckpt, device=args.device)
    fl_policy, f_phi, fcfg = load_flyada_checkpoint(args.flyada_ckpt, device=args.device)
    fly_alpha = float(fcfg.get("alpha", 0.1))
    fly_mode = str(fcfg.get("update_mode", "ema"))

    variants = {
        "vanilla": lambda: vanilla_policy.make_rollout_policy(),
        "flyada_alpha0": lambda: FlyAdaRolloutPolicy(fl_policy, f_phi, alpha=0.0, update_mode=fly_mode),
        f"flyada_frozen_after_{args.freeze_after}":
            lambda: FlyAdaRolloutPolicy(fl_policy, f_phi, alpha=fly_alpha, update_mode=fly_mode,
                                         freeze_after_steps=args.freeze_after),
        "flyada_full": lambda: FlyAdaRolloutPolicy(fl_policy, f_phi, alpha=fly_alpha, update_mode=fly_mode),
    }

    rows = []
    for name, factory in variants.items():
        print(f"[eval_ablation] running {name}…")
        m, _ = run_variant(name, factory, env_cfg_path, args.n_seeds, 820_000, HARD_COND)
        rows.append(m)

    # Write CSV
    csv_path = out_dir / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "success_rate", "collision_rate", "final_dist_mean",
                    "final_dist_p90", "ep_length_mean", "delta_u2_mean", "return_mean", "n"])
        for r in rows:
            w.writerow([r["name"], r["success_rate"], r["collision_rate"], r["final_dist_mean"],
                        r["final_dist_p90"], r["ep_length_mean"], r["delta_u2_mean"], r["return_mean"], r["n"]])

    summary = {"condition": HARD_COND, "n_seeds": args.n_seeds, "variants": {r["name"]: r for r in rows}}
    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # z_t time-series for one representative seed
    try:
        zs = collect_z_timeseries(
            lambda: FlyAdaRolloutPolicy(fl_policy, f_phi, alpha=fly_alpha, update_mode=fly_mode),
            env_cfg_path, HARD_COND, seed=820_013,
        )
        np.save(out_dir / "z_timeseries.npy", zs)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for k in range(zs.shape[1]):
            ax.plot(zs[:, k], alpha=0.7, linewidth=1.0)
        ax.set_xlabel("env step"); ax.set_ylabel("z_t component value")
        ax.set_title(f"FlyAda z_t trajectory (mass+30%, drag+100%, wind=1, delay=2)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "z_timeseries.png", dpi=130); plt.close(fig)
    except Exception as e:
        print(f"[eval_ablation] z-timeseries plot skipped: {e}")

    print("[eval_ablation] results:")
    for r in rows:
        print(f"  {r['name']:<35s}  succ={r['success_rate']:.3f}  "
              f"final_d={r['final_dist_mean']:.2f}  len={r['ep_length_mean']:.1f}")
    print(f"[eval_ablation] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
