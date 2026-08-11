"""Mismatch eval: sweep mass / drag / wind / delay for vanilla diffusion vs FlyAda.

Per PLAN §8 Experiment 2. Writes a per-condition CSV and success-vs-shift curves PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from eval.rollout import aggregate, rollout_episode
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


NOMINAL = {"mass": 1.0, "drag": 0.1, "wind": [0.0, 0.0, 0.0], "control_delay": 0}


def build_sweep() -> List[Dict]:
    out = [
        {"id": "nominal", "axis": "nominal", "level": 0.0, **NOMINAL},
    ]
    # Mass
    for frac in [0.10, 0.20, 0.30]:
        out.append({"id": f"mass+{int(frac*100)}", "axis": "mass", "level": frac,
                    **{**NOMINAL, "mass": 1.0 * (1.0 + frac)}})
    # Drag
    for frac in [0.50, 1.00]:
        out.append({"id": f"drag+{int(frac*100)}", "axis": "drag", "level": frac,
                    **{**NOMINAL, "drag": 0.1 * (1.0 + frac)}})
    # Wind magnitudes along +x for interpretability
    for wmag in [0.5, 1.0, 1.5]:
        out.append({"id": f"wind_x={wmag}", "axis": "wind", "level": wmag,
                    **{**NOMINAL, "wind": [wmag, 0.0, 0.0]}})
    # Control delay (env steps at 50 Hz)
    for d in [1, 2, 3]:
        out.append({"id": f"delay={d}", "axis": "delay", "level": float(d),
                    **{**NOMINAL, "control_delay": d}})
    return out


def make_flyada_policy(policy, f_phi, mode: str, update_mode: str = "ema",
                       alpha: float = 0.1, freeze_after: int = 0):
    if mode == "full":
        return FlyAdaRolloutPolicy(policy, f_phi, alpha=alpha, update_mode=update_mode, freeze_after_steps=-1)
    if mode == "alpha0":
        return FlyAdaRolloutPolicy(policy, f_phi, alpha=0.0, update_mode=update_mode, freeze_after_steps=-1)
    if mode == "frozen":
        return FlyAdaRolloutPolicy(policy, f_phi, alpha=alpha, update_mode=update_mode, freeze_after_steps=freeze_after)
    raise ValueError(f"Unknown FlyAda mode: {mode}")


def eval_policy_across_sweep(policy_factory, env_cfg_path, sweep, n_seeds, base_seed):
    """policy_factory: () -> stateful policy with .reset() + .predict()."""
    rows = []
    for cond in sweep:
        dyn_over = {"mass": cond["mass"], "drag": cond["drag"],
                    "wind": cond["wind"], "control_delay": cond["control_delay"]}
        env = UAVEnv(config=env_cfg_path, dynamics_overrides=dyn_over)
        pol = policy_factory()
        results = [rollout_episode(env, pol, seed=base_seed + i) for i in range(n_seeds)]
        m = aggregate(results)
        rows.append({
            "id": cond["id"],
            "axis": cond["axis"],
            "level": cond["level"],
            **{k: m[k] for k in ["success_rate", "collision_rate", "final_dist_mean",
                                  "final_dist_p90", "ep_length_mean", "delta_u2_mean", "return_mean"]},
            "n": n_seeds,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-ckpt", type=str, required=True)
    parser.add_argument("--flyada-ckpt", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--run-name", type=str, default="mismatch_v1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env_cfg_path = args.env_config
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = build_sweep()
    print(f"[eval_mismatch] sweep conditions: {[s['id'] for s in sweep]}")

    # Vanilla
    vanilla_policy = load_diffusion_checkpoint(args.vanilla_ckpt, device=args.device)
    print("[eval_mismatch] vanilla…")
    vanilla_rows = eval_policy_across_sweep(
        policy_factory=lambda: vanilla_policy.make_rollout_policy(),
        env_cfg_path=env_cfg_path, sweep=sweep, n_seeds=args.n_seeds, base_seed=800_000,
    )

    # FlyAda full
    fl_policy, f_phi, fcfg = load_flyada_checkpoint(args.flyada_ckpt, device=args.device)
    fly_alpha = float(fcfg.get("alpha", 0.1))
    fly_mode = str(fcfg.get("update_mode", "ema"))
    print(f"[eval_mismatch] flyada (full alpha={fly_alpha} update_mode={fly_mode})…")
    flyada_rows = eval_policy_across_sweep(
        policy_factory=lambda: make_flyada_policy(fl_policy, f_phi, "full",
                                                  update_mode=fly_mode, alpha=fly_alpha),
        env_cfg_path=env_cfg_path, sweep=sweep, n_seeds=args.n_seeds, base_seed=800_000,
    )

    # Write CSV
    csv_path = out_dir / "mismatch_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "id", "axis", "level",
                    "success_rate", "collision_rate", "final_dist_mean",
                    "final_dist_p90", "ep_length_mean", "delta_u2_mean", "return_mean", "n"])
        for r in vanilla_rows:
            w.writerow(["vanilla"] + [r[k] for k in ["id", "axis", "level",
                        "success_rate", "collision_rate", "final_dist_mean",
                        "final_dist_p90", "ep_length_mean", "delta_u2_mean", "return_mean", "n"]])
        for r in flyada_rows:
            w.writerow(["flyada"] + [r[k] for k in ["id", "axis", "level",
                        "success_rate", "collision_rate", "final_dist_mean",
                        "final_dist_p90", "ep_length_mean", "delta_u2_mean", "return_mean", "n"]])

    # Summary JSON
    summary = {
        "n_seeds_per_condition": args.n_seeds,
        "vanilla_mean_success": float(np.mean([r["success_rate"] for r in vanilla_rows])),
        "flyada_mean_success": float(np.mean([r["success_rate"] for r in flyada_rows])),
        "per_condition": {
            r["id"]: {
                "vanilla": v["success_rate"], "flyada": r["success_rate"],
            } for v, r in zip(vanilla_rows, flyada_rows)
        },
    }
    with open(out_dir / "mismatch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot success-vs-shift curves per axis
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        axes_order = ["mass", "drag", "wind", "delay"]
        fig, axs = plt.subplots(1, len(axes_order), figsize=(4 * len(axes_order), 3.2), sharey=True)
        for ax, ax_name in zip(axs, axes_order):
            vrows = [r for r in vanilla_rows if r["axis"] == ax_name]
            frows = [r for r in flyada_rows if r["axis"] == ax_name]
            vrows.sort(key=lambda r: r["level"]); frows.sort(key=lambda r: r["level"])
            ax.plot([r["level"] for r in vrows], [r["success_rate"] for r in vrows], "o-", label="vanilla")
            ax.plot([r["level"] for r in frows], [r["success_rate"] for r in frows], "s--", label="FlyAda")
            ax.set_title(ax_name); ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(ax_name + " level"); ax.grid(True, alpha=0.3)
        axs[0].set_ylabel("success rate")
        axs[-1].legend(loc="lower left")
        fig.suptitle(f"Mismatch robustness (n={args.n_seeds})")
        fig.tight_layout(); fig.savefig(out_dir / "mismatch_success_curves.png", dpi=130); plt.close(fig)
    except Exception as e:
        print(f"[eval_mismatch] plot skipped: {e}")

    print(f"[eval_mismatch] mean success: vanilla={summary['vanilla_mean_success']:.3f} "
          f"flyada={summary['flyada_mean_success']:.3f}")
    print(f"[eval_mismatch] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
