"""Sim-to-sim transfer test: deploy partial-obs checkpoints in a 6-DoF MuJoCo
quadrotor that has full rotor + attitude dynamics.

Pipeline:
  - Load checkpoint (vanilla / frame-stack / FlyAda).
  - Wrap a MuJoCoQuadrotorEnv that exposes the same 12-dim partial observation
    and 3-dim action interface used during training.
  - Roll out N seeds for each policy under nominal and a hard mismatch condition
    (mass scale 1.3, body-frame drag, +1 m/s wind in x).
  - Save per-seed trajectories, plot a 3D comparison, write a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mujoco_quadrotor import MuJoCoQuadrotorEnv
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


def make_policy(ckpt: str, kind: str, device: str = "cpu"):
    if kind == "flyada":
        dp, f_phi, fcfg = load_flyada_checkpoint(ckpt, device=device)
        return lambda: FlyAdaRolloutPolicy(
            dp, f_phi,
            alpha=float(fcfg.get("alpha", 0.1)),
            update_mode=str(fcfg.get("update_mode", "ema")),
        )
    if kind in ("vanilla", "frame_stack"):
        dp = load_diffusion_checkpoint(ckpt, device=device)
        return lambda: dp.make_rollout_policy()
    raise ValueError(f"unknown kind {kind}")


def rollout(env: MuJoCoQuadrotorEnv, policy_factory: Callable, n_seeds: int, base_seed: int):
    out = []
    for i in range(n_seeds):
        pol = policy_factory()
        obs, info = env.reset(seed=base_seed + i)
        traj = [obs[0:3].copy()]
        d_series = [float(obs[11])]
        steps = 0
        while True:
            action, _ = pol.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            traj.append(obs[0:3].copy())
            d_series.append(float(info["d_goal"]))
            steps += 1
            if term or trunc:
                break
        out.append({
            "seed": base_seed + i,
            "success": bool(info["success"]),
            "out_of_bounds": bool(info["out_of_bounds"]),
            "length": steps,
            "final_d": float(info["d_goal"]),
            "trajectory": np.stack(traj),
            "goal": env._goal.copy(),
            "d_series": np.asarray(d_series, dtype=np.float32),
        })
    return out


def aggregate(eps: List[Dict]) -> Dict:
    n = len(eps)
    if n == 0:
        return {}
    succ = np.array([r["success"] for r in eps])
    fd = np.array([r["final_d"] for r in eps])
    ln = np.array([r["length"] for r in eps])
    return {
        "n": int(n),
        "success_rate": float(succ.mean()),
        "final_dist_mean": float(fd.mean()),
        "final_dist_p90": float(np.percentile(fd, 90)),
        "ep_length_mean": float(ln.mean()),
    }


def plot_3d(out_dir: Path, results_by_method: Dict[str, List[Dict]], cond_name: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
    except Exception as e:
        print(f"[plot_3d] skipped: {e}")
        return
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"vanilla": "C0", "frame_stack": "C2", "flyada": "C1"}
    pretty = {"vanilla": "Vanilla", "frame_stack": "Frame-stack", "flyada": "FlyAda"}
    for method, eps in results_by_method.items():
        for i, r in enumerate(eps[:3]):
            t = r["trajectory"]
            label = pretty.get(method, method) if i == 0 else None
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=colors.get(method, "k"),
                    alpha=0.9, linewidth=1.8, label=label)
        if eps:
            goals = np.stack([r["goal"] for r in eps[:3]])
            ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], color=colors.get(method, "k"),
                       marker="*", s=120, alpha=0.95, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"mujoco_traj_{cond_name}.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-ckpt", type=str, default="results/diffusion_partial_v1/diffusion_policy.pt")
    p.add_argument("--frame-stack-ckpt", type=str, default="results/diffusion_partial_frame3_v1/diffusion_policy.pt")
    p.add_argument("--flyada-ckpt", type=str, default="results/flyada_partial_v1/flyada_policy.pt")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out-dir", type=str, default="results/mujoco_transfer_v1")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "vanilla":     make_policy(str((ROOT / args.vanilla_ckpt).resolve()),     "vanilla"),
        "frame_stack": make_policy(str((ROOT / args.frame_stack_ckpt).resolve()), "frame_stack"),
        "flyada":      make_policy(str((ROOT / args.flyada_ckpt).resolve()),      "flyada"),
    }

    conditions = {
        "nominal": dict(mass_scale=1.0, drag_world=0.0, wind_world=(0.0, 0.0, 0.0)),
        "hard":    dict(mass_scale=1.3, drag_world=0.1, wind_world=(1.0, 0.0, 0.0)),
    }

    summary: Dict[str, Dict] = {}
    for cond_name, cond_kwargs in conditions.items():
        print(f"\n=== condition: {cond_name} ({cond_kwargs}) ===", flush=True)
        env = MuJoCoQuadrotorEnv(partial_obs=True, max_steps=args.max_steps, **cond_kwargs)
        results_by_method: Dict[str, List[Dict]] = {}
        cond_summary: Dict[str, Dict] = {}
        for method, factory in methods.items():
            print(f"  {method:<12s} ...", flush=True)
            eps = rollout(env, factory, n_seeds=args.n_seeds, base_seed=300_000)
            agg = aggregate(eps)
            results_by_method[method] = eps
            cond_summary[method] = agg
            print(f"    succ={agg['success_rate']:.3f}  final_d={agg['final_dist_mean']:.2f}  "
                  f"len={agg['ep_length_mean']:.0f}", flush=True)
        plot_3d(out_dir, results_by_method, cond_name)
        summary[cond_name] = cond_summary

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[mujoco_transfer] saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
