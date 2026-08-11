"""Evaluate PPO teacher on nominal Task A over N seeds (Stage 1)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from stable_baselines3 import PPO


def run_episode(env: UAVEnv, model: PPO, seed: int, deterministic: bool):
    obs, info = env.reset(seed=seed)
    goal = info["goal"].astype(np.float32)
    traj = [obs[:3].copy()]
    ep_return = 0.0
    steps = 0
    collided = False
    success = False
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        traj.append(obs[:3].copy())
        ep_return += float(r)
        steps += 1
        if info.get("collided"):
            collided = True
        if info.get("success"):
            success = True
        if term or trunc:
            break
    traj = np.stack(traj)
    final_d = float(np.linalg.norm(obs[:3] - goal))
    return {
        "seed": seed,
        "success": bool(success),
        "collided": bool(collided),
        "final_dist": final_d,
        "length": steps,
        "return": ep_return,
        "trajectory": traj,
        "goal": goal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    out_dir = ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg["model_path"]
    if not Path(model_path).is_absolute():
        model_path = str((ROOT / model_path).resolve())

    env_cfg_path = cfg["env_config"]
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    n_seeds = int(args.n_seeds or cfg["nominal"]["n_seeds"])
    deterministic = bool(cfg["nominal"]["deterministic"])
    plot_n = int(cfg["nominal"]["plot_n_trajectories"])

    print(f"[eval_nominal] model={model_path} seeds={n_seeds}")
    model = PPO.load(model_path, device="cpu")
    env = UAVEnv(config=env_cfg_path)

    results = []
    for i in range(n_seeds):
        results.append(run_episode(env, model, seed=100_000 + i, deterministic=deterministic))

    # aggregate
    successes = np.array([r["success"] for r in results])
    collisions = np.array([r["collided"] for r in results])
    finals = np.array([r["final_dist"] for r in results])
    lengths = np.array([r["length"] for r in results])
    returns = np.array([r["return"] for r in results])

    summary = {
        "run_name": run_name,
        "n_seeds": int(n_seeds),
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "final_dist_mean": float(finals.mean()),
        "final_dist_median": float(np.median(finals)),
        "final_dist_p90": float(np.percentile(finals, 90)),
        "episode_length_mean": float(lengths.mean()),
        "episode_length_median_success": float(np.median(lengths[successes]) if successes.any() else -1.0),
        "return_mean": float(returns.mean()),
    }

    with open(out_dir / "eval_nominal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "eval_nominal_per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "success", "collided", "final_dist", "length", "return"])
        for r in results:
            w.writerow([r["seed"], int(r["success"]), int(r["collided"]),
                        f"{r['final_dist']:.4f}", r["length"], f"{r['return']:.4f}"])

    # representative 3D trajectories
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # Pick up to plot_n successful, otherwise fill with any
        succ_idx = [i for i, r in enumerate(results) if r["success"]]
        pick = succ_idx[:plot_n]
        if len(pick) < plot_n:
            extras = [i for i in range(len(results)) if i not in pick]
            pick += extras[: plot_n - len(pick)]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        for i, idx in enumerate(pick):
            tr = results[idx]["trajectory"]
            g = results[idx]["goal"]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], label=f"ep {idx} {'OK' if results[idx]['success'] else 'X'}")
            ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], marker="o", s=30)
            ax.scatter(g[0], g[1], g[2], marker="*", s=70)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{run_name} — nominal trajectories")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "eval_nominal_traj3d.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[eval_nominal] plot skipped: {e}")

    print("[eval_nominal] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
