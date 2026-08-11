"""Long-horizon Task B (waypoint-following) evaluation (PLAN §8 Experiment 4).

Reuses the Task-A-trained partial-obs policies (vanilla diffusion and FlyAda) and
rolls them out on Task B (3 sequential waypoints, 800-step budget). No retraining:
the observation format is unchanged — g_rel and d_goal always point to the current
waypoint, so the policies that learned "go to the goal you see" transfer transparently.

Metrics per PLAN §8:
  - task completion  (all waypoints reached in order)
  - cumulative tracking error  (sum_t d(current waypoint) over the episode)
  - drift from planned route  (same, reported as a vs-time curve)
  - per-waypoint completion

Outputs:
  - summary.json  (mean metrics per policy and per condition)
  - per_seed.csv
  - longhorizon_traj3d.png  (representative trajectory per policy)
  - longhorizon_drift.png  (mean ± std d-to-current-wp vs time)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


@dataclass
class TaskBEpisode:
    seed: int
    success: bool
    waypoints_reached: int
    num_waypoints: int
    length: int
    cumulative_tracking_error: float
    mean_tracking_error: float
    rmse_tracking_error: float
    max_tracking_error: float
    trajectory: np.ndarray      # [T+1, 3]
    waypoints: np.ndarray       # [num_waypoints, 3]
    d_to_current_wp: np.ndarray # [T]
    wp_reached_at_step: np.ndarray  # [num_waypoints], -1 if not reached


def rollout_taskB(env, policy, seed: int) -> TaskBEpisode:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    waypoints = np.stack(info["waypoints"])
    num_wp = len(waypoints)

    traj = [obs[:3].copy()]
    d_series: List[float] = []
    wp_reached = -np.ones(num_wp, dtype=np.int32)
    max_wp_reached = -1

    steps = 0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        traj.append(obs[:3].copy())
        d_series.append(float(info["d_goal"]))
        steps += 1
        wp_idx = int(info["wp_idx"])
        # wp_idx advances the moment we cross the threshold; record the first step each was reached
        if info.get("wp_advanced") and wp_reached[wp_idx - 1] < 0:
            wp_reached[wp_idx - 1] = steps
            max_wp_reached = max(max_wp_reached, wp_idx - 1)
        if info.get("success") and wp_reached[num_wp - 1] < 0:
            wp_reached[num_wp - 1] = steps
            max_wp_reached = max(max_wp_reached, num_wp - 1)
        if term or trunc:
            break

    waypoints_reached = int(max_wp_reached + 1) if max_wp_reached >= 0 else 0
    d_arr = np.array(d_series, dtype=np.float32)
    return TaskBEpisode(
        seed=seed,
        success=bool(info["success"]),
        waypoints_reached=waypoints_reached,
        num_waypoints=num_wp,
        length=steps,
        cumulative_tracking_error=float(d_arr.sum()),
        mean_tracking_error=float(d_arr.mean()) if len(d_arr) else 0.0,
        rmse_tracking_error=float(np.sqrt(np.mean(d_arr ** 2))) if len(d_arr) else 0.0,
        max_tracking_error=float(d_arr.max()) if len(d_arr) else 0.0,
        trajectory=np.stack(traj),
        waypoints=waypoints,
        d_to_current_wp=d_arr,
        wp_reached_at_step=wp_reached,
    )


def aggregate_taskB(results: List[TaskBEpisode]) -> dict:
    if not results:
        return {}
    n = len(results)
    succ = np.array([r.success for r in results])
    wp_frac = np.array([r.waypoints_reached / r.num_waypoints for r in results])
    length = np.array([r.length for r in results])
    cum_err = np.array([r.cumulative_tracking_error for r in results])
    mean_err = np.array([r.mean_tracking_error for r in results])
    rmse_err = np.array([r.rmse_tracking_error for r in results])
    max_err = np.array([r.max_tracking_error for r in results])
    return {
        "n": int(n),
        "task_success_rate": float(succ.mean()),
        "waypoint_completion_mean": float(wp_frac.mean()),
        "all_waypoints_rate": float(((np.array([r.waypoints_reached for r in results]))
                                     == np.array([r.num_waypoints for r in results])).mean()),
        "ep_length_mean": float(length.mean()),
        "cumulative_tracking_error_mean": float(cum_err.mean()),
        "mean_tracking_error_mean": float(mean_err.mean()),
        "rmse_tracking_error_mean": float(rmse_err.mean()),
        "max_tracking_error_mean": float(max_err.mean()),
    }


def run_condition(policy_factory: Callable, env_cfg: str, n_seeds: int,
                  base_seed: int, dyn_over: dict | None) -> List[TaskBEpisode]:
    env = UAVEnv(config=env_cfg, dynamics_overrides=dyn_over)
    out: List[TaskBEpisode] = []
    for i in range(n_seeds):
        out.append(rollout_taskB(env, policy_factory(), seed=base_seed + i))
    return out


def plot_trajectories(out_dir: Path, vanilla_eps: List[TaskBEpisode], flyada_eps: List[TaskBEpisode]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print(f"[eval_longhorizon] trajectory plot skipped: {e}")
        return

    def best(eps):
        # Pick episode reaching the most waypoints; break ties by shortest length.
        return sorted(eps, key=lambda r: (-r.waypoints_reached, r.length))[0]

    v_best = best(vanilla_eps)
    f_best = best(flyada_eps)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ft = f_best.trajectory
    ax.plot(ft[:, 0], ft[:, 1], ft[:, 2], color="C1", linewidth=2.4,
            label=f"FlyAda (wps {f_best.waypoints_reached}/{f_best.num_waypoints})")
    vt = v_best.trajectory
    ax.plot(vt[:, 0], vt[:, 1], vt[:, 2], color="C0", linewidth=1.6, alpha=0.9,
            label=f"Vanilla (wps {v_best.waypoints_reached}/{v_best.num_waypoints})")
    wps = f_best.waypoints
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], color="k", s=120, marker="*",
               label="waypoints")
    ax.scatter(ft[0, 0], ft[0, 1], ft[0, 2], color="g", s=70, marker="o", label="start")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "longhorizon_traj3d.png", dpi=140); plt.close(fig)


def plot_drift(out_dir: Path, vanilla_eps: List[TaskBEpisode], flyada_eps: List[TaskBEpisode]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[eval_longhorizon] drift plot skipped: {e}")
        return

    def stack_pad(eps):
        max_len = max(len(r.d_to_current_wp) for r in eps)
        M = np.full((len(eps), max_len), np.nan, dtype=np.float32)
        for i, r in enumerate(eps):
            M[i, :len(r.d_to_current_wp)] = r.d_to_current_wp
        return M

    Vm = stack_pad(vanilla_eps)
    Fm = stack_pad(flyada_eps)

    # Use nan-aware mean/std since episodes can terminate at different lengths.
    def mstd(M):
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0)

    vm, vs = mstd(Vm)
    fm, fs = mstd(Fm)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    t_v = np.arange(len(vm))
    t_f = np.arange(len(fm))
    ax.plot(t_v, vm, color="C0", linewidth=1.8, label=f"Vanilla (n={len(vanilla_eps)})")
    ax.fill_between(t_v, vm - vs, vm + vs, color="C0", alpha=0.18)
    ax.plot(t_f, fm, color="C1", linewidth=1.8, label=f"FlyAda (n={len(flyada_eps)})")
    ax.fill_between(t_f, fm - fs, fm + fs, color="C1", alpha=0.18)
    ax.axhline(0.5, color="k", linewidth=1.0, linestyle=":", label="advance tol")
    ax.set_xlabel("env step")
    ax.set_ylabel("distance to current waypoint (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "longhorizon_drift.png", dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla-ckpt", type=str, required=True)
    parser.add_argument("--flyada-ckpt", type=str, required=True)
    parser.add_argument("--env-config", type=str, default="configs/env_partial_taskB.yaml")
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--run-name", type=str, default="longhorizon_partial_v1")
    parser.add_argument("--include-hard", action="store_true",
                        help="Also evaluate under the combined hard dynamics condition.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    env_cfg = args.env_config
    if not Path(env_cfg).is_absolute():
        env_cfg = str((ROOT / env_cfg).resolve())

    out_dir = ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    vanilla_policy = load_diffusion_checkpoint(args.vanilla_ckpt, device=args.device)
    fl_policy, f_phi, fcfg = load_flyada_checkpoint(args.flyada_ckpt, device=args.device)
    fly_alpha = float(fcfg.get("alpha", 0.1))
    fly_mode = str(fcfg.get("update_mode", "ema"))

    conditions = [{"id": "nominal", "dyn": None}]
    if args.include_hard:
        conditions.append({
            "id": "hard", "dyn": {"mass": 1.3, "drag": 0.2, "wind": [1.0, 0.0, 0.0], "control_delay": 2},
        })

    summary = {"conditions": {}, "n_seeds": args.n_seeds}
    per_seed_rows = []

    for cond in conditions:
        print(f"[eval_longhorizon] condition={cond['id']} vanilla…")
        v_eps = run_condition(
            lambda: vanilla_policy.make_rollout_policy(),
            env_cfg, args.n_seeds, base_seed=600_000, dyn_over=cond["dyn"],
        )
        print(f"[eval_longhorizon] condition={cond['id']} flyada…")
        f_eps = run_condition(
            lambda: FlyAdaRolloutPolicy(fl_policy, f_phi, alpha=fly_alpha, update_mode=fly_mode),
            env_cfg, args.n_seeds, base_seed=600_000, dyn_over=cond["dyn"],
        )
        va = aggregate_taskB(v_eps)
        fa = aggregate_taskB(f_eps)
        summary["conditions"][cond["id"]] = {"vanilla": va, "flyada": fa}

        for r in v_eps:
            per_seed_rows.append((cond["id"], "vanilla", r.seed, int(r.success),
                                   r.waypoints_reached, r.num_waypoints, r.length,
                                   f"{r.cumulative_tracking_error:.3f}", f"{r.mean_tracking_error:.3f}"))
        for r in f_eps:
            per_seed_rows.append((cond["id"], "flyada", r.seed, int(r.success),
                                   r.waypoints_reached, r.num_waypoints, r.length,
                                   f"{r.cumulative_tracking_error:.3f}", f"{r.mean_tracking_error:.3f}"))

        if cond["id"] == "nominal":
            plot_trajectories(out_dir, v_eps, f_eps)
            plot_drift(out_dir, v_eps, f_eps)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "method", "seed", "success",
                    "waypoints_reached", "num_waypoints", "length",
                    "cumulative_tracking_error", "mean_tracking_error"])
        for row in per_seed_rows:
            w.writerow(row)

    print("[eval_longhorizon] summary:")
    for cond_id, sub in summary["conditions"].items():
        print(f"  {cond_id}:")
        for method in ["vanilla", "flyada"]:
            m = sub[method]
            print(f"    {method:<8s}  task_succ={m['task_success_rate']:.3f}  "
                  f"wp_frac={m['waypoint_completion_mean']:.3f}  "
                  f"len={m['ep_length_mean']:.0f}  "
                  f"mean_err={m['mean_tracking_error_mean']:.2f}")


if __name__ == "__main__":
    main()
