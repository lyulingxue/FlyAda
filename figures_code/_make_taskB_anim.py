"""Animate Task B partial-obs rollouts: vanilla / frame-stack / FlyAda on the same seed.

Each frame advances all three drones one env step. Trails grow behind each drone;
waypoints turn gold as they're reached by FlyAda; baselines drift past WP1 in
broad loops.

Output: paper/figures/taskB_comparison.gif
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from eval.eval_longhorizon import rollout_taskB
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


SEED = 600000   # matches the teaser
ADVANCE_TOL = 0.5


def main():
    device = "cuda"
    van = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"), device=device)
    fst = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_frame3_v1/diffusion_policy.pt"), device=device)
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device)
    alpha = float(fcfg.get("alpha", 0.1)); mode = str(fcfg.get("update_mode", "ema"))

    factories = {
        "Vanilla":     lambda: van.make_rollout_policy(),
        "Frame-stack": lambda: fst.make_rollout_policy(),
        "FlyAda":      lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=alpha, update_mode=mode),
    }

    env = UAVEnv(config=str(ROOT / "configs/env_partial_taskB.yaml"))
    rollouts: Dict[str, dict] = {}
    for name, fac in factories.items():
        ep = rollout_taskB(env, fac(), seed=SEED)
        rollouts[name] = {
            "trajectory": ep.trajectory,
            "waypoints": ep.waypoints,
            "wps": ep.waypoints_reached,
            "length": ep.length,
        }
        print(f"{name}: length={ep.length}, wps={ep.waypoints_reached}/3")

    waypoints = rollouts["FlyAda"]["waypoints"]
    n_wp = len(waypoints)

    # Total animation length = shortest non-zero policy length or capped at 300 to keep GIF small
    raw_max = max(r["length"] for r in rollouts.values())
    n_frames_total = min(raw_max + 1, 300)
    # Subsample for a reasonable GIF (one frame per sub_step env steps)
    sub_step = max(1, raw_max // n_frames_total)
    frame_indices = np.arange(0, raw_max + 1, sub_step)[:n_frames_total]

    from paper._plot_style import set_ieee_font
    set_ieee_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    colors = {"Vanilla": "#1f77b4", "Frame-stack": "#2ca02c", "FlyAda": "#ff7f0e"}
    linestyles = {"Vanilla": "--", "Frame-stack": ":", "FlyAda": "-"}

    all_pts = np.concatenate(
        [r["trajectory"] for r in rollouts.values()] + [waypoints], axis=0
    )
    xmin, xmax = float(all_pts[:, 0].min() - 0.5), float(all_pts[:, 0].max() + 0.5)
    ymin, ymax = float(all_pts[:, 1].min() - 0.5), float(all_pts[:, 1].max() + 0.5)
    zmin, zmax = float(all_pts[:, 2].min() - 0.5), float(all_pts[:, 2].max() + 0.5)

    fig = plt.figure(figsize=(7, 4.6))
    ax = fig.add_subplot(111, projection="3d")

    def render(k):
        ax.cla()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-62)

        env_t = frame_indices[k]
        for name, r in rollouts.items():
            traj = r["trajectory"]
            t_clipped = min(env_t, len(traj) - 1)
            sub = traj[: t_clipped + 1]
            ax.plot(sub[:, 0], sub[:, 1], sub[:, 2],
                    color=colors[name], linestyle=linestyles[name], linewidth=1.8, alpha=0.9)
            ax.scatter(sub[-1, 0], sub[-1, 1], sub[-1, 2],
                       color=colors[name], s=60, edgecolors="black", linewidths=0.6,
                       label=name, zorder=10)

        # Waypoints — color-code by FlyAda's reached status at this frame
        fly_traj = rollouts["FlyAda"]["trajectory"]
        fly_t = min(env_t, len(fly_traj) - 1)
        # Determine how many waypoints FlyAda has reached by env step env_t.
        wp_reached = 0
        running_target = waypoints[0]
        for i in range(fly_t + 1):
            d = float(np.linalg.norm(fly_traj[i] - running_target))
            if d < ADVANCE_TOL and wp_reached < n_wp:
                wp_reached += 1
                if wp_reached < n_wp:
                    running_target = waypoints[wp_reached]
        for i, wp in enumerate(waypoints):
            done = i < wp_reached
            ax.scatter(wp[0], wp[1], wp[2],
                       color="gold" if done else "black",
                       marker="*", s=240 if done else 200,
                       edgecolors="white", linewidths=0.8, zorder=11)
            ax.text(wp[0] + 0.2, wp[1] + 0.2, wp[2] + 0.35, f"WP{i+1}",
                    fontsize=8, color="black", fontweight="bold")

        ax.set_title(f"Task B — partial obs, env step {env_t}/{raw_max}", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0),
                  fontsize=8.5, frameon=True, facecolor="white", framealpha=0.85)

    print(f"rendering {len(frame_indices)} frames (sub_step={sub_step}, total env steps={raw_max})")
    anim = FuncAnimation(fig, render, frames=len(frame_indices), interval=80, blit=False)
    out_path = ROOT / "paper/figures/taskB_comparison.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=15), dpi=110)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
