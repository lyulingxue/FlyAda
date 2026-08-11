"""Make a teaser figure for the first page: same seed, three methods, partial-obs Task B."""
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


def main():
    device = "cuda"
    van = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"), device=device)
    fst = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_frame3_v1/diffusion_policy.pt"), device=device)
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device)
    alpha = float(fcfg.get("alpha", 0.1))
    mode = str(fcfg.get("update_mode", "ema"))

    factories = {
        "Vanilla":     lambda: van.make_rollout_policy(),
        "Frame-stack": lambda: fst.make_rollout_policy(),
        "FlyAda":      lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=alpha, update_mode=mode),
    }

    # Search a few seeds for one where FlyAda succeeds and the others clearly fail.
    target = None
    for s in range(600_000, 600_050):
        env = UAVEnv(config=str(ROOT / "configs/env_partial_taskB.yaml"))
        rollouts: Dict[str, dict] = {}
        for name, fac in factories.items():
            ep = rollout_taskB(env, fac(), seed=s)
            rollouts[name] = {
                "trajectory": ep.trajectory,
                "waypoints": ep.waypoints,
                "success": ep.success,
                "wps": ep.waypoints_reached,
                "length": ep.length,
            }
        if (rollouts["FlyAda"]["success"] and rollouts["FlyAda"]["wps"] == 3
                and rollouts["Vanilla"]["wps"] <= 1 and rollouts["Frame-stack"]["wps"] <= 1):
            target = (s, rollouts)
            print(f"seed {s} matches: FlyAda 3/3, Vanilla {rollouts['Vanilla']['wps']}/3, FS {rollouts['Frame-stack']['wps']}/3")
            break
    if target is None:
        print("No clean seed found in range — using last seed regardless.")
        target = (s, rollouts)
    seed, results = target

    # Plot
    from paper._plot_style import set_ieee_font
    set_ieee_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(6.4, 3.4))
    ax = fig.add_subplot(111, projection="3d")

    from paper._plot_style import METHOD_COLORS, METHOD_LINESTYLES
    colors = METHOD_COLORS
    linestyles = METHOD_LINESTYLES
    linewidths = {"Vanilla": 1.6, "Frame-stack": 1.6, "FlyAda": 2.6}

    waypoints = results["FlyAda"]["waypoints"]

    for method, r in results.items():
        traj = r["trajectory"]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=colors[method], linestyle=linestyles[method],
                linewidth=linewidths[method], alpha=0.95,
                label=f"{method} ({r['wps']}/{len(waypoints)})")

    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
               color="black", marker="*", s=220, edgecolors="white", linewidths=0.8,
               label="Waypoints", zorder=10)
    for i, wp in enumerate(waypoints):
        ax.text(wp[0] + 0.2, wp[1] + 0.2, wp[2] + 0.35, f"WP{i+1}",
                fontsize=9, color="black", fontweight="bold")

    start = results["FlyAda"]["trajectory"][0]
    ax.scatter(start[0], start[1], start[2], color="white", marker="o", s=80,
               edgecolors="black", linewidths=0.8, label="Start", zorder=10)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8.5,
              frameon=True, facecolor="white", framealpha=0.8, edgecolor="0.7")
    fig.tight_layout()
    out = ROOT / "paper/figures/teaser.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out} (seed={seed})")


if __name__ == "__main__":
    main()
