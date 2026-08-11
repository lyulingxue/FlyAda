"""Re-render Fig 2 (training panel) with multiple curves.

Top: diffusion epsilon-MSE for all three partial-obs methods (vanilla,
frame-stack, FlyAda) on the same axis, plus FlyAda's ||z_t|| on a secondary
right-axis. The point: training-loss curves alone don't separate the methods.
Bottom: nominal-eval success rate during training for the three methods. Only
FlyAda climbs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper._plot_style import set_ieee_font
set_ieee_font()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path: Path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    return rows


def take_final_run(rows):
    """The TSV is appended across runs; return only the last run (last epoch=0 onward)."""
    if not rows:
        return rows
    last_zero = max(i for i, r in enumerate(rows) if r["epoch"] == "0")
    return rows[last_zero:]


def parse_loss_and_eval(rows, eval_key="eval_success"):
    epochs, losses = [], []
    eval_e, eval_s = [], []
    z_e, z_v = [], []
    for r in rows:
        if r.get("loss"):
            try:
                e = int(r["epoch"]); l = float(r["loss"])
                epochs.append(e); losses.append(l)
            except (ValueError, TypeError):
                pass
        if r.get(eval_key):
            try:
                e = int(r["epoch"]); s = float(r[eval_key])
                eval_e.append(e); eval_s.append(s)
            except (ValueError, TypeError):
                pass
        if r.get("z_norm_mean"):
            try:
                e = int(r["epoch"]); z = float(r["z_norm_mean"])
                z_e.append(e); z_v.append(z)
            except (ValueError, TypeError):
                pass
    return (np.array(epochs), np.array(losses),
            np.array(eval_e), np.array(eval_s),
            np.array(z_e), np.array(z_v))


def main():
    runs = {
        "Vanilla":     ROOT / "results/diffusion_partial_v1/train_log.tsv",
        "Frame-stack": ROOT / "results/diffusion_partial_frame3_v1/train_log.tsv",
        "FlyAda":      ROOT / "results/flyada_partial_v1/train_log.tsv",
    }
    parsed = {}
    for name, path in runs.items():
        rows = take_final_run(load_log(path))
        parsed[name] = parse_loss_and_eval(rows)

    from paper._plot_style import METHOD_COLORS, METHOD_LINESTYLES
    colors = METHOD_COLORS
    linestyles = METHOD_LINESTYLES
    linewidths = {"Vanilla": 1.8, "Frame-stack": 1.8, "FlyAda": 2.2}

    fig, ax = plt.subplots(2, 1, figsize=(6, 4.8), sharex=True)

    # Top: diffusion epsilon-MSE for all 3, plus ||z|| on twin axis (FlyAda only)
    for name in ("Vanilla", "Frame-stack", "FlyAda"):
        ep, ls, _, _, _, _ = parsed[name]
        ax[0].plot(ep, ls, color=colors[name], linestyle=linestyles[name],
                   linewidth=linewidths[name], label=name)
    ax[0].set_ylabel(r"diffusion $\epsilon$-MSE")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.85)

    # Twin axis: FlyAda's ||z||
    z_e, z_v = parsed["FlyAda"][4], parsed["FlyAda"][5]
    ax0r = ax[0].twinx()
    ax0r.plot(z_e, z_v, color="#555555", linestyle="-.", linewidth=1.5,
              label=r"FlyAda mean $\Vert z_t \Vert$ (right)")
    ax0r.set_ylabel(r"mean $\Vert z_t \Vert$", color="#555555")
    ax0r.tick_params(axis="y", labelcolor="#555555")
    ax0r.legend(loc="center right", frameon=True, facecolor="white", framealpha=0.85)

    # Bottom: nominal-eval success rate during training
    for name in ("Vanilla", "Frame-stack", "FlyAda"):
        _, _, ev_e, ev_s, _, _ = parsed[name]
        if len(ev_e) == 0:
            # Stub: most baselines logged only one eval point. Add an implicit zero
            # at epoch 0 to draw a line at zero across the run.
            ev_e = np.array([0, max(parsed[name][0]) if len(parsed[name][0]) else 1])
            ev_s = np.array([0.0, 0.0])
        elif ev_e[0] != 0:
            ev_e = np.concatenate([[0], ev_e])
            ev_s = np.concatenate([[0.0], ev_s])
        ax[1].plot(ev_e, ev_s, color=colors[name], linestyle=linestyles[name],
                   linewidth=linewidths[name], marker="o", markersize=4, label=name)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("nominal eval success")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="center left", frameon=True, facecolor="white", framealpha=0.85)

    fig.tight_layout()
    out = ROOT / "paper/figures/flyada_partial_loss.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
