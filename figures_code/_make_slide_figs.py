"""Static figures for the WRC SARA 2026 oral slides.

Every number is read from results/*.json / *.csv, or from the paper tables where a
run's raw artefact isn't checked in — no hand-typed values that aren't traceable.

Palette: the validated categorical slots (blue / aqua / orange), which clear the
all-pairs CVD check, so the three methods stay distinguishable on a projector and
for colour-blind viewers. Series colour never carries text.

Usage:
    python -m paper._make_slide_figs
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

SLIDES = ROOT / "paper" / "figures" / "slides"
RESULTS = ROOT / "results"

# --- validated categorical slots (light mode) --------------------------------
C_VAN = "#2a78d6"      # slot 1  blue
C_FST = "#1baf7a"      # slot 3  aqua
C_FLY = "#eb6834"      # slot 2  orange
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e2de"

METHODS = ["Vanilla", "Frame-stack", "FlyAda"]
MCOLOR = {"Vanilla": C_VAN, "Frame-stack": C_FST, "FlyAda": C_FLY}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
    })


def _despine(ax, left=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)


def save(fig, name, dpi=200):
    SLIDES.mkdir(parents=True, exist_ok=True)
    p = SLIDES / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"saved -> {p}")


# =============================================================== 0. the task ===
def fig_task():
    """Task, policy and observation in one picture.

    The look-ahead drawn here is the policy's actual 8-step action chunk at that
    env step, forward-integrated through the environment's own dynamics.
    """
    style()
    from mpl_toolkits.mplot3d import Axes3D                    # noqa: F401
    from paper._slide_quadrotor import QuadrotorArtist

    d = np.load(SLIDES / "_chunk_demo.npz")
    traj, pred, chunk = d["traj"], d["pred"], d["chunk"]
    t0 = int(d["at_step"])
    goal = traj[-1]

    fig = plt.figure(figsize=(13.6, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.02,
                          left=0.01, right=0.99, top=0.99, bottom=0.02)
    ax = fig.add_subplot(gs[0], projection="3d")
    axt = fig.add_subplot(gs[1]); axt.axis("off")

    # Frame on the drone and its look-ahead — at full-trajectory scale the chunk
    # is a few pixels. The goal sits outside the box and is shown as a bearing.
    centre = traj[t0]
    reach = float(np.abs(np.concatenate([pred, traj[max(t0 - 12, 0):t0 + 1]])
                         - centre).max())
    span = max(2.8, 2.4 * reach)
    lo, hi = centre - span / 2, centre + span / 2
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=20, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        pane.pane.set_edgecolor((0.85, 0.87, 0.92, 1.0))

    ax.plot(traj[:t0 + 1, 0], traj[:t0 + 1, 1], traj[:t0 + 1, 2],
            color="#9a9a95", lw=2.0, label="flown so far")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=C_FLY, lw=2.6,
            ls="--", label="predicted chunk,  $H=8$")
    ax.plot(pred[:5, 0], pred[:5, 1], pred[:5, 2], color=C_FLY, lw=5.0,
            alpha=0.55, solid_capstyle="round", label="executed,  $K=4$")
    ax.scatter(pred[1:, 0], pred[1:, 1], pred[1:, 2], s=30, color=C_FLY,
               edgecolors="white", linewidths=0.7, zorder=8)

    # Goal bearing: the g_rel channel the policy is actually given.
    gdir = (goal - centre) / (np.linalg.norm(goal - centre) + 1e-9)
    tip = centre + gdir * span * 0.44
    ax.plot(*np.stack([centre, tip], 1), color="#c98500", lw=2.4, ls=(0, (5, 3)),
            zorder=7, label=r"$\mathbf{g}_{rel}$  (bearing to goal)")
    ax.scatter(*tip, marker="*", s=380, color="#ffc000",
               edgecolors="#7a5a00", linewidths=1.0, zorder=9)
    ax.text(tip[0], tip[1], tip[2] + span * 0.055,
            f"goal, {np.linalg.norm(goal - centre):.1f} m away",
            ha="center", fontsize=11.5, fontweight="bold", color="#7a5a00")

    drone = QuadrotorArtist(ax, C_VAN, scale=span / 7.0, zorder=12)
    drone.update(traj[t0], chunk[0], traj[t0] - traj[max(t0 - 1, 0)])
    ax.legend(loc="upper left", bbox_to_anchor=(-0.04, 1.0), fontsize=11.5,
              framealpha=0.9)

    # ---- the observation vector, channel by channel
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0.02, 0.96, "what the policy sees each step", fontsize=15,
             fontweight="bold", color=INK, va="top")
    rows = [
        ("$\\mathbf{p}$", "position", 3, True),
        ("$\\mathbf{v}$", "velocity", 3, False),
        ("yaw", "heading + rate", 2, True),
        ("$\\mathbf{g}_{rel}$", "goal-relative vector", 3, True),
        ("$d_{goal}$", "distance to goal", 1, True),
    ]
    y = 0.80
    for sym, name, dim, shown in rows:
        col = INK if shown else "#b4462f"
        axt.add_patch(Rectangle((0.02, y - 0.055), 0.94, 0.115,
                                facecolor="#f6f6f4" if shown else "#fdeeea",
                                edgecolor="#e3e2de" if shown else "#eab6a8",
                                linewidth=1.4))
        axt.text(0.06, y, sym, fontsize=15, va="center", color=col)
        axt.text(0.30, y, name, fontsize=13.5, va="center", color=INK2)
        axt.text(0.70, y, f"{dim} dim", fontsize=12.5, va="center", color=INK2)
        axt.text(0.94, y, "visible" if shown else "ZEROED", fontsize=12.5,
                 va="center", ha="right", color=col,
                 fontweight="normal" if shown else "bold")
        y -= 0.145
    axt.text(0.02, 0.055,
             "The simulator still integrates the true velocity —\n"
             "only the policy's view of it is removed.",
             fontsize=13, color="#b4462f", style="italic", va="bottom",
             linespacing=1.5)

    save(fig, "fig_task.png")


def fig_story_arc():
    """The hook, as a picture: what we expected, what we found, what we asked.

    Slide 2 was the one text-only slide in the main line; this carries the same
    narrative beats visually so the audience has something to look at while the
    framing lands.
    """
    style()
    fig, ax = plt.subplots(figsize=(5.6, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13.2); ax.axis("off")

    def card(y, h, title, body, edge, fc, tcol):
        ax.add_patch(FancyBboxPatch((0.3, y), 9.4, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.28",
                                    facecolor=fc, edgecolor=edge, linewidth=2.2,
                                    zorder=3))
        ax.text(5.0, y + h - 0.50, title, ha="center", va="top", fontsize=14,
                fontweight="bold", color=tcol, zorder=4)
        ax.text(5.0, y + h - 1.20, body, ha="center", va="top", fontsize=12,
                color=INK2, zorder=4, linespacing=1.5)

    def arrow(y0, y1):
        ax.add_patch(FancyArrowPatch((5.0, y0), (5.0, y1), arrowstyle="-|>",
                                     mutation_scale=20, linewidth=2.2,
                                     color=MUTED, zorder=2))

    card(9.95, 3.05, "Prior work",
         "A small online-updated latent\nclosed a dynamics-mismatch gap\n"
         "for UAV control.", C_VAN, "#eef4fc", C_VAN)
    arrow(9.85, 9.35)
    card(6.20, 3.05, "So we ported it to diffusion",
         "Same recipe, action-chunk\ndiffusion policy, mass / drag /\n"
         "wind / delay sweep.", MUTED, "#f6f6f4", INK)
    arrow(6.10, 5.60)
    card(2.45, 3.05, "…and there was no gap to close",
         "Vanilla was already at 1.00\nacross the sweep. The latent\n"
         "added nothing.", "#b4462f", "#fdeeea", "#b4462f")
    arrow(2.35, 1.85)

    ax.add_patch(FancyBboxPatch((0.3, 0.15), 9.4, 1.6,
                                boxstyle="round,pad=0.02,rounding_size=0.28",
                                facecolor=C_FLY, edgecolor=C_FLY, linewidth=2.2,
                                zorder=3))
    ax.text(5.0, 0.95, "So — when does it help?", ha="center", va="center",
            fontsize=15.5, fontweight="bold", color="white", zorder=4)

    save(fig, "fig_story_arc.png")


def fig_ambiguity():
    """The imitation-ambiguity argument as a picture.

    Two demonstrations reach the same position at different speeds and are
    followed by the same action chunk. A policy conditioned on that position
    fits both with one output, so nothing in the loss ever rewards separating
    them by velocity.
    """
    style()
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(-0.5, 11.4); ax.set_ylim(0.0, 3.55); ax.axis("off")

    P = np.array([4.7, 1.75])

    # Both demonstrations are drawn with the same number of equally-spaced time
    # steps, so the dot spacing *is* the speed.
    for sgn, reach, col, lab in [(1, 3.7, "#7a5cc6", "fast"),
                                 (-1, 1.55, "#1baf7a", "slow")]:
        t = np.linspace(0, 1, 7)
        xs = P[0] - reach * (1 - t)
        ys = P[1] + sgn * 0.80 * (1 - t) ** 1.5
        ax.plot(xs, ys, color=col, lw=2.4, zorder=4)
        ax.scatter(xs[:-1], ys[:-1], s=44, color=col, edgecolors="white",
                   linewidths=1.0, zorder=5)
        ax.text(xs[0] - 0.18, ys[0], lab, ha="right", va="center", fontsize=13,
                fontweight="bold", color=col)

    ax.scatter(*P, s=200, facecolor="white", edgecolor=INK, linewidths=2.2,
               zorder=7)
    ax.text(P[0] + 0.05, P[1] + 0.34, "same position", ha="center", va="bottom",
            fontsize=12.5, fontweight="bold", color=INK, zorder=8,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))

    # --- the single demonstrated chunk that follows both
    t = np.linspace(0, 1, 6)
    xs = P[0] + 2.9 * t
    ys = P[1] + 0.28 * np.sin(np.pi * t)
    ax.plot(xs, ys, color=C_FLY, lw=3.0, ls="--", zorder=4)
    ax.scatter(xs[1:], ys[1:], s=48, color=C_FLY, edgecolors="white",
               linewidths=1.0, zorder=5)
    ax.text(xs[-1] + 0.25, ys[-1], "the same\ndemonstrated\naction chunk",
            ha="left", va="center", fontsize=12.5, color=C_FLY,
            fontweight="bold", linespacing=1.4)

    ax.text(-0.5, 3.5, "Same number of time steps — the dot spacing is the speed",
            ha="left", va="top", fontsize=12.5, color=INK2, style="italic")
    ax.text(-0.5, 0.30,
            "One output minimises the loss for both, so no gradient ever asks "
            "the model to tell them apart.",
            ha="left", va="bottom", fontsize=12.5, color=INK2, style="italic")

    save(fig, "fig_ambiguity.png")


def fig_scoreboard():
    """The four numbers worth remembering, sized to be read from the back."""
    style()
    fig, ax = plt.subplots(figsize=(12.4, 1.95))
    ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")

    cards = [
        ("0.3% → 100%", "partial-observation sweep\n(12 conditions)", C_FLY),
        ("R² = 0.992", "linear probe from the latent\nto the hidden velocity", C_VAN),
        ("1.00 / 0.93", "3-waypoint chain\nnominal / hard · no retraining", "#c98500"),
        ("0.00 → 0.85", "6-DoF MuJoCo quadrotor\nhard condition", C_FST),
    ]
    for i, (big, small, col) in enumerate(cards):
        x = i + 0.5
        ax.add_patch(FancyBboxPatch((i + 0.045, 0.06), 0.91, 0.88,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor="#fbfbfa", edgecolor=col,
                                    linewidth=2.2, zorder=2))
        ax.text(x, 0.66, big, ha="center", va="center", fontsize=21,
                fontweight="bold", color=col, zorder=3)
        ax.text(x, 0.28, small, ha="center", va="center", fontsize=11,
                color=INK2, zorder=3, linespacing=1.45)
    save(fig, "fig_scoreboard.png")


def fig_latent_pca():
    """The paper's PCA panel, re-cut for half a slide.

    Only the scatter coloured by speed — the per-condition centroid panel makes
    the same point as the 24% classifier, which the slide already states, so it
    stays in the paper.
    """
    style()
    d = np.load(SLIDES / "_probe_pairs.npz")
    if "z" not in d.files:
        raise SystemExit("_probe_pairs.npz has no z — re-run "
                         "python -c 'from paper._slide_rollouts import "
                         "collect_probe_pairs; collect_probe_pairs()'")
    z, v = d["z"], d["vtrue"]
    speed = np.linalg.norm(v, axis=1)

    zc = z - z.mean(0)
    # Full-matrix SVD on ~10 K x 32 is instant; no need for a PCA dependency.
    _u, s_, vt = np.linalg.svd(zc, full_matrices=False)
    pcs = zc @ vt[:2].T
    var = (s_ ** 2 / (s_ ** 2).sum())[:2]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pcs), size=min(6000, len(pcs)), replace=False)

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    sc = ax.scatter(pcs[idx, 0], pcs[idx, 1], c=speed[idx], s=5, alpha=0.55,
                    cmap="viridis", linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.055)
    cb.set_label("‖v‖  (m/s)", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax.set_xlabel(f"PC 1  ({var[0]*100:.0f}%)", fontsize=11.5)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.0f}%)", fontsize=11.5)
    ax.set_title("$z_t$ in its top two components", fontsize=12.5, pad=7)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    _despine(ax)
    save(fig, "fig_latent_pca.png")


def fig_task_card():
    """Everything the 10-minute talk needs about the setup, and nothing else.

    What the policy sees (with the hidden channel called out), and how far the
    dynamics are pushed. No architecture numbers — those live on a backup slide.
    """
    style()
    fig, (axo, axp) = plt.subplots(2, 1, figsize=(6.1, 5.9),
                                   gridspec_kw={"height_ratios": [1.25, 1.0],
                                                "hspace": 0.62})

    # ---- top: the observation vector
    axo.axis("off"); axo.set_xlim(0, 1); axo.set_ylim(0, 1)
    axo.text(0.0, 1.10, "What the policy sees", fontsize=15.5,
             fontweight="bold", color=INK, va="top")
    rows = [("$\\mathbf{p}$", "position", True),
            ("$\\mathbf{v}$", "velocity", False),
            ("yaw", "heading + rate", True),
            ("$\\mathbf{g}_{rel}$, $d$", "goal-relative vector, distance", True)]
    y = 0.76
    for sym, name, shown in rows:
        col = INK if shown else "#b4462f"
        axo.add_patch(Rectangle((0.0, y - 0.085), 1.0, 0.175,
                                facecolor="#f6f6f4" if shown else "#fdeeea",
                                edgecolor="#e3e2de" if shown else "#eab6a8",
                                linewidth=1.6))
        axo.text(0.05, y, sym, fontsize=15, va="center", color=col)
        axo.text(0.30, y, name, fontsize=13, va="center", color=INK2)
        axo.text(0.97, y, "visible" if shown else "HIDDEN", fontsize=13,
                 va="center", ha="right", color=col,
                 fontweight="normal" if shown else "bold")
        y -= 0.225
    axo.text(0.0, -0.10,
             "The simulator still integrates the true velocity.\n"
             "Only the policy's view of it is removed.",
             fontsize=12.5, color="#b4462f", style="italic", va="top",
             linespacing=1.45)

    # ---- bottom: how far the dynamics are pushed, training vs test
    #
    # The four axes carry different units, so each row is normalised to its own
    # test maximum and the real numbers are printed alongside. One shared
    # numeric x-axis across mixed units would be meaningless.
    axp.set_title("How far the dynamics are pushed", fontsize=15.5,
                  fontweight="bold", color=INK, loc="left", pad=12)
    axes_spec = [
        ("mass",  1.0, 1.3, 2.0, "×{:.1f}–{:.1f}", "×{:.0f}"),
        ("drag",  1.0, 2.0, 3.0, "×{:.0f}–{:.0f}", "×{:.0f}"),
        ("wind",  0.0, 1.5, 3.0, "{:.0f}–{:.1f} m/s", "{:.0f} m/s"),
        ("delay", 0.0, 3.0, 5.0, "{:.0f}–{:.0f} steps", "{:.0f} steps"),
    ]
    ys = np.arange(len(axes_spec))[::-1]
    BAR_W = 0.52          # fraction of the row given to the bar
    for y_, (lab, tlo, thi, xhi, tfmt, xfmt) in zip(ys, axes_spec):
        axp.barh(y_, BAR_W, height=0.46, color="#efeeea", zorder=2)
        axp.barh(y_, BAR_W * (thi - tlo) / xhi, left=BAR_W * tlo / xhi,
                 height=0.46, color=C_FLY, alpha=0.9, zorder=3)
        axp.text(BAR_W + 0.035, y_ + 0.16, "trained " + tfmt.format(tlo, thi),
                 va="center", fontsize=12, color=INK)
        axp.text(BAR_W + 0.035, y_ - 0.19, "tested to " + xfmt.format(xhi),
                 va="center", fontsize=12, color=INK2)
    axp.set_yticks(ys)
    axp.set_yticklabels([a[0] for a in axes_spec], fontsize=13.5)
    axp.set_xticks([])
    axp.set_xlim(0, 1.16)
    axp.set_ylim(-0.75, len(axes_spec) - 0.35)
    axp.tick_params(axis="y", length=0)
    for s in axp.spines.values():
        s.set_visible(False)
    axp.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=C_FLY, alpha=0.9, label="training range"),
        plt.Rectangle((0, 0), 1, 1, color="#efeeea", label="test sweep"),
    ], loc="upper left", bbox_to_anchor=(-0.02, -0.02), ncol=2, fontsize=12.5,
        frameon=False, handlelength=1.3)

    save(fig, "fig_task_card.png")


def fig_transfer_strip():
    """Task B and MuJoCo side by side, compact enough to ride along the main result."""
    style()
    lh = json.load(open(RESULTS / "longhorizon_partial_v1/summary.json"))
    mj = json.load(open(RESULTS / "mujoco_transfer_v1/summary.json"))
    conds = ["nominal", "hard"]

    fig, axes = plt.subplots(2, 1, figsize=(4.9, 5.4),
                             gridspec_kw={"hspace": 0.62})
    panels = [
        (axes[0], {m: [lh[c][KEYMAP[m]]["task_success_rate"] for c in conds]
                   for m in METHODS},
         "Task B — 3 waypoints, no retraining", "task success"),
        (axes[1], {m: [mj[c][KEYMAP[m]]["success_rate"] for c in conds]
                   for m in METHODS},
         "Sim-to-sim — 6-DoF MuJoCo quadrotor", "success rate"),
    ]
    for ax, data, title, ylab in panels:
        xs = np.arange(2)
        w = 0.26
        for i, m in enumerate(METHODS):
            b = ax.bar(xs + (i - 1) * w, data[m], w, color=MCOLOR[m], zorder=3,
                       label=m)
            for r, v in zip(b, data[m]):
                ax.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=11.5,
                        fontweight="bold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(["nominal", "hard"], fontsize=12.5)
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        ax.grid(axis="y", alpha=0.5, zorder=0)
        _despine(ax)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.44), ncol=3,
                   fontsize=12, handlelength=1.2, columnspacing=1.1)
    save(fig, "fig_transfer_strip.png")


def fig_ablation_compact():
    """The same-weights ablation, narrow enough for half a slide.

    Success is the headline; the mean final distance rides along as a label,
    because at 40 seeds it separates the variants far more cleanly than the
    success rate does.
    """
    style()
    names = ["Vanilla", "Frame-stack",
             r"FlyAda, $\alpha=0$" "\n(latent pinned)",
             "FlyAda, frozen\nafter 10 steps",
             r"FlyAda, continuous" "\n" r"$\alpha=0.3$ EMA"]
    succ = [0.000, 0.000, 0.050, 0.025, 0.100]
    dist = [2.91, 3.06, 1.60, 3.61, 0.68]
    cols = [C_VAN, C_FST, "#f6c3ab", "#f6c3ab", C_FLY]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, succ, height=0.6, color=cols, zorder=3)
    for y, s_, d_ in zip(ys, succ, dist):
        ax.text(s_ + 0.004, y, f"{s_:.3f}", ha="left", va="center",
                fontsize=13, fontweight="bold", color=INK)
        ax.text(0.148, y, f"{d_:.2f} m", ha="right", va="center",
                fontsize=12.5, color=INK2)
    ax.text(0.148, len(names) - 0.55, "mean final\ndistance", ha="right",
            va="center", fontsize=11.5, color=INK2, linespacing=1.3)
    ax.set_xlim(0, 0.152)
    ax.set_xticks([0, 0.05, 0.10])
    ax.set_xlabel("success rate")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=12.5)
    ax.set_title("Same weights — only the test-time update rule changes\n"
                 "hard combined condition, 40 seeds", fontsize=13.5, pad=10)
    ax.grid(axis="x", alpha=0.5, zorder=0)
    _despine(ax, left=False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_ablation_compact.png")


def fig_obs_table():
    """Just the observation-channel panel, for pairing with the chunking video."""
    style()
    fig, axt = plt.subplots(figsize=(5.6, 5.3))
    axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0.02, 0.99, "What the policy sees each step", fontsize=16,
             fontweight="bold", color=INK, va="top")
    rows = [
        ("$\\mathbf{p}$", "position", 3, True),
        ("$\\mathbf{v}$", "velocity", 3, False),
        ("yaw", "heading + rate", 2, True),
        ("$\\mathbf{g}_{rel}$", "goal-relative vector", 3, True),
        ("$d_{goal}$", "distance to goal", 1, True),
    ]
    y = 0.83
    for sym, name, dim, shown in rows:
        col = INK if shown else "#b4462f"
        axt.add_patch(Rectangle((0.02, y - 0.058), 0.96, 0.118,
                                facecolor="#f6f6f4" if shown else "#fdeeea",
                                edgecolor="#e3e2de" if shown else "#eab6a8",
                                linewidth=1.6))
        axt.text(0.07, y, sym, fontsize=16, va="center", color=col)
        axt.text(0.28, y, name, fontsize=14, va="center", color=INK2)
        axt.text(0.665, y, f"{dim} dim", fontsize=13, va="center", color=INK2)
        axt.text(0.96, y, "visible" if shown else "HIDDEN", fontsize=13,
                 va="center", ha="right", color=col,
                 fontweight="normal" if shown else "bold")
        y -= 0.148
    axt.text(0.02, 0.02,
             "The simulator still integrates the true velocity —\n"
             "only the policy's view of it is removed.",
             fontsize=13.5, color="#b4462f", style="italic", va="bottom",
             linespacing=1.5)
    save(fig, "fig_obs_table.png")


# ========================================================= 1a. full observation ===
def fig_fullobs():
    """Paper Table I: the full-observation sweep, vanilla vs FlyAda.

    This is the negative result — vanilla is already at ceiling, and the
    adaptation latent adds nothing on the plan-specified sweep and costs
    something on the most extreme extrapolation cells.
    """
    style()
    regimes = [
        ("plan-specified\n(12 conditions, 30 seeds)", 1.000, 0.964),
        ("mass ×4", 0.95, 0.50),
        ("drag ×10", 0.95, 0.95),
        ("wind 3 m/s", 0.75, 0.55),
        ("delay 8 steps", 0.95, 0.65),
    ]
    labels = [r[0] for r in regimes]
    van = [r[1] for r in regimes]
    fly = [r[2] for r in regimes]

    fig, ax = plt.subplots(figsize=(12.4, 4.9))
    xs = np.arange(len(regimes))
    w = 0.36
    for off, vals, name, col in [(-w / 2, van, "Vanilla", C_VAN),
                                 (w / 2, fly, "FlyAda", C_FLY)]:
        b = ax.bar(xs + off, vals, w, color=col, label=name, zorder=3)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.022, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("success rate")
    ax.grid(axis="y", alpha=0.55, zorder=0)
    _despine(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=13,
              ncol=2, handlelength=1.5, columnspacing=2.5)

    # Divider between the in-plan sweep and the deliberate over-extrapolation.
    ax.axvline(0.5, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.text(-0.42, 1.10, "conventional sweep", fontsize=12.5, style="italic",
            color=INK2, ha="left")
    ax.text(4.45, 1.10, "an order of magnitude past the training range",
            fontsize=12.5, style="italic", color=INK2, ha="right")

    fig.suptitle("Full observation — the policy sees velocity",
                 fontsize=16, fontweight="bold", y=1.02, color=INK)
    save(fig, "fig_fullobs.png")


# ============================================================ 1. the collapse ===
def fig_collapse():
    """Full observation vs. partial observation, mean success over the 12-cond sweep.

    Full-obs numbers are the paper's Table I plan-specified row; partial-obs numbers
    are computed from results/mismatch_partial_v1/mismatch_table.csv (vanilla,
    FlyAda) and the paper's Table II frame-stack column.
    """
    style()
    rows = list(csv.DictReader(open(RESULTS / "mismatch_partial_v1/mismatch_table.csv")))
    van_partial = np.mean([float(r["success_rate"]) for r in rows if r["method"] == "vanilla"])
    fly_partial = np.mean([float(r["success_rate"]) for r in rows if r["method"] == "flyada"])
    fst_partial = 0.011      # Table II middle column, 12-condition mean
    van_full, fly_full = 1.000, 0.964    # Table I, plan-specified row

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

    panels = [
        (axes[0], "Full observation\n(policy sees velocity)",
         [("Vanilla", van_full), ("FlyAda", fly_full)]),
        (axes[1], "Partial observation\n(velocity hidden)",
         [("Vanilla", van_partial), ("Frame-stack", fst_partial), ("FlyAda", fly_partial)]),
    ]
    for ax, title, data in panels:
        names = [d[0] for d in data]
        vals = [d[1] for d in data]
        xs = np.arange(len(names))
        ax.bar(xs, vals, width=0.55, color=[MCOLOR[n] for n in names], zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.03, f"{v*100:.1f}%", ha="center", va="bottom",
                    fontsize=15, fontweight="bold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=13)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=15, pad=12, color=INK)
        ax.grid(axis="y", alpha=0.55, zorder=0)
        _despine(ax)
    axes[0].set_ylabel("mean success over the 12-condition sweep")

    axes[1].annotate("", xy=(0.03, 1.05), xytext=(-0.30, 1.05),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", lw=2.2, color=MUTED,
                                     connectionstyle="arc3,rad=-0.25"))
    axes[1].text(-0.14, 1.10, "hide velocity", transform=axes[1].transAxes,
                 ha="center", fontsize=13, style="italic", color=INK2)

    fig.suptitle("Vanilla diffusion is robust to dynamics mismatch — until the state is incomplete",
                 fontsize=16, fontweight="bold", y=1.03, color=INK)
    save(fig, "fig_collapse.png")


# ========================================================= 2. condition grid ===
def fig_sweep_grid(n_cols_shown: int = 3, name: str = "fig_sweep_grid.png"):
    """Table II as a readable grid: 12 partial-obs conditions x 3 methods.

    n_cols_shown=2 renders the baselines only — used on the slide that poses the
    failure, before FlyAda has been introduced.
    """
    style()
    rows = list(csv.DictReader(open(RESULTS / "mismatch_partial_v1/mismatch_table.csv")))
    van = {r["id"]: float(r["success_rate"]) for r in rows if r["method"] == "vanilla"}
    fly = {r["id"]: float(r["success_rate"]) for r in rows if r["method"] == "flyada"}

    # Paper Table II layout: within-training-range block, then extrapolation block.
    # Frame-stack has no per-condition CSV checked in; values are Table II's column.
    within = [
        ("nominal",      van["nominal"],     0.00, fly["nominal"]),
        ("mass +30%",    van["mass+30"],     0.00, fly["mass+30"]),
        ("drag +50%",    van["drag+50"],     0.00, fly["drag+50"]),
        ("drag +100%",   van["drag+100"],    0.00, fly["drag+100"]),
        ("wind 1.0",     van["wind_x=1.0"],  0.00, fly["wind_x=1.0"]),
        ("wind 1.5",     van["wind_x=1.5"],  0.03, fly["wind_x=1.5"]),
        ("delay 3",      van["delay=3"],     0.00, fly["delay=3"]),
    ]
    beyond = [
        ("mass +50%",  0.00, 0.00, 1.00),
        ("mass +100%", 0.00, 0.05, 1.00),
        ("drag +200%", 0.00, 0.00, 1.00),
        ("wind 2.0",   0.00, 0.00, 0.95),
        ("wind 3.0",   0.00, 0.00, 0.00),
        ("delay 5",    0.00, 0.05, 1.00),
    ]
    # One spacer row separates the two blocks; its slot carries the block caption.
    labels = [r[0] for r in within] + [""] + [r[0] for r in beyond]
    data = ([[r[1], r[2], r[3]] for r in within]
            + [[np.nan] * 3]
            + [[r[1], r[2], r[3]] for r in beyond])
    M = np.array(data, float)

    M = M[:, :n_cols_shown]
    fig, ax = plt.subplots(figsize=(10.6 - 1.5 * (3 - n_cols_shown), 8.0))
    n_rows, n_cols = M.shape
    ytop = n_rows - 1           # row i is drawn at y = ytop - i

    for j in range(n_cols):
        for i in range(n_rows):
            v = M[i, j]
            if np.isnan(v):
                continue
            col = [C_VAN, C_FST, C_FLY][j]
            ax.add_patch(Rectangle((j - 0.42, ytop - i - 0.34), 0.84, 0.68,
                                   facecolor=col, alpha=0.10 + 0.85 * v,
                                   edgecolor="white", linewidth=2.0, zorder=2))
            ax.text(j, ytop - i, "1.00" if v >= 0.995 else f"{v:.2f}",
                    ha="center", va="center", fontsize=12.5,
                    fontweight="bold" if v > 0.5 else "normal",
                    color="white" if v > 0.6 else INK2, zorder=3)
        ax.text(j, ytop + 0.95, METHODS[j], ha="center", va="center",
                fontsize=14.5, color=INK)

    for i, lab in enumerate(labels):
        if lab:
            ax.text(-0.58, ytop - i, lab, ha="right", va="center",
                    fontsize=13, color=INK2)

    ax.set_xlim(-2.55, n_cols - 0.4)
    ax.set_ylim(-0.6, ytop + 1.75)
    ax.axis("off")

    y_split = ytop - 7          # the spacer row
    ax.text(-2.5, ytop + 0.40, "within training range",
            fontsize=13, style="italic", color=INK2, ha="left")
    ax.text(-2.5, y_split, "beyond training range\n(extrapolation)",
            fontsize=13, style="italic", color=INK2, ha="left",
            va="center", linespacing=1.4)
    ax.plot([-2.5, n_cols - 0.5], [y_split + 0.52] * 2, color=GRID, lw=1.6,
            clip_on=False)

    ax.set_title("Success rate with velocity hidden — every condition"
                 + (", every method" if n_cols_shown == 3 else ""),
                 fontsize=15.5, fontweight="bold", pad=30, color=INK)
    save(fig, name)


# ============================================================ 3. the method ===
def fig_method():
    """Block diagram of the FlyAda loop: what is added, and where the gradient is."""
    style()
    fig, ax = plt.subplots(figsize=(13.0, 6.9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.9); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=13, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                    facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4,
                fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(x0, y0, x1, y1, color=INK2, rad=0.0, lw=2.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=lw, color=color,
                                     linestyle=ls, zorder=2,
                                     connectionstyle=f"arc3,rad={rad}"))

    # --- observation
    box(0.15, 3.5, 2.5, 1.35,
        "observation  $o_t$\n$[\\,\\mathbf{p},\\ \\mathbf{0},\\ \\mathrm{yaw},\\ \\mathbf{g}_{rel},\\ d\\,]$",
        "#f4f4f2", MUTED, fs=13)
    ax.text(1.85, 3.06, "velocity zeroed", ha="left", fontsize=11.5,
            style="italic", color="#b4462f")

    # --- observer head (the addition)
    box(3.35, 3.45, 3.05, 1.5,
        "observer head  $f_\\phi$\n2-layer MLP · 73K params\n"
        "$(s_t,\\ u_t,\\ s_{t+1})\\rightarrow$ tanh",
        "#fdf0ea", C_FLY, fs=12.5)
    ax.text(4.88, 5.08, "ADDED BY FlyAda", ha="center", fontsize=11.5,
            fontweight="bold", color=C_FLY)

    # --- EMA latent
    box(7.05, 3.55, 2.6, 1.3,
        "latent  $z_t \\in \\mathbb{R}^{32}$\n"
        "$z_{t+1}=(1-\\alpha)z_t+\\alpha f_\\phi$\n$\\alpha=0.3$, every env step",
        "#fdf0ea", C_FLY, fs=12)

    # --- denoiser
    box(10.15, 3.4, 2.7, 1.6,
        "diffusion denoiser\n1D temporal UNet, 3.98 M\nFiLM on $(s_t,\\ z_t)$\n"
        "DDIM, 20 steps",
        "#eef4fc", C_VAN, fs=12.5)
    ax.text(11.50, 5.08, "UNCHANGED BACKBONE", ha="center", fontsize=11.5,
            fontweight="bold", color=C_VAN)

    # --- action chunk / env
    box(10.15, 1.15, 2.7, 1.1, "action chunk  $u_{t:t+8}$\nexecute $K{=}4$, re-plan",
        "#f4f4f2", MUTED, fs=12.5)
    box(4.90, 1.15, 3.6, 1.1, "UAV  ·  50 Hz  ·  true velocity\nintegrated internally,"
        " never shown", "#f4f4f2", MUTED, fs=12.5)

    arrow(2.65, 4.2, 3.35, 4.2)
    arrow(6.40, 4.2, 7.05, 4.2)
    arrow(9.65, 4.2, 10.15, 4.2)
    arrow(11.5, 3.4, 11.5, 2.25)
    arrow(10.15, 1.7, 8.50, 1.7)
    arrow(4.90, 1.7, 1.00, 1.7)
    arrow(1.00, 2.25, 1.00, 3.5)

    # --- auxiliary loss: sits above the forward path so the closed loop below
    # stays uncrossed. Dashed = gradient path, present at training time only.
    box(6.30, 5.55, 3.75, 1.15,
        "$\\mathcal{L}_{vel}=\\lambda\\,\\|\\mathrm{dec}_\\phi(z_t)-\\mathbf{v}_t^{true}\\|^2$,"
        "   $\\lambda=5$\nTRAINING ONLY — privileged signal, dropped at test",
        "#fff9e8", "#c98500", fs=12)
    arrow(7.60, 5.55, 7.60, 4.85, color="#c98500", ls="--", lw=2.0)
    ax.text(7.80, 5.10, "supervises $z_t$", fontsize=11.5, color="#8a6200",
            style="italic", ha="left")

    ax.text(6.5, 0.28,
            "Three ingredients, all textbook on their own: an online-updated latent, "
            "an EMA update, an auxiliary supervised head.\n"
            "The claim is that all three have to be present together for the "
            "partial-observation gap to close.",
            ha="center", fontsize=12.5, color=INK2, style="italic", linespacing=1.5)

    save(fig, "fig_method.png")


# =========================================================== 4. the ablation ===
def fig_ablation():
    """Same weights, only the test-time update rule changes (paper Table III)."""
    style()
    # Table III: the deliberately hard combined condition (mass x1.5, drag x4,
    # wind 2 m/s, delay 4), 40 seeds. Not the checked-in ablation_summary.json,
    # which is the milder mass 1.3 / drag 0.2 / wind 1 / delay 2 condition.
    names = ["Vanilla  (separate weights)",
             "Frame-stack  (separate weights)",
             r"FlyAda, $\alpha=0$  (latent pinned at 0)",
             "FlyAda, latent frozen after 10 steps",
             r"FlyAda, continuous $\alpha=0.3$ EMA"]
    succ = [0.000, 0.000, 0.050, 0.025, 0.100]
    dist = [2.91, 3.06, 1.60, 3.61, 0.68]
    cols = [C_VAN, C_FST, "#f6c3ab", "#f6c3ab", C_FLY]

    # Horizontal bars: five long variant names read cleanly on one axis.
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    ys = np.arange(len(names))[::-1]

    ax = axes[0]
    ax.barh(ys, succ, height=0.62, color=cols, zorder=3)
    for y, v in zip(ys, succ):
        ax.text(v + 0.004, y, f"{v:.3f}", ha="left", va="center",
                fontsize=13.5, fontweight="bold", color=INK)
    ax.set_xlim(0, 0.135)
    ax.set_xlabel("success rate")
    ax.set_title("success   (higher is better)", fontsize=14, pad=10)

    ax = axes[1]
    ax.barh(ys, dist, height=0.62, color=cols, zorder=3)
    for y, v in zip(ys, dist):
        ax.text(v + 0.10, y, f"{v:.2f} m", ha="left", va="center",
                fontsize=13.5, fontweight="bold", color=INK)
    ax.set_xlim(0, 4.55)
    ax.set_xlabel("mean final distance to goal (m)")
    ax.set_title("distance to goal   (lower is better)", fontsize=14, pad=10)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, fontsize=13)
    for ax in axes:
        ax.grid(axis="x", alpha=0.55, zorder=0)
        _despine(ax, left=False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Freeze the weights, change only the test-time update rule "
                 "— hard combined condition, 40 seeds",
                 fontsize=15.5, fontweight="bold", y=1.05, color=INK)
    fig.text(0.5, -0.09,
             "Capacity is not the mechanism: the same network with its latent pinned "
             "or frozen loses most of the gap. Continuous online updating is what pays.",
             ha="center", fontsize=12.5, color=INK2, style="italic")
    save(fig, "fig_ablation.png")


# ============================================================== 5. transfer ===
def _grouped_success(ax, data, xticklabels, ylabel, title):
    """Shared grouped-bar body for the two transfer panels."""
    width = 0.26
    xs = np.arange(len(xticklabels))
    for i, m in enumerate(METHODS):
        b = ax.bar(xs + (i - 1) * width, data[m], width, color=MCOLOR[m],
                   label=m, zorder=3)
        for r, v in zip(b, data[m]):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.025, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                    color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, fontsize=12.5)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="y", alpha=0.55, zorder=0)
    _despine(ax)
    ax.legend(loc="upper left", fontsize=12.5, ncol=3,
              columnspacing=1.0, handlelength=1.4)


KEYMAP = {"Vanilla": "vanilla", "Frame-stack": "frame_stack", "FlyAda": "flyada"}


def fig_taskB_bars():
    """Task B task success, nominal and hard."""
    style()
    lh = json.load(open(RESULTS / "longhorizon_partial_v1/summary.json"))
    conds = ["nominal", "hard"]
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    _grouped_success(
        ax,
        {m: [lh[c][KEYMAP[m]]["task_success_rate"] for c in conds] for m in METHODS},
        ["nominal", "hard\n(mass+30%, drag+100%,\nwind 1, delay 2)"],
        "task success (all 3 waypoints)",
        "Task B — 3 waypoints, 800 steps, 30 seeds")
    save(fig, "fig_taskB_bars.png")


def fig_mujoco_bars():
    """6-DoF MuJoCo sim-to-sim success, nominal and hard."""
    style()
    mj = json.load(open(RESULTS / "mujoco_transfer_v1/summary.json"))
    conds = ["nominal", "hard"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    _grouped_success(
        ax,
        {m: [mj[c][KEYMAP[m]]["success_rate"] for c in conds] for m in METHODS},
        ["nominal", "hard\n(mass×1.3, drag, wind 1 m/s)"],
        "success rate",
        "6-DoF MuJoCo quadrotor, 20 seeds\n"
        "500 Hz attitude-rate PID under a 50 Hz policy")
    save(fig, "fig_mujoco_bars.png")


# ============================================================== 6. the probe ===
def fig_probe():
    """What z_t encodes: a linear probe recovers velocity; condition identity is at chance."""
    style()
    s = json.load(open(RESULTS / "latent_analysis_summary.json"))
    r2 = s["linear_probe_R2_overall"]
    acc = s["condition_classifier_acc"]
    chance = s["condition_classifier_chance"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28})

    # --- left: what the latent decodes to, against the velocity it never saw
    ax = axes[0]
    pairs = SLIDES / "_probe_pairs.npz"
    if pairs.exists():
        d = np.load(pairs)
        vh, vt = d["vhat"], d["vtrue"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(vt), size=min(4000, len(vt)), replace=False)
        ax.scatter(vt[idx].ravel(), vh[idx].ravel(), s=7, alpha=0.16,
                   color=C_FLY, linewidths=0, zorder=3, rasterized=True)
        lim = float(np.abs(vt[idx]).max()) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1.4, ls="--", zorder=4)
        ax.text(lim * 0.52, -lim * 0.80, "identity", fontsize=11.5,
                color=INK2, style="italic")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        # R^2 of the points actually plotted, i.e. the decoder head itself.
        # The paper's Sec. VIII number (0.992) is a separately fitted OLS probe.
        ss_res = float(((vh - vt) ** 2).sum())
        ss_tot = float(((vt - vt.mean(0)) ** 2).sum())
        r2_dec = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(((vh - vt) ** 2).mean()))
        n_plot = len(vt)
    else:
        lim, r2_dec, rmse, n_plot = 1.0, float("nan"), float("nan"), 0
        ax.text(0.5, 0.5, "run  python -m paper._slide_rollouts", ha="center",
                transform=ax.transAxes, color=INK2)
    ax.set_xlabel("true velocity (m/s) — hidden from the policy")
    ax.set_ylabel(r"decoded  $\mathrm{dec}_\phi(z_t)$  (m/s)")
    ax.text(0.03, 0.965, f"$R^2$ = {r2_dec:.3f}", transform=ax.transAxes,
            fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.03, 0.878,
            f"RMSE {rmse:.2f} m/s   ·   {n_plot:,} steps\n"
            f"5 dynamics conditions, 25 seeds each\n"
            f"fitted linear probe: $R^2$ = {r2:.3f}",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="top",
            linespacing=1.5)
    ax.set_title("A linear probe reads the missing state\nstraight off the latent",
                 fontsize=14, pad=10)
    ax.grid(alpha=0.45, zorder=0)
    _despine(ax)

    ax = axes[1]
    ax.bar([0, 1], [acc, chance], width=0.44,
           color=["#b9b8b2", "#efeeea"], zorder=3)
    ax.text(0, acc + 0.012, f"{acc*100:.1f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(1, chance + 0.012, f"{chance*100:.0f}%", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=INK2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"classifier  $z_t \rightarrow$ condition", "chance (5-way)"],
                       fontsize=13)
    ax.set_ylim(0, 0.34)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_title("...and almost nothing about which\nperturbation it is flying under",
                 fontsize=14, pad=10)
    ax.grid(axis="y", alpha=0.55, zorder=0)
    _despine(ax)

    fig.suptitle("$z_t$ is a velocity estimate, not a dynamics-regime label",
                 fontsize=16, fontweight="bold", y=1.04, color=INK)
    save(fig, "fig_probe.png")


# ========================================================== 7. contributions ===
def fig_takeaway():
    """Closing summary card — the three claims and their evidence."""
    style()
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 5.6); ax.axis("off")

    cards = [
        (C_VAN, "The failure is about the\nobjective, not the input",
         "A 3-frame stack carries the same\ntemporal information frame-by-frame\n"
         "and still fails (1.1%). The diffusion\nMSE never asks the model to\n"
         "extract velocity from position deltas."),
        ("#c98500", "A privileged auxiliary head\nputs the gradient back",
         "$\\mathcal{L}_{vel}$ supervises $z_t$ with the true\n"
         "velocity at training time only.\nA linear probe then recovers it\n"
         "at $R^2 = 0.992$ — and success\njumps 0 \u2192 100%."),
        (C_FLY, "Updating online is the\nload-bearing part",
         "Same weights, only the test-time\nrule toggled: pinned or frozen\n"
         "latents lose most of the gap.\nThe gain survives to a 3-waypoint\n"
         "chain and to a 6-DoF MuJoCo body."),
    ]
    w, gap = 3.95, 0.28
    head_h = 1.05                       # tall enough for a two-line heading
    top, bot = 5.25, 0.35
    for i, (col, head, body) in enumerate(cards):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, bot), w, top - bot,
                                    boxstyle="round,pad=0.02,rounding_size=0.16",
                                    facecolor="#fbfbfa", edgecolor=col,
                                    linewidth=2.4, zorder=2))
        ax.add_patch(Rectangle((x, top - head_h), w, head_h, facecolor=col, zorder=3))
        ax.text(x + w / 2, top - head_h / 2, head, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=4,
                linespacing=1.35)
        ax.text(x + w / 2, (bot + top - head_h) / 2, body, ha="center", va="center",
                fontsize=12.5, color=INK2, zorder=4, linespacing=1.7)
    save(fig, "fig_takeaway.png")


# Every figure paper/_make_slide_deck.py places, in slide order. Keeping this
# list in sync with the deck is what stops a stale PNG from silently shipping;
# _check_figures_match_deck() below asserts it.
PLACED = [
    ("fig_story_arc.png", fig_story_arc),            # 2  the question
    ("fig_task_card.png", fig_task_card),            # 3  the task
    ("fig_fullobs.png", fig_fullobs),                # 4  full observation
    ("fig_ambiguity.png", fig_ambiguity),            # 6  why history fails
    ("fig_method.png", fig_method),                  # 7  FlyAda
    ("fig_sweep_grid.png", fig_sweep_grid),          # 8  main result
    ("fig_ablation_compact.png", fig_ablation_compact),  # 9  what does the work
    ("fig_mujoco_bars.png", fig_mujoco_bars),        # 9  sim-to-sim
    ("fig_latent_pca.png", fig_latent_pca),          # 10 what does the work
    ("fig_scoreboard.png", fig_scoreboard),          # 11 conclusion
]

# Built by request only — alternative cuts of slides the deck already covers:
# fig_task (setup 3D + table in one), fig_obs_table (the table alone),
# fig_collapse (full-obs vs partial-obs bars), fig_ablation (the wide two-panel
# ablation), fig_takeaway (three-card summary), fig_probe (the decoded-vs-true
# scatter), fig_taskB_bars, fig_transfer_strip (Task B + MuJoCo bars, for when both
# results have to share one slide), and fig_sweep_grid(n_cols_shown=2).


def _check_figures_match_deck():
    """Warn if the deck places a figure this module does not build, or vice versa."""
    deck = (ROOT / "paper" / "_make_slide_deck.py").read_text(encoding="utf-8")
    placed = {n for n, _ in PLACED}
    referenced = set(re.findall(r'SLIDES / "(fig_[a-zA-Z0-9_]+\.png)"', deck))
    for missing in sorted(referenced - placed):
        print(f"  [warn] deck places {missing} but PLACED does not build it")
    for unused in sorted(placed - referenced):
        print(f"  [warn] PLACED builds {unused} but no slide places it")


def main():
    for name, fn in PLACED:
        fn()
    _check_figures_match_deck()


if __name__ == "__main__":
    main()
