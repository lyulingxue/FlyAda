"""Render the animated 3D flight comparisons used in the WRC SARA 2026 oral slides.

Produces, from cached traces (paper/_slide_rollouts.py):

  taskB_race_nominal.mp4/.gif   three quadrotors on the same partial-obs Task B seed
  taskB_race_hard.mp4/.gif      same, under the hard combined dynamics condition
  belief_tracking.mp4/.gif      FlyAda's latent decoding the hidden velocity live
  *_still.png                   a final frame of each, as a projector fallback

Usage:
    python -m paper._make_slide_anims               # everything
    python -m paper._make_slide_anims --only race_nominal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                       # noqa: E402,F401

from paper._slide_quadrotor import QuadrotorArtist            # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover - ffmpeg is only needed for the .mp4 outputs
    pass

SLIDES = ROOT / "paper" / "figures" / "slides"

COLORS = {"Vanilla": "#2a78d6", "Frame-stack": "#1baf7a", "FlyAda": "#eb6834"}
STYLES = {"Vanilla": "--", "Frame-stack": ":", "FlyAda": "-"}
METHODS = ["Vanilla", "Frame-stack", "FlyAda"]

ADVANCE_TOL = 0.5


def slide_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def load(tag: str, task: str = "taskB") -> dict:
    f = SLIDES / f"_traces_{task}_{tag}.npz"
    if not f.exists():
        raise SystemExit(f"missing {f} — run:  python -m paper._slide_rollouts")
    d = np.load(f)
    return {k: d[k] for k in d.files}


def _save(anim, stem: str, fps: int, dpi: int, make_gif: bool = False):
    mp4 = SLIDES / f"{stem}.mp4"
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                                          "-profile:v", "high", "-preset", "slow"])
        anim.save(str(mp4), writer=writer, dpi=dpi)
        print(f"  saved -> {mp4}")
    except Exception as e:
        print(f"  [warn] mp4 failed ({e}); GIF only")
    if make_gif:
        gif = SLIDES / f"{stem}.gif"
        anim.save(str(gif), writer=PillowWriter(fps=min(fps, 20)), dpi=int(dpi * 0.8))
        print(f"  saved -> {gif}")


# ============================================================== Task B race ===
def render_race(tag: str, max_steps: int = 420, stride: int = 2,
                fps: int = 25, hold_s: float = 2.0, dpi: int = 100):
    """Three quadrotors, one seed, one scene, with live diagnostics on the right."""
    slide_style()
    d = load(tag)
    wps = d["waypoints"]
    n_wp = len(wps)

    traj = {m: d[f"{m}/pos"] for m in METHODS}
    acts = {m: d[f"{m}/act"] for m in METHODS}
    vels = {m: d[f"{m}/vel"] for m in METHODS}
    dgoal = {m: d[f"{m}/d_goal"] for m in METHODS}
    wpidx = {m: d[f"{m}/wp_idx"] for m in METHODS}

    T = min(max_steps, max(len(traj[m]) for m in METHODS))
    frames = np.arange(0, T, stride)
    n_hold = int(hold_s * fps)

    all_pts = np.concatenate([traj[m][:T] for m in METHODS] + [wps], axis=0)
    lo = all_pts.min(0) - 0.8
    hi = all_pts.max(0) + 0.8
    # Keep the box roughly cubic so the drones don't look stretched.
    span = float((hi - lo).max())
    mid = (hi + lo) / 2
    lo, hi = mid - span / 2, mid + span / 2

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1.0], height_ratios=[1, 1],
                          left=0.02, right=0.965, top=0.90, bottom=0.09,
                          wspace=0.18, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    cond = ("nominal dynamics" if tag == "nominal"
            else "hard condition: mass +30%, drag +100%, 1 m/s wind, 2-step delay")
    fig.suptitle(f"Task B — 3 waypoints, velocity hidden from the policy  ({cond})",
                 fontsize=15, fontweight="bold", y=0.965)

    # ---- static 3D scene furniture
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (m)", labelpad=6)
    ax.set_ylabel("y (m)", labelpad=6)
    ax.set_zlabel("z (m)", labelpad=2)
    ax.view_init(elev=20, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.75))
    except Exception:
        pass
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        pane.pane.set_edgecolor((0.85, 0.87, 0.92, 1.0))
    ax.grid(True)

    start = traj["FlyAda"][0]
    ax.scatter(*start, s=90, facecolor="white", edgecolor="black",
               linewidths=1.4, zorder=6)
    ax.text(start[0], start[1], start[2] - 0.55, "start", fontsize=10, ha="center")

    wp_marks, wp_labels = [], []
    for i, w in enumerate(wps):
        m = ax.scatter(*w, marker="*", s=300, color="#444444",
                       edgecolors="white", linewidths=1.0, zorder=7)
        wp_marks.append(m)
        wp_labels.append(ax.text(w[0], w[1], w[2] + 0.45, f"WP{i+1}",
                                 fontsize=11, fontweight="bold", ha="center"))

    # ---- per-method artists
    drones, trails, shadows = {}, {}, {}
    for m in METHODS:
        trails[m], = ax.plot([], [], [], color=COLORS[m], lw=2.4,
                             linestyle=STYLES[m], alpha=0.95, zorder=5)
        shadows[m], = ax.plot([], [], [], color=COLORS[m], lw=1.2,
                              alpha=0.20, zorder=2)
        drones[m] = QuadrotorArtist(ax, COLORS[m], scale=span / 14.0,
                                    zorder=12 if m == "FlyAda" else 10)

    handles = [plt.Line2D([], [], color=COLORS[m], lw=2.6, linestyle=STYLES[m],
                          label=m) for m in METHODS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 0.98),
              frameon=True, framealpha=0.9, fontsize=12)

    # ---- right column: distance to current waypoint + waypoints captured
    ax_d.set_title("distance to current waypoint", fontsize=12, pad=6)
    ax_d.set_xlabel("env step"); ax_d.set_ylabel("m")
    ax_d.set_xlim(0, T)
    ymax = max(float(np.nanmax(dgoal[m][:T])) for m in METHODS) * 1.08
    ax_d.set_ylim(0, ymax)
    ax_d.axhline(ADVANCE_TOL, color="#666666", ls=":", lw=1.3)
    ax_d.text(T * 0.985, ADVANCE_TOL + ymax * 0.05, "capture tol 0.5 m",
              fontsize=9.5, ha="right", va="bottom", color="#555555",
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    dlines = {m: ax_d.plot([], [], color=COLORS[m], lw=2.0,
                           linestyle=STYLES[m])[0] for m in METHODS}

    ax_p.set_title("waypoints captured", fontsize=12, pad=6)
    ax_p.set_xlim(0, n_wp + 0.02); ax_p.set_ylim(-0.6, len(METHODS) - 0.4)
    ax_p.set_yticks(range(len(METHODS)))
    ax_p.set_yticklabels(METHODS[::-1], fontsize=11)
    ax_p.set_xticks(range(n_wp + 1))
    ax_p.set_xlabel("count")
    ax_p.grid(axis="x", alpha=0.3)
    bars = ax_p.barh(range(len(METHODS)),
                     [0] * len(METHODS),
                     color=[COLORS[m] for m in METHODS[::-1]], height=0.55)
    btexts = [ax_p.text(0.06, i, "0/3", va="center", fontsize=11,
                        fontweight="bold", color="#333333") for i in range(len(METHODS))]

    step_txt = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=12,
                         fontweight="bold", color="#333333")

    def wp_captured(m, t):
        """How many waypoints this method has captured by env step t."""
        idx = wpidx[m][:min(t + 1, len(wpidx[m]))]
        if len(idx) == 0:
            return 0
        reached = int(idx.max())
        # wp_idx points at the *current* target; a success flags the final one.
        if bool(d[f"{m}/success"]) and t >= int(d[f"{m}/length"]):
            reached = n_wp
        return min(reached, n_wp)

    def draw(k):
        t = int(frames[min(k, len(frames) - 1)])
        for m in METHODS:
            tr = traj[m]
            tt = min(t, len(tr) - 1)
            sub = tr[: tt + 1]
            trails[m].set_data(sub[:, 0], sub[:, 1])
            trails[m].set_3d_properties(sub[:, 2])
            shadows[m].set_data(sub[:, 0], sub[:, 1])
            shadows[m].set_3d_properties(np.full(len(sub), lo[2]))
            drones[m].update(sub[-1], acts[m][min(tt, len(acts[m]) - 1)],
                             vels[m][tt], blade_phase=0.9 * k)

            dl = dgoal[m][: tt + 1]
            dlines[m].set_data(np.arange(len(dl)), dl)

        for j, m in enumerate(METHODS[::-1]):
            c = wp_captured(m, t)
            bars[j].set_width(c)
            btexts[j].set_text(f"{c}/{n_wp}")
            btexts[j].set_x(max(c + 0.08, 0.08))

        # Gold-fill a waypoint once FlyAda has captured it.
        got = wp_captured("FlyAda", t)
        for i, mk in enumerate(wp_marks):
            mk.set_color("#ffc000" if i < got else "#444444")
            mk.set_sizes([420 if i < got else 300])

        step_txt.set_text(f"env step {t}")
        return []

    total = len(frames) + n_hold
    print(f"[race:{tag}] {total} frames ({len(frames)} + {n_hold} hold), T={T}")
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    _save(anim, f"taskB_race_{tag}", fps, dpi)

    draw(len(frames) - 1)
    fig.savefig(SLIDES / f"taskB_race_{tag}_still.png", dpi=140)
    print(f"  saved -> {SLIDES / f'taskB_race_{tag}_still.png'}")
    plt.close(fig)


# ========================================================= belief tracking ===
def render_belief(tag: str = "hard", fps: int = 25, hold_s: float = 2.0,
                  dpi: int = 100, stride: int = 1):
    """FlyAda's latent decoding the velocity it is never shown."""
    slide_style()
    d = load(tag, task="taskA")

    pos = d["FlyAda/pos"]
    act = d["FlyAda/act"]
    vel = d["FlyAda/vel"]          # ground truth — hidden from the policy
    vhat = d["FlyAda/vhat"]        # dec_phi(z_t)
    z = d["FlyAda/z"]
    T = len(pos)

    frames = np.arange(0, T, stride)
    n_hold = int(hold_s * fps)

    lo, hi = pos.min(0) - 0.8, pos.max(0) + 0.8
    span = float((hi - lo).max()); mid = (hi + lo) / 2
    lo, hi = mid - span / 2, mid + span / 2

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.0],
                          left=0.03, right=0.965, top=0.885, bottom=0.09,
                          wspace=0.20, hspace=0.55)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axv = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    fig.suptitle("The observer latent recovers the velocity the policy never sees",
                 fontsize=15, fontweight="bold", y=0.965)

    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
    ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)"); ax3.set_zlabel("z (m)")
    ax3.view_init(elev=20, azim=-58)
    try:
        ax3.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        pane.pane.set_edgecolor((0.85, 0.87, 0.92, 1.0))

    goal = pos[-1]
    ax3.scatter(*goal, marker="*", s=380, color="#ffc000",
                edgecolors="#7a5a00", linewidths=1.0, zorder=7)
    ax3.text(goal[0], goal[1], goal[2] + 0.5, "goal", fontsize=11,
             fontweight="bold", ha="center")
    ax3.scatter(*pos[0], s=90, facecolor="white", edgecolor="black",
                linewidths=1.4, zorder=6)

    trail, = ax3.plot([], [], [], color=COLORS["FlyAda"], lw=2.6, zorder=5)
    shadow, = ax3.plot([], [], [], color=COLORS["FlyAda"], lw=1.2, alpha=0.20, zorder=2)
    drone = QuadrotorArtist(ax3, COLORS["FlyAda"], scale=span / 12.0, zorder=12)

    ax3.legend(handles=[
        plt.Line2D([], [], color=COLORS["FlyAda"], lw=2.6,
                   label="FlyAda (partial obs, hard condition)"),
    ], loc="upper left", bbox_to_anchor=(-0.02, 0.98), fontsize=12, framealpha=0.9)

    labels = ["$v_x$", "$v_y$", "$v_z$"]
    truel, hatl = [], []
    for i, a in enumerate(axv):
        a.set_xlim(0, T)
        m = float(max(np.abs(vel[:, i]).max(), np.abs(vhat[:, i]).max())) * 1.25 + 1e-6
        a.set_ylim(-m, m)
        a.set_ylabel(f"{labels[i]} (m/s)", fontsize=11)
        a.axhline(0, color="#999999", lw=0.8)
        tl, = a.plot([], [], color="#333333", lw=2.2, label="true (hidden)")
        hl, = a.plot([], [], color=COLORS["FlyAda"], lw=2.2, ls="--",
                     label=r"decoded $\mathrm{dec}_\phi(z_t)$")
        truel.append(tl); hatl.append(hl)
        if i == 0:
            a.legend(loc="upper right", fontsize=10, ncol=1, framealpha=0.9)
        if i == 2:
            a.set_xlabel("env step")
    axv[0].set_title("what the policy is denied vs. what it infers",
                     fontsize=12, pad=6)

    err_txt = ax3.text2D(0.02, 0.02, "", transform=ax3.transAxes, fontsize=12,
                         fontweight="bold", color="#333333")

    def draw(k):
        t = int(frames[min(k, len(frames) - 1)])
        sub = pos[: t + 1]
        trail.set_data(sub[:, 0], sub[:, 1]); trail.set_3d_properties(sub[:, 2])
        shadow.set_data(sub[:, 0], sub[:, 1])
        shadow.set_3d_properties(np.full(len(sub), lo[2]))
        drone.update(sub[-1], act[min(t, len(act) - 1)], vel[t], blade_phase=0.9 * k)

        tt = np.arange(t + 1)
        for i in range(3):
            truel[i].set_data(tt, vel[: t + 1, i])
            hatl[i].set_data(tt, vhat[: t + 1, i])

        e = float(np.linalg.norm(vhat[t] - vel[t]))
        err_txt.set_text(f"env step {t}   |  decode error  {e:.2f} m/s   "
                         f"|  $\\|z_t\\|$ {np.linalg.norm(z[t]):.2f}")
        return []

    total = len(frames) + n_hold
    print(f"[belief] {total} frames, T={T}")
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    _save(anim, "belief_tracking", fps, dpi)

    draw(len(frames) - 1)
    fig.savefig(SLIDES / "belief_tracking_still.png", dpi=140)
    print(f"  saved -> {SLIDES / 'belief_tracking_still.png'}")
    plt.close(fig)


# ============================================================== overshoot ====
def render_overshoot(fps: int = 25, hold_s: float = 2.2, dpi: int = 100,
                     max_steps: int = 230, stride: int = 1):
    """Why hiding velocity breaks it: the policy cannot tell when to brake.

    Same seed, same backbone, same recipe — the only difference is whether the
    three velocity dimensions are in the observation. Success needs distance
    < 0.5 m *and* speed < 1 m/s; the partial-obs policy gets closer in position
    and still fails, because it arrives too fast.
    """
    slide_style()
    f = SLIDES / "_traces_overshoot.npz"
    if not f.exists():
        raise SystemExit("missing traces — run: python -m paper._slide_rollouts")
    d = {k: v for k, v in np.load(f).items()}

    runs = ["Full obs", "Velocity hidden"]
    col = {"Full obs": "#2a78d6", "Velocity hidden": "#eb6834"}
    pos = {k: d[f"{k}/pos"] for k in runs}
    act = {k: d[f"{k}/act"] for k in runs}
    vel = {k: d[f"{k}/vel"] for k in runs}
    spd = {k: np.linalg.norm(vel[k], axis=1) for k in runs}
    dgo = {k: d[f"{k}/d_goal"] for k in runs}
    goal = d["Full obs/goal"]

    T = min(max_steps, max(len(pos[k]) for k in runs))
    frames = np.arange(0, T, stride)
    n_hold = int(hold_s * fps)

    pts = np.concatenate([pos[k][:T] for k in runs] + [goal[None]], axis=0)
    lo, hi = pts.min(0) - 0.7, pts.max(0) + 0.7
    span = float((hi - lo).max()); mid = (hi + lo) / 2
    lo, hi = mid - span / 2, mid + span / 2

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1.0],
                          left=0.02, right=0.965, top=0.885, bottom=0.09,
                          wspace=0.16, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])

    fig.suptitle("Same policy recipe, same seed — the only difference is whether "
                 "velocity is in the observation",
                 fontsize=15, fontweight="bold", y=0.965)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=20, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.78))
    except Exception:
        pass
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        pane.pane.set_edgecolor((0.85, 0.87, 0.92, 1.0))

    # Goal, plus the 0.5 m capture ball drawn to scale.
    u, vv = np.mgrid[0:2 * np.pi:22j, 0:np.pi:12j]
    r = 0.5
    ax.plot_surface(goal[0] + r * np.cos(u) * np.sin(vv),
                    goal[1] + r * np.sin(u) * np.sin(vv),
                    goal[2] + r * np.cos(vv),
                    color="#ffc000", alpha=0.16, linewidth=0, zorder=3)
    ax.scatter(*goal, marker="*", s=380, color="#ffc000",
               edgecolors="#7a5a00", linewidths=1.0, zorder=8)
    ax.text(goal[0], goal[1], goal[2] + 1.05, "goal  (0.5 m capture ball)",
            ha="center", fontsize=11.5, fontweight="bold", color="#7a5a00")
    ax.scatter(*pos["Full obs"][0], s=90, facecolor="white", edgecolor="black",
               linewidths=1.4, zorder=6)

    trails, shadows, drones = {}, {}, {}
    for k in runs:
        trails[k], = ax.plot([], [], [], color=col[k], lw=2.6,
                             ls=("-" if k == "Full obs" else "--"), zorder=5)
        shadows[k], = ax.plot([], [], [], color=col[k], lw=1.2, alpha=0.20, zorder=2)
        drones[k] = QuadrotorArtist(ax, col[k], scale=span / 13.0, zorder=11)
    ax.legend(handles=[plt.Line2D([], [], color=col[k], lw=2.6,
                                  ls=("-" if k == "Full obs" else "--"), label=k)
                       for k in runs],
              loc="upper left", bbox_to_anchor=(-0.02, 0.99), fontsize=12,
              framealpha=0.9)

    ax_d.set_title("distance to goal", fontsize=12.5, pad=6)
    ax_d.set_ylabel("m")
    ax_d.set_xlim(0, T); ax_d.set_ylim(0, max(dgo[k][:T].max() for k in runs) * 1.08)
    ax_d.axhline(0.5, color="#666666", ls=":", lw=1.4)
    ax_d.text(T * 0.98, 0.72, "0.5 m", fontsize=10, ha="right", color="#555555")

    ax_v.set_title("speed  $\\|\\mathbf{v}\\|$", fontsize=12.5, pad=6)
    ax_v.set_ylabel("m/s"); ax_v.set_xlabel("env step")
    ax_v.set_xlim(0, T); ax_v.set_ylim(0, max(spd[k][:T].max() for k in runs) * 1.10)
    ax_v.axhline(1.0, color="#666666", ls=":", lw=1.4)
    ax_v.text(T * 0.98, 1.25, "1 m/s stop tolerance", fontsize=10, ha="right",
              color="#555555")

    dl = {k: ax_d.plot([], [], color=col[k], lw=2.2,
                       ls=("-" if k == "Full obs" else "--"))[0] for k in runs}
    vl = {k: ax_v.plot([], [], color=col[k], lw=2.2,
                       ls=("-" if k == "Full obs" else "--"))[0] for k in runs}

    step_txt = ax.text2D(0.02, 0.075, "", transform=ax.transAxes, fontsize=12.5,
                         fontweight="bold", color="#333333")
    verdict = {k: ax.text2D(0.02, 0.032 - 0.042 * i, "", transform=ax.transAxes,
                            fontsize=12.5, fontweight="bold", color=col[k],
                            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
               for i, k in enumerate(runs)}

    # The moment each policy is closest to the goal — the whole argument.
    closest = {k: int(np.argmin(dgo[k][:T])) for k in runs}

    def draw(fi):
        t = int(frames[min(fi, len(frames) - 1)])
        for k in runs:
            p = pos[k]
            tt = min(t, len(p) - 1)
            sub = p[: tt + 1]
            trails[k].set_data(sub[:, 0], sub[:, 1])
            trails[k].set_3d_properties(sub[:, 2])
            shadows[k].set_data(sub[:, 0], sub[:, 1])
            shadows[k].set_3d_properties(np.full(len(sub), lo[2]))
            drones[k].update(sub[-1], act[k][min(tt, len(act[k]) - 1)],
                             vel[k][tt], blade_phase=0.9 * fi)
            dl[k].set_data(np.arange(tt + 1), dgo[k][: tt + 1])
            vl[k].set_data(np.arange(tt + 1), spd[k][: tt + 1])

        step_txt.set_text(f"env step {t}")
        for k in runs:
            c = closest[k]
            if t >= c:
                ok = ("REACHED — stopped inside the ball" if
                      (dgo[k][c] < 0.5 and spd[k][c] < 1.0) else
                      "TOO FAST — sails straight through")
                verdict[k].set_text(
                    f"{k}: closest {dgo[k][c]:.2f} m at {spd[k][c]:.2f} m/s   →   {ok}")
        return []

    total = len(frames) + n_hold
    print(f"[overshoot] {total} frames, T={T}")
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    _save(anim, "overshoot", fps, dpi)

    draw(len(frames) - 1)
    fig.savefig(SLIDES / "overshoot_still.png", dpi=140)
    print(f"  saved -> {SLIDES / 'overshoot_still.png'}")
    plt.close(fig)


# ========================================================= receding horizon ===
def render_chunk_replan(fps: int = 22, hold_s: float = 2.0, dpi: int = 100):
    """What "action chunk + receding horizon" means, shown rather than stated.

    At each re-plan the policy commits to 8 actions and executes only the first
    4, so consecutive plans overlap and the old plan is discarded half-used.
    """
    slide_style()
    f = SLIDES / "_chunk_sequence.npz"
    if not f.exists():
        raise SystemExit("missing traces — run: python -m paper._slide_rollouts")
    d = np.load(f)
    traj, plans, plan_steps, goal = d["traj"], d["plans"], d["plan_steps"], d["goal"]

    T = len(traj)
    frames = np.arange(T)
    n_hold = int(hold_s * fps)

    # A chase view: an 8-step plan spans a few tens of centimetres, which is a
    # handful of pixels at whole-trajectory scale. Follow the drone instead.
    span = 2.6
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    fig.suptitle("Action chunking with a receding horizon — plan 8 steps, "
                 "execute 4, re-plan from where you actually are",
                 fontsize=15, fontweight="bold", y=0.955)

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=20, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.97, 0.98, 1.0, 1.0))
        pane.pane.set_edgecolor((0.85, 0.87, 0.92, 1.0))

    goal_dot = ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=400,
                          color="#ffc000", edgecolors="#7a5a00", linewidths=1.0,
                          zorder=8)
    goal_lbl = ax.text(goal[0], goal[1], goal[2] + 0.22, "goal", ha="center",
                       fontsize=12, fontweight="bold", color="#7a5a00")

    flown, = ax.plot([], [], [], color="#6f6f6a", lw=2.6, zorder=5)
    ghosts = [ax.plot([], [], [], color=COLORS["FlyAda"], lw=1.6, alpha=0.20,
                      zorder=4)[0] for _ in range(4)]
    planned, = ax.plot([], [], [], color=COLORS["FlyAda"], lw=2.6, ls="--", zorder=6)
    committed, = ax.plot([], [], [], color=COLORS["FlyAda"], lw=6.0, alpha=0.45,
                         solid_capstyle="round", zorder=5)
    plan_dots = ax.scatter([], [], [], s=34, color=COLORS["FlyAda"],
                           edgecolors="white", linewidths=0.7, zorder=9)
    drone = QuadrotorArtist(ax, COLORS["Vanilla"], scale=span / 11.0, zorder=12)

    ax.legend(handles=[
        plt.Line2D([], [], color="#6f6f6a", lw=2.6, label="flown so far"),
        plt.Line2D([], [], color=COLORS["FlyAda"], lw=2.6, ls="--",
                   label="current plan  ($H=8$ actions)"),
        plt.Line2D([], [], color=COLORS["FlyAda"], lw=6.0, alpha=0.45,
                   label="the part that gets executed  ($K=4$)"),
        plt.Line2D([], [], color=COLORS["FlyAda"], lw=1.6, alpha=0.30,
                   label="discarded tails of earlier plans"),
    ], loc="upper left", bbox_to_anchor=(-0.02, 0.99), fontsize=12, framealpha=0.9)

    txt = ax.text2D(0.02, 0.02, "", transform=ax.transAxes, fontsize=13,
                    fontweight="bold", color="#333333")

    def draw(fi):
        t = int(frames[min(fi, len(frames) - 1)])
        sub = traj[: t + 1]
        flown.set_data(sub[:, 0], sub[:, 1]); flown.set_3d_properties(sub[:, 2])

        # Chase camera: recentre the box on the drone every frame.
        c = sub[-1]
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
        ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
        ax.set_zlim(c[2] - span * 0.4, c[2] + span * 0.4)
        inside = np.all(np.abs(goal - c) < span / 2)
        goal_dot.set_visible(bool(inside))
        goal_lbl.set_visible(bool(inside))

        pi = int(np.searchsorted(plan_steps, t, side="right") - 1)
        pi = max(pi, 0)
        pl = plans[pi]
        planned.set_data(pl[:, 0], pl[:, 1]); planned.set_3d_properties(pl[:, 2])
        committed.set_data(pl[:5, 0], pl[:5, 1]); committed.set_3d_properties(pl[:5, 2])
        plan_dots._offsets3d = (pl[1:, 0], pl[1:, 1], pl[1:, 2])

        for j, g in enumerate(ghosts):
            k = pi - (j + 1)
            if k >= 0:
                q = plans[k]
                g.set_data(q[:, 0], q[:, 1]); g.set_3d_properties(q[:, 2])
                g.set_alpha(0.22 - 0.045 * j)
            else:
                g.set_data([], []); g.set_3d_properties([])

        drone.update(sub[-1], pl[1] - pl[0], sub[-1] - traj[max(t - 1, 0)],
                     blade_phase=0.9 * fi)
        txt.set_text(f"env step {t}      re-plan #{pi + 1} of {len(plans)}"
                     f"      (a fresh chunk every {int(np.diff(plan_steps).min())} steps)")
        return []

    total = len(frames) + n_hold
    print(f"[chunk-replan] {total} frames, {len(plans)} plans")
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    _save(anim, "chunk_replan", fps, dpi)

    # Poster frame from mid-flight: at the last frame the drone is parked and
    # the plan collapses to a point, which says nothing about the mechanism.
    draw(int(len(frames) * 0.45))
    fig.savefig(SLIDES / "chunk_replan_still.png", dpi=140)
    print(f"  saved -> {SLIDES / 'chunk_replan_still.png'}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["race_nominal", "race_hard", "belief",
                             "overshoot", "chunk_replan"])
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    SLIDES.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "race_nominal"):
        render_race("nominal", dpi=args.dpi)
    if args.only in (None, "race_hard"):
        render_race("hard", dpi=args.dpi)
    if args.only in (None, "belief"):
        render_belief("hard", dpi=args.dpi)
    if args.only in (None, "overshoot"):
        render_overshoot(dpi=args.dpi)
    if args.only in (None, "chunk_replan"):
        render_chunk_replan(dpi=args.dpi)


if __name__ == "__main__":
    main()
