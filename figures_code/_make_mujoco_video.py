"""Render a real 3D MuJoCo flight for the sim-to-sim slide.

Two rendered viewports side by side — vanilla and FlyAda flying the same seed of
the 6-DoF quadrotor under the hard condition — with a shared distance-to-goal
trace underneath. Rendering uses envs/mujoco_quadrotor_render.xml, which is
dynamically identical to the physics model (see envs/_render_parity_check.py);
only the appearance differs.

Usage:
    python -m paper._make_mujoco_video
Output:
    paper/figures/slides/mujoco_flight.mp4  (+ _still.png)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco                                                  # noqa: E402
import torch                                                   # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter   # noqa: E402

from envs.mujoco_quadrotor import MuJoCoQuadrotorEnv           # noqa: E402
from models.checkpoint import (                                # noqa: E402
    load_diffusion_checkpoint, load_flyada_checkpoint)
from trainers.train_flyada import FlyAdaRolloutPolicy          # noqa: E402

try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

SLIDES = ROOT / "paper" / "figures" / "slides"
RENDER_XML = str(ROOT / "envs" / "mujoco_quadrotor_render.xml")

C_VAN = "#2a78d6"
C_FLY = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"

# Paper Table V hard condition.
HARD = dict(mass_scale=1.3, drag_world=0.1, wind_world=(1.0, 0.0, 0.0))

RGBA = {"Vanilla": (0.16, 0.47, 0.84, 1.0), "FlyAda": (0.92, 0.41, 0.20, 1.0)}


# ------------------------------------------------------------------ rollout ---
def roll(name: str, policy, seed: int, max_steps: int):
    # DDIM sampling draws its own noise, so without this the rendered episode
    # differs every run and the framing/axis limits stop being reproducible.
    torch.manual_seed(seed)
    env = MuJoCoQuadrotorEnv(xml_path=RENDER_XML, partial_obs=True,
                             max_steps=max_steps, **HARD)
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    goal = np.asarray(info["goal"], np.float32)

    qpos, dgoal, contact = [env.data.qpos.copy()], [], []
    success = False
    for _ in range(max_steps):
        a, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        qpos.append(env.data.qpos.copy())
        dgoal.append(float(info["d_goal"]))
        # Ground contact is the dominant vanilla failure here — 11 of 12 seeds —
        # so the plot marks it rather than leaving the viewer to infer it.
        contact.append(env.data.ncon > 0)
        success = bool(info.get("success", False))
        if term or trunc:
            break
    contact = np.asarray(contact, bool)
    print(f"  {name:8s} len={len(dgoal):3d}  final d={dgoal[-1]:.2f} m  "
          f"success={success}  ground-contact steps={int(contact.sum())}")
    return {"qpos": np.asarray(qpos), "d": np.asarray(dgoal),
            "contact": contact, "goal": goal, "success": success}


# ------------------------------------------------------------------ render ----
class Viewport:
    """One MuJoCo renderer plus the model/data pair whose pose it draws."""

    def __init__(self, width, height, trail_rgba, max_trail=90):
        self.model = mujoco.MjModel.from_xml_path(RENDER_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.scene.maxgeom  # touch, so a bad build fails loudly here
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        self.cam.elevation = -16.0
        self.trail_rgba = np.asarray(trail_rgba, np.float32)
        self.max_trail = max_trail
        self.mocap_id = self.model.body("goalmarker").mocapid[0]

    def frame(self, qpos, goal, trail, azimuth):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[self.mocap_id] = goal
        mujoco.mj_forward(self.model, self.data)

        pos = qpos[:3]
        sep = float(np.linalg.norm(goal - pos))
        # Bias the framing toward the drone so the airframe stays large and
        # legible; the goal marker drifts toward the edge when it wanders far.
        self.cam.lookat[:] = 0.66 * pos + 0.34 * goal
        self.cam.distance = float(np.clip(0.95 * sep + 2.4, 3.4, 8.5))
        self.cam.azimuth = azimuth

        self.renderer.update_scene(self.data, self.cam)

        # Breadcrumb trail, drawn as scene geoms so it lives in the 3D scene
        # (correctly occluded) rather than being pasted on in 2D.
        scene = self.renderer.scene
        n = len(trail)
        for i, p in enumerate(trail[-self.max_trail:]):
            if scene.ngeom >= scene.maxgeom:
                break
            frac = (i + 1) / min(n, self.max_trail)
            rgba = self.trail_rgba.copy()
            rgba[3] = 0.12 + 0.68 * frac
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g, int(mujoco.mjtGeom.mjGEOM_SPHERE),
                np.array([0.035 + 0.03 * frac] * 3, np.float64),
                np.asarray(p, np.float64), np.eye(3).flatten(), rgba)
            scene.ngeom += 1

        return self.renderer.render()


def main():
    ap = argparse.ArgumentParser()
    # Seed 10: vanilla sinks, recovers, sinks again and drifts to 4.5 m —
    # the typical failure (11 of 12 seeds touch down), and it stays
    # visually alive rather than parking on the floor at step 56.
    ap.add_argument("--seed", type=int, default=10)
    # 260 steps is past FlyAda's convergence (~110) and shows two full vanilla
    # oscillations, without the late fly-away that blows up the distance axis.
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--height", type=int, default=430)
    args = ap.parse_args()

    device = "cuda"
    van = load_diffusion_checkpoint(
        str(ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"), device=device)
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(
        str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device)

    print("[mujoco-video] rolling out")
    runs = {
        "Vanilla": roll("Vanilla", van.make_rollout_policy(), args.seed, args.max_steps),
        "FlyAda": roll("FlyAda",
                       FlyAdaRolloutPolicy(fl_dp, f_phi,
                                           alpha=float(fcfg.get("alpha", 0.3)),
                                           update_mode=str(fcfg.get("update_mode", "ema"))),
                       args.seed, args.max_steps),
    }

    T = max(len(r["qpos"]) for r in runs.values())
    frames = np.arange(0, T, args.stride)
    hold = int(1.6 * args.fps)

    views = {k: Viewport(args.width, args.height, RGBA[k]) for k in runs}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0],
                          left=0.035, right=0.975, top=0.885, bottom=0.085,
                          wspace=0.06, hspace=0.30)
    axv = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate(runs)}
    axd = fig.add_subplot(gs[1, :])

    fig.suptitle("Sim-to-sim — 6-DoF MuJoCo quadrotor, velocity hidden, hard condition "
                 "(mass ×1.3, drag, 1 m/s wind)",
                 fontsize=15, fontweight="bold", y=0.965, color=INK)

    ims = {}
    for k, ax in axv.items():
        ax.axis("off")
        ims[k] = ax.imshow(np.zeros((args.height, args.width, 3), np.uint8))
        ax.set_title(k, fontsize=15, fontweight="bold",
                     color=C_VAN if k == "Vanilla" else C_FLY, pad=8)

    axd.set_xlim(0, T)
    axd.set_ylim(0, max(float(r["d"].max()) for r in runs.values()) * 1.08)
    axd.set_xlabel("env step (50 Hz)", fontsize=12, color=INK2)
    axd.set_ylabel("distance to goal (m)", fontsize=12, color=INK2)
    axd.grid(alpha=0.3)
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    axd.axhline(0.5, color="#666666", ls=":", lw=1.3)
    axd.text(T * 0.995, 0.62, "success tolerance", fontsize=10, ha="right",
             color="#555555")

    # Ground-contact band for vanilla, along the bottom of the trace.
    ymax = axd.get_ylim()[1]
    band_lo, band_hi = -0.035 * ymax, 0.02 * ymax
    axd.set_ylim(band_lo * 1.6, ymax)
    c = runs["Vanilla"]["contact"]
    if c.any():
        axd.fill_between(np.arange(len(c)), band_lo, band_hi, where=c,
                         color="#b4462f", alpha=0.9, step="mid", lw=0)
        first = int(np.argmax(c))
        # Keep the label clear of the FlyAda curve, which also runs near zero
        # around the first touchdown — otherwise it reads as labelling FlyAda.
        axd.annotate("vanilla touches the ground",
                     xy=(first, band_hi), xytext=(first + T * 0.05, ymax * 0.72),
                     fontsize=11.5, color="#b4462f", fontweight="bold",
                     arrowprops=dict(arrowstyle="-|>", color="#b4462f", lw=1.6,
                                     connectionstyle="arc3,rad=-0.2"))
    dl = {k: axd.plot([], [], color=(C_VAN if k == "Vanilla" else C_FLY),
                      lw=2.2, ls=("--" if k == "Vanilla" else "-"),
                      label=k)[0] for k in runs}
    axd.legend(loc="upper right", fontsize=12, frameon=False, ncol=2)

    def draw(fi):
        t = int(frames[min(fi, len(frames) - 1)])
        az = 132.0 + 0.055 * t          # slow orbit, so the scene reads as 3D
        for k, r in runs.items():
            q = r["qpos"]
            tt = min(t, len(q) - 1)
            trail = q[: tt + 1, :3][::2]
            ims[k].set_data(views[k].frame(q[tt], r["goal"], trail, az))
            d = r["d"][: max(tt, 1)]
            dl[k].set_data(np.arange(len(d)), d)
        return []

    total = len(frames) + hold
    print(f"[mujoco-video] rendering {total} frames")
    anim = FuncAnimation(fig, draw, frames=total, interval=1000 / args.fps)
    out = SLIDES / "mujoco_flight.mp4"
    anim.save(str(out), writer=FFMpegWriter(
        fps=args.fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow"]), dpi=100)
    print(f"  saved -> {out}")

    draw(len(frames) - 1)
    fig.savefig(SLIDES / "mujoco_flight_still.png", dpi=140)
    print(f"  saved -> {SLIDES / 'mujoco_flight_still.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
