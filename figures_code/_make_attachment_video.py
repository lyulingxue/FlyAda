"""Narrated explainer-style slideshow video for the FlyAda v1 paper.

Each slide has:
  - a title that fades in
  - one or more spoken-style narrative lines that appear progressively
  - a figure or animated diagram that fades in / pans
  - per-frame compositing so the content builds up over time

Output: paper/figures/attachment_video.mp4 (mp4, 1280x720, <10 MB).
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
import imageio.v3 as iio

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = FIG / "attachment_video.mp4"

W, H, DPI = 1280, 720, 100
FPS = 25
plt.rcParams.update({"font.family": "serif"})


def new_fig(bg="white"):
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def render(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def fade_in(t, total):
    """Linear alpha ramp, clamped to [0,1]."""
    if total <= 0: return 1.0
    return float(np.clip(t / total, 0.0, 1.0))


# --------------------------------------------------------------------------
# Title card with fade-in
# --------------------------------------------------------------------------
def title_card(line1, line2, line3, hold_s=4.0):
    frames = []
    n = int(hold_s * FPS)
    fade_n = int(0.7 * FPS)
    for i in range(n):
        a_main = fade_in(i, fade_n)
        a_sub  = fade_in(i - 8, fade_n)
        a_tag  = fade_in(i - 16, fade_n)
        fig, ax = new_fig(bg="#101620")
        ax.text(0.5, 0.62, line1, ha="center", va="center",
                  fontsize=28, fontweight="bold",
                  color=(1, 1, 1, a_main))
        ax.text(0.5, 0.46, line2, ha="center", va="center",
                  fontsize=18, color=(0.62, 0.74, 0.86, a_sub),
                  style="italic")
        if line3:
            ax.text(0.5, 0.36, line3, ha="center", va="center",
                      fontsize=13, color=(0.62, 0.74, 0.86, a_tag))
        ax.text(0.5, 0.06, "video attachment to anonymous submission",
                  ha="center", va="bottom", fontsize=10,
                  color=(0.36, 0.47, 0.57, a_tag))
        frames.append(render(fig))
    return frames


def closing_card(title, lines, hold_s=5.0):
    frames = []
    n = int(hold_s * FPS)
    fade_n = int(0.7 * FPS)
    for i in range(n):
        fig, ax = new_fig(bg="#101620")
        ax.text(0.5, 0.78, title, ha="center", va="center",
                  fontsize=28, fontweight="bold",
                  color=(1, 1, 1, fade_in(i, fade_n)))
        for k, ln in enumerate(lines):
            a = fade_in(i - 10 - 10 * k, fade_n)
            ax.text(0.5, 0.58 - 0.10 * k, ln, ha="center", va="center",
                      fontsize=16, color=(0.62, 0.74, 0.86, a),
                      style="italic" if k == 0 else "normal")
        ax.text(0.5, 0.06, "video attachment to anonymous submission",
                  ha="center", va="bottom", fontsize=10,
                  color=(0.36, 0.47, 0.57, fade_in(i - 25, fade_n)))
        frames.append(render(fig))
    return frames


# --------------------------------------------------------------------------
# Content slide: title + progressively-revealed bullets + figure (fade in)
# --------------------------------------------------------------------------
def _wrap_bullets(bullets, bullet_box):
    """Wrap each bullet text to fit the bullet column.  bullets is a list of
    either strings or (delay, text) tuples; returns the same shape with text
    wrapped.  Approximate char width: 0.011 of canvas width at fontsize 12."""
    bx, by, bw, bh = bullet_box
    # Leave ~0.025 for the bullet glyph + small right padding.
    text_width_fig = bw - 0.04
    char_w_fig = 0.0095   # tuned for fontsize=12 serif at 1280x720
    max_chars = max(20, int(text_width_fig / char_w_fig))
    out = []
    for b in bullets:
        if isinstance(b, tuple):
            delay, text = b
            wrapped = _wrap_one(text, max_chars)
            out.append((delay, wrapped))
        else:
            out.append(_wrap_one(b, max_chars))
    return out


def _wrap_one(text, max_chars):
    """Wrap text to max_chars per line, preserving the user's explicit \\n
    breaks as paragraph boundaries.  Continuation lines are indented by 2
    spaces so they read as part of the bullet body."""
    paragraphs = text.split("\n")
    wrapped_paragraphs = []
    for p in paragraphs:
        # Strip leading whitespace from continuation lines the caller might
        # have added as visual indent (we'll re-add it).
        stripped = p.lstrip()
        was_continuation = stripped != p
        ww = textwrap.wrap(stripped, width=max_chars,
                              subsequent_indent="  ")
        if not ww:
            ww = [""]
        if was_continuation:
            ww[0] = "  " + ww[0]
        wrapped_paragraphs.append("\n".join(ww))
    return "\n".join(wrapped_paragraphs)


def _bullet_y_positions(bullets, top, line_step=0.045, bullet_gap=0.025):
    """Pre-compute y position for each bullet given its line count, stepping
    down by line_step per text line plus bullet_gap between bullets."""
    ys = []
    cur = top
    for entry in bullets:
        text = entry[1] if isinstance(entry, tuple) else entry
        ys.append(cur)
        n_lines = text.count("\n") + 1
        cur -= line_step * n_lines + bullet_gap
    return ys


def content_slide(title, bullets, image_path=None, image_box=(0.10, 0.16, 0.55, 0.55),
                    bullet_box=(0.66, 0.20, 0.32, 0.50),
                    bullet_lead_s=1.6, image_fade_s=0.8, hold_s=8.0,
                    image_caption=None):
    """
    image_box / bullet_box: (left, bottom, width, height) in figure coords.
    Use image_box=None to skip image; bullets fill width.
    bullets: list of (delay_frames, text) tuples OR plain strings (auto-delayed).
    """
    if isinstance(bullets[0], str):
        bullets = [(int((k + 1) * bullet_lead_s * FPS), b) for k, b in enumerate(bullets)]
    bullets = _wrap_bullets(bullets, bullet_box)
    total_s = max(hold_s, bullets[-1][0] / FPS + 1.5)
    n_frames = int(total_s * FPS)
    image_fade_n = int(image_fade_s * FPS)

    img = mpimg.imread(image_path) if image_path else None

    bx, by, bw, bh = bullet_box
    bullet_ys = _bullet_y_positions(bullets, top=by + bh)

    frames = []
    for i in range(n_frames):
        fig, ax = new_fig(bg="white")
        # Title (fades in over first 0.6s).
        ax.text(0.05, 0.93, title, ha="left", va="top",
                  fontsize=22, fontweight="bold",
                  color=(0.10, 0.10, 0.10, fade_in(i, int(0.6 * FPS))))
        # Horizontal divider.
        ax.plot([0.05, 0.95], [0.875, 0.875],
                  color=(0.7, 0.7, 0.7, fade_in(i - 4, int(0.6 * FPS))), lw=1)

        # Image (fade in).
        if img is not None:
            a = fade_in(i - 6, image_fade_n)
            if a > 0:
                ax_img = fig.add_axes(image_box)
                ax_img.imshow(img, alpha=a); ax_img.set_axis_off()
            if image_caption:
                ax.text(image_box[0] + image_box[2] / 2,
                          image_box[1] - 0.02,
                          image_caption, ha="center", va="top",
                          fontsize=10, color=(0.30, 0.30, 0.30,
                                              fade_in(i - 10, image_fade_n)),
                          style="italic")

        # Bullets (each fades in at its delay).
        for k, (delay, text) in enumerate(bullets):
            a = fade_in(i - delay, int(0.5 * FPS))
            if a <= 0: continue
            y = bullet_ys[k]
            ax.text(bx, y, "•", ha="left", va="top",
                      fontsize=18, color=(0.20, 0.45, 0.75, a),
                      fontweight="bold")
            ax.text(bx + 0.025, y, text, ha="left", va="top",
                      fontsize=12, color=(0.15, 0.15, 0.15, a))
        frames.append(render(fig))
    return frames


# --------------------------------------------------------------------------
# Method diagram: data-flow boxes that appear one at a time
# --------------------------------------------------------------------------
def method_diagram_slide(hold_s=14.0):
    """Build the FlyAda data-flow diagram progressively over time."""
    n_frames = int(hold_s * FPS)
    # Element appearance times (in seconds).
    # 0: title; 1: obs box; 2: policy box; 3: action arrow; 4: observer head;
    # 5: latent box + arrow back to policy; 6: EMA loop; 7: aux loss arrow.
    appear_s = [0.4, 1.4, 2.4, 3.4, 4.6, 6.0, 7.6, 9.2]

    frames = []
    for i in range(n_frames):
        t = i / FPS
        fig, ax = new_fig(bg="white")

        def alive(idx):
            return t >= appear_s[idx]
        def alpha(idx, dur=0.6):
            return float(np.clip((t - appear_s[idx]) / dur, 0, 1))

        ax.text(0.05, 0.94, "Method: FlyAda observer head + EMA-updated latent",
                  ha="left", va="top", fontsize=22, fontweight="bold",
                  color=(0.10, 0.10, 0.10, alpha(0)))
        ax.plot([0.05, 0.95], [0.885, 0.885],
                  color=(0.7, 0.7, 0.7, alpha(0)), lw=1)

        # Box helpers (positions in axes coords).
        def box(x, y, w, h, color, label, sublabel=None, alpha_v=1.0):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.005",
                facecolor=color, edgecolor=(0.15, 0.15, 0.15, alpha_v),
                linewidth=1.2, alpha=alpha_v))
            ax.text(x + w / 2, y + h / 2 + (0.02 if sublabel else 0),
                      label, ha="center", va="center",
                      fontsize=12, fontweight="bold",
                      color=(0.10, 0.10, 0.10, alpha_v))
            if sublabel:
                ax.text(x + w / 2, y + h / 2 - 0.025, sublabel,
                          ha="center", va="center", fontsize=10,
                          color=(0.30, 0.30, 0.30, alpha_v), style="italic")

        def arrow(x0, y0, x1, y1, color="#222", lw=1.6, alpha_v=1.0,
                    style="-|>"):
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle=style,
                mutation_scale=14, lw=lw,
                color=(color[1:3] if isinstance(color, str) else color),
                alpha=alpha_v,
                shrinkA=2, shrinkB=2))

        # Manually-routed arrows because matplotlib FancyArrowPatch wants
        # color in RGBA or named colour, not hex tuple — convert.
        from matplotlib.colors import to_rgba

        def arr(x0, y0, x1, y1, color="#222", lw=1.6, alpha_v=1.0,
                  style="-|>", connectionstyle="arc3,rad=0"):
            r, g, b, _ = to_rgba(color)
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle=style,
                mutation_scale=14, lw=lw,
                color=(r, g, b, alpha_v),
                connectionstyle=connectionstyle,
                shrinkA=2, shrinkB=2))

        # 1. Observation box.
        if alive(1):
            box(0.05, 0.58, 0.18, 0.14, "#dfe9f3",
                  "Observation $o_t$",
                  "pos, goal, velocity = $\\mathbf{0}$",
                  alpha(1))
        # 2. Diffusion policy box (centre).
        if alive(2):
            box(0.34, 0.55, 0.20, 0.20, "#fde8c4",
                  "Diffusion BC",
                  "denoiser $\\pi_\\theta$",
                  alpha(2))
            arr(0.23, 0.65, 0.34, 0.65, alpha_v=alpha(2))
        # 3. Action chunk.
        if alive(3):
            box(0.66, 0.58, 0.18, 0.14, "#dfe9f3",
                  "Action chunk",
                  "$a_t,\\ldots,a_{t+K-1}$", alpha(3))
            arr(0.54, 0.65, 0.66, 0.65, alpha_v=alpha(3))

        # 4. Observer head (below the policy).
        if alive(4):
            box(0.34, 0.30, 0.20, 0.13, "#cfe8d3",
                  "Observer head",
                  "73K-param MLP, $z_t$", alpha(4))
            # Arrow from policy down to head.
            arr(0.44, 0.55, 0.44, 0.43, alpha_v=alpha(4), color="#3a3a3a")

        # 5. Aux velocity-decoding loss.
        if alive(5):
            box(0.66, 0.30, 0.20, 0.13, "#f5d6d6",
                  "Auxiliary loss",
                  "decode true $v_t$", alpha(5))
            arr(0.54, 0.365, 0.66, 0.365, alpha_v=alpha(5), color="#a23a3a")
            ax.text(0.6, 0.40, "supervised at training",
                      fontsize=9, color=(0.5, 0.2, 0.2, alpha(5)),
                      style="italic")

        # 6. Latent feedback into policy conditioning.
        if alive(6):
            # Curved arrow from observer head back up to policy.
            arr(0.40, 0.43, 0.40, 0.55, alpha_v=alpha(6), color="#1a6b1a",
                  connectionstyle="arc3,rad=-0.3")
            ax.text(0.27, 0.50, "$z_t$ conditions denoiser",
                      fontsize=10, color=(0.10, 0.40, 0.10, alpha(6)),
                      style="italic")

        # 7. EMA online update at deployment.
        if alive(7):
            box(0.05, 0.30, 0.18, 0.13, "#e9d9f5",
                  "EMA update", "$z_{t+1} = (1-\\beta) z_t + \\beta\\,h(o_t)$",
                  alpha(7))
            arr(0.34, 0.365, 0.23, 0.365, alpha_v=alpha(7), color="#5a2a8a")
            ax.text(0.06, 0.255, "applied online at deployment",
                      fontsize=9, color=(0.4, 0.2, 0.6, alpha(7)),
                      style="italic")

        # Tagline (bottom).
        a_tag = float(np.clip((t - 10.5) / 1.0, 0, 1))
        ax.text(0.5, 0.10,
                  "Same diffusion backbone; add a small head with a supervised aux loss,\n"
                  "then update its latent online with an EMA.  Frame-stack alone does not do this.",
                  ha="center", va="center", fontsize=12,
                  color=(0.15, 0.15, 0.15, a_tag))
        frames.append(render(fig))
    return frames


# --------------------------------------------------------------------------
# GIF / video-backed slide: loop frames from a GIF or MP4 with caption
# --------------------------------------------------------------------------
def _load_anim_frames(path):
    path = str(path)
    if path.lower().endswith((".mp4", ".mpg", ".mpeg", ".mov")):
        v = iio.imread(path)
        return [np.array(v[i]) for i in range(len(v))]
    g = Image.open(path)
    out = []
    try:
        while True:
            out.append(np.array(g.copy().convert("RGB")))
            g.seek(g.tell() + 1)
    except EOFError:
        pass
    return out


def gif_slide(title, gif_path, narrative_bullets, hold_s=10.0,
                image_box=(0.07, 0.15, 0.55, 0.62),
                bullet_box=(0.66, 0.22, 0.32, 0.55),
                playback_factor=1):
    """Cycle through animation frames while bullets fade in on the right.
    playback_factor controls speed: 1 = source-rate (every frame), 2 = half
    rate (each frame shown twice), etc."""
    src_frames = _load_anim_frames(gif_path)
    n_src = len(src_frames)
    n_frames = int(hold_s * FPS)
    fade_n = int(0.6 * FPS)
    bullet_delays = [int((k + 1) * 1.4 * FPS) for k in range(len(narrative_bullets))]

    bx, by, bw, bh = bullet_box
    bullet_pairs = list(zip(bullet_delays, narrative_bullets))
    bullet_pairs = _wrap_bullets(bullet_pairs, bullet_box)
    narrative_bullets = [t for _, t in bullet_pairs]
    bullet_ys = _bullet_y_positions(bullet_pairs, top=by + bh)

    out = []
    for i in range(n_frames):
        src_idx = (i // playback_factor) % n_src
        fig, ax = new_fig(bg="white")
        ax.text(0.05, 0.93, title, ha="left", va="top",
                  fontsize=22, fontweight="bold",
                  color=(0.10, 0.10, 0.10, fade_in(i, fade_n)))
        ax.plot([0.05, 0.95], [0.875, 0.875],
                  color=(0.7, 0.7, 0.7, fade_in(i - 4, fade_n)), lw=1)

        a = fade_in(i - 6, fade_n)
        if a > 0:
            ax_img = fig.add_axes(image_box)
            ax_img.imshow(src_frames[src_idx], alpha=a); ax_img.set_axis_off()

        for k, text in enumerate(narrative_bullets):
            a = fade_in(i - bullet_delays[k], int(0.5 * FPS))
            if a <= 0: continue
            y = bullet_ys[k]
            ax.text(bx, y, "•", ha="left", va="top",
                      fontsize=18, color=(0.20, 0.45, 0.75, a),
                      fontweight="bold")
            ax.text(bx + 0.025, y, text, ha="left", va="top",
                      fontsize=12, color=(0.15, 0.15, 0.15, a))
        out.append(render(fig))
    return out


# --------------------------------------------------------------------------
# Compose the video
# --------------------------------------------------------------------------
frames = []

# --- 1. Title card ---
frames += title_card(
    "FlyAda: Belief-State Adaptation",
    "for Diffusion Policies under Partial Observation",
    "UAV goal-reaching, hidden velocity, online latent update",
    hold_s=4.0)

# --- 2. The setup ---
frames += content_slide(
    title="The setup",
    bullets=[
        "Quadrotor goal-reaching, action-chunk diffusion BC.",
        "Perturbations: mass, drag, wind, control-delay.",
        "Question: when does online adaptation actually help?",
        "We pick the diffusion-policy regime because it is the\n  default for contact-rich imitation now.",
    ],
    image_path=FIG / "teaser.png",
    image_box=(0.06, 0.15, 0.55, 0.60),
    image_caption="Same task, three policies, single seed.",
    hold_s=10.0)

# --- 3. Surprise: full obs already works ---
frames += content_slide(
    title="Full observation: vanilla already wins",
    bullets=[
        "Vanilla diffusion reaches ~100% on the conventional sweep.",
        "Even at 4x mass, 10x drag, 8-step delay extrapolation.",
        "Reason: receding-horizon re-sampling at K=4 lets the next chunk\n  absorb the previous chunk's dynamics error.",
        "Adding an adaptation latent on top: did not change the picture.",
        "We had to *restrict* the observation to find a gap.",
    ],
    image_path=FIG / "mismatch_success_curves.png",
    image_box=(0.06, 0.17, 0.55, 0.55),
    image_caption="Success curves across the mass / drag / wind / delay sweep.",
    hold_s=11.0)

# --- 4. Hiding velocity breaks the closed loop (animated 2D rollout) ---
frames += gif_slide(
    title="Hiding velocity breaks the closed loop",
    gif_path=FIG / "anim_rollout_2d.mp4",
    narrative_bullets=[
        "Velocity is zeroed in the policy's observation,\n  at training and at test.",
        "Vanilla overshoots every waypoint.",
        "3-frame stack: same pattern (input is there,\n  gradient is not).",
        "FlyAda's latent carries the velocity estimate;\n  the policy can brake into each waypoint.",
        "Vanilla 0.3% / Frame-stack 1.1% / FlyAda near 100%.",
    ],
    image_box=(0.02, 0.13, 0.55, 0.68),
    bullet_box=(0.60, 0.16, 0.38, 0.62),
    hold_s=12.0)

# --- 5. Animated method diagram ---
frames += method_diagram_slide(hold_s=14.0)

# --- 6. What the latent learns (animated z_t vs true velocity) ---
frames += gif_slide(
    title="What the latent encodes",
    gif_path=FIG / "anim_latent_vs_vel.mp4",
    narrative_bullets=[
        "True velocity (blue) is hidden from the policy.",
        "z_t (orange dashed) is the EMA-updated latent\n  recovered at deployment by the observer head.",
        "Linear probe on z_t recovers the hidden velocity\n  to R^2 = 0.992.",
        "Same-weights ablation: toggle only the test-time\n  update rule. The continuous online update is the\n  part that matters.",
    ],
    image_box=(0.02, 0.14, 0.55, 0.65),
    bullet_box=(0.60, 0.18, 0.38, 0.58),
    hold_s=11.0)

# --- 7. Result trajectories (animated GIF) ---
frames += gif_slide(
    title="Result: FlyAda threads the partial-obs task",
    gif_path=FIG / "taskB_comparison.gif",
    narrative_bullets=[
        "Animation: same seed, three policies.",
        "Vanilla (blue) and frame-stack (green) overshoot the\n  first waypoint and never recover.",
        "FlyAda (orange) settles into each waypoint and threads\n  all three.",
        "Near-full success on the partial-obs sweep within the\n  training range; degrades only at wind well above training\n  (0.95 at 2 m/s, 0 at 3 m/s).",
    ],
    hold_s=12.0)

# --- 7b. Animated sweep-bar payoff: success rate across all 12 cells ---
frames += gif_slide(
    title="Partial-obs sweep: vanilla vs frame-stack vs FlyAda",
    gif_path=FIG / "anim_sweep_bars.mp4",
    narrative_bullets=[
        "12 conditions: nominal, mass +10/20/30%,\n  drag +50/100%, wind 0.5/1.0/1.5 m/s,\n  delay 1/2/3 steps.",
        "Vanilla diffusion: 0% on every cell.",
        "3-frame stack: 0% on every cell.",
        "FlyAda: 100% on every cell of the sweep.",
        "Same backbone, same demos —\n  the auxiliary head + online latent\n  is the only delta.",
    ],
    image_box=(0.02, 0.13, 0.55, 0.66),
    bullet_box=(0.60, 0.18, 0.38, 0.58),
    hold_s=12.0)

# --- 8. MuJoCo sim-to-sim ---
frames += content_slide(
    title="Sim-to-sim transfer to a 6-DoF MuJoCo quadrotor",
    bullets=[
        "Home sim is a 3-DoF point-mass with linear drag.",
        "MuJoCo: 1 kg, 4 rotors, cascaded attitude controller at 500 Hz.",
        "No retraining; only the inner controller wraps the policy.",
        "The same partial-obs gap reappears.",
        "FlyAda closes it with the same checkpoint.",
    ],
    image_path=FIG / "mujoco_hard_teaser.png",
    image_box=(0.06, 0.13, 0.55, 0.62),
    image_caption="MuJoCo hard condition: mass x 1.3, drag, wind 1 m/s.",
    hold_s=11.0)

# --- 9. Closing card ---
frames += closing_card(
    "Take-aways",
    [
        "Partial-obs failure is about the *objective*, not the input.",
        "Auxiliary velocity head + online EMA latent close the gap.",
        "Frame-stacking does not. Latent recovers velocity to R² = 0.992.",
        "Same recipe transfers to a 6-DoF MuJoCo quadrotor.",
    ],
    hold_s=6.0)

print(f"Encoding {len(frames)} frames -> {OUT}")
iio.imwrite(OUT, np.stack(frames), fps=FPS, codec="libx264",
              macro_block_size=1)
print(f"saved -> {OUT}  ({len(frames)/FPS:.1f} s, "
       f"{OUT.stat().st_size/1024:.1f} KB)")
