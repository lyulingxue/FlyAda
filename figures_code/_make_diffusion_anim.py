"""Animate the DDIM denoising of an action chunk into a trajectory.

For one fixed conditioning state we run the same 20-step DDIM sampler the policy
uses at deployment, but expose the intermediate states. At each step we plot the
denoiser's current best estimate of the clean action chunk x_0 — integrated from
the current pos/vel — as a 3D trajectory. The viewer sees random scribbles in
early steps collapse into a coherent path to the goal in the last few steps.

Output: paper/figures/diffusion_denoising.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from models.checkpoint import load_flyada_checkpoint


DT = 0.02
ACTION_SCALE = 5.0


def integrate_chunk(start_pos: np.ndarray, start_vel: np.ndarray, chunk: np.ndarray,
                    mass: float = 1.0, drag: float = 0.1, horizon_steps: int = 50) -> np.ndarray:
    """Integrate an action chunk into a position trajectory using nominal dynamics.

    The chunk has H actions covering H*dt seconds (~0.16 s for H=8). To make the
    predicted motion visible at goal-scale axes, we extend the integration to
    `horizon_steps` env-steps by repeating the chunk's last action.
    """
    pos, vel = start_pos.copy(), start_vel.copy()
    out = [pos.copy()]
    H = len(chunk)
    for k in range(horizon_steps):
        u = chunk[min(k, H - 1)]
        u_clipped = np.clip(u, -1, 1) * ACTION_SCALE
        accel = u_clipped / mass - drag * vel
        vel = vel + DT * accel
        pos = pos + DT * vel
        out.append(pos.copy())
    return np.stack(out)


@torch.no_grad()
def ddim_with_intermediates(policy, state: torch.Tensor, latent: torch.Tensor):
    """Mirror DiffusionPolicy.sample() but yield (x_t, x0_pred) at every step."""
    B = state.shape[0]
    cfg = policy.cfg
    H, A = cfg.horizon, cfg.action_dim
    T_s = cfg.T_sample
    device = policy.device

    x = torch.randn(B, H, A, device=device)
    cond = policy.build_cond(state, latent=latent)
    step_indices = torch.linspace(cfg.T_train - 1, 0, T_s, device=device).long()

    intermediates = []   # list of (x_t.cpu(), x0_pred.cpu())
    for i, idx in enumerate(step_indices):
        t_batch = torch.full((B,), int(idx), device=device, dtype=torch.long)
        eps = policy.model(x, t_batch, cond)
        alpha_bar_t = policy.alpha_bar[idx]
        x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
        x0_pred_clamped = x0_pred.clamp(-1.0, 1.0)
        intermediates.append((x.clone().cpu().numpy(), x0_pred_clamped.cpu().numpy()))

        if i + 1 < len(step_indices):
            idx_next = step_indices[i + 1]
            alpha_bar_next = policy.alpha_bar[idx_next]
            eps_used = (x - alpha_bar_t.sqrt() * x0_pred_clamped) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
            x = alpha_bar_next.sqrt() * x0_pred_clamped + (1.0 - alpha_bar_next).sqrt() * eps_used
        else:
            x = x0_pred_clamped
    intermediates.append((x.cpu().numpy(), x.cpu().numpy()))   # final state
    return intermediates


def main():
    device = "cuda"
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(
        str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device
    )

    # Pick a state with a far goal so the predicted chunk motion is large enough
    # to read at goal-scale axes.
    env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"))
    chosen = None
    for s in range(600_000, 600_200):
        obs, info = env.reset(seed=s)
        pos = np.asarray(env._pos, dtype=np.float32).copy()
        goal = np.asarray(info["goal"], dtype=np.float32)
        d = float(np.linalg.norm(goal - pos))
        if d > 6.0:
            chosen = s
            break
    print(f"using seed {chosen} (d_goal={d:.2f} m)")
    vel = np.asarray(env._vel, dtype=np.float32).copy()
    state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # Initialize latent at zero (cold-start episode start, like the teaser)
    z = torch.zeros(1, f_phi.latent_dim, device=device)

    intermediates = ddim_with_intermediates(fl_dp, state, latent=z)
    print(f"captured {len(intermediates)} denoising steps")

    # Integrate the chunk for 40 env steps (~0.8 s) — long enough to read but
    # short enough to avoid runaway extrapolation past the chunk's actual horizon.
    pred_trajs = []
    for x_t, x0 in intermediates:
        chunk = x0[0]
        traj = integrate_chunk(pos, vel, chunk, horizon_steps=40)
        pred_trajs.append(traj)
    pred_trajs = np.stack(pred_trajs)
    n_frames = pred_trajs.shape[0]
    step_indices = np.linspace(fl_dp.cfg.T_train - 1, 0, fl_dp.cfg.T_sample, dtype=int)
    alpha_bars = fl_dp.alpha_bar.cpu().numpy()[step_indices]

    from paper._plot_style import set_ieee_font
    set_ieee_font()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # Pin axes to the start-goal bounding box (with margin) so the visual frame
    # is consistent regardless of which way a noisy chunk's prediction wanders.
    box = np.stack([pos, goal])
    margin = 1.5
    xmin, xmax = float(box[:, 0].min() - margin), float(box[:, 0].max() + margin)
    ymin, ymax = float(box[:, 1].min() - margin), float(box[:, 1].max() + margin)
    zmin, zmax = float(box[:, 2].min() - margin), float(box[:, 2].max() + margin)

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    def render_frame(k):
        ax.cla()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-65)

        traj = pred_trajs[k]
        # Fade earlier frames behind for context
        for j in range(max(0, k - 3), k):
            ax.plot(pred_trajs[j][:, 0], pred_trajs[j][:, 1], pred_trajs[j][:, 2],
                    color="0.7", linewidth=0.8, alpha=0.25)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color="#ff7f0e", linewidth=2.4, label="$x_0$ prediction")
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color="green", s=60,
                   edgecolors="black", linewidths=0.8, label="current pos")
        ax.scatter(goal[0], goal[1], goal[2], color="black", marker="*", s=200,
                   edgecolors="white", linewidths=0.8, label="goal")

        is_final = (k == n_frames - 1)
        if k == 0:
            step_label = "step 0/20  (initial noise)"
            ab_label = ""
        elif is_final:
            step_label = f"step {k}/{n_frames-1} (final)"
            ab_label = ""
        else:
            step_label = f"step {k}/{n_frames-1}"
            ab_label = rf"  $\bar\alpha_t = {alpha_bars[min(k-1, len(alpha_bars)-1)]:.2f}$"
        ax.set_title(f"DDIM denoising — {step_label}{ab_label}", fontsize=11)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True,
                  facecolor="white", framealpha=0.85)
        return ()

    anim = FuncAnimation(fig, render_frame, frames=n_frames, interval=350, blit=False)
    out_path = ROOT / "paper/figures/diffusion_denoising.gif"
    writer = PillowWriter(fps=4)
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
