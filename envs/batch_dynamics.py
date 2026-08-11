"""GPU-batched replay of UAV dynamics under per-episode perturbations.

Matches envs/dynamics.py but vectorized over a batch for fast FlyAda training.
Obstacle collisions are ignored during replay — we only need the transition
sequence (s'_t, u_t, s'_{t+1}) to feed f_phi, not the reward/termination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


DT = 0.02
ACTION_SCALE = 5.0
STATE_DIM = 12          # matches UAVEnv: [p(3), v(3), yaw, yaw_dot, g_rel(3), d_goal]


@dataclass
class BatchPerturb:
    mass: torch.Tensor          # [B]
    drag: torch.Tensor          # [B]
    wind: torch.Tensor          # [B, 3]
    delay: torch.Tensor         # [B] int64 steps

    def to(self, device: str) -> "BatchPerturb":
        return BatchPerturb(
            mass=self.mass.to(device),
            drag=self.drag.to(device),
            wind=self.wind.to(device),
            delay=self.delay.to(device),
        )


def sample_perturbations(B: int, cfg: Dict, rng: np.random.Generator, nominal_frac: float = 0.1) -> BatchPerturb:
    """Sample B perturbations from config ranges. `nominal_frac` of them use nominal dynamics.

    This keeps the training distribution anchored so FlyAda on nominal Task A
    stays as good as vanilla diffusion.
    """
    mass = rng.uniform(*cfg["mass_range"], size=B).astype(np.float32)
    drag = rng.uniform(*cfg["drag_range"], size=B).astype(np.float32)
    wmag = rng.uniform(*cfg["wind_mag_range"], size=B).astype(np.float32)
    wdir = rng.standard_normal((B, 3)).astype(np.float32)
    wdir /= np.maximum(np.linalg.norm(wdir, axis=-1, keepdims=True), 1e-6)
    wind = wmag[:, None] * wdir
    d_lo, d_hi = cfg["delay_steps"]
    delay = rng.integers(d_lo, d_hi + 1, size=B).astype(np.int64)

    # Anchor some fraction to nominal
    if nominal_frac > 0:
        n_nom = max(1, int(B * nominal_frac))
        nom_idx = rng.choice(B, size=n_nom, replace=False)
        mass[nom_idx] = 1.0
        drag[nom_idx] = 0.1
        wind[nom_idx] = 0.0
        delay[nom_idx] = 0

    return BatchPerturb(
        mass=torch.from_numpy(mass),
        drag=torch.from_numpy(drag),
        wind=torch.from_numpy(wind),
        delay=torch.from_numpy(delay),
    )


def batch_replay(
    s0: torch.Tensor,                   # [B, STATE_DIM] initial full state (for goal extraction)
    actions: torch.Tensor,              # [B, T, 3], assumed already clipped to [-1, 1]
    perturb: BatchPerturb,
) -> torch.Tensor:
    """Replay actions under per-episode perturbed dynamics on GPU.

    Returns state trajectory [B, T+1, STATE_DIM] matching UAVEnv's 12-dim state layout.
    """
    B, T, A = actions.shape
    device = actions.device
    dtype = actions.dtype

    pos = s0[:, 0:3].clone()
    vel = s0[:, 3:6].clone()
    # goal = pos + g_rel (from s0)
    goal = pos + s0[:, 8:11]

    perturb = perturb.to(device)
    mass = perturb.mass.to(dtype).unsqueeze(-1)     # [B, 1]
    drag = perturb.drag.to(dtype).unsqueeze(-1)     # [B, 1]
    wind = perturb.wind.to(dtype)                   # [B, 3]
    delay = perturb.delay                           # [B]

    # Build delay-shifted action stream:
    #   applied[:, t] = actions[:, t - delay[i]] if t >= delay[i] else 0
    max_d = int(delay.max().item())
    pad = torch.zeros(B, max_d, A, device=device, dtype=dtype)
    padded = torch.cat([pad, actions], dim=1) if max_d > 0 else actions.clone()
    # Build an index tensor [B, T] that picks padded[b, t + (max_d - delay[b])]
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    shift = (max_d - delay).unsqueeze(-1)           # [B, 1]
    gather_idx = (t_idx + shift).clamp(min=0, max=padded.shape[1] - 1)
    applied = torch.gather(padded, 1, gather_idx.unsqueeze(-1).expand(-1, -1, A))

    applied = applied.clamp(-1.0, 1.0) * ACTION_SCALE

    zeros2 = torch.zeros(B, 2, device=device, dtype=dtype)  # yaw, yaw_dot
    g_rel0 = goal - pos
    d_goal0 = g_rel0.norm(dim=-1, keepdim=True)
    states = [torch.cat([pos, vel, zeros2, g_rel0, d_goal0], dim=-1)]

    for t in range(T):
        u_t = applied[:, t]
        accel = u_t / mass - drag * vel + wind
        vel = vel + DT * accel
        pos = pos + DT * vel
        g_rel = goal - pos
        d_goal = g_rel.norm(dim=-1, keepdim=True)
        states.append(torch.cat([pos, vel, zeros2, g_rel, d_goal], dim=-1))

    return torch.stack(states, dim=1)   # [B, T+1, STATE_DIM]
