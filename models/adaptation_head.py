"""FlyAda adaptation head f_phi (PLAN Section 5.3).

Update rule: z_{t+1} = z_t + alpha * f_phi(s_t, u_t, s_{t+1})
Input: [s_t, u_t, s_{t+1}, s_{t+1}-s_t] concatenated.
Output: latent_dim tensor, tanh-activated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptationHead(nn.Module):
    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 3,
        latent_dim: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        in_dim = state_dim + action_dim + state_dim + state_dim  # s_t, u_t, s_{t+1}, delta_s
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, latent_dim)
        self.latent_dim = latent_dim
        # Zero-init the final projection so f_phi outputs exactly 0 at init.
        # Without this, a randomly-initialized f_phi dominates the policy's
        # conditioning (||z|| ~ 3 after a few dozen steps), destroying the
        # nominal-dynamics behaviour before f_phi has learned anything.
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        # Optional velocity-decoder head: predicts observed 3D velocity from z.
        # Used only during training as an auxiliary supervised signal so z actually
        # encodes velocity when the policy's observation is partial.
        self.vel_head = nn.Linear(latent_dim, 3)

    def forward(self, s_t: torch.Tensor, u_t: torch.Tensor, s_tp1: torch.Tensor) -> torch.Tensor:
        delta = s_tp1 - s_t
        inp = torch.cat([s_t, u_t, s_tp1, delta], dim=-1)
        h = F.mish(self.fc1(inp))
        h = F.mish(self.fc2(h))
        return torch.tanh(self.fc_out(h))

    def decode_velocity(self, z: torch.Tensor) -> torch.Tensor:
        """Auxiliary: predict the 3D velocity encoded in z."""
        return self.vel_head(z)


def roll_latent(
    f_phi: AdaptationHead,
    states: torch.Tensor,           # [B, T+1, state_dim]
    actions: torch.Tensor,          # [B, T, action_dim]
    alpha: float = 0.1,
    mode: str = "ema",
) -> torch.Tensor:
    """Roll z_t forward over a trajectory. Returns [B, T+1, latent_dim].

    mode="add" (plan default): z_{t+1} = z_t + alpha * f_phi(...).  Good for
        accumulating evidence about a static dynamics offset (e.g., wrong mass).
    mode="ema"  (plan-allowed alternative): z_{t+1} = (1-alpha)*z_t + alpha*f_phi(...).
        Good for *tracking* a signal that varies within the episode — e.g., current
        velocity inferred from position deltas under partial observability. With
        tanh-bounded f_phi this also bounds ||z|| regardless of episode length.
    """
    B, T_plus1, _ = states.shape
    T = T_plus1 - 1
    zs = [torch.zeros(B, f_phi.latent_dim, device=states.device, dtype=states.dtype)]
    for t in range(T):
        delta_z = f_phi(states[:, t], actions[:, t], states[:, t + 1])
        if mode == "add":
            zs.append(zs[-1] + alpha * delta_z)
        elif mode == "ema":
            zs.append((1.0 - alpha) * zs[-1] + alpha * delta_z)
        else:
            raise ValueError(f"Unknown latent update mode: {mode}")
    return torch.stack(zs, dim=1)   # [B, T+1, latent_dim]
