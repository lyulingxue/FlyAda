"""1D temporal UNet denoiser over action-chunk time axis.

Input: action chunk x [B, H, A], diffusion timestep t [B], conditioning cond [B, C].
Output: epsilon prediction of the same shape as x.

Conditioning is FiLM-applied inside each residual block from a combined
(timestep-embed + cond-embed) vector.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 5, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(groups, out_c), out_c)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock1D(nn.Module):
    """Two-stage conv block with FiLM conditioning applied between stages."""

    def __init__(self, in_c: int, out_c: int, cond_dim: int, kernel: int = 5):
        super().__init__()
        self.conv1 = Conv1dBlock(in_c, out_c, kernel=kernel)
        self.conv2 = Conv1dBlock(out_c, out_c, kernel=kernel)
        self.film = nn.Linear(cond_dim, 2 * out_c)
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma[..., None]) + beta[..., None]
        h = self.conv2(h)
        return h + self.skip(x)


class TemporalUNet(nn.Module):
    """Small 1D UNet: one down, one up, with a mid stack. Designed for H in [4, 16]."""

    def __init__(
        self,
        action_dim: int = 3,
        cond_input_dim: int = 12,
        hidden: int = 128,
        latent_dim: int = 0,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_input_dim = cond_input_dim
        self.latent_dim = latent_dim
        self.hidden = hidden

        cond_dim = hidden * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim + latent_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.in_conv = nn.Conv1d(action_dim, hidden, 3, padding=1)

        self.down1 = ResBlock1D(hidden, hidden, cond_dim)
        self.down2 = ResBlock1D(hidden, hidden * 2, cond_dim)
        self.downsample = nn.Conv1d(hidden * 2, hidden * 2, 3, stride=2, padding=1)

        self.mid1 = ResBlock1D(hidden * 2, hidden * 2, cond_dim)
        self.mid2 = ResBlock1D(hidden * 2, hidden * 2, cond_dim)

        self.upsample = nn.ConvTranspose1d(hidden * 2, hidden * 2, 4, stride=2, padding=1)
        # After concatenating the skip from down2 (which has hidden*2 channels):
        self.up1 = ResBlock1D(hidden * 2 + hidden * 2, hidden, cond_dim)
        self.up2 = ResBlock1D(hidden + hidden, hidden, cond_dim)

        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_act = nn.Mish()
        self.out_conv = nn.Conv1d(hidden, action_dim, 1)

    def forward(
        self,
        x: torch.Tensor,                # [B, H, A]
        t: torch.Tensor,                # [B]
        cond: torch.Tensor,             # [B, cond_input_dim (+latent_dim)]
    ) -> torch.Tensor:
        # Transpose to channel-first for Conv1d
        x = x.transpose(1, 2)           # [B, A, H]

        t_emb = self.time_mlp(t)        # [B, cond_dim]
        c_emb = self.cond_mlp(cond)     # [B, cond_dim]
        c = t_emb + c_emb

        h0 = self.in_conv(x)            # [B, hidden, H]
        d1 = self.down1(h0, c)          # [B, hidden, H]
        d2 = self.down2(d1, c)          # [B, hidden*2, H]
        down = self.downsample(d2)      # [B, hidden*2, H/2]

        m = self.mid1(down, c)
        m = self.mid2(m, c)

        up = self.upsample(m)           # [B, hidden*2, H]
        # Guard against odd-H mismatches if they ever arise
        if up.shape[-1] != d2.shape[-1]:
            up = F.interpolate(up, size=d2.shape[-1], mode="nearest")
        u1 = self.up1(torch.cat([up, d2], dim=1), c)
        u2 = self.up2(torch.cat([u1, d1], dim=1), c)

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out.transpose(1, 2)      # [B, H, A]
