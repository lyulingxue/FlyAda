"""Diffusion policy wrapper: DDPM training + DDIM sampling + receding-horizon rollout.

Action chunks are u_{t:t+H}. Conditioning is a state-derived vector (state only, or
state concatenated with FlyAda latent z_t).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule of alpha_bar from Nichol & Dhariwal. Length T (indices 0..T-1)."""
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / T + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = betas.clamp(0.0, 0.999)
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0).float()
    return alpha_bar  # [T]


@dataclass
class DiffusionConfig:
    action_dim: int = 3
    horizon: int = 8
    cond_dim: int = 12                  # total conditioning input dim (state_dim * frame_stack)
    latent_dim: int = 0
    hidden: int = 128
    T_train: int = 50
    T_sample: int = 20
    exec_k: int = 4
    frame_stack: int = 1                # 1 = single obs; >1 = concatenate last K obs at rollout time


class DiffusionPolicy:
    """Wraps a denoiser with DDPM training + DDIM sampling + receding-horizon execution.

    All tensors are on `device`. Normalization statistics are stored here so that
    training inputs match deployment inputs.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        cfg: DiffusionConfig,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        device: str = "cuda",
    ):
        self.model = denoiser.to(device)
        self.cfg = cfg
        self.device = device
        self.alpha_bar = cosine_alpha_bar(cfg.T_train).to(device)
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------ utils
    def _norm_state(self, s: torch.Tensor) -> torch.Tensor:
        return (s - self.state_mean) / (self.state_std + 1e-6)

    def build_cond(self, state: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self._norm_state(state)
        if self.cfg.latent_dim > 0:
            if latent is None:
                latent = torch.zeros(c.shape[0], self.cfg.latent_dim, device=c.device)
            c = torch.cat([c, latent], dim=-1)
        return c

    # ------------------------------------------------------------------ train
    def training_loss(
        self,
        action_chunk: torch.Tensor,         # [B, H, A], already in [-1, 1]
        state: torch.Tensor,                # [B, cond_dim_raw]
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = action_chunk.shape[0]
        t = torch.randint(0, self.cfg.T_train, (B,), device=self.device)
        noise = torch.randn_like(action_chunk)
        alpha_bar_t = self.alpha_bar[t][:, None, None]
        x_t = alpha_bar_t.sqrt() * action_chunk + (1.0 - alpha_bar_t).sqrt() * noise

        cond = self.build_cond(state, latent=latent)
        eps_pred = self.model(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,                # [B, cond_dim_raw]
        latent: Optional[torch.Tensor] = None,
        T_sample: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        B = state.shape[0]
        H, A = self.cfg.horizon, self.cfg.action_dim
        T_s = int(T_sample or self.cfg.T_sample)

        x = torch.randn(B, H, A, device=self.device, generator=generator)
        cond = self.build_cond(state, latent=latent)

        # DDIM-style index walk from T_train-1 down to 0
        step_indices = torch.linspace(self.cfg.T_train - 1, 0, T_s, device=self.device).long()

        for i, idx in enumerate(step_indices):
            t_batch = torch.full((B,), int(idx), device=self.device, dtype=torch.long)
            eps = self.model(x, t_batch, cond)
            alpha_bar_t = self.alpha_bar[idx]
            x0_pred = (x - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(step_indices):
                idx_next = step_indices[i + 1]
                alpha_bar_next = self.alpha_bar[idx_next]
                # DDIM deterministic update (eta=0)
                eps_used = (x - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-8)
                x = alpha_bar_next.sqrt() * x0_pred + (1.0 - alpha_bar_next).sqrt() * eps_used
            else:
                x = x0_pred
        return x.clamp(-1.0, 1.0)

    # ---------------------------------------------------------- rollout helper
    def make_rollout_policy(self):
        """Returns a stateful callable: policy(obs_numpy) -> action_numpy.

        The callable caches the most recent action chunk and re-samples every
        cfg.exec_k env steps (receding-horizon). Call policy.reset() at episode start.
        When cfg.frame_stack > 1, returns a FrameStackRolloutPolicy that
        maintains a history buffer and feeds stacked obs.
        """
        if self.cfg.frame_stack > 1:
            from .frame_stack_policy import FrameStackRolloutPolicy
            return FrameStackRolloutPolicy(self, self.cfg.frame_stack)

        K = self.cfg.exec_k
        device = self.device

        class _Wrapper:
            def __init__(inner):
                inner._chunk: Optional[np.ndarray] = None
                inner._pos_in_chunk: int = 0

            def reset(inner):
                inner._chunk = None
                inner._pos_in_chunk = 0

            def predict(inner, obs: np.ndarray, deterministic: bool = True):
                if inner._chunk is None or inner._pos_in_chunk >= K:
                    state = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    chunk = self.sample(state)
                    inner._chunk = chunk[0].cpu().numpy()
                    inner._pos_in_chunk = 0
                action = inner._chunk[inner._pos_in_chunk]
                inner._pos_in_chunk += 1
                return action.astype(np.float32), None

        return _Wrapper()
