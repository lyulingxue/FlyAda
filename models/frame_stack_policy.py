"""Frame-stacking rollout wrapper for a DiffusionPolicy.

Maintains the last K observations internally and feeds the stacked vector
[s_{t-K+1}, ..., s_t] as the policy's conditioning each time a new action chunk
is sampled. Before K obs are accumulated, the first obs is repeated — same
convention as in training.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .diffusion_policy import DiffusionPolicy


class FrameStackRolloutPolicy:
    def __init__(self, diffusion: DiffusionPolicy, frame_stack: int):
        self.dp = diffusion
        self.K = int(frame_stack)
        self.reset()

    def reset(self):
        self._history: list[np.ndarray] = []
        self._chunk: Optional[np.ndarray] = None
        self._pos: int = 0

    def _stacked(self, obs: np.ndarray) -> np.ndarray:
        if not self._history:
            self._history = [obs.copy() for _ in range(self.K)]
        else:
            self._history.pop(0)
            self._history.append(obs.copy())
        return np.concatenate(self._history).astype(np.float32)

    @torch.no_grad()
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        stacked = self._stacked(obs)
        if self._chunk is None or self._pos >= self.dp.cfg.exec_k:
            state = torch.as_tensor(stacked, dtype=torch.float32, device=self.dp.device).unsqueeze(0)
            chunk = self.dp.sample(state)
            self._chunk = chunk[0].cpu().numpy()
            self._pos = 0
        action = self._chunk[self._pos]
        self._pos += 1
        return action.astype(np.float32), None
