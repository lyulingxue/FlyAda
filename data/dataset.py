"""Windowed (s_t, u_{t:t+H}) dataset over demos.npz respecting episode boundaries.

Short-episode padding: if an episode ends before t+H, the last valid action is
repeated to fill the chunk — same convention as the diffusion-policy reference
implementation. The policy runs receding-horizon so trailing noise in padded
windows doesn't affect deployment.

Optional `frame_stack`: the returned state concatenates the last K observations
[s_{t-K+1}, ..., s_t]. At episode start we left-pad by repeating s_0. Used by
the frame-stacking baseline that compensates for partial observability by giving
the policy access to recent history directly rather than via a learned latent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, demos_path: str | Path, horizon: int = 8, frame_stack: int = 1):
        d = np.load(str(demos_path))
        self.s = d["s"].astype(np.float32)
        self.u = d["u"].astype(np.float32)
        self.done = d["done"].astype(bool)
        self.g = d["g"].astype(np.float32)
        self.H = int(horizon)
        self.frame_stack = int(frame_stack)

        ep_end_idx = np.nonzero(self.done)[0]
        if len(ep_end_idx) == 0:
            ep_end_idx = np.array([len(self.s) - 1])
        ep_start_idx = np.concatenate([[0], ep_end_idx[:-1] + 1])

        tts: list[int] = []
        ends: list[int] = []
        starts: list[int] = []
        for s, e in zip(ep_start_idx, ep_end_idx):
            for t in range(int(s), int(e) + 1):
                tts.append(t)
                ends.append(int(e))
                starts.append(int(s))
        self._t = np.asarray(tts, dtype=np.int64)
        self._end = np.asarray(ends, dtype=np.int64)
        self._start = np.asarray(starts, dtype=np.int64)

        # Per-frame state normalization stats (tiled across the stack at model construction time)
        self.state_mean = self.s.mean(axis=0).astype(np.float32)
        self.state_std = self.s.std(axis=0).astype(np.float32)
        self.state_std = np.where(self.state_std < 1e-3, 1.0, self.state_std).astype(np.float32)

    def __len__(self) -> int:
        return int(len(self._t))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t = int(self._t[i])
        e = int(self._end[i])
        ep_start = int(self._start[i])

        if self.frame_stack <= 1:
            state = self.s[t]
        else:
            K = self.frame_stack
            # Gather [s_{t-K+1}, ..., s_t], clamping at episode start (repeat s_{ep_start})
            hist_idx = np.maximum(np.arange(-K + 1, 1) + t, ep_start)
            state = self.s[hist_idx].reshape(-1)   # [K * state_dim]

        chunk_idx = np.minimum(np.arange(self.H) + t, e)
        return {
            "state": torch.from_numpy(state),
            "action_chunk": torch.from_numpy(self.u[chunk_idx]).clamp(-1.0, 1.0),
        }
