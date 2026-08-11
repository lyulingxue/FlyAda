"""Episode-level loader for demos.npz. FlyAda training samples whole episodes, not windows."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_episodes(demos_path: str | Path, min_len: int = 4) -> List[dict]:
    d = np.load(str(demos_path))
    s = d["s"].astype(np.float32)
    u = d["u"].astype(np.float32)
    g = d["g"].astype(np.float32)
    done = d["done"].astype(bool)

    ep_end_idx = np.nonzero(done)[0]
    if len(ep_end_idx) == 0:
        ep_end_idx = np.array([len(s) - 1])
    ep_start_idx = np.concatenate([[0], ep_end_idx[:-1] + 1])

    episodes: List[dict] = []
    for s_i, e_i in zip(ep_start_idx, ep_end_idx):
        if (e_i - s_i + 1) < min_len:
            continue
        episodes.append({
            "s": s[s_i : e_i + 1].copy(),             # [T, state_dim]
            "u": u[s_i : e_i + 1].copy(),             # [T, action_dim]
            "g": g[s_i].copy(),                        # [3] goal (constant per episode)
            "T": int(e_i - s_i + 1),
        })
    return episodes


def global_state_stats(demos_path: str | Path):
    d = np.load(str(demos_path))
    s = d["s"].astype(np.float32)
    mean = s.mean(axis=0).astype(np.float32)
    std = s.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-3, 1.0, std).astype(np.float32)
    return mean, std
