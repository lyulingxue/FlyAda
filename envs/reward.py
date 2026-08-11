"""Shaping + terminal reward for Task A (PLAN.md Section 4.6)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RewardParams:
    dist_coeff: float = 0.1
    ctrl_coeff: float = 0.01
    collision_penalty: float = 1.0
    success_bonus: float = 100.0


def compute_reward(
    params: RewardParams,
    d_goal: float,
    action_norm: np.ndarray,
    collided: bool,
    success: bool,
) -> float:
    r = -params.dist_coeff * float(d_goal)
    r -= params.ctrl_coeff * float(np.sum(np.square(action_norm)))
    if collided:
        r -= params.collision_penalty
    if success:
        r += params.success_bonus
    return r
