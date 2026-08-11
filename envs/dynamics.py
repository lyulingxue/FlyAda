"""Discrete UAV dynamics model (PLAN.md Section 4.3)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DynamicsParams:
    dt: float = 0.02
    mass: float = 1.0
    drag: float = 0.1
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    control_delay: int = 0
    action_scale: float = 5.0


class UAVDynamics:
    """3-DoF translational UAV: acceleration-command, first-order drag, optional wind + input delay."""

    def __init__(self, params: DynamicsParams):
        self.p = params
        self._delay_buf: deque[np.ndarray] = deque()

    def reset(self):
        self._delay_buf.clear()
        # Pre-fill delay buffer with zeros so the first k steps use zero action.
        for _ in range(self.p.control_delay):
            self._delay_buf.append(np.zeros(3, dtype=np.float32))

    def step(self, pos: np.ndarray, vel: np.ndarray, action_norm: np.ndarray):
        """One discrete-time step. `action_norm` is in [-1, 1]; scaled to acceleration here."""
        u_cmd = np.clip(action_norm, -1.0, 1.0).astype(np.float32) * self.p.action_scale

        if self.p.control_delay > 0:
            self._delay_buf.append(u_cmd)
            u_applied = self._delay_buf.popleft()
        else:
            u_applied = u_cmd

        accel = u_applied / self.p.mass - self.p.drag * vel + self.p.wind
        v_next = vel + self.p.dt * accel
        p_next = pos + self.p.dt * v_next
        return p_next.astype(np.float32), v_next.astype(np.float32), u_applied.astype(np.float32)
