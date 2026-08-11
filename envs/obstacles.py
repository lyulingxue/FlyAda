"""Axis-aligned box and vertical cylinder obstacles with signed-distance collision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Box:
    center: np.ndarray      # (3,)
    half: np.ndarray        # (3,) half-extents

    def sdf(self, p: np.ndarray) -> float:
        q = np.abs(p - self.center) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return float(outside + inside)


@dataclass
class Cylinder:
    """Vertical cylinder, axis along z."""
    center_xy: np.ndarray   # (2,)
    z_min: float
    z_max: float
    radius: float

    def sdf(self, p: np.ndarray) -> float:
        dxy = np.linalg.norm(p[:2] - self.center_xy) - self.radius
        dz_below = self.z_min - p[2]
        dz_above = p[2] - self.z_max
        dz = max(dz_below, dz_above)
        if dxy <= 0.0 and dz <= 0.0:
            return float(max(dxy, dz))
        dxy_p = max(dxy, 0.0)
        dz_p = max(dz, 0.0)
        return float(np.hypot(dxy_p, dz_p))


class ObstacleField:
    def __init__(self, primitives: List):
        self.prims = list(primitives)

    def min_sdf(self, p: np.ndarray) -> float:
        if not self.prims:
            return 1e6
        return min(prim.sdf(p) for prim in self.prims)

    def collides(self, p: np.ndarray, margin: float = 0.0) -> bool:
        return self.min_sdf(p) < margin
