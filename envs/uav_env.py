"""Low-dim UAV gym environment (PLAN.md Section 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from .dynamics import DynamicsParams, UAVDynamics
from .obstacles import Box, Cylinder, ObstacleField
from .reward import RewardParams, compute_reward


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class UAVEnv(gym.Env):
    """Task A — goal-reaching UAV env.

    State (12-dim):
      [p_x, p_y, p_z, v_x, v_y, v_z, yaw, yaw_dot, g_rel_x, g_rel_y, g_rel_z, d_goal]
    Action (3-dim): normalized acceleration command in [-1, 1] along each axis.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: str | Path | Dict[str, Any] | None = None,
        dynamics_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if config is None:
            cfg = _load_yaml(Path(__file__).resolve().parents[1] / "configs" / "env.yaml")
        elif isinstance(config, (str, Path)):
            cfg = _load_yaml(config)
        else:
            cfg = dict(config)

        self._cfg = cfg
        self.dt = float(cfg["dt"])
        self.max_steps = int(cfg["max_steps"])
        self.state_dim = int(cfg["state_dim"])
        self.action_dim = int(cfg["action_dim"])
        self.obs_mode = str(cfg.get("obs_mode", "full"))  # "full" or "partial" (velocity zeroed)
        self.pos_bound = float(cfg["pos_bound"])
        self.goal_distance_range = tuple(cfg["goal_distance_range"])
        self.success_pos_tol = float(cfg["success_pos_tol"])
        self.success_vel_tol = float(cfg["success_vel_tol"])

        # Task A (goal-reaching) is the default. Task B (waypoint-following) extends
        # it by chaining N sequential waypoints; the observation always points to
        # the current waypoint so policies trained on Task A transfer without retraining.
        self.task = str(cfg.get("task", "A")).upper()
        self.num_waypoints = int(cfg.get("num_waypoints", 1 if self.task == "A" else 3))
        self.waypoint_step_distance_range = tuple(cfg.get("waypoint_step_distance_range", [2.0, 4.0]))
        self.waypoint_advance_tol = float(cfg.get("waypoint_advance_tol", self.success_pos_tol))
        self.require_stop_on_final = bool(cfg.get("require_stop_on_final_wp", True))

        self.num_obstacles_range = tuple(cfg["num_obstacles_range"])
        self.obstacle_box_half_range = tuple(cfg["obstacle_box_half_range"])
        self.obstacle_cyl_radius_range = tuple(cfg["obstacle_cyl_radius_range"])
        self.obstacle_cyl_height_range = tuple(cfg["obstacle_cyl_height_range"])
        self.obstacle_clearance = float(cfg["obstacle_clearance"])

        rcfg = cfg["reward"]
        self.reward_params = RewardParams(
            dist_coeff=float(rcfg["dist_coeff"]),
            ctrl_coeff=float(rcfg["ctrl_coeff"]),
            collision_penalty=float(rcfg["collision_penalty"]),
            success_bonus=float(rcfg["success_bonus"]),
        )

        dyn_cfg = dict(
            dt=self.dt,
            mass=float(cfg["mass"]),
            drag=float(cfg["drag"]),
            wind=np.asarray(cfg.get("wind", [0.0, 0.0, 0.0]), dtype=np.float32),
            control_delay=int(cfg.get("control_delay", 0)),
            action_scale=float(cfg.get("action_scale", 5.0)),
        )
        if dynamics_overrides:
            for k, v in dynamics_overrides.items():
                if k == "wind":
                    dyn_cfg["wind"] = np.asarray(v, dtype=np.float32)
                else:
                    dyn_cfg[k] = v
        self._dyn_params = DynamicsParams(**dyn_cfg)
        self.dynamics = UAVDynamics(self._dyn_params)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        high = np.full((self.state_dim,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # per-episode state
        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._yaw_dot = 0.0
        self._goal = np.zeros(3, dtype=np.float32)
        self._waypoints: List[np.ndarray] = []
        self._wp_idx: int = 0
        self._step = 0
        self._obstacles = ObstacleField([])

        self._np_random: np.random.Generator | None = None

    # ------------------------------------------------------------------ util
    def _sample_free_point(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(100):
            p = rng.uniform(-self.pos_bound, self.pos_bound, size=3).astype(np.float32)
            if self._obstacles.min_sdf(p) > self.obstacle_clearance:
                return p
        # Fallback: ignore clearance to avoid infinite loops in crowded configs
        return rng.uniform(-self.pos_bound, self.pos_bound, size=3).astype(np.float32)

    def _sample_goal(self, rng: np.random.Generator, start: np.ndarray) -> np.ndarray:
        lo, hi = self.goal_distance_range
        for _ in range(200):
            direction = rng.standard_normal(3).astype(np.float32)
            direction /= max(np.linalg.norm(direction), 1e-6)
            dist = rng.uniform(lo, hi)
            g = start + direction * dist
            # keep goal inside workspace
            if np.all(np.abs(g) <= self.pos_bound) and self._obstacles.min_sdf(g) > self.obstacle_clearance:
                return g.astype(np.float32)
        return (start + np.array([lo, 0.0, 0.0], dtype=np.float32)).astype(np.float32)

    def _sample_waypoint_chain(self, rng: np.random.Generator, start: np.ndarray) -> List[np.ndarray]:
        """Task B: sample N sequential waypoints, each at a random distance from the previous.

        Waypoints may extend beyond pos_bound (multi-waypoint flight paths span a
        larger effective volume). We clip to 1.8 * pos_bound so they stay in a
        reachable region, and fall back to a straight-line layout if rejection
        sampling fails.
        """
        lo, hi = self.waypoint_step_distance_range
        wps: List[np.ndarray] = []
        current = start.copy()
        bound = self.pos_bound * 1.8
        for _ in range(self.num_waypoints):
            for _ in range(200):
                direction = rng.standard_normal(3).astype(np.float32)
                direction /= max(np.linalg.norm(direction), 1e-6)
                dist = float(rng.uniform(lo, hi))
                wp = current + direction * dist
                if np.all(np.abs(wp) <= bound) and self._obstacles.min_sdf(wp) > self.obstacle_clearance:
                    wps.append(wp.astype(np.float32))
                    current = wp
                    break
            else:
                wp = (current + np.array([lo, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
                wps.append(wp)
                current = wp
        return wps

    def _sample_obstacles(self, rng: np.random.Generator) -> List:
        n_lo, n_hi = self.num_obstacles_range
        n = int(rng.integers(n_lo, n_hi + 1))
        prims: List = []
        for _ in range(n):
            if rng.random() < 0.5:
                center = rng.uniform(-self.pos_bound * 0.7, self.pos_bound * 0.7, size=3).astype(np.float32)
                lo, hi = self.obstacle_box_half_range
                half = rng.uniform(lo, hi, size=3).astype(np.float32)
                prims.append(Box(center=center, half=half))
            else:
                center_xy = rng.uniform(-self.pos_bound * 0.7, self.pos_bound * 0.7, size=2).astype(np.float32)
                lo_r, hi_r = self.obstacle_cyl_radius_range
                lo_h, hi_h = self.obstacle_cyl_height_range
                radius = float(rng.uniform(lo_r, hi_r))
                h = float(rng.uniform(lo_h, hi_h))
                z_c = float(rng.uniform(-self.pos_bound * 0.5, self.pos_bound * 0.5))
                prims.append(Cylinder(center_xy=center_xy, z_min=z_c - h / 2, z_max=z_c + h / 2, radius=radius))
        return prims

    def _obs(self) -> np.ndarray:
        g_rel = self._goal - self._pos
        d_goal = float(np.linalg.norm(g_rel))
        obs = np.zeros(self.state_dim, dtype=np.float32)
        obs[0:3] = self._pos
        obs[3:6] = self._vel
        obs[6] = self._yaw
        obs[7] = self._yaw_dot
        obs[8:11] = g_rel
        obs[11] = d_goal
        if self.obs_mode == "partial":
            # Velocity hidden from the policy. The env still integrates with the true
            # velocity internally; only the observation is masked.
            obs[3:6] = 0.0
        return obs

    # ----------------------------------------------------------------- gym API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        self._obstacles = ObstacleField(self._sample_obstacles(rng))
        self._pos = self._sample_free_point(rng)
        if self.task == "B":
            self._waypoints = self._sample_waypoint_chain(rng, self._pos)
            self._wp_idx = 0
            self._goal = self._waypoints[0]
        else:
            self._goal = self._sample_goal(rng, self._pos)
            self._waypoints = [self._goal]
            self._wp_idx = 0
        self._vel = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._yaw_dot = 0.0
        self._step = 0
        self.dynamics.reset()
        return self._obs(), {
            "goal": self._goal.copy(),
            "waypoints": [wp.copy() for wp in self._waypoints],
            "wp_idx": self._wp_idx,
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self._pos, self._vel, u_applied = self.dynamics.step(self._pos, self._vel, action)
        self._step += 1

        d_goal = float(np.linalg.norm(self._goal - self._pos))
        v_norm = float(np.linalg.norm(self._vel))
        collided = self._obstacles.collides(self._pos, margin=0.0)
        out_of_bounds = bool(np.any(np.abs(self._pos) > self.pos_bound * 2.0))

        wp_advanced = False
        if self.task == "B":
            is_final = (self._wp_idx == len(self._waypoints) - 1)
            advance = d_goal < self.waypoint_advance_tol
            if is_final and self.require_stop_on_final:
                advance = advance and (v_norm < self.success_vel_tol)
            if advance:
                if is_final:
                    success = True
                else:
                    self._wp_idx += 1
                    self._goal = self._waypoints[self._wp_idx]
                    wp_advanced = True
                    d_goal = float(np.linalg.norm(self._goal - self._pos))
                    success = False
            else:
                success = False
        else:
            success = (d_goal < self.success_pos_tol) and (v_norm < self.success_vel_tol)

        reward = compute_reward(
            self.reward_params,
            d_goal=d_goal,
            action_norm=action,
            collided=collided,
            success=success,
        )
        terminated = bool(success or collided)
        truncated = bool(self._step >= self.max_steps)

        info = {
            "d_goal": d_goal,
            "v_norm": v_norm,
            "collided": collided,
            "success": success,
            "out_of_bounds": out_of_bounds,
            "u_applied": u_applied,
            "wp_idx": self._wp_idx,
            "num_waypoints": len(self._waypoints),
            "wp_advanced": wp_advanced,
        }
        return self._obs(), float(reward), terminated, truncated, info

    # --------------------------------------------------- introspection helpers
    @property
    def goal(self) -> np.ndarray:
        return self._goal.copy()

    @property
    def waypoints(self) -> List[np.ndarray]:
        return [wp.copy() for wp in self._waypoints]

    @property
    def wp_idx(self) -> int:
        return int(self._wp_idx)

    @property
    def obstacles(self) -> ObstacleField:
        return self._obstacles

    @property
    def dynamics_params(self) -> DynamicsParams:
        return self._dyn_params
