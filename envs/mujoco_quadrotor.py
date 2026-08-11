"""6-DoF quadrotor in MuJoCo, used as a sim-to-sim transfer testbed.

Public interface mirrors UAVEnv:
  - 12-dim observation: [p, v, yaw, yaw_dot, g_rel, d_goal]; v zeroed when partial_obs.
  - 3-dim action: normalized world-frame acceleration in [-1, 1], scaled by 5 m/s^2.

Internally an inner cascaded controller maps that acceleration command into 4 rotor
forces by (i) computing the required total body-z thrust, (ii) deriving the desired
body-z tilt direction from the desired world-frame net force, (iii) running an
attitude-rate PID, and (iv) mixing the resulting (T, tau_x, tau_y, tau_z) into rotor
commands. The inner loop runs at the MuJoCo simulation rate (500 Hz) under the
50 Hz outer-loop policy update, matching real-quadrotor cascaded control.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import mujoco


XML_PATH = str(Path(__file__).resolve().parent / "mujoco_quadrotor.xml")
ARM_LEN = 0.15
YAW_KT = 0.02      # rotor-drag yaw torque per unit force
ACTION_SCALE = 5.0
GRAVITY = 9.81
SUCCESS_POS_TOL = 0.5
SUCCESS_VEL_TOL = 1.0


class MuJoCoQuadrotorEnv:
    def __init__(
        self,
        xml_path: str = XML_PATH,
        partial_obs: bool = True,
        max_steps: int = 400,
        dt_ctrl: float = 0.02,
        goal_distance_range: Tuple[float, float] = (3.0, 5.0),
        success_pos_tol: float = SUCCESS_POS_TOL,
        success_vel_tol: float = SUCCESS_VEL_TOL,
        # Task B (waypoint chain): set num_waypoints > 1 to enable.
        task: str = "A",
        num_waypoints: int = 1,
        waypoint_step_distance_range: Tuple[float, float] = (2.5, 4.0),
        waypoint_advance_tol: float = 0.5,
        # Optional dynamics perturbations applied AFTER the policy is loaded.
        mass_scale: float = 1.0,
        drag_world: float = 0.0,
        wind_world: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.partial_obs = bool(partial_obs)
        self.max_steps = int(max_steps)
        self.dt_ctrl = float(dt_ctrl)
        self.dt_sim = float(self.model.opt.timestep)
        self.steps_per_ctrl = max(1, int(round(self.dt_ctrl / self.dt_sim)))
        self.goal_distance_range = tuple(goal_distance_range)
        self.success_pos_tol = float(success_pos_tol)
        self.success_vel_tol = float(success_vel_tol)
        self.task = str(task).upper()
        self.num_waypoints = int(num_waypoints if self.task == "B" else 1)
        self.waypoint_step_distance_range = tuple(waypoint_step_distance_range)
        self.waypoint_advance_tol = float(waypoint_advance_tol)

        # Apply mass perturbation by scaling body inertial properties.
        body_id = self.model.body("quad").id
        self.model.body_mass[body_id] *= float(mass_scale)
        self.model.body_inertia[body_id] *= float(mass_scale)
        self.mass = float(self.model.body_mass[body_id])
        self.drag_world = float(drag_world)
        self.wind_world = np.asarray(wind_world, dtype=np.float32)

        # Mixer matrix: [T, tau_x, tau_y, tau_z]^T = M @ [F1, F2, F3, F4]^T
        L = ARM_LEN
        kT = YAW_KT
        self.mixer = np.array([
            [1.0,   1.0,   1.0,   1.0 ],
            [ L,     L,    -L,    -L  ],
            [-L,     L,     L,    -L  ],
            [-kT,    kT,   -kT,    kT ],
        ], dtype=np.float32)
        self.mixer_inv = np.linalg.inv(self.mixer).astype(np.float32)

        # Outer-loop / inner-loop PID gains. Tuned by inspection on hover + step.
        self.kp_att  = 18.0   # attitude P (tilt-error -> body-rate target)
        self.kp_rate = 4.5    # rate P (rate error -> body torque)
        self.kp_yaw  = 8.0    # yaw error -> yaw-rate target

        self._step_count = 0
        self._goal = np.zeros(3, dtype=np.float32)
        self._waypoints: list = []
        self._wp_idx = 0
        self._init_height = 1.5

    # ---- gym-like API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              goal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, self._init_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quaternion (w,x,y,z)
        self.data.qvel[:] = 0.0

        if self.task == "B":
            # Sample N sequential waypoints from the start position.
            start_pos = np.array([0.0, 0.0, self._init_height], dtype=np.float32)
            self._waypoints = self._sample_waypoint_chain(rng, start_pos)
            self._wp_idx = 0
            self._goal = self._waypoints[0]
        elif goal is None:
            lo, hi = self.goal_distance_range
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            d_xy = float(rng.uniform(lo, hi))
            dz = float(rng.uniform(-1.0, 1.0))
            self._goal = np.array([
                d_xy * np.cos(angle),
                d_xy * np.sin(angle),
                self._init_height + dz,
            ], dtype=np.float32)
            self._waypoints = [self._goal]; self._wp_idx = 0
        else:
            self._goal = np.asarray(goal, dtype=np.float32)
            self._waypoints = [self._goal]; self._wp_idx = 0

        self._step_count = 0
        return self._obs(), {
            "goal": self._goal.copy(),
            "waypoints": [wp.copy() for wp in self._waypoints],
            "wp_idx": self._wp_idx,
        }

    def _sample_waypoint_chain(self, rng: np.random.Generator, start: np.ndarray) -> list:
        lo, hi = self.waypoint_step_distance_range
        wps = []
        cur = start.copy()
        for _ in range(self.num_waypoints):
            for _ in range(100):
                dxy_dir = rng.standard_normal(2).astype(np.float32)
                dxy_dir /= max(np.linalg.norm(dxy_dir), 1e-6)
                dxy = float(rng.uniform(lo, hi))
                dz = float(rng.uniform(-0.6, 0.6))
                wp = cur + np.array([dxy_dir[0] * dxy, dxy_dir[1] * dxy, dz], dtype=np.float32)
                # keep z above the ground
                if wp[2] >= 0.5:
                    wps.append(wp.astype(np.float32))
                    cur = wp
                    break
            else:
                wp = (cur + np.array([lo, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
                wps.append(wp); cur = wp
        return wps

    def _quat_to_R(self, w: float, x: float, y: float, z: float) -> np.ndarray:
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        pos = self.data.qpos[0:3].astype(np.float32)
        vel = self.data.qvel[0:3].astype(np.float32)
        w, x, y, z = self.data.qpos[3:7]
        yaw = float(np.arctan2(2.0 * (w * z + x * y),
                               1.0 - 2.0 * (y * y + z * z)))
        ang_vel = self.data.qvel[3:6].astype(np.float32)
        yaw_dot = float(ang_vel[2])
        g_rel = (self._goal - pos).astype(np.float32)
        d_goal = float(np.linalg.norm(g_rel))

        obs = np.zeros(12, dtype=np.float32)
        obs[0:3] = pos
        obs[3:6] = vel
        obs[6] = yaw
        obs[7] = yaw_dot
        obs[8:11] = g_rel
        obs[11] = d_goal
        if self.partial_obs:
            obs[3:6] = 0.0
        return obs

    def step(self, action_norm: np.ndarray):
        a_des_world = (np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)
                       * ACTION_SCALE)

        for _ in range(self.steps_per_ctrl):
            qw, qx, qy, qz = self.data.qpos[3:7]
            R = self._quat_to_R(qw, qx, qy, qz)
            body_z = R[:, 2]

            # Optional extra world drag + wind (defaults are off).
            if self.drag_world > 0.0 or np.any(self.wind_world != 0.0):
                v_world = self.data.qvel[0:3].astype(np.float32)
                drag = -self.drag_world * v_world + self.wind_world
                self.data.qfrc_applied[0:3] = self.mass * drag
            else:
                self.data.qfrc_applied[0:3] = 0.0

            # Outer (acceleration) loop: required net world force.
            f_des_world = self.mass * np.array([a_des_world[0],
                                                 a_des_world[1],
                                                 a_des_world[2] + GRAVITY], dtype=np.float32)

            # Required total thrust along current body-z.
            T_des = float(np.dot(f_des_world, body_z))
            T_des = float(np.clip(T_des, 0.0, 20.0))

            # Desired body-z direction (where we want to tilt).
            f_norm = float(np.linalg.norm(f_des_world))
            z_des = (f_des_world / f_norm) if f_norm > 1e-3 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Tilt error: rotation needed to bring body-z onto z_des, expressed
            # in the body frame for the rate controller.
            err_world = np.cross(body_z, z_des)
            err_body = R.T @ err_world
            ang_vel_body = R.T @ self.data.qvel[3:6].astype(np.float32)

            yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))
            rate_des = np.array([
                self.kp_att * err_body[0],
                self.kp_att * err_body[1],
                self.kp_yaw * (-yaw),
            ], dtype=np.float32)

            tau_body = self.kp_rate * (rate_des - ang_vel_body)

            # Solve mixer for rotor forces.
            T_tau = np.array([T_des, tau_body[0], tau_body[1], tau_body[2]], dtype=np.float32)
            F = self.mixer_inv @ T_tau
            F = np.clip(F, 0.0, 6.0).astype(np.float32)

            self.data.ctrl[:] = F
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        # Recompute proprioception (post-step) for waypoint advancement check.
        pos_now = self.data.qpos[0:3].astype(np.float32)
        v_norm = float(np.linalg.norm(self.data.qvel[0:3]))
        d_now = float(np.linalg.norm(self._goal - pos_now))

        wp_advanced = False
        if self.task == "B":
            is_final = (self._wp_idx == len(self._waypoints) - 1)
            advance = d_now < self.waypoint_advance_tol
            if is_final:
                advance = advance and (v_norm < self.success_vel_tol)
            if advance:
                if is_final:
                    success = True
                else:
                    self._wp_idx += 1
                    self._goal = self._waypoints[self._wp_idx]
                    wp_advanced = True
                    success = False
            else:
                success = False
        else:
            success = (d_now < self.success_pos_tol) and (v_norm < self.success_vel_tol)

        obs = self._obs()
        oob = bool(np.any(np.abs(self.data.qpos[0:3]) > 20.0))
        terminated = bool(success or oob)
        truncated = bool(self._step_count >= self.max_steps)
        info = dict(d_goal=float(obs[11]), v_norm=v_norm, success=success,
                    out_of_bounds=oob, wp_idx=self._wp_idx,
                    num_waypoints=len(self._waypoints), wp_advanced=wp_advanced)
        return obs, 0.0, terminated, truncated, info
