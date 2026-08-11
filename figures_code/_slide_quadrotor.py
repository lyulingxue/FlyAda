"""A small 3D quadrotor renderer for matplotlib Axes3D.

Draws an X-configuration quadrotor (four arms, four rotor disks with spinning
blades, a body box) at a given position and attitude. Attitude is derived from
the commanded acceleration so the drone visibly banks into its turns, which is
what makes the baseline-vs-FlyAda comparison readable at a glance.
"""
from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

G = 9.81


# ---------------------------------------------------------------- attitude ---
def attitude_from_command(a_cmd: np.ndarray, vel: np.ndarray, action_scale: float = 5.0):
    """Roll/pitch from the horizontal acceleration command, yaw from heading.

    The home simulator is a translational point mass, so there is no true
    attitude to read out. For the visualisation we invert the usual small-angle
    quadrotor relation a_xy = g * [theta, -phi], which is exactly the tilt a real
    vehicle would need to produce the commanded acceleration.
    """
    a = np.asarray(a_cmd, float) * action_scale
    pitch = np.arctan2(a[0], G)
    roll = np.arctan2(-a[1], G)
    v = np.asarray(vel, float)
    yaw = np.arctan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 1e-3 else 0.0
    return roll, pitch, yaw


def rot_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ------------------------------------------------------------------ meshes ---
_ARM_DIRS = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], float)
_ARM_DIRS /= np.linalg.norm(_ARM_DIRS, axis=1, keepdims=True)

_DISK_THETA = np.linspace(0, 2 * np.pi, 20)


def quadrotor_geometry(pos, R, arm_len=0.55, rotor_r=0.26, blade_phase=0.0):
    """Return (arm_segments, rotor_polys, blade_segments) in world coordinates.

    All returned arrays are ready to hand to Line3DCollection / Poly3DCollection.
    """
    pos = np.asarray(pos, float)
    hubs_body = _ARM_DIRS * arm_len

    arms, disks, blades = [], [], []
    for i, hub_b in enumerate(hubs_body):
        hub_w = pos + R @ hub_b
        arms.append([pos, hub_w])

        # Rotor disk: a circle in the body xy-plane, lifted slightly above the arm.
        circle_b = np.stack(
            [rotor_r * np.cos(_DISK_THETA), rotor_r * np.sin(_DISK_THETA),
             np.zeros_like(_DISK_THETA)], axis=1
        ) + hub_b + np.array([0, 0, 0.05])
        disks.append((R @ circle_b.T).T + pos)

        # A single blade across the disk, spun by blade_phase (counter-rotating pairs).
        ph = blade_phase * (1 if i % 2 == 0 else -1)
        d = np.array([np.cos(ph), np.sin(ph), 0.0]) * rotor_r
        b0 = hub_b + np.array([0, 0, 0.05]) - d
        b1 = hub_b + np.array([0, 0, 0.05]) + d
        blades.append([pos + R @ b0, pos + R @ b1])

    return np.array(arms), disks, np.array(blades)


def body_box(pos, R, half=(0.20, 0.14, 0.07)):
    """Six quad faces of the fuselage box."""
    hx, hy, hz = half
    c = np.array([[sx * hx, sy * hy, sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    c = (R @ c.T).T + np.asarray(pos, float)
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
           (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [c[list(f)] for f in idx]


class QuadrotorArtist:
    """Persistent collection of artists for one drone, updated in place."""

    def __init__(self, ax, color, arm_len=0.55, rotor_r=0.26, scale=1.0, zorder=10):
        self.color = color
        self.arm_len = arm_len * scale
        self.rotor_r = rotor_r * scale
        self.scale = scale

        self.arms = Line3DCollection([], colors=color, linewidths=2.4, zorder=zorder)
        self.blades = Line3DCollection([], colors="#222222", linewidths=1.1,
                                       alpha=0.85, zorder=zorder + 2)
        self.disks = Poly3DCollection([], facecolors=color, edgecolors=color,
                                      alpha=0.30, linewidths=1.0, zorder=zorder + 1)
        self.body = Poly3DCollection([], facecolors=color, edgecolors="#1a1a1a",
                                     alpha=0.95, linewidths=0.5, zorder=zorder + 1)
        for a in (self.arms, self.disks, self.body, self.blades):
            # autolim=False: the artists start empty and the caller has already
            # fixed the axis limits from the trajectory extent.
            try:
                ax.add_collection3d(a, autolim=False)
            except TypeError:
                ax.add_collection3d(a)

    def update(self, pos, a_cmd, vel, blade_phase=0.0, action_scale=5.0):
        roll, pitch, yaw = attitude_from_command(a_cmd, vel, action_scale)
        R = rot_matrix(roll, pitch, yaw)
        arms, disks, blades = quadrotor_geometry(
            pos, R, self.arm_len, self.rotor_r, blade_phase)
        self.arms.set_segments(list(arms))
        self.blades.set_segments(list(blades))
        self.disks.set_verts(disks)
        self.body.set_verts(body_box(pos, R, half=(0.20 * self.scale,
                                                   0.14 * self.scale,
                                                   0.07 * self.scale)))
        return R

    def set_visible(self, flag: bool):
        for a in (self.arms, self.disks, self.body, self.blades):
            a.set_visible(flag)
