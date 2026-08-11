"""Deploy a trained FlyAda (or vanilla / frame-stack) partial-obs checkpoint in AirSim.

Usage (client side, after launching an AirSim binary):
    python -m eval.airsim_rollout \
        --ckpt results/flyada_partial_v1/flyada_policy.pt \
        --type flyada --n-seeds 5 --goal-distance 4.0

Coordinate conventions:
    - Our simulator is right-handed, +z up (hover is at z>0).
    - AirSim NED is right-handed, +z DOWN.
    - We flip z at the simulator/AirSim boundary (positions and velocities).

Control pipeline per step:
    1. Query AirSim for kinematics.
    2. Build our 12-dim state [p, v=0, yaw, yaw_dot, g_rel, d_goal] — velocity zeroed
       to match partial-obs training.
    3. Query policy; policy emits a 3-dim normalised acceleration in [-1, 1].
    4. Scale by action_scale (5 m/s^2), integrate into target velocity
       v_target = v_current + a * dt_ctrl, clip to max.
    5. Send via moveByVelocityAsync.

This tests whether the policy trained on our low-dim point-mass simulator transfers
to AirSim's full quadrotor dynamics (rotor thrust, attitude control, nonlinear
aerodynamics). Any degradation quantifies the sim-to-sim gap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DT_CTRL = 0.02          # matches the simulator's 50 Hz training rate
ACTION_SCALE = 5.0      # m/s^2 per axis, matches configs/env.yaml
V_MAX = 5.0             # clip target velocity before sending to AirSim
MAX_STEPS = 800         # per episode
SUCCESS_POS_TOL = 0.5   # same as Task A
SUCCESS_VEL_TOL = 1.0


# ---------------------------------------------------------------------------
# Coordinate conversion: our sim ↔ AirSim NED
# ---------------------------------------------------------------------------

def sim_to_ned(v: np.ndarray) -> np.ndarray:
    """(x, y, z_up) -> (x, y, z_down)."""
    out = v.copy().astype(np.float32)
    out[2] = -out[2]
    return out


def ned_to_sim(v: np.ndarray) -> np.ndarray:
    out = v.copy().astype(np.float32)
    out[2] = -out[2]
    return out


# ---------------------------------------------------------------------------
# Policy loading (dispatches on type)
# ---------------------------------------------------------------------------

def load_policy(ckpt_path: str, policy_type: str, device: str = "cpu"):
    from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint

    if policy_type == "flyada":
        dp, f_phi, fcfg = load_flyada_checkpoint(ckpt_path, device=device)
        from trainers.train_flyada import FlyAdaRolloutPolicy
        return FlyAdaRolloutPolicy(
            dp, f_phi,
            alpha=float(fcfg.get("alpha", 0.1)),
            update_mode=str(fcfg.get("update_mode", "ema")),
        )
    elif policy_type in ("vanilla", "frame_stack"):
        dp = load_diffusion_checkpoint(ckpt_path, device=device)
        return dp.make_rollout_policy()
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


# ---------------------------------------------------------------------------
# AirSim bridge
# ---------------------------------------------------------------------------

def build_state_vec(ned_pos: np.ndarray, ned_vel: np.ndarray, yaw: float,
                    sim_goal: np.ndarray, partial: bool = True) -> np.ndarray:
    """Assemble our 12-dim state from AirSim kinematics.

    Velocity is zeroed when partial=True — that's the regime where FlyAda /
    vanilla-partial / frame-stack were all trained. Full-obs policies (not
    targeted here) would set partial=False.
    """
    sim_pos = ned_to_sim(ned_pos)
    sim_vel = ned_to_sim(ned_vel)
    g_rel = sim_goal - sim_pos
    d_goal = float(np.linalg.norm(g_rel))

    s = np.zeros(12, dtype=np.float32)
    s[0:3] = sim_pos
    s[3:6] = 0.0 if partial else sim_vel
    s[6] = yaw
    s[7] = 0.0
    s[8:11] = g_rel
    s[11] = d_goal
    return s


def run_episode(client, policy, sim_goal: np.ndarray, partial: bool = True):
    """Returns dict with trajectory + success + length."""
    traj_sim = []           # positions in our-sim frame (+z up)
    vels_sim = []
    actions = []
    d_series = []

    policy.reset() if hasattr(policy, "reset") else None

    # Issue current velocity target repeatedly; AirSim's control loop runs much
    # faster than ours so we re-issue at each env step.
    client.enableApiControl(True)
    client.armDisarm(True)

    # Hover at origin first, then take off to a small altitude so we're not at 0.
    client.takeoffAsync().join()
    # Give a moment for steady state.
    time.sleep(0.5)

    # Record origin (after takeoff)
    state = client.getMultirotorState()
    ned_pos0 = np.array([
        state.kinematics_estimated.position.x_val,
        state.kinematics_estimated.position.y_val,
        state.kinematics_estimated.position.z_val,
    ], dtype=np.float32)
    sim_pos0 = ned_to_sim(ned_pos0)

    # Express goal in absolute NED so moves are relative to the spawn origin.
    # sim_goal was provided relative to sim origin (0,0,h); shift to account for
    # the actual post-takeoff position.
    sim_goal_abs = sim_pos0 + sim_goal

    target_v_sim = np.zeros(3, dtype=np.float32)
    success = False
    collided = False

    for step in range(MAX_STEPS):
        state = client.getMultirotorState()
        ned_pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
        ], dtype=np.float32)
        ned_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
        ], dtype=np.float32)
        # AirSim orientation quaternion (w, x, y, z). yaw ≈ 2*atan2(qz*qw, ...)
        # For goal-reaching we rely on yaw≈0 (matches our training distribution).
        ori = state.kinematics_estimated.orientation
        yaw = float(np.arctan2(2.0 * (ori.w_val * ori.z_val + ori.x_val * ori.y_val),
                               1.0 - 2.0 * (ori.y_val**2 + ori.z_val**2)))

        s_vec = build_state_vec(ned_pos, ned_vel, yaw, sim_goal_abs, partial=partial)
        traj_sim.append(s_vec[0:3].copy())
        vels_sim.append(ned_to_sim(ned_vel))
        d_series.append(float(s_vec[11]))

        # Early termination (Task-A-style).
        v_norm = float(np.linalg.norm(vels_sim[-1]))
        if s_vec[11] < SUCCESS_POS_TOL and v_norm < SUCCESS_VEL_TOL:
            success = True
            break

        # Collision check
        coll = client.simGetCollisionInfo()
        if coll.has_collided:
            collided = True
            break

        action, _ = policy.predict(s_vec, deterministic=True)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        actions.append(action)

        # Integrate acceleration command into velocity target (in sim frame),
        # then send to AirSim in NED.
        accel_sim = action * ACTION_SCALE
        target_v_sim = target_v_sim + accel_sim * DT_CTRL
        speed = float(np.linalg.norm(target_v_sim))
        if speed > V_MAX:
            target_v_sim = target_v_sim / speed * V_MAX
        target_v_ned = sim_to_ned(target_v_sim)
        client.moveByVelocityAsync(
            float(target_v_ned[0]), float(target_v_ned[1]), float(target_v_ned[2]),
            duration=DT_CTRL,
        )
        time.sleep(DT_CTRL)   # throttle to ~50 Hz

    client.hoverAsync().join()

    return {
        "success": success,
        "collided": collided,
        "length": len(traj_sim),
        "trajectory": np.stack(traj_sim) if traj_sim else np.zeros((0, 3)),
        "velocities": np.stack(vels_sim) if vels_sim else np.zeros((0, 3)),
        "actions": np.stack(actions) if actions else np.zeros((0, 3)),
        "d_series": np.asarray(d_series, dtype=np.float32),
        "goal_abs_sim": sim_goal_abs,
    }


def sample_goal(rng: np.random.Generator, distance: float) -> np.ndarray:
    direction = rng.standard_normal(3).astype(np.float32)
    direction /= max(np.linalg.norm(direction), 1e-6)
    direction[2] = 0.2 * direction[2]   # keep goals mostly horizontal
    direction /= max(np.linalg.norm(direction), 1e-6)
    return direction * distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--type", type=str, choices=["flyada", "vanilla", "frame_stack"], required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--goal-distance", type=float, default=4.0)
    p.add_argument("--out-dir", type=str, default="results/airsim_flyada_v1")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--full-obs", action="store_true",
                   help="Pass true velocity in the state (not partial). Only use with full-obs ckpts.")
    args = p.parse_args()

    import airsim

    ckpt = args.ckpt
    if not Path(ckpt).is_absolute():
        ckpt = str((ROOT / ckpt).resolve())
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(ckpt, args.type, device=args.device)
    print(f"[airsim_rollout] loaded {args.type} from {ckpt}")

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[airsim_rollout] connected to AirSim")

    rng = np.random.default_rng(42)
    results = []
    for i in range(args.n_seeds):
        print(f"[airsim_rollout] seed {i}: reset + takeoff …")
        client.reset()
        time.sleep(0.3)
        client.enableApiControl(True)
        client.armDisarm(True)

        goal = sample_goal(rng, args.goal_distance)
        print(f"  goal (sim frame, relative) = {goal}")
        ep = run_episode(client, policy, goal, partial=not args.full_obs)
        print(f"  length={ep['length']}  success={ep['success']}  "
              f"collided={ep['collided']}  final_d={ep['d_series'][-1]:.2f}")
        results.append(ep)

    client.armDisarm(False)
    client.enableApiControl(False)

    # Save
    np.savez_compressed(
        out_dir / f"rollouts_{args.type}.npz",
        trajectories=np.array([r["trajectory"] for r in results], dtype=object),
        velocities=np.array([r["velocities"] for r in results], dtype=object),
        actions=np.array([r["actions"] for r in results], dtype=object),
        d_series=np.array([r["d_series"] for r in results], dtype=object),
        goals=np.stack([r["goal_abs_sim"] for r in results]),
        success=np.asarray([r["success"] for r in results], dtype=bool),
        collided=np.asarray([r["collided"] for r in results], dtype=bool),
        length=np.asarray([r["length"] for r in results], dtype=np.int32),
    )
    summary = {
        "type": args.type,
        "n_seeds": args.n_seeds,
        "goal_distance_m": args.goal_distance,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "collision_rate": float(np.mean([r["collided"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "mean_final_dist": float(np.mean([r["d_series"][-1] for r in results])),
    }
    with open(out_dir / f"summary_{args.type}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[airsim_rollout] summary: {summary}")
    print(f"[airsim_rollout] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
