"""Collect rich rollout traces for the WRC SARA 2026 oral-presentation visuals.

Unlike eval/*, this records the *hidden* state as well: true velocity, the FlyAda
latent z_t, and the velocity decoded from z_t. Those are what the belief-state
animations show.

Traces are cached to paper/figures/slides/_traces_<tag>.npz so the renderers can
be iterated on without re-running the policies.

Usage:
    python -m paper._slide_rollouts            # collect everything
    python -m paper._slide_rollouts --taskB    # just Task B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv                                       # noqa: E402
from models.checkpoint import (                               # noqa: E402
    load_diffusion_checkpoint,
    load_flyada_checkpoint,
)
from trainers.train_flyada import FlyAdaRolloutPolicy         # noqa: E402

OUT_DIR = ROOT / "paper" / "figures" / "slides"

VANILLA_CKPT = ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"
FRAME3_CKPT = ROOT / "results/diffusion_partial_frame3_v1/diffusion_policy.pt"
FLYADA_CKPT = ROOT / "results/flyada_partial_v1/flyada_policy.pt"

METHODS = ["Vanilla", "Frame-stack", "FlyAda"]

# The hard combined condition used in the paper's Task B "hard" row.
HARD_DYN = {"mass": 1.3, "drag": 0.2, "wind": [1.0, 0.0, 0.0], "control_delay": 2}


def build_policies(device: str = "cuda"):
    van = load_diffusion_checkpoint(str(VANILLA_CKPT), device=device)
    fst = load_diffusion_checkpoint(str(FRAME3_CKPT), device=device)
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(str(FLYADA_CKPT), device=device)
    alpha = float(fcfg.get("alpha", 0.3))
    mode = str(fcfg.get("update_mode", "ema"))

    factories = {
        "Vanilla": lambda: van.make_rollout_policy(),
        "Frame-stack": lambda: fst.make_rollout_policy(),
        "FlyAda": lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=alpha, update_mode=mode),
    }
    return factories, f_phi


def trace_episode(env: UAVEnv, policy, seed: int, f_phi=None) -> Dict[str, np.ndarray]:
    """Roll one episode, recording the observable *and* the hidden channels."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)

    pos, vel, acts, zs, vhat, dgoal, wpidx = [], [], [], [], [], [], []
    pos.append(env._pos.copy())
    vel.append(env._vel.copy())

    def _record_latent():
        """z_t and dec(z_t) if this policy carries a latent; zeros otherwise."""
        if f_phi is None or not hasattr(policy, "_z"):
            return np.zeros(32, np.float32), np.zeros(3, np.float32)
        z = policy._z.detach()
        with torch.no_grad():
            v = f_phi.decode_velocity(z)
        return z[0].cpu().numpy(), v[0].cpu().numpy()

    z0, v0 = _record_latent()
    zs.append(z0)
    vhat.append(v0)

    steps = 0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        steps += 1

        pos.append(env._pos.copy())
        vel.append(env._vel.copy())
        acts.append(np.asarray(action, np.float32))
        z, v = _record_latent()
        zs.append(z)
        vhat.append(v)
        dgoal.append(float(info["d_goal"]))
        wpidx.append(int(info.get("wp_idx", 0)))

        if term or trunc:
            break

    # Pad the action series so every array shares the T+1 time base.
    acts.append(acts[-1] if acts else np.zeros(3, np.float32))
    dgoal.append(dgoal[-1] if dgoal else 0.0)
    wpidx.append(wpidx[-1] if wpidx else 0)

    return {
        "pos": np.asarray(pos, np.float32),
        "vel": np.asarray(vel, np.float32),
        "act": np.asarray(acts, np.float32),
        "z": np.asarray(zs, np.float32),
        "vhat": np.asarray(vhat, np.float32),
        "d_goal": np.asarray(dgoal, np.float32),
        "wp_idx": np.asarray(wpidx, np.int32),
        "success": np.asarray(bool(info.get("success", False))),
        "length": np.asarray(steps),
    }


def collect_taskB(tag: str, dyn_over: dict | None, seed: int, device: str = "cuda"):
    factories, f_phi = build_policies(device)
    env = UAVEnv(config=str(ROOT / "configs/env_partial_taskB.yaml"),
                 dynamics_overrides=dyn_over)

    blob: Dict[str, np.ndarray] = {}
    for name in METHODS:
        pol = factories[name]()
        tr = trace_episode(env, pol, seed=seed, f_phi=f_phi if name == "FlyAda" else None)
        for k, v in tr.items():
            blob[f"{name}/{k}"] = v
        # env.reset re-samples waypoints deterministically from the seed, so any
        # method's waypoints are the same; store once from the last rollout.
        print(f"  {tag:8s} {name:12s} len={int(tr['length']):4d} "
              f"wp_reached={int(tr['wp_idx'][-1])} success={bool(tr['success'])}")

    _obs, info = env.reset(seed=seed)
    blob["waypoints"] = np.stack(info["waypoints"]).astype(np.float32)
    blob["seed"] = np.asarray(seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"_traces_taskB_{tag}.npz"
    np.savez_compressed(out, **blob)
    print(f"saved -> {out}")
    return blob


def collect_taskA(tag: str, dyn_over: dict | None, seed: int, device: str = "cuda"):
    """Single-goal Task A traces — used for the belief-tracking figure."""
    factories, f_phi = build_policies(device)
    env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"),
                 dynamics_overrides=dyn_over)

    blob: Dict[str, np.ndarray] = {}
    for name in METHODS:
        pol = factories[name]()
        tr = trace_episode(env, pol, seed=seed, f_phi=f_phi if name == "FlyAda" else None)
        for k, v in tr.items():
            blob[f"{name}/{k}"] = v
        print(f"  {tag:8s} {name:12s} len={int(tr['length']):4d} success={bool(tr['success'])}")

    obs, info = env.reset(seed=seed)
    blob["goal"] = np.asarray(env._pos + obs[6:9], np.float32) if obs.shape[0] > 9 else np.zeros(3, np.float32)
    blob["seed"] = np.asarray(seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"_traces_taskA_{tag}.npz"
    np.savez_compressed(out, **blob)
    print(f"saved -> {out}")
    return blob


def collect_overshoot(seed: int = 900_011, device: str = "cuda"):
    """The same vanilla recipe with and without velocity in the observation.

    configs/env.yaml and configs/env_partial.yaml differ only in obs_mode, so the
    same seed gives the same start, goal and obstacles — the only difference the
    animation shows is what the policy could see.
    """
    full = load_diffusion_checkpoint(
        str(ROOT / "results/diffusion_v1/diffusion_policy.pt"), device=device)
    part = load_diffusion_checkpoint(str(VANILLA_CKPT), device=device)

    runs = {
        "Full obs": (full, "configs/env.yaml"),
        "Velocity hidden": (part, "configs/env_partial.yaml"),
    }
    blob: Dict[str, np.ndarray] = {}
    for name, (dp, cfg) in runs.items():
        env = UAVEnv(config=str(ROOT / cfg))
        tr = trace_episode(env, dp.make_rollout_policy(), seed=seed)
        for k, v in tr.items():
            blob[f"{name}/{k}"] = v
        obs, info = env.reset(seed=seed)
        blob[f"{name}/goal"] = np.asarray(env._pos + obs[8:11], np.float32)
        print(f"  {name:16s} len={int(tr['length']):4d} "
              f"final_d={float(tr['d_goal'][-2]):.2f} m success={bool(tr['success'])}")

    blob["seed"] = np.asarray(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "_traces_overshoot.npz"
    np.savez_compressed(out, **blob)
    print(f"saved -> {out}")


def collect_chunk_sequence(seed: int = 900_003, device: str = "cuda"):
    """Every action chunk a FlyAda rollout emits, with its forward-integrated path.

    Used by the receding-horizon animation: at each re-plan the policy commits to
    8 actions and executes only the first 4, so consecutive plans overlap.
    """
    import copy as _copy

    factories, f_phi = build_policies(device)
    env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"))
    pol = factories["FlyAda"]()
    obs, info = env.reset(seed=seed)
    pol.reset()

    traj = [env._pos.copy()]
    plans, plan_steps = [], []
    last_chunk_id = None

    for t in range(400):
        action, _ = pol.predict(obs, deterministic=True)
        # _pos is reset to 0 exactly on the step a fresh chunk was sampled.
        if pol._pos == 1 and id(pol._chunk) != last_chunk_id:
            last_chunk_id = id(pol._chunk)
            dyn = _copy.deepcopy(env.dynamics)
            p, v = env._pos.copy(), env._vel.copy()
            pts = [p.copy()]
            for u in np.asarray(pol._chunk, np.float32):
                p, v, _ = dyn.step(p, v, u)
                pts.append(p.copy())
            plans.append(np.asarray(pts, np.float32))
            plan_steps.append(t)
        obs, r, term, trunc, info = env.step(action)
        traj.append(env._pos.copy())
        if term or trunc:
            break

    goal = np.asarray(env._pos, np.float32) if info.get("success") else \
        np.asarray(traj[-1], np.float32)
    out = OUT_DIR / "_chunk_sequence.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, traj=np.asarray(traj, np.float32),
                        plans=np.stack(plans), plan_steps=np.asarray(plan_steps),
                        goal=goal)
    print(f"saved -> {out}  ({len(plans)} re-plans over {len(traj)} steps)")


def collect_chunk_demo(seed: int = 900_003, at_step: int = 18, device: str = "cuda"):
    """One rollout plus the raw 8-step action chunk the policy emits at `at_step`.

    The chunk is forward-integrated through the same dynamics the environment
    uses, so the plotted look-ahead is the policy's real plan, not a sketch.
    """
    import copy as _copy

    factories, f_phi = build_policies(device)
    env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"))
    pol = factories["FlyAda"]()

    obs, info = env.reset(seed=seed)
    pol.reset()
    traj, chunk, pred = [env._pos.copy()], None, None

    for t in range(400):
        action, _ = pol.predict(obs, deterministic=True)
        if t == at_step:
            chunk = np.asarray(pol._chunk, np.float32).copy()
            # Replay the whole chunk from the current state on a copy of the sim.
            dyn = _copy.deepcopy(env.dynamics)
            p, v = env._pos.copy(), env._vel.copy()
            pts = [p.copy()]
            for u in chunk:
                p, v, _ = dyn.step(p, v, u)
                pts.append(p.copy())
            pred = np.asarray(pts, np.float32)
        obs, r, term, trunc, info = env.step(action)
        traj.append(env._pos.copy())
        if term or trunc:
            break

    goal = env._pos.copy() if info.get("success") else None
    out = OUT_DIR / "_chunk_demo.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        traj=np.asarray(traj, np.float32),
        chunk=chunk, pred=pred,
        at_step=np.asarray(at_step),
        goal=goal if goal is not None else np.zeros(3, np.float32),
        obstacles=np.asarray(getattr(env, "_obstacle_spheres", []), np.float32),
    )
    print(f"saved -> {out}  (chunk {None if chunk is None else chunk.shape})")


def collect_probe_pairs(n_seeds: int = 25, warmup: int = 5, device: str = "cuda"):
    """(decoded velocity, true velocity) pairs across the latent-analysis sweep.

    Mirrors paper/_make_latent_analysis.py's CONDITIONS so the scatter on the
    slide and the R^2 reported in the paper come from the same distribution.
    """
    conditions = [
        ("nominal",   dict(mass=1.0, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=0)),
        ("mass+30%",  dict(mass=1.3, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=0)),
        ("drag+100%", dict(mass=1.0, drag=0.2, wind=[0.0, 0.0, 0.0], control_delay=0)),
        ("wind=1.0",  dict(mass=1.0, drag=0.1, wind=[1.0, 0.0, 0.0], control_delay=0)),
        ("delay=2",   dict(mass=1.0, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=2)),
    ]
    factories, f_phi = build_policies(device)

    vh, vt, zs, cond_id = [], [], [], []
    for ci, (name, dyn) in enumerate(conditions):
        env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"), dynamics_overrides=dyn)
        for i in range(n_seeds):
            tr = trace_episode(env, factories["FlyAda"](), seed=700_000 + i, f_phi=f_phi)
            vh.append(tr["vhat"][warmup:])
            vt.append(tr["vel"][warmup:])
            zs.append(tr["z"][warmup:])
            cond_id.append(np.full(len(tr["vel"]) - warmup, ci, np.int32))
        print(f"  {name:10s} done ({n_seeds} seeds)")

    out = OUT_DIR / "_probe_pairs.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vhat=np.concatenate(vh), vtrue=np.concatenate(vt),
        z=np.concatenate(zs),
        cond=np.concatenate(cond_id),
        cond_names=np.array([c[0] for c in conditions]),
    )
    print(f"saved -> {out}  ({len(np.concatenate(cond_id))} pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--taskB-seed", type=int, default=600_000)
    ap.add_argument("--taskA-seed", type=int, default=900_003)
    args = ap.parse_args()

    print("[traces] Task B, nominal partial-obs")
    collect_taskB("nominal", None, args.taskB_seed, args.device)
    print("[traces] Task B, hard combined condition")
    collect_taskB("hard", HARD_DYN, args.taskB_seed, args.device)
    print("[traces] Task A, hard combined condition (belief tracking)")
    collect_taskA("hard", HARD_DYN, args.taskA_seed, args.device)


if __name__ == "__main__":
    main()
