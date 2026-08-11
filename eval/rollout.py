"""Shared evaluation helpers: rollout a stateful policy on the UAV env, aggregate metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    collided: bool
    final_dist: float
    final_vel: float
    length: int
    total_return: float
    trajectory: np.ndarray
    goal: np.ndarray
    actions: np.ndarray
    states: np.ndarray


def rollout_episode(env, policy, seed: int, deterministic: bool = True) -> EpisodeResult:
    """Policy duck-types either an SB3 model (predict(obs) -> (action, state))
    or a custom wrapper with .predict() and .reset().
    """
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    goal = info["goal"].astype(np.float32)

    traj: List[np.ndarray] = [obs[:3].copy()]
    states: List[np.ndarray] = [obs.copy()]
    acts: List[np.ndarray] = []
    ep_r = 0.0
    steps = 0
    collided = False
    success = False

    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        acts.append(np.asarray(action, dtype=np.float32))
        obs, r, term, trunc, info = env.step(action)
        states.append(obs.copy())
        traj.append(obs[:3].copy())
        ep_r += float(r)
        steps += 1
        if info.get("collided"):
            collided = True
        if info.get("success"):
            success = True
        if term or trunc:
            break

    return EpisodeResult(
        seed=seed,
        success=bool(success),
        collided=bool(collided),
        final_dist=float(np.linalg.norm(obs[:3] - goal)),
        final_vel=float(np.linalg.norm(obs[3:6])),
        length=steps,
        total_return=ep_r,
        trajectory=np.stack(traj),
        goal=goal,
        actions=np.stack(acts) if acts else np.zeros((0, env.action_space.shape[0]), dtype=np.float32),
        states=np.stack(states),
    )


def aggregate(results: List[EpisodeResult]) -> Dict[str, float]:
    if not results:
        return {}
    succ = np.array([r.success for r in results])
    coll = np.array([r.collided for r in results])
    fd = np.array([r.final_dist for r in results])
    fv = np.array([r.final_vel for r in results])
    ln = np.array([r.length for r in results])
    ret = np.array([r.total_return for r in results])

    out = {
        "n": int(len(results)),
        "success_rate": float(succ.mean()),
        "collision_rate": float(coll.mean()),
        "final_dist_mean": float(fd.mean()),
        "final_dist_median": float(np.median(fd)),
        "final_dist_p90": float(np.percentile(fd, 90)),
        "final_vel_mean": float(fv.mean()),
        "ep_length_mean": float(ln.mean()),
        "return_mean": float(ret.mean()),
    }
    # Control smoothness (E[|Δu|^2]) for successful episodes
    smooths = []
    for r in results:
        if len(r.actions) >= 2:
            d = np.diff(r.actions, axis=0)
            smooths.append(float(np.mean(np.sum(d * d, axis=-1))))
    out["delta_u2_mean"] = float(np.mean(smooths)) if smooths else -1.0
    return out
