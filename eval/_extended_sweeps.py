"""One-off: re-run partial-obs evals at extrapolation conditions so the headline
tables don't look like 0.000-vs-1.000 gimmicks. Saves merged JSON for the paper."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from eval.rollout import aggregate, rollout_episode
from eval.eval_longhorizon import rollout_taskB, aggregate_taskB
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


N_SEEDS = 20  # smaller than 30 so the extrapolation sweep stays reasonable

# Extrapolation conditions for the partial-obs Task A mismatch table.
# These extend beyond the FlyAda training range (mass [0.9,1.3], drag [0.05,0.25],
# wind [0,1], delay [0,3]) so FlyAda is no longer trivially 1.0 everywhere.
EXTRAP_CONDS = [
    {"id": "mass+50",  "axis": "mass",  "level": 0.50, "mass": 1.5, "drag": 0.1, "wind": [0,0,0], "control_delay": 0},
    {"id": "mass+100", "axis": "mass",  "level": 1.00, "mass": 2.0, "drag": 0.1, "wind": [0,0,0], "control_delay": 0},
    {"id": "drag+200", "axis": "drag",  "level": 2.00, "mass": 1.0, "drag": 0.3, "wind": [0,0,0], "control_delay": 0},
    {"id": "wind=2.0", "axis": "wind",  "level": 2.0,  "mass": 1.0, "drag": 0.1, "wind": [2.0,0,0], "control_delay": 0},
    {"id": "wind=3.0", "axis": "wind",  "level": 3.0,  "mass": 1.0, "drag": 0.1, "wind": [3.0,0,0], "control_delay": 0},
    {"id": "delay=5",  "axis": "delay", "level": 5.0,  "mass": 1.0, "drag": 0.1, "wind": [0,0,0], "control_delay": 5},
]

HARDER_COND = dict(mass=1.5, drag=0.4, wind=[2.0, 0.0, 0.0], control_delay=4)


def make_factories(device: str = "cuda"):
    fs_cfg = load_flyada_checkpoint(str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device)
    fl_dp, f_phi, fcfg = fs_cfg
    fly_alpha = float(fcfg.get("alpha", 0.1)); fly_mode = str(fcfg.get("update_mode", "ema"))

    van = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"), device=device)
    fst = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_frame3_v1/diffusion_policy.pt"), device=device)

    return {
        "vanilla":     lambda: van.make_rollout_policy(),
        "frame_stack": lambda: fst.make_rollout_policy(),
        "flyada":      lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=fly_alpha, update_mode=fly_mode),
    }


def run_taskA_extrap(factories: Dict[str, Callable]) -> Dict:
    out: Dict[str, Dict] = {}
    for cond in EXTRAP_CONDS:
        dyn = {"mass": cond["mass"], "drag": cond["drag"], "wind": cond["wind"], "control_delay": cond["control_delay"]}
        env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"), dynamics_overrides=dyn)
        out[cond["id"]] = {"axis": cond["axis"], "level": cond["level"]}
        for name, fac in factories.items():
            results = [rollout_episode(env, fac(), seed=820_000 + i) for i in range(N_SEEDS)]
            m = aggregate(results)
            out[cond["id"]][name] = {
                "success": m["success_rate"],
                "final_d": m["final_dist_mean"],
                "ep_length": m["ep_length_mean"],
            }
            print(f"  {cond['id']:<12s} {name:<12s} succ={m['success_rate']:.3f}  final_d={m['final_dist_mean']:.2f}  len={m['ep_length_mean']:.0f}",
                  flush=True)
    return out


def run_ablation_harder(device: str = "cuda") -> Dict:
    from eval.eval_ablation import HARD_COND, run_variant   # we'll override HARD_COND
    from envs import UAVEnv as _UAVEnv

    van = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_v1/diffusion_policy.pt"), device=device)
    fst = load_diffusion_checkpoint(str(ROOT / "results/diffusion_partial_frame3_v1/diffusion_policy.pt"), device=device)
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device)
    fly_alpha = float(fcfg.get("alpha", 0.1)); fly_mode = str(fcfg.get("update_mode", "ema"))

    env_cfg = str(ROOT / "configs/env_partial.yaml")
    cond = HARDER_COND

    variants = {
        "vanilla":            lambda: van.make_rollout_policy(),
        "frame_stack":        lambda: fst.make_rollout_policy(),
        "flyada_alpha0":      lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=0.0, update_mode=fly_mode),
        "flyada_frozen10":    lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=fly_alpha,
                                                          update_mode=fly_mode, freeze_after_steps=10),
        "flyada_full":        lambda: FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=fly_alpha, update_mode=fly_mode),
    }
    out = {"condition": cond, "n_seeds": 40, "variants": {}}
    for name, factory in variants.items():
        m, _ = run_variant(name, factory, env_cfg, n_seeds=40, base_seed=820_000, cond=cond)
        out["variants"][name] = {k: float(v) if isinstance(v, (int, float)) else v
                                 for k, v in m.items() if k not in ("name",)}
        print(f"  {name:<22s}  succ={m['success_rate']:.3f}  final_d={m['final_dist_mean']:.2f}  len={m['ep_length_mean']:.0f}",
              flush=True)
    return out


def run_taskB_extreme(factories: Dict[str, Callable]) -> Dict:
    extreme = dict(mass=1.5, drag=0.4, wind=[2.0, 0.0, 0.0], control_delay=4)
    env = UAVEnv(config=str(ROOT / "configs/env_partial_taskB.yaml"), dynamics_overrides=extreme)
    out: Dict[str, Dict] = {"condition": extreme, "n_seeds": 30}
    for name, fac in factories.items():
        eps = []
        for i in range(30):
            eps.append(rollout_taskB(env, fac(), seed=600_000 + i))
        m = aggregate_taskB(eps)
        out[name] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
        print(f"  TaskB-extreme  {name:<12s}  task_succ={m['task_success_rate']:.3f}  "
              f"wp_frac={m['waypoint_completion_mean']:.3f}  len={m['ep_length_mean']:.0f}",
              flush=True)
    return out


def main():
    factories = make_factories(device="cuda")
    print("\n=== EXTRAP MISMATCH SWEEP (n=20) ===", flush=True)
    extrap = run_taskA_extrap(factories)
    print("\n=== ABLATION ON HARDER COND (n=40) ===", flush=True)
    abl = run_ablation_harder(device="cuda")
    print("\n=== TASK B EXTREME (n=30) ===", flush=True)
    taskb = run_taskB_extreme(factories)

    out_dir = ROOT / "results" / "extended_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"extrap_mismatch": extrap, "ablation_harder": abl, "taskB_extreme": taskb}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[extended] saved -> {out_dir/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
