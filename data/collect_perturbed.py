"""Collect demos under random dynamics perturbations using the PPO teacher.

Each episode samples a perturbation (mass, drag, wind, delay) from ranges in
configs/train_flyada.yaml::perturb, then rolls out the teacher. The teacher was
trained on nominal dynamics so its actions under perturbation are "best effort"
rather than optimal — but they're still (state, action) pairs consistent with
each perturbed trajectory, which is what FlyAda needs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import UAVEnv
from stable_baselines3 import PPO


def sample_perturb(rng: np.random.Generator, cfg):
    mass = float(rng.uniform(*cfg["mass_range"]))
    drag = float(rng.uniform(*cfg["drag_range"]))
    wmag = float(rng.uniform(*cfg["wind_mag_range"]))
    wdir = rng.standard_normal(3).astype(np.float32)
    wdir /= max(np.linalg.norm(wdir), 1e-6)
    wind = (wmag * wdir).tolist()
    d_lo, d_hi = cfg["delay_steps"]
    delay = int(rng.integers(d_lo, d_hi + 1))
    return {"mass": mass, "drag": drag, "wind": wind, "control_delay": delay}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flyada-config", type=str, default="configs/train_flyada.yaml")
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--teacher", type=str, default="results/ppo_teacher_v2/ppo_teacher.zip")
    parser.add_argument("--out", type=str, default="results/ppo_teacher_v2/demos_perturbed.npz")
    parser.add_argument("--n-transitions", type=int, default=50000)
    parser.add_argument("--min-episodes", type=int, default=400)
    parser.add_argument("--nominal-frac", type=float, default=0.2,
                        help="Fraction of episodes collected under exactly nominal dynamics.")
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--mask-velocity", action="store_true",
                        help="Zero velocity dims [3:6] in saved observations.")
    args = parser.parse_args()

    with open(ROOT / args.flyada_config, "r") as f:
        flyada_cfg = yaml.safe_load(f)
    perturb_cfg = flyada_cfg["perturb"]

    env_cfg_path = args.env_config
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())
    teacher_path = args.teacher
    if not Path(teacher_path).is_absolute():
        teacher_path = str((ROOT / teacher_path).resolve())
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[collect_perturbed] teacher={teacher_path}")
    print(f"[collect_perturbed] perturb ranges: {perturb_cfg}")
    model = PPO.load(teacher_path, device="cpu")
    rng = np.random.default_rng(args.seed)

    s_list, g_list, u_list, s2_list, done_list, succ_list = [], [], [], [], [], []
    ep_perturb_id: list[int] = []
    ep_mass, ep_drag, ep_wind_mag, ep_delay = [], [], [], []

    episodes_done = 0
    total = 0
    t0 = time.time()

    while total < args.n_transitions or episodes_done < args.min_episodes:
        if rng.random() < args.nominal_frac:
            perturb = {"mass": 1.0, "drag": 0.1, "wind": [0.0, 0.0, 0.0], "control_delay": 0}
        else:
            perturb = sample_perturb(rng, perturb_cfg)

        env = UAVEnv(config=env_cfg_path, dynamics_overrides=perturb)
        obs, info = env.reset(seed=args.seed + episodes_done + 1)
        goal = info["goal"].astype(np.float32)
        ep_perturb_id.append(episodes_done)
        ep_mass.append(perturb["mass"])
        ep_drag.append(perturb["drag"])
        ep_wind_mag.append(float(np.linalg.norm(perturb["wind"])))
        ep_delay.append(int(perturb["control_delay"]))

        while True:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            if args.mask_velocity:
                s_save = obs.copy(); s_save[3:6] = 0.0
                s2_save = next_obs.copy(); s2_save[3:6] = 0.0
            else:
                s_save, s2_save = obs, next_obs
            s_list.append(s_save.astype(np.float32))
            g_list.append(goal.copy())
            u_list.append(np.asarray(action, dtype=np.float32))
            s2_list.append(s2_save.astype(np.float32))
            done_list.append(bool(terminated or truncated))
            succ_list.append(bool(step_info.get("success", False)))
            obs = next_obs
            total += 1
            if terminated or truncated:
                break

        episodes_done += 1
        if episodes_done % 50 == 0:
            print(f"[collect_perturbed] episodes={episodes_done} transitions={total} "
                  f"elapsed={time.time()-t0:.1f}s")

    s = np.stack(s_list); g = np.stack(g_list); u = np.stack(u_list)
    s2 = np.stack(s2_list); dones = np.asarray(done_list, dtype=bool)
    succ = np.asarray(succ_list, dtype=bool)

    ep_success_any = []
    start = 0
    for i, d in enumerate(dones):
        if d:
            ep_success_any.append(bool(succ[start:i + 1].any()))
            start = i + 1
    ep_success_any = np.asarray(ep_success_any, dtype=bool)

    np.savez_compressed(
        out_path,
        s=s, g=g, u=u, s2=s2, done=dones, success=succ, episode_success=ep_success_any,
        ep_mass=np.asarray(ep_mass, dtype=np.float32),
        ep_drag=np.asarray(ep_drag, dtype=np.float32),
        ep_wind_mag=np.asarray(ep_wind_mag, dtype=np.float32),
        ep_delay=np.asarray(ep_delay, dtype=np.int32),
    )
    print(f"[collect_perturbed] saved {total} transitions across {episodes_done} episodes -> {out_path}")
    print(f"[collect_perturbed] teacher episode-success under perturbation: {ep_success_any.mean():.3f}")


if __name__ == "__main__":
    main()
