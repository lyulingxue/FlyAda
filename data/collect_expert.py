"""Roll out a trained PPO teacher and save demos (Stage 1)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_ppo.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override path to PPO .zip (else results/<run>/ppo_teacher.zip).")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n-transitions", type=int, default=None)
    parser.add_argument("--mask-velocity", action="store_true",
                        help="Zero velocity dims [3:6] in saved observations (for partial-obs policies). "
                             "Teacher still sees full obs during action selection.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or cfg["run_name"]
    out_dir = ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else out_dir / "ppo_teacher.zip"

    demo_cfg = cfg.get("demo_collection", {})
    n_transitions = int(args.n_transitions or demo_cfg.get("n_transitions", 50000))
    min_episodes = int(demo_cfg.get("min_episodes", 200))
    deterministic = bool(demo_cfg.get("deterministic", True))
    out_path = Path(args.out or (out_dir / demo_cfg.get("output", "demos.npz")))

    env_cfg_path = cfg["env_config"]
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    print(f"[collect_expert] loading model: {model_path}")
    model = PPO.load(str(model_path), device="cpu")   # CPU is fine for rollouts
    env = UAVEnv(config=env_cfg_path)

    s_list, g_list, u_list, s2_list, done_list, succ_list = [], [], [], [], [], []
    episodes_done = 0
    total = 0
    t0 = time.time()
    seed = 12345

    while total < n_transitions or episodes_done < min_episodes:
        obs, info = env.reset(seed=seed + episodes_done)
        goal = info["goal"].astype(np.float32)
        ep_len = 0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
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
            succ_list.append(bool(info.get("success", False)))
            obs = next_obs
            ep_len += 1
            total += 1
            if terminated or truncated:
                break
        episodes_done += 1
        if episodes_done % 50 == 0:
            print(f"[collect_expert] episodes={episodes_done} transitions={total} "
                  f"elapsed={time.time()-t0:.1f}s")

    s = np.stack(s_list); g = np.stack(g_list); u = np.stack(u_list)
    s2 = np.stack(s2_list); dones = np.asarray(done_list, dtype=bool)
    succ = np.asarray(succ_list, dtype=bool)

    # episode-level success: any True within each episode-span defined by done flags
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
    )
    print(f"[collect_expert] saved {total} transitions across {episodes_done} episodes -> {out_path}")
    print(f"[collect_expert] episode success rate (teacher): {ep_success_any.mean():.3f}")


if __name__ == "__main__":
    main()
