"""Train PPO teacher on Task A nominal dynamics (Stage 1)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Allow `python trainers/train_ppo.py` as well as `python -m trainers.train_ppo`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from envs import UAVEnv
from trainers.utils import TSVLogger, results_dir, set_global_seed

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def make_env_fn(env_config_path: str, seed: int):
    def _thunk():
        env = UAVEnv(config=env_config_path)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _thunk


class SuccessCallback(BaseCallback):
    """Logs rolling success rate from Monitor info dicts."""

    def __init__(self, logger: TSVLogger, log_every: int = 2048):
        super().__init__()
        self._logger = logger
        self._log_every = log_every
        self._successes: list[int] = []
        self._ep_returns: list[float] = []
        self._last_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._ep_returns.append(float(ep["r"]))
                self._successes.append(1 if info.get("success") else 0)
        if self.num_timesteps - self._last_log >= self._log_every and self._ep_returns:
            window = self._ep_returns[-100:]
            sr = float(np.mean(self._successes[-100:])) if self._successes else 0.0
            self._logger.log({
                "timesteps": int(self.num_timesteps),
                "ep_return_mean": float(np.mean(window)),
                "ep_return_std": float(np.std(window)),
                "success_rate_100": sr,
                "episodes": len(self._ep_returns),
            })
            self._last_log = self.num_timesteps
        return True


def build_vec_env(env_config_path: str, n_envs: int, seed: int):
    fns = [make_env_fn(env_config_path, seed + i) for i in range(n_envs)]
    # SubprocVecEnv avoids GIL contention but has Windows-spawn cost; DummyVecEnv is simpler
    # and the env is fast enough (no heavy numpy). Use DummyVecEnv for Stage-1 smoke.
    return DummyVecEnv(fns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_ppo.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override config's total_timesteps (useful for smoke tests).")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or cfg["run_name"]
    total_timesteps = int(args.total_timesteps or cfg["total_timesteps"])
    seed = int(cfg["seed"])
    set_global_seed(seed)

    env_cfg_path = cfg["env_config"]
    if not Path(env_cfg_path).is_absolute():
        env_cfg_path = str((ROOT / env_cfg_path).resolve())

    out_dir = results_dir(run_name, root=ROOT / "results")
    log_path = out_dir / "train_log.tsv"
    logger = TSVLogger(
        path=log_path,
        columns=["timesteps", "ep_return_mean", "ep_return_std", "success_rate_100", "episodes"],
    )

    n_envs = int(cfg["n_envs"])
    vec_env = build_vec_env(env_cfg_path, n_envs, seed)

    # Eval env — single env wrapped in DummyVecEnv for SB3 compatibility
    eval_env = DummyVecEnv([make_env_fn(env_cfg_path, seed + 10_000)])

    ppo_cfg = cfg["ppo"]
    policy_kwargs = dict(
        net_arch=list(ppo_cfg["policy_kwargs"]["net_arch"]),
        activation_fn=ACTIVATIONS[str(ppo_cfg["policy_kwargs"]["activation_fn"]).lower()],
    )

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_ppo] CUDA unavailable, falling back to CPU.")
        device = "cpu"
    torch.set_float32_matmul_precision("high")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
        seed=seed,
    )

    cb_success = SuccessCallback(logger=logger, log_every=max(1, n_envs * int(ppo_cfg["n_steps"]) // 2))
    eval_cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=int(cfg["eval"]["n_episodes"]),
        eval_freq=max(1, int(cfg["eval"]["eval_freq"]) // n_envs),
        best_model_save_path=str(out_dir),
        log_path=str(out_dir / "sb3_eval"),
        deterministic=True,
        render=False,
        verbose=0,
    )

    t0 = time.time()
    print(f"[train_ppo] run={run_name} device={device} n_envs={n_envs} "
          f"total_timesteps={total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=[cb_success, eval_cb], progress_bar=False)
    elapsed = time.time() - t0

    model.save(str(out_dir / "ppo_teacher.zip"))

    # Rollout a quick final-eval pass to write a summary json
    rewards, successes, lens = [], [], []
    obs = eval_env.reset()
    for _ in range(30):
        obs = eval_env.reset()
        done = [False]
        ep_r, ep_len, succ = 0.0, 0, False
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = eval_env.step(action)
            ep_r += float(r[0])
            ep_len += 1
            if info[0].get("success"):
                succ = True
        rewards.append(ep_r); successes.append(int(succ)); lens.append(ep_len)

    summary = {
        "run_name": run_name,
        "total_timesteps": total_timesteps,
        "wall_time_sec": elapsed,
        "final_eval_episodes": len(rewards),
        "final_eval_return_mean": float(np.mean(rewards)),
        "final_eval_return_std": float(np.std(rewards)),
        "final_eval_success_rate": float(np.mean(successes)),
        "final_eval_length_mean": float(np.mean(lens)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot training curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv

        ts, ret, sr = [], [], []
        with open(log_path, "r", newline="") as f:
            r = _csv.DictReader(f, delimiter="\t")
            for row in r:
                if not row.get("timesteps"):
                    continue
                ts.append(int(row["timesteps"]))
                ret.append(float(row["ep_return_mean"]))
                sr.append(float(row["success_rate_100"]))
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].plot(ts, ret); ax[0].set_xlabel("env steps"); ax[0].set_ylabel("ep return mean")
        ax[0].set_title("Return (rolling 100 eps)")
        ax[1].plot(ts, sr); ax[1].set_xlabel("env steps"); ax[1].set_ylabel("success rate")
        ax[1].set_title("Success (rolling 100 eps)"); ax[1].set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(out_dir / "train_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[train_ppo] plotting skipped: {e}")

    print(f"[train_ppo] done in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
