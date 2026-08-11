"""Latent analysis for FlyAda's $z_t$.

Rolls the partial-obs FlyAda policy on a small sweep of dynamics conditions,
records (z_t, true velocity, condition) at every env step, and produces:

  - A linear probe (z -> v) R^2 score, fitted on a held-out split, reported per
    axis and overall.
  - A 2-panel PCA scatter of the latent (z reduced to 2D) coloured by ||v|| on
    the left, by perturbation type on the right.
  - A temporal smoothness number: mean ||z_{t+1} - z_t|| in steady state.

Output: paper/figures/latent_analysis.png and a small JSON with the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper._plot_style import set_ieee_font
set_ieee_font()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import UAVEnv
from models.checkpoint import load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy


CONDITIONS: List[Tuple[str, dict]] = [
    ("nominal",   dict(mass=1.0, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=0)),
    ("mass+30%",  dict(mass=1.3, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=0)),
    ("drag+100%", dict(mass=1.0, drag=0.2, wind=[0.0, 0.0, 0.0], control_delay=0)),
    ("wind=1.0",  dict(mass=1.0, drag=0.1, wind=[1.0, 0.0, 0.0], control_delay=0)),
    ("delay=2",   dict(mass=1.0, drag=0.1, wind=[0.0, 0.0, 0.0], control_delay=2)),
]

N_SEEDS = 25       # per condition
WARMUP_STEPS = 5   # drop the first few steps where z is still ramping up from 0


def collect():
    device = "cuda"
    fl_dp, f_phi, fcfg = load_flyada_checkpoint(
        str(ROOT / "results/flyada_partial_v1/flyada_policy.pt"), device=device
    )
    alpha = float(fcfg.get("alpha", 0.1))
    mode = str(fcfg.get("update_mode", "ema"))

    all_z, all_v, all_cond, all_t = [], [], [], []
    z_diffs_by_cond: Dict[int, List[float]] = {i: [] for i in range(len(CONDITIONS))}

    for ci, (name, dyn) in enumerate(CONDITIONS):
        env = UAVEnv(config=str(ROOT / "configs/env_partial.yaml"), dynamics_overrides=dyn)
        print(f"  [{name}] rolling {N_SEEDS} eps", flush=True)
        for seed in range(700_000, 700_000 + N_SEEDS):
            pol = FlyAdaRolloutPolicy(fl_dp, f_phi, alpha=alpha, update_mode=mode)
            obs, info = env.reset(seed=seed)
            prev_z = None
            t = 0
            while True:
                action, _ = pol.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                t += 1
                if t < WARMUP_STEPS:
                    if term or trunc:
                        break
                    continue
                z = pol._z.detach().cpu().numpy().reshape(-1).astype(np.float32)
                v_true = env._vel.copy().astype(np.float32)
                all_z.append(z); all_v.append(v_true)
                all_cond.append(ci); all_t.append(t)
                if prev_z is not None:
                    z_diffs_by_cond[ci].append(float(np.linalg.norm(z - prev_z)))
                prev_z = z
                if term or trunc:
                    break

    all_z = np.stack(all_z); all_v = np.stack(all_v)
    all_cond = np.asarray(all_cond, dtype=np.int32)
    return all_z, all_v, all_cond, z_diffs_by_cond


def linear_probe(z: np.ndarray, v: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    """Solve v ≈ z @ W + b on a train split; report R^2 on the test split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    n_test = int(len(z) * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    Z_tr, V_tr = z[train_idx], v[train_idx]
    Z_te, V_te = z[test_idx], v[test_idx]

    Z_tr_aug = np.concatenate([Z_tr, np.ones((Z_tr.shape[0], 1))], axis=1)
    Z_te_aug = np.concatenate([Z_te, np.ones((Z_te.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(Z_tr_aug, V_tr, rcond=None)

    V_pred = Z_te_aug @ W
    ss_res = np.sum((V_te - V_pred) ** 2, axis=0)
    ss_tot = np.sum((V_te - V_te.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_axis = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2_overall = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
    return float(r2_overall), r2_per_axis.tolist()


def pca_2d(z: np.ndarray):
    zc = z - z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(zc, full_matrices=False)
    z_2d = (zc @ Vt[:2].T)
    explained = (S ** 2) / (S ** 2).sum()
    return z_2d, explained[:2]


def _cov_ellipse_xy(points: np.ndarray, n_std: float = 1.0):
    """Return parametric (x, y) of an n_std covariance ellipse around `points`."""
    if len(points) < 3:
        return points[:, 0], points[:, 1]
    mu = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    a, b = n_std * np.sqrt(eigvals[0]), n_std * np.sqrt(eigvals[1])
    t = np.linspace(0, 2 * np.pi, 80)
    x = a * np.cos(t); y = b * np.sin(t)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    xy = R @ np.stack([x, y])
    return mu[0] + xy[0], mu[1] + xy[1]


def condition_classifier_accuracy(z: np.ndarray, cond: np.ndarray, seed: int = 0):
    """Linear (multinomial logistic, OLS-on-onehot) classifier accuracy on a 70/30 split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    n_test = int(len(z) * 0.3)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    n_classes = int(cond.max()) + 1
    Y_tr = np.eye(n_classes)[cond[train_idx]]
    Z_tr_aug = np.concatenate([z[train_idx], np.ones((len(train_idx), 1))], axis=1)
    Z_te_aug = np.concatenate([z[test_idx], np.ones((len(test_idx), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Z_tr_aug, Y_tr, rcond=None)
    pred = (Z_te_aug @ W).argmax(axis=1)
    acc = float((pred == cond[test_idx]).mean())
    return acc, 1.0 / n_classes


def render(z_2d, all_v, all_cond, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # Left: coloured by ||v||
    v_mag = np.linalg.norm(all_v, axis=1)
    sc = ax[0].scatter(z_2d[:, 0], z_2d[:, 1], c=v_mag, cmap="viridis",
                       s=6, alpha=0.6, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax[0], pad=0.02)
    cbar.set_label(r"$\Vert v \Vert$ (m/s)")
    ax[0].set_xlabel("PC 1"); ax[0].set_ylabel("PC 2")
    ax[0].set_title(r"$z_t$ scatter, coloured by $\Vert v \Vert$")
    ax[0].grid(True, alpha=0.25)

    # Right: per-condition centroid + 1-sigma ellipse, with light scatter behind
    from paper._plot_style import CONDITION_COLORS
    palette = CONDITION_COLORS
    ax[1].scatter(z_2d[:, 0], z_2d[:, 1], color="0.85", s=4, alpha=0.4, linewidths=0)
    for ci, (name, _) in enumerate(CONDITIONS):
        m = (all_cond == ci)
        pts = z_2d[m]
        mu = pts.mean(axis=0)
        ex, ey = _cov_ellipse_xy(pts, n_std=1.0)
        c = palette[ci % len(palette)]
        ax[1].plot(ex, ey, color=c, linewidth=1.8, label=name)
        ax[1].scatter(mu[0], mu[1], color=c, s=70, edgecolors="white",
                      linewidths=1.0, zorder=10)
    ax[1].set_xlabel("PC 1"); ax[1].set_ylabel("PC 2")
    ax[1].set_title("per-condition centroid $\pm 1\\sigma$")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="best", frameon=True, facecolor="white", framealpha=0.85)
    # Match left axis range so the comparison is fair.
    ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(ax[0].get_ylim())

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    z, v, cond, smoothness = collect()
    print(f"collected {z.shape[0]} (z,v) pairs across {len(CONDITIONS)} conditions",
          flush=True)

    r2_overall, r2_axes = linear_probe(z, v)
    print(f"linear probe (z -> v): R^2 = {r2_overall:.3f}  per-axis = "
          f"({r2_axes[0]:.3f}, {r2_axes[1]:.3f}, {r2_axes[2]:.3f})", flush=True)

    cond_acc, chance = condition_classifier_accuracy(z, cond)
    print(f"linear classifier (z -> condition): test accuracy {cond_acc:.3f} (chance {chance:.3f})",
          flush=True)

    z_2d, expl = pca_2d(z)
    print(f"PCA explained variance ratio: {expl[0]:.2f}, {expl[1]:.2f} "
          f"(sum {sum(expl):.2f})", flush=True)

    smoothness_per_cond = {CONDITIONS[ci][0]: float(np.mean(d)) for ci, d in smoothness.items()}
    smooth_overall = float(np.mean([v for vals in smoothness.values() for v in vals]))
    print(f"mean ||z_{{t+1}} - z_t||: overall {smooth_overall:.3f}; "
          f"per-condition {smoothness_per_cond}", flush=True)

    out = ROOT / "paper/figures/latent_analysis.png"
    render(z_2d, v, cond, out)
    print(f"saved -> {out}", flush=True)

    summary = {
        "n_pairs": int(z.shape[0]),
        "linear_probe_R2_overall": r2_overall,
        "linear_probe_R2_per_axis": r2_axes,
        "condition_classifier_acc": cond_acc,
        "condition_classifier_chance": chance,
        "pca_explained_var": [float(e) for e in expl],
        "smoothness_per_cond": smoothness_per_cond,
        "smoothness_overall": smooth_overall,
    }
    with open(ROOT / "results/latent_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
