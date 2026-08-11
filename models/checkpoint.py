"""Load trained diffusion / FlyAda checkpoints into a usable rollout policy."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from models.adaptation_head import AdaptationHead
from models.diffusion_policy import DiffusionConfig, DiffusionPolicy
from models.temporal_unet import TemporalUNet


def load_diffusion_checkpoint(ckpt_path: str | Path, device: str = "cuda") -> DiffusionPolicy:
    blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    c = blob["config"]
    frame_stack = int(c.get("frame_stack", 1))
    denoiser = TemporalUNet(
        action_dim=c["action_dim"],
        cond_input_dim=c["state_dim"],
        latent_dim=c.get("latent_dim", 0),
        hidden=c["hidden"],
    )
    denoiser.load_state_dict(blob["model_state"])
    denoiser.to(device).eval()
    dcfg = DiffusionConfig(
        action_dim=c["action_dim"],
        horizon=c["horizon"],
        cond_dim=c["state_dim"],
        latent_dim=c.get("latent_dim", 0),
        hidden=c["hidden"],
        T_train=c["T_train"],
        T_sample=c["T_sample"],
        exec_k=c["exec_k"],
        frame_stack=frame_stack,
    )
    policy = DiffusionPolicy(
        denoiser, dcfg,
        state_mean=blob["state_mean"], state_std=blob["state_std"],
        device=device,
    )
    return policy


def load_flyada_checkpoint(ckpt_path: str | Path, device: str = "cuda"):
    blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    c = blob["config"]
    denoiser = TemporalUNet(
        action_dim=c["action_dim"],
        cond_input_dim=c["state_dim"],
        latent_dim=c["latent_dim"],
        hidden=c["hidden"],
    )
    denoiser.load_state_dict(blob["model_state"])
    denoiser.to(device).eval()

    f_phi = AdaptationHead(
        state_dim=c["state_dim"],
        action_dim=c["action_dim"],
        latent_dim=c["latent_dim"],
        hidden=c["adaptation_hidden"],
    )
    f_phi.load_state_dict(blob["f_phi_state"])
    f_phi.to(device).eval()

    dcfg = DiffusionConfig(
        action_dim=c["action_dim"],
        horizon=c["horizon"],
        cond_dim=c["state_dim"],
        latent_dim=c["latent_dim"],
        hidden=c["hidden"],
        T_train=c["T_train"],
        T_sample=c["T_sample"],
        exec_k=c["exec_k"],
    )
    policy = DiffusionPolicy(
        denoiser, dcfg,
        state_mean=blob["state_mean"], state_std=blob["state_std"],
        device=device,
    )
    # Pass update_mode back so callers can construct FlyAdaRolloutPolicy with the
    # matching rule. Legacy checkpoints (no update_mode key) used additive accum.
    c = dict(c)
    c.setdefault("update_mode", "add")
    return policy, f_phi, c
