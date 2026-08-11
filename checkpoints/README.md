# Trained policies

Four checkpoints — every video and figure in `media/` was rendered from these.

| Directory | Policy | Observation | Used for |
|---|---|---|---|
| `flyada_partial/` | FlyAda | velocity hidden | the method, every FlyAda trace |
| `vanilla_partial/` | vanilla diffusion | velocity hidden | the failing baseline |
| `framestack_partial/` | 3-frame stack | velocity hidden | the "more history" baseline |
| `vanilla_full_obs/` | vanilla diffusion | full state | the blue trace in `overshoot` |

Each file holds the model weights plus the config needed to rebuild the module,
so no separate YAML is required:

```python
from models.checkpoint import load_diffusion_checkpoint, load_flyada_checkpoint
from trainers.train_flyada import FlyAdaRolloutPolicy

dp, f_phi, cfg = load_flyada_checkpoint(
    "checkpoints/flyada_partial/flyada_policy.pt", device="cuda")
policy = FlyAdaRolloutPolicy(dp, f_phi, alpha=cfg["alpha"],
                             update_mode=cfg["update_mode"])

obs, info = env.reset(seed=0)
action, _ = policy.predict(obs, deterministic=True)
```

`policy.reset()` at the start of every episode — FlyAda carries state (`z_t`)
across steps, and forgetting to reset it leaks the previous episode's belief
into the next one.
