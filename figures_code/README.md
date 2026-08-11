# How everything in `media/` was made

These are the generators, copied verbatim from the research tree. Their rendered
outputs are already in `media/`, so you never need to run them — they are here so
every figure and animation in the talk is traceable to code rather than to a
screenshot.

| Script | Produces |
|---|---|
| `_slide_rollouts.py` | the cached rollouts in `media/traces/` — records the *hidden* state (true velocity, latent `z_t`, decoded velocity) alongside the observable one |
| `_slide_quadrotor.py` | the 3D quadrotor renderer: arms, rotor discs, spinning blades, attitude derived from the commanded acceleration |
| `_make_slide_anims.py` | `overshoot`, `taskB_race_nominal`, `belief_tracking`, `chunk_replan` |
| `_make_mujoco_video.py` | `mujoco_flight` — a real MuJoCo render, not a plot |
| `_make_slide_figs.py` | every PNG in `media/figures/` |
| `_make_slide_deck.py`, `_slide_script.py` | the slides and the narration |

## Running them

They expect the layout of the full research repo, not this artefact bundle:

- inputs are read from `results/` (present here) and `paper/figures/slides/`
  (here it is `media/traces/` — symlink or copy it across);
- outputs are written to `paper/figures/slides/`;
- `_slide_rollouts.py` and `_make_mujoco_video.py` need the checkpoints in
  `checkpoints/` and a CUDA device; everything else runs from the cached traces
  on CPU.

The quickest way to re-render a figure is to copy `figures_code/*.py` into a
`paper/` package, put `media/traces/*.npz` in `paper/figures/slides/`, and run
`python -m paper._make_slide_figs`.
