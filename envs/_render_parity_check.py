"""Assert that mujoco_quadrotor_render.xml is dynamically identical to the physics model.

The render model adds cosmetic geoms, a textured floor and a mocap goal marker so
the sim-to-sim slide can show a real rendered quadrotor. None of that is allowed
to change what the policy flies. This asserts it: same model constants, and
bit-identical state trajectories under the same action sequence.

Usage:
    python -m envs._render_parity_check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mujoco_quadrotor import MuJoCoQuadrotorEnv    # noqa: E402

PHYS = ROOT / "envs" / "mujoco_quadrotor.xml"
REND = ROOT / "envs" / "mujoco_quadrotor_render.xml"


def main(n_steps: int = 300, seed: int = 4321) -> int:
    a = MuJoCoQuadrotorEnv(xml_path=str(PHYS), partial_obs=True)
    b = MuJoCoQuadrotorEnv(xml_path=str(REND), partial_obs=True)

    fails = []

    # --- model constants the controller depends on
    for name, x, y in [
        ("timestep", a.model.opt.timestep, b.model.opt.timestep),
        ("gravity", a.model.opt.gravity, b.model.opt.gravity),
        ("nq", a.model.nq, b.model.nq),
        ("nv", a.model.nv, b.model.nv),
        ("nu", a.model.nu, b.model.nu),
        ("body_mass[quad]", a.model.body_mass[a.model.body("quad").id],
         b.model.body_mass[b.model.body("quad").id]),
        ("body_inertia[quad]", a.model.body_inertia[a.model.body("quad").id],
         b.model.body_inertia[b.model.body("quad").id]),
        ("actuator_gear", a.model.actuator_gear, b.model.actuator_gear),
        ("ctrlrange", a.model.actuator_ctrlrange, b.model.actuator_ctrlrange),
    ]:
        if not np.allclose(np.asarray(x, float), np.asarray(y, float), rtol=0, atol=0):
            fails.append(f"{name}: {x} != {y}")

    # --- the collision set must match exactly
    #
    # Cosmetic geoms are contype/conaffinity 0 and invisible to the solver. Any
    # difference in the *collidable* set is a physics change: an earlier version
    # of the render model silently dropped the airframe's collision box, and the
    # vehicle flew through the ground plane in the rendered video.
    def collidable(model):
        return sorted(
            (int(model.geom_type[g]), tuple(np.round(model.geom_size[g], 6)))
            for g in range(model.ngeom)
            if model.geom_contype[g] or model.geom_conaffinity[g]
        )
    if collidable(a.model) != collidable(b.model):
        fails.append(f"collidable geoms differ:\n"
                     f"      physics {collidable(a.model)}\n"
                     f"      render  {collidable(b.model)}")

    # --- identical state trajectory under an identical action sequence
    rng = np.random.default_rng(seed)
    oa, _ = a.reset(seed=seed)
    ob, _ = b.reset(seed=seed)
    if not np.array_equal(oa, ob):
        fails.append("reset observation differs")

    max_dq = 0.0
    for t in range(n_steps):
        u = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        oa, *_ = a.step(u)
        ob, *_ = b.step(u)
        max_dq = max(max_dq, float(np.abs(a.data.qpos - b.data.qpos).max()),
                     float(np.abs(a.data.qvel - b.data.qvel).max()))
    if max_dq != 0.0:
        fails.append(f"state diverged: max |dq|,|dv| = {max_dq:.3e}")

    # --- and again through ground contact, which the flight test never reaches
    a.reset(seed=seed); b.reset(seed=seed)
    # A zero action means zero *acceleration* — the controller cancels gravity and
    # the vehicle hovers. Command a full-scale descent to actually reach the floor.
    descend = np.array([0.0, 0.0, -1.0], np.float32)
    drop_dq, touched = 0.0, False
    for _ in range(400):
        a.step(descend); b.step(descend)
        drop_dq = max(drop_dq, float(np.abs(a.data.qpos - b.data.qpos).max()),
                      float(np.abs(a.data.qvel - b.data.qvel).max()))
        touched = touched or a.data.ncon > 0
    if not touched:
        fails.append("drop test never made contact — it is not testing anything")
    if drop_dq != 0.0:
        fails.append(f"state diverged under ground contact: max = {drop_dq:.3e}")
    final_z = float(a.data.qpos[2])
    if final_z < -0.05:
        fails.append(f"physics model fell through the floor (z={final_z:.3f})")

    if fails:
        print("PARITY FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print(f"parity OK — {n_steps} powered steps + a 400-step drop onto the "
          f"ground, max state difference exactly 0.0")
    print(f"  render model adds {b.model.ngeom - a.model.ngeom} cosmetic geoms "
          f"and {b.model.nbody - a.model.nbody} mocap body; "
          f"the collidable set is identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
