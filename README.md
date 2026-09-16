# learn-robotics

A 3-DOF articulated arm, built from first principles — as a way of learning the
layers above motors and encoders: frames, kinematics, dynamics, control,
planning, and eventually sim-to-real.

## Quick start

```bash
nix develop          # or `direnv allow`
just                 # list commands
just test            # 47 tests, ~4s
just viz             # trace a circular path → runs/fk_sweep.{rrd,parquet}
just view            # open that recording in the Rerun viewer
```

## What exists

* **`config/arm.toml`** — every physical parameter, validated on load
* **`arm.kinematics`** — FK, closed-form IK and the Jacobian, derived by hand
* **`arm.models`** — URDF and MJCF generated from the TOML, never hand-edited
* **`arm.telemetry`** — runs emit an `.rrd` to look at *and* a parquet table to
  assert on

Verified three ways: the hand-derived maths, Pinocchio, and MuJoCo all agree to
1e-9, sharing nothing but `arm.toml`.

## Where to start reading

[`docs/`](docs/README.md) — concepts, the derivation, the stack, and the
decision records.

## Status

| | |
|---|---|
| Kinematics, IK, Jacobian | done, verified |
| Model generation | done |
| Visualisation | done |
| MuJoCo simulation | next |
| Trajectory generation, control | after that |
| CAD, hardware, middleware | later — see [docs/00-stack.md](docs/00-stack.md) |
