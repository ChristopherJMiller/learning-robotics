# learn-robotics

A 3-DOF arm, built from first principles — as a way of learning the layers above
motors and encoders: frames, kinematics, dynamics, control, state estimation,
planning, and sim-to-real.

Everything is measured rather than asserted. Where a measurement contradicted
the textbook answer, the measurement is what the code does and the contradiction
is [written down](docs/README.md#things-the-measurements-contradicted).

## Quick start

```bash
nix develop          # or `direnv allow`
just                 # list commands
just test            # 211 tests, ~30s
just execute         # plan a route, time it, drive the arm along it
just view plan_execute
```

## The pipeline

```
goal pose
  → inverse kinematics       hand-derived, closed form
  → RRT + shortcutting       a collision-free route
  → spline + TOPP            a route the actuators can execute
  → feedforward PD           torques that track it
  → MuJoCo
```

`just execute` runs exactly that, with quantised encoders and a Kalman filter in
the loop: **0.26 mm RMS tip error, arriving within 0.10 mm of the goal, zero
contacts**, and nothing hand-tuned along the way.

## Worth looking at

| | |
|---|---|
| `just cspace` | draw configuration space — only possible at three joints |
| `just plan` | RRT vs RRT\*: is rewiring worth its cost? |
| `just sensing` | what encoder quantisation costs, and what one measured parameter is worth |
| `just track` | four controllers on the same trajectory |
| `just impedance` | push the arm; it yields by exactly `F/K` |
| `just execute-watch` | the pipeline in MuJoCo's viewer, at quarter speed |

## How it is verified

Nothing is trusted because it was derived carefully. The hand-written
kinematics, Pinocchio and MuJoCo all read the same `config/arm.toml` by
different routes and must agree to 1e-9 — and since none of them shares code
with the others, agreement is evidence rather than tautology.

`nix flake check` runs the tests, the CAD tests, and lint hermetically.

## Where to start reading

[`docs/`](docs/README.md) — concepts, the derivations, the stack, and ten
decision records.

## Status

| | |
|---|---|
| Kinematics, IK, Jacobian | done, verified three ways |
| Dynamics, control, impedance | done |
| Sensing and state estimation | done |
| CAD and derived inertials | done |
| Planning, trajectories, TOPP | done |
| **Hardware driver** | **not started** — the layer between this and a real arm |
| **Safety** | **not started** — watchdog, e-stop, limit enforcement |
| Perception, middleware, electronics | later — see [docs/00-stack.md](docs/00-stack.md) |
