# Documentation

Notes written while building, in the order they become useful.

| Document | What it covers |
|---|---|
| [01-concepts.md](01-concepts.md) | Degrees of freedom, SE(3), frames, the Jacobian, singularities |
| [03-kinematics.md](03-kinematics.md) | The derivation this arm uses, and how it is verified |
| [04-planning.md](04-planning.md) | Configuration space, RRT, and path versus trajectory |
| [00-stack.md](00-stack.md) | Every tool, why it is here, and when it arrives |
| [02-repo-organization.md](02-repo-organization.md) | Why the tree is shaped the way it is |

## Decisions

Architecture decision records — what was chosen, what was rejected, and why, so
a later reader can tell a deliberate choice from an accident. Most of them
record a measurement that contradicted the obvious answer.

| ADR | Decision |
|---|---|
| [0001](adr/0001-actuator-tier-a-dynamixel.md) | Dynamixel servos, not gimbal BLDC or moteus |
| [0002](adr/0002-single-source-of-truth.md) | One TOML file owns every physical parameter |
| [0003](adr/0003-nix-derivations-not-fhs.md) | Real derivations for the CAD chain, no FHS environment |
| [0004](adr/0004-hand-derived-kinematics.md) | Derive kinematics by hand; use a library for dynamics |
| [0005](adr/0005-rerun-viewer-from-wheel.md) | Extract a version-matched Rerun viewer from the upstream wheel |
| [0006](adr/0006-feedforward-over-computed-torque.md) | Feedforward PD as the default controller, not computed torque |
| [0007](adr/0007-cad-derived-inertials.md) | Link inertials come from CAD, not from estimates |
| [0008](adr/0008-kalman-gains-from-the-datasheet.md) | Kalman gains derived from the encoder, not chosen |
| [0009](adr/0009-rrt-with-shortcutting.md) | RRT with shortcutting as the default, not RRT\* |
| [0010](adr/0010-time-optimal-parameterisation.md) | Solve the velocity profile, don't stretch the trajectory |

## Things the measurements contradicted

Collected because they are the most useful part of the record — in each case the
textbook answer was reasonable and the measurement disagreed.

* **Computed torque is not better here.** Reflected rotor inertia is ~19× the
  elbow link inertia, so the gearbox linearises the plant and `M(q)` varies only
  1.48× across the workspace against 25.4× direct-driven. Feedforward PD wins.
  ([0006](adr/0006-feedforward-over-computed-torque.md))
* **Estimated inertials were 45–74% wrong**, and the base was *qualitatively*
  wrong — the estimate had `Izz` largest, the geometry has it smallest.
  ([0007](adr/0007-cad-derived-inertials.md))
* **A model-based estimator is worse than a model-free one when its model is
  wrong**, and no amount of process noise fixes it, because `Q` describes
  zero-mean noise and friction is a bias.
  ([0008](adr/0008-kalman-gains-from-the-datasheet.md))
* **Rewiring buys 8% for 12× the cost**, and what makes it worth paying for is
  topology — genuinely alternative routes — not obstacle count.
  ([0009](adr/0009-rrt-with-shortcutting.md))
* **Slowing a trajectory cannot fix gravity.** Peak torque converges to the
  gravity requirement, not to zero.
  ([0010](adr/0010-time-optimal-parameterisation.md))
