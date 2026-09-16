# Documentation

Notes written while building, in the order they become useful.

| Document | What it covers |
|---|---|
| [01-concepts.md](01-concepts.md) | Degrees of freedom, SE(3), frames, the Jacobian, singularities |
| [03-kinematics.md](03-kinematics.md) | The derivation this arm actually uses, and how it is verified |
| [00-stack.md](00-stack.md) | Every tool, why it is here, and when it arrives |
| [02-repo-organization.md](02-repo-organization.md) | Why the tree is shaped the way it is |

## Decisions

Architecture decision records — what was chosen, what was rejected, and the
reason, so that a future reader can tell a deliberate choice from an accident.

| ADR | Decision |
|---|---|
| [0001](adr/0001-actuator-tier-a-dynamixel.md) | Dynamixel servos, not gimbal BLDC or moteus |
| [0002](adr/0002-single-source-of-truth.md) | One TOML file owns every physical parameter |
| [0003](adr/0003-nix-derivations-not-fhs.md) | Real derivations for the CAD chain, no FHS environment |
| [0004](adr/0004-hand-derived-kinematics.md) | Derive kinematics by hand; use a library for dynamics |
| [0005](adr/0005-rerun-viewer-from-wheel.md) | Extract a version-matched Rerun viewer from the upstream wheel |
| [0006](adr/0006-feedforward-over-computed-torque.md) | Feedforward PD as the default controller, not computed torque |
| [0007](adr/0007-cad-derived-inertials.md) | Link inertials come from CAD, not from estimates |
