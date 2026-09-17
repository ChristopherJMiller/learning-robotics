# The stack

Why each tool is here, what it is actually for, and when it arrives. Tools are
adopted at the point they solve a problem we have, not up front.

## Status at a glance

| Layer | Tool | State |
|---|---|---|
| Parameters | `config/arm.toml` + pydantic | **in use** |
| Kinematics | hand-derived numpy | **in use** |
| Validation oracles | Pinocchio, MuJoCo | **in use** |
| Model formats | generated URDF + MJCF | **in use** |
| Introspection | Rerun + parquet | **in use** |
| Simulation | MuJoCo | **in use** |
| Control | feedforward PD, impedance | **in use** |
| Sensing | encoder model + Kalman filter | **in use** |
| Planning | RRT, RRT\*, shortcutting | **in use** |
| Trajectories | spline + TOPP | **in use** |
| CAD | build123d | **in use** — inertials derived |
| Hardware driver | none | **the next gap** |
| Safety | none | **the next gap** |
| Middleware | Zenoh | not yet — see below |
| Electronics | KiCad | not yet |
| Firmware | none | **by design** — see actuators |
| Perception | none | not yet — see below |

---

## Parameters: TOML plus a validated model

`config/arm.toml` is the single source of truth. Nothing else may hard-code a
length, mass or limit. The CAD model, the generated URDF, the generated MJCF,
the kinematics and (later) the hardware driver all read it through
`arm.params`.

TOML rather than YAML or JSON, because:

* comments are essential here — `# datasheet 1.5 N·m, derated 60%` is the kind
  of thing that must live next to the number;
* `tomllib` is in the Python 3.11+ standard library, so reading costs nothing;
* `serde` support in Rust is first class, which matters if the motor driver
  ends up there;
* YAML's footguns are real (`no` parsing as `False`, sexagesimal literals), and
  JSON has no comments at all.

**The file is not the contract; `arm.params` is.** The TOML is parsed into a
frozen pydantic model at load, which catches unit slips and sign errors
immediately rather than three layers deep inside a simulation. Validation
includes ordering of joint limits, unit-norm joint axes, positive masses,
inertia tensors that satisfy the triangle inequality (an unphysical one makes a
simulator behave bizarrely rather than error), and a controller rate that does
not exceed the physics rate.

Units live in the key names — `length_m`, `mass_kg`, `limit_lower_rad`,
`stall_torque_nm`. Free to do, and it permanently kills the mm/m
factor-of-1000 bug.

## Kinematics: derived by hand, verified against libraries

`arm.kinematics` implements forward kinematics, closed-form inverse kinematics
and the Jacobian by hand. See [03-kinematics.md](03-kinematics.md) for the
derivation and [adr/0004](adr/0004-hand-derived-kinematics.md) for why.

The short version: for a 3-DOF arm the closed form is roughly twenty lines,
runs in microseconds, returns *all four* solution branches, and never fails to
converge — so it is not merely a teaching exercise, it is genuinely the right
tool at this size. At 6 DOF that reverses completely and you would reach for
IKFast or a numerical solver instead.

## Validation: two independent oracles

This is the part that makes the hand derivation defensible.

```
                   config/arm.toml
                          |
          +---------------+----------------+
          |                                |
     ArmParams                      arm.models
          |                                |
          v                                v
  arm.kinematics.fk(q)          models/generated/arm.urdf
   hand-derived algebra                    |
          |                        +-------+--------+
          |                        v                v
          |                 Pinocchio FK      MuJoCo FK
          |                        |                |
          +------------+-----------+----------------+
                       v
             agree to 1e-9, or the build fails
```

Nothing is shared between the paths except the TOML. Pinocchio never calls our
code and our code never calls Pinocchio, so agreement is evidence rather than
tautology. MuJoCo provides a third implementation for free.

A failure means *either* the algebra is wrong *or* the model generator is
wrong. To tell them apart, check `q = 0` by hand: the tip must be at
`(L1 + L2, 0, L0)`.

Pinocchio is currently a **test-only** dependency. It graduates to a runtime
dependency at the dynamics stage, where `crba` and `rnea` replace hand-derived
mass matrices — hand-rolling recursive Newton–Euler has poor
learning-per-hour and is a well-known source of subtle bugs.

## Model formats: generated, never written

`arm.models` emits both from the TOML. Neither is ever hand-edited; both are
committed so that a parameter change produces a reviewable diff, and a test
asserts regeneration is a no-op so hand edits are caught.

* **URDF** — the interchange format. Pinocchio, MuJoCo, Rerun and the ROS
  ecosystem all read it. Describes the tree, joint origins and axes, limits,
  and link inertials.
* **MJCF** — MuJoCo's native format. Everything URDF has, plus what URDF has no
  vocabulary for: actuators, sensors, solver settings, and the scene.

URDF can only express **trees**. Closed loops — four-bar linkages, parallel
mechanisms, belt coupling between joints — cannot be represented. Irrelevant
for this arm; the reason MJCF exists alongside it.

## Simulation: MuJoCo now, Gazebo later (maybe)

MuJoCo is Apache-2.0, fast, has an excellent contact model, first-class Python
bindings and a built-in viewer. It reads both URDF and MJCF. For a 3-DOF arm it
is more than sufficient.

**Gazebo is deliberately deferred.** Its real differentiator is *sensor*
simulation — cameras, depth, lidar, IMU — bridged into ROS 2 via
`ros_gz_bridge`. We have no sensors. It is also a substantial packaging burden
on NixOS. It becomes interesting at the same moment ROS 2 does, and for the
same reason, so the two arrive together or not at all.

## Introspection: Rerun for eyes, parquet for assertions

Every run writes two artefacts:

* `runs/<name>.rrd` — opened in the Rerun viewer, scrubbed on a timeline,
  inspected in 3D. Rerun's transform hierarchy is effectively tf2 for
  visualisation: log a `Transform3D` per entity path and children inherit
  their parent's pose.
* `runs/<name>.parquet` — a flat numeric table that tests assert against.

**A simulation that only produces a GUI cannot be verified by CI, or by an
agent that cannot see a screen.** Rerun documents a dataframe query API that
reads an `.rrd` back into Pandas, which would remove the need for the second
sink -- but it is not reachable as `rerun.dataframe` in 0.37.2, and that module
is absent from the official upstream wheel, not just the nixpkgs build. The API
has moved since those docs. Writing the table directly costs a few lines and
keeps assertions stable across SDK upgrades.

The viewer itself needed packaging: see
[adr/0005](adr/0005-rerun-viewer-from-wheel.md).

## Sensing: model what the controller is actually allowed to know

Every result up to a point read exact joint angles and velocities straight out
of the simulator. Hardware provides neither: a Dynamixel reports position from a
4096-count encoder and **does not measure velocity at all**.

`arm.sensing` puts that between the simulator and the controller —
quantisation, plus a choice of velocity estimator. `SensedArm` is itself an
`ArmBackend`, so the abstraction built for swapping simulation and hardware took
a sensing model without any controller noticing.

The cost is larger than it looks. Differentiating the encoder quantum gives
0.307 rad/s, which at `kd = 0.6` is 0.184 N·m of pure noise against a 0.6 N·m
limit — **about a third of the actuator spent on quantisation**.

`KalmanVelocity` derives its gains rather than having them chosen, and with
`use_model` predicts using the arm's dynamics instead of assuming constant
velocity, which recovers almost everything quantisation took away. See
[adr/0008](adr/0008-kalman-gains-from-the-datasheet.md) — including why a
model-based estimator becomes *worse* than a model-free one when its model is
wrong.

## Planning and trajectories

`arm.cspace` gives collision checking and, because this arm has only three
joints, a drawable occupancy grid of configuration space. `arm.planning`
implements RRT, RRT\* and shortcutting; `arm.timing` and `arm.topp` turn a
geometric path into something the actuators can execute.

Concepts in [04-planning.md](04-planning.md); the two measured decisions in
[adr/0009](adr/0009-rrt-with-shortcutting.md) and
[adr/0010](adr/0010-time-optimal-parameterisation.md).

Worth stating here because it drove a change to the world: with only a floor,
**107 of 107 randomly sampled valid configuration pairs were connectable by a
straight line**. The free region is nearly convex and a planner had nothing to
do, so `config/arm.toml` gained an obstacle.

## Actuators and firmware: Tier A, Dynamixel

See [adr/0001](adr/0001-actuator-tier-a-dynamixel.md) for the full comparison.

The decision that matters: **Dynamixel means no firmware.** The servo owns its
control loop; we write a serial driver. That defers an entire discipline while
still providing encoder feedback and a current-control mode, so gravity
compensation and impedance control remain reachable.

Gimbal BLDC with SimpleFOC was rejected on torque, not cost — a 2804 gives
~0.029 N·m against the ~1.3 N·m this arm needs at the shoulder. A ~45x
shortfall means a 30–50:1 reduction, which is months of gearbox design rather
than kinematics and control.

What Tier A hides is exactly one layer: the innermost ~1 kHz servo loop. If
that layer becomes the point, moteus is the upgrade, and the hardware
abstraction boundary is designed so only the driver changes.

## Middleware: Zenoh, at the process boundary, not before

**ROS 2 is not a middleware.** It is a framework containing a pluggable
middleware slot (RMW) plus a client library, a message IDL, a build and launch
system, introspection tooling, and an ecosystem (tf2, MoveIt, Nav2, rviz).
Zenoh fills only that first slot; `rmw_zenoh` is the designated successor to
the DDS default.

Roughly: **Zenoh is to ROS 2 as HTTP is to Django.**

For a single process on one machine, middleware is pure cost — latency plus a
serialization bug surface — and teaches nothing. It is adopted at the seam
where `controller` and `sim-or-hardware` become separate processes. At that
point it teaches the real lessons: timestamps, latency, dropped messages.

Standalone Zenoh (Rust core, first-class Rust and Python bindings, trivial to
package on Nix) is the near-term option. Full ROS 2 means `nix-ros-overlay` and
is worth it only for MoveIt, rviz or Nav2 specifically — none of which help an
arm whose IK is already solved in closed form.

## CAD: build123d authors, FreeCAD only views

build123d is Python code-CAD on OpenCASCADE, so geometry is diffable,
reviewable and machine-editable. It is the source of truth. FreeCAD is a
*viewer* — its native format is effectively opaque to git — and lives in its
own devShell because its closure includes TeX Live.

The payoff that matters: **OCCT computes volume, centre of mass and the
inertia tensor.** Link inertials become *derived from geometry* rather than
guessed. Every link in `arm.toml` carries a `provenance` field, currently
`"estimate"`; those flip to `"cad"` with computed numbers, and a test will
assert the URDF matches what the CAD says.

Packaging build123d required six derivations that do not exist in nixpkgs —
see [adr/0003](adr/0003-nix-derivations-not-fhs.md).

## Electronics: KiCad, much later

Board dimensions matter less than the **mechanical/electrical interface**:
motor mount bolt patterns, magnetic-encoder standoff distance, connector
placement, cable routing and strain relief. KiCad round-trips with build123d
via STEP in both directions, its files are s-expression text so they diff
cleanly, and it is already packaged in nixpkgs.

## Perception: not yet, and the thing that changes the stack

Still deliberately absent. Everything so far closes the loop on **joint
encoders**: the arm knows where it is because its motors say so, and it knows
where the obstacle is because `config/arm.toml` says so. Neither is true of a
robot that has to deal with a world it was not told about.

This is also the point at which the two deferred tools earn their place.
**Gazebo's** real differentiator is sensor simulation — cameras, depth, lidar,
IMU — and **ROS 2's** is the ecosystem around it. Both were deferred precisely
because there were no sensors; adding one is what makes them worth their cost.

## What is missing before this drives a real arm

Two gaps, both structural rather than incremental.

**A hardware driver.** `arm.hardware.ArmBackend` exists and `MujocoArm`
implements it; nothing implements it for real servos. That is the single layer
between this repository and a moving arm, and it is the reason the boundary was
designed on day one — see [adr/0001](adr/0001-actuator-tier-a-dynamixel.md).

**A safety layer.** No watchdog, no e-stop, no limit enforcement in the control
loop, no startup homing. Irrelevant in simulation and serious with 1.5 N·m
servos. This is the one pillar from the original brief — "real-time and safety"
— that has not been touched at all.
