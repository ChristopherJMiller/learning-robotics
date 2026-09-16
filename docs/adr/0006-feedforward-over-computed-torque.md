# 0006 — Feedforward PD as the default controller, not computed torque

**Status:** accepted · 2026-09-16

## Context

Computed torque is the textbook answer for manipulator control: cancel `M(q)`
and the closed-loop error obeys `ë + kd·ė + kp·e = 0` regardless of
configuration, so one set of gains works everywhere. It looked like the obvious
target to build toward.

Measuring the arm said otherwise.

Reflected rotor inertia — the `armature`, `J_rotor × ratio²` — is about **2.4×
the shoulder's link inertia and 19× the elbow's** for these geared servos.
Sampling `M(q)` across the joint-limit box:

| | inertia variation | joint coupling |
|---|---|---|
| direct drive (armature removed) | **25.4×** | 2.55 |
| geared servo (ours) | **1.48×** | 0.30 |

**The gearbox linearises and decouples the plant.** Cancelling a
configuration-dependent `M(q)` is the entire value of computed torque, so when
`M` varies by 1.5× there is very little left to cancel. This is why industrial
robots ran independent-joint PID for decades, and why direct-drive and
quasi-direct-drive machines suddenly needed whole-body torque control — they
removed the gearbox that had been linearising everything for them.

## Decision

`feedforward_pd` is the default: inverse-dynamics feedforward evaluated at the
**planned** state, plus PD on the error, plus a small clamped integral.

```
τ = rnea(q_d, q̇_d, q̈_d) + D·q̇_d + F_c·sign(q̇_d)   feedforward from the plan
  + kp·e + kd·ė                                      feedback
  + Ki·∫e  (clamped, back-calculated)                residual
```

`computed_torque` is kept, because the comparison is the point and because the
answer reverses the day a joint becomes direct-drive.

## Consequences

**Measured, on a feasible 1 Hz trajectory:**

| controller | RMS tip error | saturated |
|---|---|---|
| PD | 12.54 mm | 0 |
| PD + gravity compensation | 9.05 mm | 0 |
| **feedforward PD** | **0.17 mm** | 0 |
| computed torque | 0.22 mm | 0 |

Feedforward is 53× better than gravity-compensated PD, and — as predicted —
computed torque is *not* better. That is the armature result confirmed in a
measurement rather than an argument.

**Noise.** Computed torque evaluates the model at the measured state, pushing
quantised encoder counts and numerically differentiated velocity through `M(q)`
and the Coriolis terms. Feedforward uses smooth planned quantities and could be
computed offline. On hardware this gap widens in feedforward's favour.

**Gains became two types.** `JointGains` (N·m/rad) and `AccelGains` (1/s²) are
now distinct classes. They were one class, and passing one where the other
belonged produced ζ = 0.11 — a controller that looked broken but was merely
mis-specified. Same failure mode as absolute-vs-relative transforms: a
distinction the type system could not see until it was given a name.
`AccelGains.from_spec(bandwidth_hz, damping_ratio)` derives gains from a stated
response and refuses bandwidths near the control rate.

**Saturation became observable.** `write_torque` returns the applied torque, so
a controller can compute `commanded − applied` and unwind its integrator. An
integrator that cannot distinguish "slow to respond" from "the actuator has
nothing left" will wind up every time it saturates.

**Impedance control is available and verified.** `cartesian_impedance` realises
a commanded stiffness: pushed with 2 N, the tip deflects 40.0 / 20.0 / 10.0 /
5.0 mm at K = 50 / 100 / 200 / 400 N/m, matching `F/K` to the millimetre.

## Revisit this if

`tests/test_dynamics.py::test_armature_is_not_negligible` fails. That test
exists to fail loudly if the armature stops dominating link inertia, because
the whole argument above depends on it.
