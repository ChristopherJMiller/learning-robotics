# 0008 — Kalman gains derived from the encoder, not chosen

**Status:** accepted · 2026-09-16

## Context

A Dynamixel reports position from a 4096-count magnetic encoder and does not
measure velocity at all. Velocity has to be differentiated from quantised
positions, and differentiation is precisely the operation that turns a small
position quantum into a large velocity one:

```
1.534 mrad / 0.005 s = 0.307 rad/s   →   × kd=0.6  =  0.184 N·m
```

against a 0.6 N·m continuous limit. **About a third of the actuator spent on
quantisation noise**, and the single biggest reason a controller that works in
simulation buzzes on hardware.

The first attempt was an alpha-beta filter with gains picked by eye.

## Decision

`arm.sensing.KalmanVelocity` computes the gains instead, from two stated
quantities rather than tuned ones.

`R` is exactly computable: quantisation error is uniform across one count, and a
uniform distribution over `[−Δ/2, Δ/2]` has variance `Δ²/12 = 1.96e-7 rad²`. No
tuning, straight from the datasheet.

`Q` states how fast the prediction goes stale. With `use_model` the prediction
runs the arm's real dynamics (`pinocchio.aba`) using the commanded torque rather
than assuming constant velocity, which makes it much better and justifies a
smaller `Q`.

## Consequences

**The hand-picked gains were wrong.** Derived: `α = 0.770, β = 0.542`, against
the `0.7 / 0.35` chosen by eye — beta 35% low, so the velocity estimate was
sluggish. Fixing that alone took alpha-beta from 2.72 mm to 1.63 mm RMS.

| estimator | RMS error | chatter |
|---|---|---|
| perfect state | 0.161 mm | 4.59 mNm |
| finite difference | 0.826 mm | 116.78 mNm |
| alpha-beta (guessed) | 2.720 mm | 29.87 mNm |
| alpha-beta (derived) | 1.633 mm | 45.96 mNm |
| **Kalman, kinematic** | 1.646 mm | 47.19 mNm |
| **Kalman, model-based** | **0.144 mm** | 29.32 mNm |

**At steady state the filter reduces to alpha-beta**, empirically as well as
algebraically — the kinematic Kalman row and the derived alpha-beta row land on
the same numbers because they are the same filter. Alpha-beta was never a
different algorithm, only the same one with the answer guessed.

**The covariance never sees the data.** `P` and `K` depend only on `F`, `H`,
`Q`, `R`, so they converge to the same steady state from any starting confidence
with no measurements at all. Measuring reduces uncertainty; the measurement
agreeing with the prediction does not. Asserted in `tests/test_kalman.py`.

**The estimate beats the sensor.** Steady-state position uncertainty is
0.389 mrad against a 1.534 mrad encoder quantum, because the filter fuses the
sensor with a model.

## The caveat that matters more than the win

A model-based estimator is only as good as its model. With Coulomb friction it
does not know about, it becomes **worse than the kinematic filter that never
trusted a model** — 3.70 mm against 2.92 mm at 0.05 N·m. Raising `Q` recovers
part of it and never all, for a reason worth understanding: **`Q` describes
zero-mean noise, while friction is a systematic bias that always opposes
motion.** No value of `Q` represents a bias.

That is the argument for system identification rather than tuning, and it is
measured in `scripts/sim_sensing.py`: telling both the estimator and the
controller one measured friction value takes tracking error from 3.70 mm to
0.33 mm, an 11× improvement. See [0006](0006-feedforward-over-computed-torque.md)
for why the controller gains more from it than the estimator does.

## Related

Finding this also surfaced a real bug: `forward_dynamics` subtracted viscous
damping but not Coulomb friction, so the filter could not model friction *even
when given the correct value*. Predicted acceleration was wrong by 13.4 rad/s²
against a plant with 0.05 N·m of friction.
