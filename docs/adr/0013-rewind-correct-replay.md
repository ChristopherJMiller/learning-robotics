# 0013 — Rewind to the shutter time; never fuse an observation as "now"

**Status:** accepted · 2026-09-17

## Context

An encoder answers in under a millisecond, which at 200 Hz is a fraction of a
control period — close enough to instantaneous. A camera is not. An image has
to be exposed, read off the sensor, carried over USB, undistorted, thresholded,
decoded, and run through `solvePnP`. Tens of milliseconds is ordinary.

So by the time a camera observation exists, **it describes where the arm was.**

## Decision

`arm.fusion.FusedEstimator` keeps a bounded history of filter states. When an
observation arrives it **rewinds** to the state as it was at `taken_s`,
**corrects** there, and **replays** every control step since from the stored
encoder readings. The corrected history is kept, so a second late observation
rewinds onto the already-corrected timeline rather than undoing the first.

`MarkerObservation` carries `taken_s` as a required field. An observation that
records only a value is an observation that can only be fused wrongly.

## Why, measured

Marker moving at 137 mm/s, camera at 30 Hz, control at 200 Hz, tip RMS error:

| latency | `encoder_only` | `naive` | `rewind` |
|---|---|---|---|
| 0 ms | 2.700 mm | 2.407 mm | 2.407 mm |
| 20 ms | 2.700 mm | 2.815 mm | 2.583 mm |
| 40 ms | 2.700 mm | 3.760 mm | 2.676 mm |
| 80 ms | 2.700 mm | 6.145 mm | 2.701 mm |
| 160 ms | 2.700 mm | 11.471 mm | 2.697 mm |
| 240 ms | 2.700 mm | 16.773 mm | 2.700 mm |

Three things fall out.

**The camera is worth having.** At zero latency it takes 2.700 mm to 2.407 mm,
about 11%.

**Handled correctly, being late costs nothing.** `rewind` decays back *to* the
encoder-only baseline and stops there, because an observation about further in
the past has less left to say once the encoders have already covered the
interval. It does not decay *through* it.

**Handled naively, being late diverges.** `naive` grows linearly with
`velocity × latency` and does not converge: 6.2× the baseline error at 240 ms.
The crossover — where fusing the camera becomes worse than never wiring it up —
sits between **20 and 40 ms**, which is an utterly ordinary USB camera pipeline.

So: **a sensor with a mishandled timestamp is worse than no sensor.** One adds
noise that can be characterised; the other adds a speed-dependent bias that
cannot. And it is invisible in any static test — calibrate at a standstill,
verify at a standstill, and the bug ships.

## What had to be true for this to be measurable

Two things, both discovered by getting them wrong first.

**A camera does not beat an encoder at measuring a joint.** On a rigid arm,
quantisation puts the marker within **0.149 mm** against the camera's 1.5 mm
laterally and 7.4 mm in depth. The filter correctly gives the camera almost no
weight, and none of the three strategies differ by more than 0.02 mm. The
camera earns its place only when the encoder is on the far side of a gearbox
from a compliant printed link — `arm.fusion.LinkSideError`. A degree of
backlash or flex costs 5.87 mm at the marker, well past what the camera
resolves. **A camera beats an encoder at measuring a link, not a joint.**

**The link is the truth; the encoder is the flawed observer.** A first attempt
had the *link* jitter and the encoder read cleanly, which made the camera
actively harmful (−4%) — correctly, because there was nothing coherent left to
estimate. The direction of the error model is not a detail.

## Alternatives rejected

**Extrapolate the measurement forward** to the present before fusing, using the
current velocity estimate. Cheaper — no buffer, no replay — and it removes most
of the first-order error. Rejected because the correction is only as good as
the velocity estimate the camera is supposed to be improving, and the
approximation degrades exactly when the arm accelerates, which is when the
estimate matters most. Worth revisiting if the buffer ever becomes a problem.

**Augment the state with a delayed clone** (stochastic cloning). Exact, and it
avoids replay, but it grows the state by six per outstanding observation and is
considerably more to get right. Replay is exact too, and at 100 buffered steps
the arithmetic is negligible.

**Run the filter at camera rate.** Throws away the 200 Hz encoder stream to
avoid handling two rates, which is a much larger loss than the one being
avoided.

## Consequences

- `KalmanVelocity.update` is split into `predict` and `correct`, plus
  `snapshot`/`restore`. Fused as one step they could only ever be applied to
  "now", which is precisely the bug.
- `correct` takes an *innovation* rather than a measurement, so the same code
  serves the linear encoder update and the nonlinear camera one.
- `KalmanVelocity.measurement_noise` is settable. `R` means "how well does this
  measurement pin down the state", the state is the *link* angle, and an
  encoder that can be a degree out about the link does not pin it down to a
  quantisation step. Leaving `R` at the quantisation floor makes the filter
  overconfident in exactly the sensor that is wrong.
- `arm.kinematics.point_jacobian` gives `∂p/∂q` for any point on the arm, not
  just the tip, via `a × (p − o)`. Verified two ways: exactly against the
  analytic `jacobian()` at the tip, and against finite differences at the marker.
- An observation older than the buffer is **refused**, not clamped to the oldest
  state kept — clamping would reintroduce the error this exists to remove, and
  invisibly.

## What this does not fix

A *systematic* deflection is a bias, and rewinding replays the encoder updates
faithfully, bias included. Measured with gravity droop at 15 N·m/rad,
`rewind` scored 3.84 mm against 4.02 mm for ignoring the camera entirely —
essentially no help. This is the same lesson ADR 0008 records for friction:
`Q` and `R` describe zero-mean noise, and no amount of inflating them
represents a bias. The fixes are to identify and model it, or to estimate it as
an augmented state.
