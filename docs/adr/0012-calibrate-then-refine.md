# 0012 — Closed-form hand-eye is an initial guess, not an answer

**Status:** accepted · 2026-09-17

## Context

`config/arm.toml` declares where the camera is, which is a convenience of
simulation and a fiction on hardware. A camera gets clamped to a tripod; its
optical centre is inside the lens barrel and its axes are defined by the sensor,
not the housing. Nobody measures that with a tape.

Hand-eye calibration recovers it from motion. For every configuration one chain
must hold:

```
T_cam_marker  =  T_cam_base · T_base_link(q) · T_link_marker
   measured        unknown        known FK        unknown
```

Eliminating the constant marker mounting between two poses gives `AX = XB`, with
`A` the relative arm motion and `B` the relative camera motion.

OpenCV offers seven solvers across two entry points. All seven are
*non-iterative*: each makes an algebraic simplification to get a one-shot
solution.

## Decision

**Solve in closed form, then refine against the corner pixels.** `arm.handeye`
runs a linear solver for an initial guess and then minimises reprojection error
over both unknown transforms with `scipy.optimize.least_squares`.

## Why, measured

On 21 rendered observations the seven closed-form solvers disagreed by a factor
of five — 3.57 mm (`shah`) to 19.17 mm (`daniilidis`) — a spread no amount of
*choosing between them* resolves.

Refinement collapses it. **All seven converge to the same 0.661 mm and
0.032°**, at 0.174 px RMS. The choice of closed-form method stops mattering once
it is only a starting point.

The reason is what each stage optimises. The closed-form methods minimise an
algebraic residual in *pose* space, where a millimetre of depth error and a
millimetre of lateral error count the same — but they are not the same, because
depth from a single fiducial is 3–5× less certain (ADR 0011). Minimising
*reprojection* error weights each observation by what the camera actually
resolved, so the well-conditioned directions dominate, as they should.

The marker mounting comes free, recovered to 0.552 mm — which matters on
hardware, where the marker is stuck on by hand and its offset from the link
frame is as unmeasured as the camera pose.

## The failure mode this is guarded against

`AX = XB` does not always have a unique answer. Each relative motion constrains
`X`'s translation only in the plane perpendicular to that motion's rotation
axis. **If every rotation axis is parallel, the translation along that shared
axis is constrained by nothing at all.**

This is not hypothetical on an RRR arm: joints 2 and 3 are both pitch, so
holding the base joint still makes every relative rotation share one axis. On
synthetic noise-free data that alone costs 214 mm. On rendered observations it
fails three different ways at once:

| | result |
|---|---|
| `andreff`, `shah`, `li` | raise from inside OpenCV |
| `park` | returns a matrix of `NaN` |
| `tsai`, `horaud`, `daniilidis` | 228 mm, 213 mm, 837 mm — and no complaint |

Only the last row is dangerous, because only it looks like an answer.

**And refinement makes it worse.** Given that set, least squares slid the camera
to **428 metres** from the base while *improving* the residual to 0.116 px —
better than the 0.174 px of the correct answer. Not a bug in the optimiser: the
data genuinely does not constrain that direction, so travelling along it is free
and buys a fractionally better fit to the noise.

So: **a small residual proves consistency, not correctness.** `arm.handeye`
computes `axis_spread`, refuses to solve below `MIN_AXIS_SPREAD = 0.05`, and
names the reason. The check is worth running *before* moving anything, because
it needs only forward kinematics.

## Alternatives rejected

**Pick the best closed-form method and stop.** `shah` is the best here at
3.57 mm, but "best on this pose set" is not a property that transfers, and it is
still 5× worse than refinement. Refinement makes the choice irrelevant, which is
a better outcome than making it correctly.

**Refine only, from identity.** Reprojection error is non-convex in pose; a bad
start finds a local minimum or none. The linear solve is cheap and its job is to
land in the right basin.

**Trust the residual as the acceptance test.** The degenerate case above is
precisely the counterexample. The residual is still worth reporting — it is the
only check available on hardware — but it must be read next to the axis spread.

## Consequences

- `calibrate()` raises `DegenerateGeometryError` rather than returning a
  confident wrong answer; `require_observability=False` opts out, and
  `just calibrate --degenerate` demonstrates what that costs.
- A `SolverFailure` is raised when a solver returns a non-finite or
  non-orthonormal pose. `park` returns `NaN` rather than raising, and an
  unchecked `NaN` first surfaced three calls downstream as "SVD did not
  converge", naming nothing useful.
- `METHODS` records which OpenCV entry point each method belongs to. The two
  enums overlap — `CALIB_HAND_EYE_TSAI` and `CALIB_ROBOT_WORLD_HAND_EYE_SHAH`
  are both `0` — and passing one family's constant to the other function is
  silently accepted. That mistake cost a camera 600 mm and 120° out, with
  nothing raised.
- Both argument mappings were confirmed numerically against synthetic data
  before being trusted, each reproducing the planted transforms to machine
  precision.
