# 0010 — Solve the velocity profile, don't stretch the trajectory

**Status:** accepted · 2026-09-17

## Context

`arm.timing.time_optimal_scale` makes a trajectory feasible by stretching all of
it by one factor until the hardest instant fits. Correct, simple, and wasteful:
measured on a planned path, the torque limit binds at **9% of the path** and the
median utilisation is **0.32**. For most of the move the arm crawls for no
reason, because a single scalar is set by the worst moment.

## Decision

`arm.topp` varies speed *along* the path instead. The change of variable is what
makes that tractable: stop describing the motion as `q(t)` and describe it as a
fixed geometric path `q = f(s)` plus a timing `s(t)` to be solved for. Then

```
q̇ = f'(s)·ṡ            q̈ = f'(s)·s̈ + f''(s)·ṡ²
```

and substituting into the manipulator equation gives

```
τ(s) = a(s)·s̈ + b(s)·ṡ² + c(s)
```

**Torque is linear in `s̈` and in `ṡ²`.** So `τ_min ≤ τ ≤ τ_max` is a linear
inequality and the feasible acceleration interval at each point is *solved for*,
not searched. With `u = ṡ²` and `du/ds = 2s̈`, the classical algorithm (Bobrow;
Shin and McKay; later Pham's TOPP-RA) is: find the maximum-velocity curve,
integrate forward from rest at maximum acceleration, integrate backward from
rest at maximum deceleration, take the pointwise minimum.

The backward pass is what stops the arm arriving somewhere too fast to slow down
for, which a purely greedy forward pass would happily do.

`time_optimal_scale` is kept, both for comparison and because it is the
right tool when a path is already nearly feasible.

## Consequences

Measured on `tuck_and_extend`:

| | duration | median utilisation | binding |
|---|---|---|---|
| uniform scaling | 2.845 s | 0.55 | 9% of the path |
| **TOPP** | **0.803 s** | **0.92** | **93%** |

**3.5× faster, and for the stated reason** — the arm is near its limit almost
everywhere rather than at one instant.

The win depends on how much the torque demand varies along the path. On the
short symmetric `reach_across` problem it is only 1.25×; on `tuck_and_extend`,
where the arm retracts and then extends far out, it is 3.5×.

## Three bugs, each plausible-looking code with absurd output

**A clamped geometric spline.** `arm.timing` uses `bc_type="clamped"` so the arm
starts and finishes at rest. Reusing that here forces `f'(0) = 0`, which bakes
"start at rest" into the **geometry** — a category error, because stopping
belongs in the velocity profile as `u = 0`. With `f'(0) = 0` the coefficient
`a = M·f'` is zero too, no path acceleration reaches any joint, the forward pass
never leaves rest, and `dt = ds/√u` diverges: **a 15,076-second trajectory**.
TOPP needs a *natural* spline.

**Trapezoid rule over a singularity.** `ṡ` is zero at both ends, so `1/ṡ` is
singular there. Integrating `dt = ds/ṡ` by trapezoid rule evaluates that as an
enormous finite number: **5,026 seconds**. The singularity is integrable — near
rest `ṡ ~ √(2·s̈·s)` — and averaging the *speed* across each interval rather
than its reciprocal respects that.

**Exactly at the limit is over the limit.** The solved profile sits on the
constraint by construction, so any discretisation error puts the realised
trajectory over it: 13% over at 50 nodes, still 2% at 1600, converging only as
`1/√nodes`. It now solves against a reduced limit (`margin = 0.92`) and
**verifies against the true one**, with a final uniform stretch if needed, so the
result is feasible rather than nearly feasible.

## Known limitation

The path is smoothed by `arm.timing` before being handed here, using a cubic
spline through the planner's waypoints. That rounds corners, which can move the
path into an obstacle — `path_stays_valid` rechecks, but a failure means going
back and shortcutting less aggressively rather than anything automatic.

A genuinely integrated approach would optimise geometry and timing together.
This keeps them separate, which is simpler to understand and what most practical
pipelines do.
