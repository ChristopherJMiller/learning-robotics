# 0004 — Derive kinematics by hand; use a library for dynamics

**Status:** accepted · 2026-09-15

## Context

Pinocchio, Drake, MoveIt and the Robotics Toolbox all compute forward
kinematics, Jacobians and inverse kinematics. In professional work almost
nobody hand-writes these.

Being honest about what real practice looks like:

| Piece | Industry practice |
|---|---|
| Forward kinematics | **always** a library |
| Jacobian | **always** a library |
| Dynamics | **always** a library, emphatically |
| Inverse kinematics | mixed — numerical solvers, or closed-form **generated** by IKFast / `ik_geo` |

Even where closed-form IK matters in production — and it does, for speed
(microseconds vs milliseconds inside a planner) and completeness (all
solutions, never fails to converge) — practitioners *generate* it rather than
derive it.

## Decision

**Hand-derive forward kinematics, closed-form inverse kinematics and the
Jacobian. Use a library for dynamics.**

That line is drawn where learning-per-hour falls off. FK/IK/J for this geometry
is roughly eighty lines and an afternoon's derivation, and every later concept
rests on it: without it, singularities are "the solver warned me" rather than
"I watched `det(J)` go to zero", and damped least squares is a magic λ copied
from a paper.

Mass matrices, Coriolis terms and recursive Newton–Euler are the opposite —
error-prone tedium with poor returns. `pinocchio.crba` and `pinocchio.rnea`
will own those.

## Consequences

**The hand-derived code is genuinely production-grade at this size.** For 3 DOF
the closed form is ~20 lines, runs in microseconds, returns all four branches,
and cannot fail to converge. Measured peak tracking error over a 400-waypoint
circular path is **1.76e-16 m** — machine precision. This is not a toy kept for
sentiment; it is the right tool for this arm.

**It does not generalise, and that is accepted.** At 6 DOF this reverses
entirely — IKFast or a numerical solver, without hesitation. The
`params.py` → URDF pipeline already makes that a contained swap: delete
`kinematics.py`, keep everything else.

**Verification is mandatory, and is the actual deliverable.** Hand-derived code
is only defensible because it is checked against two independent engines that
share nothing with it but the TOML. See
[03-kinematics.md](../03-kinematics.md#how-it-is-verified).

**Pinocchio is currently test-only.** It graduates to a runtime dependency at
the dynamics stage.

## What this actually buys, professionally

Not the ability to write it — the ability to **debug** it. When MoveIt reports
"no IK solution found" for an obviously reachable target, or an arm lurches
crossing mid-workspace, the people who fix it are those who know what a
singularity is and what a branch flip looks like.

**The library spares you the typing, not the concepts.**
