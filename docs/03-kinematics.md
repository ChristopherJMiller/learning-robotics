# Kinematics of this arm

The derivation implemented in `arm.kinematics`, and how it is kept honest.

## Conventions

Zero configuration puts the arm **straight out along +x, horizontal**. Positive
θ₂ and θ₃ **lift**. Angles in radians, lengths in metres.

```
                    elbow  θ₃
               ─────●────── L₂ ──▶ tip
              /
           L₁/
            ●  shoulder  θ₂
            │
            │ L₀
       ═════╧═════  base  θ₁ (yaw)
```

Every joint-to-joint transform here is a **pure translation along a single
axis**, which is why `[geometry]` needs only three scalars:

```
base → shoulder :  +z by L₀
shoulder → elbow:  +x by L₁
elbow → tip     :  +x by L₂
```

This is a Level-1 shorthand. The general case needs a full 6-number pose per
joint (URDF's `<origin xyz rpy>`) or 4 under the Denavit–Hartenberg convention.
**Add a link offset or a wrist and this shorthand must be replaced**, along
with the closed-form IK that depends on it.

Note that "link length" means **the distance between consecutive joint axes**,
not the length of the physical part. A 180 mm bar whose bearing bores sit
150 mm apart is a 150 mm link. Conflating the two produces an IK that is
silently wrong by the difference, with no error anywhere.

## Forward kinematics

In the vertical plane selected by θ₁:

```
r = L₁·cos θ₂ + L₂·cos(θ₂ + θ₃)
z = L₀ + L₁·sin θ₂ + L₂·sin(θ₂ + θ₃)

x = r·cos θ₁
y = r·sin θ₁
```

Always succeeds, exactly one answer, a few multiplications. `fk_frames()`
returns the same result as a chain of `T_base_*` transforms, which is the same
computation written as matrix products.

## Inverse kinematics

**Not a function** — it is set-valued. The asymmetry with FK is the whole
lesson:

| | Forward | Inverse |
|---|---|---|
| Always an answer? | yes | **no** — target may be unreachable |
| How many? | exactly one | 0, 2, 4 … or infinitely many |
| Closed form? | always | only for special geometries |
| Cost | microseconds | trig, or an iterative solve |

**Step 1 — decouple.** Joints 2 and 3 rotate about parallel horizontal axes, so
they only move the tip inside a vertical plane. Joint 1 chooses the plane:

```
θ₁ = atan2(y, x)
```

**Step 2 — collapse to 2D.** `r = √(x² + y²)`, `h = z − L₀`.

**Step 3 — law of cosines.** With `D = √(r² + h²)` the shoulder-to-target
distance, and the interior elbow angle being `π − θ₃` so the sign flips:

```
cos θ₃ = (r² + h² − L₁² − L₂²) / (2·L₁·L₂)
```

**Step 4 — reachability is free.** `cos θ₃` must lie in [−1, 1]. Outside that,
no triangle exists and the target is unreachable. No separate workspace check
is needed: `|cos| > 1` means either `D > L₁+L₂` or `D < |L₁−L₂|`.

**Step 5 — the ± is the elbow branch.** `arccos` returns [0, π]; negating gives
elbow-up. Two genuine solutions, mirrored across the shoulder→target line.

**Step 6 — θ₂ is a difference of angles.**

```
θ₂ = atan2(h, r) − atan2(L₂·sin θ₃, L₁ + L₂·cos θ₃)
     └ point at the target ┘   └ correct for the bend ┘
```

**Step 7 — the shoulder flip.** `θ₁ + π` with `r → −r` is the same planar
problem solved backwards over the shoulder. Usually outside joint limits, so it
is returned **flagged** rather than dropped — a caller can then see *why* a
target failed.

**Step 8 — degenerate guard.** When `|cos θ₃| ≈ 1` both branches coincide; the
duplicate is suppressed.

Up to **2 shoulder × 2 elbow = 4** solutions.

### Choosing among them

Because IK is set-valued, every consumer needs a **selection policy**. This is
a safety requirement, not a nicety: if consecutive waypoints resolve to
different branches, the arm tries to flip through configuration space
instantly. `ik_nearest()` picks the in-limit solution closest to the current
configuration, and `tests/test_kinematics.py` asserts no branch flip across a
40-step path.

## The Jacobian and its determinant

With `c1 = cos θ₁`, `s23 = sin(θ₂+θ₃)` and so on:

```
        ⎡ −r·s1     c1·∂r/∂θ₂    c1·∂r/∂θ₃ ⎤
J(q) =  ⎢  r·c1     s1·∂r/∂θ₂    s1·∂r/∂θ₃ ⎥
        ⎣   0       L₁c₂+L₂c₂₃    L₂·c₂₃   ⎦
```

Working the 3×3 determinant through by hand collapses to something strikingly
compact:

```
det(J) = −r · L₁ · L₂ · sin θ₃
```

which names both singularities directly — `sin θ₃ = 0` (straight or folded
elbow, a boundary case) and `r = 0` (tip on the base axis, the dangerous
interior case). See [01-concepts.md](01-concepts.md#singularities).

## How it is verified

Nothing above is trusted because it was derived carefully. It is trusted
because three independent implementations agree.

```
                   config/arm.toml
                          │
          ┌───────────────┴────────────────┐
     ArmParams                      arm.models
          │                                │
          ▼                                ▼
  arm.kinematics.fk(q)          models/generated/arm.urdf
   hand-derived                            │
          │                        ┌───────┴────────┐
          │                        ▼                ▼
          │                 Pinocchio FK      MuJoCo FK
          └────────────┬───────────┴────────────────┘
                       ▼
               agree to 1e-9 or CI fails
```

The paths share only the TOML. Pinocchio never calls our code and our code
never calls Pinocchio, so agreement is evidence rather than tautology.

Also asserted: `fk(ik(p)) ≈ p` over thousands of sampled configurations; the
analytic Jacobian against finite differences; the closed-form determinant
against `np.linalg.det`; Newton-iteration IK against the closed form; and both
singularities landing exactly where the geometry predicts.

**A cross-check failure means one of two things**: the algebra is wrong, or the
model generator is wrong. To tell them apart, check `q = 0` by hand — the tip
must be at `(L₁+L₂, 0, L₀)` = `(0.300, 0, 0.050)`.
