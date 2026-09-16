# Concepts

The vocabulary, in the order it is needed.

## Degrees of freedom

Two different conversations use this phrase, and people switch between them
without saying so.

**Task space.** A free rigid body in 3D has exactly **6 DOF**: three
translations and three rotations. That is a full *pose*. This number comes from
physics; it is not a property of your robot.

**Joint space.** "3-DOF arm" means **three independently actuated joints**. The
configuration is a point `q = (θ₁, θ₂, θ₃)`. The count is *how many motors*,
not *which world axes*.

So "which three degrees?" has no universal answer — it depends on how the
joints are arranged:

| Arrangement | Joints | Result |
|---|---|---|
| **Articulated (RRR)** ← ours | base yaw, shoulder pitch, elbow pitch | any position in a shell-shaped volume |
| Cylindrical (RPP) | base rotation, vertical slide, radial slide | same volume, different singularities |
| Cartesian (PPP) | three linear slides | a 3D printer |
| SCARA (RRP) | two vertical-axis revolutes + Z | fast planar pick-and-place |

R = revolute, P = prismatic. **The name describes the construction**, not the
motions you get.

### The consequence

Forward kinematics maps `q ∈ ℝ³ → SE(3)`, and SE(3) is 6-dimensional. Three
inputs into a six-dimensional space means the reachable set is at most a
3-dimensional sliver of it. Nothing is projected *down*; most of the space is
simply unreachable.

Practically: this arm can put its tip at any **position** in the workspace, but
the **orientation is whatever falls out**. You do not get to choose.

That is precisely why industrial arms have 6 DOF — three joints to place the
wrist, three more to orient the tool — and why 7-DOF arms exist, so the elbow
can move while the tool pose is held.

Because orientation is not controllable here, `ik()` takes a **3-vector
position**, not a full pose. The function signature encodes the limitation.

## SE(3)

The **Special Euclidean group in 3 dimensions**: all rigid-body motions.
Rotation plus translation, no stretching, no mirroring.

* **E**uclidean — preserves distances
* **S**pecial — determinant +1, no reflections (a left glove cannot be moved
  into a right glove)

A 4×4 homogeneous matrix:

```
T = [ R  p ]      R = 3×3 rotation
    [ 0  1 ]      p = translation
```

The `[0 0 0 1]` row exists so that composition is ordinary matrix
multiplication. **SE(3) is what "a frame" means** — and chaining frames *is*
forward kinematics:

```
T_base_ee = T_base_shoulder @ T_shoulder_elbow @ T_elbow_ee
```

### Naming convention

`T_a_b` = "the pose of b expressed in a". Adjacent subscripts cancel:

```
T_a_b @ T_b_c == T_a_c
```

If the inner indices do not match, the expression is a bug. Most robotics
defects are frame defects, and this convention makes a large class of them
visible by reading. Implemented in `arm.frames`.

### Absolute vs. relative transforms

The convention above says *which frames* a transform relates. It does not say
whether a transform is **absolute** (relative to the world) or **relative to a
parent**, and that distinction has its own failure mode.

`fk_frames` returns absolute poses -- `T_base_shoulder`, `T_base_elbow`,
`T_base_ee` -- which is what you want for computing where things are.

A **transform hierarchy** wants the opposite. Rerun's entity tree, tf2, and a
URDF joint chain all store each node *in its parent* and compose back down the
tree themselves:

```
world/base                     <- T_base_shoulder
world/base/upper_arm           <- T_shoulder_elbow      (NOT T_base_elbow)
world/base/upper_arm/forearm   <- T_elbow_ee            (NOT T_base_ee)
```

Feed absolute poses into a hierarchy and the consumer multiplies them together.
For `q = (0.3, 0.6, -1.2)` that put the tip at `(0.199, 0.225, 0.360)` instead
of `(0.237, 0.073, 0.050)` -- the arm rendered at impossible angles while every
number computed elsewhere stayed perfectly correct.

**Nothing can catch this for you.** Both are `np.ndarray` of shape (4, 4). The
types are identical, no exception is raised, and the error appears only in a
picture. `arm.kinematics.fk_frames_relative` exists to name the difference, and
`tests/test_kinematics.py` pins the contract: composing the relative chain must
reproduce the absolute frames and land on `fk(q)`.

The general lesson is the one that motivates this whole section -- when a
distinction matters but the type system cannot see it, **the distinction has to
live in a name and be enforced by a test.**

### Why orientation is awkward

Orientation genuinely has 3 DOF, but there is no clean way to write it as three
numbers without a catch:

| Representation | Numbers | Catch |
|---|---|---|
| Euler angles | 3 | **gimbal lock**; ~12 incompatible conventions |
| Rotation matrix | 9 | redundant, drifts numerically |
| Quaternion | 4 | double-covers (`q` and `−q` are the same) |
| Axis-angle | 3 | discontinuous at π |

SE(3) is a 6-dimensional *manifold*: 16 numbers in the matrix, 6 of them free.

> **The number of DOF is a property of the thing. The number of numbers you
> write down is a property of your representation.** Gimbal lock is a bug in
> the representation, not in the robot.

Velocity has no such problem. A **twist** `V = [vx, vy, vz, ωx, ωy, ωz]` is an
honest 6-vector, and it is what the Jacobian produces.

## The Jacobian

For any `f: ℝⁿ → ℝᵐ`, the Jacobian is the m×n matrix of partial derivatives —
the best *linear* approximation of `f` at a point.

Here `f` is forward kinematics, so `J(q)` answers: **spin the joints at these
rates, and how fast does the tip move, in which direction?**

```
ẋ = J(q) q̇
```

`J` **depends on `q`**. It is not a constant; how the arm responds to joint
velocity changes completely with configuration.

For this arm, caring only about position, `J` is **3×3 — square**, which
unlocks three things:

1. **Forward velocity** — `ẋ = J q̇`
2. **Inverse velocity** — `q̇ = J⁻¹ẋ`. Cartesian jogging, and iterating it is
   Newton's method for numerical IK.
3. **Force duality** — `τ = Jᵀ F`. The *transpose* maps a desired tip force to
   the joint torques producing it. It falls out of conservation of power
   (`F·ẋ = τ·q̇`) and underpins force control and gravity compensation.

An analytic Jacobian can **always** be checked against finite differences of
`fk` — nudge each joint by 1e-6, see where the tip went, divide. That is a
test, not an opinion, and `tests/test_kinematics.py` runs it.

## Singularities

A configuration where **`J` loses rank** (`det(J) = 0` for square `J`).
Physically: **the arm has instantaneously lost the ability to move in some
direction**, regardless of what the motors do.

Why it hurts: `q̇ = J⁻¹ẋ` blows up. *Near* a singularity `J⁻¹` has enormous
entries, so a modest tip velocity demands absurd joint velocities. Real robots
hit their limits and fault, or whip.

This arm has exactly two, and `det(J) = -r·L₁·L₂·sin(θ₃)` names both:

**Elbow (boundary), `sin θ₃ = 0`.** Arm straight or fully folded. You are at
the edge of reach — you can still swing sideways but cannot move radially
outward. Intuitive: one direction is simply gone.

**Shoulder (interior), `r = 0`.** The tip sits on the base rotation axis, so
spinning θ₁ moves it nowhere. Vicious because it is in the *middle* of the
workspace — reachable during a perfectly reasonable straight-line move. Passing
near it, the IK answer for θ₁ flips ~180° over a tiny distance and the base
tries to slew impossibly fast. The canonical "why did the robot go berserk".

**Mitigations.** Damped least squares — use `Jᵀ(JJᵀ + λ²I)⁻¹` instead of
`J⁻¹`, trading accuracy for bounded joint speed. A **manipulability** measure
`w = √det(JJᵀ)` as a scalar warning. Planning that keeps the elbow bent and
avoids the base axis. A redundant joint to escape through the null space.

### Two things the tests found

*Manipulability is not monotonic* approaching the elbow singularity: for this
arm `w = r·L₁·L₂·|sin θ₃|`, and straightening the elbow shrinks `|sin θ₃|`
while simultaneously **growing** `r`. The product ticks upward before it
collapses.

*The square root costs half your precision.* Because `w = √det(JJᵀ)`, a
determinant accurate to ~1e-20 yields only ~1e-10 in `w`. **Use `det(J)`
directly for a tight singularity test.**
