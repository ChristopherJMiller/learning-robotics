# 5 — Perception

Everything before this chapter closed the loop on **proprioception**: the arm
knew where it was because its motors said so. Forward kinematics turned encoder
counts into a tip position, and nothing ever checked that answer against the
world.

Perception is the first component that has to *find out*. That changes the shape
of the problem in two ways worth naming up front:

1. **Every number now has an error bar**, and the error is not isotropic. Some
   directions are measured well and others barely at all, and knowing which is
   which is most of the skill.
2. **Observations are not guaranteed.** A configuration can be perfectly
   reachable, have the marker fully inside the frame, and still produce nothing.

Run it:

```
just perception              # what the camera sees, scored against truth
just calibrate               # find the camera without ever measuring it
just calibrate --degenerate  # ... and watch the same code get it badly wrong
```

## 5.1 The camera

`arm.camera` is the pinhole model and nothing more: a 3×3 intrinsic matrix `K`
built from the configured field of view, and a pose in the world.

```
      [ fx   0  cx ]                     height / 2
K  =  [  0  fy  cy ]        fy  =  ---------------------
      [  0   0   1 ]                 tan(fovy / 2)
```

A point in the camera frame projects to `(u, v) = (K·p)/p_z`. That is the whole
model. What makes it fiddly is not the algebra but the **conventions**.

### Two conventions, one diag(1, −1, −1)

MuJoCo's camera looks down **−z** with **+y up**. OpenCV's looks down **+z**
with **+y down**. Both are right-handed, both are standard, and they differ by

```python
_MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
```

post-multiplied onto the rotation. The operation is its own inverse, which is
why `opencv_from_mujoco` needs no partner function. Getting this wrong does not
produce garbage — it produces a *plausible* pose, mirrored, which is much worse.

`RenderedView` bundles one frame with everything needed to interpret it: RGB,
depth, `T_world_camera`, `K`, and a timestamp. The timestamp is there because
Phase 4 will need it — an observation that arrives late has to be fused at the
time it was *taken*, not the time it arrived.

## 5.2 Fiducials

A camera on its own sees pixels. A **fiducial** is a pattern engineered so a
detector can find it reliably and recover a full 6-DOF pose from a single image
— a remarkable amount of information from one frame, and the reason fiducials
are the standard on-ramp to robot vision.

ArUco works because the pattern is a binary code from a known dictionary. The
detector finds quadrilaterals, reads the bits, rejects anything that is not a
valid codeword, and returns the four corners **in a known order**. Knowing which
corner is which is what makes the pose unambiguous.

### The quiet zone is not part of the marker

A detector needs white space around the black square to find the boundary. It is
generated as part of the texture and is **not** part of `size_m`, because the
pose solver must be told the size of the *black square alone*. Conflating them
is a pure scale error in every recovered translation — which then looks like a
calibration failure rather than a measurement mistake.

### Depth is the weak axis, and by a lot

A single marker gives a full pose, but not equally well in every direction.

| | error |
|---|---|
| along the optical axis | **7.40 mm** |
| across it | **1.42 mm** |

That is conditioning, not a defect. Depth is inferred from **apparent size**, so
a corner error of `σ` pixels propagates as

```
ΔZ  ≈  Z²·σ / (f·L)          L = marker side length
```

which predicts 7.9 mm at one pixel of corner error — matching the measurement.
Two consequences worth internalising:

- Error grows with the **square** of distance. A marker twice as far away is
  four times worse.
- Error falls only **linearly** with marker size. So the cheapest accuracy fix
  is almost always a **bigger marker**, not a better camera.

It is also why sub-pixel corner refinement is on by default. It costs almost
nothing and took that same measurement **from 7.53 mm to 1.19 mm**.

### Not every reachable pose is observable

Detection held below about **49°** of obliquity and failed at **51.8°** — with
all four corners comfortably inside the frame. So it is not visibility that
fails, it is obliquity: an oblique square projects to a smaller quadrilateral
and the code cells inside it become unreadable.

`viewing_angle_deg` exists for this. It needs no render, so candidate poses can
be filtered before anything moves — which is what makes pose collection a
**search** rather than a sweep.

## 5.3 Hand-eye calibration

`arm.toml` declares where the camera is. On hardware nobody knows that: a camera
gets clamped to a tripod, its optical centre is inside the lens barrel, and its
axes are defined by the sensor rather than the housing.

Calibration recovers it from motion alone. For every configuration, one chain
must hold:

```
T_cam_marker  =  T_cam_base · T_base_link(q) · T_link_marker
   measured        unknown        known FK        unknown
```

Two unknowns, both constant. Eliminate the marker mounting between two poses and
what is left is the classic form:

```
A X = X B
```

with `A` the relative **arm** motion and `B` the relative **camera** motion.
That is the whole idea: the unknown transform is whatever makes the arm's motion
and the camera's observation of that motion agree.

This arm is **eye-to-hand** — camera fixed, marker on the forearm — because
three joints cannot control tip orientation, so a camera mounted there would
point wherever the geometry happened to aim it. See ADR 0011.

### Two stages, because closed form is not the answer

OpenCV's seven solvers are all non-iterative. On the same 21 observations they
disagreed by a factor of five:

```
shah 3.57 mm   li 3.92 mm   andreff 6.16 mm   park 8.81 mm
horaud 8.82 mm   tsai 11.94 mm   daniilidis 19.17 mm
```

Refining against the corner pixels collapses that spread completely — **all
seven converge to the same 0.661 mm and 0.032°**, at 0.174 px RMS. The marker
mounting comes out at 0.552 mm as a by-product.

The reason is what each stage optimises. Closed form minimises an algebraic
residual in *pose* space, where a millimetre of depth and a millimetre of
lateral error count the same. They are not the same (§5.2). Minimising
*reprojection* error weights each observation by what the camera actually
resolved.

This is the standard shape of a real calibration pipeline: **a linear method for
the initial guess, nonlinear least squares for the answer.**

### The question the data never asked

`AX = XB` does not always have a unique solution. Each relative motion
constrains `X`'s translation only in the plane perpendicular to that motion's
rotation axis. **If every rotation axis is parallel, the translation along that
shared axis is constrained by nothing.**

On an RRR arm this is one careless decision away. Joints 2 and 3 are both pitch,
so holding the base joint still makes every relative rotation share an axis.
`just calibrate --degenerate`:

| | result |
|---|---|
| `andreff`, `shah`, `li` | raise from inside OpenCV |
| `park` | returns a matrix of `NaN` |
| `tsai`, `horaud`, `daniilidis` | 228 mm, 213 mm, 837 mm — no complaint |

Only the last row is dangerous, because only it looks like an answer.

**And refinement makes it worse.** Least squares slid the camera to **428
metres** from the base while *improving* the reprojection residual to 0.116 px
— better than the 0.174 px of the correct answer. The optimiser is not at
fault. The data does not constrain that direction, so moving along it is free
and buys a fractionally better fit to the noise.

> **A small residual proves consistency, not correctness.**

The residual is still the right thing to report — on hardware there is no ground
truth and it is all you have — but it must be read next to `axis_spread`, which
is computed from forward kinematics alone and so can be checked on candidate
poses *before* moving anything. `calibrate()` refuses below `MIN_AXIS_SPREAD`
and names the reason.

## 5.4 What this chapter earned

- A camera pose recovered to **0.661 mm** and **0.032°** without ever being
  measured, scored against a truth the pipeline never reads.
- The marker mounting to **0.552 mm**, which on hardware is as unknown as the
  camera.
- A named, cheap, *predictive* test for whether a pose set can answer the
  question at all.

## 5.5 Where this goes

| Phase | What | The lesson |
|---|---|---|
| ~~1. Camera~~ | ~~pinhole model, MuJoCo rendering~~ | ~~conventions, and `diag(1,−1,−1)`~~ |
| ~~2. Fiducials~~ | ~~ArUco detection and `solvePnP`~~ | ~~depth is the weak axis~~ |
| ~~3. Hand-eye~~ | ~~`AX = XB` → `T_base_camera`~~ | ~~consistency is not correctness~~ |
| **4. Latency** | timestamped fusion of delayed observations | a measurement is about the past |
| 5. Servoing | vision in the loop; obstacles from depth | closing on what is seen |

Phase 4 is where the timestamp on `RenderedView` starts to matter. A camera
frame takes time to arrive, and fusing it as though it described *now* injects
an error proportional to speed — the faster the arm moves, the worse the
estimate, which is exactly backwards from what you want.

## See also

- ADR 0011 — eye-to-hand with a single fiducial
- ADR 0012 — closed-form hand-eye is an initial guess, not an answer
- `src/arm/camera.py`, `src/arm/fiducial.py`, `src/arm/handeye.py`
