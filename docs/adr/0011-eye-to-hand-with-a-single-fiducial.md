# 0011 — Eye-to-hand, with one marker on the forearm

**Status:** accepted · 2026-09-17

## Context

Everything through Phase 2 of the control stack assumed the arm knew where it
was because its motors said so. Perception is the first component that has to
*find out*, and it needs a physical arrangement before it needs any code.

Two arrangements are standard:

- **Eye-in-hand** — the camera rides on the end effector.
- **Eye-to-hand** — the camera is fixed, and the arm carries the target.

## Decision

**Eye-to-hand.** The camera is a fixed `[[cameras]]` entry in `config/arm.toml`;
a 40 mm `DICT_4X4_50` ArUco marker rides on the forearm as a `[[markers]]` entry
and is baked into the MJCF as a texture, so MuJoCo renders it and OpenCV reads
it back with nothing shared but the geometry.

The deciding argument is the arm itself. **Three joints cannot control
end-effector orientation.** Base yaw plus two pitches positions the tip and
leaves its orientation a function of where it ended up. A camera bolted there
would point wherever the geometry happened to aim it — so any pose that was
good for reaching would be arbitrary for seeing, and the two objectives would
fight for the only three degrees of freedom available. Fixing the camera
decouples them.

A single planar fiducial, rather than a checkerboard rig, because it recovers a
full 6-DOF pose from one image with a known correspondence order, and because
`solvePnP` with `IPPE_SQUARE` is specialised for exactly this case.

## What this costs, measured

**Depth is the weak axis, by 3–5×.** Depth is inferred from apparent size, so a
corner error of `σ` pixels propagates as `ΔZ ≈ Z²σ / (f·L)`. On a 40 mm marker
at 427 mm that predicts 7.9 mm per pixel of corner error, and the measurement
agreed: **7.40 mm along the optical axis against 1.42 mm across it**. The
practical consequences are worth internalising — error grows with the *square*
of distance and falls only *linearly* with marker size, so the cheapest accuracy
fix is almost always a bigger marker rather than a better camera.

**Sub-pixel refinement is free and large.** Turning on `CORNER_REFINE_SUBPIX`
took the same measurement from **7.53 mm to 1.19 mm**.

**Not every reachable pose yields an observation.** Detection held below about
49° of obliquity and failed at 51.8°, with all four corners comfortably inside
the frame. So visibility is not the binding constraint — obliquity is. This is
why `arm.fiducial.viewing_angle_deg` exists and why collecting calibration poses
is a *search* rather than a sweep.

## Alternatives rejected

**Eye-in-hand.** Rejected on the orientation argument above. It becomes the
right answer the moment a wrist is added, and this decision should be revisited
then — it is a consequence of 3 DOF, not a general preference.

**A checkerboard or ChArUco target.** More corners and better conditioning, but
it needs a rigid printed board mounted on a link, and it gives one pose per
image just as the marker does. Not worth the mechanical complication at this
scale; ChArUco would be the answer if intrinsics also had to be estimated.

**Depth-only sensing.** The depth buffer is rendered and available, but a depth
camera gives geometry without identity — it cannot say *which* link it is
looking at. The marker is what makes the observation attributable to a frame.

## Consequences

- `config/arm.toml` now carries `[[cameras]]` and `[[markers]]`; both are
  emitted into the MJCF, so the simulator and the perception code cannot drift.
- Marker textures are generated (`just markers`) and committed like the CAD
  meshes, with a test asserting regeneration is a no-op.
- The MJCF references them through a *relative* `texturedir`. An absolute path
  worked locally and broke `nix flake check`, which has no `/home`.
- Anything mounted to a link must use `arm.kinematics.link_frames`, not
  `fk_frames`: a body frame includes its own joint's rotation and a joint frame
  does not. Conflating them put the marker's ground truth **151 mm** from where
  the simulator had drawn it, and `test_link_frames_match_the_simulator` pins it.
