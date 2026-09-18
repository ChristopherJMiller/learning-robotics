#!/usr/bin/env python3
"""What the camera sees, and how well it recovers where the arm is.

Everything before this closed the loop on proprioception: the arm knew where it
was because its motors said so. This is the first component that has to *find
out*, and the interesting part is the error -- a recovered pose is scored
against ground truth the perception code never sees.

Two results worth watching for.

**Depth is the weak axis.** A single marker gives a full pose, but not equally
well in every direction. Depth is inferred from apparent size, so its error
grows as ``Z^2 / (f * L)`` while lateral error grows only linearly -- typically
several times worse, which is why the cheapest accuracy fix is a bigger marker
rather than a better camera.

**Not every reachable pose produces an observation.** A configuration can be
perfectly valid for the arm, have the marker fully in frame, and still be too
edge-on for the detector to decode. That is what makes collecting calibration
poses a search.

    just perception
    just view perception
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from arm.camera import SimulatedCamera, opencv_from_mujoco
from arm.fiducial import detect, marker_pose_world, viewing_angle_deg
from arm.frames import inverse
from arm.kinematics import fk_frames
from arm.params import default_params
from arm.sim import MujocoArm
from arm.telemetry import RUNS_DIR


def pose_error(estimated, truth):
    position = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    relative = inverse(truth) @ estimated
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
    return position, angle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=24)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()

    params = default_params()
    marker = params.markers[0]
    camera_position = np.asarray(params.cameras[0].pos_m)

    arm = MujocoArm(params)
    camera = SimulatedCamera(arm.model, arm.data, params.cameras[0].name, params)

    rr.init("learn_robotics_perception", spawn=args.spawn)
    if not args.spawn:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        rr.save(str(RUNS_DIR / "perception.rrd"))

    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="world", origin="/world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="camera", origin="/camera"),
                rrb.TimeSeriesView(name="error", origin="/metrics"),
            ),
        )
    )
    rr.log(
        "world/obstacles",
        rr.Boxes3D(
            centers=[o.pos_m for o in params.obstacles],
            half_sizes=[o.half_size_m for o in params.obstacles],
            colors=[[170, 120, 95]],
        ),
        static=True,
    )
    rr.log(
        "world/camera",
        rr.Points3D([camera_position], radii=0.012, colors=[[120, 200, 255]]),
        static=True,
    )

    print(f"{'q (deg)':>26} {'angle':>7} {'found':>6} {'pos err':>9} {'depth':>8} {'lateral':>8}")
    print("-" * 70)

    missed = 0
    positions, depths, laterals = [], [], []

    for index in range(args.poses):
        phase = index / args.poses
        q = np.array(
            [
                0.9 + 0.5 * np.sin(2 * np.pi * phase),
                0.45 + 0.25 * np.sin(4 * np.pi * phase),
                -1.1 + 0.3 * np.cos(2 * np.pi * phase),
            ]
        )
        arm.reset(q)
        view = camera.capture()
        truth = marker_pose_world(marker, q, params)
        angle = viewing_angle_deg(marker, q, camera_position, params)

        rr.set_time("pose", sequence=index)
        rr.log("camera/image", rr.Image(view.rgb))
        skeleton = np.array([f[:3, 3] for f in fk_frames(q, params)])
        rr.log("world/arm", rr.LineStrips3D([skeleton], radii=0.004))
        rr.log(
            "world/marker_true", rr.Points3D([truth[:3, 3]], radii=0.008, colors=[[90, 220, 120]])
        )
        rr.log("metrics/viewing_angle_deg", rr.Scalars(angle))

        detections = detect(view, params)
        if not detections:
            missed += 1
            print(
                f"{str(np.round(np.degrees(q), 0)):>26} {angle:>6.1f}d {'no':>6}"
                f" {'--':>9} {'--':>8} {'--':>8}"
            )
            rr.log("metrics/detected", rr.Scalars(0.0))
            continue

        detection = detections[0]
        estimated = opencv_from_mujoco(view.pose_world) @ detection.pose_camera
        position, _ = pose_error(estimated, truth)

        axis = -view.pose_world[:3, 2]
        error = estimated[:3, 3] - truth[:3, 3]
        depth = abs(float(np.dot(error, axis)))
        lateral = float(np.linalg.norm(error - np.dot(error, axis) * axis))

        positions.append(position)
        depths.append(depth)
        laterals.append(lateral)

        rr.log("metrics/detected", rr.Scalars(1.0))
        rr.log("metrics/position_error_mm", rr.Scalars(position * 1000))
        rr.log("metrics/depth_error_mm", rr.Scalars(depth * 1000))
        rr.log("metrics/lateral_error_mm", rr.Scalars(lateral * 1000))
        rr.log(
            "world/marker_seen",
            rr.Points3D([estimated[:3, 3]], radii=0.008, colors=[[255, 140, 40]]),
        )
        rr.log(
            "camera/corners",
            rr.Points2D(detection.corners_px, radii=3.0, colors=[[255, 210, 60]]),
        )

        print(
            f"{str(np.round(np.degrees(q), 0)):>26} {angle:>6.1f}d {'yes':>6} "
            f"{position * 1000:>7.2f}mm {depth * 1000:>6.2f}mm {lateral * 1000:>6.2f}mm"
        )

    print()
    if positions:
        print(
            f"detected {len(positions)}/{args.poses} poses; "
            f"mean position error {np.mean(positions) * 1000:.2f} mm"
        )
        print(
            f"mean depth error {np.mean(depths) * 1000:.2f} mm against "
            f"{np.mean(laterals) * 1000:.2f} mm lateral -- "
            f"{np.mean(depths) / max(np.mean(laterals), 1e-9):.1f}x worse"
        )
    if missed:
        print(
            f"\n{missed} pose(s) produced no observation at all. Reachable, in frame,"
            "\nand too oblique to decode -- which is why calibration has to search for"
            "\nusable poses rather than sweep a grid."
        )

    stream = rr.get_global_data_recording()
    if stream is not None:
        stream.flush()
    if not args.spawn:
        print(f"\nwrote {RUNS_DIR / 'perception.rrd'}   (just view perception)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
