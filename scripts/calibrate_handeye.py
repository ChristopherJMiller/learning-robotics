#!/usr/bin/env python3
"""Find the camera without ever measuring it -- and see how it fails.

``arm.toml`` declares where the camera is, so we know the answer. This script
hides it: calibration sees only arm configurations and images, and the declared
pose is used at the very end to grade the result.

Three things worth watching.

**Closed form is a starting guess.** The seven solvers disagree by a factor of
five on the same data. Refining each against the corner pixels collapses them
onto one answer, which is why real pipelines are linear-solve-then-least-squares.

**Some questions the data never asks.** Joints 2 and 3 are both pitch, so
holding the base joint still makes every relative rotation share an axis, and
the camera position along it is unconstrained. Nothing in the fit complains.

**A small residual is not a correct answer.** The degenerate solve fits the
images *better* than the correct one -- while putting the camera metres away.

    just calibrate
    just calibrate --degenerate
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from arm.fiducial import viewing_angle_deg
from arm.handeye import (
    DEFAULT_METHOD,
    METHODS,
    SolverFailure,
    calibrate,
    collect,
    pose_error,
    refine,
    reprojection_residuals,
    search_configurations,
    truth,
)
from arm.params import default_params


def rms_px(calibration, observations, params):
    residuals = reprojection_residuals(
        calibration.pose_base_camera, calibration.pose_link_marker, observations, params
    )
    return float(np.sqrt(np.mean(residuals**2)))


def pinned_configurations(params, count, rng, theta1=1.0):
    """Poses with the base joint frozen -- every relative rotation shares an axis."""
    lower = np.array([joint.limit_lower_rad for joint in params.joints])
    upper = np.array([joint.limit_upper_rad for joint in params.joints])
    marker, camera = params.markers[0], params.cameras[0].pos_m

    found = []
    while len(found) < count:
        q = lower + rng.random(3) * (upper - lower)
        q[0] = theta1
        if viewing_angle_deg(marker, q, camera, params) <= 45.0:
            found.append(q)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--degenerate",
        action="store_true",
        help="pin the base joint, so the answer is unobservable",
    )
    args = parser.parse_args()

    params = default_params()
    rng = np.random.default_rng(args.seed)
    pose_base_camera, pose_link_marker = truth(params)

    if args.degenerate:
        configurations = pinned_configurations(params, args.poses, rng)
        print("Base joint pinned: joints 2 and 3 are both pitch, so every relative")
        print("rotation shares one axis and the camera position along it is")
        print("constrained by nothing at all.\n")
    else:
        configurations = search_configurations(
            params, count=args.poses, rng=rng, max_attempts=50 * args.poses
        )

    observations = collect(configurations, params)
    print(f"{len(configurations)} configurations -> {len(observations)} usable observations")
    print(
        f"viewing angles {min(o.viewing_angle_deg for o in observations):.0f}"
        f"-{max(o.viewing_angle_deg for o in observations):.0f} deg, "
        f"worst PnP reprojection {max(o.reprojection_error_px for o in observations):.3f} px"
    )

    from arm.handeye import axis_spread

    spread = axis_spread(observations, params)
    print(f"rotation-axis spread {spread:.4f}", end="  ")
    print("-- unobservable" if spread < 0.05 else "-- observable")

    print(f"\n{'':>12} {'closed form':>21}   {'after refinement':>22}   {'px rms':>7}")
    print("-" * 72)

    best = None
    best_method = None
    for method in METHODS:
        try:
            closed = calibrate(observations, params, method=method, require_observability=False)
        except (cv2.error, SolverFailure) as failure:
            label = "raised" if isinstance(failure, cv2.error) else "returned NaN"
            print(f"{method:>12} {'solver ' + label:>21}")
            continue

        before = pose_error(closed.pose_base_camera, pose_base_camera)
        refined = refine(closed, observations, params)
        after = pose_error(refined.pose_base_camera, pose_base_camera)

        def fmt(error):
            metres, degrees = error
            unit = f"{metres * 1000:8.2f}mm" if metres < 1 else f"{metres:9.2f}m"
            return f"{unit} {degrees:6.2f}d"

        print(
            f"{method:>12} {fmt(before):>21}   {fmt(after):>22}   "
            f"{rms_px(refined, observations, params):7.4f}"
        )
        # Prefer the default method, but report whatever survived -- on a
        # degenerate set the default is one of the three that refuse outright.
        if best is None or method == DEFAULT_METHOD:
            best, best_method = refined, method

    if best is None:
        print("\nEvery solver refused this pose set. That is the honest outcome:")
        print("nothing here could have answered the question that was asked.")
        return 1

    position, angle = pose_error(best.pose_base_camera, pose_base_camera)
    mount_position, mount_angle = pose_error(best.pose_link_marker, pose_link_marker)

    print(f"\n{best_method} + refinement:")

    def magnitude(metres):
        return f"{metres * 1000:.3f} mm" if metres < 1 else f"{metres:.1f} m"

    print(f"  camera  true {np.round(pose_base_camera[:3, 3], 4)}")
    print(f"       recovered {np.round(best.pose_base_camera[:3, 3], 4)}")
    print(f"          error {magnitude(position)}, {angle:.3f} deg")
    print(f"  marker mount  error {magnitude(mount_position)}, {mount_angle:.3f} deg")
    print(f"  reprojection  {rms_px(best, observations, params):.4f} px rms")

    if args.degenerate:
        print(
            "\nNote what just happened. The fit is sub-pixel -- as good as the"
            "\ncorrect answer, or better -- and the camera is nowhere near where it"
            "\nactually is. The residual measures consistency, not correctness, and"
            "\nno amount of extra poses along the same axis would help."
            "\n\nThis is why `calibrate` refuses this pose set by default."
        )
    else:
        print(
            "\nThe camera was never measured, and nothing in the pipeline read its"
            "\ndeclared pose. Run with --degenerate to see the same code produce a"
            "\nconfident wrong answer."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
