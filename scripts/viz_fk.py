#!/usr/bin/env python3
"""Drive the arm along a Cartesian path and log the result.

Sweeps the base through a wide arc while the reach and height also change, so
all three joints are exercised. A circle in the y-z plane -- the obvious first
choice -- barely moves the base at all, because theta1 = atan2(y, x) spans only
about +/-17 degrees there and the arm appears to pivot in place.

    just viz            write runs/fk_sweep.rrd + .parquet
    just viz-live       open the Rerun viewer live instead
    just view           open a recorded run
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.kinematics import fk, ik_nearest, jacobian_det, manipulability
from arm.params import default_params
from arm.telemetry import Telemetry

YAW_AMPLITUDE_RAD = 2.4  # +/-137 degrees, inside the +/-175 degree limit
REACH_MID_M, REACH_SWING_M = 0.20, 0.045
HEIGHT_MID_M, HEIGHT_SWING_M = 0.10, 0.040


def path_point(t: float) -> np.ndarray:
    """Target position at normalised time ``t`` in [0, 1)."""
    phase = 2.0 * np.pi * t
    yaw = YAW_AMPLITUDE_RAD * np.sin(phase)
    reach = REACH_MID_M + REACH_SWING_M * np.cos(2.0 * phase)
    height = HEIGHT_MID_M + HEIGHT_SWING_M * np.sin(2.0 * phase)
    return np.array([reach * np.cos(yaw), reach * np.sin(yaw), height])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spawn", action="store_true", help="open the viewer live")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--name", default="fk_sweep")
    args = parser.parse_args()

    params = default_params()
    q = np.array([0.0, 0.6, -1.2])
    missed = 0

    with Telemetry(args.name, params, spawn=args.spawn) as log:
        for step in range(args.steps):
            t = step / args.steps
            target = path_point(t)

            solution = ik_nearest(target, q, params)
            if solution is None:
                missed += 1
                continue
            q = solution.q

            achieved = fk(q, params)
            error = float(np.linalg.norm(achieved - target))
            determinant = jacobian_det(q, params)

            log.at(t)
            log.log_arm(q)
            log.log_target(target)
            log.log_scalars(
                det_jacobian=determinant,
                manipulability=manipulability(q, params),
                tracking_error_m=error,
            )
            log.row(
                sim_time=t,
                q=q,
                target=target,
                achieved=achieved,
                det_jacobian=determinant,
                manipulability=manipulability(q, params),
                tracking_error_m=error,
            )

        print(f"wrote {log.rrd_path}")
        print(f"wrote {log.table_path}")

    if missed:
        print(f"WARNING: {missed}/{args.steps} waypoints had no in-limit IK solution")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
