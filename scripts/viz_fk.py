#!/usr/bin/env python3
"""Drive the arm along a Cartesian path and log the result.

Traces a circle in a vertical plane using closed-form IK at every waypoint,
which exercises the parts worth watching: branch continuity, how the
determinant of the Jacobian moves as the arm folds and extends, and where the
conditioning gets poor.

    just viz            write runs/fk_sweep.rrd + .parquet
    just viz --spawn    open the Rerun viewer live instead
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.kinematics import fk, ik_nearest, jacobian_det, manipulability
from arm.params import default_params
from arm.telemetry import Telemetry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spawn", action="store_true", help="open the viewer live")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--name", default="fk_sweep")
    args = parser.parse_args()

    params = default_params()
    centre = np.array([0.20, 0.0, 0.10])
    radius = 0.06

    q = np.array([0.0, 0.6, -1.2])
    missed = 0

    with Telemetry(args.name, params, spawn=args.spawn) as log:
        for step in range(args.steps):
            t = step / args.steps
            angle = 2.0 * np.pi * t

            target = centre + radius * np.array([0.0, np.cos(angle), np.sin(angle)])

            solution = ik_nearest(target, q, params)
            if solution is None:
                missed += 1
                continue
            q = solution.q

            achieved = fk(q, params)
            log.at(t)
            log.log_arm(q)
            log.log_scalars(
                det_jacobian=jacobian_det(q, params),
                manipulability=manipulability(q, params),
                tracking_error_m=float(np.linalg.norm(achieved - target)),
            )
            log.row(
                sim_time=t,
                q=q,
                target=target,
                achieved=achieved,
                det_jacobian=jacobian_det(q, params),
                manipulability=manipulability(q, params),
                tracking_error_m=float(np.linalg.norm(achieved - target)),
            )

        print(f"wrote {log.rrd_path}")
        print(f"wrote {log.table_path}")

    if missed:
        print(f"WARNING: {missed}/{args.steps} waypoints had no in-limit IK solution")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
