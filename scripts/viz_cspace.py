#!/usr/bin/env python3
"""Draw configuration space, and show why a straight line is not a path.

Two views of the same thing, side by side in Rerun:

* the arm in the world, where obstacles are obvious and simple;
* configuration space, where the same obstacles become a curved volume and the
  arm is a single point.

The straight line between two valid configurations is drawn through both. In
the world view it looks unremarkable. In C-space you can see it pass straight
through the blocked region -- which is the whole argument for planning.

Only possible because this arm has three joints. At six the C-space is
six-dimensional and cannot be drawn at all.

    just cspace
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from arm.cspace import CollisionChecker, occupancy_grid
from arm.kinematics import fk_frames
from arm.params import default_params
from arm.telemetry import RUNS_DIR

# Reaching to either side of the post: bearing 65 and 115 degrees, r = 0.17 m,
# z = 0.16 m. Both configurations are valid, and half the straight line between
# them is not.
START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


def log_arm_at(path: str, q: np.ndarray, params, colour) -> None:
    """Draw the arm as a skeleton at one configuration, in the world view."""
    points = np.array([frame[:3, 3] for frame in fk_frames(q, params)])
    rr.log(path, rr.LineStrips3D([points], radii=0.004, colors=[colour]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=44)
    parser.add_argument("--spawn", action="store_true", help="open the viewer live")
    args = parser.parse_args()

    params = default_params()
    checker = CollisionChecker(params)

    for name, q in (("start", START), ("goal", GOAL)):
        if not checker.is_valid(q):
            raise SystemExit(f"{name} configuration {q} is not valid")

    print(
        f"building a {args.resolution}^3 occupancy grid "
        f"({args.resolution**3:,} collision checks) ..."
    )
    grid = occupancy_grid(args.resolution, params)
    print(f"  {grid.blocked_fraction * 100:.1f}% of configuration space is blocked")

    rr.init("learn_robotics_cspace", spawn=args.spawn)
    if not args.spawn:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        rr.save(str(RUNS_DIR / "cspace.rrd"))

    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="the world", origin="/world"),
            rrb.Spatial3DView(name="configuration space", origin="/cspace"),
        )
    )

    # -- configuration space ------------------------------------------------
    blocked = grid.blocked_points()
    rr.log(
        "cspace/blocked",
        rr.Points3D(blocked, radii=0.012, colors=[200, 70, 70]),
        static=True,
    )

    # The straight line between start and goal, in C-space.
    samples = np.linspace(0.0, 1.0, 200)
    line = np.array([START + (GOAL - START) * s for s in samples])
    valid = np.array([checker.is_valid(q) for q in line])

    rr.log(
        "cspace/straight_line",
        rr.LineStrips3D([line], radii=0.006, colors=[80, 160, 255]),
        static=True,
    )
    if not valid.all():
        rr.log(
            "cspace/straight_line_blocked",
            rr.Points3D(line[~valid], radii=0.018, colors=[255, 210, 60]),
            static=True,
        )
    rr.log(
        "cspace/endpoints",
        rr.Points3D(np.array([START, GOAL]), radii=0.03, colors=[[90, 220, 120], [255, 140, 40]]),
        static=True,
    )

    # -- the world, swept along that same line ------------------------------
    log_arm_at("world/start", START, params, [90, 220, 120])
    log_arm_at("world/goal", GOAL, params, [255, 140, 40])
    rr.log(
        "world/floor",
        rr.Boxes3D(centers=[[0.0, 0.0, -0.005]], half_sizes=[[0.5, 0.5, 0.005]]),
        static=True,
    )
    if params.obstacles:
        rr.log(
            "world/obstacles",
            rr.Boxes3D(
                centers=[o.pos_m for o in params.obstacles],
                half_sizes=[o.half_size_m for o in params.obstacles],
                colors=[[150, 110, 90]],
            ),
            static=True,
        )

    for index, (q, ok) in enumerate(zip(line, valid, strict=True)):
        rr.set_time("step", sequence=index)
        log_arm_at("world/arm", q, params, [90, 220, 120] if ok else [220, 60, 60])
        rr.log("cspace/cursor", rr.Points3D([q], radii=0.025, colors=[255, 255, 255]))

    blocked_count = int((~valid).sum())
    print(
        f"\nthe straight line from start to goal passes through "
        f"{blocked_count}/{len(line)} blocked configurations"
    )
    if blocked_count:
        print("so it is not a path, and something has to search for one instead.")
    else:
        print("this particular pair happens to be directly connectable.")

    stream = rr.get_global_data_recording()
    if stream is not None:
        stream.flush()
    if not args.spawn:
        print(f"\nwrote {RUNS_DIR / 'cspace.rrd'}   (open with: just view cspace)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
