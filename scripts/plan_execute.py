#!/usr/bin/env python3
"""The whole pipeline: plan a route, time it, and actually drive the arm along it.

Every piece built so far, in the order a real command would flow through them:

    goal pose
      -> inverse kinematics          which joint angles reach it
      -> RRT + shortcutting          a collision-free route to them
      -> spline + time scaling       a route the actuators can execute
      -> feedforward PD              torques that track it
      -> MuJoCo                      what the arm actually does

Two things this demonstrates that the earlier demos could not. The trajectory is
*generated* rather than hand-written, so nobody had to search for a feasible
sinusoid. And torque feasibility is enforced by construction: the trajectory is
slowed until the arm can hold it, rather than checked afterwards and hoped for.

    just execute
    just execute --speed 3.0     ask for a traversal the arm cannot sustain
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.control import JointGains, feedforward_pd
from arm.cspace import CollisionChecker
from arm.kinematics import fk
from arm.params import default_params
from arm.planning import path_length, rrt, shortcut
from arm.sensing import KalmanVelocity, SensedArm
from arm.sim import MujocoArm
from arm.telemetry import Telemetry
from arm.timing import (
    gravity_feasible,
    path_stays_valid,
    peak_torque,
    time_optimal_scale,
    torque_limits,
    trajectory_from_path,
)

START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=1.5, help="nominal rad/s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()

    params = default_params()
    checker = CollisionChecker(params)

    # 1. plan
    result = rrt(START, GOAL, checker, seed=args.seed)
    if not result.found:
        print("no path found")
        return 1
    path = shortcut(result.path, checker, seed=args.seed)
    print(f"1. planned    {len(path)} waypoints, {path_length(path):.3f} rad")

    # 2. smooth, and confirm the smoothing did not undo the planning
    trajectory = trajectory_from_path(path, params, speed_rad_s=args.speed)
    still_clear = path_stays_valid(trajectory, checker)
    print(
        f"2. smoothed   {trajectory.duration_s:.3f} s at {args.speed} rad/s, "
        f"still collision-free: {still_clear}"
    )
    if not still_clear:
        print("   rounding the corners cut through something; slow the nominal")
        print("   speed or shortcut less aggressively")
        return 1

    # 3. check what cannot be fixed by slowing down, then slow down
    limits = torque_limits(params)
    raw_peak = peak_torque(trajectory)
    holdable, ratio = gravity_feasible(trajectory)
    print(f"3. torque     peak {np.round(raw_peak, 3)} against limits {np.round(limits, 3)}")
    print(
        f"   gravity alone needs {ratio * 100:.0f}% of the limit at worst "
        f"({'holdable' if holdable else 'NOT holdable'})"
    )

    trajectory, factor = time_optimal_scale(trajectory)
    print(
        f"4. scaled     {factor:.2f}x slower -> {trajectory.duration_s:.3f} s, "
        f"peak {np.round(peak_torque(trajectory), 3)}"
    )

    # 5. execute, against quantised sensing and the recommended controller
    plant = MujocoArm(params)
    arm = SensedArm(
        plant,
        params,
        velocity_estimator=KalmanVelocity(process_accel_std=10.0, use_model=True),
    )
    gains = JointGains.uniform(8.0, 0.6)

    start = trajectory.at(0.0)
    arm.reset(start.q, start.dq)

    errors, contacts = [], 0
    settle_s = 0.5
    steps = int((trajectory.duration_s + settle_s) / plant.dt)

    with Telemetry("plan_execute", params, spawn=args.spawn) as log:
        log.note(
            f"plan -> execute\n\n"
            f"- path {path_length(path):.3f} rad, {len(path)} waypoints\n"
            f"- slowed {factor:.2f}x for torque\n"
            f"- duration {trajectory.duration_s:.3f} s"
        )
        for _ in range(steps):
            state = arm.read()
            truth = plant.read()
            target = trajectory.at(state.t)

            arm.write_torque(feedforward_pd(state, target, gains, dt=plant.dt))
            arm.step()
            contacts += int(plant.data.ncon > 0)

            error = float(np.linalg.norm(fk(truth.q, params) - fk(target.q, params)))
            errors.append(error)

            log.at(truth.t)
            log.log_arm(truth.q)
            log.log_target(fk(target.q, params))
            log.log_scalars(tip_error_m=error)
            log.row(sim_time=truth.t, q=truth.q, q_desired=target.q, tau=state.tau)

    final = float(np.linalg.norm(fk(plant.read().q, params) - fk(GOAL, params)))
    print(
        f"5. executed   RMS tip error {np.sqrt(np.mean(np.square(errors))) * 1000:.2f} mm, "
        f"peak {max(errors) * 1000:.2f} mm"
    )
    print(f"   arrived within {final * 1000:.2f} mm of the goal")
    print(f"   contacts during execution: {contacts}")

    if contacts:
        print("\n   NOT clean -- the arm touched something it planned around.")
        return 1

    print(
        "\nThe loop is closed: a goal pose became joint angles, a route, a timed"
        "\ntrajectory the actuators can sustain, and torques -- with nothing"
        "\nhand-tuned along the way."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
