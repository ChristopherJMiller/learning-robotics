#!/usr/bin/env python3
"""Track a moving trajectory and compare controllers honestly.

Three failures had to be fixed before this comparison meant anything, and each
one looked exactly like a badly tuned controller:

1. **Gains that do not transfer.** ``kp`` is N·m/rad for PD and 1/s² for
   computed torque. Passing one as the other gave a damping ratio of 0.11.
2. **A model mismatch.** The MJCF carried armature and damping the URDF did
   not, so the controller inverted a robot whose elbow inertia was 19x too
   small.
3. **An infeasible trajectory.** The original path drove the arm 72 mm through
   the floor; the contact solver -- correctly -- refused to follow it, which
   produced a large, gain-independent error.

Hence the feasibility check before anything runs. Tuning cannot fix any of the
three, so it is worth ruling them out first.

    just track
    just track --hz 1.25
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from arm.control import (
    AccelGains,
    Integrator,
    JointGains,
    computed_torque,
    feedforward_pd,
    pd_torque,
    pd_with_gravity_compensation,
)
from arm.kinematics import fk
from arm.params import default_params
from arm.sim import MujocoArm
from arm.telemetry import Telemetry
from arm.trajectory import check_feasibility, sinusoid

CENTRE = np.array([0.0, 0.9, -0.7])
AMPLITUDE = np.array([0.6, 0.30, 0.35])


def run(name: str, seconds: float, hz: float, *, spawn: bool = False, viewer: bool = False):
    params = default_params()
    arm = MujocoArm(params)

    joint_gains = JointGains.uniform(8.0, 0.6)
    accel_gains = AccelGains.from_spec(bandwidth_hz=3.0, damping_ratio=1.0)
    integrator = Integrator(ki=2.0, limit=0.15)

    start = sinusoid(0.0, CENTRE, AMPLITUDE, hz)
    arm.reset(start.q, start.dq)

    errors: list[float] = []
    saturated = 0
    refused = np.zeros(3)

    handle = arm.launch_viewer() if viewer else None
    with Telemetry(f"track_{name}", params, spawn=spawn) as log:
        for _ in range(int(seconds / arm.dt)):
            state = arm.read()
            target = sinusoid(state.t, CENTRE, AMPLITUDE, hz)

            if name == "pd":
                tau = pd_torque(state, target.q, joint_gains, target.dq)
            elif name == "pd_gravity":
                tau = pd_with_gravity_compensation(state, target.q, joint_gains, target.dq)
            elif name == "feedforward_pd":
                tau = feedforward_pd(state, target, joint_gains, integrator, arm.dt, refused)
            else:
                tau = computed_torque(state, target, accel_gains)

            arm.write_torque(tau)
            refused = arm.refused_torque
            saturated += int(arm.is_saturated)
            arm.step()

            if handle is not None:
                handle.sync()
                time.sleep(arm.dt)  # play at wall-clock speed so it is watchable

            error = float(np.linalg.norm(fk(state.q, params) - fk(target.q, params)))
            errors.append(error)

            log.at(state.t)
            log.log_arm(state.q)
            log.log_target(fk(target.q, params))
            log.log_scalars(tip_error_m=error, saturated=float(arm.is_saturated))
            log.row(
                sim_time=state.t,
                q=state.q,
                q_desired=target.q,
                tau=state.tau,
                tip_error_m=error,
            )

    if handle is not None:
        handle.close()

    settle = int(0.5 / hz / arm.dt)  # discard the first half cycle
    tail = np.array(errors[settle:])
    return float(np.sqrt(np.mean(tail**2))), float(np.max(tail)), saturated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--spawn", action="store_true", help="live Rerun viewer")
    parser.add_argument("--viewer", action="store_true", help="MuJoCo's own physics viewer")
    parser.add_argument("--only", help="run a single controller instead of all four")
    args = parser.parse_args()

    samples = [
        sinusoid(t, CENTRE, AMPLITUDE, args.hz) for t in np.linspace(0.0, 1.0 / args.hz, 240)
    ]
    report = check_feasibility(samples)
    print(f"trajectory: {report.describe()}")
    if not report.feasible:
        print("\nrefusing to run: no controller can follow an infeasible path.")
        return 1

    joint_gains = JointGains.uniform(8.0, 0.6)
    accel_gains = AccelGains.from_spec(bandwidth_hz=3.0, damping_ratio=1.0)
    print(f"joint gains : {joint_gains.describe()}")
    print(
        f"accel gains : wn = {np.round(accel_gains.natural_frequency_rad_s, 1)} rad/s, "
        f"zeta = {np.round(accel_gains.damping_ratio, 2)}\n"
    )

    print(f"{'controller':<18} {'RMS tip error':>15} {'peak':>10} {'saturated':>11}")
    print("-" * 57)
    names = (args.only,) if args.only else ("pd", "pd_gravity", "feedforward_pd", "computed_torque")
    results = {}
    for name in names:
        rms, peak, saturated = run(
            name, args.seconds, args.hz, spawn=args.spawn, viewer=args.viewer
        )
        results[name] = rms
        print(f"{name:<18} {rms * 1000:>12.2f} mm {peak * 1000:>7.2f} mm {saturated:>11}")

    if len(names) < 4:
        return 0

    print(
        f"\nGravity compensation alone cuts error {results['pd'] / results['pd_gravity']:.1f}x."
        f"\nAdding inverse-dynamics feedforward cuts it a further "
        f"{results['pd_gravity'] / results['feedforward_pd']:.1f}x."
    )
    print(
        "\nfeedforward_pd and computed_torque land close together here, and that is"
        "\nthe expected result: reflected rotor inertia makes M(q) vary only 1.5x"
        "\nacross this workspace, so there is little configuration dependence left"
        "\nfor computed torque to cancel. On a direct-drive arm M varies 25x and"
        "\nthe ordering reverses."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
