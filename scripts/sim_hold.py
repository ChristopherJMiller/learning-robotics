#!/usr/bin/env python3
"""Hold a pose against gravity, three ways, and measure the difference.

The point of this demo is the *structural* failure of pure PD control. A PD
controller produces a restoring torque proportional to error, so to balance a
gravity load it must first permit enough error to generate that torque. The
resulting droop is not a tuning problem -- raising the gains shrinks it but can
never remove it, and high gains make the arm stiff and jittery.

Feeding gravity forward removes the cause instead of fighting the symptom.

    just hold                write runs/hold_*.{rrd,parquet} and print a table
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.control import JointGains, pd_torque, pd_with_gravity_compensation
from arm.kinematics import fk
from arm.params import default_params
from arm.sim import MujocoArm
from arm.telemetry import Telemetry

# Only the two joint-space position controllers are compared here. Computed
# torque and feedforward both collapse to gravity compensation when the target
# is stationary -- a held pose asks for zero acceleration -- so they belong in
# the tracking demo instead. See scripts/sim_track.py.
CONTROLLERS = {
    "pd": pd_torque,
    "pd_gravity": pd_with_gravity_compensation,
}


def run(name: str, gains: JointGains, q_desired: np.ndarray, seconds: float, *, spawn=False):
    params = default_params()
    arm = MujocoArm(params)
    arm.reset(q_desired)  # start exactly at the target; only gravity disturbs it

    controller = CONTROLLERS[name]
    steps = int(seconds / arm.dt)
    errors: list[float] = []

    with Telemetry(f"hold_{name}", params, spawn=spawn) as log:
        for _ in range(steps):
            state = arm.read()
            arm.write_torque(controller(state, q_desired, gains))
            arm.step()

            joint_error = float(np.linalg.norm(state.q - q_desired))
            tip_error = float(np.linalg.norm(fk(state.q, params) - fk(q_desired, params)))
            errors.append(tip_error)

            log.at(state.t)
            log.log_arm(state.q)
            log.log_scalars(joint_error_rad=joint_error, tip_error_m=tip_error)
            log.row(
                sim_time=state.t,
                q=state.q,
                q_desired=q_desired,
                tau=state.tau,
                joint_error_rad=joint_error,
                tip_error_m=tip_error,
            )

    settled = errors[-len(errors) // 4 :]  # final quarter, after transients
    return float(np.mean(settled)), float(np.max(np.abs(arm.read().tau)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--kp", type=float, default=8.0)
    parser.add_argument("--kd", type=float, default=0.6)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()

    # Arm held out horizontally -- the worst case for gravity, and the pose
    # where the difference between the controllers is most visible.
    q_desired = np.array([0.0, 0.0, 0.0])
    gains = JointGains.uniform(args.kp, args.kd)

    print(
        f"holding q = {np.round(np.degrees(q_desired), 1)} deg "
        f"(arm horizontal), kp={args.kp}, kd={args.kd}\n"
    )
    print(f"{'controller':<18} {'steady tip error':>18} {'peak torque':>14}")
    print("-" * 52)

    results = {}
    for name in CONTROLLERS:
        error, torque = run(name, gains, q_desired, args.seconds, spawn=args.spawn)
        results[name] = error
        millimetres = error * 1000.0
        shown = f"{millimetres:>15.3f} mm" if millimetres >= 1e-3 else f"{millimetres:>15.2e} mm"
        print(f"{name:<18} {shown} {torque:>12.3f} Nm")

    print(
        f"\nPD droops {results['pd'] * 1000:.2f} mm. Feeding gravity forward leaves "
        f"{results['pd_gravity'] * 1000:.1e} mm,"
        "\nwhich is integrator noise rather than a tuning result."
    )
    print(
        "\nPeak torque is identical for both: holding against gravity costs what it"
        "\ncosts. What differs is whether the arm has to be *wrong* in order to"
        "\nproduce it -- PD's only source of torque is kp*e, so e_ss = g(q)/kp."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
