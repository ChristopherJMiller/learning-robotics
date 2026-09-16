#!/usr/bin/env python3
"""Cartesian impedance control: choose a stiffness instead of demanding a pose.

A position controller answers a disturbance by increasing torque until it wins
or something breaks. An impedance controller behaves like a spring: push it and
it yields by a predictable amount, pushing back with a bounded force you chose.

    tau = J^T (K (x_d - x) + D (dx_d - dx)) + g(q)

At equilibrium, with gravity cancelled and the arm at rest, the joint torques
balance:

    J^T K (x_d - x) + J^T F_ext = 0     =>     deflection = -F_ext / K

so the deflection under a known tip force is a direct measurement of the
stiffness actually realised. That is what this demo checks: command a stiffness,
push with a known force, and confirm the arm yields by exactly F/K.

The external force is injected as ``J^T F_ext`` -- joint torques equivalent to
pushing on the tip, which is what a hand or an obstacle does.

    just impedance
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.control import JointGains, cartesian_impedance, pd_with_gravity_compensation
from arm.kinematics import fk, jacobian
from arm.params import default_params
from arm.sim import MujocoArm
from arm.telemetry import Telemetry

HOLD_Q = np.array([0.0, 0.9, -0.9])


def settle(arm, controller, seconds: float, force: np.ndarray, params) -> np.ndarray:
    """Run to equilibrium under a constant tip force; return the final tip position."""
    for _ in range(int(seconds / arm.dt)):
        state = arm.read()
        tau = controller(state)
        # An external push on the tip, expressed as joint torques.
        external = jacobian(state.q, params).T @ force
        arm.write_torque(tau + external)
        arm.step()
    return fk(arm.read().q, params)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", type=float, default=2.0, help="tip push, newtons")
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    params = default_params()
    target = fk(HOLD_Q, params)
    push = np.array([0.0, -args.force, 0.0])  # sideways, well away from singularity

    print(f"holding the tip at {np.round(target, 4)} m")
    print(f"pushing sideways with {args.force} N\n")
    print(f"{'controller':<28} {'deflection':>12} {'predicted':>11} {'peak torque':>13}")
    print("-" * 68)

    # 1. Stiff joint-space position control: fights the push, no chosen compliance.
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    gains = JointGains.uniform(20.0, 1.5)
    peak = np.zeros(3)

    def stiff(state):
        nonlocal peak
        tau = pd_with_gravity_compensation(state, HOLD_Q, gains)
        peak = np.maximum(peak, np.abs(tau))
        return tau

    final = settle(arm, stiff, args.seconds, push, params)
    print(
        f"{'joint PD + gravity comp':<28} "
        f"{np.linalg.norm(final - target) * 1000:>9.2f} mm {'--':>11} "
        f"{np.max(peak):>10.3f} Nm"
    )

    # 2. Cartesian impedance at a range of commanded stiffnesses.
    for stiffness in (50.0, 100.0, 200.0, 400.0):
        arm = MujocoArm(params)
        arm.reset(HOLD_Q)
        peak = np.zeros(3)
        damping = 2.0 * np.sqrt(stiffness * 0.3)  # ~critical for ~0.3 kg apparent mass

        def impedance(state, k=stiffness, d=damping):
            nonlocal peak
            tau = cartesian_impedance(state, target, k, d, params=params)
            peak = np.maximum(peak, np.abs(tau))
            return tau

        with Telemetry(f"impedance_k{int(stiffness)}", params) as log:
            for _ in range(int(args.seconds / arm.dt)):
                state = arm.read()
                external = jacobian(state.q, params).T @ push
                arm.write_torque(impedance(state) + external)
                arm.step()
                log.at(state.t)
                log.log_arm(state.q)
                log.log_target(target)
                log.row(sim_time=state.t, q=state.q, tau=state.tau)

        final = fk(arm.read().q, params)
        measured = np.linalg.norm(final - target)
        predicted = args.force / stiffness
        print(
            f"{f'impedance K={stiffness:.0f} N/m':<28} "
            f"{measured * 1000:>9.2f} mm {predicted * 1000:>8.2f} mm "
            f"{np.max(peak):>10.3f} Nm"
        )

    print(
        "\nThe deflection tracks F/K because that is what stiffness means. Position"
        "\ncontrol has no such number: its compliance is an accident of the gains,"
        "\nthe configuration, and the inertia -- not something you chose."
    )
    print(
        "\nThis is what makes contact safe. A stiff controller meeting an obstacle"
        "\nescalates torque until it wins or breaks; an impedance controller pushes"
        "\nwith a force you specified in newtons and stops there."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
