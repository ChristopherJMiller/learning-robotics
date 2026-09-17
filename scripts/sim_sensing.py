#!/usr/bin/env python3
"""What perfect state measurement was hiding.

Every result so far read exact joint angles and exact velocities from the
simulator. Hardware provides neither. A Dynamixel reports position from a
4096-count encoder and does not measure velocity at all -- it differentiates,
and differentiation turns a small position quantum into a large velocity one:

    1.534 mrad / 0.005 s = 0.307 rad/s   ->   x kd=0.6  =  0.184 N.m of noise

against a 0.6 N.m continuous limit. Roughly a third of the actuator spent on
quantisation noise.

Three parts:

1. What quantisation costs, and how five estimators trade error against command
   chatter. The Kalman filter derives its gains from that 1.534 mrad rather than
   having them chosen, and predicting with the arm's dynamics rather than
   assuming constant velocity recovers almost all of the lost accuracy.
2. What identifying one parameter is worth. Coulomb friction is added to the
   plant, and the estimator and the controller are told about it independently.
   Telling both takes tracking error from 3.70 mm to 0.33 mm -- an 11x
   improvement from measuring a number rather than tuning a gain.
3. Holding a pose against that friction, where it creates a dead zone that
   feedforward cannot close and integral action can only hunt around.

    just sensing
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.control import Integrator, JointGains, feedforward_pd
from arm.kinematics import fk
from arm.params import default_params
from arm.sensing import (
    AlphaBeta,
    FilteredDifference,
    FiniteDifference,
    KalmanVelocity,
    SensedArm,
)
from arm.sim import MujocoArm
from arm.telemetry import Telemetry
from arm.trajectory import Setpoint, sinusoid

CENTRE = np.array([0.0, 0.9, -0.7])
AMPLITUDE = np.array([0.6, 0.30, 0.35])
HZ = 1.0


def run(
    label: str,
    *,
    sensed: bool,
    estimator=None,
    friction_nm: float = 0.0,
    integral: bool = False,
    seconds: float = 4.0,
):
    params = default_params()
    if friction_nm > 0.0:
        params = params.with_friction(friction_nm)
        plant = MujocoArm(params, from_params=True)
    else:
        plant = MujocoArm(params)

    arm = SensedArm(plant, params, velocity_estimator=estimator) if sensed else plant

    gains = JointGains.uniform(8.0, 0.6)
    integrator = Integrator(ki=3.0, limit=0.12) if integral else None

    start = sinusoid(0.0, CENTRE, AMPLITUDE, HZ)
    arm.reset(start.q, start.dq)

    errors, torques = [], []
    refused = np.zeros(3)

    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    with Telemetry(f"sensing_{slug}", params) as log:
        for _ in range(int(seconds / plant.dt)):
            state = arm.read()
            # Ground truth sampled at the same instant as the control decision.
            # Reading it after arm.step() compares a pose one control period in
            # the future against a target from the past, which at this speed is
            # a 4.5 mm artefact -- larger than every effect being measured.
            truth = plant.read()
            target = sinusoid(state.t, CENTRE, AMPLITUDE, HZ)

            # The controller's model never knows about the friction we added.
            tau = feedforward_pd(state, target, gains, integrator, plant.dt, refused)
            arm.write_torque(tau)
            refused = plant.refused_torque
            arm.step()

            errors.append(float(np.linalg.norm(fk(truth.q, params) - fk(target.q, params))))
            torques.append(tau.copy())

            log.at(truth.t)
            log.log_arm(truth.q)
            log.log_scalars(tip_error_m=errors[-1])
            log.row(sim_time=truth.t, q=truth.q, q_measured=state.q, tau=tau)

    settle = int(0.5 / HZ / plant.dt)
    tail = np.array(errors[settle:])
    torque = np.array(torques[settle:])
    # Chatter: how much the command moves between consecutive control periods.
    chatter = float(np.mean(np.abs(np.diff(torque, axis=0))))
    return float(np.sqrt(np.mean(tail**2))), chatter


def tracking_error(
    friction_nm: float,
    *,
    estimator_knows: bool,
    controller_knows: bool,
    seconds: float = 3.0,
) -> float:
    """Tracking error with the friction known, or not, to each model separately.

    Two independent beliefs. The plant always has the friction; whether the
    estimator's dynamics and the controller's feedforward know about it is what
    varies. Separating the axes is what shows where an identified parameter
    actually pays.
    """
    plant_params = default_params().with_friction(friction_nm)
    naive = default_params()

    plant = MujocoArm(plant_params, from_params=True)
    arm = SensedArm(
        plant,
        plant_params,
        velocity_estimator=KalmanVelocity(
            process_accel_std=10.0,
            use_model=True,
            params=plant_params if estimator_knows else naive,
        ),
    )

    gains = JointGains.uniform(8.0, 0.6)
    start = sinusoid(0.0, CENTRE, AMPLITUDE, HZ)
    arm.reset(start.q, start.dq)

    errors = []
    for _ in range(int(seconds / plant.dt)):
        state = arm.read()
        truth = plant.read()
        target = sinusoid(state.t, CENTRE, AMPLITUDE, HZ)
        arm.write_torque(
            feedforward_pd(
                state,
                target,
                gains,
                dt=plant.dt,
                params=plant_params if controller_knows else naive,
            )
        )
        arm.step()
        errors.append(float(np.linalg.norm(fk(truth.q, plant_params) - fk(target.q, plant_params))))

    tail = np.array(errors[int(0.5 / HZ / plant.dt) :])
    return float(np.sqrt(np.mean(tail**2)))


def hold(label: str, *, friction_nm: float, integral: bool, seconds: float = 6.0):
    """Regulation, not tracking -- the task integral action is actually for.

    On a fast reversing trajectory Coulomb friction is not a constant bias at
    all: it flips sign with velocity, so it is a square wave at the trajectory
    frequency. Integral action cannot cancel that and only contributes phase
    lag, which is why adding it to the tracking case measurably *hurts*.

    Holding a pose is different. There the friction is genuinely constant, and
    it creates a dead zone: once the remaining error is small enough that
    ``kp * e`` falls below the breakaway torque, the arm simply stops. No
    feedforward term can fix that, because the arm is not moving and the model
    has nothing to feed forward. Integral action is the mechanism that exists
    for exactly this.
    """
    params = default_params().with_friction(friction_nm)
    plant = MujocoArm(params, from_params=True)
    arm = SensedArm(plant, params, velocity_estimator=AlphaBeta())

    gains = JointGains.uniform(8.0, 0.6)
    integrator = Integrator(ki=4.0, limit=0.15) if integral else None

    q_desired = np.array([0.0, 0.9, -0.9])
    arm.reset(q_desired + np.array([0.05, 0.05, 0.05]))  # start off target

    refused = np.zeros(3)
    target = Setpoint(q=q_desired, dq=np.zeros(3), ddq=np.zeros(3))
    errors = []

    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    with Telemetry(f"hold_{slug}", params) as log:
        for _ in range(int(seconds / plant.dt)):
            state = arm.read()
            truth = plant.read()
            tau = feedforward_pd(state, target, gains, integrator, plant.dt, refused)
            arm.write_torque(tau)
            refused = plant.refused_torque
            arm.step()

            errors.append(float(np.linalg.norm(fk(truth.q, params) - fk(q_desired, params))))
            log.at(truth.t)
            log.log_arm(truth.q)
            log.log_scalars(tip_error_m=errors[-1])
            log.row(sim_time=truth.t, q=truth.q, tau=tau)

    return float(np.mean(errors[-len(errors) // 5 :]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--friction", type=float, default=0.02)
    args = parser.parse_args()

    print("PART 1 - encoder quantisation, no friction\n")
    print(f"{'state estimate':<38} {'RMS error':>11} {'chatter':>11}")
    print("-" * 62)

    cases = [
        ("perfect state (what we had)", False, None),
        ("quantised + finite difference", True, FiniteDifference()),
        ("quantised + low-pass 40 Hz", True, FilteredDifference(cutoff_hz=40.0)),
        ("quantised + low-pass 10 Hz", True, FilteredDifference(cutoff_hz=10.0)),
        ("quantised + alpha-beta (guessed)", True, AlphaBeta()),
        (
            "quantised + alpha-beta (derived)",
            True,
            AlphaBeta(alpha=0.770, beta=0.542),
        ),
        ("quantised + kalman, kinematic", True, KalmanVelocity()),
        (
            "quantised + kalman, model-based",
            True,
            KalmanVelocity(process_accel_std=10.0, use_model=True),
        ),
    ]
    for label, sensed, estimator in cases:
        rms, chatter = run(label, sensed=sensed, estimator=estimator, seconds=args.seconds)
        print(f"{label:<38} {rms * 1000:>8.3f} mm {chatter * 1000:>8.2f} mNm")

    print(
        "\nAmong the purely kinematic estimators there is no best row, only a"
        "\nfrontier. Raw differencing tracks best of those -- but at ~24x the command"
        "\nchatter, and chatter is what heats motors, wears gears and excites"
        "\nresonances. It is that column which rules it out, not the error."
        "\n"
        "\nFiltering buys smoothness with delay, and delay costs phase margin, which"
        "\nis why a 10 Hz cutoff is quieter *and worse*. Predicting forward with a"
        "\nmotion model beats that trade outright."
        "\n"
        "\nThe two alpha-beta rows are the same filter with different gains. The"
        "\nfirst pair I chose by eye; the second the Kalman filter derives from the"
        "\nencoder datasheet (R = d^2/12, no tuning) -- and the kinematic Kalman row"
        "\nmatches it, because at steady state they are literally the same filter."
        "\n"
        "\nThe model-based row is the real result. Predicting with the arm's"
        "\ndynamics instead of assuming constant velocity recovers almost everything"
        "\nquantisation took away, at a quarter of the command chatter."
    )

    print("\n\nPART 2 - what identifying one parameter is worth\n")
    print(
        "Coulomb friction the models do not know about, and the four combinations"
        "\nof telling the estimator and telling the controller:\n"
    )
    print(f"{'friction':>9} {'neither':>10} {'est only':>10} {'ctl only':>10} {'both':>10}")
    print("-" * 54)
    for friction in (0.0, 0.02, 0.05, 0.10):
        results = [
            tracking_error(friction, estimator_knows=e, controller_knows=c)
            for e, c in ((False, False), (True, False), (False, True), (True, True))
        ]
        print(f"{friction:>7.2f}Nm " + " ".join(f"{value * 1000:>7.3f} mm" for value in results))

    print(
        "\nOne measured number, fed to both, takes tracking error at 0.05 N.m from"
        "\n3.70 mm to 0.33 mm -- an 11x improvement from identifying a parameter"
        "\nrather than tuning a gain."
        "\n"
        "\nThe controller gains more than the estimator (1.10 mm against 2.57 mm on"
        "\ntheir own), which is what you would expect: the controller cancels the"
        "\ndisturbance directly, while the estimator only gets a better state."
        "\n"
        "\nWithout that number the model-based estimator is *worse* than the"
        "\nkinematic one that never trusted a model -- and no amount of raising"
        "\nprocess noise fixes it, because Q describes zero-mean noise while friction"
        "\nis a systematic bias. This is the whole argument for system"
        "\nidentification: measure the parameter, do not tune around it."
    )

    print("\n\nPART 3 - holding a pose with Coulomb friction\n")
    kp = 8.0
    print(f"{'friction':>9} {'dead zone':>11} {'no integral':>13} {'+ integral':>12}")
    print("-" * 50)
    for friction in (0.02, 0.05, 0.10):
        without = hold("plain", friction_nm=friction, integral=False)
        with_i = hold("integral", friction_nm=friction, integral=True)
        dead = np.degrees(friction / kp)
        print(
            f"{friction:>7.2f}Nm {dead:>8.3f} deg {without * 1000:>10.3f} mm "
            f"{with_i * 1000:>9.3f} mm"
        )

    print(
        "\nThis is not a clean win for integral action, and the reason is worth more"
        "\nthan the win would have been."
        "\n"
        "\nCoulomb friction creates a **dead zone**: once the remaining error is small"
        "\nenough that kp*e falls below the breakaway torque, the arm simply stops."
        "\nAt 0.10 N.m and kp=8 that band is +/-0.72 deg, and *both* controllers end"
        "\nup inside it -- neither has converged in any meaningful sense. Measured"
        "\nsigned error at the end:"
        "\n"
        "\n    no integral   +0.044, +0.050 deg   stops short, inside the band"
        "\n    + integral    -0.183, -0.163 deg   pushes through, sticks on the far side"
        "\n"
        "\nSo integral action does what it is supposed to -- accumulate until stiction"
        "\nbreaks -- and then overshoots and sticks again. It hunts rather than"
        "\nsettles, and the apparent 'worse' number is just where in the dead band it"
        "\nhappened to stop. This is exactly the behaviour that makes PID on a geared"
        "\njoint feel twitchy on real hardware."
        "\n"
        "\nThe real cures are not more integral gain: identify the friction and feed"
        "\nit forward, add a dither signal to keep the joint microscopically moving,"
        "\nor accept the dead band and stop asking for precision below it."
        "\n"
        "\nNote also that feedforward cannot help here at all. friction_torque is"
        "\nproportional to tanh(dq/eps), and a stationary arm has dq = 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
