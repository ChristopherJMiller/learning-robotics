"""Joint-space trajectory generation.

A controller needs more than a target position. Position alone tells it where
to go but nothing about how fast it should already be moving when it gets
there, which is why step commands produce overshoot and jerk.

Each generator here returns the full triple ``(q, q̇, q̈)``:

* ``q``  — where to be
* ``q̇``  — feed-forward velocity, so the derivative term is not fighting the motion
* ``q̈``  — feed-forward acceleration, which is what ``computed_torque`` needs to
  ask the arm for a specific acceleration rather than inferring one from error

Without ``q̈`` a computed-torque controller degenerates into gravity-compensated
PD, which is exactly why the hold demo showed no difference between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Setpoint:
    """A complete instantaneous command."""

    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


def quintic(t: float, duration: float, start: np.ndarray, end: np.ndarray) -> Setpoint:
    """Minimum-jerk point-to-point motion.

    A fifth-order polynomial is the lowest order that can satisfy six boundary
    conditions -- position, velocity and acceleration at both ends -- so the
    motion starts and finishes at rest *and* with zero acceleration. That
    second property is what stops the arm jolting at the endpoints.

    Time is clamped, so evaluating past ``duration`` holds the final pose.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    s = float(np.clip(t / duration, 0.0, 1.0))
    delta = end - start

    # s(τ) = 10τ³ − 15τ⁴ + 6τ⁵, chosen so s(0)=0, s(1)=1, s'=s''=0 at both ends
    position = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    velocity = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration
    acceleration = (60.0 * s - 180.0 * s**2 + 120.0 * s**3) / duration**2

    if s >= 1.0:  # held past the end: no residual velocity or acceleration
        velocity = acceleration = 0.0

    return Setpoint(
        q=start + delta * position,
        dq=delta * velocity,
        ddq=delta * acceleration,
    )


def sinusoid(
    t: float,
    centre: np.ndarray,
    amplitude: np.ndarray,
    frequency_hz: float,
) -> Setpoint:
    """Continuous oscillation, useful for exercising the dynamics.

    Acceleration scales with the square of frequency, so raising
    ``frequency_hz`` is the direct way to make inertial and Coriolis terms
    dominate gravity -- which is the regime where computed torque separates
    from gravity-compensated PD.
    """
    centre = np.asarray(centre, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    phase = omega * t

    return Setpoint(
        q=centre + amplitude * np.sin(phase),
        dq=amplitude * omega * np.cos(phase),
        ddq=-amplitude * omega**2 * np.sin(phase),
    )


@dataclass(frozen=True)
class FeasibilityReport:
    """Whether a trajectory can actually be executed, and why not if it cannot."""

    within_joint_limits: bool
    lowest_point_m: float
    peak_torque_nm: np.ndarray
    torque_limit_nm: np.ndarray
    clearance_m: float

    @property
    def clears_floor(self) -> bool:
        return self.lowest_point_m >= self.clearance_m

    @property
    def within_torque_limits(self) -> bool:
        return bool(np.all(self.peak_torque_nm <= self.torque_limit_nm))

    @property
    def feasible(self) -> bool:
        return self.within_joint_limits and self.clears_floor and self.within_torque_limits

    def describe(self) -> str:
        if self.feasible:
            headroom = np.min(self.torque_limit_nm - self.peak_torque_nm)
            return (
                f"feasible: lowest point {self.lowest_point_m * 1000:.0f} mm above "
                f"the floor, {headroom * 1000:.0f} mNm of torque headroom"
            )

        reasons = []
        if not self.within_joint_limits:
            reasons.append("exceeds joint limits")
        if not self.clears_floor:
            reasons.append(
                f"drives the arm to {self.lowest_point_m * 1000:.0f} mm "
                f"(floor is 0, wanted >= {self.clearance_m * 1000:.0f} mm)"
            )
        if not self.within_torque_limits:
            over = self.peak_torque_nm > self.torque_limit_nm
            reasons.append(
                f"needs {np.round(self.peak_torque_nm[over], 3)} Nm where the "
                f"actuators hold {np.round(self.torque_limit_nm[over], 3)} Nm"
            )
        return "INFEASIBLE: " + "; ".join(reasons)


def check_feasibility(setpoints, params=None, *, clearance_m: float = 0.03) -> FeasibilityReport:
    """Test a trajectory against joint limits, the floor, and actuator torque.

    Written because all three of these were violated at various points while
    developing the tracking demo, and each failure looked exactly like a badly
    tuned controller:

    * a trajectory passing through the floor produces a large, gain-independent
      tracking error, because the contact solver -- correctly -- refuses to let
      the arm follow it;
    * a trajectory demanding more torque than the actuators hold saturates, and
      saturation invalidates every linear control result;
    * a trajectory outside joint limits is stopped by the joint stops.

    None of these is fixed by tuning. Check before blaming the controller.
    """
    from arm.dynamics import damping_torque, inverse_dynamics
    from arm.kinematics import fk_frames
    from arm.params import default_params

    params = params or default_params()
    limits = params.joint_limits_rad
    torque_limit = np.array(
        [params.actuators[j.actuator].continuous_torque_nm for j in params.joints]
    )

    within_limits = True
    lowest = float("inf")
    peak = np.zeros(3)

    for setpoint in setpoints:
        within_limits &= bool(
            np.all(setpoint.q >= limits[:, 0]) and np.all(setpoint.q <= limits[:, 1])
        )
        lowest = min(lowest, *(frame[2, 3] for frame in fk_frames(setpoint.q, params)[1:]))
        torque = inverse_dynamics(setpoint.q, setpoint.dq, setpoint.ddq)
        peak = np.maximum(peak, np.abs(torque + damping_torque(setpoint.dq)))

    return FeasibilityReport(
        within_joint_limits=within_limits,
        lowest_point_m=float(lowest),
        peak_torque_nm=peak,
        torque_limit_nm=torque_limit,
        clearance_m=clearance_m,
    )
