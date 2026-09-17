"""Turning a path into a trajectory.

A planner returns geometry: an ordered list of configurations, with no statement
about when the arm is at any of them. That is not executable. A controller needs
``q``, ``q̇`` and ``q̈`` at every instant, and the arm needs those to be within
what its joints and motors can actually do.

Three problems have to be solved, in this order.

**Corners.** A piecewise-linear path changes direction instantly at each
waypoint, and traversing a corner at non-zero speed requires infinite
acceleration. Either stop at every waypoint, or round the corners -- and
rounding moves the path, so the rounded version has to be re-checked for
collision. A smoother that quietly cuts a corner through an obstacle has
undone the planner's work.

**Timing.** Assigning durations to segments is where velocity and acceleration
first exist, and therefore the first point at which torque can even be asked
about. This is why ``check_feasibility`` takes setpoints rather than
configurations.

**Feasibility.** If the result exceeds what the actuators can hold, slow it
down. That works -- but not unconditionally, and the reason is worth knowing::

    stretching a trajectory by a factor s scales velocity by 1/s
    and acceleration by 1/s^2

so inertial torque falls quadratically with slowing. **Gravity does not scale at
all.** A configuration whose gravity load alone exceeds the continuous torque
limit cannot be rescued by going slower, and a time-scaling loop that assumes
otherwise will iterate forever. :func:`gravity_feasible` separates the two cases
before any scaling is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from arm.cspace import CollisionChecker
from arm.dynamics import damping_torque, friction_torque, gravity_torque, inverse_dynamics
from arm.params import ArmParams, default_params
from arm.trajectory import Setpoint

__all__ = [
    "Trajectory",
    "gravity_feasible",
    "peak_torque",
    "time_optimal_scale",
    "trajectory_from_path",
]


@dataclass(frozen=True)
class Trajectory:
    """A path with timing: sample it at any instant to get a full setpoint.

    Stores the waypoints and knot times rather than only the fitted spline, so
    that rescaling rebuilds it under identical boundary conditions. An earlier
    version kept just the spline and refitted from ``spline.x`` and
    ``spline(spline.x)``, which silently dropped the clamped end conditions and
    produced a *different curve* -- slowing the trajectory by 1% appeared to cut
    peak torque sixfold, which is impossible and was the giveaway.
    """

    waypoints: np.ndarray
    times_s: np.ndarray
    params: ArmParams

    @property
    def spline(self) -> CubicSpline:
        return CubicSpline(self.times_s, self.waypoints, bc_type="clamped")

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    def at(self, t: float, spline: CubicSpline | None = None) -> Setpoint:
        """Position, velocity and acceleration at time ``t``.

        Clamped at both ends, and held with zero velocity past the end so that
        a controller reading beyond the trajectory holds the final pose rather
        than extrapolating a spline off into nonsense.

        ``spline`` may be passed to avoid refitting on every call, which matters
        inside a control loop.
        """
        spline = self.spline if spline is None else spline
        if t >= self.duration_s:
            return Setpoint(
                q=np.asarray(spline(self.duration_s), dtype=float),
                dq=np.zeros(3),
                ddq=np.zeros(3),
            )
        t = max(t, 0.0)
        return Setpoint(
            q=np.asarray(spline(t), dtype=float),
            dq=np.asarray(spline(t, 1), dtype=float),
            ddq=np.asarray(spline(t, 2), dtype=float),
        )

    def sample(self, count: int = 200) -> list[Setpoint]:
        spline = self.spline
        return [self.at(t, spline) for t in np.linspace(0.0, self.duration_s, count)]

    def rescaled(self, factor: float) -> Trajectory:
        """The same geometry, traversed ``factor`` times slower.

        Only the knot times are stretched; the waypoints and boundary conditions
        are untouched, so the curve through space is identical and velocity
        scales by exactly ``1/factor``, acceleration by ``1/factor**2``.
        """
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        return Trajectory(
            waypoints=self.waypoints,
            times_s=self.times_s * factor,
            params=self.params,
        )


def _knot_times(path, speed_rad_s: float) -> np.ndarray:
    """Assign a time to each waypoint, at roughly constant joint-space speed.

    Crude on purpose. A genuinely time-optimal parameterisation solves for the
    fastest traversal subject to every constraint at once (TOPP and its
    descendants); this assigns a nominal speed and then scales the whole
    trajectory until it fits. The result is slower than optimal but correct,
    and the failure modes are the interesting part rather than the algorithm.
    """
    points = np.asarray(path, dtype=float)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # A minimum spacing keeps duplicate or near-duplicate waypoints from
    # producing a zero-length interval, which would make the spline singular.
    times = np.concatenate([[0.0], np.cumsum(np.maximum(distances, 1e-6))])
    return times / speed_rad_s


def trajectory_from_path(
    path,
    params: ArmParams | None = None,
    *,
    speed_rad_s: float = 1.0,
) -> Trajectory:
    """Fit a smooth trajectory through the waypoints of a path.

    A cubic spline with clamped ends: zero velocity at the start and finish, and
    continuous acceleration throughout, which is what makes the result something
    a controller can feed forward without jolting.

    The spline does **not** pass exactly along the straight segments between
    waypoints -- it rounds the corners, which is the whole point, and also the
    risk. Use :func:`path_stays_valid` to confirm the rounded version is still
    collision-free.
    """
    params = params or default_params()
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        raise ValueError("a trajectory needs at least two waypoints")

    times = _knot_times(points, speed_rad_s)
    return Trajectory(waypoints=points, times_s=times, params=params)


def path_stays_valid(trajectory: Trajectory, checker: CollisionChecker, samples: int = 400) -> bool:
    """Whether the smoothed trajectory is still collision-free.

    Smoothing rounds corners, and a rounded corner cuts inside the original
    path. That can push the arm into an obstacle the planner had carefully
    avoided, which would silently undo the planning. Worth checking every time.
    """
    return all(checker.is_valid(setpoint.q) for setpoint in trajectory.sample(samples))


def peak_torque(trajectory: Trajectory, samples: int = 200) -> np.ndarray:
    """Largest absolute torque each joint must produce over the trajectory."""
    peak = np.zeros(3)
    for setpoint in trajectory.sample(samples):
        torque = (
            inverse_dynamics(setpoint.q, setpoint.dq, setpoint.ddq)
            + damping_torque(setpoint.dq)
            + friction_torque(setpoint.dq, params=trajectory.params)
        )
        peak = np.maximum(peak, np.abs(torque))
    return peak


def torque_limits(params: ArmParams | None = None) -> np.ndarray:
    params = params or default_params()
    return np.array([params.actuators[j.actuator].continuous_torque_nm for j in params.joints])


def gravity_feasible(trajectory: Trajectory, samples: int = 200):
    """Can the arm even *hold* every pose on this path, moving arbitrarily slowly?

    The question that has to be asked before any time scaling, because gravity
    torque is a function of configuration alone. Slowing down divides inertial
    torque by the square of the slowdown and leaves gravity untouched, so if
    holding a pose already saturates a joint, no amount of slowing will help.

    Returns ``(feasible, worst_ratio)``, where the ratio is the largest gravity
    torque seen as a fraction of the continuous limit.
    """
    limits = torque_limits(trajectory.params)
    worst = 0.0
    for setpoint in trajectory.sample(samples):
        ratio = float(np.max(np.abs(gravity_torque(setpoint.q)) / limits))
        worst = max(worst, ratio)
    return worst <= 1.0, worst


def time_optimal_scale(
    trajectory: Trajectory,
    *,
    max_factor: float = 20.0,
    tolerance: float = 0.01,
) -> tuple[Trajectory, float]:
    """Find the fastest traversal of this path the actuators can sustain.

    Binary search on the slowdown factor. Torque is monotone in it -- inertial
    terms fall as ``1/s^2`` while gravity stays put -- so bisection is
    well-founded and converges quickly.

    Raises if gravity alone makes the path infeasible, rather than searching to
    ``max_factor`` and returning something that still does not work. That
    distinction matters: one failure means "go slower", the other means "this
    arm cannot hold this pose at all", and they call for completely different
    responses.
    """
    holdable, ratio = gravity_feasible(trajectory)
    if not holdable:
        raise ValueError(
            f"gravity alone needs {ratio:.2f}x the continuous torque limit "
            "somewhere on this path; slowing down cannot fix that"
        )

    limits = torque_limits(trajectory.params)
    if np.all(peak_torque(trajectory) <= limits):
        return trajectory, 1.0

    low, high = 1.0, 2.0
    while high <= max_factor:
        if np.all(peak_torque(trajectory.rescaled(high)) <= limits):
            break
        low, high = high, high * 2.0
    else:
        raise ValueError(
            f"still infeasible at {max_factor}x slowdown, which should not happen "
            "once gravity is known to be within limits"
        )

    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if np.all(peak_torque(trajectory.rescaled(middle)) <= limits):
            high = middle
        else:
            low = middle

    return trajectory.rescaled(high), high
