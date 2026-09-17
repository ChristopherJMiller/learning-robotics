"""Time-optimal path parameterisation: go slowly only where you must.

``arm.timing`` stretches a whole trajectory by one factor until the hardest
instant fits. That is correct and wasteful -- measured on a planned path, the
torque limit binds at 9% of the path and the median utilisation is 0.32, so for
most of the move the arm crawls for no reason.

The fix is to vary speed *along* the path. The idea that makes it tractable is a
change of variable: stop describing the motion as ``q(t)`` and describe it as a
fixed geometric path ``q = f(s)`` plus a timing ``s(t)`` to be solved for. Then::

    q̇  = f'(s)·ṡ
    q̈  = f'(s)·s̈ + f''(s)·ṡ²

and substituting into ``τ = M(q)q̈ + C(q,q̇)q̇ + g(q)`` gives, after collecting
terms::

    τ(s) = a(s)·s̈ + b(s)·ṡ² + c(s)

**Torque is linear in ``s̈`` and in ``ṡ²``.** That is the whole reason this
works: the constraint ``τ_min ≤ τ ≤ τ_max`` becomes a linear inequality, so at
any point the feasible range of ``s̈`` is an interval that can be *solved for*
rather than searched.

Writing ``u = ṡ²`` and noting ``du/ds = 2s̈``, the classical algorithm
(Bobrow, Shin and McKay; later Pham's TOPP-RA) is then:

1. find the maximum-velocity curve -- the largest ``u`` at each ``s`` for which
   any ``s̈`` at all is feasible;
2. integrate forward from rest at maximum acceleration;
3. integrate backward from rest at maximum deceleration;
4. take the pointwise minimum of the three.

The backward pass is what stops the arm arriving at a corner too fast to slow
down for, which a purely greedy forward pass would happily do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm.dynamics import damping_torque, inverse_dynamics
from arm.params import ArmParams, default_params
from arm.timing import Trajectory, torque_limits
from arm.trajectory import Setpoint

__all__ = ["TimeOptimalTrajectory", "parameterise"]

_ZERO_TOLERANCE = 1e-9


def _coefficients(spline, s: float, params: ArmParams):
    """``a(s)``, ``b(s)``, ``c(s)`` with ``τ = a·s̈ + b·ṡ² + c``.

    Recovered from three inverse-dynamics evaluations rather than by forming
    ``M`` and ``C`` explicitly, which keeps this honest about using the same
    dynamics the controller does:

    * ``c = rnea(q, 0, 0)`` is gravity alone;
    * ``a = rnea(q, 0, f') - c`` is ``M·f'``, the response to unit ``s̈``;
    * ``b = rnea(q, f', f'') - c`` is ``M·f'' + C(q,f')f'``, the ``ṡ²`` term.
    """
    q = np.asarray(spline(s), dtype=float)
    first = np.asarray(spline(s, 1), dtype=float)
    second = np.asarray(spline(s, 2), dtype=float)

    gravity = inverse_dynamics(q, np.zeros(3), np.zeros(3))
    a = inverse_dynamics(q, np.zeros(3), first) - gravity
    b = inverse_dynamics(q, first, second) - gravity
    return a, b, gravity, first


def _acceleration_bounds(a, b, c, u: float, limits, damping):
    """Feasible ``[α, β]`` for ``s̈`` at this point, given ``u = ṡ²``.

    Returns ``(-inf, inf)`` reversed -- that is, ``α > β`` -- when no ``s̈``
    satisfies every joint, which is how the maximum-velocity search detects
    that ``u`` is already too fast.
    """
    # Viscous damping is proportional to q̇ = f'·ṡ, so it contributes a term in
    # sqrt(u) rather than u. Folded into the constant part at this u.
    offset = c + b * u + damping * np.sqrt(max(u, 0.0))
    upper = limits - offset
    lower = -limits - offset

    alpha, beta = -np.inf, np.inf
    for index in range(len(a)):
        coefficient = a[index]
        if coefficient > _ZERO_TOLERANCE:
            beta = min(beta, upper[index] / coefficient)
            alpha = max(alpha, lower[index] / coefficient)
        elif coefficient < -_ZERO_TOLERANCE:
            beta = min(beta, lower[index] / coefficient)
            alpha = max(alpha, upper[index] / coefficient)
        elif not (lower[index] <= 0.0 <= upper[index]):
            # s̈ has no influence on this joint here, and the rest of the
            # torque already violates its limit: no acceleration can help.
            return 1.0, -1.0
    return alpha, beta


def _maximum_velocity(a, b, c, limits, damping, ceiling: float) -> float:
    """Largest ``u`` at this point for which some ``s̈`` is feasible.

    Bisection. ``u`` appears linearly in the offset, so feasibility is monotone
    in it and the search is well founded.
    """
    low, high = 0.0, ceiling
    alpha, beta = _acceleration_bounds(a, b, c, high, limits, damping)
    if alpha <= beta:
        return high

    for _ in range(40):
        middle = 0.5 * (low + high)
        alpha, beta = _acceleration_bounds(a, b, c, middle, limits, damping)
        if alpha <= beta:
            low = middle
        else:
            high = middle
    return low


@dataclass(frozen=True)
class TimeOptimalTrajectory:
    """A geometric path plus a velocity profile solved for, not assumed."""

    spline: object  # CubicSpline over the path coordinate s in [0, 1]
    s_nodes: np.ndarray
    u_nodes: np.ndarray  # squared path speed at each node
    t_nodes: np.ndarray
    params: ArmParams

    @property
    def duration_s(self) -> float:
        return float(self.t_nodes[-1])

    def _state_at(self, t: float):
        s = float(np.interp(t, self.t_nodes, self.s_nodes))
        u = float(np.interp(s, self.s_nodes, self.u_nodes))
        speed = float(np.sqrt(max(u, 0.0)))

        # s̈ from the profile itself: du/ds = 2 s̈.
        gradient = np.gradient(self.u_nodes, self.s_nodes)
        accel = 0.5 * float(np.interp(s, self.s_nodes, gradient))
        return s, speed, accel

    def at(self, t: float) -> Setpoint:
        if t >= self.duration_s:
            return Setpoint(
                q=np.asarray(self.spline(1.0), dtype=float),
                dq=np.zeros(3),
                ddq=np.zeros(3),
            )
        s, speed, accel = self._state_at(max(t, 0.0))
        first = np.asarray(self.spline(s, 1), dtype=float)
        second = np.asarray(self.spline(s, 2), dtype=float)
        return Setpoint(
            q=np.asarray(self.spline(s), dtype=float),
            dq=first * speed,
            ddq=first * accel + second * speed**2,
        )

    def sample(self, count: int = 200) -> list[Setpoint]:
        return [self.at(t) for t in np.linspace(0.0, self.duration_s, count)]

    def rescaled(self, factor: float) -> TimeOptimalTrajectory:
        """Uniformly slower, keeping the same geometry and profile shape.

        Used only as a final safety correction: the solved profile can exceed
        the torque limit slightly through discretisation, and a small uniform
        stretch removes that without redoing the solve.
        """
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        return TimeOptimalTrajectory(
            spline=self.spline,
            s_nodes=self.s_nodes,
            u_nodes=self.u_nodes / factor**2,
            t_nodes=self.t_nodes * factor,
            params=self.params,
        )

    def peak_torque(self, samples: int = 300) -> np.ndarray:
        peak = np.zeros(3)
        for setpoint in self.sample(samples):
            torque = inverse_dynamics(setpoint.q, setpoint.dq, setpoint.ddq) + damping_torque(
                setpoint.dq
            )
            peak = np.maximum(peak, np.abs(torque))
        return peak


def parameterise(
    trajectory: Trajectory,
    *,
    nodes: int = 400,
    speed_ceiling: float = 400.0,
    margin: float = 0.92,
) -> TimeOptimalTrajectory:
    """Solve for the fastest timing of a path the actuators can sustain.

    Takes an ``arm.timing.Trajectory`` for its geometry only -- whatever timing
    it already had is discarded and replaced.

    ``margin`` shrinks the torque limits used while solving. The solved profile
    is exactly at the limit by construction, so *any* discretisation error puts
    the realised trajectory over it: measured, 13% over at 50 nodes and still
    2% over at 1600, converging only as about ``1/sqrt(nodes)`` because of the
    square-root singularity in ``ṡ`` at each end. Solving against a slightly
    lower limit absorbs that, and the result is verified afterwards and given a
    final uniform stretch if it is still over -- so the returned trajectory is
    feasible rather than nearly feasible.
    """
    params = trajectory.params
    true_limits = torque_limits(params)
    limits = true_limits * margin

    # Re-fit the geometry against a normalised path coordinate, so that s runs
    # 0 to 1 regardless of what timing the input happened to carry.
    #
    # A **natural** spline, not the clamped one arm.timing uses, and the
    # difference is not cosmetic. Clamped ends force f'(0) = f'(1) = 0, which
    # bakes "start and finish at rest" into the *geometry*. Here that is a
    # category error: stopping belongs in the velocity profile, as u = 0 at the
    # ends. With f'(0) = 0 the coefficient a = M·f' is zero too, so no path
    # acceleration influences any joint, the forward pass can never leave
    # u = 0, and dt = ds/sqrt(u) diverges -- measured, a 15076-second
    # trajectory for a move that should take under a second.
    from scipy.interpolate import CubicSpline

    span = trajectory.times_s[-1]
    spline = CubicSpline(trajectory.times_s / span, trajectory.waypoints, bc_type="natural")

    s_nodes = np.linspace(0.0, 1.0, nodes)
    step = s_nodes[1] - s_nodes[0]

    coefficients = []
    for s in s_nodes:
        a, b, c, first = _coefficients(spline, s, params)
        damping = damping_torque(first)  # scales with sqrt(u); see the bounds
        coefficients.append((a, b, c, damping))

    ceiling = np.array(
        [
            _maximum_velocity(a, b, c, limits, damping, speed_ceiling)
            for a, b, c, damping in coefficients
        ]
    )

    # Forward from rest, accelerating as hard as each point allows.
    forward = np.zeros(nodes)
    for index in range(nodes - 1):
        a, b, c, damping = coefficients[index]
        _, beta = _acceleration_bounds(a, b, c, forward[index], limits, damping)
        if not np.isfinite(beta):
            beta = 0.0
        forward[index + 1] = min(max(forward[index] + 2.0 * beta * step, 0.0), ceiling[index + 1])

    # Backward from rest, so the arm is never travelling faster than it can
    # still stop from. Without this pass a greedy forward profile arrives at a
    # tight corner too fast and there is no valid deceleration left.
    backward = np.zeros(nodes)
    for index in range(nodes - 1, 0, -1):
        a, b, c, damping = coefficients[index]
        alpha, _ = _acceleration_bounds(a, b, c, backward[index], limits, damping)
        if not np.isfinite(alpha):
            alpha = 0.0
        backward[index - 1] = min(
            max(backward[index] - 2.0 * alpha * step, 0.0), ceiling[index - 1]
        )

    u_nodes = np.minimum(np.minimum(forward, backward), ceiling)

    # Integrate dt = ds/ṡ using the *average speed* across each interval,
    #
    #     dt = 2·ds / (ṡ_k + ṡ_{k+1})
    #
    # rather than the average of 1/ṡ. The profile is exactly zero at both ends,
    # so 1/ṡ is singular there, and a trapezoid rule over 1/ṡ evaluates that
    # singularity as an enormous finite number -- it produced a 5026-second
    # trajectory. The singularity is integrable: near rest the arm accelerates
    # at roughly constant s̈, so ṡ ~ sqrt(2·s̈·s) and the integral of ds/sqrt(s)
    # converges. Averaging the speed rather than its reciprocal respects that,
    # and gives a finite, correct first interval.
    speeds = np.sqrt(np.maximum(u_nodes, 0.0))
    pair_sum = speeds[:-1] + speeds[1:]
    intervals = np.where(pair_sum > 1e-12, 2.0 * step / np.maximum(pair_sum, 1e-12), 0.0)
    t_nodes = np.concatenate([[0.0], np.cumsum(intervals)])

    result = TimeOptimalTrajectory(
        spline=spline,
        s_nodes=s_nodes,
        u_nodes=u_nodes,
        t_nodes=t_nodes,
        params=params or default_params(),
    )

    # Verify against the *true* limits and correct if the margin was not enough.
    # Torque is quadratic in speed, so the stretch needed is the square root of
    # the overshoot; a little extra covers the gravity term, which does not
    # scale and so makes the true requirement slightly larger than that.
    overshoot = float(np.max(result.peak_torque() / true_limits))
    if overshoot > 1.0:
        result = result.rescaled(np.sqrt(overshoot) * 1.02)
    return result
