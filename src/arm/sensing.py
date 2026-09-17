"""What the controller actually gets to measure.

Every controller so far has read perfect state straight out of the simulator:
exact joint angles and exact velocities, at no delay. Real hardware provides
neither.

A Dynamixel reports position from a **4096-count magnetic encoder**, so angles
arrive quantised to 2π/4096 = 1.534 mrad. Velocity is not measured at all -- it
is differentiated from those quantised positions, and differentiation is exactly
the operation that turns a small position quantum into a large velocity one::

    velocity quantum = 1.534e-3 rad / 0.005 s = 0.307 rad/s

With ``kd = 0.6`` that is ±0.184 N·m of pure noise on a joint whose continuous
limit is 0.6 N·m -- **about 30% of the actuator's capability spent on quantisation
noise**. This is the single biggest reason a controller that works perfectly in
simulation buzzes and overheats on hardware.

The fix is not a bigger encoder. It is to stop treating the derivative of a
quantised signal as if it were a measurement, and to filter -- which trades
delay for noise, and delay costs phase margin. That trade is the whole subject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm.hardware import ArmState
from arm.params import ArmParams, default_params

__all__ = [
    "AlphaBeta",
    "FilteredDifference",
    "FiniteDifference",
    "KalmanVelocity",
    "SensedArm",
    "encoder_resolution_rad",
    "quantize",
]


def encoder_resolution_rad(params: ArmParams | None = None) -> np.ndarray:
    """Smallest angle each joint's encoder can distinguish."""
    params = params or default_params()
    return np.array(
        [
            2.0 * np.pi / params.actuators[joint.actuator].encoder_counts_rev
            for joint in params.joints
        ]
    )


def quantize(q: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    """Snap angles to the nearest encoder count."""
    resolution = np.asarray(resolution, dtype=float)
    return np.round(np.asarray(q, dtype=float) / resolution) * resolution


@dataclass
class FiniteDifference:
    """Backward difference. The obvious estimator, and the one that fails.

    ``(q[k] - q[k-1]) / dt`` is unbiased and instant, which is why it is
    tempting. It also multiplies the position quantum by ``1/dt``, so at 200 Hz
    it amplifies encoder noise 200-fold. Included precisely so the failure can
    be measured rather than asserted.
    """

    _previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, q: np.ndarray, dt: float, tau: np.ndarray | None = None) -> np.ndarray:
        del tau  # a kinematic estimator has no use for the commanded torque
        q = np.asarray(q, dtype=float)
        if self._previous is None:
            self._previous = q
            return np.zeros_like(q)
        velocity = (q - self._previous) / dt
        self._previous = q
        return velocity


@dataclass
class FilteredDifference:
    """Finite difference followed by a first-order low-pass.

    ``cutoff_hz`` buys noise rejection with delay. Roughly, a first-order filter
    at cutoff ``f_c`` costs about ``1/(2π f_c)`` seconds of lag, and lag eats
    phase margin in the loop that consumes it. Setting the cutoff far below the
    control rate makes the velocity clean and the controller sluggish; setting
    it high does the reverse. There is no setting that avoids the trade.
    """

    cutoff_hz: float = 20.0
    _difference: FiniteDifference = field(default_factory=FiniteDifference)
    _estimate: np.ndarray | None = None

    def reset(self) -> None:
        self._difference.reset()
        self._estimate = None

    def update(self, q: np.ndarray, dt: float, tau: np.ndarray | None = None) -> np.ndarray:
        del tau
        raw = self._difference.update(q, dt)
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        if self._estimate is None:
            self._estimate = raw
        else:
            self._estimate = self._estimate + alpha * (raw - self._estimate)
        return self._estimate.copy()


@dataclass
class AlphaBeta:
    """A two-state tracking filter over position and velocity.

    Rather than differentiating and then smoothing, this keeps an internal
    estimate of both position and velocity, predicts forward, and corrects
    against the measurement. That is a fixed-gain special case of the same idea
    as a Kalman filter -- and the natural bridge to one, where the gains come
    from stated process and measurement noise instead of being chosen.

    ``alpha`` corrects position, ``beta`` corrects velocity. Both in (0, 1);
    larger means trust the measurement more and the model less.

    Measured against differentiate-then-smooth on the tracking benchmark, this
    **dominates** a first-order low-pass: 2.74 mm error at 30 mNm of command
    chatter, against 5.77 mm at 24 mNm for a 10 Hz cutoff. Roughly half the
    error for comparable smoothness, because predicting forward with a motion
    model extracts more from the same quantised samples than filtering a
    derivative after the fact.
    """

    alpha: float = 0.7
    beta: float = 0.35
    _position: np.ndarray | None = None
    _velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._position = None
        self._velocity = None

    def update(self, q: np.ndarray, dt: float, tau: np.ndarray | None = None) -> np.ndarray:
        del tau
        q = np.asarray(q, dtype=float)
        if self._position is None:
            self._position = q
            self._velocity = np.zeros_like(q)
            return self._velocity.copy()

        predicted_position = self._position + self._velocity * dt
        residual = q - predicted_position

        self._position = predicted_position + self.alpha * residual
        self._velocity = self._velocity + (self.beta / dt) * residual
        return self._velocity.copy()


class SensedArm:
    """Wraps a backend so the controller sees what hardware would report.

    An :class:`arm.hardware.ArmBackend` itself, which is the point: the
    abstraction that was built for swapping simulation and hardware also lets a
    sensing model be inserted between them without any controller noticing.
    """

    def __init__(
        self,
        backend,
        params: ArmParams | None = None,
        *,
        quantize_position: bool = True,
        velocity_estimator=None,
        position_noise_rad: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.params = params or default_params()
        self.quantize_position = quantize_position
        self.velocity_estimator = (
            velocity_estimator if velocity_estimator is not None else FiniteDifference()
        )
        self.position_noise_rad = float(position_noise_rad)
        self._resolution = encoder_resolution_rad(self.params)
        self._rng = np.random.default_rng(seed)

    # -- ArmBackend ----------------------------------------------------------

    @property
    def dt(self) -> float:
        return self.backend.dt

    def read(self) -> ArmState:
        true_state = self.backend.read()
        measured = true_state.q.copy()

        if self.position_noise_rad > 0.0:
            measured = measured + self._rng.normal(
                0.0, self.position_noise_rad, size=measured.shape
            )
        if self.quantize_position:
            measured = quantize(measured, self._resolution)

        # The torque that was actually applied last cycle. A controller knows
        # this -- write_torque returns it -- so using it in the estimator is
        # legitimate, unlike reading true state.
        velocity = self.velocity_estimator.update(measured, self.dt, true_state.tau)
        return ArmState(t=true_state.t, q=measured, dq=velocity, tau=true_state.tau)

    def true_state(self) -> ArmState:
        """Ground truth, for measuring error. A controller must not use this."""
        return self.backend.read()

    def write_torque(self, tau: np.ndarray) -> np.ndarray:
        return self.backend.write_torque(tau)

    def step(self) -> None:
        self.backend.step()

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        self.backend.reset(q, dq)
        self.velocity_estimator.reset()

    # -- passthrough ---------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self.backend, name)


@dataclass
class KalmanVelocity:
    """A Kalman filter over joint position and velocity.

    The same shape as :class:`AlphaBeta` -- predict forward, compare against the
    measurement, correct by a fraction of the disagreement -- but the fraction is
    *derived* rather than chosen. Two numbers are stated instead:

    ``R``  how noisy one measurement is. For a quantised encoder this is exactly
           computable: the error is uniform over one count, and a uniform
           distribution over ``[-d/2, d/2]`` has variance ``d^2/12``. No tuning.
    ``Q``  how fast the prediction goes stale -- the unmodelled acceleration the
           arm experiences between control periods.

    The state is ``x = [q; dq]`` (6 elements) and only position is measured, so
    ``H = [I 0]``. The cycle::

        predict   x = F x (+ model),   P = F P F' + Q      uncertainty GROWS
        update    K = P H' (H P H' + R)^-1
                  x = x + K (z - H x),  P = (I - K H) P    uncertainty SHRINKS

    Two properties worth knowing, both asserted in ``tests/test_kalman.py``:

    * **The covariance never touches the measured values.** ``P`` and ``K``
      depend only on ``F``, ``H``, ``Q``, ``R``, so they converge to the same
      steady state from any starting confidence, with no data at all. Measuring
      reduces uncertainty; the measurement coming out *as predicted* does not.
    * **At steady state this reduces to alpha-beta.** The converged gain is
      constant, and that constant is exactly ``(alpha, beta/dt)``. Running with
      ``model=False`` and letting it settle reproduces a hand-tuned alpha-beta
      filter -- with the right gains, which is the point: the derived values are
      ``alpha=0.770, beta=0.542`` against the ``0.7 / 0.35`` chosen by eye.

    Set ``use_model`` to predict with the arm's actual dynamics
    (``pinocchio.aba``) rather than assuming constant velocity. Knowing the
    commanded torque makes the forecast better, which justifies a smaller ``Q``,
    which yields a smaller gain, which leans less on the noisy measurement. On
    the tracking benchmark that recovers almost everything quantisation took
    away -- 0.14 mm against 0.16 mm for a controller reading exact state, and
    1.65 mm for the kinematic filter.

    **But a model-based estimator is only as good as its model.** Add Coulomb
    friction the estimator does not know about and it becomes *worse* than the
    kinematic filter -- 5.4 mm against 2.9 mm at 0.05 N·m. Raising
    ``process_accel_std`` to admit the model is unreliable recovers part of it
    (down to 3.1 mm) but never all, and the reason is worth understanding:
    ``Q`` describes *zero-mean* noise, while friction is a systematic bias that
    always opposes motion. No amount of inflating Q represents a bias. The
    proper fixes are to identify the friction and model it, or to estimate it
    as an augmented state -- not to keep turning knobs.
    """

    process_accel_std: float = 20.0  # rad/s^2 of unmodelled acceleration
    use_model: bool = False
    params: ArmParams | None = None
    _x: np.ndarray | None = None
    _P: np.ndarray | None = None
    _R: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.params = self.params or default_params()
        resolution = encoder_resolution_rad(self.params)
        # Quantisation error is uniform across one count: variance = d^2 / 12.
        self._R = np.diag(resolution**2 / 12.0)

    def reset(self) -> None:
        self._x = None
        self._P = None

    @property
    def covariance(self) -> np.ndarray | None:
        """The 6x6 state covariance, or None before the first measurement."""
        return None if self._P is None else self._P.copy()

    @property
    def position_std_rad(self) -> np.ndarray:
        """One-sigma position uncertainty per joint."""
        if self._P is None:
            return np.full(3, np.inf)
        return np.sqrt(np.diag(self._P)[:3])

    @property
    def velocity_std_rad_s(self) -> np.ndarray:
        if self._P is None:
            return np.full(3, np.inf)
        return np.sqrt(np.diag(self._P)[3:])

    def _transition(self, dt: float) -> np.ndarray:
        eye = np.eye(3)
        return np.block([[eye, dt * eye], [np.zeros((3, 3)), eye]])

    def _process_noise(self, dt: float) -> np.ndarray:
        """Continuous white-noise acceleration model.

        An unknown acceleration acting over ``dt`` moves position by
        ``0.5 dt^2`` and velocity by ``dt``, so ``Q = G G' sigma_a^2`` with
        ``G = [0.5 dt^2; dt]``. The off-diagonal terms matter: position and
        velocity errors from a common acceleration are correlated, and a
        diagonal Q would throw that away.
        """
        gain = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
        return gain @ gain.T * self.process_accel_std**2

    def _predict_mean(self, dt: float, tau: np.ndarray | None) -> np.ndarray:
        position, velocity = self._x[:3], self._x[3:]

        if self.use_model and tau is not None:
            from arm.dynamics import forward_dynamics

            acceleration = forward_dynamics(position, velocity, tau)
        else:
            acceleration = np.zeros(3)

        return np.concatenate(
            [
                position + velocity * dt + 0.5 * acceleration * dt**2,
                velocity + acceleration * dt,
            ]
        )

    def update(self, q: np.ndarray, dt: float, tau: np.ndarray | None = None) -> np.ndarray:
        measurement = np.asarray(q, dtype=float)

        if self._x is None:
            self._x = np.concatenate([measurement, np.zeros(3)])
            # Position is known to sensor accuracy; velocity is not known at all.
            self._P = np.diag(np.concatenate([np.diag(self._R), np.full(3, 1.0)]))
            return np.zeros(3)

        transition = self._transition(dt)

        # Predict. The mean may follow the nonlinear dynamics while the
        # covariance is propagated with the kinematic transition -- an EKF-style
        # approximation that skips the Jacobian of the dynamics. It costs a
        # little optimality and no correctness, because a Kalman filter stays
        # stable under an imperfect F so long as Q is honest about it.
        self._x = self._predict_mean(dt, tau)
        self._P = transition @ self._P @ transition.T + self._process_noise(dt)

        # Update.
        observation = np.hstack([np.eye(3), np.zeros((3, 3))])
        innovation = measurement - observation @ self._x
        innovation_covariance = observation @ self._P @ observation.T + self._R
        gain = self._P @ observation.T @ np.linalg.inv(innovation_covariance)

        self._x = self._x + gain @ innovation
        identity = np.eye(6)
        # Joseph form: stays symmetric and positive definite under round-off,
        # where the shorter (I - KH)P does not.
        factor = identity - gain @ observation
        self._P = factor @ self._P @ factor.T + gain @ self._R @ gain.T

        return self._x[3:].copy()

    def steady_state_gain(self, dt: float, iterations: int = 500) -> np.ndarray:
        """The gain this filter converges to, computed without any data.

        Demonstrates the property directly: iterate the covariance recursion
        alone and it settles, because nothing in it depends on what is measured.
        """
        transition = self._transition(dt)
        observation = np.hstack([np.eye(3), np.zeros((3, 3))])
        covariance = np.eye(6)
        gain = np.zeros((6, 3))

        for _ in range(iterations):
            covariance = transition @ covariance @ transition.T + self._process_noise(dt)
            innovation_covariance = observation @ covariance @ observation.T + self._R
            gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
            covariance = (np.eye(6) - gain @ observation) @ covariance

        return gain

    def equivalent_alpha_beta(self, dt: float) -> tuple[float, float]:
        """The alpha-beta gains this filter is equivalent to at steady state."""
        gain = self.steady_state_gain(dt)
        return float(gain[0, 0]), float(gain[3, 0] * dt)
