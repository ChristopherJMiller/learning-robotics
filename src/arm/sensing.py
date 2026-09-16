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

    def update(self, q: np.ndarray, dt: float) -> np.ndarray:
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

    def update(self, q: np.ndarray, dt: float) -> np.ndarray:
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

    def update(self, q: np.ndarray, dt: float) -> np.ndarray:
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

        velocity = self.velocity_estimator.update(measured, self.dt)
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
