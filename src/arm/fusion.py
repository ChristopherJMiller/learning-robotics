"""Fusing a camera that is always telling you about the past.

Every estimator so far consumed measurements the instant they were true. An
encoder read over a serial bus is close enough to that: a Dynamixel answers in
well under a millisecond, which at 200 Hz is a fraction of a control period.

A camera is not close enough. An image has to be exposed, read off the sensor,
carried over USB, undistorted, thresholded, decoded, and run through
``solvePnP``. Tens of milliseconds is ordinary. So by the time the answer
exists, **it is an answer about where the arm was, not where it is.**

The naive thing is to fuse it anyway
------------------------------------
Apply that observation as though it described the present and the filter is
told the marker is somewhere it has already left. The induced error is::

    error  ~  velocity  x  latency

which has a nasty shape: **the faster the arm moves, the worse the estimate
gets.** That is exactly backwards from what an extra sensor is supposed to buy,
and it is invisible in any static test. Calibrate at a standstill, verify at a
standstill, and the bug ships.

Measured here on the marker, at 80 ms of latency, this is not a small effect --
it is worse than *ignoring the camera entirely*, which is the result worth
keeping: **a sensor with a mishandled timestamp is worse than no sensor.** One
adds noise you can characterise; the other adds a speed-dependent bias you
cannot.

Rewind, correct, replay
-----------------------
The fix is to stop pretending and use the timestamp. Keep a short history of
filter states, and when an observation arrives:

1. **Rewind** to the state as it was at the moment the shutter opened.
2. **Correct** there, where the observation is actually valid.
3. **Replay** every control step since, from the stored encoder readings.

The corrected history is kept, so a second late observation rewinds onto the
already-corrected timeline rather than undoing the first. This is the standard
answer -- it appears as back-propagation, or state buffering, or
out-of-sequence measurement handling -- and it is exact, at the cost of storing
a buffer and redoing the arithmetic since.

What the camera actually measures
---------------------------------
Not joint angles: a 3D point in the world. So the update is nonlinear, with

    h(q)  =  marker position from forward kinematics
    H     =  [ point_jacobian(q, marker) | 0 ]

and the zero block is the honest part -- a single frame says nothing about
velocity. Velocity only improves because position and velocity are *correlated*
in ``P``, so correcting one moves the other. That correlation is the entire
reason a filter beats a smoother.

The noise is anisotropic, and it matters
----------------------------------------
A single fiducial's depth error is 3-5x its lateral error (see
:mod:`arm.fiducial`), so a spherical ``R`` would be wrong in both directions at
once -- overconfident along the optical axis, underconfident across it.
:func:`camera_noise_world` builds the ellipsoid in camera axes and rotates it
into the world, which lets the filter weight each direction by what the camera
genuinely resolved.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from arm.fiducial import marker_pose_world
from arm.kinematics import point_jacobian
from arm.params import ArmParams, default_params
from arm.sensing import ENCODER_OBSERVATION, KalmanVelocity, encoder_resolution_rad, quantize

__all__ = [
    "FusedEstimator",
    "LinkSideError",
    "MarkerObservation",
    "STRATEGIES",
    "camera_noise_world",
    "encoder_reading",
    "observe_marker",
]

STRATEGIES = ("encoder_only", "naive", "rewind")

# Measured in tests/test_fiducial.py on a 40 mm marker at ~430 mm. Depth is the
# weak axis because it is inferred from apparent size; see arm.fiducial for the
# Z^2/(f*L) derivation these come from.
DEPTH_STD_M = 7.4e-3
LATERAL_STD_M = 1.5e-3


def camera_noise_world(
    pose_world_camera: np.ndarray,
    *,
    depth_std_m: float = DEPTH_STD_M,
    lateral_std_m: float = LATERAL_STD_M,
) -> np.ndarray:
    """A 3x3 measurement covariance, elongated along the optical axis.

    Built diagonally in camera axes -- where the uncertainty ellipsoid actually
    aligns -- and rotated into the world, because that is the frame the filter's
    state lives in. ``R_world = R · diag(...) · Rᵀ``.

    Accepts the camera pose in either convention: only the third column's
    *direction* matters, and a covariance is unchanged by flipping an axis.
    """
    rotation = np.asarray(pose_world_camera, dtype=float)[:3, :3]
    diagonal = np.diag([lateral_std_m**2, lateral_std_m**2, depth_std_m**2])
    return rotation @ diagonal @ rotation.T


@dataclass(frozen=True)
class MarkerObservation:
    """Where the marker was seen, and **when the shutter opened**.

    ``taken_s`` is the whole point of this class. An observation that carries
    only a value is an observation that can only be fused wrongly.
    """

    taken_s: float
    position_world: np.ndarray
    noise: np.ndarray


@dataclass
class _Record:
    """One control step, kept so it can be replayed if the past changes."""

    time_s: float
    state: np.ndarray
    covariance: np.ndarray
    measurement: np.ndarray
    dt: float
    torque: np.ndarray | None


@dataclass
class FusedEstimator:
    """Encoders at control rate, plus camera fixes that arrive late.

    ``strategy`` selects how the latency is handled:

    ``encoder_only``  ignore the camera entirely -- the baseline to beat
    ``naive``         fuse each observation as though it described now
    ``rewind``        rewind to ``taken_s``, correct there, replay since

    The three share every other line of code, so a difference between them is a
    difference in timestamp handling and nothing else.
    """

    params: ArmParams | None = None
    strategy: str = "rewind"
    process_accel_std: float = 20.0
    use_model: bool = False
    history_s: float = 0.5
    encoder_std_rad: np.ndarray | None = None

    _filter: KalmanVelocity = field(init=False)
    _history: deque = field(init=False)
    _time_s: float = field(init=False, default=0.0)
    innovations: list = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected one of {STRATEGIES}")
        self.params = self.params or default_params()
        self._filter = KalmanVelocity(
            process_accel_std=self.process_accel_std,
            use_model=self.use_model,
            params=self.params,
        )
        if self.encoder_std_rad is not None:
            self._filter.measurement_noise = np.diag(np.asarray(self.encoder_std_rad) ** 2)
        self._history = deque()
        self.innovations = []

    # -- the encoder loop ----------------------------------------------------

    def step(
        self, time_s: float, q: np.ndarray, dt: float, tau: np.ndarray | None = None
    ) -> np.ndarray:
        """One control period: predict, fold in the encoder, remember it."""
        measurement = np.asarray(q, dtype=float)
        self._time_s = float(time_s)

        if self._filter.state is None:
            self._filter.update(measurement, dt, tau)
        else:
            self._filter.predict(dt, tau)
            self._filter.correct(
                measurement - ENCODER_OBSERVATION @ self._filter.state,
                ENCODER_OBSERVATION,
                self._filter.measurement_noise,
            )

        state, covariance = self._filter.snapshot()
        self._history.append(
            _Record(
                self._time_s,
                state,
                covariance,
                measurement,
                dt,
                None if tau is None else np.asarray(tau),
            )
        )
        self._trim()
        return state

    def _trim(self) -> None:
        cutoff = self._time_s - self.history_s
        while len(self._history) > 2 and self._history[0].time_s < cutoff:
            self._history.popleft()

    # -- the camera --------------------------------------------------------

    def observe(self, observation: MarkerObservation) -> float:
        """Fold in a camera fix. Returns the innovation magnitude, in metres.

        The innovation is worth watching in its own right: it is how far the
        camera disagreed with the prediction, and under ``naive`` it grows with
        speed while under ``rewind`` it stays at the noise floor. That is the
        whole effect, visible without any ground truth -- which matters,
        because on hardware there is none.
        """
        if self.strategy == "encoder_only" or self._filter.state is None:
            return 0.0

        if self.strategy == "naive":
            # Fuse against the present state, as though the image were current.
            innovation = self._correct_camera(observation)
            self.innovations.append(innovation)
            return innovation

        index = self._index_at(observation.taken_s)
        if index is None:
            return 0.0

        # Rewind to the moment the shutter opened.
        record = self._history[index]
        self._filter.restore((record.state, record.covariance))
        innovation = self._correct_camera(observation)
        record.state, record.covariance = self._filter.snapshot()

        # Replay every control step since, onto the corrected past. The stored
        # snapshots are rewritten as we go, so a later observation rewinds onto
        # this correction rather than undoing it.
        for following in list(self._history)[index + 1 :]:
            self._filter.predict(following.dt, following.torque)
            self._filter.correct(
                following.measurement - ENCODER_OBSERVATION @ self._filter.state,
                ENCODER_OBSERVATION,
                self._filter.measurement_noise,
            )
            following.state, following.covariance = self._filter.snapshot()

        self.innovations.append(innovation)
        return innovation

    def _index_at(self, taken_s: float) -> int | None:
        """The last control step at or before the shutter time.

        Returns None if the observation predates the buffer -- which is the
        honest response to an observation so late that the state it describes
        has already been forgotten. Silently fusing it at the oldest available
        state would reintroduce exactly the error this class exists to avoid.
        """
        if not self._history or taken_s < self._history[0].time_s:
            return None
        for index in range(len(self._history) - 1, -1, -1):
            if self._history[index].time_s <= taken_s:
                return index
        return None

    def _correct_camera(self, observation: MarkerObservation) -> float:
        """One EKF update against a 3D point, linearised at the current state."""
        state = self._filter.state
        predicted = marker_pose_world(self.params.markers[0], state[:3], self.params)[:3, 3]
        jacobian = point_jacobian(state[:3], predicted, self.params)

        observation_matrix = np.hstack([jacobian, np.zeros((3, 3))])
        innovation = np.asarray(observation.position_world, dtype=float) - predicted
        self._filter.correct(innovation, observation_matrix, observation.noise)
        return float(np.linalg.norm(innovation))

    # -- readout -------------------------------------------------------------

    @property
    def state(self) -> np.ndarray | None:
        return self._filter.state

    @property
    def q(self) -> np.ndarray:
        return self._filter.state[:3].copy()

    @property
    def dq(self) -> np.ndarray:
        return self._filter.state[3:].copy()


def observe_marker(
    q_true: np.ndarray,
    taken_s: float,
    pose_world_camera: np.ndarray,
    params: ArmParams | None = None,
    *,
    rng: np.random.Generator | None = None,
    depth_std_m: float = DEPTH_STD_M,
    lateral_std_m: float = LATERAL_STD_M,
) -> MarkerObservation:
    """A synthetic marker fix with realistic anisotropic noise.

    Stands in for render-and-detect where speed matters. The noise is drawn
    from the same covariance the filter is told about, which is the *optimistic*
    case -- real detection noise is not exactly Gaussian, and the numbers here
    should be read as a lower bound on the error rather than a prediction of it.
    """
    params = params or default_params()
    rng = rng or np.random.default_rng(0)
    noise = camera_noise_world(
        pose_world_camera, depth_std_m=depth_std_m, lateral_std_m=lateral_std_m
    )

    truth = marker_pose_world(params.markers[0], q_true, params)[:3, 3]
    perturbation = np.linalg.cholesky(noise) @ rng.standard_normal(3)
    return MarkerObservation(
        taken_s=float(taken_s), position_world=truth + perturbation, noise=noise
    )


def encoder_reading(q_true: np.ndarray, params: ArmParams | None = None) -> np.ndarray:
    """What the encoders would report: the truth, snapped to a count."""
    params = params or default_params()
    return quantize(q_true, encoder_resolution_rad(params))


@dataclass(frozen=True)
class LinkSideError:
    """The encoder measures the *motor*. The camera measures the *link*.

    This is why cameras earn their place on real arms, and it took a
    measurement here to see it. On a perfectly rigid arm the encoder simply
    wins: quantisation puts the marker within **0.149 mm**, against the
    camera's 1.5 mm laterally and 7.4 mm in depth. The filter correctly gives
    the camera almost no weight, latency handling changes nothing, and the
    whole exercise is pointless.

    But an encoder on a Dynamixel's output shaft reports where the *gearbox*
    got to. Between there and the end of a printed PLA link sit gear backlash,
    horn compliance, and the flex of the part itself. That error splits in two,
    and the halves behave completely differently.

    **Deterministic flex** -- the link droops under gravity by ``g(q)/k``, which
    at a plausible 15 N·m/rad is 0.78 deg at the shoulder and **2.9 mm at the
    marker**. This is a *bias*, and the filter cannot remove it. Measured:
    correct latency handling scored 3.84 mm against 4.02 mm for ignoring the
    camera entirely -- essentially no help, because rewinding replays the
    encoder updates and faithfully re-applies their bias every time. The proper
    fixes are the same ones :mod:`arm.sensing` names for friction: identify it
    and model it, or estimate it as an augmented state. Not more filtering.

    **Zero-mean play** -- backlash taken up in whichever direction the joint
    last moved, and measurement noise on top. This the camera *can* fix,
    because averaging works on it, and it is what the rest of this module is
    demonstrated against.

    A degree of either costs 5.87 mm at the marker, which is well past what the
    camera can resolve. So the honest statement of the trade is: **a camera
    does not beat an encoder at measuring a joint. It beats an encoder at
    measuring a link.**
    """

    play_rad: float = np.radians(1.0)
    stiffness_nm_rad: float | None = None

    def deflection(self, q: np.ndarray) -> np.ndarray:
        """Deterministic droop under gravity -- a bias, not noise."""
        if self.stiffness_nm_rad is None:
            return np.zeros(3)
        from arm.dynamics import gravity_torque

        return -gravity_torque(np.asarray(q, dtype=float)) / self.stiffness_nm_rad

    def encoder_reading(
        self, link_q: np.ndarray, rng: np.random.Generator, params: ArmParams | None = None
    ) -> np.ndarray:
        """What the encoder reports, given where the link actually is.

        Note the direction. The link is the physical truth and the encoder is a
        flawed observer of it -- not the other way round. Getting that backwards
        makes the *link* jitter at control rate, which no estimator can help
        with because there is nothing coherent to estimate, and it quietly turns
        the whole experiment into a measurement of the noise you injected.

        ``play_rad`` is modelled as white, matching what the filter is told
        ``R`` is. Real backlash is correlated in time -- it is taken up in
        whichever direction the joint last moved and stays there until a
        reversal -- so this is the optimistic case, and a correlated error would
        be harder for any filter to average away.
        """
        link_q = np.asarray(link_q, dtype=float)
        motor = link_q - self.deflection(link_q)
        noisy = motor + rng.normal(0.0, self.play_rad, 3)
        return quantize(noisy, encoder_resolution_rad(params))

    def position_std_rad(self, q: np.ndarray | None = None) -> np.ndarray:
        """What to tell the filter ``R`` is -- and why it is not quantisation.

        ``R`` means "how well does this measurement pin down the state". The
        state is the *link* angle, and an encoder that can be a degree out about
        the link does not pin it down to a quantisation step. Leaving ``R`` at
        the quantisation floor makes the filter overconfident in precisely the
        sensor that is wrong, which is how a correctly implemented filter
        produces a confidently incorrect answer.
        """
        floor = encoder_resolution_rad() ** 2 / 12.0
        flex = np.zeros(3) if q is None else self.deflection(q) ** 2
        return np.sqrt(self.play_rad**2 + flex + floor)
