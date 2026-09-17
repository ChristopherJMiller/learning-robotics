"""Joint-space and task-space controllers.

Design notes, because the obvious answer is wrong for this arm
-------------------------------------------------------------
Reflected rotor inertia dominates link inertia here: the armature is about 2.4x
the shoulder's link inertia and 19x the elbow's. The consequence is measurable
and it drives every choice below --

    inertia variation across the workspace     coupling
    direct drive        25.4x                    2.55
    geared (ours)        1.48x                   0.30

**The gearbox linearises and decouples the plant.** Cancelling a
configuration-dependent ``M(q)`` is the entire selling point of computed torque,
so when ``M`` barely varies, computed torque is solving a problem we do not
have. This is why industrial robots ran independent-joint PID for decades, and
why direct-drive and quasi-direct-drive machines suddenly needed whole-body
torque control -- they discarded the gearbox that was linearising everything.

So the controller to reach for here is :func:`feedforward_pd`, not
:func:`computed_torque`. Both are provided, because the comparison is the point,
and because the day this arm gains a direct-drive joint the answer flips.

Why feedforward beats inverse dynamics on hardware
--------------------------------------------------
:func:`computed_torque` evaluates the model at the *measured* state, pushing
quantised encoder counts and numerically differentiated velocity through
``M(q)`` and the Coriolis terms -- amplifying noise through the model.
:func:`feedforward_pd` evaluates it at the *planned* state, which is smooth,
known, and could be computed offline. The feedback path then stays simple.
Given ``M`` varies only 1.5x here, that gives up almost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm.dynamics import (
    damping_torque,
    friction_torque,
    gravity_torque,
    inverse_dynamics,
    mass_matrix,
)
from arm.hardware import ArmState
from arm.kinematics import jacobian
from arm.params import ArmParams, default_params
from arm.trajectory import Setpoint

__all__ = [
    "AccelGains",
    "Integrator",
    "JointGains",
    "cartesian_impedance",
    "computed_torque",
    "feedforward_pd",
    "pd_torque",
    "pd_with_gravity_compensation",
]


def _as_triple(value, name: str) -> np.ndarray:
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.size == 1:
        array = np.full(3, float(array[0]))
    if array.shape != (3,):
        raise ValueError(f"{name} must be a scalar or 3-vector, got {array.shape}")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


@dataclass(frozen=True)
class JointGains:
    """PD gains in **torque** units: kp in N·m/rad, kd in N·m·s/rad.

    Distinct from :class:`AccelGains` on purpose. Both hold three numbers, so
    nothing but the type can prevent passing one where the other belongs -- and
    doing so silently produced a damping ratio of 0.11 during development, which
    looked exactly like a controller that simply did not work.

    Closed-loop behaviour depends on ``M(q)``:
    ``ωₙ = √(kp/M)``, ``ζ = kd / (2√(kp·M))``.
    """

    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "kp", _as_triple(self.kp, "kp"))
        object.__setattr__(self, "kd", _as_triple(self.kd, "kd"))

    @staticmethod
    def uniform(kp: float, kd: float) -> JointGains:
        return JointGains(kp=kp, kd=kd)

    def describe(self, params: ArmParams | None = None) -> str:
        """Report the implied response, using the mean inertia over the workspace."""
        inertia = np.diag(mass_matrix(np.zeros(3)))
        omega = np.sqrt(self.kp / inertia)
        zeta = self.kd / (2.0 * np.sqrt(self.kp * inertia))
        return (
            f"wn ~ {np.round(omega, 1)} rad/s "
            f"({np.round(omega / (2 * np.pi), 2)} Hz), zeta ~ {np.round(zeta, 2)}"
        )


@dataclass(frozen=True)
class AccelGains:
    """Gains in **acceleration** units for inverse-dynamics control.

    kp in 1/s², kd in 1/s. Because ``M(q)`` is cancelled, the closed-loop error
    obeys ``ë + kd·ė + kp·e = 0`` independently of configuration -- which is
    what lets one set of gains work across the whole workspace, and what lets
    them be *derived from a specification* instead of tuned by hand.
    """

    kp: np.ndarray
    kd: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "kp", _as_triple(self.kp, "kp"))
        object.__setattr__(self, "kd", _as_triple(self.kd, "kd"))

    @classmethod
    def from_spec(
        cls, bandwidth_hz: float, damping_ratio: float = 1.0, control_hz: float = 200.0
    ) -> AccelGains:
        """Gains from a stated closed-loop response.

        ``kp = ωₙ²``, ``kd = 2ζωₙ``. Requesting a bandwidth near the control
        rate is refused: a discrete loop needs many samples per oscillation, and
        past roughly a tenth of the sample rate the continuous-time design stops
        predicting what the discrete system does.
        """
        if bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive")
        if bandwidth_hz > control_hz / 10.0:
            raise ValueError(
                f"bandwidth_hz={bandwidth_hz} is too close to the {control_hz} Hz "
                f"control rate; keep it below {control_hz / 10:.0f} Hz so the "
                "discrete loop has enough samples per oscillation"
            )
        omega = 2.0 * np.pi * bandwidth_hz
        return cls(kp=omega**2, kd=2.0 * damping_ratio * omega)

    @property
    def natural_frequency_rad_s(self) -> np.ndarray:
        return np.sqrt(self.kp)

    @property
    def damping_ratio(self) -> np.ndarray:
        return self.kd / (2.0 * np.sqrt(self.kp))


@dataclass
class Integrator:
    """Clamped integral action with back-calculation anti-windup.

    Integral action exists to discover a constant bias by accumulating error.
    Gravity *is* a constant bias, so a gravity-ignorant PID spends its
    integrator continuously rediscovering ``g(q)`` -- a large quantity, hence a
    large accumulated state, hence windup whenever the actuator saturates or the
    motion reverses.

    **Feeding gravity forward is the structural cure**; what remains for the
    integrator is residual friction and model error, which is small enough that
    ``ki`` can stay small and the clamp can stay tight.

    Two tactical defences on top:

    * the accumulated term is hard-clamped to ``limit``;
    * ``back_calculation`` unwinds the integrator in proportion to how much
      torque the actuator actually refused, so a saturated loop stops charging.
    """

    ki: np.ndarray
    limit: np.ndarray
    back_calculation: float = 1.0
    _accumulated: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.ki = _as_triple(self.ki, "ki")
        self.limit = _as_triple(self.limit, "limit")

    def reset(self) -> None:
        self._accumulated = np.zeros(3)

    @property
    def value(self) -> np.ndarray:
        return self._accumulated.copy()

    def update(
        self, error: np.ndarray, dt: float, refused_torque: np.ndarray | None = None
    ) -> np.ndarray:
        """Advance the integrator and return its torque contribution.

        ``refused_torque`` is ``commanded - applied`` from the previous cycle:
        nonzero only while saturated. Without it the integrator cannot tell the
        difference between "the arm is slow to respond" and "the actuator has
        nothing left to give", which is exactly when windup happens.
        """
        self._accumulated = self._accumulated + self.ki * np.asarray(error) * dt

        if refused_torque is not None:
            self._accumulated = self._accumulated - self.back_calculation * np.asarray(
                refused_torque
            )

        self._accumulated = np.clip(self._accumulated, -self.limit, self.limit)
        return self._accumulated.copy()


# ---------------------------------------------------------------------------
# Joint-space controllers
# ---------------------------------------------------------------------------


def pd_torque(
    state: ArmState,
    q_desired: np.ndarray,
    gains: JointGains,
    dq_desired: np.ndarray | None = None,
) -> np.ndarray:
    """Plain PD. Cannot hold a pose against gravity without steady-state error.

    Its only source of torque is ``kp·e``, so it must first permit the error
    that generates the load it is resisting: ``e_ss = g(q) / kp``. Raising kp
    shrinks that but never removes it, and buys stiffness and noise sensitivity.
    """
    q_desired = np.asarray(q_desired, dtype=float)
    dq_desired = np.zeros(3) if dq_desired is None else np.asarray(dq_desired, float)
    return gains.kp * (q_desired - state.q) + gains.kd * (dq_desired - state.dq)


def pd_with_gravity_compensation(
    state: ArmState,
    q_desired: np.ndarray,
    gains: JointGains,
    dq_desired: np.ndarray | None = None,
) -> np.ndarray:
    """PD plus a feed-forward term cancelling gravity at the *measured* pose.

    Evaluated at ``state.q`` rather than ``q_desired`` so the compensation stays
    correct while moving, not only once the arm has arrived.
    """
    return pd_torque(state, q_desired, gains, dq_desired) + gravity_torque(state.q)


def feedforward_pd(
    state: ArmState,
    setpoint: Setpoint,
    gains: JointGains,
    integrator: Integrator | None = None,
    dt: float = 0.005,
    refused_torque: np.ndarray | None = None,
    params: ArmParams | None = None,
) -> np.ndarray:
    """Inverse-dynamics feedforward from the **plan**, plus PD on the error.

    The recommended controller for this arm. The feedforward term is computed
    entirely from the planned trajectory, so it introduces no sensor noise and
    could be precomputed offline; the feedback term only has to reject what the
    model missed.

    ``params`` is what the controller *believes* about the arm, which need not
    match the plant. Passing a model whose friction has been identified is how
    a measurement turns into better tracking.
    """
    feedforward = (
        inverse_dynamics(setpoint.q, setpoint.dq, setpoint.ddq)
        + damping_torque(setpoint.dq)
        + friction_torque(setpoint.dq, params=params)
    )
    feedback = gains.kp * (setpoint.q - state.q) + gains.kd * (setpoint.dq - state.dq)

    if integrator is not None:
        feedback = feedback + integrator.update(setpoint.q - state.q, dt, refused_torque)
    return feedforward + feedback


def computed_torque(
    state: ArmState,
    setpoint: Setpoint,
    gains: AccelGains,
    params: ArmParams | None = None,
) -> np.ndarray:
    """Inverse-dynamics control evaluated at the **measured** state.

    ``τ = M(q)·(q̈_d + kp·e + kd·ė) + C(q,q̇)q̇ + g(q) + friction``

    Theoretically the stronger controller -- it cancels the true ``M(q)`` rather
    than the planned one -- but it differentiates measured state through the
    model, so on hardware it amplifies encoder noise. Worth having, worth
    measuring against :func:`feedforward_pd`, and genuinely the right choice
    once a joint is direct-drive.
    """
    commanded_acceleration = (
        setpoint.ddq + gains.kp * (setpoint.q - state.q) + gains.kd * (setpoint.dq - state.dq)
    )
    # rnea(q, dq, 0) is exactly the Coriolis + gravity bias.
    bias = inverse_dynamics(state.q, state.dq, np.zeros(3))
    return (
        mass_matrix(state.q) @ commanded_acceleration
        + bias
        + damping_torque(state.dq)
        + friction_torque(state.dq, params=params)
    )


# ---------------------------------------------------------------------------
# Task-space control
# ---------------------------------------------------------------------------


def cartesian_impedance(
    state: ArmState,
    position_desired: np.ndarray,
    stiffness_n_m,
    damping_ns_m,
    velocity_desired: np.ndarray | None = None,
    params: ArmParams | None = None,
) -> np.ndarray:
    """Make the end effector behave like a spring-damper in Cartesian space.

    ``τ = Jᵀ · (K·(x_d − x) + D·(ẋ_d − ẋ)) + g(q)``

    This is what torque control actually unlocks, and it is a different way of
    thinking about the problem. Instead of demanding a position and rejecting
    everything that opposes it, you specify a *stiffness* -- in newtons per
    metre, a physical quantity -- and let the arm yield when pushed.

    Consequences worth knowing:

    * Contact stops being a failure mode. A stiff position controller responds
      to an obstacle by increasing torque until something breaks; an impedance
      controller pushes with a bounded, chosen force.
    * It is safe to be near. Stiffness is the thing a person feels.
    * Steady-state position error under load is intended behaviour, not droop:
      a 200 N/m spring holding a 1 N load sits 5 mm off target by definition.

    The ``Jᵀ`` is the force-duality relation earning its keep: the transpose of
    the Jacobian maps a desired tip force to the joint torques that produce it,
    which falls out of conservation of power.

    Near a singularity ``J`` loses rank and a whole Cartesian direction becomes
    uncontrollable -- but note this degrades *gracefully*, unlike ``J⁻¹``
    schemes which blow up. The arm simply cannot push that way.
    """
    from arm.kinematics import fk

    params = params or default_params()
    stiffness = _as_triple(stiffness_n_m, "stiffness_n_m")
    damping = _as_triple(damping_ns_m, "damping_ns_m")

    position_desired = np.asarray(position_desired, dtype=float)
    velocity_desired = (
        np.zeros(3) if velocity_desired is None else np.asarray(velocity_desired, float)
    )

    jac = jacobian(state.q, params)
    position = fk(state.q, params)
    velocity = jac @ state.dq

    wrench = stiffness * (position_desired - position) + damping * (velocity_desired - velocity)
    return jac.T @ wrench + gravity_torque(state.q)
