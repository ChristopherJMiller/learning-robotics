"""Controller behaviour, including the properties that motivate each one.

These are not smoke tests. Each asserts the specific claim that justifies a
controller existing: that PD *must* droop, that gravity compensation removes
the cause rather than the symptom, that feedforward beats feedback alone, and
that an impedance controller realises the stiffness you asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.control import (
    AccelGains,
    Integrator,
    JointGains,
    cartesian_impedance,
    feedforward_pd,
    pd_torque,
    pd_with_gravity_compensation,
)
from arm.dynamics import gravity_torque
from arm.kinematics import fk, jacobian
from arm.sim import MujocoArm
from arm.trajectory import sinusoid

HOLD_Q = np.array([0.0, 0.9, -0.9])


def _settle(arm, controller, seconds, params, external=None):
    for _ in range(int(seconds / arm.dt)):
        state = arm.read()
        tau = controller(state)
        if external is not None:
            tau = tau + jacobian(state.q, params).T @ external
        arm.write_torque(tau)
        arm.step()
    return arm.read()


# --------------------------------------------------------------------------
# Gains are distinct types on purpose
# --------------------------------------------------------------------------


def test_accel_gains_from_spec_matches_the_requested_response():
    gains = AccelGains.from_spec(bandwidth_hz=3.0, damping_ratio=1.0)
    np.testing.assert_allclose(gains.natural_frequency_rad_s, 2 * np.pi * 3.0, rtol=1e-12)
    np.testing.assert_allclose(gains.damping_ratio, 1.0, rtol=1e-12)


def test_accel_gains_reject_bandwidth_near_the_control_rate():
    """A discrete loop needs many samples per oscillation; 100 Hz at 200 Hz is not."""
    with pytest.raises(ValueError, match="too close to the"):
        AccelGains.from_spec(bandwidth_hz=100.0, control_hz=200.0)


def test_gains_reject_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        JointGains.uniform(-1.0, 0.5)


def test_joint_and_accel_gains_are_not_interchangeable():
    """The distinction that caused a damping ratio of 0.11 during development."""
    assert JointGains is not AccelGains
    assert not isinstance(JointGains.uniform(8.0, 0.6), AccelGains)


# --------------------------------------------------------------------------
# Integral action and windup
# --------------------------------------------------------------------------


def test_integrator_accumulates_then_clamps():
    integrator = Integrator(ki=1.0, limit=0.05)
    error = np.full(3, 1.0)
    for _ in range(1000):
        value = integrator.update(error, dt=0.005)
    np.testing.assert_allclose(value, np.full(3, 0.05))


def test_back_calculation_unwinds_a_saturated_integrator():
    """The whole point: a saturated loop must stop charging.

    Without back-calculation the integrator keeps accumulating while the
    actuator refuses the extra torque, and the stored energy has to be paid
    back as overshoot once the error reverses.
    """
    error = np.full(3, 1.0)
    refused = np.full(3, 0.02)

    naive = Integrator(ki=1.0, limit=10.0, back_calculation=0.0)
    guarded = Integrator(ki=1.0, limit=10.0, back_calculation=1.0)

    for _ in range(200):
        naive.update(error, dt=0.005)
        guarded.update(error, dt=0.005, refused_torque=refused)

    assert np.all(guarded.value < naive.value), "back-calculation did not unwind"
    assert np.all(guarded.value <= 0.05), "guarded integrator still wound up"


def test_integrator_resets():
    integrator = Integrator(ki=1.0, limit=1.0)
    integrator.update(np.ones(3), dt=0.1)
    assert np.any(integrator.value != 0.0)
    integrator.reset()
    np.testing.assert_allclose(integrator.value, np.zeros(3))


# --------------------------------------------------------------------------
# The structural failure of PD, and its cure
# --------------------------------------------------------------------------


def test_pd_alone_droops_under_gravity(params):
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    gains = JointGains.uniform(8.0, 0.6)

    state = _settle(arm, lambda s: pd_torque(s, HOLD_Q, gains), 3.0, params)
    assert np.linalg.norm(state.q - HOLD_Q) > 1e-3, "PD should not hold this pose"


def test_pd_steady_state_error_matches_gravity_over_kp(params):
    """The droop is structural: e_ss = g(q)/kp, not a tuning artefact."""
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    kp = 8.0
    gains = JointGains.uniform(kp, 1.2)

    state = _settle(arm, lambda s: pd_torque(s, HOLD_Q, gains), 6.0, params)
    predicted = -gravity_torque(state.q) / kp
    np.testing.assert_allclose(state.q - HOLD_Q, predicted, atol=2e-3)


def test_gravity_compensation_removes_the_droop(params):
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    gains = JointGains.uniform(8.0, 0.6)

    state = _settle(arm, lambda s: pd_with_gravity_compensation(s, HOLD_Q, gains), 3.0, params)
    np.testing.assert_allclose(state.q, HOLD_Q, atol=1e-6)


def test_feedforward_beats_feedback_alone_on_a_moving_target(params):
    """Feedforward from the plan is the recommended controller for this arm."""
    centre = np.array([0.0, 0.9, -0.7])
    amplitude = np.array([0.6, 0.30, 0.35])
    gains = JointGains.uniform(8.0, 0.6)

    def track(use_feedforward: bool) -> float:
        arm = MujocoArm(params)
        start = sinusoid(0.0, centre, amplitude, 1.0)
        arm.reset(start.q, start.dq)
        errors = []
        for _ in range(int(3.0 / arm.dt)):
            state = arm.read()
            target = sinusoid(state.t, centre, amplitude, 1.0)
            tau = (
                feedforward_pd(state, target, gains, dt=arm.dt)
                if use_feedforward
                else pd_with_gravity_compensation(state, target.q, gains, target.dq)
            )
            arm.write_torque(tau)
            arm.step()
            errors.append(np.linalg.norm(fk(state.q, params) - fk(target.q, params)))
        return float(np.sqrt(np.mean(np.array(errors[len(errors) // 2 :]) ** 2)))

    assert track(True) < track(False) / 10.0, (
        "inverse-dynamics feedforward should cut tracking error by an order of magnitude"
    )


# --------------------------------------------------------------------------
# Impedance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stiffness", [50.0, 100.0, 200.0, 400.0])
def test_impedance_realises_the_commanded_stiffness(stiffness, params):
    """Deflection under a known tip force must equal F/K.

    This is the defining property. A position controller has no equivalent
    number -- its compliance is an accident of gains, configuration and
    inertia rather than something chosen.
    """
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    target = fk(HOLD_Q, params)
    force = np.array([0.0, -2.0, 0.0])
    damping = 2.0 * np.sqrt(stiffness * 0.3)

    state = _settle(
        arm,
        lambda s: cartesian_impedance(s, target, stiffness, damping, params=params),
        5.0,
        params,
        external=force,
    )

    deflection = np.linalg.norm(fk(state.q, params) - target)
    np.testing.assert_allclose(deflection, np.linalg.norm(force) / stiffness, rtol=0.05)


def test_impedance_deflects_along_the_applied_force(params):
    arm = MujocoArm(params)
    arm.reset(HOLD_Q)
    target = fk(HOLD_Q, params)
    force = np.array([0.0, -2.0, 0.0])

    state = _settle(
        arm,
        lambda s: cartesian_impedance(s, target, 100.0, 11.0, params=params),
        5.0,
        params,
        external=force,
    )
    offset = fk(state.q, params) - target
    assert offset[1] < 0.0, "should yield in the direction of the push"
    assert abs(offset[1]) > 5.0 * abs(offset[2]), "deflection should follow the force"


# --------------------------------------------------------------------------
# Saturation must be observable
# --------------------------------------------------------------------------


def test_backend_reports_refused_torque(params):
    arm = MujocoArm(params)
    arm.reset(np.zeros(3))
    applied = arm.write_torque(np.array([5.0, 0.0, 0.0]))

    np.testing.assert_allclose(applied[0], arm.torque_limit[0])
    assert arm.is_saturated
    np.testing.assert_allclose(arm.refused_torque[0], 5.0 - arm.torque_limit[0])


def test_unsaturated_commands_report_nothing_refused(params):
    arm = MujocoArm(params)
    arm.reset(np.zeros(3))
    arm.write_torque(np.array([0.01, 0.01, 0.01]))
    assert not arm.is_saturated
    np.testing.assert_allclose(arm.refused_torque, np.zeros(3))
