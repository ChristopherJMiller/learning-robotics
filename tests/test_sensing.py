"""What the controller measures, and what that costs.

These pin the numbers that justify the sensing model existing: the encoder
quantum, the velocity quantum that differentiating it produces, and the
noise-versus-delay frontier the estimators sit on.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.sensing import (
    AlphaBeta,
    FilteredDifference,
    FiniteDifference,
    SensedArm,
    encoder_resolution_rad,
    quantize,
)
from arm.sim import MujocoArm


def test_encoder_resolution_matches_the_datasheet(params):
    expected = np.array(
        [2 * np.pi / params.actuators[j.actuator].encoder_counts_rev for j in params.joints]
    )
    np.testing.assert_allclose(encoder_resolution_rad(params), expected)
    # 4096 counts is 1.534 mrad; the number the whole module is about.
    np.testing.assert_allclose(expected, 1.5340e-3, atol=1e-6)


def test_differentiating_the_quantum_amplifies_it_by_the_rate(params):
    """The core claim: 1.534 mrad over a 5 ms period is 0.307 rad/s."""
    resolution = encoder_resolution_rad(params)[0]
    dt = 1.0 / params.sim.control_hz
    np.testing.assert_allclose(resolution / dt, 0.3068, atol=1e-3)


def test_quantize_snaps_to_counts():
    resolution = np.full(3, 1.0e-3)
    quantized = quantize(np.array([0.0004, 0.0006, -0.0016]), resolution)
    np.testing.assert_allclose(quantized, [0.0, 0.001, -0.002])


def test_quantize_error_never_exceeds_half_a_count():
    resolution = encoder_resolution_rad()
    rng = np.random.default_rng(0)
    angles = rng.uniform(-3.0, 3.0, size=(500, 3))
    error = np.abs(quantize(angles, resolution) - angles)
    assert np.all(error <= resolution / 2 + 1e-15)


def test_finite_difference_recovers_a_constant_velocity():
    estimator = FiniteDifference()
    dt, velocity = 0.005, np.array([1.0, -2.0, 0.5])
    q = np.zeros(3)
    for _ in range(5):
        q = q + velocity * dt
        estimate = estimator.update(q, dt)
    np.testing.assert_allclose(estimate, velocity, atol=1e-12)


def test_filtered_difference_lags_the_raw_derivative():
    """The trade this module exists to make visible: smoothness costs delay.

    Compared against the unfiltered derivative rather than an absolute
    threshold, because how fast the filter settles depends on the cutoff and
    the sample rate -- a 20 Hz cutoff at 200 Hz has alpha = 0.47 and is already
    within 5% after five samples.
    """
    dt, velocity = 0.005, np.full(3, 1.0)
    raw, filtered = FiniteDifference(), FilteredDifference(cutoff_hz=20.0)

    q = np.zeros(3)
    raw_history, filtered_history = [], []
    for _ in range(400):
        q = q + velocity * dt
        raw_history.append(raw.update(q, dt))
        filtered_history.append(filtered.update(q, dt))

    # The raw derivative is exact from its second sample; the filter is not.
    np.testing.assert_allclose(raw_history[1], velocity, atol=1e-12)
    assert np.all(filtered_history[1] < 0.6 * velocity), "filter shows no lag"
    assert np.all(filtered_history[3] < velocity), "filter should still be catching up"

    # Both converge eventually.
    np.testing.assert_allclose(filtered_history[-1], velocity, rtol=1e-3)


@pytest.mark.parametrize("estimator", [FiniteDifference(), FilteredDifference(), AlphaBeta()])
def test_estimators_start_at_zero_and_reset(estimator):
    first = estimator.update(np.array([1.0, 2.0, 3.0]), 0.005)
    np.testing.assert_allclose(first, np.zeros(3))
    estimator.update(np.array([1.1, 2.1, 3.1]), 0.005)
    estimator.reset()
    np.testing.assert_allclose(estimator.update(np.array([5.0, 5.0, 5.0]), 0.005), np.zeros(3))


def test_alpha_beta_is_quieter_than_raw_differencing_on_quantised_input(params):
    """Both track; only one is usable as a derivative term."""
    resolution = encoder_resolution_rad(params)
    dt = 0.005
    truth = np.array([0.0, 0.0, 0.0])
    velocity = np.full(3, 0.4)

    raw, smooth = FiniteDifference(), AlphaBeta()
    raw_estimates, smooth_estimates = [], []
    for _ in range(400):
        truth = truth + velocity * dt
        measured = quantize(truth, resolution)
        raw_estimates.append(raw.update(measured, dt))
        smooth_estimates.append(smooth.update(measured, dt))

    raw_noise = np.std(np.array(raw_estimates[100:]), axis=0)
    smooth_noise = np.std(np.array(smooth_estimates[100:]), axis=0)
    assert np.all(smooth_noise < raw_noise / 2.0)


# --------------------------------------------------------------------------
# The wrapper is itself a backend
# --------------------------------------------------------------------------


def test_sensed_arm_quantises_what_the_controller_reads(params):
    arm = SensedArm(MujocoArm(params), params)
    arm.reset(np.array([0.123456, 0.654321, -0.999999]))
    measured = arm.read().q
    resolution = encoder_resolution_rad(params)
    np.testing.assert_allclose(measured, quantize(measured, resolution), atol=1e-15)


def test_sensed_arm_exposes_truth_separately(params):
    """Measuring error needs ground truth; a controller must never read it."""
    plant = MujocoArm(params)
    arm = SensedArm(plant, params)
    q = np.array([0.1234567, 0.5, -0.5])
    arm.reset(q)

    np.testing.assert_allclose(arm.true_state().q, q, atol=1e-12)
    assert not np.allclose(arm.read().q, q, atol=1e-9), "measurement should differ"


def test_sensed_arm_passes_torque_through(params):
    plant = MujocoArm(params)
    arm = SensedArm(plant, params)
    arm.reset(np.zeros(3))
    applied = arm.write_torque(np.array([5.0, 0.0, 0.0]))
    np.testing.assert_allclose(applied[0], plant.torque_limit[0])
    assert arm.is_saturated


def test_friction_variant_does_not_disturb_the_baseline(params):
    """The committed configuration stays the verified-equivalent baseline."""
    variant = params.with_friction(0.05)
    assert all(a.coulomb_friction_nm == 0.05 for a in variant.actuators.values())
    assert all(a.coulomb_friction_nm == 0.0 for a in params.actuators.values())

    np.testing.assert_allclose(MujocoArm(variant, from_params=True).model.dof_frictionloss, 0.05)
    np.testing.assert_allclose(MujocoArm(params).model.dof_frictionloss, 0.0)
