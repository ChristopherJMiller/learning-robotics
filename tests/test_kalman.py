"""The Kalman filter, and the properties that make it more than a tuned gain.

Everything here is about the claims in the module docstring being true rather
than merely plausible: that the covariance never sees the data, that steady
state reduces to alpha-beta, and that the measurement noise comes from the
datasheet rather than from tuning.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.dynamics import forward_dynamics, inverse_dynamics
from arm.sensing import (
    AlphaBeta,
    KalmanVelocity,
    encoder_resolution_rad,
    quantize,
)

DT = 0.005


# --------------------------------------------------------------------------
# R comes from the datasheet, not from tuning
# --------------------------------------------------------------------------


def test_measurement_noise_is_the_quantisation_variance(params):
    """A uniform distribution over one encoder count has variance d^2/12."""
    resolution = encoder_resolution_rad(params)[0]
    expected = resolution**2 / 12.0

    filt = KalmanVelocity(params=params)
    np.testing.assert_allclose(np.diag(filt._R), expected, rtol=1e-12)
    np.testing.assert_allclose(expected, 1.9609e-7, rtol=1e-4)


def test_quantisation_error_really_is_uniform_over_a_count(params):
    """The assumption R rests on, checked against the actual quantiser."""
    resolution = encoder_resolution_rad(params)
    rng = np.random.default_rng(0)
    angles = rng.uniform(-3.0, 3.0, size=(20000, 3))

    error = quantize(angles, resolution) - angles
    np.testing.assert_allclose(error.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(error.var(axis=0), resolution**2 / 12.0, rtol=0.05)


# --------------------------------------------------------------------------
# The covariance never sees the data
# --------------------------------------------------------------------------


def test_gain_converges_from_any_starting_confidence():
    """P and K depend only on F, H, Q, R -- never on what was measured.

    This is the property that answers "does uncertainty fall because things
    went well?". It does not. Uncertainty falls because a measurement was
    taken, by an amount fixed in advance by the sensor and the model.
    """
    filt = KalmanVelocity()
    gain = filt.steady_state_gain(DT)

    # Same recursion from a wildly different starting covariance.
    transition = filt._transition(DT)
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    covariance = np.diag(np.concatenate([np.full(3, 1e-2), np.full(3, 100.0)]))

    for _ in range(500):
        covariance = transition @ covariance @ transition.T + filt._process_noise(DT)
        innovation = observation @ covariance @ observation.T + filt._R
        other = covariance @ observation.T @ np.linalg.inv(innovation)
        covariance = (np.eye(6) - other @ observation) @ covariance

    np.testing.assert_allclose(other, gain, rtol=1e-6)


def test_running_the_filter_reproduces_the_predicted_steady_state():
    """The gain derived with no data is the one the filter actually reaches.

    ``steady_state_gain`` iterates the covariance recursion with no measurements
    at all. Running the filter over real (quantised) data must converge to the
    same place, because the recursion the data drives is the identical one.
    """
    filt = KalmanVelocity()
    predicted = filt.steady_state_gain(DT)

    position = np.zeros(3)
    for _ in range(400):
        position = position + np.array([0.3, -0.2, 0.1]) * DT
        filt.update(quantize(position, encoder_resolution_rad()), DT)

    # Recover the gain implied by the filter's own converged covariance, by
    # running one more predict/update step by hand.
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    transition = filt._transition(DT)
    covariance = transition @ filt.covariance @ transition.T + filt._process_noise(DT)
    innovation_covariance = observation @ covariance @ observation.T + filt._R
    reached = covariance @ observation.T @ np.linalg.inv(innovation_covariance)

    np.testing.assert_allclose(reached, predicted, rtol=1e-6)


def test_uncertainty_shrinks_on_measurement_and_grows_without_one():
    """The breathing cycle, asserted directly."""
    filt = KalmanVelocity()
    resolution = encoder_resolution_rad()

    position = np.zeros(3)
    for _ in range(200):
        position = position + np.array([0.2, 0.2, 0.2]) * DT
        filt.update(quantize(position, resolution), DT)

    settled = filt.covariance
    predicted = filt._transition(DT) @ settled @ filt._transition(DT).T + filt._process_noise(DT)

    # Predicting alone increases uncertainty ...
    assert np.diag(predicted)[0] > np.diag(settled)[0]
    # ... and the next measurement brings it back down.
    position = position + np.array([0.2, 0.2, 0.2]) * DT
    filt.update(quantize(position, resolution), DT)
    np.testing.assert_allclose(np.diag(filt.covariance)[0], np.diag(settled)[0], rtol=0.05)


def test_estimate_is_more_certain_than_a_single_measurement():
    """Fusing a model with repeated readings beats the sensor alone."""
    filt = KalmanVelocity()
    resolution = encoder_resolution_rad()

    position = np.zeros(3)
    for _ in range(300):
        position = position + np.array([0.4, 0.4, 0.4]) * DT
        filt.update(quantize(position, resolution), DT)

    measurement_std = resolution / np.sqrt(12.0)
    assert np.all(filt.position_std_rad < measurement_std)


# --------------------------------------------------------------------------
# Steady state is alpha-beta
# --------------------------------------------------------------------------


def test_steady_state_reduces_to_alpha_beta():
    """The converged gain IS the alpha-beta gain pair.

    Which is the honest framing of what alpha-beta was: the same filter with
    the answer guessed instead of computed.
    """
    alpha, beta = KalmanVelocity().equivalent_alpha_beta(DT)
    assert 0.0 < alpha < 1.0
    assert 0.0 < beta < 1.0
    np.testing.assert_allclose([alpha, beta], [0.770, 0.542], atol=5e-3)


def test_converged_filter_tracks_like_its_equivalent_alpha_beta():
    """Empirical confirmation, not just algebraic."""
    alpha, beta = KalmanVelocity().equivalent_alpha_beta(DT)
    kalman = KalmanVelocity()
    equivalent = AlphaBeta(alpha=alpha, beta=beta)
    resolution = encoder_resolution_rad()

    position = np.zeros(3)
    velocity = np.array([0.5, -0.3, 0.2])
    for _ in range(300):
        position = position + velocity * DT
        measured = quantize(position, resolution)
        from_kalman = kalman.update(measured, DT)
        from_alpha_beta = equivalent.update(measured, DT)

    np.testing.assert_allclose(from_kalman, from_alpha_beta, atol=2e-3)


# --------------------------------------------------------------------------
# Estimation quality
# --------------------------------------------------------------------------


def test_tracks_a_constant_velocity_without_bias():
    """The filter is unbiased, not noiseless.

    Asserted on the mean of the settled estimates rather than the last one: with
    the default process noise the instantaneous estimate scatters by around 10%,
    which is the filter correctly reflecting that a quantised position tells you
    very little about velocity in any single sample. Averaging reveals there is
    no systematic error underneath that scatter.
    """
    filt = KalmanVelocity()
    resolution = encoder_resolution_rad()
    velocity = np.array([0.8, -0.5, 0.35])

    position = np.zeros(3)
    estimates = []
    for _ in range(600):
        position = position + velocity * DT
        estimates.append(filt.update(quantize(position, resolution), DT))

    settled = np.array(estimates[200:])
    np.testing.assert_allclose(settled.mean(axis=0), velocity, atol=0.01)
    assert np.all(settled.std(axis=0) < 0.15), "scatter larger than expected"


def test_quieter_than_raw_differencing():
    from arm.sensing import FiniteDifference

    resolution = encoder_resolution_rad()
    kalman, raw = KalmanVelocity(), FiniteDifference()

    position = np.zeros(3)
    kalman_history, raw_history = [], []
    for _ in range(400):
        position = position + np.full(3, 0.4) * DT
        measured = quantize(position, resolution)
        kalman_history.append(kalman.update(measured, DT))
        raw_history.append(raw.update(measured, DT))

    kalman_noise = np.std(np.array(kalman_history[150:]), axis=0)
    raw_noise = np.std(np.array(raw_history[150:]), axis=0)
    # Roughly 2.4x quieter at the default process noise. The filter is tuned
    # fast (sigma_a = 20 rad/s^2, matching the peak acceleration a kinematic
    # model leaves entirely unexplained), so it deliberately keeps a high gain;
    # a smaller sigma_a trades responsiveness for more smoothing.
    assert np.all(kalman_noise < raw_noise / 2.0)


def test_model_based_prediction_uses_the_torque():
    """With use_model the commanded torque changes the estimate; without, it cannot."""
    resolution = encoder_resolution_rad()
    position = np.array([0.0, 0.5, -0.5])
    torque = np.array([0.0, 0.3, 0.1])

    for use_model, should_differ in ((True, True), (False, False)):
        with_torque = KalmanVelocity(use_model=use_model)
        without = KalmanVelocity(use_model=use_model)
        for _ in range(20):
            measured = quantize(position, resolution)
            a = with_torque.update(measured, DT, torque)
            b = without.update(measured, DT, None)
        differs = not np.allclose(a, b, atol=1e-9)
        assert differs is should_differ


def test_reset_clears_the_estimate():
    filt = KalmanVelocity()
    filt.update(np.array([0.1, 0.2, 0.3]), DT)
    filt.update(np.array([0.11, 0.21, 0.31]), DT)
    assert filt.covariance is not None

    filt.reset()
    assert filt.covariance is None
    np.testing.assert_allclose(filt.update(np.zeros(3), DT), np.zeros(3))


def test_covariance_stays_symmetric_and_positive_definite():
    """The Joseph form exists to keep this true under round-off."""
    filt = KalmanVelocity()
    resolution = encoder_resolution_rad()
    rng = np.random.default_rng(3)

    position = np.zeros(3)
    for _ in range(500):
        position = position + rng.normal(0, 0.5, 3) * DT
        filt.update(quantize(position, resolution), DT)
        covariance = filt.covariance
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-18)
        assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


# --------------------------------------------------------------------------
# The dynamics the model-based filter predicts with
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_and_inverse_dynamics_round_trip(seed):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, 3)
    dq = rng.uniform(-1.0, 1.0, 3)
    ddq = rng.uniform(-5.0, 5.0, 3)

    from arm.dynamics import damping_torque

    torque = inverse_dynamics(q, dq, ddq) + damping_torque(dq)
    np.testing.assert_allclose(forward_dynamics(q, dq, torque), ddq, atol=1e-9)
