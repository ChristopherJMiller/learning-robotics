"""Fusing observations that arrive late.

The three strategies share every line of code except how they treat
``taken_s``, so any difference between them is a difference in timestamp
handling and nothing else. That is what makes these comparisons mean something.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.camera import opencv_from_mujoco, pose_in_world
from arm.fusion import (
    STRATEGIES,
    FusedEstimator,
    LinkSideError,
    MarkerObservation,
    camera_noise_world,
    observe_marker,
)
from arm.kinematics import fk
from arm.sensing import encoder_resolution_rad

CAMERA_HZ = 30.0
SPEED = 2.0


def link_angles(t, speed=SPEED):
    return np.array(
        [
            0.9 + 0.5 * np.sin(speed * t),
            0.45 + 0.3 * np.sin(0.7 * speed * t),
            -1.1 + 0.3 * np.cos(speed * t),
        ]
    )


def simulate(params, strategy, latency, *, seconds=3.0, play_deg=1.0, seed=0):
    """Returns tip RMS error in metres."""
    rng = np.random.default_rng(seed)
    flex = LinkSideError(play_rad=np.radians(play_deg))
    pose_world_camera = opencv_from_mujoco(pose_in_world(params.cameras[0]))
    dt = 1.0 / params.sim.control_hz

    estimator = FusedEstimator(params, strategy=strategy, encoder_std_rad=flex.position_std_rad())
    pending, errors = [], []
    next_frame = 0.0

    for step in range(int(seconds / dt)):
        t = step * dt
        link = link_angles(t)
        estimator.step(t, flex.encoder_reading(link, rng, params), dt, None)

        if t >= next_frame:
            pending.append(
                (t + latency, observe_marker(link, t, pose_world_camera, params, rng=rng))
            )
            next_frame += 1.0 / CAMERA_HZ
        while pending and pending[0][0] <= t:
            estimator.observe(pending.pop(0)[1])

        if t > 1.0:
            errors.append(np.linalg.norm(fk(estimator.q, params) - fk(link, params)))

    return float(np.sqrt(np.mean(np.square(errors))))


# --------------------------------------------------------------------------
# The result this module exists for
# --------------------------------------------------------------------------


def test_with_no_latency_the_camera_helps(params):
    """Otherwise none of the rest is interesting."""
    assert simulate(params, "rewind", 0.0) < simulate(params, "encoder_only", 0.0)


def test_naive_fusion_degrades_with_latency(params):
    """The error grows with ``velocity x latency``, without bound."""
    errors = [simulate(params, "naive", latency) for latency in (0.0, 0.08, 0.24)]
    assert errors[0] < errors[1] < errors[2]
    assert errors[2] > 3 * errors[0], f"expected a large effect, got {errors}"


def test_rewind_is_insensitive_to_latency(params):
    """Flat, because the observation is applied where it was actually valid."""
    errors = [simulate(params, "rewind", latency) for latency in (0.0, 0.08, 0.24)]
    assert max(errors) - min(errors) < 0.4e-3, f"expected flat, got {errors}"


def test_naive_fusion_becomes_worse_than_no_camera_at_all(params):
    """The result worth keeping: a mishandled timestamp is worse than no sensor.

    One adds noise that can be characterised; the other adds a speed-dependent
    bias that cannot. 80 ms is an ordinary USB camera pipeline, not a
    pathological one.
    """
    assert simulate(params, "naive", 0.08) > simulate(params, "encoder_only", 0.08)


def test_rewind_converges_to_the_encoder_baseline_rather_than_past_it(params):
    """Late information is worth less and less, and that is all.

    An observation about further in the past has less left to say once the
    encoders have already covered the interval, so the benefit decays to
    nothing -- but it decays *to* the baseline rather than through it, which is
    the difference between a sensor that stops helping and one that starts
    hurting.

    Not asserted as "never worse", because that would overclaim: when the
    camera's information content has decayed to nothing, folding it in still
    contributes its own noise, so rewind sits a fraction of a percent above the
    baseline at long latency. The point is that it *converges* while naive
    diverges without bound.
    """
    baseline = simulate(params, "encoder_only", 0.0, seconds=6.0)
    errors = [simulate(params, "rewind", latency, seconds=6.0) for latency in (0.04, 0.08, 0.24)]

    for error in errors:
        assert error == pytest.approx(baseline, rel=0.02), f"{error} vs {baseline}"

    # And unlike naive, it does not keep climbing.
    assert max(errors) - min(errors) < 0.05 * baseline


def test_zero_latency_makes_the_strategies_agree(params):
    """With nothing to mishandle, naive and rewind must be identical."""
    assert simulate(params, "naive", 0.0) == pytest.approx(simulate(params, "rewind", 0.0))


# --------------------------------------------------------------------------
# The estimator's parts
# --------------------------------------------------------------------------


def test_an_observation_older_than_the_buffer_is_refused(params):
    """The honest response, rather than fusing it at the oldest state kept.

    Silently clamping to the front of the buffer would reintroduce exactly the
    error this class exists to remove, and would do it invisibly.
    """
    estimator = FusedEstimator(params, strategy="rewind", history_s=0.05)
    dt = 1.0 / params.sim.control_hz
    for step in range(200):
        estimator.step(step * dt, link_angles(step * dt), dt, None)

    before = estimator.state
    stale = MarkerObservation(
        taken_s=0.0, position_world=np.array([1.0, 1.0, 1.0]), noise=np.eye(3) * 1e-6
    )
    assert estimator.observe(stale) == 0.0
    np.testing.assert_array_equal(estimator.state, before)


def test_a_camera_fix_moves_the_estimate_toward_the_truth(params):
    """One EKF update against a 3D point, in isolation."""
    rng = np.random.default_rng(0)
    flex = LinkSideError(play_rad=np.radians(3.0))
    pose_world_camera = opencv_from_mujoco(pose_in_world(params.cameras[0]))
    dt = 1.0 / params.sim.control_hz
    link = np.array([0.9, 0.45, -1.1])

    estimator = FusedEstimator(params, strategy="rewind", encoder_std_rad=flex.position_std_rad())
    for step in range(400):
        estimator.step(step * dt, flex.encoder_reading(link, rng, params), dt, None)

    def error():
        return np.linalg.norm(fk(estimator.q, params) - fk(link, params))

    before = error()
    estimator.observe(observe_marker(link, 400 * dt - 0.08, pose_world_camera, params, rng=rng))
    assert error() < before


def test_encoder_only_ignores_observations_entirely(params):
    estimator = FusedEstimator(params, strategy="encoder_only")
    dt = 1.0 / params.sim.control_hz
    for step in range(50):
        estimator.step(step * dt, link_angles(step * dt), dt, None)

    before = estimator.state
    estimator.observe(
        MarkerObservation(
            taken_s=40 * dt, position_world=np.array([9.0, 9.0, 9.0]), noise=np.eye(3) * 1e-9
        )
    )
    np.testing.assert_array_equal(estimator.state, before)


def test_unknown_strategy_is_refused(params):
    with pytest.raises(ValueError, match="unknown strategy"):
        FusedEstimator(params, strategy="guess")


def test_history_is_bounded(params):
    estimator = FusedEstimator(params, strategy="rewind", history_s=0.1)
    dt = 1.0 / params.sim.control_hz
    for step in range(1000):
        estimator.step(step * dt, link_angles(step * dt), dt, None)
    assert len(estimator._history) <= int(0.1 / dt) + 2


# --------------------------------------------------------------------------
# Noise and geometry
# --------------------------------------------------------------------------


def test_camera_noise_is_elongated_along_the_optical_axis(params):
    """A spherical R would be wrong in both directions at once."""
    pose = opencv_from_mujoco(pose_in_world(params.cameras[0]))
    noise = camera_noise_world(pose)

    axis = pose[:3, 2]  # optical axis in world
    along = float(axis @ noise @ axis)
    across = float(pose[:3, 0] @ noise @ pose[:3, 0])
    assert along > 10 * across


def test_camera_noise_is_a_valid_covariance(params):
    noise = camera_noise_world(opencv_from_mujoco(pose_in_world(params.cameras[0])))
    np.testing.assert_allclose(noise, noise.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(noise) > 0)


def test_camera_noise_ignores_the_axis_convention(params):
    """A covariance is unchanged by flipping an axis, so both conventions agree."""
    mujoco_pose = pose_in_world(params.cameras[0])
    np.testing.assert_allclose(
        camera_noise_world(mujoco_pose),
        camera_noise_world(opencv_from_mujoco(mujoco_pose)),
        atol=1e-18,
    )


def test_the_base_joint_does_not_droop(params):
    """Its axis is vertical, so gravity exerts no torque about it."""
    flex = LinkSideError(play_rad=0.0, stiffness_nm_rad=15.0)
    assert flex.deflection(np.array([0.9, 0.45, -1.1]))[0] == pytest.approx(0.0, abs=1e-12)


def test_link_side_error_widens_r_past_quantisation(params):
    """R means "how well does this pin down the state", and the state is the link."""
    flex = LinkSideError(play_rad=np.radians(1.0))
    assert np.all(flex.position_std_rad() > encoder_resolution_rad(params) / np.sqrt(12.0))


def test_every_strategy_is_exercised():
    assert set(STRATEGIES) == {"encoder_only", "naive", "rewind"}
