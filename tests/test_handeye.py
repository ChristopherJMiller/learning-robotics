"""Hand-eye calibration, scored against the camera pose we planted.

The whole point of these tests is that ``arm.handeye`` never reads
``params.cameras``. It sees arm configurations and rendered images, and has to
work out where the camera is. ``truth()`` exists only to grade the answer.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from arm.fiducial import viewing_angle_deg
from arm.frames import inverse, se3
from arm.handeye import (
    METHODS,
    MIN_AXIS_SPREAD,
    Calibration,
    DegenerateGeometryError,
    average_poses,
    axis_spread,
    calibrate,
    collect,
    marker_mount,
    pose_error,
    refine,
    reprojection_residuals,
    search_configurations,
    truth,
)
from arm.kinematics import link_frames

# Rendering is not cheap, so the shared observation sets are module-scoped.


@pytest.fixture(scope="module")
def observations(params):
    found = collect(
        search_configurations(params, count=40, rng=np.random.default_rng(1), max_attempts=2000),
        params,
    )
    assert len(found) >= 10, f"only {len(found)} usable observations"
    return found


@pytest.fixture(scope="module")
def degenerate(params):
    """Poses with the base joint pinned, so every rotation shares one axis."""
    rng = np.random.default_rng(3)
    lower = np.array([joint.limit_lower_rad for joint in params.joints])
    upper = np.array([joint.limit_upper_rad for joint in params.joints])
    marker, camera = params.markers[0], params.cameras[0].pos_m

    configurations = []
    while len(configurations) < 30:
        q = lower + rng.random(3) * (upper - lower)
        q[0] = 1.0
        if viewing_angle_deg(marker, q, camera, params) <= 45.0:
            configurations.append(q)
    found = collect(configurations, params)
    assert len(found) >= 6
    return found


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------


def test_the_camera_is_recovered_without_being_told(params, observations):
    """The headline: find a camera nobody measured, to under a millimetre."""
    pose_base_camera, _ = truth(params)
    calibration = refine(calibrate(observations, params), observations, params)

    position, angle = pose_error(calibration.pose_base_camera, pose_base_camera)
    assert position < 2e-3, f"{position * 1000:.2f} mm"
    assert angle < 0.2, f"{angle:.3f} deg"


def test_the_marker_mounting_comes_free(params, observations):
    """Calibration recovers *where the marker actually is* as a by-product.

    Worth having in its own right: on hardware the marker is stuck on by hand
    and its offset from the link frame is as unmeasured as the camera pose.
    """
    _, pose_link_marker = truth(params)
    calibration = refine(calibrate(observations, params), observations, params)

    position, angle = pose_error(calibration.pose_link_marker, pose_link_marker)
    assert position < 2e-3, f"{position * 1000:.2f} mm"
    assert angle < 0.5, f"{angle:.3f} deg"


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_converges_to_the_same_answer_after_refinement(params, observations, method):
    """The closed-form solvers are initialisers, and it stops mattering which.

    Before refinement these disagree by a factor of five. After it they land on
    the same answer, which is why a real pipeline uses a linear method for a
    starting guess and nonlinear least squares for the result.
    """
    pose_base_camera, _ = truth(params)
    refined = refine(calibrate(observations, params, method=method), observations, params)

    position, angle = pose_error(refined.pose_base_camera, pose_base_camera)
    assert position < 2e-3, f"{method}: {position * 1000:.2f} mm"
    assert angle < 0.2, f"{method}: {angle:.3f} deg"


def test_refinement_improves_on_the_closed_form(params, observations):
    pose_base_camera, _ = truth(params)
    closed = calibrate(observations, params)
    refined = refine(closed, observations, params)

    before, _ = pose_error(closed.pose_base_camera, pose_base_camera)
    after, _ = pose_error(refined.pose_base_camera, pose_base_camera)
    assert after < before


def test_refinement_lowers_the_reprojection_residual(params, observations):
    """It is minimising pixels, so this is the thing it is guaranteed to improve."""
    closed = calibrate(observations, params)
    refined = refine(closed, observations, params)

    def rms(calibration):
        residuals = reprojection_residuals(
            calibration.pose_base_camera, calibration.pose_link_marker, observations, params
        )
        return float(np.sqrt(np.mean(residuals**2)))

    assert rms(refined) < rms(closed)


def test_calibration_never_reads_the_declared_camera_pose(params, observations):
    """Move the camera in the parameters; the answer must not follow.

    A calibration that quietly consulted ``params.cameras`` would pass every
    accuracy test above while doing nothing at all.
    """
    moved = params.model_copy(
        update={"cameras": (params.cameras[0].model_copy(update={"pos_m": (1.5, -2.0, 3.0)}),)}
    )
    baseline = calibrate(observations, params).pose_base_camera
    shifted = calibrate(observations, moved).pose_base_camera
    np.testing.assert_allclose(baseline, shifted, atol=1e-12)


# --------------------------------------------------------------------------
# Observability -- the part that bites
# --------------------------------------------------------------------------


def test_parallel_rotation_axes_are_refused(params, degenerate):
    """Pinning the base joint makes the answer unobservable, and it must say so.

    Joints 2 and 3 are both pitch, so with the base joint held still every
    relative rotation is about the same axis and the camera position along that
    axis is constrained by nothing.
    """
    assert axis_spread(degenerate, params) < MIN_AXIS_SPREAD
    with pytest.raises(DegenerateGeometryError, match="near-parallel"):
        calibrate(degenerate, params)


def test_a_varied_pose_set_is_observable(params, observations):
    assert axis_spread(observations, params) > 10 * MIN_AXIS_SPREAD


def test_forcing_a_degenerate_solve_is_badly_wrong(params, degenerate):
    """And the error is metres, not millimetres -- this is not a small effect."""
    pose_base_camera, _ = truth(params)
    forced = calibrate(degenerate, params, method="tsai", require_observability=False)

    position, _ = pose_error(forced.pose_base_camera, pose_base_camera)
    assert position > 0.1, f"expected a large error, got {position * 1000:.1f} mm"


def test_a_small_residual_does_not_mean_a_correct_answer(params, degenerate):
    """The lesson worth keeping: consistency is not correctness.

    Refining the degenerate set drives the camera metres away from the truth
    while *lowering* the reprojection residual. The optimiser is not at fault --
    the data genuinely does not constrain that direction, so moving along it is
    free, and it buys a slightly better fit to the noise.
    """
    pose_base_camera, _ = truth(params)
    forced = calibrate(degenerate, params, method="tsai", require_observability=False)
    refined = refine(forced, degenerate, params)

    def rms(calibration):
        residuals = reprojection_residuals(
            calibration.pose_base_camera, calibration.pose_link_marker, degenerate, params
        )
        return float(np.sqrt(np.mean(residuals**2)))

    position, _ = pose_error(refined.pose_base_camera, pose_base_camera)
    assert position > 0.1, "expected a wildly wrong camera"
    assert rms(refined) < 1.0, "and yet it fits the images sub-pixel"


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def test_observations_carry_no_knowledge_of_the_camera(params, observations):
    """Each one is an encoder reading and an image measurement, nothing more."""
    for observation in observations[:3]:
        assert observation.q.shape == (3,)
        assert observation.corners_px.shape == (4, 2)
        assert observation.reprojection_error_px < 1.0


def test_pose_base_link_matches_forward_kinematics(params, observations):
    observation = observations[0]
    index = [link.name for link in params.links].index(params.markers[0].link)
    np.testing.assert_allclose(
        observation.pose_base_link(params), link_frames(observation.q, params)[index]
    )


def test_search_finds_only_readable_poses(params):
    """Rejection sampling on viewing angle, which needs no render to evaluate."""
    marker, camera = params.markers[0], params.cameras[0].pos_m
    for q in search_configurations(params, count=15, rng=np.random.default_rng(7)):
        assert viewing_angle_deg(marker, q, camera, params) <= 45.0


def test_too_few_poses_is_refused(params, observations):
    with pytest.raises(ValueError, match="at least 3"):
        calibrate(observations[:2], params)


def test_unknown_method_is_refused(params, observations):
    with pytest.raises(ValueError, match="unknown method"):
        calibrate(observations, params, method="nonesuch")


def test_average_poses_recovers_a_common_pose():
    rng = np.random.default_rng(0)
    base = se3(cv2.Rodrigues(np.array([0.3, -0.2, 0.7]))[0], np.array([0.1, 0.2, 0.3]))

    jittered = []
    for _ in range(50):
        delta = se3(cv2.Rodrigues(rng.normal(0, 0.01, 3))[0], rng.normal(0, 0.001, 3))
        jittered.append(base @ delta)

    position, angle = pose_error(average_poses(jittered), base)
    assert position < 1e-3
    assert angle < 0.5


def test_marker_mount_matches_the_declared_offset(params):
    marker = params.markers[0]
    np.testing.assert_allclose(marker_mount(marker)[:3, 3], marker.pos_m)


def test_residuals_are_per_observation(params, observations):
    calibration = refine(calibrate(observations, params), observations, params)
    assert calibration.residuals(observations, params).shape == (len(observations),)
    assert reprojection_residuals(
        calibration.pose_base_camera, calibration.pose_link_marker, observations, params
    ).shape == (8 * len(observations),)


def test_the_chain_closes_on_the_recovered_transforms(params, observations):
    """The equation this module exists to satisfy, checked directly.

    It closes to a few millimetres rather than a few microns, and the shape of
    what is left over says why: the leftover is **along the optical axis**, not
    across it. Measured on the worst observation here, 8.32 mm of an 8.33 mm
    discrepancy is depth -- and that same observation's marker detection is
    independently 8.33 mm from ground truth, all of it depth.

    So this is not calibration error at all. It is the depth conditioning of a
    single fiducial (see :mod:`arm.fiducial`) showing up in the one place that
    can see it, and the calibration averaged it away rather than inheriting it:
    the recovered camera is good to 0.66 mm while the observations feeding it
    are individually several times worse.
    """
    calibration = refine(calibrate(observations, params), observations, params)
    pose_camera_base = inverse(calibration.pose_base_camera)

    depth, lateral = [], []
    for observation in observations:
        predicted = (
            pose_camera_base @ observation.pose_base_link(params) @ calibration.pose_link_marker
        )
        # In the camera frame the optical axis *is* z, so the split is direct.
        error = predicted[:3, 3] - observation.pose_camera_marker[:3, 3]
        depth.append(abs(error[2]))
        lateral.append(float(np.linalg.norm(error[:2])))

    assert max(depth) < 15e-3, f"depth {max(depth) * 1000:.2f} mm"
    assert max(lateral) < 3e-3, f"lateral {max(lateral) * 1000:.2f} mm"
    assert np.mean(depth) > 2 * np.mean(lateral), "expected the residual to be depth-dominated"


def test_calibration_reports_what_produced_it(params, observations):
    calibration = calibrate(observations, params, method="li")
    assert isinstance(calibration, Calibration)
    assert calibration.method == "li"
    assert calibration.observations == len(observations)
    assert refine(calibration, observations, params).method == "li+refined"
