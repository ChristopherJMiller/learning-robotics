"""Fiducial detection, scored against the pose the simulator actually used.

Everything here compares a *recovered* pose against ground truth the perception
code never sees, which is the only way to know a vision pipeline works.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.camera import SimulatedCamera, opencv_from_mujoco
from arm.fiducial import (
    detect,
    marker_corners_local,
    marker_pose_world,
    marker_texture_path,
    texture_size_m,
)
from arm.frames import inverse
from arm.kinematics import link_frames
from arm.sim import MujocoArm

# Configurations where the marker faces the camera well enough to be read.
# Not every reachable pose qualifies -- see test_detection_fails_at_oblique_angles.
POSES = [
    np.array([1.2, 0.4, -1.0]),
    np.array([1.0, 0.6, -1.2]),
    np.array([0.9, 0.7, -1.3]),
]

# All four corners in frame, but too oblique to decode.
OBLIQUE = np.array([1.4, 0.3, -0.8])


def _capture(params, q):
    arm = MujocoArm(params)
    arm.reset(q)
    return SimulatedCamera(arm.model, arm.data, "workspace", params).capture()


def _pose_error(estimated, truth):
    position = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    relative = inverse(truth) @ estimated
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
    return position, angle


# --------------------------------------------------------------------------
# The link-frame convention, which is where this went wrong
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", POSES)
def test_link_frames_match_the_simulator(params, q):
    """A regression test for a 151 mm error.

    Anything attached to a link -- a fiducial, a sensor, a tool -- hangs off the
    *body* frame, and a body frame includes its own joint's rotation.
    ``fk_frames`` does not: its frames sit at joints and carry only the rotations
    before them. Using one for the other put the marker at the right point with
    the wrong orientation, and the supposed ground truth was 151 mm from where
    the simulator had actually drawn it.
    """
    import mujoco

    arm = MujocoArm(params)
    arm.reset(q)
    mujoco.mj_forward(arm.model, arm.data)

    for index, link in enumerate(params.links):
        body = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, link.name)
        expected_position = arm.data.xpos[body]
        expected_rotation = arm.data.xmat[body].reshape(3, 3)

        frame = link_frames(q, params)[index]
        np.testing.assert_allclose(frame[:3, 3], expected_position, atol=1e-9)
        np.testing.assert_allclose(frame[:3, :3], expected_rotation, atol=1e-9)


def test_link_frames_differ_from_fk_frames(params):
    """So the distinction is real and the test above is worth having."""
    from arm.kinematics import fk_frames

    q = np.array([0.3, 0.5, -0.9])
    body = link_frames(q, params)[2]
    joint = fk_frames(q, params)[2]

    # Same origin ...
    np.testing.assert_allclose(body[:3, 3], joint[:3, 3], atol=1e-12)
    # ... different orientation, by exactly the elbow rotation.
    assert not np.allclose(body[:3, :3], joint[:3, :3], atol=1e-6)


# --------------------------------------------------------------------------
# Geometry of the marker itself
# --------------------------------------------------------------------------


def test_the_texture_is_larger_than_the_black_square(params):
    """The quiet zone is part of the print and not part of ``size_m``.

    Conflating them is a pure scale error in every recovered translation, which
    then looks like a calibration failure rather than a measurement mistake.
    """
    marker = params.markers[0]
    assert texture_size_m(marker) > marker.size_m


def test_the_marker_texture_exists(params):
    """Committed like the meshes: the MJCF references it at load time."""
    for marker in params.markers:
        assert marker_texture_path(marker).exists(), "run `just markers`"


def test_marker_corners_are_the_black_square(params):
    marker = params.markers[0]
    corners = marker_corners_local(marker)

    assert corners.shape == (4, 3)
    np.testing.assert_allclose(corners[:, 2], 0.0)  # planar
    side = np.linalg.norm(corners[0] - corners[1])
    np.testing.assert_allclose(side, marker.size_m, rtol=1e-12)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", POSES)
def test_the_marker_is_found(params, q):
    detections = detect(_capture(params, q), params)
    assert [d.aruco_id for d in detections] == [params.markers[0].aruco_id]


@pytest.mark.parametrize("q", POSES)
def test_the_recovered_pose_is_close_to_the_truth(params, q):
    view = _capture(params, q)
    detection = detect(view, params)[0]

    estimated = opencv_from_mujoco(view.pose_world) @ detection.pose_camera
    truth = marker_pose_world(params.markers[0], q, params)

    position, angle = _pose_error(estimated, truth)
    assert position < 5e-3, f"{position * 1000:.2f} mm"
    assert angle < 2.0, f"{angle:.2f} deg"


@pytest.mark.parametrize("q", POSES)
def test_reprojection_error_is_sub_pixel(params, q):
    """The only cheap signal that a pose is wrong rather than merely uncertain."""
    detection = detect(_capture(params, q), params)[0]
    assert detection.reprojection_error_px < 1.0


def test_depth_is_the_weak_axis(params):
    """Conditioning, not a defect, and worth knowing before trusting a number.

    Depth is inferred from apparent size, so it degrades as ``Z^2 / (f · L)``
    while lateral position degrades linearly. Measured here, depth error is
    several times the lateral error -- which is why a marker twice as far away
    is four times worse, and why the cheapest fix is a bigger marker.
    """
    q = POSES[0]
    view = _capture(params, q)
    detection = detect(view, params)[0]

    estimated = opencv_from_mujoco(view.pose_world) @ detection.pose_camera
    truth = marker_pose_world(params.markers[0], q, params)

    axis = -view.pose_world[:3, 2]  # camera forward, in world
    error = estimated[:3, 3] - truth[:3, 3]
    along = abs(float(np.dot(error, axis)))
    across = float(np.linalg.norm(error - np.dot(error, axis) * axis))

    assert along > across, "expected depth to be the worse axis"


def test_nothing_is_detected_when_the_marker_is_hidden(params):
    """An empty result, not an exception or a stale pose."""
    hidden = params.model_copy(update={"markers": ()})
    assert detect(_capture(params, POSES[0]), hidden) == []


def test_detection_fails_at_oblique_angles(params):
    """Reachable, fully in frame, and still unreadable.

    This pose puts all four marker corners comfortably inside the image, so it
    is not a visibility failure -- the marker is simply too edge-on for the
    detector to decode. Worth pinning because it is what makes collecting
    calibration poses a search rather than a sweep: a configuration can be
    perfectly valid for the arm and produce no observation at all.
    """
    view = _capture(params, OBLIQUE)
    marker = params.markers[0]

    truth = marker_pose_world(marker, OBLIQUE, params)
    corners = np.array([(truth @ np.append(c, 1.0))[:3] for c in marker_corners_local(marker)])
    pixels = view.project(corners)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < params.cameras[0].width_px)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < params.cameras[0].height_px)
    )
    assert inside.all(), "this pose should be in frame; the point is the angle"
    assert detect(view, params) == []


@pytest.mark.parametrize("q", POSES)
def test_usable_poses_are_less_oblique_than_the_failing_one(params, q):
    """Names the mechanism rather than leaving it as a magic list of poses."""
    from arm.fiducial import viewing_angle_deg

    camera = params.cameras[0].pos_m
    marker = params.markers[0]
    assert viewing_angle_deg(marker, q, camera, params) < viewing_angle_deg(
        marker, OBLIQUE, camera, params
    )
