"""The camera model, checked against the renderer rather than against itself.

A projection function that agrees with its own inverse proves nothing. The test
that matters is whether our pinhole model predicts where MuJoCo's renderer
actually puts a point -- two independent implementations of the same geometry,
which is the same standard the kinematics was held to.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.camera import (
    SimulatedCamera,
    camera_axes,
    intrinsic_matrix,
    opencv_from_mujoco,
    pose_in_world,
)
from arm.params import Camera
from arm.sim import MujocoArm


@pytest.fixture(scope="module")
def scene(params):
    arm = MujocoArm(params)
    arm.reset(np.array([1.2, 0.4, -1.0]))
    return arm


@pytest.fixture(scope="module")
def view(scene, params):
    return SimulatedCamera(scene.model, scene.data, "workspace", params).capture()


# --------------------------------------------------------------------------
# Intrinsics and axes
# --------------------------------------------------------------------------


def test_focal_length_follows_from_the_field_of_view(params):
    camera = params.cameras[0]
    matrix = intrinsic_matrix(camera)

    expected = 0.5 * camera.height_px / np.tan(0.5 * np.radians(camera.fov_y_deg))
    np.testing.assert_allclose(matrix[0, 0], expected)
    assert matrix[0, 0] == matrix[1, 1], "pixels should be square"
    np.testing.assert_allclose(
        [matrix[0, 2], matrix[1, 2]], [camera.width_px / 2, camera.height_px / 2]
    )


def test_camera_axes_are_orthonormal_and_right_handed(params):
    right, up, backward = camera_axes(params.cameras[0])
    basis = np.column_stack([right, up, backward])

    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(basis), 1.0, atol=1e-12)


def test_the_camera_looks_at_what_it_was_told_to(params):
    camera = params.cameras[0]
    _, _, backward = camera_axes(camera)

    direction = np.asarray(camera.lookat_m) - np.asarray(camera.pos_m)
    direction /= np.linalg.norm(direction)
    # MuJoCo cameras look along -z.
    np.testing.assert_allclose(-backward, direction, atol=1e-12)


def test_looking_straight_down_does_not_produce_nonsense():
    """Degenerate: the view direction is parallel to world up.

    There is no preferred "right" for a camera looking straight down, so the
    naive cross product is zero and every axis comes back NaN. Silent NaNs
    propagate into a pose and then into a calibration.
    """
    camera = Camera(
        name="overhead",
        pos_m=(0.0, 0.0, 0.5),
        lookat_m=(0.0, 0.0, 0.0),
        width_px=64,
        height_px=64,
        fov_y_deg=45.0,
    )
    basis = np.column_stack(camera_axes(camera))
    assert np.all(np.isfinite(basis))
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_a_camera_must_look_somewhere_else(params):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="no view direction"):
        Camera(
            name="nowhere",
            pos_m=(0.1, 0.1, 0.1),
            lookat_m=(0.1, 0.1, 0.1),
            width_px=64,
            height_px=64,
            fov_y_deg=45.0,
        )


def test_the_opencv_conversion_is_its_own_inverse(params):
    pose = pose_in_world(params.cameras[0])
    np.testing.assert_allclose(opencv_from_mujoco(opencv_from_mujoco(pose)), pose)


# --------------------------------------------------------------------------
# Projection, checked against the renderer
# --------------------------------------------------------------------------


def test_the_scene_renders(view, params):
    camera = params.cameras[0]
    assert view.rgb.shape == (camera.height_px, camera.width_px, 3)
    assert view.depth.shape == (camera.height_px, camera.width_px)
    assert view.rgb.dtype == np.uint8
    assert view.gray.shape == view.depth.shape


def test_projection_agrees_with_what_the_renderer_drew(view, scene, params):
    """The check that matters: our model against MuJoCo's, not against itself.

    The marker's four corners are known exactly from forward kinematics, so
    where our pinhole model says they land can be compared with where a
    detector finds them in the actual render. Agreement to about a pixel means
    the intrinsics, the axis conventions and the frame chain are all right --
    and a disagreement of 242 px is what a wrong link-frame convention looked
    like before it was fixed.
    """
    from arm.fiducial import detect, marker_corners_local, marker_pose_world

    marker = params.markers[0]
    truth = marker_pose_world(marker, scene.data.qpos.copy(), params)
    corners_world = np.array(
        [(truth @ np.append(c, 1.0))[:3] for c in marker_corners_local(marker)]
    )

    predicted = view.project(corners_world)
    detected = detect(view, params)
    assert detected, "the marker should be visible from this camera"

    error = np.linalg.norm(predicted - detected[0].corners_px, axis=1)
    assert error.max() < 2.0, f"pinhole model disagrees with the renderer: {error}"


def test_points_behind_the_camera_are_rejected(view):
    """A point behind the lens has a valid-looking projection, mirrored.

    Accepting it silently is worse than an error, because the pixel looks
    perfectly reasonable.
    """
    behind = view.pose_world[:3, 3] + view.pose_world[:3, 2] * 0.5
    assert np.all(np.isnan(view.project(behind)))


def test_a_pixel_and_its_ray_round_trip(view):
    """A pixel names a ray; a point on that ray projects back to the pixel."""
    for pixel in ([320.0, 240.0], [100.0, 400.0], [560.0, 80.0]):
        direction = view.ray(pixel)
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0)

        point = view.pose_world[:3, 3] + direction * 0.5
        np.testing.assert_allclose(view.project(point)[0], pixel, atol=1e-6)


def test_unprojecting_with_depth_recovers_a_world_point(view, params):
    """Depth is what turns a ray back into a point."""
    from arm.fiducial import marker_pose_world

    centre = marker_pose_world(params.markers[0], np.array([1.2, 0.4, -1.0]), params)[:3, 3]
    pixel = view.project(centre)[0]
    assert np.all(np.isfinite(pixel))

    recovered = view.unproject(pixel)
    # Limited by the depth buffer's own resolution and by rounding the pixel to
    # an integer, so millimetres rather than microns.
    np.testing.assert_allclose(recovered, centre, atol=3e-3)


def test_the_view_carries_the_time_it_was_captured(view, scene):
    """An image is evidence about when it was taken, not about now."""
    assert view.time_s == pytest.approx(float(scene.data.time))
