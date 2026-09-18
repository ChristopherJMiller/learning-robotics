"""Cameras: turning a scene into pixels, and pixels back into rays.

Everything before this closed the loop on **proprioception**. The arm knew where
it was because its motors said so, and knew where the obstacle was because
``config/arm.toml`` said so. Both were *given*. A camera is the first component
that has to find something out.

The model
---------
A pinhole camera projects a point in the camera's own frame to a pixel::

    [u]        [X]              [f_x   0   c_x]
    [v] ~  K · [Y]        K  =  [ 0   f_y  c_y]
    [1]        [Z]              [ 0    0    1 ]

``K`` is the *intrinsic* matrix, and it is a **model rather than a fact**. Here
it is derived exactly from the simulated field of view; on real hardware it is
measured, imperfectly, by photographing a known pattern -- and lens distortion
means the pinhole model is an approximation even after that.

Two conventions that are easy to get wrong
------------------------------------------
**MuJoCo cameras look down their own −z axis**, with +x right and +y *up*. That
is the OpenGL convention.

**OpenCV looks down +z**, with +y *down*. So the two disagree about the sign of
two axes, and a pose handed from one to the other without conversion comes back
mirrored -- which looks like a calibration failure rather than a convention
error. :func:`opencv_from_mujoco` is the one place that flip lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from arm.frames import se3
from arm.params import ArmParams, Camera, default_params

__all__ = [
    "RenderedView",
    "SimulatedCamera",
    "camera_axes",
    "intrinsic_matrix",
    "opencv_from_mujoco",
]

# OpenCV's camera frame relative to MuJoCo's: flip y and z.
_MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def intrinsic_matrix(camera: Camera) -> np.ndarray:
    """The pinhole matrix ``K`` implied by a simulated camera.

    MuJoCo specifies a *vertical* field of view, so the focal length follows
    from half the image height over the tangent of the half angle. Pixels are
    square, so ``f_x == f_y``.

    The principal point sits at ``width / 2``, not ``(width - 1) / 2``. Both
    appear in the wild; this one matches what MuJoCo's renderer actually does,
    and ``tests/test_camera.py`` checks that by projecting a point and looking
    at where it lands in a real render rather than taking either on faith.
    """
    focal = 0.5 * camera.height_px / np.tan(0.5 * np.radians(camera.fov_y_deg))
    return np.array(
        [
            [focal, 0.0, camera.width_px / 2.0],
            [0.0, focal, camera.height_px / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def camera_axes(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The camera frame's x, y and z axes in world coordinates.

    Built from ``pos_m`` and ``lookat_m`` because those are what a person can
    reason about, while MuJoCo wants axes. Follows MuJoCo's convention: the
    camera looks along **−z**, x points right and y points up.

    Degenerate when the view direction is parallel to world up -- a camera
    looking straight down has no preferred "right" -- so a second reference is
    used in that case rather than producing silent NaNs.
    """
    position = np.asarray(camera.pos_m, dtype=float)
    forward = np.asarray(camera.lookat_m, dtype=float) - position
    forward /= np.linalg.norm(forward)

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, reference))) > 0.999:
        reference = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, reference)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    return right, up, -forward


def pose_in_world(camera: Camera) -> np.ndarray:
    """``T_world_camera`` using MuJoCo's axis convention."""
    right, up, backward = camera_axes(camera)
    rotation = np.column_stack([right, up, backward])
    return se3(rotation, np.asarray(camera.pos_m, dtype=float))


def opencv_from_mujoco(pose: np.ndarray) -> np.ndarray:
    """Convert a camera pose between the two conventions.

    MuJoCo looks down −z with +y up; OpenCV looks down +z with +y down. The
    rotation is post-multiplied by ``diag(1, −1, −1)``, and the operation is its
    own inverse.
    """
    pose = np.asarray(pose, dtype=float)
    return se3(pose[:3, :3] @ _MUJOCO_TO_OPENCV, pose[:3, 3])


@dataclass(frozen=True)
class RenderedView:
    """One frame, and what it would take to interpret it."""

    rgb: np.ndarray
    depth: np.ndarray
    pose_world: np.ndarray  # T_world_camera, MuJoCo convention
    intrinsics: np.ndarray
    time_s: float

    @property
    def gray(self) -> np.ndarray:
        """Luminance, which is what a fiducial detector wants."""
        return (
            0.299 * self.rgb[..., 0] + 0.587 * self.rgb[..., 1] + 0.114 * self.rgb[..., 2]
        ).astype(np.uint8)

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World points to pixels, in this view.

        Returns ``NaN`` for anything behind the camera rather than a plausible
        pixel. A point behind the lens still has a mathematically valid
        projection -- it lands mirrored on the far side of the image -- and
        silently accepting that is a genuinely nasty bug, because the number
        looks reasonable.
        """
        points = np.atleast_2d(np.asarray(points_world, dtype=float))
        rotation = self.pose_world[:3, :3]
        translation = self.pose_world[:3, 3]

        # World into the camera frame, then into OpenCV's axes.
        local = (points - translation) @ rotation
        local = local @ _MUJOCO_TO_OPENCV

        pixels = np.full((len(points), 2), np.nan)
        in_front = local[:, 2] > 1e-9
        if np.any(in_front):
            homogeneous = local[in_front] / local[in_front, 2:3]
            pixels[in_front] = (homogeneous @ self.intrinsics.T)[:, :2]
        return pixels

    def ray(self, pixel) -> np.ndarray:
        """A unit direction in world coordinates for a pixel.

        The inverse of :meth:`project` up to depth: a pixel names a ray, not a
        point. Recovering the point needs the depth buffer, or a second view, or
        a known plane the point lies on.
        """
        u, v = float(pixel[0]), float(pixel[1])
        inverse = np.linalg.inv(self.intrinsics)
        direction = inverse @ np.array([u, v, 1.0])
        direction = _MUJOCO_TO_OPENCV @ direction
        direction = self.pose_world[:3, :3] @ direction
        return direction / np.linalg.norm(direction)

    def unproject(self, pixel) -> np.ndarray:
        """The world point a pixel sees, using the depth buffer."""
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        distance = float(self.depth[v, u])
        direction = self.ray(pixel)
        # depth is measured along the optical axis, not along the ray
        axis = -self.pose_world[:3, 2]
        return self.pose_world[:3, 3] + direction * (distance / float(np.dot(direction, axis)))


class SimulatedCamera:
    """Renders a MuJoCo scene from a camera declared in ``config/arm.toml``."""

    def __init__(
        self,
        model,
        data,
        camera: Camera | str = 0,
        params: ArmParams | None = None,
    ) -> None:
        self.params = params or default_params()
        if isinstance(camera, str):
            camera = next(c for c in self.params.cameras if c.name == camera)
        elif isinstance(camera, int):
            camera = self.params.cameras[camera]

        self.camera = camera
        self.model = model
        self.data = data
        self.intrinsics = intrinsic_matrix(camera)
        self.pose_world = pose_in_world(camera)
        self._renderer = mujoco.Renderer(model, height=camera.height_px, width=camera.width_px)

    def capture(self, *, depth: bool = True) -> RenderedView:
        """Render the current state.

        The timestamp is the *simulation* time at which the scene was captured,
        carried alongside the image because an image is evidence about when it
        was taken rather than about now. That matters as soon as a detection
        result is fused with anything.
        """
        mujoco.mj_forward(self.model, self.data)

        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(self.data, camera=self.camera.name)
        rgb = self._renderer.render().copy()

        if depth:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(self.data, camera=self.camera.name)
            distances = self._renderer.render().copy()
            self._renderer.disable_depth_rendering()
        else:
            distances = np.zeros(rgb.shape[:2])

        return RenderedView(
            rgb=rgb,
            depth=distances,
            pose_world=self.pose_world,
            intrinsics=self.intrinsics,
            time_s=float(self.data.time),
        )
