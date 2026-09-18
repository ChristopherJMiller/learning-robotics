"""Hand-eye calibration: finding the camera without ever measuring it.

Everything up to here assumed ``T_base_camera`` was known, because ``arm.toml``
declares it. On real hardware nobody knows it. A camera gets clamped to a
tripod, and its pose relative to the robot base is a number nobody measured and
nobody can measure well with a tape -- the camera's optical centre is inside the
lens barrel, and its axes are defined by the sensor, not the housing.

Hand-eye calibration recovers it from motion alone. Move the arm to several
configurations, observe the marker in each, and the constraint that the same
rigid transform must explain *every* observation pins it down.

The equation
------------
This arm is **eye-to-hand**: the camera is fixed and the marker rides on the
arm. (Eye-in-hand -- camera on the end effector -- is the other arrangement, and
was rejected here because a 3-DOF arm cannot control tip orientation, so a
mounted camera would point wherever the geometry happened to aim it.)

For every configuration, one chain must hold::

    T_cam_marker  =  T_cam_base · T_base_link(q) · T_link_marker
       measured        unknown        known FK        unknown

Two unknowns, both constant. ``T_base_link(q)`` changes with every pose and is
known exactly from forward kinematics; ``T_cam_marker`` is what ``solvePnP``
returns. Eliminating the constant marker mounting between two poses gives the
classic form::

    A X  =  X B

with ``A`` the relative *arm* motion and ``B`` the relative *camera* motion.
That is the whole idea: the unknown transform is whatever makes the arm's motion
and the camera's observation of that motion agree.

Two stages, because closed form is not the answer
-------------------------------------------------
OpenCV's seven solvers are all *non-iterative*: each makes an algebraic
simplification -- separating rotation from translation, or linearising a
quaternion product -- to get a solution in one shot. On the same 21 rendered
observations they disagreed by a factor of five::

    shah 3.57 mm   li 3.92 mm   andreff 6.16 mm   park 8.81 mm
    horaud 8.82 mm   tsai 11.94 mm   daniilidis 19.17 mm

Refining against the corner pixels collapses that spread entirely -- all seven
converge to **the same 0.661 mm and 0.032 deg**, at 0.17 px RMS. The choice of
closed-form method stops mattering once it is only an initial guess, which is
exactly how real calibration pipelines are built. See :func:`refine`.

Why the axes must not be parallel
---------------------------------
``AX = XB`` does not always have a unique answer. Each relative motion
constrains ``X``'s translation only in the plane perpendicular to that motion's
rotation axis, so if every rotation axis is parallel, the translation along that
shared axis is **never constrained by anything**.

That is not a hypothetical on an RRR arm. Joints 2 and 3 are both pitch about
parallel axes, so holding the base joint still makes every relative rotation
share one axis. On synthetic noise-free data that alone costs 214 mm. On real
rendered observations it is worse, and it fails in three different ways at once
(``just calibrate --degenerate``)::

    andreff, shah, li   raise from inside OpenCV
    park                returns a matrix of NaN
    tsai                228 mm      horaud 213 mm      daniilidis 837 mm

Only the last row is dangerous, because only it looks like an answer.

**And refinement makes it worse, not better.** Given that same set, least
squares slid the camera to **428 metres** from the base -- while *improving*
the reprojection residual to 0.116 px, better than the 0.174 px of the correct
answer. That is not a bug in the optimiser. The data genuinely does not
constrain that direction, so the optimiser is free to travel along it, and it
will, because doing so buys a fractionally better fit to the noise.

What the residual can and cannot tell you
-----------------------------------------
So: **a small residual proves consistency, not correctness.** A calibration that
explains every observation perfectly can still be metres wrong, if the
observations never asked the question. :func:`Calibration.residuals` and the
reprojection RMS are the checks that survive contact with hardware, where no
ground truth exists -- but they must be read next to :attr:`Calibration.axis_spread`,
which is computed from forward kinematics alone and so can be evaluated on
candidate poses *before* moving anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from arm.camera import SimulatedCamera, opencv_from_mujoco
from arm.fiducial import detect, viewing_angle_deg
from arm.frames import inverse, rot_x, rot_y, rot_z, se3
from arm.kinematics import link_frames
from arm.params import ArmParams, Marker, default_params

__all__ = [
    "Calibration",
    "DEFAULT_METHOD",
    "DegenerateGeometryError",
    "SolverFailure",
    "MIN_AXIS_SPREAD",
    "METHODS",
    "average_poses",
    "Observation",
    "axis_spread",
    "calibrate",
    "collect",
    "marker_mount",
    "pose_error",
    "refine",
    "reprojection_residuals",
    "search_configurations",
    "truth",
]

# OpenCV ships two *different* families of hand-eye solver behind two
# independent and overlapping enums -- CALIB_HAND_EYE_TSAI is 0, and so is
# CALIB_ROBOT_WORLD_HAND_EYE_SHAH. Passing one family's constant to the other
# function is silently accepted and returns nonsense: a first cut here sent
# HORAUD (2) to calibrateRobotWorldHandEye, which has no method 2, and got a
# camera 600 mm and 120 degrees out with nothing raised and no warning.
#
# So each method records which entry point it belongs to.
#   AX = XB   calibrateHandEye           -- camera pose alone
#   AX = ZB   calibrateRobotWorldHandEye -- camera pose and marker mount together
METHODS = {
    "tsai": ("handeye", cv2.CALIB_HAND_EYE_TSAI),
    "park": ("handeye", cv2.CALIB_HAND_EYE_PARK),
    "horaud": ("handeye", cv2.CALIB_HAND_EYE_HORAUD),
    "andreff": ("handeye", cv2.CALIB_HAND_EYE_ANDREFF),
    "daniilidis": ("handeye", cv2.CALIB_HAND_EYE_DANIILIDIS),
    "shah": ("robotworld", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
    "li": ("robotworld", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI),
}

DEFAULT_METHOD = "shah"

# Below this axis spread the translation along the shared rotation axis is
# unobservable and the result is meaningless. The value is not tuned: a
# well-spread set on this arm scores around 0.55-0.67 and a degenerate one
# scores 0.0 to within floating point, so anything in between is already a
# warning sign. It is deliberately loud rather than a silent clamp -- the whole
# failure mode is that nothing else complains.
MIN_AXIS_SPREAD = 0.05


class DegenerateGeometryError(ValueError):
    """The poses do not constrain the answer, whatever the residual says."""


class SolverFailure(RuntimeError):
    """A closed-form solver returned something that is not a pose.

    Distinct from :class:`DegenerateGeometryError` because it is a different
    kind of report: the geometry error says the *question* was unanswerable,
    this says one particular solver fell over trying. On degenerate input the
    seven methods do both -- ``andreff``, ``shah`` and ``li`` raise from inside
    OpenCV, ``park`` hands back a matrix full of NaN, and the rest return a
    confident wrong answer. Only the last of those is dangerous.
    """


def _validated(pose: np.ndarray, method: str) -> np.ndarray:
    """Reject a solver result that is not a rigid transform.

    ``park`` returns NaN rather than raising when the geometry is degenerate,
    and an unchecked NaN propagates a long way before anything notices -- it
    first surfaced here as "SVD did not converge" inside pose averaging, three
    calls downstream, naming nothing useful.
    """
    if not np.isfinite(pose).all():
        raise SolverFailure(f"{method!r} returned a non-finite pose; the geometry is degenerate")

    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise SolverFailure(f"{method!r} returned a non-orthonormal rotation")
    if np.linalg.det(rotation) < 0:
        raise SolverFailure(f"{method!r} returned a reflection rather than a rotation")
    return pose


def average_poses(poses) -> np.ndarray:
    """Chordal mean of a tightly clustered set of poses.

    Translations average arithmetically; rotations are averaged by summing the
    matrices and projecting back onto SO(3) with an SVD, which gives the closest
    rotation in the Frobenius sense. Valid here because the poses differ only by
    measurement noise -- it would be the wrong tool for widely spread rotations.
    """
    poses = list(poses)
    rotation = sum(pose[:3, :3] for pose in poses)
    u, _, vt = np.linalg.svd(rotation)
    if np.linalg.det(u @ vt) < 0:  # a reflection, not a rotation
        u[:, -1] *= -1
    return se3(u @ vt, np.mean([pose[:3, 3] for pose in poses], axis=0))


def marker_mount(marker: Marker) -> np.ndarray:
    """``T_link_marker`` as declared -- the truth calibration has to rediscover."""
    roll, pitch, yaw = np.radians(marker.euler_deg)
    rotation = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return se3(rotation, np.asarray(marker.pos_m, dtype=float))


def pose_error(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """``(metres, degrees)`` between two poses."""
    position = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
    relative = inverse(truth) @ estimated
    angle = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
    return position, angle


@dataclass(frozen=True)
class Observation:
    """One arm configuration and what the camera saw there.

    ``pose_camera_marker`` comes from ``solvePnP`` and is the only measured
    quantity; ``q`` is what the encoders reported. Nothing here knows where the
    camera is, which is the point.
    """

    q: np.ndarray
    pose_camera_marker: np.ndarray  # T_cam_marker, OpenCV convention
    corners_px: np.ndarray  # (4, 2) -- the actual measurement, kept for refine()
    intrinsics: np.ndarray
    reprojection_error_px: float
    viewing_angle_deg: float

    def pose_base_link(self, params: ArmParams | None = None) -> np.ndarray:
        """``T_base_link`` from forward kinematics -- the other half of the chain."""
        params = params or default_params()
        marker = params.markers[0]
        index = [link.name for link in params.links].index(marker.link)
        return link_frames(self.q, params)[index]


def search_configurations(
    params: ArmParams | None = None,
    count: int = 20,
    *,
    rng: np.random.Generator | None = None,
    max_attempts: int = 400,
    max_viewing_angle_deg: float = 45.0,
) -> list[np.ndarray]:
    """Find configurations where the marker is likely to be readable.

    A *search*, not a sweep. Rejection sampling against
    :func:`arm.fiducial.viewing_angle_deg` is cheap -- it needs no render -- and
    it filters out the poses that are reachable, fully in frame, and still too
    edge-on for the detector to decode.

    The sampler deliberately spans the base joint. That is not cosmetic: see
    :func:`axis_spread` for what happens when it does not.
    """
    params = params or default_params()
    rng = rng or np.random.default_rng(0)
    marker = params.markers[0]
    camera_position = np.asarray(params.cameras[0].pos_m)
    lower = np.array([joint.limit_lower_rad for joint in params.joints])
    upper = np.array([joint.limit_upper_rad for joint in params.joints])

    found: list[np.ndarray] = []
    for _ in range(max_attempts):
        if len(found) >= count:
            break
        q = lower + rng.random(3) * (upper - lower)
        if viewing_angle_deg(marker, q, camera_position, params) <= max_viewing_angle_deg:
            found.append(q)
    return found


def collect(
    configurations,
    params: ArmParams | None = None,
    *,
    max_reprojection_error_px: float = 1.0,
) -> list[Observation]:
    """Render each configuration, detect the marker, keep what was seen.

    Configurations that produce no detection are silently dropped -- which is
    the honest behaviour, since on hardware a missed frame is simply a missed
    frame. ``max_reprojection_error_px`` rejects poses the solver fitted badly;
    a bad fit is the one cheap signal that a pose is wrong rather than merely
    uncertain.
    """
    from arm.sim import MujocoArm

    params = params or default_params()
    marker = params.markers[0]
    camera_position = np.asarray(params.cameras[0].pos_m)

    arm = MujocoArm(params)
    camera = SimulatedCamera(arm.model, arm.data, params.cameras[0].name, params)

    observations: list[Observation] = []
    for q in configurations:
        q = np.asarray(q, dtype=float)
        arm.reset(q)
        view = camera.capture()
        detections = [d for d in detect(view, params) if d.aruco_id == marker.aruco_id]
        if not detections:
            continue
        detection = detections[0]
        if detection.reprojection_error_px > max_reprojection_error_px:
            continue
        observations.append(
            Observation(
                corners_px=detection.corners_px,
                intrinsics=view.intrinsics,
                q=q,
                pose_camera_marker=detection.pose_camera,
                reprojection_error_px=detection.reprojection_error_px,
                viewing_angle_deg=viewing_angle_deg(marker, q, camera_position, params),
            )
        )
    return observations


def axis_spread(observations, params: ArmParams | None = None) -> float:
    """How non-parallel the relative rotation axes are, in ``[0, 1]``.

    The smallest over the largest singular value of every relative motion's
    rotation axis stacked into a matrix. Zero means every axis is parallel and
    the translation along that shared axis is unobservable, no matter how many
    poses are collected or how clean the data is.

    This is the diagnostic to run *before* believing a calibration. It is
    computed from forward kinematics alone, so it can be evaluated on candidate
    poses without moving anything.
    """
    params = params or default_params()
    poses = [observation.pose_base_link(params) for observation in observations]

    axes = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            rotation = (inverse(poses[j]) @ poses[i])[:3, :3]
            angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
            if angle < 1e-6 or abs(angle - np.pi) < 1e-6:
                continue
            axis = np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]
            ) / (2 * np.sin(angle))
            axes.append(axis)

    if len(axes) < 2:
        return 0.0
    singular = np.linalg.svd(np.array(axes), compute_uv=False)
    return float(singular[-1] / singular[0])


@dataclass(frozen=True)
class Calibration:
    """The recovered camera pose, and the marker mounting that came free."""

    pose_base_camera: np.ndarray  # T_base_camera, OpenCV convention
    pose_link_marker: np.ndarray  # T_link_marker -- where the marker really sits
    axis_spread: float
    method: str
    observations: int

    def residuals(self, observations, params: ArmParams | None = None) -> np.ndarray:
        """Per-observation position disagreement, in metres, without ground truth.

        Re-predicts ``T_cam_marker`` from the recovered transforms and compares
        against what was measured. This is the check that survives contact with
        hardware -- but a degenerate pose set produces small residuals and a
        badly wrong answer, so read it next to :attr:`axis_spread`.
        """
        params = params or default_params()
        pose_camera_base = inverse(self.pose_base_camera)
        errors = []
        for observation in observations:
            predicted = (
                pose_camera_base @ observation.pose_base_link(params) @ self.pose_link_marker
            )
            errors.append(np.linalg.norm(predicted[:3, 3] - observation.pose_camera_marker[:3, 3]))
        return np.array(errors)


def calibrate(
    observations,
    params: ArmParams | None = None,
    *,
    method: str = DEFAULT_METHOD,
    require_observability: bool = True,
) -> Calibration:
    """Recover the camera pose from observations alone.

    Both argument mappings below are unobvious, and both were confirmed
    numerically against synthetic noise-free data rather than read off the
    documentation -- each reproduces the planted transforms to machine
    precision, which is the only way to be sure the frames were not transposed
    somewhere.

    **AX = XB** (``calibrateHandEye``, five methods). Written for eye-in-hand,
    it is turned into eye-to-hand by feeding it the *inverted* arm poses: supply
    ``T_link_base`` where it expects "gripper2base" and ``T_cam_marker`` where it
    expects "target2cam", and its "cam2gripper" output is exactly
    ``T_base_camera``.

    **AX = ZB** (``calibrateRobotWorldHandEye``, two methods). Supply
    ``T_cam_marker`` as "world2cam" and ``T_base_link`` as "base2gripper"; the
    two outputs invert to ``T_link_marker`` and ``T_base_camera``.

    The marker mount is recovered the same way for every method -- by averaging
    ``T_base_link⁻¹ · T_base_camera · T_cam_marker`` over the observations --
    so that the comparison between solvers is about the camera pose and nothing
    else. Once ``T_base_camera`` is known the mount is a direct measurement, not
    another optimisation.

    This never reads ``params.cameras``: the camera pose is the output, and
    ``params`` is used only for forward kinematics.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}")
    if len(observations) < 3:
        raise ValueError(f"hand-eye calibration needs at least 3 poses, got {len(observations)}")

    params = params or default_params()
    spread = axis_spread(observations, params)
    if spread < MIN_AXIS_SPREAD and require_observability:
        raise DegenerateGeometryError(
            f"relative rotation axes are near-parallel (spread {spread:.4g} < "
            f"{MIN_AXIS_SPREAD}); the camera position along that axis is "
            f"unobservable and the result would be arbitrary. Vary the base "
            f"joint across the pose set. Pass require_observability=False to "
            f"solve anyway and see what that costs."
        )

    arm_poses = [observation.pose_base_link(params) for observation in observations]
    camera_poses = [observation.pose_camera_marker for observation in observations]
    family, flag = METHODS[method]

    if family == "handeye":
        inverted = [inverse(pose) for pose in arm_poses]
        rotation, translation = cv2.calibrateHandEye(
            [pose[:3, :3] for pose in inverted],
            [pose[:3, 3] for pose in inverted],
            [pose[:3, :3] for pose in camera_poses],
            [pose[:3, 3] for pose in camera_poses],
            method=flag,
        )
        pose_base_camera = _validated(se3(rotation, translation.ravel()), method)
    else:
        _, _, rotation, translation = cv2.calibrateRobotWorldHandEye(
            [pose[:3, :3] for pose in camera_poses],
            [pose[:3, 3] for pose in camera_poses],
            [pose[:3, :3] for pose in arm_poses],
            [pose[:3, 3] for pose in arm_poses],
            method=flag,
        )
        pose_base_camera = _validated(inverse(se3(rotation, translation.ravel())), method)

    # T_cam_marker = T_cam_base · T_base_link · T_link_marker, so once the camera
    # pose is known the mount falls straight out: T_link_marker is
    # T_base_link⁻¹ · T_base_camera · T_cam_marker, one estimate per observation.
    mount = average_poses(
        inverse(arm) @ pose_base_camera @ camera
        for arm, camera in zip(arm_poses, camera_poses, strict=True)
    )

    return Calibration(
        pose_base_camera=pose_base_camera,
        pose_link_marker=mount,
        axis_spread=spread,
        method=method,
        observations=len(observations),
    )


def truth(params: ArmParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(T_base_camera, T_link_marker)`` as declared, for scoring only.

    A calibration routine must never read this; it is the answer being hidden.
    """
    from arm.camera import pose_in_world

    params = params or default_params()
    return (
        opencv_from_mujoco(pose_in_world(params.cameras[0])),
        marker_mount(params.markers[0]),
    )


def _pack(pose_base_camera: np.ndarray, pose_link_marker: np.ndarray) -> np.ndarray:
    """Two poses as twelve numbers: axis-angle plus translation, twice."""
    return np.concatenate(
        [
            cv2.Rodrigues(pose_base_camera[:3, :3])[0].ravel(),
            pose_base_camera[:3, 3],
            cv2.Rodrigues(pose_link_marker[:3, :3])[0].ravel(),
            pose_link_marker[:3, 3],
        ]
    )


def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        se3(cv2.Rodrigues(x[0:3])[0], x[3:6]),
        se3(cv2.Rodrigues(x[6:9])[0], x[9:12]),
    )


def reprojection_residuals(
    pose_base_camera: np.ndarray,
    pose_link_marker: np.ndarray,
    observations,
    params: ArmParams | None = None,
) -> np.ndarray:
    """Predicted minus detected corner pixels, flattened.

    The residual in the units the noise actually lives in. Every other error
    measure in this module -- millimetres of camera position, degrees of
    rotation -- is derived; *this* is what was measured, and it is the only
    quantity available on hardware.
    """
    from arm.fiducial import marker_corners_local

    params = params or default_params()
    corners_local = marker_corners_local(params.markers[0])
    pose_camera_base = inverse(pose_base_camera)

    errors = []
    for observation in observations:
        pose = pose_camera_base @ observation.pose_base_link(params) @ pose_link_marker
        points = corners_local @ pose[:3, :3].T + pose[:3, 3]
        homogeneous = points @ observation.intrinsics.T
        predicted = homogeneous[:, :2] / homogeneous[:, 2:3]
        errors.append((predicted - observation.corners_px).ravel())
    return np.concatenate(errors)


def refine(calibration: Calibration, observations, params: ArmParams | None = None) -> Calibration:
    """Least-squares polish of both transforms against the corner pixels.

    The closed-form solvers are **initialisers, not answers**. Each one makes a
    different algebraic simplification -- separating rotation from translation,
    or linearising a quaternion product -- to get a non-iterative solution, and
    each pays for it differently on noisy data. Measured on the same ten
    observations, they disagreed by a factor of eight, from 4 mm to 32 mm,
    which is a spread no amount of choosing between them resolves.

    Refinement dissolves that disagreement because it optimises the right thing.
    The closed-form methods minimise an algebraic residual in pose space, where
    a millimetre of depth error and a millimetre of lateral error count the
    same -- but they are not the same, since depth is far less certain
    (see :mod:`arm.fiducial`). Minimising *reprojection error* weights each
    observation by what the camera actually resolved, so the well-conditioned
    directions dominate, as they should.

    This is the standard shape of a real calibration pipeline: a linear method
    for an initial guess, then nonlinear least squares for the answer.
    """
    from scipy.optimize import least_squares

    params = params or default_params()
    result = least_squares(
        lambda x: reprojection_residuals(*_unpack(x), observations, params),
        _pack(calibration.pose_base_camera, calibration.pose_link_marker),
        method="lm",
        xtol=1e-14,
        ftol=1e-14,
    )
    pose_base_camera, pose_link_marker = _unpack(result.x)

    return Calibration(
        pose_base_camera=pose_base_camera,
        pose_link_marker=pose_link_marker,
        axis_spread=calibration.axis_spread,
        method=f"{calibration.method}+refined",
        observations=calibration.observations,
    )
