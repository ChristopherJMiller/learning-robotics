"""Fiducial markers: making the arm observable to a camera.

A camera on its own sees pixels. A fiducial is a pattern engineered so that a
detector can find it reliably and recover a **full six-degree-of-freedom pose**
from a single image -- which is a remarkable amount of information from one
frame, and the reason fiducials are the standard on-ramp to robot vision.

ArUco markers work because the pattern is a binary code with a known dictionary:
the detector finds quadrilaterals, reads the bits, rejects anything that is not
a valid codeword, and returns the four corners in a *known order*. Knowing which
corner is which is what makes the pose unambiguous.

Two details that cause more trouble than they should
----------------------------------------------------
**The quiet zone.** A detector needs white space around the black square to find
the boundary. It is generated as part of the texture but is *not* part of
``size_m``, because the pose solver must be told the size of the black square
alone. Conflating them produces a pure scale error in the recovered translation,
which looks like a calibration failure rather than a measurement mistake.

**Planar pose ambiguity.** A square viewed nearly face-on has two poses that
project to almost the same four corners -- a genuine ambiguity, not noise. It
appears as the estimated pose flipping between two orientations on consecutive
frames. Viewing the marker at an angle resolves it, which is one reason
calibration wants varied poses rather than many similar ones.

Depth is the weak axis, and by a lot
------------------------------------
A single marker gives a full pose, but not equally well in every direction.
Measured on a noise-free render of a 40 mm marker at 427 mm, the error was
**7.40 mm along the optical axis and 1.42 mm across it** -- depth roughly five
times worse.

That is conditioning rather than a defect. Depth is inferred from *apparent
size*, so a corner error of ``sigma`` pixels propagates as::

    dZ  ~  Z^2 * sigma / (f * L)          L = marker side length

which for those numbers predicts 7.9 mm at one pixel of corner error, matching
what was measured. The consequences are practical: **depth error grows with the
square of distance and falls only linearly with marker size**, so a marker twice
as far away is four times worse, and the cheapest fix is almost always a bigger
marker rather than a better camera.

It is also why sub-pixel corner refinement is on by default below. It costs
almost nothing and took the same measurement from 7.53 mm to 1.19 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arm.frames import se3
from arm.params import REPO_ROOT, ArmParams, Marker, default_params

__all__ = [
    "Detection",
    "detect",
    "marker_corners_local",
    "marker_texture_path",
    "write_marker_texture",
]

TEXTURE_DIR = REPO_ROOT / "assets" / "markers"

# Fraction of the black square's width added as white border on each side.
# Detectors want at least one marker cell; this is comfortably more.
QUIET_ZONE_FRACTION = 0.25


def marker_texture_path(marker: Marker) -> Path:
    return TEXTURE_DIR / f"{marker.name}.png"


def write_marker_texture(marker: Marker, pixels: int = 400) -> Path:
    """Render the marker to a PNG, with its quiet zone.

    Committed like the link meshes: the generated MJCF references it at load
    time, so the model is only self-contained if it is present.
    """
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker.dictionary))
    code = cv2.aruco.generateImageMarker(dictionary, marker.aruco_id, pixels)

    border = int(round(pixels * QUIET_ZONE_FRACTION))
    canvas = np.full((pixels + 2 * border, pixels + 2 * border), 255, np.uint8)
    canvas[border : border + pixels, border : border + pixels] = code

    path = marker_texture_path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    return path


def texture_size_m(marker: Marker) -> float:
    """Side length of the whole printed tile, black square plus quiet zone."""
    return marker.size_m * (1.0 + 2.0 * QUIET_ZONE_FRACTION)


def marker_corners_local(marker: Marker) -> np.ndarray:
    """The four corners of the black square, in the marker's own frame.

    Ordered to match what ``cv2.aruco`` returns -- top-left, top-right,
    bottom-right, bottom-left -- because ``solvePnP`` pairs these with the
    detected pixels by index. Getting the order wrong yields a pose that is
    confidently wrong rather than an error.

    The marker lies in its own xy plane with +z out of its face.
    """
    half = marker.size_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ]
    )


@dataclass(frozen=True)
class Detection:
    """One marker found in one image."""

    aruco_id: int
    corners_px: np.ndarray  # (4, 2), detector order
    pose_camera: np.ndarray  # T_camera_marker, OpenCV convention
    reprojection_error_px: float
    time_s: float

    @property
    def position_camera(self) -> np.ndarray:
        return self.pose_camera[:3, 3]


def detect(view, params: ArmParams | None = None) -> list[Detection]:
    """Find every configured marker in a rendered view and solve for its pose.

    ``IPPE_SQUARE`` is used rather than the general solver: it is the method
    specialised for planar square targets, and it is both faster and better
    conditioned than treating four coplanar points as a general PnP problem.

    The reprojection error is carried on every detection because it is the only
    cheap signal that a pose is wrong. A detection can be crisp and its pose
    still nonsense -- from the planar ambiguity, or a mis-stated marker size --
    and reprojection error is what distinguishes those from a good fit.
    """
    import cv2

    params = params or default_params()
    if not params.markers:
        return []

    gray = view.gray
    results: list[Detection] = []

    by_dictionary: dict[str, list[Marker]] = {}
    for marker in params.markers:
        by_dictionary.setdefault(marker.dictionary, []).append(marker)

    for name, markers in by_dictionary.items():
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        detector_params = cv2.aruco.DetectorParameters()
        # Sub-pixel refinement, because the default corner locations are integer
        # pixels and depth is inferred from apparent size. Measured on the same
        # frame, this took position error from 7.53 mm to 1.19 mm for no
        # meaningful cost -- the single cheapest accuracy gain in the pipeline.
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corner_sets, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue

        wanted = {marker.aruco_id: marker for marker in markers}
        for corners, identifier in zip(corner_sets, ids.ravel(), strict=True):
            marker = wanted.get(int(identifier))
            if marker is None:
                continue

            image_points = corners[0].astype(np.float64)
            object_points = marker_corners_local(marker)

            ok, rotation, translation = cv2.solvePnP(
                object_points,
                image_points,
                view.intrinsics,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            projected, _ = cv2.projectPoints(
                object_points, rotation, translation, view.intrinsics, None
            )
            error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))

            matrix, _ = cv2.Rodrigues(rotation)
            results.append(
                Detection(
                    aruco_id=int(identifier),
                    corners_px=image_points,
                    pose_camera=se3(matrix, translation.ravel()),
                    reprojection_error_px=error,
                    time_s=view.time_s,
                )
            )

    return results


def viewing_angle_deg(
    marker: Marker, q: np.ndarray, camera_position, params: ArmParams | None = None
) -> float:
    """Angle between the marker's normal and the line of sight, in degrees.

    Zero is face-on. Detection degrades as this grows, because an oblique square
    projects to a smaller quadrilateral and the code cells inside it get harder
    to read. Measured on this setup, markers are found reliably below about 49
    degrees and were missed at 52 -- with all four corners comfortably inside
    the frame, so it is obliquity rather than visibility that fails.

    This is what makes collecting calibration poses a search rather than a
    sweep: a configuration can be perfectly reachable and still produce no
    observation at all.
    """
    pose = marker_pose_world(marker, q, params)
    normal = pose[:3, 2]
    to_camera = np.asarray(camera_position, dtype=float) - pose[:3, 3]
    to_camera = to_camera / np.linalg.norm(to_camera)
    return float(np.degrees(np.arccos(np.clip(np.dot(normal, to_camera), -1.0, 1.0))))


def marker_pose_world(marker: Marker, q: np.ndarray, params: ArmParams | None = None) -> np.ndarray:
    """Where a marker actually is, from forward kinematics.

    Ground truth, for scoring a detection. A perception pipeline must never use
    this -- it is exactly the thing the camera is supposed to find out.
    """
    from arm.kinematics import link_frames

    params = params or default_params()
    index = [link.name for link in params.links].index(marker.link)
    # link_frames, not fk_frames: a marker is attached to a *body*, and a body's
    # frame includes its own joint rotation. See link_frames for why that
    # distinction is worth a separate function.
    link_pose = link_frames(q, params)[index]

    roll, pitch, yaw = np.radians(marker.euler_deg)
    from arm.frames import rot_x, rot_y, rot_z

    rotation = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return link_pose @ se3(rotation, np.asarray(marker.pos_m, dtype=float))
