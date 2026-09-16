"""Forward and inverse kinematics for the 3-DOF articulated (RRR) arm.

Everything here is derived by hand rather than delegated to a library. That is
deliberate: for this geometry the derivation is short, and owning it is what
turns "the solver warned me about a singularity" into "I can see why
``det(J)`` goes to zero here". ``tests/test_kinematics.py`` checks all of it
against Pinocchio and against finite differences, so the hand derivation is
verified rather than merely trusted.

Conventions
-----------
Joint zero has the arm extended straight out along +x, horizontal. Positive
``theta2``/``theta3`` lift the arm. Angles are radians, lengths are metres.

Geometry, in the vertical plane reached by rotating the base through ``theta1``::

    r = L1*cos(t2) + L2*cos(t2 + t3)
    z = L0 + L1*sin(t2) + L2*sin(t2 + t3)
    x = r*cos(t1)
    y = r*sin(t1)

This arm has three joints and therefore three degrees of freedom, so it can
reach a *position* anywhere in its workspace but cannot independently choose
the end-effector *orientation*. That is why every function here takes or
returns a 3-vector position rather than a full SE(3) pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from arm.frames import rot_y, rot_z, se3, wrap_to_pi
from arm.params import ArmParams, default_params

__all__ = [
    "Elbow",
    "IKSolution",
    "Shoulder",
    "denominator_det",
    "fk",
    "fk_frames",
    "ik",
    "ik_nearest",
    "jacobian",
    "jacobian_det",
    "manipulability",
    "within_limits",
]


class Elbow(str, Enum):
    """Which of the two planar branches the elbow takes to reach a point."""

    UP = "up"
    DOWN = "down"


class Shoulder(str, Enum):
    """Whether the base faces the target or is flipped 180 degrees away."""

    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class IKSolution:
    """One of the (up to four) joint configurations reaching a given point."""

    q: np.ndarray
    elbow: Elbow
    shoulder: Shoulder
    within_limits: bool

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        degrees = np.degrees(self.q)
        limits = "ok" if self.within_limits else "OUT OF LIMITS"
        return (
            f"IKSolution({degrees[0]:.1f} deg, {degrees[1]:.1f} deg, "
            f"{degrees[2]:.1f} deg, {self.elbow.value}/{self.shoulder.value}, {limits})"
        )


def _lengths(params: ArmParams | None) -> tuple[float, float, float]:
    return (params or default_params()).link_lengths_m


def fk(q: np.ndarray, params: ArmParams | None = None) -> np.ndarray:
    """Forward kinematics: joint angles -> end-effector position.

    Always succeeds, always returns exactly one answer. This is the easy
    direction; see :func:`ik` for why the inverse is not symmetric with it.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (3,):
        raise ValueError(f"q must be a 3-vector, got shape {q.shape}")

    length_base, length_upper, length_fore = _lengths(params)
    t1, t2, t3 = q

    radius = length_upper * np.cos(t2) + length_fore * np.cos(t2 + t3)
    height = length_base + length_upper * np.sin(t2) + length_fore * np.sin(t2 + t3)

    return np.array([radius * np.cos(t1), radius * np.sin(t1), height])


def fk_frames(q: np.ndarray, params: ArmParams | None = None) -> list[np.ndarray]:
    """Every intermediate frame, as ``T_base_x`` transforms.

    Returns ``[T_base_base, T_base_shoulder, T_base_elbow, T_base_ee]``. Useful
    for drawing the arm and for seeing that forward kinematics really is just
    a chain of matrix multiplications.
    """
    q = np.asarray(q, dtype=float)
    length_base, length_upper, length_fore = _lengths(params)
    t1, t2, t3 = q

    t_base = np.eye(4)
    t_shoulder = t_base @ se3(rot_z(t1), np.array([0.0, 0.0, length_base]))
    # se3(R, p) maps v -> R@v + p, i.e. p is measured in the PARENT frame.
    # A link offset must therefore be rotated into place before it is applied:
    # rotate about the joint axis first, then translate along the new x axis.
    t_elbow = (
        t_shoulder
        @ se3(rot_y(-t2), np.zeros(3))
        @ se3(np.eye(3), np.array([length_upper, 0.0, 0.0]))
    )
    t_ee = (
        t_elbow
        @ se3(rot_y(-t3), np.zeros(3))
        @ se3(np.eye(3), np.array([length_fore, 0.0, 0.0]))
    )
    return [t_base, t_shoulder, t_elbow, t_ee]


def jacobian(q: np.ndarray, params: ArmParams | None = None) -> np.ndarray:
    """Analytic position Jacobian: ``dx = J(q) dq``.

    Shape (3, 3) because this arm controls position only. Being square means
    it is invertible away from singularities, which is what makes resolved-rate
    control and Newton-style numerical IK straightforward here.

    ``J`` depends on ``q``: how the arm responds to joint velocity changes
    completely with configuration.
    """
    q = np.asarray(q, dtype=float)
    _, length_upper, length_fore = _lengths(params)
    t1, t2, t3 = q

    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)

    radius = length_upper * c2 + length_fore * c23
    d_radius_d_t2 = -(length_upper * s2 + length_fore * s23)
    d_radius_d_t3 = -length_fore * s23
    d_height_d_t2 = length_upper * c2 + length_fore * c23
    d_height_d_t3 = length_fore * c23

    return np.array(
        [
            [-radius * s1, c1 * d_radius_d_t2, c1 * d_radius_d_t3],
            [radius * c1, s1 * d_radius_d_t2, s1 * d_radius_d_t3],
            [0.0, d_height_d_t2, d_height_d_t3],
        ]
    )


def jacobian_det(q: np.ndarray, params: ArmParams | None = None) -> float:
    """Closed-form determinant of the Jacobian.

    Working the 3x3 determinant through by hand collapses to a strikingly
    simple result::

        det(J) = -r * L1 * L2 * sin(t3)

    which names both of this arm's singularities directly:

    * ``sin(t3) == 0`` -- the elbow is straight or fully folded. A *boundary*
      singularity: the tip is at the edge of the workspace and can no longer
      move radially outward.
    * ``r == 0`` -- the tip sits on the base rotation axis. An *interior*
      singularity, and the dangerous one: rotating ``theta1`` moves the tip
      nowhere, so a straight-line path passing near it demands enormous base
      velocity. This is the classic "the robot suddenly went berserk mid-path".

    :func:`jacobian_det` and ``np.linalg.det(jacobian(q))`` agree to machine
    precision; the test suite asserts it.
    """
    q = np.asarray(q, dtype=float)
    _, length_upper, length_fore = _lengths(params)
    t1, t2, t3 = q

    radius = length_upper * np.cos(t2) + length_fore * np.cos(t2 + t3)
    return float(-radius * length_upper * length_fore * np.sin(t3))


# Kept as an explicit alias: the phrase "the determinant" is ambiguous once
# damped least squares enters, where the damped denominator matters instead.
denominator_det = jacobian_det


def manipulability(q: np.ndarray, params: ArmParams | None = None) -> float:
    """Yoshikawa's measure ``sqrt(det(J J^T))`` -- distance from singularity.

    Zero exactly at a singularity, larger where the arm can move freely in all
    directions. Plotting this over the workspace makes the singular surfaces
    visible.
    """
    jac = jacobian(q, params)
    return float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))


def within_limits(q: np.ndarray, params: ArmParams | None = None) -> bool:
    """Whether every joint angle lies inside its configured range."""
    q = np.asarray(q, dtype=float)
    limits = (params or default_params()).joint_limits_rad
    return bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))


def ik(
    target: np.ndarray,
    params: ArmParams | None = None,
    *,
    tolerance: float = 1e-9,
) -> list[IKSolution]:
    """Inverse kinematics: position -> every joint configuration reaching it.

    Unlike :func:`fk`, this is **not a function** -- it is set-valued. A target
    may have zero solutions (outside the workspace), or up to four:

    * two planar branches, elbow up and elbow down;
    * times two base orientations, facing the target or flipped 180 degrees.

    Solutions violating joint limits are returned with ``within_limits=False``
    rather than dropped, so that callers can see *why* a target is
    unreachable in practice rather than merely that it is.

    Because the caller must choose among the results, every consumer needs a
    selection policy; see :func:`ik_nearest`. Choosing arbitrarily is a real
    hazard: if consecutive waypoints resolve to different branches, the arm
    will try to flip through configuration space instantly.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"target must be a 3-vector, got shape {target.shape}")

    length_base, length_upper, length_fore = _lengths(params)
    x, y, z = target
    height = z - length_base
    radius = float(np.hypot(x, y))

    base_forward = float(np.arctan2(y, x))
    solutions: list[IKSolution] = []

    for shoulder in (Shoulder.FORWARD, Shoulder.BACKWARD):
        if shoulder is Shoulder.FORWARD:
            t1 = base_forward
            reach = radius
        else:
            # Flipping the base 180 degrees lets the arm reach "backwards",
            # which is the same planar problem with a negated radius.
            t1 = float(wrap_to_pi(base_forward + np.pi))
            reach = -radius

        # Law of cosines on the planar two-link subproblem.
        cos_t3 = (
            reach**2 + height**2 - length_upper**2 - length_fore**2
        ) / (2.0 * length_upper * length_fore)

        if cos_t3 > 1.0 + tolerance or cos_t3 < -1.0 - tolerance:
            continue  # target is outside this branch's annulus
        cos_t3 = float(np.clip(cos_t3, -1.0, 1.0))

        for elbow in (Elbow.UP, Elbow.DOWN):
            t3 = -np.arccos(cos_t3) if elbow is Elbow.UP else np.arccos(cos_t3)
            t2 = np.arctan2(height, reach) - np.arctan2(
                length_fore * np.sin(t3), length_upper + length_fore * np.cos(t3)
            )
            q = np.array([t1, float(wrap_to_pi(t2)), float(t3)])
            solutions.append(
                IKSolution(
                    q=q,
                    elbow=elbow,
                    shoulder=shoulder,
                    within_limits=within_limits(q, params),
                )
            )

            if abs(cos_t3) >= 1.0 - tolerance:
                break  # degenerate: both branches coincide, do not duplicate

    return solutions


def ik_nearest(
    target: np.ndarray,
    q_current: np.ndarray,
    params: ArmParams | None = None,
    *,
    require_limits: bool = True,
) -> IKSolution | None:
    """Pick the reachable solution closest to the current configuration.

    The default selection policy, and the one that keeps a trajectory
    continuous: never change branch mid-path unless forced to.
    """
    q_current = np.asarray(q_current, dtype=float)
    candidates = ik(target, params)
    if require_limits:
        candidates = [s for s in candidates if s.within_limits]
    if not candidates:
        return None
    return min(candidates, key=lambda s: float(np.linalg.norm(s.q - q_current)))
