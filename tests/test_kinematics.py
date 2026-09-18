"""Verification of the hand-derived kinematics.

The derivation in ``arm.kinematics`` is done by hand, so it needs independent
confirmation. Three kinds of check appear here:

* **Round trips** -- ``fk(ik(p)) == p``, which ties the two directions together.
* **Independent recomputation** -- the analytic Jacobian against finite
  differences of ``fk``, and the closed-form determinant against a general
  matrix determinant.
* **Named behaviour** -- the two singularities land exactly where the geometry
  says they should.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from arm.kinematics import (
    Elbow,
    fk,
    fk_frames,
    fk_frames_relative,
    ik,
    ik_nearest,
    jacobian,
    jacobian_det,
    manipulability,
)
from conftest import joint_angles, nonsingular_joint_angles

SETTINGS = settings(max_examples=250, deadline=None)


# --------------------------------------------------------------------------
# Forward kinematics
# --------------------------------------------------------------------------


def test_zero_configuration_is_straight_out_along_x(params):
    length_base, length_upper, length_fore = params.link_lengths_m
    position = fk(np.zeros(3), params)
    np.testing.assert_allclose(position, [length_upper + length_fore, 0.0, length_base], atol=1e-12)


def test_base_rotation_sweeps_the_plane(params):
    """theta1 rotates the whole arm about z without changing radius or height."""
    q = np.array([0.0, 0.3, -0.4])
    reference = fk(q, params)
    for angle in np.linspace(-np.pi, np.pi, 17):
        rotated = fk(np.array([angle, q[1], q[2]]), params)
        assert np.isclose(np.hypot(*rotated[:2]), np.hypot(*reference[:2]))
        assert np.isclose(rotated[2], reference[2])


@given(q=joint_angles())
@SETTINGS
def test_fk_frames_chain_reproduces_fk(q):
    """The last frame of the chain is the end effector -- FK really is matmul."""
    frames = fk_frames(q)
    np.testing.assert_allclose(frames[-1][:3, 3], fk(q), atol=1e-12)


@given(q=joint_angles())
@SETTINGS
def test_every_frame_is_a_valid_rigid_transform(q):
    for transform in fk_frames(q):
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)
        np.testing.assert_allclose(transform[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-15)


@given(q=joint_angles())
@SETTINGS
def test_tip_stays_within_reach(
    q,
):
    from arm.params import default_params

    geometry = default_params().geometry
    position = fk(q)
    radius = np.hypot(*position[:2])
    height = position[2] - geometry.base_height_m
    distance = np.hypot(radius, height)
    assert distance <= geometry.max_reach_m + 1e-9


# --------------------------------------------------------------------------
# Inverse kinematics
# --------------------------------------------------------------------------


@given(q=joint_angles())
@SETTINGS
def test_fk_ik_round_trip(q):
    """Every reachable point admits at least one solution that reaches it.

    This is the headline property: the two directions are consistent.
    """
    target = fk(q)
    solutions = ik(target)
    assert solutions, f"no IK solution for a point produced by FK: {target}"
    errors = [np.linalg.norm(fk(s.q) - target) for s in solutions]
    assert min(errors) < 1e-9


@given(q=nonsingular_joint_angles())
@SETTINGS
def test_every_ik_solution_actually_reaches_the_target(q):
    """Not just the best one -- all four branches must be genuine solutions."""
    target = fk(q)
    for solution in ik(target):
        np.testing.assert_allclose(fk(solution.q), target, atol=1e-9)


@given(q=nonsingular_joint_angles())
@SETTINGS
def test_ik_finds_both_elbow_branches(q):
    """Away from singularities the elbow-up and elbow-down branches are distinct."""
    solutions = ik(fk(q))
    elbows = {s.elbow for s in solutions}
    assert elbows == {Elbow.UP, Elbow.DOWN}


def test_ik_returns_nothing_outside_the_workspace(params):
    unreachable = np.array([params.geometry.max_reach_m * 3.0, 0.0, 0.0])
    assert ik(unreachable, params) == []


def test_ik_nearest_preserves_branch_continuity(params):
    """Consecutive waypoints must not flip the elbow; that would be a lurch."""
    q = np.array([0.0, 0.5, -1.0])
    for _ in range(40):
        target = fk(q, params) + np.array([0.0, 0.002, 0.0])
        solution = ik_nearest(target, q, params)
        assert solution is not None
        assert np.linalg.norm(solution.q - q) < 0.15, "branch flip between waypoints"
        q = solution.q


def test_ik_reports_limit_violations_rather_than_hiding_them(params):
    """Out-of-limit solutions are returned flagged, so callers can see why."""
    target = fk(np.array([0.0, 0.4, -0.8]), params)
    solutions = ik(target, params)
    assert any(not s.within_limits for s in solutions), (
        "expected the backward-shoulder family to violate limits"
    )
    assert any(s.within_limits for s in solutions)


# --------------------------------------------------------------------------
# Jacobian
# --------------------------------------------------------------------------


@given(q=joint_angles())
@SETTINGS
def test_analytic_jacobian_matches_finite_differences(q):
    """The check that makes a hand-derived Jacobian trustworthy."""
    step = 1e-6
    numerical = np.zeros((3, 3))
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = step
        numerical[:, j] = (fk(q + delta) - fk(q - delta)) / (2.0 * step)
    np.testing.assert_allclose(jacobian(q), numerical, atol=1e-7)


@given(q=joint_angles())
@SETTINGS
def test_closed_form_determinant_matches_general_determinant(q):
    """det(J) = -r * L1 * L2 * sin(theta3), worked out by hand."""
    assert np.isclose(jacobian_det(q), np.linalg.det(jacobian(q)), atol=1e-12)


@given(q=nonsingular_joint_angles())
@SETTINGS
def test_velocity_round_trip_away_from_singularities(q):
    """J maps joint rates to tip velocity, and its inverse maps back."""
    jac = jacobian(q)
    q_dot = np.array([0.3, -0.2, 0.45])
    x_dot = jac @ q_dot
    np.testing.assert_allclose(np.linalg.solve(jac, x_dot), q_dot, atol=1e-8)


@given(q=nonsingular_joint_angles())
@SETTINGS
def test_numerical_ik_converges_to_a_closed_form_solution(q):
    """Newton iteration on the Jacobian must agree with the algebraic answer.

    Two entirely independent methods, one target: strong evidence both are right.
    """
    target = fk(q)
    guess = q + np.random.default_rng(0).normal(scale=0.05, size=3)

    for _ in range(100):
        error = target - fk(guess)
        if np.linalg.norm(error) < 1e-12:
            break
        guess = guess + np.linalg.solve(jacobian(guess), error)

    np.testing.assert_allclose(fk(guess), target, atol=1e-8)


# --------------------------------------------------------------------------
# Singularities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("elbow_angle", [0.0, np.pi, -np.pi])
def test_straight_or_folded_elbow_is_singular(elbow_angle, params):
    """Boundary singularity: nothing left to extend, so radial motion is lost."""
    q = np.array([0.3, 0.4, elbow_angle])
    assert np.isclose(jacobian_det(q, params), 0.0, atol=1e-12)
    assert np.isclose(manipulability(q, params), 0.0, atol=1e-9)


def test_tip_on_the_base_axis_is_singular(params):
    """Interior singularity -- the dangerous one, reachable mid-path.

    Folding the arm so the tip sits on the base rotation axis makes theta1
    move the tip nowhere.
    """
    _, length_upper, length_fore = params.link_lengths_m
    assert np.isclose(length_upper, length_fore), "this construction needs equal links"

    q = np.array([0.0, 0.7, np.pi - 2 * 0.7])
    radius = np.hypot(*fk(q, params)[:2])
    assert np.isclose(radius, 0.0, atol=1e-12), "expected the tip on the base axis"
    assert np.isclose(jacobian_det(q, params), 0.0, atol=1e-12)

    # The first Jacobian column is exactly zero: theta1 does nothing here.
    np.testing.assert_allclose(jacobian(q, params)[:, 0], np.zeros(3), atol=1e-12)


@given(q=joint_angles())
@SETTINGS
def test_manipulability_is_never_negative(q):
    assert manipulability(q) >= 0.0


def test_manipulability_collapses_at_the_elbow_singularity(params):
    """Manipulability vanishes as the elbow straightens -- but not monotonically.

    Since |det J| = r * L1 * L2 * |sin(t3)|, straightening the elbow shrinks
    |sin(t3)| while simultaneously *growing* r, because the arm reaches
    further out. The product can therefore tick upward before it collapses,
    so the honest property is the limit, not monotonicity.
    """
    angles = np.linspace(-1.2, -0.02, 12)
    values = [manipulability(np.array([0.2, 0.3, a]), params) for a in angles]

    # Well away from the singularity it is comfortably conditioned ...
    assert values[0] > 1e-3
    # ... and it has collapsed by two orders of magnitude on approach ...
    assert values[-1] < values[0] / 30.0
    # ... reaching zero when the elbow is straight. Note the tolerance: because
    # manipulability is sqrt(det(J J^T)), the square root halves the available
    # significant digits near zero, so a determinant good to ~1e-20 only yields
    # ~1e-10 here. Use det(J) directly when you need a tight singularity test.
    assert np.isclose(manipulability(np.array([0.2, 0.3, 0.0]), params), 0.0, atol=1e-8)
    assert np.isclose(jacobian_det(np.array([0.2, 0.3, 0.0]), params), 0.0, atol=1e-15)

    # The decline is monotonic over the final stretch, where sin(t3) dominates.
    tail = values[3:]
    assert all(later < earlier for earlier, later in zip(tail, tail[1:], strict=False))


# --------------------------------------------------------------------------
# Transform hierarchy contract
# --------------------------------------------------------------------------


@given(q=joint_angles())
@SETTINGS
def test_relative_frames_compose_back_to_absolute(q):
    """Composing the relative chain must reproduce the absolute frames.

    This is the contract every transform hierarchy relies on -- Rerun's entity
    tree, tf2, a URDF joint chain. It is stated as a test because absolute and
    relative transforms have identical types: both are 4x4 arrays, so nothing
    else can tell them apart.

    Passing absolute poses where relative ones are expected is a real bug that
    produces no error and no wrong numbers anywhere else -- only a robot drawn
    at impossible angles, because the consumer multiplies the poses together.
    """
    absolute = fk_frames(q)
    relative = fk_frames_relative(q)

    composed = np.eye(4)
    for step, parent_to_child in enumerate(relative, start=1):
        composed = composed @ parent_to_child
        np.testing.assert_allclose(composed, absolute[step], atol=1e-12)


@given(q=joint_angles())
@SETTINGS
def test_composed_hierarchy_reaches_the_end_effector(q):
    """The full chain composed from relatives must land on fk(q)."""
    composed = np.eye(4)
    for parent_to_child in fk_frames_relative(q):
        composed = composed @ parent_to_child
    np.testing.assert_allclose(composed[:3, 3], fk(q), atol=1e-12)


# --------------------------------------------------------------------------
# point_jacobian -- the same question as jacobian(), asked anywhere on the arm
# --------------------------------------------------------------------------


def test_point_jacobian_matches_the_analytic_one_at_the_tip(params):
    """Two independent derivations of the same quantity must agree exactly.

    ``jacobian`` comes from differentiating the closed-form FK; ``point_jacobian``
    from the geometric construction ``a x (p - o)``. They share no code.
    """
    from arm.kinematics import point_jacobian

    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-2.0, 2.0, 3)
        np.testing.assert_allclose(
            point_jacobian(q, fk(q, params), params), jacobian(q, params), atol=1e-12
        )


def test_point_jacobian_matches_finite_differences_at_the_marker(params):
    """Off the tip, where ``jacobian`` cannot answer, check against calculus."""
    from arm.fiducial import marker_pose_world
    from arm.kinematics import point_jacobian

    marker = params.markers[0]
    rng = np.random.default_rng(1)
    step = 1e-6

    for _ in range(20):
        q = rng.uniform(-1.5, 1.5, 3)
        numeric = np.zeros((3, 3))
        for index in range(3):
            forward, backward = q.copy(), q.copy()
            forward[index] += step
            backward[index] -= step
            numeric[:, index] = (
                marker_pose_world(marker, forward, params)[:3, 3]
                - marker_pose_world(marker, backward, params)[:3, 3]
            ) / (2 * step)

        position = marker_pose_world(marker, q, params)[:3, 3]
        np.testing.assert_allclose(point_jacobian(q, position, params), numeric, atol=1e-8)


def test_a_point_on_the_base_axis_does_not_move_with_the_base_joint(params):
    """Rotating about an axis leaves points *on* that axis alone."""
    from arm.kinematics import point_jacobian

    q = np.array([0.4, 0.6, -0.9])
    on_axis = np.array([0.0, 0.0, 0.12])  # the base joint's axis is world z
    np.testing.assert_allclose(point_jacobian(q, on_axis, params)[:, 0], 0.0, atol=1e-15)
