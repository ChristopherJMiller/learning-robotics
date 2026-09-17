"""Configuration space and collision checking.

The claims worth pinning are about *connectivity*, not just about the checker
working: whether planning is necessary at all is a property of the world, and it
turned out to be a close call. With only a floor, every randomly sampled pair of
valid configurations was directly connectable by a straight line -- the free
region is nearly convex and a planner would have nothing to do. One post in the
workspace changes that, and these tests make sure it stays changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.cspace import CollisionChecker, occupancy_grid
from arm.kinematics import fk

# Reaching to either side of the post: bearing 65 and 115 degrees.
START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


@pytest.fixture(scope="module")
def checker():
    return CollisionChecker()


def test_the_world_contains_an_obstacle(params):
    """Without one, there is nothing to plan around."""
    assert params.obstacles, "no obstacles configured; planning would be trivial"


def test_obstacle_reaches_the_simulator(checker, params):
    import mujoco

    names = {
        mujoco.mj_id2name(checker.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(checker.model.ngeom)
    }
    for obstacle in params.obstacles:
        assert f"obstacle_{obstacle.name}" in names


def test_straight_line_endpoints_are_valid(checker):
    assert checker.is_valid(START)
    assert checker.is_valid(GOAL)


def test_the_straight_line_between_them_is_not(checker):
    """The demo case. If this ever passes, the demo stops making its point."""
    assert not checker.segment_is_valid(START, GOAL)


def test_a_configuration_inside_the_obstacle_is_blocked(checker, params):
    """Sanity: drive the tip into the post and the checker must notice."""
    post = params.obstacles[0]
    from arm.kinematics import ik_nearest

    solution = ik_nearest(np.array(post.pos_m), np.zeros(3), params)
    assert solution is not None, "the post should be inside the workspace"
    assert checker.in_collision(solution.q)


def test_limits_are_enforced_separately_from_collision(checker, params):
    """A configuration can be collision-free and still out of bounds."""
    beyond = params.joint_limits_rad[:, 1] + 0.5
    assert not checker.within_limits(beyond)
    assert not checker.is_valid(beyond)


def test_sample_stays_inside_the_limits(checker):
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert checker.within_limits(checker.sample(rng))


def test_sample_valid_returns_collision_free_configurations(checker):
    rng = np.random.default_rng(1)
    for _ in range(50):
        assert not checker.in_collision(checker.sample_valid(rng))


def test_about_half_of_configuration_space_is_blocked(checker):
    """Mostly the floor: the joint limits let the arm swing below the table.

    Worth pinning because it is the number that makes planning non-trivial, and
    a change to joint limits or link geometry would move it silently.
    """
    rng = np.random.default_rng(0)
    samples = rng.uniform(checker.limits[:, 0], checker.limits[:, 1], size=(4000, 3))
    blocked = np.mean([checker.in_collision(q) for q in samples])
    assert 0.45 < blocked < 0.65


def test_free_space_is_not_fully_connected(checker):
    """The property that makes a planner necessary rather than decorative.

    With only a floor this failed: 107 of 107 valid pairs were connectable in a
    straight line. The obstacle is what makes some pairs genuinely need a search.
    """
    rng = np.random.default_rng(7)
    connectable = pairs = 0
    while pairs < 60:
        start, goal = checker.sample(rng), checker.sample(rng)
        if not (checker.is_valid(start) and checker.is_valid(goal)):
            continue
        pairs += 1
        connectable += checker.segment_is_valid(start, goal, resolution_rad=0.05)

    assert connectable < pairs, "every pair is directly connectable; nothing to plan"


def test_segment_check_is_symmetric(checker):
    assert checker.segment_is_valid(START, GOAL) == checker.segment_is_valid(GOAL, START)


def test_a_zero_length_segment_matches_the_point(checker):
    assert checker.segment_is_valid(START, START) == checker.is_valid(START)


def test_occupancy_grid_shape_and_axes():
    grid = occupancy_grid(resolution=8)
    assert grid.occupied.shape == (8, 8, 8)
    assert len(grid.axes) == 3
    assert all(len(axis) == 8 for axis in grid.axes)
    assert 0.0 < grid.blocked_fraction < 1.0


def test_grid_points_round_trip_to_coordinates():
    grid = occupancy_grid(resolution=6)
    blocked = grid.blocked_points()
    free = grid.free_points()

    assert len(blocked) + len(free) == 6**3
    assert blocked.shape[1] == 3

    checker = CollisionChecker()
    for q in blocked[:: max(1, len(blocked) // 10)]:
        assert checker.in_collision(q)
    for q in free[:: max(1, len(free) // 10)]:
        assert not checker.in_collision(q)


def test_the_demo_endpoints_reach_opposite_sides_of_the_post(params):
    """Guards the demo's premise, not just its arithmetic."""
    post_y = params.obstacles[0].pos_m[1]
    start_tip, goal_tip = fk(START, params), fk(GOAL, params)

    # Straddling the post in x, at matching height and distance out.
    assert start_tip[0] > 0.0 > goal_tip[0]
    assert start_tip[1] > 0.0 and goal_tip[1] > 0.0
    assert post_y > 0.0
    np.testing.assert_allclose(start_tip[1], goal_tip[1], atol=1e-3)
    np.testing.assert_allclose(start_tip[2], goal_tip[2], atol=1e-3)


def test_the_obstacle_is_clear_of_the_control_demos(params):
    """The post must not silently turn tracking results into contact results.

    Every control demo works between -34 and +34 degrees of base bearing. An
    obstacle inside that arc would change what sim_track, sim_sensing and
    sim_hold are measuring without any of them mentioning it.
    """
    for obstacle in params.obstacles:
        x, y, _ = obstacle.pos_m
        bearing = abs(np.degrees(np.arctan2(y, x)))
        assert bearing > 45.0, (
            f"obstacle {obstacle.name!r} at {bearing:.0f} deg intrudes on the "
            "control demos' workspace"
        )
