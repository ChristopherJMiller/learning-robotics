"""Motion planning.

The properties worth asserting about a randomised algorithm are not "it returns
the right answer" -- there isn't one -- but that whatever it returns is *valid*,
that it succeeds reliably across seeds, and that the pieces do what their names
claim. Path quality is measured in ``scripts/plan_compare.py`` rather than
pinned here, because it is a distribution, not a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.cspace import CollisionChecker
from arm.planning import path_length, rrt, rrt_star, shortcut

# Reaching to either side of the post: bearing 65 and 115 degrees.
START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


@pytest.fixture(scope="module")
def checker():
    return CollisionChecker()


def _is_executable(path, checker, resolution_rad: float = 0.02) -> bool:
    """Every segment of the path must be traversable, not merely its waypoints.

    The distinction matters: a path whose waypoints are all valid can still
    sweep the arm through an obstacle between two of them.
    """
    # zip over consecutive pairs is deliberately ragged -- path[1:] is one
    # shorter -- so strict=True could never hold here.
    return all(
        checker.segment_is_valid(first, second, resolution_rad)
        for first, second in zip(path, path[1:], strict=False)
    )


# --------------------------------------------------------------------------
# The problem is real
# --------------------------------------------------------------------------


def test_the_straight_line_is_blocked(checker):
    """If this stops holding, every test below is testing nothing."""
    assert checker.is_valid(START)
    assert checker.is_valid(GOAL)
    assert not checker.segment_is_valid(START, GOAL)


# --------------------------------------------------------------------------
# RRT
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_rrt_finds_a_path_on_every_seed(checker, seed):
    result = rrt(START, GOAL, checker, seed=seed)
    assert result.found, result.describe()


@pytest.mark.parametrize("seed", range(3))
def test_rrt_paths_are_executable(checker, seed):
    """Valid waypoints are not enough; the segments between them must be clear."""
    result = rrt(START, GOAL, checker, seed=seed)
    assert _is_executable(result.path, checker)


def test_rrt_path_starts_and_ends_where_asked(checker):
    result = rrt(START, GOAL, checker, seed=0)
    np.testing.assert_allclose(result.path[0], START)
    np.testing.assert_allclose(result.path[-1], GOAL)


def test_rrt_is_deterministic_for_a_seed(checker):
    first = rrt(START, GOAL, checker, seed=3)
    second = rrt(START, GOAL, checker, seed=3)
    assert path_length(first.path) == path_length(second.path)


def test_rrt_rejects_an_invalid_start(checker, params):
    inside_post = np.array(params.obstacles[0].pos_m)
    from arm.kinematics import ik_nearest

    blocked = ik_nearest(inside_post, np.zeros(3), params)
    with pytest.raises(ValueError, match="start configuration"):
        rrt(blocked.q, GOAL, checker)


def test_rrt_reports_not_found_rather_than_hanging(checker):
    """Failure means 'not found in the budget', never 'impossible'."""
    result = rrt(START, GOAL, checker, max_iterations=3, seed=0)
    assert not result.found
    assert result.iterations == 3
    assert result.length == float("inf")


def test_goal_bias_helps(checker):
    """Sanity check on the one parameter that steers the search."""
    without = [
        rrt(START, GOAL, checker, goal_bias=0.0, max_iterations=400, seed=s).found for s in range(6)
    ]
    with_bias = [
        rrt(START, GOAL, checker, goal_bias=0.1, max_iterations=400, seed=s).found for s in range(6)
    ]
    assert sum(with_bias) >= sum(without)


# --------------------------------------------------------------------------
# Shortcutting
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_shortcutting_never_lengthens_a_path(checker, seed):
    result = rrt(START, GOAL, checker, seed=seed)
    shortened = shortcut(result.path, checker, seed=seed)
    assert path_length(shortened) <= path_length(result.path) + 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_shortcut_paths_are_still_executable(checker, seed):
    """The failure that would matter: a shorter path that cuts a corner."""
    result = rrt(START, GOAL, checker, seed=seed)
    shortened = shortcut(result.path, checker, seed=seed)
    assert _is_executable(shortened, checker)


def test_shortcutting_preserves_the_endpoints(checker):
    result = rrt(START, GOAL, checker, seed=0)
    shortened = shortcut(result.path, checker, seed=0)
    np.testing.assert_allclose(shortened[0], START)
    np.testing.assert_allclose(shortened[-1], GOAL)


def test_shortcutting_reduces_variance_across_seeds(checker):
    """The main practical benefit, and the one most easily overlooked.

    Raw RRT is erratic -- sometimes near-optimal, sometimes wandering. For a
    planner that runs on every command, predictability matters more than the
    average case.
    """
    raw, cut = [], []
    for seed in range(8):
        result = rrt(START, GOAL, checker, seed=seed)
        raw.append(path_length(result.path))
        cut.append(path_length(shortcut(result.path, checker, seed=seed)))

    assert np.std(cut) < np.std(raw) / 3.0
    assert np.mean(cut) < np.mean(raw)


def test_a_coarse_resolution_really_does_produce_colliding_paths(checker):
    """Justifies the 0.01 rad default, which is otherwise an arbitrary number.

    Segment checking subdivides and samples; it does not prove. At the original
    0.05 rad default, two of eight planned paths were genuinely in collision
    when rechecked more finely -- paths that would have driven the arm through
    the post. This test asserts the danger is real so the default is not
    loosened back for speed.
    """
    coarse_failures = 0
    for seed in range(8):
        result = rrt(START, GOAL, checker, seed=seed, resolution_rad=0.05)
        if not result.found:
            continue
        path = shortcut(result.path, checker, seed=seed, resolution_rad=0.05)
        if not _is_executable(path, checker, resolution_rad=0.005):
            coarse_failures += 1

    assert coarse_failures > 0, (
        "a 0.05 rad check no longer misses collisions; the default could be "
        "revisited, but re-measure before loosening it"
    )


@pytest.mark.parametrize("seed", range(6))
def test_the_default_resolution_survives_a_finer_recheck(checker, seed):
    """The other half: at the chosen default, paths hold up under scrutiny."""
    result = rrt(START, GOAL, checker, seed=seed)
    path = shortcut(result.path, checker, seed=seed)
    assert _is_executable(path, checker, resolution_rad=0.005)


def test_shortcutting_a_trivial_path_is_a_no_op(checker):
    assert shortcut(None, checker) == []
    single = [START]
    assert len(shortcut(single, checker)) == 1


def test_shortcut_charges_its_own_collision_checks(checker):
    """So a comparison cannot make it look free."""
    result = rrt(START, GOAL, checker, seed=0)
    spent = [0]
    shortcut(result.path, checker, seed=0, checks=spent)
    assert spent[0] > 0


# --------------------------------------------------------------------------
# RRT*
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_rrt_star_finds_executable_paths(checker, seed):
    result = rrt_star(START, GOAL, checker, max_iterations=600, seed=seed)
    assert result.found, result.describe()
    assert _is_executable(result.path, checker)


def test_rrt_star_beats_raw_rrt_on_length(checker):
    """Asymptotic optimality should be visible, even at modest sample counts."""
    plain = [path_length(rrt(START, GOAL, checker, seed=s).path) for s in range(5)]
    starred = [
        path_length(rrt_star(START, GOAL, checker, max_iterations=800, seed=s).path)
        for s in range(5)
    ]
    assert np.mean(starred) < np.mean(plain)


def test_rrt_star_costs_more_checks_than_rrt(checker):
    """Rewiring is not free: each sample triggers a neighbourhood search."""
    plain = rrt(START, GOAL, checker, seed=0)
    starred = rrt_star(START, GOAL, checker, max_iterations=600, seed=0)
    assert starred.collision_checks > plain.collision_checks


def test_rrt_star_gains_little_from_shortcutting(checker):
    """Because it has already found roughly the path shortcutting would give.

    The converse of the RRT result, and the clearest sign the rewiring is
    actually doing what it claims.
    """
    result = rrt_star(START, GOAL, checker, max_iterations=800, seed=0)
    shortened = shortcut(result.path, checker, seed=0)
    assert path_length(shortened) > 0.95 * path_length(result.path)


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# What actually makes rewiring worth paying for
# --------------------------------------------------------------------------


def _wall(params):
    """A low wall across the workspace, leaving a route over it and around it."""
    from arm.params import Obstacle

    posts = []
    for index, radius in enumerate((0.10, 0.16, 0.22, 0.28)):
        posts.append(
            Obstacle(
                name=f"w{index}",
                pos_m=(0.0, radius, 0.06),
                half_size_m=(0.020, 0.020, 0.06),
            )
        )
    return params.with_obstacles(posts)


def test_collision_checker_respects_the_params_it_is_given(params):
    """A regression test for a bug that silently invalidated an experiment.

    CollisionChecker used to load the committed MJCF regardless of the
    parameters passed to it, so a world with extra obstacles was checked
    against the old one. A clutter comparison returned byte-identical numbers
    for one obstacle and for seven, which is what gave it away.
    """
    import mujoco

    checker = CollisionChecker(_wall(params))
    names = {
        mujoco.mj_id2name(checker.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(checker.model.ngeom)
    }
    assert "obstacle_w3" in names
    assert "obstacle_post" not in names


def test_a_wall_creates_distinct_routes(params):
    """The property that makes rewiring pay, stated as a measurement.

    A single post leaves essentially one sensible way around, so every seed
    finds roughly the same path. A wall offers a choice -- over the top, or
    around the end -- and RRT commits to whichever it stumbles into, so its
    path lengths spread out across seeds.

    That spread is the thing shortcutting cannot fix, because it can only
    polish the route it was handed.
    """
    one_post = CollisionChecker(params)
    wall = CollisionChecker(_wall(params))

    def spread(checker) -> float:
        lengths = []
        for seed in range(8):
            result = rrt(START, GOAL, checker, seed=seed, max_iterations=8000)
            if result.found:
                lengths.append(path_length(shortcut(result.path, checker, seed=seed)))
        return max(lengths) / min(lengths)

    assert spread(wall) > spread(one_post)


def test_rewiring_helps_more_when_routes_differ(params):
    """More obstacles is not the point; different routes is the point."""
    wall = CollisionChecker(_wall(params))

    plain, starred = [], []
    for seed in range(6):
        result = rrt(START, GOAL, wall, seed=seed, max_iterations=8000)
        if result.found:
            plain.append(path_length(shortcut(result.path, wall, seed=seed)))
        best = rrt_star(START, GOAL, wall, max_iterations=1200, seed=seed)
        if best.found:
            starred.append(path_length(shortcut(best.path, wall, seed=seed)))

    assert plain and starred
    assert np.mean(starred) < np.mean(plain)


def test_path_length_of_a_straight_line_is_the_distance():
    start, end = np.zeros(3), np.array([0.3, 0.4, 0.0])
    midpoint = (start + end) / 2
    np.testing.assert_allclose(path_length([start, midpoint, end]), 0.5)


def test_path_length_handles_degenerate_input():
    assert path_length(None) == 0.0
    assert path_length([np.zeros(3)]) == 0.0


def test_tree_edges_are_only_kept_when_asked(checker):
    assert rrt(START, GOAL, checker, seed=0).tree_edges == []
    assert rrt(START, GOAL, checker, seed=0, keep_tree=True).tree_edges
