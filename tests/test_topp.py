"""Time-optimal path parameterisation.

The properties that matter are that the result is *feasible*, that it is
genuinely faster than stretching everything uniformly, and that it gets there by
keeping the arm near its limit for most of the path rather than for one instant.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.cspace import CollisionChecker
from arm.dynamics import damping_torque, inverse_dynamics
from arm.planning import rrt, shortcut
from arm.scenarios import TUCK_AND_EXTEND
from arm.timing import time_optimal_scale, torque_limits, trajectory_from_path
from arm.topp import parameterise

START, GOAL = TUCK_AND_EXTEND.start, TUCK_AND_EXTEND.goal


@pytest.fixture(scope="module")
def geometry():
    checker = CollisionChecker()
    result = rrt(START, GOAL, checker, seed=0)
    assert result.found
    path = shortcut(result.path, checker, seed=0)
    return trajectory_from_path(path, speed_rad_s=8.0)


def _utilisation(trajectory, samples: int = 300):
    """Torque as a fraction of the limit, sampled along the trajectory."""
    limits = torque_limits(trajectory.params)
    ratios = []
    for setpoint in trajectory.sample(samples):
        torque = inverse_dynamics(setpoint.q, setpoint.dq, setpoint.ddq) + damping_torque(
            setpoint.dq
        )
        ratios.append(float(np.max(np.abs(torque) / limits)))
    return np.array(ratios)


# --------------------------------------------------------------------------
# The result must be executable
# --------------------------------------------------------------------------


def test_the_result_respects_the_torque_limits(geometry):
    """Feasible, not nearly feasible.

    The solved profile sits exactly at the limit by construction, so any
    discretisation error puts the realised trajectory over it. parameterise
    solves against a reduced limit and then verifies against the true one.
    """
    trajectory = parameterise(geometry)
    assert np.all(_utilisation(trajectory) <= 1.0 + 1e-6)


def test_it_starts_and_ends_at_rest(geometry):
    trajectory = parameterise(geometry)
    np.testing.assert_allclose(trajectory.at(0.0).dq, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(trajectory.at(trajectory.duration_s).dq, np.zeros(3), atol=1e-9)


def test_it_reaches_both_endpoints(geometry):
    trajectory = parameterise(geometry)
    np.testing.assert_allclose(trajectory.at(0.0).q, geometry.waypoints[0], atol=1e-6)
    np.testing.assert_allclose(
        trajectory.at(trajectory.duration_s).q, geometry.waypoints[-1], atol=1e-6
    )


def test_the_duration_is_finite_and_sensible(geometry):
    """Guards the two integration bugs that produced absurd durations.

    A clamped geometric spline made f'(0) = 0, so no path acceleration reached
    any joint and the profile never left rest: 15076 seconds. Integrating
    dt = ds/sqrt(u) with a trapezoid rule over the singular endpoints gave
    5026 seconds. Both looked like plausible code and neither was.
    """
    trajectory = parameterise(geometry)
    assert 0.1 < trajectory.duration_s < 10.0


# --------------------------------------------------------------------------
# It should actually be faster, and for the stated reason
# --------------------------------------------------------------------------


def test_it_beats_uniform_scaling(geometry):
    uniform, _ = time_optimal_scale(geometry)
    optimal = parameterise(geometry)
    assert optimal.duration_s < uniform.duration_s


def test_it_keeps_the_arm_near_its_limit_for_most_of_the_path(geometry):
    """The mechanism, not just the outcome.

    Uniform scaling is set by the single hardest instant, so the rest of the
    path crawls. Solving the profile keeps the arm near its limit throughout,
    which is *why* it is faster.
    """
    uniform, _ = time_optimal_scale(geometry)
    optimal = parameterise(geometry)

    assert np.median(_utilisation(optimal)) > np.median(_utilisation(uniform))
    assert np.mean(_utilisation(optimal) > 0.8) > np.mean(_utilisation(uniform) > 0.8)


def test_the_geometry_is_unchanged(geometry):
    """Only the timing is solved for; the path itself is the planner's."""
    optimal = parameterise(geometry)
    checker = CollisionChecker()
    for setpoint in optimal.sample(200):
        assert checker.is_valid(setpoint.q)


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1.5, 2.0])
def test_rescaling_slows_it_uniformly(geometry, factor):
    trajectory = parameterise(geometry)
    slower = trajectory.rescaled(factor)

    np.testing.assert_allclose(slower.duration_s, trajectory.duration_s * factor, rtol=1e-12)
    middle = trajectory.duration_s / 2.0
    np.testing.assert_allclose(
        trajectory.at(middle).dq, slower.at(middle * factor).dq * factor, rtol=1e-6
    )


def test_rescaling_rejects_a_non_positive_factor(geometry):
    with pytest.raises(ValueError, match="must be positive"):
        parameterise(geometry).rescaled(0.0)


def test_more_nodes_changes_the_answer_only_slightly(geometry):
    """A coarse discretisation should not be a different problem."""
    coarse = parameterise(geometry, nodes=200)
    fine = parameterise(geometry, nodes=800)
    np.testing.assert_allclose(coarse.duration_s, fine.duration_s, rtol=0.1)


def test_sampling_past_the_end_holds_the_final_pose(geometry):
    trajectory = parameterise(geometry)
    beyond = trajectory.at(trajectory.duration_s * 2.0)
    np.testing.assert_allclose(beyond.dq, np.zeros(3))
    np.testing.assert_allclose(beyond.ddq, np.zeros(3))
