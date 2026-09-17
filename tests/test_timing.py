"""Turning paths into trajectories.

The properties here are mostly about *scaling*, because that is where the
subtlety lives: stretching a trajectory divides inertial torque by the square of
the stretch and leaves gravity entirely alone, so "slow down until it fits" is
sound for one kind of infeasibility and useless for the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from arm.cspace import CollisionChecker
from arm.planning import rrt, shortcut
from arm.timing import (
    gravity_feasible,
    path_stays_valid,
    peak_torque,
    time_optimal_scale,
    torque_limits,
    trajectory_from_path,
)

START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


@pytest.fixture(scope="module")
def checker():
    return CollisionChecker()


@pytest.fixture(scope="module")
def path(checker):
    result = rrt(START, GOAL, checker, seed=0)
    assert result.found
    return shortcut(result.path, checker, seed=0)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_trajectory_passes_through_its_waypoints(path):
    trajectory = trajectory_from_path(path)
    for time, waypoint in zip(trajectory.times_s, path, strict=True):
        np.testing.assert_allclose(trajectory.at(time).q, waypoint, atol=1e-9)


def test_trajectory_starts_and_ends_at_rest(path):
    """Clamped ends: a controller can feed this forward without a jolt."""
    trajectory = trajectory_from_path(path)
    np.testing.assert_allclose(trajectory.at(0.0).dq, np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(trajectory.at(trajectory.duration_s).dq, np.zeros(3), atol=1e-9)


def test_sampling_past_the_end_holds_the_final_pose(path):
    trajectory = trajectory_from_path(path)
    beyond = trajectory.at(trajectory.duration_s * 3.0)
    np.testing.assert_allclose(beyond.q, path[-1], atol=1e-9)
    np.testing.assert_allclose(beyond.dq, np.zeros(3))
    np.testing.assert_allclose(beyond.ddq, np.zeros(3))


def test_a_faster_nominal_speed_gives_a_shorter_duration(path):
    slow = trajectory_from_path(path, speed_rad_s=1.0)
    fast = trajectory_from_path(path, speed_rad_s=4.0)
    assert fast.duration_s < slow.duration_s
    np.testing.assert_allclose(slow.duration_s / fast.duration_s, 4.0, rtol=1e-9)


def test_a_trajectory_needs_two_waypoints():
    with pytest.raises(ValueError, match="at least two"):
        trajectory_from_path([np.zeros(3)])


def test_smoothing_is_rechecked_for_collision(path, checker):
    """Rounding corners moves the path, which can undo the planning."""
    assert path_stays_valid(trajectory_from_path(path), checker)


# --------------------------------------------------------------------------
# Rescaling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1.5, 2.0, 3.0])
def test_rescaling_preserves_the_curve_through_space(path, factor):
    """A regression test for a real bug.

    Rescaling used to refit the spline from its own knots, which silently
    dropped the clamped end conditions and produced a *different curve*.
    Slowing a trajectory by 1% appeared to cut peak torque sixfold, which is
    impossible, and was how it was noticed.
    """
    trajectory = trajectory_from_path(path, speed_rad_s=3.0)
    scaled = trajectory.rescaled(factor)

    original = np.array([s.q for s in trajectory.sample(60)])
    stretched = np.array([s.q for s in scaled.sample(60)])
    np.testing.assert_allclose(original, stretched, atol=1e-12)


@pytest.mark.parametrize("factor", [1.5, 2.0, 3.0])
def test_rescaling_scales_velocity_by_one_over_the_factor(path, factor):
    trajectory = trajectory_from_path(path, speed_rad_s=3.0)
    scaled = trajectory.rescaled(factor)

    middle = trajectory.duration_s / 2.0
    fast = trajectory.at(middle).dq
    slow = scaled.at(middle * factor).dq
    np.testing.assert_allclose(fast, slow * factor, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("factor", [1.5, 2.0, 3.0])
def test_rescaling_scales_acceleration_by_the_square(path, factor):
    trajectory = trajectory_from_path(path, speed_rad_s=3.0)
    scaled = trajectory.rescaled(factor)

    middle = trajectory.duration_s / 2.0
    fast = trajectory.at(middle).ddq
    slow = scaled.at(middle * factor).ddq
    np.testing.assert_allclose(fast, slow * factor**2, rtol=1e-9, atol=1e-12)


def test_rescaling_rejects_a_non_positive_factor(path):
    with pytest.raises(ValueError, match="must be positive"):
        trajectory_from_path(path).rescaled(0.0)


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------


def test_slowing_down_reduces_torque_but_not_to_zero(path):
    """The gravity floor, stated as a measurement.

    Inertial torque falls as 1/s^2, so a large enough slowdown removes it
    entirely -- but what remains is the gravity load, which does not scale at
    all. This is why 'just go slower' has a limit.
    """
    trajectory = trajectory_from_path(path, speed_rad_s=5.0)
    limits = torque_limits(trajectory.params)
    _, gravity_ratio = gravity_feasible(trajectory)

    ratios = [
        float(np.max(peak_torque(trajectory.rescaled(f)) / limits))
        for f in (1.0, 2.0, 4.0, 20.0, 200.0)
    ]

    # Monotonically decreasing ...
    for faster, slower in zip(ratios, ratios[1:], strict=False):
        assert slower <= faster + 1e-9

    # ... toward the gravity requirement, never below it.
    assert all(ratio >= gravity_ratio - 1e-9 for ratio in ratios)
    np.testing.assert_allclose(ratios[-1], gravity_ratio, rtol=0.01)

    # The approach is slow near the end, and for a reason worth knowing:
    # inertial torque falls as 1/s^2 and is gone by 20x, but viscous damping
    # falls only as 1/s, so it is still 5% above the floor there.
    assert ratios[-2] - gravity_ratio > ratios[-1] - gravity_ratio


def test_time_optimal_scale_produces_a_feasible_trajectory(path):
    trajectory = trajectory_from_path(path, speed_rad_s=8.0)
    assert np.any(peak_torque(trajectory) > torque_limits())

    scaled, factor = time_optimal_scale(trajectory)
    assert factor > 1.0
    assert np.all(peak_torque(scaled) <= torque_limits() + 1e-9)


def test_time_optimal_scale_is_not_needlessly_slow(path):
    """It should sit at the limit, not far below it."""
    trajectory = trajectory_from_path(path, speed_rad_s=8.0)
    scaled, _ = time_optimal_scale(trajectory)
    assert np.max(peak_torque(scaled) / torque_limits()) > 0.9


def test_an_already_feasible_trajectory_is_left_alone(path):
    trajectory = trajectory_from_path(path, speed_rad_s=0.5)
    scaled, factor = time_optimal_scale(trajectory)
    assert factor == 1.0
    assert scaled.duration_s == trajectory.duration_s


@pytest.mark.parametrize("speed", [4.0, 8.0, 16.0])
def test_scaling_converges_to_the_same_traversal(path, speed):
    """Whatever nominal speed is asked for, the result is the fastest feasible one.

    A pleasant consequence of scaling rather than re-timing: ``speed_rad_s`` is
    only a starting guess, and the search removes it.
    """
    scaled, _ = time_optimal_scale(trajectory_from_path(path, speed_rad_s=speed))
    reference, _ = time_optimal_scale(trajectory_from_path(path, speed_rad_s=8.0))
    np.testing.assert_allclose(scaled.duration_s, reference.duration_s, rtol=0.02)


def test_gravity_infeasibility_is_reported_rather_than_searched_around(params):
    """The failure that slowing down cannot fix, distinguished from the one it can.

    Weakening the actuators far enough makes holding the path impossible at any
    speed. A scaling loop that did not check this would search to its limit and
    return something that still does not work.
    """
    feeble = params.model_copy(
        update={
            "actuators": {
                name: actuator.model_copy(update={"stall_torque_nm": 0.02})
                for name, actuator in params.actuators.items()
            }
        }
    )
    trajectory = trajectory_from_path([START, GOAL], feeble, speed_rad_s=0.1)
    holdable, ratio = gravity_feasible(trajectory)
    assert not holdable
    assert ratio > 1.0

    with pytest.raises(ValueError, match="gravity alone"):
        time_optimal_scale(trajectory)
