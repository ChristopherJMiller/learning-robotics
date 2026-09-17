"""Shared fixtures and reachable-workspace sampling strategies."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from hypothesis import strategies as st

from arm.params import default_params


@pytest.fixture(scope="session")
def params():
    return default_params()


def joint_angles() -> st.SearchStrategy[np.ndarray]:
    """Random joint configurations inside the configured limits."""
    limits = default_params().joint_limits_rad

    return st.tuples(
        *(
            st.floats(
                min_value=float(low),
                max_value=float(high),
                allow_nan=False,
                allow_infinity=False,
            )
            for low, high in limits
        )
    ).map(lambda triple: np.array(triple, dtype=float))


def nonsingular_joint_angles(margin: float = 0.15) -> st.SearchStrategy[np.ndarray]:
    """Joint configurations comfortably away from both singularities.

    Excludes a band around ``sin(theta3) == 0`` (elbow straight or folded) and
    configurations whose tip is near the base axis, so that tests about
    well-conditioned behaviour are not fed degenerate inputs.
    """
    from arm.kinematics import fk

    return joint_angles().filter(
        lambda q: abs(np.sin(q[2])) > margin and np.hypot(*fk(q)[:2]) > margin * 0.5
    )


@lru_cache(maxsize=1)
def _collision_checker():
    from arm.cspace import CollisionChecker

    return CollisionChecker()


def collision_free_joint_angles() -> st.SearchStrategy:
    """Configurations the simulator considers free of *all* contact.

    A large part of the joint-limit box puts the arm through something: the
    limits constrain each joint independently, while collision is a constraint
    on the whole chain against the whole world. Any comparison against a model
    with no collision geometry -- Pinocchio, for instance -- must exclude these,
    because MuJoCo will be applying contact forces the other engine cannot know
    about. That exclusion is not a workaround; it is the difference between the
    articulated dynamics and the constrained dynamics.

    Asks MuJoCo rather than reimplementing the geometry. Two versions of this
    filter were wrong before, both in the same way -- each approximated the
    collision test and each produced a test that passed or failed depending on
    which examples Hypothesis happened to draw:

    * the first compared joint frame origins against the floor, ignoring that
      the links have a 15 mm radius around those frames;
    * the second corrected for the radius but still only knew about the floor,
      so adding an obstacle to the world broke it immediately.

    There is no third approximation worth writing. The simulator already answers
    this question exactly, and it is the same answer the dynamics comparison
    depends on.
    """
    checker = _collision_checker()
    return joint_angles().filter(lambda q: not checker.in_collision(q))
