"""Shared fixtures and reachable-workspace sampling strategies."""

from __future__ import annotations

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


def collision_free_joint_angles(clearance_m: float = 0.005) -> st.SearchStrategy:
    """Configurations that are inside the joint limits *and* above the floor.

    A surprisingly large part of the joint-limit box puts the arm through the
    ground: the limits constrain each joint independently, but the floor is a
    constraint on the whole chain. Any comparison against a model with no
    collision geometry -- Pinocchio, for instance -- must exclude these, because
    MuJoCo will be applying contact forces the other engine cannot know about.

    The exclusion is not a workaround. It is the difference between the
    articulated dynamics and the constrained dynamics.
    """
    from arm.kinematics import fk_frames

    return joint_angles().filter(
        lambda q: min(frame[2, 3] for frame in fk_frames(q)[1:]) > clearance_m
    )
