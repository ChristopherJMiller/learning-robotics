"""Named start/goal pairs, so the demos and tests agree on what they mean.

These are problems, not robot parameters, which is why they live here rather
than in ``config/arm.toml``. Each was found by search against the real collision
checker and is valid only for the world the committed configuration describes --
move the post and they must be re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["REACH_ACROSS", "TUCK_AND_EXTEND", "Scenario"]


@dataclass(frozen=True)
class Scenario:
    """Two valid configurations whose straight-line connection is not."""

    name: str
    start: np.ndarray
    goal: np.ndarray
    description: str


REACH_ACROSS = Scenario(
    name="reach_across",
    start=np.array([1.1345, -0.2556, 1.6598]),
    goal=np.array([2.0071, -0.2556, 1.6598]),
    description=(
        "Reaching to either side of the post at bearing 65 and 115 degrees, at "
        "the same height and radius. Symmetric and short, so it isolates the "
        "planner: about half the straight line between them is blocked."
    ),
)

TUCK_AND_EXTEND = Scenario(
    name="tuck_and_extend",
    start=np.array([0.7854, 0.0436, 1.8235]),
    goal=np.array([2.0989, 0.3867, -0.7734]),
    description=(
        "Tucked in close and high at (0.075, 0.075, 0.20), finishing extended "
        "far out and low at (-0.14, 0.24, 0.05). The arm has to retract, swing "
        "past the post, and reach out again, which makes the motion legible in "
        "a viewer and gives the time parameterisation something to work with -- "
        "the torque demand varies a great deal along it."
    ),
)
