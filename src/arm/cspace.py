"""Configuration space: the arm as a single point in a space of joint angles.

The mental shift that makes planning tractable. Stop thinking about an
articulated body moving through the world, and start thinking about one point
moving through a space whose axes are joint angles. Every pose of the arm is a
point in that space; every motion is a curve through it.

For this arm C-space is three-dimensional -- one axis per joint, bounded by the
joint limits -- which is unusually lucky, because it means it can be *drawn*.
At six degrees of freedom it cannot, which is why most explanations of planning
stay abstract.

The asymmetry that shapes every planning algorithm
--------------------------------------------------
Computing the blocked region in closed form is intractable: a flat floor in the
world becomes a curved, possibly disconnected volume in C-space once pushed
through the kinematics. But checking whether a *single* configuration is blocked
costs about 9 microseconds.

You cannot have the map. You can only probe it. Planning algorithms are
strategies for probing efficiently.

The grid built here is therefore a teaching aid rather than a planner: at 3 DOF
a 60^3 grid is 216,000 checks and takes a couple of seconds, while at 6 DOF the
same resolution would be 4.7e10 cells. Sampling-based planners never build it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from arm.params import ArmParams, default_params
from arm.sim import MujocoArm

__all__ = [
    "CollisionChecker",
    "CSpaceGrid",
    "occupancy_grid",
]


class CollisionChecker:
    """Answers "would the arm intersect anything in this pose?".

    Deliberately narrow. It does not step physics, apply torques or advance
    time -- it places the arm and runs collision detection. Nothing moves, and
    on real hardware nothing would move either: planning happens entirely inside
    a geometric model before the arm is commanded anywhere.

    That also means the model *is* the planner's whole world. Geometry that is
    wrong here produces plans that confidently crash, which is a large part of
    why the link meshes were worth deriving from CAD.
    """

    def __init__(self, params: ArmParams | None = None) -> None:
        self.params = params or default_params()
        arm = MujocoArm(self.params)
        self.model = arm.model
        self.data = arm.data
        self.limits = self.params.joint_limits_rad

    def in_collision(self, q: np.ndarray) -> bool:
        self.data.qpos[:] = np.asarray(q, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        return bool(self.data.ncon > 0)

    def within_limits(self, q: np.ndarray) -> bool:
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))

    def is_valid(self, q: np.ndarray) -> bool:
        """Inside the joint limits and not intersecting anything."""
        return self.within_limits(q) and not self.in_collision(q)

    def segment_is_valid(
        self, start: np.ndarray, end: np.ndarray, resolution_rad: float = 0.02
    ) -> bool:
        """Whether a straight line between two configurations stays valid.

        Checked by subdivision, which is the honest weak point of every
        practical planner: it samples the segment rather than proving it clear,
        so an obstacle thinner than ``resolution_rad`` can be stepped over
        entirely. Choosing that number is trading planning time against the
        chance of missing something thin.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        steps = max(2, int(np.ceil(distance / resolution_rad)) + 1)

        return all(
            self.is_valid(start + (end - start) * fraction)
            for fraction in np.linspace(0.0, 1.0, steps)
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """A uniformly random configuration inside the joint limits.

        Note it may well be in collision -- for this arm around half of them
        are, because most of the joint-limit box points the arm through the
        floor.
        """
        return rng.uniform(self.limits[:, 0], self.limits[:, 1])

    def sample_valid(self, rng: np.random.Generator, attempts: int = 1000) -> np.ndarray:
        for _ in range(attempts):
            q = self.sample(rng)
            if not self.in_collision(q):
                return q
        raise RuntimeError(f"no collision-free sample in {attempts} attempts")


@dataclass(frozen=True)
class CSpaceGrid:
    """A sampled occupancy map of configuration space."""

    occupied: np.ndarray  # bool, shape (n, n, n)
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]

    @property
    def blocked_fraction(self) -> float:
        return float(self.occupied.mean())

    def blocked_points(self) -> np.ndarray:
        """Coordinates of the blocked cells, as an (m, 3) array of angles."""
        indices = np.argwhere(self.occupied)
        return np.stack([self.axes[axis][indices[:, axis]] for axis in range(3)], axis=1)

    def free_points(self) -> np.ndarray:
        indices = np.argwhere(~self.occupied)
        return np.stack([self.axes[axis][indices[:, axis]] for axis in range(3)], axis=1)


def occupancy_grid(resolution: int = 48, params: ArmParams | None = None) -> CSpaceGrid:
    """Evaluate collision on a regular grid over the joint limits.

    ``resolution**3`` collision checks. Tractable here and nowhere else: the
    cost is ``resolution ** degrees_of_freedom``, so the same 48 steps per axis
    would be 1.2e10 checks on a six-jointed arm.
    """
    checker = CollisionChecker(params)
    limits = checker.limits

    axes = tuple(np.linspace(limits[axis, 0], limits[axis, 1], resolution) for axis in range(3))

    occupied = np.zeros((resolution,) * 3, dtype=bool)
    configuration = np.empty(3)
    for i, first in enumerate(axes[0]):
        configuration[0] = first
        for j, second in enumerate(axes[1]):
            configuration[1] = second
            for k, third in enumerate(axes[2]):
                configuration[2] = third
                occupied[i, j, k] = checker.in_collision(configuration)

    return CSpaceGrid(occupied=occupied, axes=axes)
