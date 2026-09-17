"""Sampling-based motion planning in configuration space.

The problem these algorithms exist for: the blocked region of C-space cannot be
computed in closed form, but a single configuration can be checked in about nine
microseconds. You cannot have the map; you can only probe it. Everything here is
a strategy for probing efficiently.

Three of them, deliberately kept separate so the comparison is measurable rather
than asserted:

* :func:`rrt` grows a tree from the start toward random samples. Fast, simple,
  probabilistically complete -- and provably *not* optimal: its solution
  converges to a suboptimal length with probability one.
* :func:`shortcut` takes any path and repeatedly tries replacing two waypoints
  with the straight segment between them. Cheap, and it removes most of the
  random zigzag.
* :func:`rrt_star` is :func:`rrt` plus two additions -- choose the cheapest
  parent rather than the nearest, then rewire existing neighbours through the
  new node. That second step is the dynamic-programming one, relaxing edges in
  structure already built, and it is what buys asymptotic optimality.

Whether rewiring is worth its cost *given* shortcutting is a real question with
a measurable answer; ``scripts/plan_compare.py`` answers it for this arm rather
than appealing to the literature.

A note on distance. Path length is measured as Euclidean distance in joint
space, which treats a radian of base rotation as equivalent to a radian of
elbow. That is a modelling choice, not a fact -- the base joint swings the whole
arm and costs far more energy. Weighting the metric by something like the mass
matrix would be more honest, and would change which path is "shortest".

A note on ``resolution_rad``, which matters more than it looks
--------------------------------------------------------------
Segments are validated by subdivision: sample along the line and check each
point. That *samples* the segment rather than proving it clear, so an obstacle
narrower than the spacing can be stepped straight over.

The default started at 0.05 rad, and measurement showed **two of eight planned
paths were genuinely in collision** when rechecked at 0.02. Not a test artefact
-- paths that would have driven the arm through the post. At 0.01 every path
survives rechecking at 0.005.

Finer sampling narrows the window without ever closing it, because no finite
spacing is a proof. The principled fix is to inflate the collision geometry by a
safety margin, so that a penetration smaller than the margin is still clear of
the real obstacle. That is what production planners do, and it is the honest
reason a real robot's collision model is fatter than the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm.cspace import CollisionChecker

__all__ = [
    "PlanResult",
    "path_length",
    "rrt",
    "rrt_star",
    "shortcut",
]


@dataclass
class PlanResult:
    """A path, plus what it cost to find it."""

    path: list[np.ndarray] | None
    iterations: int
    collision_checks: int
    nodes: int
    tree_edges: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> float:
        return path_length(self.path) if self.path else float("inf")

    def describe(self) -> str:
        if not self.found:
            return f"no path after {self.iterations} iterations ({self.nodes} nodes)"
        return (
            f"length {self.length:.3f} rad, {len(self.path)} waypoints, "
            f"{self.nodes} nodes, {self.collision_checks} checks"
        )


def path_length(path) -> float:
    """Total Euclidean distance through the waypoints, in joint space."""
    if path is None or len(path) < 2:
        return 0.0
    points = np.asarray(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


class _CountingChecker:
    """Wraps a checker to count probes, since that is the real cost of planning."""

    def __init__(self, checker: CollisionChecker) -> None:
        self.checker = checker
        self.count = 0

    def is_valid(self, q) -> bool:
        self.count += 1
        return self.checker.is_valid(q)

    def segment_is_valid(self, start, end, resolution_rad: float) -> bool:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end - start))
        steps = max(2, int(np.ceil(distance / resolution_rad)) + 1)
        for fraction in np.linspace(0.0, 1.0, steps):
            if not self.is_valid(start + (end - start) * fraction):
                return False
        return True


def _steer(origin: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
    """Move from ``origin`` toward ``target``, at most ``step`` radians."""
    delta = target - origin
    distance = float(np.linalg.norm(delta))
    if distance <= step:
        return target.copy()
    return origin + delta * (step / distance)


def _extract(nodes: list[np.ndarray], parents: list[int], index: int):
    path = []
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    return path[::-1]


def rrt(
    start: np.ndarray,
    goal: np.ndarray,
    checker: CollisionChecker,
    *,
    step_rad: float = 0.25,
    goal_bias: float = 0.05,
    max_iterations: int = 20000,
    resolution_rad: float = 0.01,
    seed: int = 0,
    keep_tree: bool = False,
) -> PlanResult:
    """Rapidly-exploring Random Tree.

    The loop is four lines: sample a configuration, find the nearest node in the
    tree, step from it toward the sample, and keep the step if the segment is
    clear.

    The name describes an emergent property rather than an instruction. Because
    each sample extends the node *nearest* to it, nodes with large unexplored
    regions around them -- large Voronoi cells -- get extended most often. The
    tree reaches into empty space on its own, without being told where the empty
    space is.

    ``goal_bias`` replaces the random sample with the goal a small fraction of
    the time. Without it the tree explores beautifully and takes a long while to
    notice the goal; with too much of it the search degenerates into repeatedly
    driving at a wall.

    Probabilistically complete: if a path exists the probability of finding it
    tends to one. It can never prove that none exists, so failure here means
    "not found within ``max_iterations``", not "impossible".
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    counting = _CountingChecker(checker)
    rng = np.random.default_rng(seed)

    if not counting.is_valid(start):
        raise ValueError("start configuration is not valid")
    if not counting.is_valid(goal):
        raise ValueError("goal configuration is not valid")

    nodes = [start]
    parents = [-1]

    for iteration in range(1, max_iterations + 1):
        target = goal if rng.random() < goal_bias else checker.sample(rng)

        distances = np.linalg.norm(np.asarray(nodes) - target, axis=1)
        nearest = int(np.argmin(distances))
        candidate = _steer(nodes[nearest], target, step_rad)

        if not counting.segment_is_valid(nodes[nearest], candidate, resolution_rad):
            continue

        nodes.append(candidate)
        parents.append(nearest)

        if np.linalg.norm(candidate - goal) <= step_rad and counting.segment_is_valid(
            candidate, goal, resolution_rad
        ):
            nodes.append(goal)
            parents.append(len(nodes) - 2)
            return PlanResult(
                path=_extract(nodes, parents, len(nodes) - 1),
                iterations=iteration,
                collision_checks=counting.count,
                nodes=len(nodes),
                tree_edges=_edges(nodes, parents) if keep_tree else [],
            )

    return PlanResult(
        path=None,
        iterations=max_iterations,
        collision_checks=counting.count,
        nodes=len(nodes),
        tree_edges=_edges(nodes, parents) if keep_tree else [],
    )


def rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    checker: CollisionChecker,
    *,
    step_rad: float = 0.25,
    goal_bias: float = 0.05,
    max_iterations: int = 20000,
    resolution_rad: float = 0.01,
    seed: int = 0,
    gamma: float = 3.0,
    keep_tree: bool = False,
) -> PlanResult:
    """RRT with a cost-to-come at every node, a best-parent choice, and rewiring.

    Two additions to :func:`rrt`, both operating on the set of nodes within a
    shrinking radius of the new one:

    1. **Choose the best parent.** Attach the new node to whichever neighbour
       gives the lowest total cost, not merely the nearest.
    2. **Rewire.** Then check whether any neighbour would be cheaper routed
       *through* the new node, and re-parent it if so.

    Step 2 propagates improvements backward through structure that already
    exists -- the same relaxation a shortest-path algorithm performs, applied to
    a tree that is still growing.

    The radius shrinks as ``gamma * (log(n)/n)^(1/d)``, which is the rate that
    keeps the expected number of neighbours roughly constant while still
    guaranteeing asymptotic optimality. It is also why RRT* costs meaningfully
    more per iteration: each sample triggers a neighbourhood search and several
    extra segment checks.

    Unlike :func:`rrt` this does not stop at the first connection. It keeps
    sampling to ``max_iterations``, improving the solution as it goes, which is
    what asymptotic optimality means in practice.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    counting = _CountingChecker(checker)
    rng = np.random.default_rng(seed)
    dimension = len(start)

    if not counting.is_valid(start):
        raise ValueError("start configuration is not valid")
    if not counting.is_valid(goal):
        raise ValueError("goal configuration is not valid")

    nodes = [start]
    parents = [-1]
    costs = [0.0]
    goal_index: int | None = None

    for _iteration in range(1, max_iterations + 1):
        target = goal if rng.random() < goal_bias else checker.sample(rng)
        array = np.asarray(nodes)

        nearest = int(np.argmin(np.linalg.norm(array - target, axis=1)))
        candidate = _steer(nodes[nearest], target, step_rad)

        if not counting.segment_is_valid(nodes[nearest], candidate, resolution_rad):
            continue

        count = len(nodes)
        radius = min(
            step_rad * 3.0,
            gamma * (np.log(count + 1) / (count + 1)) ** (1.0 / dimension),
        )
        near = np.flatnonzero(np.linalg.norm(array - candidate, axis=1) <= radius)

        # 1. cheapest parent among the neighbourhood
        best_parent = nearest
        best_cost = costs[nearest] + float(np.linalg.norm(candidate - nodes[nearest]))
        for index in near:
            index = int(index)
            through = costs[index] + float(np.linalg.norm(candidate - nodes[index]))
            if through < best_cost and counting.segment_is_valid(
                nodes[index], candidate, resolution_rad
            ):
                best_parent, best_cost = index, through

        nodes.append(candidate)
        parents.append(best_parent)
        costs.append(best_cost)
        new_index = len(nodes) - 1

        # 2. rewire neighbours that would be cheaper through the new node
        for index in near:
            index = int(index)
            through = best_cost + float(np.linalg.norm(nodes[index] - candidate))
            if through < costs[index] - 1e-12 and counting.segment_is_valid(
                candidate, nodes[index], resolution_rad
            ):
                parents[index] = new_index
                _propagate(nodes, parents, costs, index, costs[index] - through)

        if (
            goal_index is None
            and np.linalg.norm(candidate - goal) <= step_rad
            and counting.segment_is_valid(candidate, goal, resolution_rad)
        ):
            nodes.append(goal)
            parents.append(new_index)
            costs.append(best_cost + float(np.linalg.norm(goal - candidate)))
            goal_index = len(nodes) - 1

    return PlanResult(
        path=None if goal_index is None else _extract(nodes, parents, goal_index),
        iterations=max_iterations,
        collision_checks=counting.count,
        nodes=len(nodes),
        tree_edges=_edges(nodes, parents) if keep_tree else [],
    )


def _propagate(nodes, parents, costs, index: int, saving: float) -> None:
    """Push a cost reduction down to every descendant of a rewired node."""
    costs[index] -= saving
    stack = [index]
    while stack:
        current = stack.pop()
        for child, parent in enumerate(parents):
            if parent == current:
                costs[child] -= saving
                stack.append(child)


def _edges(nodes, parents):
    return [(nodes[parent], nodes[child]) for child, parent in enumerate(parents) if parent >= 0]


def shortcut(
    path,
    checker: CollisionChecker,
    *,
    iterations: int = 200,
    resolution_rad: float = 0.01,
    seed: int = 0,
    checks: list[int] | None = None,
) -> list[np.ndarray]:
    """Repeatedly try to replace a span of a path with a straight segment.

    Cheap, and it removes most of what makes a raw RRT path look random. The
    limit is structural rather than a matter of iterations: shortcutting can only
    improve the route it was given. If the planner happened to go the long way
    around an obstacle, this will smooth the long way beautifully and never
    discover the short one.

    That is exactly the gap rewiring is supposed to close, and the reason both
    are worth measuring rather than choosing by argument.

    Pass a one-element list as ``checks`` to have the collision checks spent here
    added to it, so a comparison can charge shortcutting for its own cost rather
    than making it look free.
    """
    if path is None or len(path) < 3:
        return list(path) if path else []

    rng = np.random.default_rng(seed)
    counting = _CountingChecker(checker)
    current = [np.asarray(q, dtype=float) for q in path]

    for _ in range(iterations):
        if len(current) < 3:
            break
        first, second = sorted(rng.choice(len(current), size=2, replace=False))
        if second - first < 2:
            continue
        if counting.segment_is_valid(current[first], current[second], resolution_rad):
            current = current[: first + 1] + current[second:]

    if checks is not None:
        checks[0] += counting.count
    return current
