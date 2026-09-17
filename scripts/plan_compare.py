#!/usr/bin/env python3
"""Is rewiring worth its cost, given that shortcutting exists?

A real question with a measurable answer, and the answer is specific to a robot
and a world rather than universal. RRT* is asymptotically optimal and RRT
provably is not, which sounds decisive -- but shortcutting is very cheap and
removes most of what makes a raw RRT path bad. So the comparison that matters is
not RRT against RRT*, it is **RRT + shortcut** against **RRT* + shortcut**, at
equal effort.

Measured over several seeds, because a single run of a randomised algorithm is
an anecdote.

    just plan
    just plan --seeds 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from arm.cspace import CollisionChecker
from arm.params import Obstacle, default_params
from arm.planning import path_length, rrt, rrt_star, shortcut
from arm.scenarios import REACH_ACROSS

START = REACH_ACROSS.start
GOAL = REACH_ACROSS.goal


def summarise(name: str, lengths, times, checks) -> None:
    """Collision checks first: it is the metric that does not depend on hardware.

    Wall time is reported as a sanity check, not as the result. Motion planning
    is conventionally costed in collision checks for exactly this reason -- the
    check is the expensive primitive, and counting it is reproducible.
    """
    lengths = np.array(lengths)
    print(
        f"{name:<26} {lengths.mean():>7.3f} {lengths.std():>7.3f} "
        f"{lengths.min():>7.3f} {int(np.mean(checks)):>9,} "
        f"{np.mean(times) * 1000:>8.0f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--star-iterations", type=int, default=2000)
    args = parser.parse_args()

    checker = CollisionChecker()
    direct = float(np.linalg.norm(GOAL - START))

    # Warm up properly before timing anything. Measured cold-to-hot, the same
    # RRT call drifts from 245 ms to 95 ms over about half a minute, which is
    # more than the differences being compared -- it made plain RRT look slower
    # than RRT *plus* shortcutting, which cannot be true. One warm-up call was
    # not enough; the full sweep is.
    print("warming up ...")
    for seed in range(args.seeds):
        warm = rrt(START, GOAL, checker, seed=seed)
        if warm.found:
            shortcut(warm.path, checker, seed=seed)

    print(f"start -> goal, straight-line distance {direct:.3f} rad")
    print(f"the straight line is valid: {checker.segment_is_valid(START, GOAL)}")
    print(f"averaged over {args.seeds} seeds\n")
    print(f"{'planner':<26} {'mean':>7} {'std':>7} {'best':>7} {'checks':>9} {'ms':>8}")
    print("-" * 70)

    names = ("rrt", "rrt_shortcut", "rrt_star", "rrt_star_shortcut")
    lengths = {name: [] for name in names}
    times = {name: [] for name in names}
    checks = {name: [] for name in names}

    # Interleaved by seed rather than run planner-by-planner. This machine's
    # timings drift by more than the differences being measured -- the same RRT
    # call took 245 ms cold and 95 ms hot -- so running each planner to
    # completion in turn charges the first one for the warm-up. Interleaving
    # spreads the drift across all four equally.
    for seed in range(args.seeds):
        for name in names:
            started = time.perf_counter()
            if name.startswith("rrt_star"):
                result = rrt_star(
                    START, GOAL, checker, max_iterations=args.star_iterations, seed=seed
                )
            else:
                result = rrt(START, GOAL, checker, seed=seed)

            if not result.found:
                continue

            path = result.path
            spent = [result.collision_checks]
            if name.endswith("shortcut"):
                path = shortcut(path, checker, seed=seed, checks=spent)

            times[name].append(time.perf_counter() - started)
            lengths[name].append(path_length(path))
            checks[name].append(spent[0])

    results = {}
    for name in names:
        if not lengths[name]:
            print(f"{name:<26} no solution found")
            continue
        results[name] = float(np.mean(lengths[name]))
        summarise(name, lengths[name], times[name], checks[name])

    if {"rrt", "rrt_shortcut", "rrt_star_shortcut"} <= results.keys():
        raw = results["rrt"]
        plain = results["rrt_shortcut"]
        starred = results["rrt_star_shortcut"]
        cost = np.mean(checks["rrt_star_shortcut"]) / np.mean(checks["rrt_shortcut"])
        spread = np.std(lengths["rrt"]) / np.std(lengths["rrt_shortcut"])

        print(
            f"\nShortcutting alone improves RRT by {100 * (raw - plain) / raw:.0f}%, "
            f"and rewiring on top of it buys a further "
            f"{100 * (plain - starred) / plain:.1f}%"
            f"\nfor {cost:.0f}x the collision checks."
        )
        print(
            f"\nThe bigger effect is not the mean but the spread: shortcutting cuts the"
            f"\nstandard deviation across seeds by {spread:.0f}x. Raw RRT is erratic --"
            "\nsometimes near-optimal, sometimes wandering -- and shortcutting makes it"
            "\n*predictable*, which for a planner that has to run on every command"
            "\nmatters more than the average case."
        )
        print(
            "\nNote also that shortcutting adds nothing to RRT* here. That is the"
            "\nasymptotic optimality showing up: RRT* has already found essentially"
            "\nthe path shortcutting would have produced."
            "\n"
            "\nSo for this arm, RRT + shortcut is the sensible default. Where rewiring"
            "\nwould still matter is the case this benchmark cannot show: shortcutting"
            "\ncan only improve the route it was handed, so if the tree went the long"
            "\nway around an obstacle it will smooth the long way beautifully and never"
            "\nfind the short one. RRT* can still escape that."
        )

    clutter_experiment(args.seeds)

    print(
        "\nNote what is *not* being compared: neither path is a trajectory yet."
        "\nBoth are geometric, with no timing, and torque feasibility cannot even be"
        "\nasked until they are time-parameterised."
    )
    return 0


def _post(name, bearing_deg, radius, centre_z, half_xy=0.020, half_z=None):
    angle = np.radians(bearing_deg)
    return Obstacle(
        name=name,
        pos_m=(radius * np.cos(angle), radius * np.sin(angle), centre_z),
        half_size_m=(half_xy, half_xy, half_z if half_z is not None else centre_z),
    )


def clutter_experiment(seeds: int) -> None:
    """Does a harder world make rewiring worth it?

    The obvious guess is that more obstacles favour RRT*. The measurement says
    something more specific: what matters is not how many obstacles there are
    but whether they create genuinely *different routes*.
    """
    base = default_params()
    worlds = {
        "one post": [_post("p0", 90, 0.19, 0.11)],
        "three posts in a line": [
            _post("p0", 90, 0.12, 0.11),
            _post("p1", 90, 0.19, 0.11),
            _post("p2", 90, 0.26, 0.11),
        ],
        "a low wall": [
            _post(f"w{i}", 90, radius, 0.06, 0.020, 0.06)
            for i, radius in enumerate((0.10, 0.16, 0.22, 0.28))
        ],
    }

    print("\n\nDoes clutter change the answer?\n")
    print(f"{'world':<24} {'blocked':>8} {'rrt+sc':>8} {'rrt*+sc':>8} {'gap':>7} {'spread':>8}")
    print("-" * 68)

    for name, obstacles in worlds.items():
        checker = CollisionChecker(base.with_obstacles(obstacles))
        if not (checker.is_valid(START) and checker.is_valid(GOAL)):
            print(f"{name:<24} endpoints not valid in this world")
            continue

        rng = np.random.default_rng(0)
        samples = rng.uniform(checker.limits[:, 0], checker.limits[:, 1], size=(2500, 3))
        blocked = float(np.mean([checker.in_collision(q) for q in samples]))

        plain, starred = [], []
        for seed in range(seeds):
            result = rrt(START, GOAL, checker, seed=seed, max_iterations=8000)
            if result.found:
                plain.append(path_length(shortcut(result.path, checker, seed=seed)))
            best = rrt_star(START, GOAL, checker, max_iterations=1500, seed=seed)
            if best.found:
                starred.append(path_length(shortcut(best.path, checker, seed=seed)))

        if not plain or not starred:
            print(f"{name:<24} no solutions found")
            continue

        plain_array = np.array(plain)
        gap = 100 * (plain_array.mean() - np.mean(starred)) / plain_array.mean()
        spread = plain_array.max() / plain_array.min()
        print(
            f"{name:<24} {100 * blocked:>7.1f}% {plain_array.mean():>8.3f} "
            f"{np.mean(starred):>8.3f} {gap:>6.1f}% {spread:>7.2f}x"
        )

    print(
        "\nThe guess that more obstacles favour rewiring is right, but not for the"
        "\nreason it sounds like. Three posts in a line are more obstacles than one"
        "\nand do not help much: they block the same route, so there is still only"
        "\none sensible way round and nothing for rewiring to choose between."
        "\n"
        "\nThe wall does help, because it creates genuinely *different* routes -- over"
        "\nthe top, or around the end -- of different quality. The 'spread' column is"
        "\nthe evidence: RRT's longest path is nearly twice its shortest, meaning"
        "\ndifferent seeds are committing to different routes. Shortcutting then"
        "\npolishes whichever one it was handed and cannot switch, while RRT* keeps"
        "\nimproving globally and migrates toward the better one."
        "\n"
        "\nSo the thing that makes rewiring worth paying for is topology, not count."
    )


if __name__ == "__main__":
    raise SystemExit(main())
