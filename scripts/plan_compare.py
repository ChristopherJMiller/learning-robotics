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
from arm.planning import path_length, rrt, rrt_star, shortcut

# Reaching to either side of the post: bearing 65 and 115 degrees.
START = np.array([1.1345, -0.2556, 1.6598])
GOAL = np.array([2.0071, -0.2556, 1.6598])


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

    print(
        "\nNote what is *not* being compared: neither path is a trajectory yet."
        "\nBoth are geometric, with no timing, and torque feasibility cannot even be"
        "\nasked until they are time-parameterised."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
