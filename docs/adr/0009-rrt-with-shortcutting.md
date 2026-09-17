# 0009 — RRT with shortcutting as the default, not RRT*

**Status:** accepted · 2026-09-16

## Context

RRT\* is asymptotically optimal and RRT provably is not — its solution converges
to a suboptimal length with probability one. That sounds decisive.

But shortcutting is very cheap and removes most of what makes a raw RRT path
bad, so the comparison that matters is not RRT against RRT\*. It is
**RRT + shortcut** against **RRT\* + shortcut**, at equal effort.

## Decision

`arm.planning` implements all three separately so the comparison is a
measurement rather than an argument, and `scripts/plan_compare.py` runs it.
**RRT + shortcut is the default.**

Averaged over 8 seeds on a problem whose straight line is blocked:

| planner | mean | std | best | checks | ms |
|---|---|---|---|---|---|
| rrt | 1.463 | 0.408 | 1.095 | 8,307 | 166 |
| **rrt + shortcut** | **1.080** | **0.114** | 0.927 | **11,039** | **203** |
| rrt\* | 0.991 | 0.045 | 0.927 | 129,118 | 1912 |
| rrt\* + shortcut | 0.991 | 0.045 | 0.927 | 131,171 | 1800 |

Shortcutting alone improves RRT by 26%; rewiring on top buys a further 8% for
**12× the collision checks**.

## Consequences

**The spread matters more than the mean.** Shortcutting cuts the standard
deviation across seeds by about 4×. Raw RRT is erratic — sometimes near-optimal,
sometimes wandering — and shortcutting makes it *predictable*. For a planner
that runs on every command that is worth more than a better average.

**Shortcutting adds nothing to RRT\*** (0.991 → 0.991). That is the asymptotic
optimality showing up: it has already found what shortcutting would produce, and
it is the clearest sign the rewiring does what it claims.

**Collision checks are the metric, not wall time.** This machine's timings drift
from 245 ms to 95 ms as it warms — more than the differences being measured, and
enough to make plain RRT look slower than RRT *plus* shortcutting. The benchmark
interleaves by seed and leads with checks, which are deterministic.

## When rewiring would be worth it

Not obstacle count — **topology**. Measured across three worlds:

| world | blocked | rrt+sc | rrt\*+sc | gap | spread |
|---|---|---|---|---|---|
| one post | 54.3% | 1.024 | 0.947 | 7.6% | 1.23× |
| three posts in a line | 57.6% | 3.943 | 3.684 | 6.6% | 1.36× |
| **a low wall** | 57.4% | 4.741 | 4.150 | **12.5%** | **1.93×** |

Three posts are *more obstacles* than one and help slightly **less** — they
block the same route, so there is still one sensible way round and nothing for
rewiring to choose between. The wall helps because it creates genuinely
different routes, over the top or around the end, of different quality.

The spread column is the evidence: under the wall RRT's longest path is nearly
twice its shortest, so different seeds commit to different routes. Its
*minimum* is close to RRT\*'s, meaning RRT can find the good route and simply
does not do so reliably. Shortcutting polishes whichever one it was handed and
cannot switch; RRT\* migrates toward the better one.

**So: revisit this decision when the workspace gains genuinely alternative
routes, not when it merely gains obstacles.**

## A bug worth recording

`resolution_rad` defaulted to 0.05, and segment checking subdivides rather than
proves. **Two of eight planned paths were genuinely in collision** when
rechecked at 0.02 — paths that would have driven the arm through the post. The
default is now 0.01, where every path survives rechecking at 0.005, and tests
pin both halves: that 0.05 really does miss collisions, and that 0.01 holds up.

No finite spacing is ever a proof. The principled fix is to inflate the
collision geometry by a safety margin, which is the honest reason a real robot's
collision model is fatter than the robot.
