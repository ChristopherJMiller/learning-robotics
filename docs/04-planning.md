# Planning and trajectories

How a goal pose becomes something the arm can execute.

## Configuration space

The mental shift: stop thinking about the arm as a shape moving through the
world, and start thinking about it as a **single point** moving through a space
whose axes are joint angles.

This arm's configuration is `q = (θ₁, θ₂, θ₃)`, so C-space is a 3-dimensional
box bounded by the joint limits. Every pose is one point in it; every motion is
a curve through it. A hard question — does this whole articulated body intersect
anything? — collapses into a simple one: is this point in a blocked region?

At three joints C-space can be *drawn*, which `just cspace` does. At six it
cannot, which is why most explanations of planning stay abstract.

### The asymmetry that shapes every algorithm

| | |
|---|---|
| Compute the blocked region in closed form | **intractable** |
| Check whether one configuration is blocked | **~9 µs** |

You cannot have the map. You can only probe it. Every planning algorithm is a
strategy for probing efficiently.

A flat floor in the world becomes a curved, possibly disconnected volume in
C-space once pushed through the kinematics. For this arm **about 54% of the
joint-limit box is blocked**, almost all of it the arm swinging below the table.

### Why the world needed an obstacle

Measured before adding one: **107 of 107 randomly sampled valid configuration
pairs were connectable by a straight line.** With only a floor the free region
is nearly convex and a planner has nothing to do.

One post drops that to 65%. That is not decoration — it is what makes planning a
real problem rather than a decorative one, and it is the honest situation
anyway, since a real arm shares its table with something.

## Why a straight line is not a path

Two tempting shortcuts, both wrong:

**Straight line in Cartesian space**, running IK at each waypoint. IK may switch
branch mid-path (the arm lurches), pass through a singularity, or hit points
that are unreachable even though both ends are fine.

**Straight line in C-space.** Simple, and *sometimes* right — but with a third
of valid pairs unconnectable, usually it passes through an obstacle.

## RRT

```
tree ← {q_start}
repeat:
    q_rand ← random configuration      (5% of the time, the goal itself)
    q_near ← nearest node in the tree
    q_new  ← step from q_near toward q_rand
    if the segment q_near → q_new is clear:
        add q_new
    if q_new is close enough to the goal: done
```

"Rapidly-exploring" is an emergent property, not an instruction. Because each
sample extends the node *nearest* to it, nodes with large unexplored regions
around them — large Voronoi cells — get extended most often. The tree reaches
into empty space on its own.

`goal_bias` replaces the random sample with the goal a small fraction of the
time. Without it the tree explores beautifully and takes a while to notice the
goal; with too much of it the search degenerates into driving at a wall.

**Probabilistically complete**, not complete: if a path exists the probability of
finding it tends to one, but failure never proves impossibility.

### Shortcutting, and whether rewiring is worth it

Raw RRT paths are jagged and far from optimal. Shortcutting repeatedly tries
replacing two waypoints with a straight segment, keeping it if clear. It is
cheap and it removes most of the randomness.

RRT\* adds a cost-to-come at every node, chooses the cheapest parent rather than
the nearest, and rewires existing neighbours through new nodes — dynamic
programming on a tree that is still growing. That buys asymptotic optimality.

**Measured, the answer for this arm is that shortcutting does most of the work
and rewiring buys 8% for 12× the collision checks.** See
[adr/0009](adr/0009-rrt-with-shortcutting.md), including the finding that what
makes rewiring worth paying for is *topology* — genuinely alternative routes —
not obstacle count.

### The weak point of every practical planner

Segments are validated by **subdivision**: sample along the line, check each
point. That samples the segment rather than proving it clear, so an obstacle
thinner than the spacing is stepped straight over. At a 0.05 rad spacing, two of
eight planned paths were genuinely in collision.

Finer sampling narrows the window without closing it. The principled fix is to
inflate the collision geometry, which is why a real robot's collision model is
fatter than the robot.

## Path is not trajectory

| | |
|---|---|
| **Path** | a geometric sequence of configurations. No time. |
| **Trajectory** | a path *plus timing*, respecting velocity, acceleration and torque |

This distinction is why torque could not be discussed until now. Torque depends
on `(q, q̇, q̈)` jointly:

```
τ = M(q)·q̈  +  C(q,q̇)·q̇  +  g(q)
```

so there is **no such thing as a torque-infeasible configuration**, only a
torque-infeasible trajectory. The same pose is fine slowly and impossible
quickly.

But note the last term: `g(q)` depends on configuration alone, and *is*
checkable per-pose. That split matters.

### Turning one into the other

**Corners.** A piecewise-linear path changes direction instantly, and a corner
at non-zero speed needs infinite acceleration. `arm.timing` fits a clamped cubic
spline through the waypoints — zero velocity at the ends, continuous
acceleration throughout. That rounds the corners, which is the point and also
the risk: `path_stays_valid` rechecks, because a smoother that cuts a corner
through an obstacle has undone the planning.

**Timing.** Assigning durations is where velocity and acceleration first exist.

**Feasibility.** If it exceeds what the actuators can hold, slow down — but not
unconditionally:

```
stretching by a factor s scales velocity by 1/s and acceleration by 1/s²
```

Inertial torque falls quadratically. **Gravity does not scale at all.** Measured,
peak torque converges to the gravity requirement rather than to zero — 4.74× the
limit unscaled, converging to 0.312×, exactly the gravity-only ratio. So
`gravity_feasible` asks that question *first*: one failure means "go slower", the
other means "this arm cannot hold this pose", and they call for different
responses.

### Going slowly only where you must

Uniform scaling is set by the single hardest instant, so the rest of the path
crawls — binding at 9%, median utilisation 0.32.

`arm.topp` solves the velocity profile instead. Reparameterising by path
position makes torque **linear in `s̈` and `ṡ²`**, so the feasible acceleration
interval at each point is solved rather than searched. Forward pass from rest,
backward pass from rest, take the minimum.

**3.5× faster on a path whose torque demand varies.** See
[adr/0010](adr/0010-time-optimal-parameterisation.md), including three bugs that
each produced plausible code and absurd output.

## The pipeline, end to end

```
goal pose
  → inverse kinematics          which joint angles reach it
  → RRT + shortcutting          a collision-free route
  → spline + TOPP               a route the actuators can execute
  → feedforward PD              torques that track it
  → the arm
```

`just execute` runs exactly that: 0.26 mm RMS tip error, arriving within 0.10 mm
of the goal, zero contacts — with quantised sensing and a Kalman filter in the
loop, and nothing hand-tuned along the way.

One honest caveat the pipeline demonstrates: a collision-free **plan** does not
guarantee collision-free **execution** if tracking error is large. An earlier
bug that broke time scaling drove the arm at saturated torque, and it collided
with the post it had carefully planned around.
