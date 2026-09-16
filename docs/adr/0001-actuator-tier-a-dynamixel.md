# 0001 — Dynamixel servos over gimbal BLDC or moteus

**Status:** accepted · 2026-09-15

## Context

The actuator choice determines how much of the robotics stack is actually
reachable. Three tiers were costed.

| Tier | Build | Feedback | Torque control | ~Cost |
|---|---|---|---|---|
| A | Dynamixel XL430 / XL330 | encoder | yes (current mode) | $200–280 |
| B | 2804 gimbal BLDC + SimpleFOC | encoder | true FOC | $160–185 |
| C | moteus r4.11 | encoder | true FOC | $900–1000 |

Hobby servos were excluded outright: position-commanded black boxes with no
feedback hide the layer worth learning.

## Decision

**Tier A.**

Tier B was rejected on **torque, not cost**. A 2804 gimbal motor produces
~0.029 N·m. This arm needs roughly

```
τ = 0.45 kg × 9.81 m/s² × 0.15 m ≈ 0.66 N·m   (static, arm horizontal)
×2 for dynamics margin            ≈ 1.3 N·m
```

at the shoulder — a **~45× shortfall**. Gimbal motors are designed for
near-zero-load camera stabilisation, not for holding a limb against gravity.
Closing that gap needs a 30–50:1 reduction: belt stages, planetary or
cycloidal. **Tier B's real cost is mechanical**, and it is months of gearbox
iteration instead of kinematics and control.

Tier C is the genuine "real robotics" option and remains the upgrade path, but
at roughly 4× the cost for a first arm.

## Consequences

**Gained.** Integrated gearing, so the torque problem disappears. **No
firmware** — the servo owns its control loop and we write a serial driver,
deferring an entire discipline. Current-control mode keeps gravity compensation
and impedance control reachable. Fastest path to an arm that moves and can be
iterated on.

**Cost.** The bus sustains a few hundred Hz, not 1 kHz, and the inner servo
loop is a black box. **Tier A hides exactly one layer: the innermost FOC
loop.**

**Mitigation.** A hardware abstraction boundary is maintained from the start —
kinematics, trajectories and parameters never know what is underneath. Moving
to moteus should touch only the driver.

## Note on Rust firmware

If Tier B or C is ever revisited, the Rust FOC ecosystem is more mature than
first assumed: the `foc` and `foc-simple` crates provide Park/Clarke
transforms, fixed-point PI/PID and Embassy/RTIC compatibility, with STM32
CORDIC acceleration. They are **primitives, not a turnkey SimpleFOC
equivalent** — no autotuning, no calibration routines, no large community
troubleshooting corpus. Viable, but a project in itself.
