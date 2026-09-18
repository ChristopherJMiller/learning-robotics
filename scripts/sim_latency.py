#!/usr/bin/env python3
"""A camera always tells you about the past. What that costs, measured.

An image has to be exposed, read off the sensor, carried over USB, and decoded
before ``solvePnP`` can say anything. Tens of milliseconds is ordinary. So by
the time the answer exists, it describes where the arm *was*.

Three ways to handle that, differing only in timestamp handling:

    encoder_only   ignore the camera -- the baseline to beat
    naive          fuse each fix as though it described now
    rewind         rewind to when the shutter opened, correct, replay since

The arm here has a compliant link, because otherwise none of this matters: on a
rigid arm the encoder puts the marker within 0.149 mm and the camera's 1.5 mm
is simply worse. A camera does not beat an encoder at measuring a joint -- it
beats one at measuring a *link*, past the gearbox and the flex of a printed
part.

    just latency
    just latency --speed 4
"""

from __future__ import annotations

import argparse

import numpy as np

from arm.camera import opencv_from_mujoco, pose_in_world
from arm.fiducial import marker_pose_world
from arm.fusion import STRATEGIES, FusedEstimator, LinkSideError, observe_marker
from arm.kinematics import fk, point_jacobian
from arm.params import default_params


def link_angles(t: float, speed: float) -> np.ndarray:
    """Where the links actually are: the physical truth the camera sees."""
    return np.array(
        [
            0.9 + 0.5 * np.sin(speed * t),
            0.45 + 0.3 * np.sin(0.7 * speed * t),
            -1.1 + 0.3 * np.cos(speed * t),
        ]
    )


def run(strategy, speed, latency, params, flex, *, camera_hz, seconds, seed=0):
    """Returns (tip RMS error in metres, mean marker speed in m/s)."""
    rng = np.random.default_rng(seed)
    pose_world_camera = opencv_from_mujoco(pose_in_world(params.cameras[0]))
    dt = 1.0 / params.sim.control_hz

    estimator = FusedEstimator(params, strategy=strategy, encoder_std_rad=flex.position_std_rad())
    pending, errors, speeds = [], [], []
    next_frame = 0.0

    for step in range(int(seconds / dt)):
        t = step * dt
        link = link_angles(t, speed)  # where the arm really is
        reading = flex.encoder_reading(link, rng, params)

        estimator.step(t, reading, dt, None)

        if t >= next_frame:
            pending.append(
                (t + latency, observe_marker(link, t, pose_world_camera, params, rng=rng))
            )
            next_frame += 1.0 / camera_hz
        while pending and pending[0][0] <= t:
            estimator.observe(pending.pop(0)[1])

        if t > 1.0:  # skip the filter's warm-up
            errors.append(np.linalg.norm(fk(estimator.q, params) - fk(link, params)))
            velocity = (link_angles(t + 1e-5, speed) - link_angles(t - 1e-5, speed)) / 2e-5
            marker = marker_pose_world(params.markers[0], link, params)[:3, 3]
            speeds.append(np.linalg.norm(point_jacobian(link, marker, params) @ velocity))

    return float(np.sqrt(np.mean(np.square(errors)))), float(np.mean(speeds))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed", type=float, default=2.0, help="motion rate scale")
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--play-deg", type=float, default=1.0, help="zero-mean link-side error")
    args = parser.parse_args()

    params = default_params()
    flex = LinkSideError(play_rad=np.radians(args.play_deg))

    print(
        f"link-side play {args.play_deg:.2f} deg (zero-mean), camera {args.camera_hz:.0f} Hz, "
        f"control {params.sim.control_hz:.0f} Hz"
    )
    _, marker_speed = run(
        "rewind", args.speed, 0.0, params, flex, camera_hz=args.camera_hz, seconds=args.seconds
    )
    print(f"marker moving at {marker_speed * 1000:.0f} mm/s\n")

    print(f"{'latency':>8} {'v x L':>9} " + "".join(f"{s:>13}" for s in STRATEGIES))
    print("-" * 58)

    rows = []
    for latency in [0.0, 0.02, 0.04, 0.08, 0.16, 0.24]:
        values = [
            run(
                s, args.speed, latency, params, flex, camera_hz=args.camera_hz, seconds=args.seconds
            )[0]
            for s in STRATEGIES
        ]
        rows.append((latency, values))
        print(
            f"{latency * 1000:6.0f}ms {marker_speed * latency * 1000:7.2f}mm "
            + "".join(f"{v * 1000:12.3f} " for v in values)
        )

    baseline = rows[0][1][STRATEGIES.index("encoder_only")]
    perfect = rows[0][1][STRATEGIES.index("rewind")]
    worst = rows[-1][1][STRATEGIES.index("naive")]

    print(
        f"\nWith no latency the camera is worth having: {baseline * 1000:.3f} mm ->"
        f" {perfect * 1000:.3f} mm, about {100 * (1 - perfect / baseline):.0f}%."
    )
    print(
        "\nHandled correctly, latency erodes that benefit and never goes past it --"
        "\nrewind converges to the encoder-only baseline, because an observation"
        "\nabout further in the past has less left to say once the encoders have"
        "\nalready covered the interval. It costs nothing to be late."
    )
    print(
        f"\nHandled naively it does not converge, it diverges: {worst * 1000:.2f} mm at"
        f" {rows[-1][0] * 1000:.0f} ms,"
        f"\n{worst / baseline:.1f}x worse than never wiring the camera up at all. The"
        "\ncrossover is somewhere around 20-40 ms -- which is an utterly ordinary"
        "\nUSB camera pipeline, not a pathological one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
