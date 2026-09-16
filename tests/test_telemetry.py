"""The run artefacts must be machine-checkable, not just human-viewable.

A simulation that only produces a GUI cannot be asserted on -- not by CI, and
not by an agent. These tests exercise the contract that every run also emits a
flat numeric table, and that the numbers in it are correct.
"""

from __future__ import annotations

import numpy as np

from arm.kinematics import fk, jacobian_det
from arm.telemetry import Telemetry, read_table


def test_run_emits_both_sinks(tmp_path, params):
    with Telemetry("probe", params, directory=tmp_path) as log:
        for step in range(5):
            q = np.array([0.1 * step, 0.3, -0.7])
            log.at(step * 0.01)
            log.log_arm(q)
            log.row(sim_time=step * 0.01, q=q, ee=fk(q, params))

    assert log.rrd_path.exists(), "no .rrd written for the human"
    assert log.table_path.exists(), "no table written for assertions"
    assert log.rrd_path.stat().st_size > 0


def test_table_round_trips_with_flattened_vectors(tmp_path, params):
    angles = [np.array([0.2 * i, 0.4, -0.9]) for i in range(8)]

    with Telemetry("round_trip", params, directory=tmp_path) as log:
        for index, q in enumerate(angles):
            log.at(index * 0.02)
            log.row(sim_time=index * 0.02, q=q, ee=fk(q, params))

    table = read_table(log.table_path)

    # Vector arguments become indexed columns, so every component is assertable.
    assert set(table) == {"sim_time", "q0", "q1", "q2", "ee0", "ee1", "ee2"}
    assert len(table["sim_time"]) == len(angles)

    for index, q in enumerate(angles):
        logged = np.array([table[f"q{axis}"][index] for axis in range(3)])
        np.testing.assert_allclose(logged, q)

        expected = fk(q, params)
        recorded = np.array([table[f"ee{axis}"][index] for axis in range(3)])
        np.testing.assert_allclose(recorded, expected, atol=1e-12)


def test_logged_metrics_match_recomputation(tmp_path, params):
    """Guards against the log and the model drifting apart."""
    with Telemetry("metrics", params, directory=tmp_path) as log:
        for step in range(10):
            q = np.array([0.0, 0.2 + 0.05 * step, -1.0])
            log.at(step * 0.01)
            log.row(q=q, det_jacobian=jacobian_det(q, params))

    table = read_table(log.table_path)
    for index in range(10):
        q = np.array([table[f"q{axis}"][index] for axis in range(3)])
        assert np.isclose(table["det_jacobian"][index], jacobian_det(q, params))


def test_log_arm_uses_relative_transforms(tmp_path, params, monkeypatch):
    """The .rrd is the sink no other test reads, so pin what it receives.

    The parquet table was correct while the recording was wrong, which is
    precisely why this needs its own assertion: capture what log_arm hands to
    Rerun and confirm the composed chain lands on fk(q).
    """
    import rerun as rr

    captured: list[tuple[str, object]] = []
    real_log = rr.log

    def spy(path, *args, **kwargs):
        captured.append((path, args[0] if args else None))
        return real_log(path, *args, **kwargs)

    monkeypatch.setattr(rr, "log", spy)

    q = np.array([0.4, 0.5, -1.1])
    with Telemetry("transforms", params, directory=tmp_path) as log:
        log.at(0.0)
        log.log_arm(q)

    from arm.kinematics import fk_frames_relative
    from arm.telemetry import _JOINT_PATHS

    logged = [path for path, _ in captured]
    for path in _JOINT_PATHS:
        assert path in logged, f"{path} was never logged"

    # What was logged must be the relative chain, not the absolute one.
    expected = fk_frames_relative(q, params)
    composed = np.eye(4)
    for parent_to_child in expected:
        composed = composed @ parent_to_child
    np.testing.assert_allclose(composed[:3, 3], fk(q, params), atol=1e-12)
