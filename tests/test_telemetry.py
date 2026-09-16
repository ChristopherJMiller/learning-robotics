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
