"""Dual-sink logging: a Rerun stream for a human, a table for assertions.

Every run emits the same data twice, because the two consumers need different
things:

* ``runs/<name>.rrd`` -- opened in the Rerun viewer, scrubbed on a timeline,
  inspected in 3D. This is how *you* understand what the arm did.
* ``runs/<name>.parquet`` -- a flat numeric table. This is how *tests* (and an
  agent, which cannot see a GUI) assert that what the arm did was correct.

A note on why both exist. Rerun documents a dataframe query API that can read
an ``.rrd`` back into Pandas, which would make the second sink unnecessary. It
is not reachable as ``rerun.dataframe`` in 0.37.2 -- that module is absent from
the official upstream wheel, not merely from the nixpkgs build, so the API has
moved since the version those docs describe. Writing the table ourselves costs
a few lines, removes the dependency on an API that has already relocated once,
and keeps the assertions stable across SDK upgrades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr

from arm.kinematics import fk_frames
from arm.params import REPO_ROOT, ArmParams, default_params

RUNS_DIR = REPO_ROOT / "runs"
TIMELINE = "sim_time"

_JOINT_PATHS = ["world/base", "world/base/upper_arm", "world/base/upper_arm/forearm"]


class Telemetry:
    """Collects a run, then writes both sinks on close."""

    def __init__(
        self,
        name: str,
        params: ArmParams | None = None,
        *,
        directory: Path | None = None,
        spawn: bool = False,
    ) -> None:
        self.name = name
        self.params = params or default_params()
        self.directory = directory or RUNS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict[str, float]] = []
        self._spawn = spawn

        rr.init(f"learn_robotics_{name}", spawn=spawn)
        if not spawn:
            rr.save(str(self.rrd_path))

    @property
    def rrd_path(self) -> Path:
        return self.directory / f"{self.name}.rrd"

    @property
    def table_path(self) -> Path:
        return self.directory / f"{self.name}.parquet"

    # -- time ---------------------------------------------------------------

    def at(self, seconds: float) -> None:
        rr.set_time(TIMELINE, duration=float(seconds))

    # -- 3D ------------------------------------------------------------------

    def log_arm(self, q: np.ndarray) -> None:
        """Log the kinematic chain as a transform hierarchy plus a skeleton.

        Logging each joint as a ``Transform3D`` on a nested entity path is the
        visualisation equivalent of tf2: Rerun composes them down the tree, so
        child geometry inherits its parent's pose automatically.
        """
        frames = fk_frames(q, self.params)

        for path, transform in zip(_JOINT_PATHS, frames[1:], strict=True):
            rr.log(
                path,
                rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )

        points = np.array([frame[:3, 3] for frame in frames])
        rr.log("world/skeleton", rr.LineStrips3D([points], radii=0.006))
        rr.log("world/ee", rr.Points3D([points[-1]], radii=0.012))

    def log_scalars(self, **values: float) -> None:
        for key, value in values.items():
            rr.log(f"metrics/{key}", rr.Scalars(float(value)))

    # -- table ---------------------------------------------------------------

    def row(self, **values: Any) -> None:
        """Record one numeric row. Arrays are flattened into indexed columns."""
        flat: dict[str, float] = {}
        for key, value in values.items():
            array = np.atleast_1d(np.asarray(value, dtype=float))
            if array.size == 1:
                flat[key] = float(array[0])
            else:
                for index, element in enumerate(array):
                    flat[f"{key}{index}"] = float(element)
        self._rows.append(flat)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._rows:
            columns = list(self._rows[0])
            table = pa.table({name: [row[name] for row in self._rows] for name in columns})
            pq.write_table(table, self.table_path)
        # rerun 0.37 exposes flush on the stream, not at module level.
        stream = rr.get_global_data_recording()
        if stream is not None:
            stream.flush()

    def __enter__(self) -> Telemetry:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_table(path: Path) -> dict[str, np.ndarray]:
    """Read a run's table back as plain numpy columns, for tests."""
    table = pq.read_table(path)
    return {name: table[name].to_numpy() for name in table.column_names}
