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

from arm.kinematics import fk_frames, fk_frames_relative
from arm.params import REPO_ROOT, ArmParams, default_params

RUNS_DIR = REPO_ROOT / "runs"
TIMELINE = "sim_time"

_LINK_THICKNESS_M = 0.030
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
        self.log_static_geometry()

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

    def log_static_geometry(self) -> None:
        """Draw the links once, as geometry in each frame's own coordinates.

        Each entity inherits its parent's transform, so a box logged here moves
        with the arm without being re-logged every tick. Note the negative
        offsets: every frame sits at its link's *distal* joint, so the link
        extends backwards towards its parent.
        """
        length_base, length_upper, length_fore = self.params.link_lengths_m
        half = _LINK_THICKNESS_M / 2.0

        spans = [
            (_JOINT_PATHS[0], (0.0, 0.0, -length_base / 2.0), (half, half, length_base / 2.0)),
            (_JOINT_PATHS[1], (-length_upper / 2.0, 0.0, 0.0), (length_upper / 2.0, half, half)),
            (_JOINT_PATHS[2], (-length_fore / 2.0, 0.0, 0.0), (length_fore / 2.0, half, half)),
        ]
        for path, centre, half_size in spans:
            rr.log(
                f"{path}/link",
                rr.Boxes3D(centers=[centre], half_sizes=[half_size]),
                static=True,
            )

    def log_arm(self, q: np.ndarray) -> None:
        """Log the kinematic chain as a transform hierarchy plus a skeleton.

        Transforms are logged **relative to each parent**, because Rerun
        composes them down the entity tree -- the same contract tf2 uses.
        Logging absolute poses here silently multiplies them together.
        """
        relative = fk_frames_relative(q, self.params)

        for path, transform in zip(_JOINT_PATHS, relative, strict=True):
            rr.log(
                path,
                rr.Transform3D(translation=transform[:3, 3], mat3x3=transform[:3, :3]),
            )

        absolute = fk_frames(q, self.params)
        points = np.array([frame[:3, 3] for frame in absolute])
        rr.log("world/skeleton", rr.LineStrips3D([points], radii=0.004))
        rr.log("world/ee", rr.Points3D([points[-1]], radii=0.012))

    def log_target(self, target: np.ndarray) -> None:
        """Log the commanded point, for comparison against where the tip went."""
        rr.log("world/target", rr.Points3D([np.asarray(target, dtype=float)], radii=0.008))

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
