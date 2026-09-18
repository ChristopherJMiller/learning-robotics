#!/usr/bin/env python3
"""Render the configured fiducial markers to PNG textures.

Committed like the CAD meshes: the generated MJCF references them at load time,
so the model is only self-contained with them present.

    just markers
"""

from arm.fiducial import write_marker_texture
from arm.params import default_params

if __name__ == "__main__":
    for marker in default_params().markers:
        path = write_marker_texture(marker)
        print(
            f"{marker.name:<16} id {marker.aruco_id:<3} {marker.dictionary:<12} "
            f"{marker.size_m * 1000:.0f} mm black square -> {path}"
        )
