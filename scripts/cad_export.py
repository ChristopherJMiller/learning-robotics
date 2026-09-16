#!/usr/bin/env python3
"""Export the links as STEP and STL.

STEP is the exchange format that keeps real geometry -- it round-trips into
KiCad for fit checking, or into a mesher. STL is triangles only, which is what
a slicer and a physics engine want.

    nix develop .#cad --command python scripts/cad_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))

from build123d import export_step, export_stl  # noqa: E402

from arm.params import REPO_ROOT, default_params  # noqa: E402
from links import LINKS, link_properties  # noqa: E402

OUTPUT = REPO_ROOT / "cad" / "export"


def main() -> int:
    params = default_params()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, builder in LINKS.items():
        part = builder(params)
        export_step(part, str(OUTPUT / f"{name}.step"))
        export_stl(part, str(OUTPUT / f"{name}.stl"))
        properties = link_properties(name, params)
        print(f"{name:<11} {properties.mass_kg * 1000:6.1f} g  -> {name}.step, {name}.stl")

    print(f"\nwrote to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
