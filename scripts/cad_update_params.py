#!/usr/bin/env python3
"""Replace estimated link inertials in config/arm.toml with CAD-derived ones.

This closes the loop the ``provenance`` field was put there for. Until now every
link carried ``provenance = "estimate"`` and numbers I guessed; after this they
carry geometry-derived values and ``provenance = "cad"``.

Requires the CAD shell:

    nix develop .#cad --command python scripts/cad_update_params.py --write

Editing the TOML textually rather than round-tripping it through a writer is
deliberate: the comments in that file explain *why* numbers are what they are,
and a TOML writer would silently discard all of them.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))

from arm.params import DEFAULT_CONFIG, default_params  # noqa: E402
from links import link_properties  # noqa: E402


def _clean(value: float, tolerance: float = 1e-12) -> float:
    """Snap floating-point dust to zero.

    A symmetric part's centre of mass comes back as 4.85e-19 rather than 0.
    Writing that into a configuration file implies a precision that does not
    exist and makes diffs noisy on every regeneration.
    """
    return 0.0 if abs(value) < tolerance else value


def _format_block(name: str, properties, existing: str) -> str:
    com = ", ".join(f"{_clean(v):.6g}" for v in properties.com_m)
    inertia = ", ".join(f"{_clean(v):.6g}" for v in properties.diagonal_inertia_kgm2)

    # Replace only the value, never the trailing comment: the comments in this
    # file carry the reasoning and losing them would defeat the point of hand
    # editing the TOML instead of round-tripping it through a writer.
    def replace(pattern: str, replacement: str, text: str) -> str:
        return re.sub(pattern, lambda _: replacement, text, count=1)

    block = existing
    block = replace(
        r"mass_kg\s*=\s*[-\d.eE+]+",
        f"mass_kg      = {properties.mass_kg:.6g}",
        block,
    )
    block = replace(r"com_m\s*=\s*\[[^\]]*\]", f"com_m        = [{com}]", block)
    block = replace(r"inertia_kgm2\s*=\s*\[[^\]]*\]", f"inertia_kgm2 = [{inertia}]", block)
    block = replace(r'provenance\s*=\s*"[^"]*"', 'provenance   = "cad"', block)
    return block


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="modify config/arm.toml")
    args = parser.parse_args()

    params = default_params()
    text = DEFAULT_CONFIG.read_text()

    print(f"{'link':<11} {'mass':>18} {'|com| shift':>13} {'max inertia change':>20}")
    print("-" * 66)

    for link in params.links:
        properties = link_properties(link.name, params)

        pattern = re.compile(
            r"\[\[links\]\]\nname\s*=\s*\""
            + re.escape(link.name)
            + r"\".*?provenance\s*=\s*\"[^\"]*\"",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match is None:
            raise SystemExit(f"could not locate the [[links]] block for {link.name!r}")

        old_inertia = np.array(link.inertia_kgm2)
        new_inertia = properties.diagonal_inertia_kgm2
        shift = float(np.linalg.norm(properties.com_m - np.array(link.com_m)))
        change = float(np.max(np.abs(new_inertia - old_inertia) / old_inertia))

        print(
            f"{link.name:<11} {link.mass_kg * 1000:>6.1f} -> {properties.mass_kg * 1000:>6.1f} g "
            f"{shift * 1000:>10.1f} mm {change * 100:>17.0f}%"
        )

        text = (
            text[: match.start()]
            + _format_block(link.name, properties, match.group(0))
            + text[match.end() :]
        )

    if args.write:
        DEFAULT_CONFIG.write_text(text)
        print(f"\nwrote {DEFAULT_CONFIG}")
        print("run `just models` to regenerate the URDF and MJCF")
    else:
        print("\ndry run; pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
