# 0003 — Real derivations for the CAD chain, not an FHS environment

**Status:** accepted · 2026-09-15

## Context

An audit of nixpkgs showed the simulation and maths stack is fully packaged,
and only CAD is missing:

| Package | nixpkgs |
|---|---|
| `mujoco`, `rerun-sdk`, `pinocchio`, `pytest`, `hypothesis` | present |
| `uv`, `just`, `kicad`, `freecad`, `opencascade-occt` | present |
| **`build123d`, `cadquery`, `OCP`** | **absent** |

`build123d` itself is a pure-Python wheel, but it pulls `cadquery-ocp-novtk` —
a 67 MB compiled OpenCASCADE binding — plus several smaller packages absent
from nixpkgs.

The usual shortcut is an FHS environment or `nix-ld`, letting PyPI wheels find
a conventional filesystem. That was explicitly rejected: it trades
reproducibility for convenience and leaves an impure layer permanently in the
build.

## Decision

Package every missing dependency as a normal derivation. Binary wheels are
handled with `autoPatchelfHook`, which rewrites the ELF interpreter and RPATHs
against nixpkgs libraries, producing an ordinary hermetic store path.

Seven derivations under `nix/pkgs/`:

| Package | Notes |
|---|---|
| `cadquery-ocp-proxy` | tiny shim the rest of the chain depends on |
| `cadquery-ocp-novtk` | 67 MB binary wheel, the hard one |
| `ocpsvg` | **pinned < 0.7** — newer requires OCP 8.x |
| `ocp-gordon` | **pinned < 0.3** — same reason |
| `trianglesolver`, `lib3mf` | small |
| `webcolors` 24.8.0 | scoped override; build123d pins `~=24.8.0`, nixpkgs has 25.x with a changed API |
| `build123d` | pure-Python wheel |

Building OCP from source was not attempted: it requires the pywrap/clang
binding generator and hours of OpenCASCADE compilation for no benefit over the
upstream wheel.

## Consequences

**It works.** `auto-patchelf: 0 dependencies could not be satisfied`, `import
OCP` succeeds, and a box-minus-hole solid computes a volume of 8.8995e-5 m³
against an analytically exact 8.8995e-5 — so the geometry kernel genuinely
functions, not merely imports.

**Impurity is zero.** No FHS layer, no `nix-ld`, no network at build time.
`nix flake check` runs the full suite hermetically.

**Layered by weight.** CAD lives in its own `cad` devShell. The everyday shell
never pays for it — which is affordable precisely because **CAD is the thing
you need last**: kinematics, IK, the Jacobian, trajectories and simulation all
predate any need for geometry.

**FreeCAD is separate again.** It is viewer-only and its closure includes TeX
Live, so it has its own `viewer` shell rather than burdening `cad`.

**Cost.** Seven derivations to maintain, two with version pins that must be
revisited when build123d's constraints move. Each should be deleted the moment
nixpkgs carries an equivalent.

Related: [adr/0005](0005-rerun-viewer-from-wheel.md) applies the same technique
to the Rerun viewer.
