# Repository organization

The tree is arranged by **dependency weight and rate of change**, so that fast,
pure work never waits on slow, heavy work.

```
config/arm.toml         ★ single source of truth — every physical parameter
src/arm/
  params.py               TOML → validated, frozen, typed model
  frames.py               SE(3) helpers and the T_a_b convention
  kinematics.py           FK, closed-form IK, Jacobian — hand-derived
  models.py               → URDF + MJCF generators
  telemetry.py            dual-sink logging (Rerun + parquet)
models/generated/       derived artefacts, committed, never hand-edited
scripts/                thin entry points; logic lives in src/
tests/                  property tests + cross-engine validation
runs/                   run outputs (gitignored)
cad/                    build123d models — heavy deps, needed last
nix/
  overlay.nix             packages missing from nixpkgs
  pkgs/*.nix              one derivation each
docs/adr/               decisions, with what was rejected and why
```

## The organizing principles

**One file owns the physical parameters.** `config/arm.toml` is the only place
a length, mass or limit may appear. Everything else derives from it. Without
this you get drift — numpy says L₂ = 150 mm, the URDF says 160 mm, and a
weekend disappears into "sim-to-real error" that is a typo. See
[adr/0002](adr/0002-single-source-of-truth.md).

**Separate by dependency weight.** The kinematics and simulation stack is pure
nixpkgs and its tests run in under four seconds. The CAD chain needs a 67 MB
OpenCASCADE binding plus six custom derivations. Keeping them in separate
directories *and* separate devShells means the everyday loop never pays for CAD
— which matters because CAD is the thing you need **last**.

**Generated files are committed but never edited.** `models/generated/`
contains derived artefacts. Committing them makes a parameter change show up as
a reviewable diff ("link 2 grew 30 mm") instead of invisible drift. A test
asserts that regeneration produces no diff, which catches hand edits.

**Scripts are thin.** Anything worth testing lives in `src/arm/`. `scripts/`
holds argument parsing and printing. If a script grows logic, that logic has no
tests.

**Every run emits numbers, not only pixels.** A simulation whose sole output is
a GUI cannot be checked by CI, or by an agent that cannot see a screen. Runs
write both an `.rrd` and a parquet table of the same series.

## The shells

| Shell | Contains | When |
|---|---|---|
| `default` | python, mujoco, rerun, pinocchio, pytest, ruff, just | everyday |
| `cad` | + build123d chain | designing geometry |
| `viewer` | FreeCAD | looking at geometry (own shell: its closure includes TeX Live) |
| `electronics` | KiCad | much later |

`nix flake check` runs the tests and lints hermetically, which is possible
because the default shell has no impure dependencies.

## Conventions

* **SI units everywhere**, radians internally, units in key names
  (`length_m`, `stall_torque_nm`).
* **`T_a_b`** for transforms — adjacent subscripts must cancel.
* Physical parameters are **data**; behaviour is **code**. If a number
  describes the robot, it belongs in the TOML.
