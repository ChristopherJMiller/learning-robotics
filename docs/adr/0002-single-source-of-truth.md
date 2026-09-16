# 0002 — One TOML file owns every physical parameter

**Status:** accepted · 2026-09-15

## Context

The same numbers — link lengths, masses, joint limits, motor specs — are needed
by the kinematics, the URDF, the MJCF, the CAD model and eventually the
hardware driver. Any duplication drifts.

The failure mode is specific and expensive: the kinematics believes L₂ is
150 mm, the URDF says 160 mm, and the resulting mismatch presents as
"sim-to-real error" rather than as a typo. Days disappear.

## Decision

`config/arm.toml` is the sole location for physical parameters. Nothing else
may hard-code one. It is parsed into a **frozen, validated pydantic model**
(`arm.params`) at load, and every consumer reads the model rather than the file.

TOML over the alternatives: comments are essential (`# datasheet 1.5 N·m,
derated 60%` must live beside the number), `tomllib` is stdlib from Python
3.11, `serde` support is first class in Rust, and YAML's footguns are real
(`no` → `False`, sexagesimal literals). JSON has no comments and is
disqualified.

**Units appear in key names** — `length_m`, `mass_kg`, `limit_lower_rad`,
`stall_torque_nm`. Free, and it permanently kills the mm/m factor-of-1000 bug.

## Consequences

**The file is not the contract; the typed model is.** Validation at load catches
what would otherwise surface much later:

* joint limits ordered, joint axes unit-norm
* masses positive
* inertia tensors satisfying the triangle inequality — an unphysical one makes
  a simulator behave bizarrely rather than raise
* control rate not exceeding the physics rate

**Derived artefacts are generated, committed, and diff-checked.** URDF and MJCF
come from the model. Committing them makes a parameter change reviewable; a
test asserts regeneration is a no-op, catching hand edits.

**Provenance is tracked.** Every link carries `provenance = "estimate"`. When
the CAD model exists, build123d computes real mass, centre of mass and inertia
via OpenCASCADE, those flip to `"cad"`, and a test will assert the URDF matches
the geometry. This field is the seam between guessed and measured.

**Cost.** Adding a parameter means touching the TOML, the pydantic model and
sometimes the generators. That friction is the point — it is what keeps them
consistent.
