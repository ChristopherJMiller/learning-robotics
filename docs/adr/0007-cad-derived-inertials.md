# 0007 — Link inertials come from CAD, not from estimates

**Status:** accepted · 2026-09-16

## Context

Every link in `config/arm.toml` carried `provenance = "estimate"` and numbers I
made up. That is the third of three ways the simulation was flattering us — the
other two being perfect state measurement and zero friction, both since closed.

Guessed inertias are worse than obviously-wrong ones: a simulator fed them is
*confidently* wrong, and every controller result derived from it inherits the
error silently.

## Decision

`cad/links.py` models each link in build123d, driven by the same
`config/arm.toml` the kinematics reads, and OpenCASCADE computes mass, centre
of mass and inertia from the geometry.
`scripts/cad_update_params.py --write` writes those numbers back and flips
`provenance` to `"cad"`.

Three details that matter:

* **OCCT reports inertia about the centre of mass**, for unit density — exactly
  URDF's convention, so no parallel-axis shift is needed. Asserted in
  `test_occt_inertia_is_about_the_centre_of_mass` rather than assumed, because
  getting it wrong would be invisible in a diff.
* **Composite bodies need the parallel axis theorem.** Shell, infill and servo
  are combined with `I_total = Σ [ I_i + m_i(|d_i|²E − d_id_iᵀ) ]`. Summing
  tensors directly is wrong unless the parts share a centre of mass, and
  `test_combining_two_halves_reproduces_the_whole` checks that the naive version
  really does differ.
* **A printed part is not solid.** The CAD draws walls explicitly and leaves the
  cavity empty; `material.infill_fraction` accounts for what the slicer puts
  back.

Servos are modelled as uniform blocks of their datasheet mass. Their mass is
known exactly and their shape is not interesting — inventing geometry for them
would add error rather than remove it.

## Consequences

**The estimates were bad, and one was qualitatively wrong:**

| link | estimated | from CAD | COM shift | max inertia error |
|---|---|---|---|---|
| base | 150.0 g | 102.2 g | 8.7 mm | **74%** |
| upper_arm | 120.0 g | 96.4 g | 14.0 mm | **54%** |
| forearm | 80.0 g | 71.7 g | 4.4 mm | **45%** |

The base was worse than a magnitude error. I had guessed
`[1.0e-4, 1.0e-4, 1.5e-4]`, making `Izz` the *largest* principal moment; the
geometry gives `[7.3e-5, 6.7e-5, 3.9e-5]`, making it the *smallest*. The
estimate had the shape of the part wrong, not merely its scale.

Shoulder gravity torque dropped from 0.229 to 0.164 N·m as a result, so every
previously reported control number was computed against an arm 30% heavier than
the one we are actually going to build.

**`provenance` now means something.**
`test_configuration_matches_the_cad_it_claims_to_come_from` fails if the CAD
changes without `just cad-sync` being rerun, and
`test_no_link_is_still_an_estimate` guards against new guesses creeping back in.

**The CAD tests run in CI.** They need build123d, which is not in the default
shell, so `nix flake check` gained a `cad-tests` check that runs them in the CAD
environment. Otherwise "skipped" would quietly have meant "never run" — the same
failure `nix flake check` itself had before it was first executed.

**Still estimated:** `material.infill_fraction` and the PETG density. Both are
answerable with a kitchen scale once a real part exists, which is the cheapest
system identification available.
