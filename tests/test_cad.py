"""The CAD -> mass properties -> URDF chain.

These need build123d, which lives only in the `cad` devShell, so the default
suite skips them. They are run for real by `nix flake check` via the dedicated
`cad-tests` check -- otherwise "skipped" would quietly mean "never run".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("build123d")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))

from build123d import Box, Pos  # noqa: E402

from links import (  # noqa: E402
    LINKS,
    MassProperties,
    combine,
    link_properties,
    mass_properties,
)


def test_occt_inertia_is_about_the_centre_of_mass():
    """The convention the whole pipeline depends on.

    OCCT could plausibly report inertia about the global origin, which would
    need a parallel-axis shift before it could go into a URDF. It does not --
    an identical box gives an identical tensor wherever it sits. Asserted rather
    than assumed, because getting it wrong would be invisible in a diff and
    catastrophic in a simulation.
    """
    centred = mass_properties(Box(0.1, 0.02, 0.03), 1000.0)
    shifted = mass_properties(Pos(1.0, 0.0, 0.0) * Box(0.1, 0.02, 0.03), 1000.0)

    np.testing.assert_allclose(centred.inertia_kgm2, shifted.inertia_kgm2, atol=1e-15)
    np.testing.assert_allclose(shifted.com_m, [1.0, 0.0, 0.0], atol=1e-12)


def test_box_inertia_matches_the_analytic_formula():
    """A solid box is (1/12)m(b^2+c^2) about each axis. No excuse for error."""
    width, depth, height, density = 0.1, 0.02, 0.03, 1270.0
    properties = mass_properties(Box(width, depth, height), density)

    mass = width * depth * height * density
    expected = (mass / 12.0) * np.array(
        [
            depth**2 + height**2,
            width**2 + height**2,
            width**2 + depth**2,
        ]
    )
    np.testing.assert_allclose(properties.mass_kg, mass, rtol=1e-12)
    np.testing.assert_allclose(properties.diagonal_inertia_kgm2, expected, rtol=1e-9)


def test_combining_two_halves_reproduces_the_whole():
    """The parallel-axis theorem, checked against a shape it cannot argue with.

    Splitting a box and recombining the halves must give back the original
    tensor. Naively summing the two tensors would not -- which is the mistake
    this function exists to avoid.
    """
    density = 1270.0
    whole = mass_properties(Box(0.2, 0.02, 0.03), density)
    left = mass_properties(Pos(-0.05, 0, 0) * Box(0.1, 0.02, 0.03), density)
    right = mass_properties(Pos(0.05, 0, 0) * Box(0.1, 0.02, 0.03), density)

    recombined = combine([left, right])
    np.testing.assert_allclose(recombined.mass_kg, whole.mass_kg, rtol=1e-12)
    np.testing.assert_allclose(recombined.com_m, whole.com_m, atol=1e-12)
    np.testing.assert_allclose(recombined.inertia_kgm2, whole.inertia_kgm2, rtol=1e-9, atol=1e-15)

    # And the naive version really is wrong, so the test above is meaningful.
    naive = left.inertia_kgm2 + right.inertia_kgm2
    assert not np.allclose(naive, whole.inertia_kgm2, rtol=1e-3)


def test_every_link_builds_and_has_positive_mass(params):
    for name in LINKS:
        properties = link_properties(name, params)
        assert properties.mass_kg > 0.0
        assert properties.volume_m3 > 0.0
        assert np.all(properties.diagonal_inertia_kgm2 > 0.0)


def test_links_are_close_to_their_principal_axes(params):
    """Justifies using a diagonal inertia in the URDF at all."""
    for name in LINKS:
        ratio = link_properties(name, params).off_diagonal_ratio
        assert ratio < 1e-6, f"{name} is not aligned with its principal axes"


def test_inertia_satisfies_the_triangle_inequality(params):
    """The same physical realisability check arm.params enforces on load."""
    for name in LINKS:
        ixx, iyy, izz = link_properties(name, params).diagonal_inertia_kgm2
        assert ixx + iyy >= izz
        assert iyy + izz >= ixx
        assert izz + ixx >= iyy


def test_link_lengths_come_from_the_shared_configuration(params):
    """CAD and kinematics must not disagree about how long a link is."""
    _, upper, fore = params.link_lengths_m
    assert link_properties("upper_arm", params).com_m[0] < upper
    assert link_properties("forearm", params).com_m[0] < fore


# --------------------------------------------------------------------------
# The loop the provenance field exists to close
# --------------------------------------------------------------------------


def test_configuration_matches_the_cad_it_claims_to_come_from(params):
    """Every link marked provenance='cad' must actually match the geometry.

    This is what makes the provenance field mean something. Change the CAD
    without rerunning scripts/cad_update_params.py and this fails.
    """
    for link in params.links:
        if link.provenance != "cad":
            pytest.skip(f"{link.name} is still an estimate")

        computed = link_properties(link.name, params)
        np.testing.assert_allclose(link.mass_kg, computed.mass_kg, rtol=1e-4)
        np.testing.assert_allclose(link.com_m, computed.com_m, atol=1e-6)
        np.testing.assert_allclose(link.inertia_kgm2, computed.diagonal_inertia_kgm2, rtol=1e-4)


def test_no_link_is_still_an_estimate(params):
    """Fails until the CAD loop is closed for every link, then guards it."""
    estimated = [link.name for link in params.links if link.provenance != "cad"]
    assert not estimated, f"still guessed: {estimated}"


def test_mass_properties_scale_linearly_with_density():
    one = mass_properties(Box(0.1, 0.02, 0.03), 1000.0)
    two = mass_properties(Box(0.1, 0.02, 0.03), 2000.0)
    np.testing.assert_allclose(two.mass_kg, 2 * one.mass_kg, rtol=1e-12)
    np.testing.assert_allclose(two.inertia_kgm2, 2 * one.inertia_kgm2, rtol=1e-12)


def test_combine_rejects_massless_input():
    with pytest.raises(ValueError, match="no mass"):
        combine([MassProperties(0.0, 0.0, np.zeros(3), np.zeros((3, 3)))])


def test_collision_boxes_match_the_printed_section():
    """models.py duplicates the cross-section because it cannot import cad/.

    build123d is not in the default shell, so the generator carries its own
    copy of the section dimensions. This is the test that stops the two
    drifting apart.
    """
    import links as cad_links
    from arm import models

    assert models.SECTION_WIDTH_M == cad_links.SECTION_WIDTH_M
    assert models.SECTION_HEIGHT_M == cad_links.SECTION_HEIGHT_M
