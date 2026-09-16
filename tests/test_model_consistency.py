"""Cross-checks between the hand-derived maths and two independent engines.

The point of this file is that nothing here shares code with
``arm.kinematics``. Pinocchio and MuJoCo both read ``models/generated/``,
which is produced from ``config/arm.toml`` -- the only thing the two paths
have in common. Agreement is therefore evidence, not tautology.

A failure means one of two things, and they are worth distinguishing:

1. the hand-derived algebra in ``arm.kinematics`` is wrong, or
2. ``arm.models`` does not faithfully encode the same robot.

To tell them apart, check a configuration by hand: ``q = 0`` must place the
tip at ``(L1 + L2, 0, L0)``. If ``fk`` gets that right and the engines do not,
the generator is at fault -- most likely a joint axis sign.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from arm.kinematics import fk, jacobian
from arm.models import MJCF_PATH, URDF_PATH, build_mjcf, build_urdf
from conftest import joint_angles

SETTINGS = settings(max_examples=150, deadline=None)


# --------------------------------------------------------------------------
# Generated artefacts are committed, so they must never drift from the source
# --------------------------------------------------------------------------


def test_generated_urdf_is_up_to_date():
    """Catches both a stale commit and a hand edit of a generated file."""
    assert URDF_PATH.read_text() == build_urdf(), (
        "models/generated/arm.urdf is stale or was edited by hand; run `just models`"
    )


def test_generated_mjcf_is_up_to_date():
    assert MJCF_PATH.read_text() == build_mjcf(), (
        "models/generated/arm.xml is stale or was edited by hand; run `just models`"
    )


def test_generation_is_deterministic():
    """Byte-stable output, or every regeneration produces diff noise."""
    assert build_urdf() == build_urdf()
    assert build_mjcf() == build_mjcf()


def test_urdf_masses_match_the_configuration(params):
    urdf = build_urdf()
    for link in params.links:
        assert f'value="{link.mass_kg:.9g}"' in urdf


# --------------------------------------------------------------------------
# Pinocchio
# --------------------------------------------------------------------------

pinocchio = pytest.importorskip("pinocchio")


@pytest.fixture(scope="session")
def pin():
    model = pinocchio.buildModelFromUrdf(str(URDF_PATH))
    return model, model.createData(), model.getFrameId("ee")


def test_pinocchio_model_has_three_actuated_joints(pin):
    model, _, _ = pin
    assert model.nq == 3, f"expected 3 configuration variables, got {model.nq}"


def test_pinocchio_agrees_at_the_zero_configuration(pin, params):
    """The hand-checkable case, which disambiguates a generator bug."""
    model, data, ee = pin
    length_base, length_upper, length_fore = params.link_lengths_m

    pinocchio.framesForwardKinematics(model, data, np.zeros(3))
    expected = np.array([length_upper + length_fore, 0.0, length_base])

    np.testing.assert_allclose(data.oMf[ee].translation, expected, atol=1e-12)
    np.testing.assert_allclose(fk(np.zeros(3), params), expected, atol=1e-12)


@given(q=joint_angles())
@SETTINGS
def test_fk_agrees_with_pinocchio(pin, q):
    """The headline cross-check: two independent paths, one answer."""
    model, data, ee = pin
    pinocchio.framesForwardKinematics(model, data, q)
    np.testing.assert_allclose(data.oMf[ee].translation, fk(q), atol=1e-9)


@given(q=joint_angles())
@SETTINGS
def test_jacobian_agrees_with_pinocchio(pin, q):
    """LOCAL_WORLD_ALIGNED gives the Jacobian in world axes at the frame origin,
    which is the convention our position Jacobian uses."""
    model, data, ee = pin
    full = pinocchio.computeFrameJacobian(
        model, data, q, ee, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    np.testing.assert_allclose(full[:3, :], jacobian(q), atol=1e-9)


# --------------------------------------------------------------------------
# MuJoCo -- a third, entirely separate implementation
# --------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="session")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return model, mujoco.MjData(model), site


def test_mujoco_model_loads_with_three_joints(mj):
    model, _, _ = mj
    assert model.nq == 3
    assert model.nu == 3, "expected one actuator per joint"


@given(q=joint_angles())
@SETTINGS
def test_fk_agrees_with_mujoco(mj, q):
    model, data, site = mj
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    np.testing.assert_allclose(data.site_xpos[site], fk(q), atol=1e-9)


# --------------------------------------------------------------------------
# Collision geometry
# --------------------------------------------------------------------------


def test_no_spurious_contacts_in_the_nominal_workspace(mj):
    """The arm must not be permanently touching something.

    It was. The base collision capsule ran from z=0 upward, and a capsule's
    hemispherical cap extends a full radius past its endpoint, so it sat 15 mm
    inside the floor -- a contact constraint solved on every step of every
    simulation. Replacing capsules with boxes sized to the printed section
    fixed the geometry; excluding the bolted-down pedestal from floor contact
    fixed the remaining coincident-surface case.
    """
    model, data, _ = mj
    centre = np.array([0.0, 0.9, -0.7])
    amplitude = np.array([0.6, 0.30, 0.35])

    for t in np.linspace(0.0, 1.0, 120):
        phase = 2.0 * np.pi * t
        data.qpos[:] = centre + amplitude * np.sin(phase)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0, (
            f"unexpected contact at q={np.round(data.qpos, 3)}: {data.ncon} contact(s)"
        )


def test_collision_geoms_are_boxes_not_capsules(mj):
    """Capsules overhang their endpoints by a radius; boxes do not."""
    model, _, _ = mj
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name and name.endswith("_collision"):
            assert model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX, name


def test_visual_geometry_does_not_collide(mj):
    """Meshes are for looking at; collision stays on the simple proxies."""
    model, _, _ = mj
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name and name.endswith("_visual"):
            assert model.geom_contype[index] == 0, name
            assert model.geom_conaffinity[index] == 0, name
