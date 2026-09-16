"""Does the controller's model describe the same robot the simulator runs?

Comparing mass matrices is not enough. The honest check is **forward dynamics**:
apply a torque, and confirm both engines predict the same acceleration. That
exercises inertia, gravity, Coriolis, armature and damping together, which is
exactly the set of things that can silently disagree.

This file exists because they *did* disagree. The MJCF carried
``armature="0.01"`` and ``damping="0.05"`` as hard-coded defaults that were
absent from the URDF, so Pinocchio modelled a robot whose elbow inertia was
19x too small. Computed-torque control was inverting the wrong dynamics, and
the symptom looked exactly like bad gain tuning.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pinocchio
import pytest
from hypothesis import given, settings

from arm.dynamics import (
    _model,
    damping_torque,
    gravity_torque,
    inverse_dynamics,
    joint_damping,
    mass_matrix,
)
from arm.sim import MujocoArm
from conftest import collision_free_joint_angles, joint_angles

SETTINGS = settings(max_examples=80, deadline=None)


@pytest.fixture(scope="module")
def arm():
    return MujocoArm()


def _mujoco_mass_matrix(arm: MujocoArm) -> np.ndarray:
    full = np.zeros((arm.model.nv, arm.model.nv))
    mujoco.mj_fullM(arm.model, arm.data, full)
    return full


# --------------------------------------------------------------------------
# The parameters themselves must reach both engines
# --------------------------------------------------------------------------


def test_armature_reaches_both_engines(arm, params):
    expected = np.array([params.actuators[j.actuator].armature_kgm2 for j in params.joints])
    np.testing.assert_allclose(arm.model.dof_armature, expected)
    np.testing.assert_allclose(_model()[0].armature, expected)


def test_damping_reaches_both_engines(arm, params):
    expected = np.array([params.actuators[j.actuator].damping_nms_rad for j in params.joints])
    np.testing.assert_allclose(arm.model.dof_damping, expected)
    np.testing.assert_allclose(joint_damping(), expected)


def test_armature_is_not_negligible(params):
    """If this ever stops holding, the controller advice in the docs changes.

    Reflected rotor inertia dominates link inertia on a geared servo, which is
    what makes M(q) nearly constant and fixed-gain PD nearly optimal here. On a
    direct-drive arm the opposite is true and computed torque becomes essential.
    """
    link_only = mass_matrix(np.zeros(3)) - np.diag(
        [params.actuators[j.actuator].armature_kgm2 for j in params.joints]
    )
    armature = np.array([params.actuators[j.actuator].armature_kgm2 for j in params.joints])
    assert np.all(armature > np.diag(link_only)), (
        "armature no longer dominates link inertia; revisit the control design"
    )


# --------------------------------------------------------------------------
# Agreement between engines
# --------------------------------------------------------------------------


@given(q=joint_angles())
@SETTINGS
def test_mass_matrices_agree(arm, q):
    arm.reset(q)
    np.testing.assert_allclose(_mujoco_mass_matrix(arm), mass_matrix(q), atol=1e-12)


@given(q=collision_free_joint_angles())
@SETTINGS
def test_gravity_agrees_with_mujoco_bias_at_rest(arm, q):
    """At zero velocity MuJoCo's bias term is purely gravity."""
    arm.reset(q)
    np.testing.assert_allclose(arm.bias_torque(), gravity_torque(q), atol=1e-12)


@given(q=collision_free_joint_angles())
@SETTINGS
def test_forward_dynamics_agree(arm, q):
    """The end-to-end check: same torque in, same acceleration out.

    MuJoCo applies joint damping as a passive force, so Pinocchio is given
    ``tau - D·q̇`` to compare like with like.

    Restricted to collision-free configurations: where the arm intersects the
    floor MuJoCo adds contact forces that Pinocchio has no geometry for, and
    the two *should* disagree there. Measured penetrating the floor, MuJoCo
    reports accelerations of order 1000 rad/s^2 while Pinocchio reports 10.
    """
    rng = np.random.default_rng(abs(hash(tuple(np.round(q, 6)))) % 2**32)
    dq = rng.uniform(-2.0, 2.0, size=3)
    tau = rng.uniform(-0.2, 0.2, size=3)

    arm.reset(q, dq)
    arm.data.ctrl[:] = tau
    mujoco.mj_forward(arm.model, arm.data)
    mujoco_acceleration = arm.data.qacc.copy()

    model, data = _model()
    pinocchio_acceleration = pinocchio.aba(model, data, q, dq, tau - damping_torque(dq))

    np.testing.assert_allclose(mujoco_acceleration, pinocchio_acceleration, atol=1e-9, rtol=1e-7)


@given(q=joint_angles())
@SETTINGS
def test_inverse_and_forward_dynamics_are_inverses(q):
    """rnea and aba must undo each other -- the simulator/controller duality."""
    rng = np.random.default_rng(abs(hash(tuple(np.round(q, 5)))) % 2**32)
    dq = rng.uniform(-1.5, 1.5, size=3)
    ddq = rng.uniform(-8.0, 8.0, size=3)

    tau = inverse_dynamics(q, dq, ddq)
    model, data = _model()
    recovered = pinocchio.aba(model, data, q, dq, tau)

    np.testing.assert_allclose(recovered, ddq, atol=1e-9)


@given(q=joint_angles())
@SETTINGS
def test_mass_matrix_is_symmetric_positive_definite(q):
    matrix = mass_matrix(q)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)
