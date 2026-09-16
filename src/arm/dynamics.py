"""Rigid-body dynamics, delegated to Pinocchio.

This is the line drawn in ``docs/adr/0004``: kinematics is derived by hand
because the derivation is short and everything downstream rests on it; dynamics
is not, because mass matrices and recursive Newton-Euler are error-prone tedium
with poor returns.

So Pinocchio graduates here from a test-only oracle to a runtime dependency. It
reads the same generated URDF the cross-checks use, which means the dynamics and
the kinematics are guaranteed to describe the same robot.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pinocchio

from arm.models import URDF_PATH
from arm.params import default_params


@lru_cache(maxsize=1)
def _model() -> tuple[pinocchio.Model, pinocchio.Data]:
    model = pinocchio.buildModelFromUrdf(str(URDF_PATH))

    # URDF has no vocabulary for reflected rotor inertia, so it must be applied
    # here from the same parameters the MJCF is generated from. Skipping this
    # is not a small error: for these geared servos the armature is roughly 3x
    # the shoulder link inertia and 19x the elbow's, so a controller that omits
    # it is inverting the dynamics of a completely different robot.
    params = default_params()
    model.armature = np.array(
        [params.actuators[joint.actuator].armature_kgm2 for joint in params.joints]
    )
    return model, model.createData()


def joint_damping() -> np.ndarray:
    """Viscous damping coefficients, per joint (N·m·s/rad)."""
    params = default_params()
    return np.array([params.actuators[joint.actuator].damping_nms_rad for joint in params.joints])


def joint_friction() -> np.ndarray:
    """Coulomb friction magnitudes, per joint (N·m)."""
    params = default_params()
    return np.array(
        [params.actuators[joint.actuator].coulomb_friction_nm for joint in params.joints]
    )


def friction_torque(dq: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Coulomb friction: a constant torque opposing motion, independent of speed.

    Non-smooth at zero velocity, which is what makes it hard. ``sign`` is
    softened over ``epsilon`` so a controller feeding this forward does not
    chatter at rest -- a real technique, not a simulation convenience, since a
    discontinuous feed-forward term will excite every resonance the arm has.

    This is the dominant unmodelled effect on real geared servos, and the reason
    an integral term is still needed after gravity has been fed forward.
    """
    dq = np.asarray(dq, dtype=float)
    return joint_friction() * np.tanh(dq / epsilon)


def damping_torque(dq: np.ndarray) -> np.ndarray:
    """Torque lost to viscous friction at joint velocity ``dq``.

    The simulator applies this as a passive resisting force. A controller that
    wants to produce a commanded acceleration has to add it back, exactly as it
    adds back gravity.
    """
    return joint_damping() * np.asarray(dq, dtype=float)


def gravity_torque(q: np.ndarray) -> np.ndarray:
    """Joint torques that exactly cancel gravity at configuration ``q``.

    In the manipulator equation ``M(q)q̈ + C(q,q̇)q̇ + g(q) = τ``, this is
    ``g(q)``. Commanding it holds the arm still against gravity without any
    position error -- which is the whole point, because a pure PD controller
    can only resist gravity by first *allowing* the error that generates the
    restoring torque.
    """
    model, data = _model()
    return np.asarray(
        pinocchio.computeGeneralizedGravity(model, data, np.asarray(q, dtype=float))
    ).copy()


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """The configuration-dependent inertia matrix ``M(q)`` (CRBA)."""
    model, data = _model()
    return np.asarray(pinocchio.crba(model, data, np.asarray(q, dtype=float))).copy()


def inverse_dynamics(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    """Torques producing acceleration ``ddq`` from state ``(q, dq)`` (RNEA).

    The basis of computed-torque control: ask for the acceleration you want and
    get the torque that produces it, gravity and Coriolis included.
    """
    model, data = _model()
    return np.asarray(
        pinocchio.rnea(
            model,
            data,
            np.asarray(q, dtype=float),
            np.asarray(dq, dtype=float),
            np.asarray(ddq, dtype=float),
        )
    ).copy()
