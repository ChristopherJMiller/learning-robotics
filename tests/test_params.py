"""The configuration contract: validation must reject physically impossible input."""

from __future__ import annotations

import tomllib

import numpy as np
import pytest
from pydantic import ValidationError

from arm.params import DEFAULT_CONFIG, ArmParams, load_params


def test_default_config_loads(params):
    assert params.meta.name == "learn-arm-3dof"
    assert len(params.joints) == 3
    assert len(params.links) == 3


def test_link_lengths_are_positive(params):
    assert all(length > 0 for length in params.link_lengths_m)


def test_reach_bounds_are_consistent(params):
    geometry = params.geometry
    assert geometry.max_reach_m == geometry.upper_arm_m + geometry.forearm_m
    assert geometry.min_reach_m < geometry.max_reach_m


def test_every_joint_resolves_its_actuator(params):
    for joint in params.joints:
        actuator = params.actuator_for(joint.name)
        assert actuator.continuous_torque_nm < actuator.stall_torque_nm


def test_joint_limits_shape(params):
    limits = params.joint_limits_rad
    assert limits.shape == (3, 2)
    assert np.all(limits[:, 0] < limits[:, 1])


def _raw() -> dict:
    with DEFAULT_CONFIG.open("rb") as handle:
        return tomllib.load(handle)


def test_rejects_inverted_joint_limits():
    raw = _raw()
    raw["joints"][0]["limit_lower_rad"] = 1.0
    raw["joints"][0]["limit_upper_rad"] = -1.0
    with pytest.raises(ValidationError, match="limit_lower_rad"):
        ArmParams.model_validate(raw)


def test_rejects_non_unit_joint_axis():
    raw = _raw()
    raw["joints"][0]["axis"] = [0.0, 0.0, 2.0]
    with pytest.raises(ValidationError, match="unit vector"):
        ArmParams.model_validate(raw)


def test_rejects_negative_mass():
    raw = _raw()
    raw["links"][0]["mass_kg"] = -0.1
    with pytest.raises(ValidationError):
        ArmParams.model_validate(raw)


def test_rejects_physically_impossible_inertia():
    # Violates the triangle inequality: a simulator would accept this happily
    # and then behave inexplicably, so it must be caught at load time.
    raw = _raw()
    raw["links"][0]["inertia_kgm2"] = [1.0e-6, 1.0e-6, 1.0]
    with pytest.raises(ValidationError, match="triangle inequality"):
        ArmParams.model_validate(raw)


def test_rejects_controller_faster_than_physics():
    raw = _raw()
    raw["sim"]["control_hz"] = 100_000.0
    with pytest.raises(ValidationError, match="exceeds the physics rate"):
        ArmParams.model_validate(raw)


def test_rejects_unknown_actuator_reference():
    raw = _raw()
    raw["joints"][0]["actuator"] = "does_not_exist"
    with pytest.raises(ValidationError, match="unknown actuator"):
        ArmParams.model_validate(raw)


def test_params_are_immutable(params):
    with pytest.raises(ValidationError):
        params.geometry.upper_arm_m = 0.2


def test_load_params_accepts_explicit_path():
    assert load_params(DEFAULT_CONFIG).meta.name == "learn-arm-3dof"
