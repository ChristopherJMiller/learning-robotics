"""Typed, validated access to ``config/arm.toml``.

The TOML file is data; this module is the contract. Everything downstream --
kinematics, URDF/MJCF generation, CAD, the hardware driver -- consumes the
validated model here and never parses the TOML itself.

Validation runs at load time so that a unit slip or a sign error surfaces
immediately rather than three layers deep inside a simulation.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "arm.toml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Meta(_Frozen):
    name: str
    description: str


class Geometry(_Frozen):
    base_height_m: float = Field(gt=0)
    upper_arm_m: float = Field(gt=0)
    forearm_m: float = Field(gt=0)

    @property
    def max_reach_m(self) -> float:
        """Furthest radial distance from the base axis the tip can achieve."""
        return self.upper_arm_m + self.forearm_m

    @property
    def min_reach_m(self) -> float:
        """Closest radial distance; nonzero when the links differ in length."""
        return abs(self.upper_arm_m - self.forearm_m)


class Link(_Frozen):
    name: str
    mass_kg: float = Field(gt=0)
    com_m: tuple[float, float, float]
    inertia_kgm2: tuple[float, float, float]
    provenance: str

    @model_validator(mode="after")
    def _inertia_positive(self) -> Link:
        if any(i <= 0 for i in self.inertia_kgm2):
            raise ValueError(f"link {self.name!r}: inertia entries must be positive")
        ixx, iyy, izz = self.inertia_kgm2
        # Physically realisable diagonal inertia must satisfy the triangle
        # inequalities; violating them makes a simulator behave bizarrely
        # rather than error, so catch it here.
        if not (ixx + iyy >= izz and iyy + izz >= ixx and izz + ixx >= iyy):
            raise ValueError(
                f"link {self.name!r}: inertia {self.inertia_kgm2} violates the "
                "triangle inequality and is not physically realisable"
            )
        return self


class Joint(_Frozen):
    name: str
    axis: tuple[float, float, float]
    limit_lower_rad: float
    limit_upper_rad: float
    actuator: str

    @model_validator(mode="after")
    def _limits_ordered(self) -> Joint:
        if self.limit_lower_rad >= self.limit_upper_rad:
            raise ValueError(
                f"joint {self.name!r}: limit_lower_rad must be < limit_upper_rad"
            )
        if not np.isclose(np.linalg.norm(self.axis), 1.0):
            raise ValueError(f"joint {self.name!r}: axis must be a unit vector")
        return self


class Actuator(_Frozen):
    model: str
    stall_torque_nm: float = Field(gt=0)
    no_load_speed_rads: float = Field(gt=0)
    encoder_counts_rev: int = Field(gt=0)
    mass_kg: float = Field(gt=0)
    derate_factor: float = Field(gt=0, le=1.0)

    @property
    def continuous_torque_nm(self) -> float:
        """Stall torque is a datasheet peak; plan trajectories against this."""
        return self.stall_torque_nm * self.derate_factor


class Sim(_Frozen):
    gravity_ms2: tuple[float, float, float]
    timestep_s: float = Field(gt=0)
    control_hz: float = Field(gt=0)

    @model_validator(mode="after")
    def _control_slower_than_physics(self) -> Sim:
        if self.control_hz > 1.0 / self.timestep_s:
            raise ValueError(
                f"control_hz ({self.control_hz}) exceeds the physics rate "
                f"({1.0 / self.timestep_s:.0f} Hz); the controller cannot run "
                "faster than the simulator steps"
            )
        return self


class ArmParams(_Frozen):
    meta: Meta
    geometry: Geometry
    links: tuple[Link, ...]
    joints: tuple[Joint, ...]
    actuators: dict[str, Actuator]
    sim: Sim

    @model_validator(mode="after")
    def _cross_references_resolve(self) -> ArmParams:
        if len(self.joints) != 3:
            raise ValueError(f"expected 3 joints, got {len(self.joints)}")
        if len(self.links) != 3:
            raise ValueError(f"expected 3 links, got {len(self.links)}")
        for joint in self.joints:
            if joint.actuator not in self.actuators:
                raise ValueError(
                    f"joint {joint.name!r} references unknown actuator "
                    f"{joint.actuator!r}"
                )
        return self

    @property
    def link_lengths_m(self) -> tuple[float, float, float]:
        """(L0, L1, L2) -- the only three numbers the kinematics needs."""
        g = self.geometry
        return (g.base_height_m, g.upper_arm_m, g.forearm_m)

    @property
    def joint_limits_rad(self) -> np.ndarray:
        """Shape (3, 2) array of [lower, upper] per joint."""
        return np.array(
            [[j.limit_lower_rad, j.limit_upper_rad] for j in self.joints],
            dtype=float,
        )

    def actuator_for(self, joint_name: str) -> Actuator:
        for joint in self.joints:
            if joint.name == joint_name:
                return self.actuators[joint.actuator]
        raise KeyError(f"no joint named {joint_name!r}")


def load_params(path: Path | str | None = None) -> ArmParams:
    """Parse and validate the arm configuration."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return ArmParams.model_validate(raw)


@lru_cache(maxsize=1)
def default_params() -> ArmParams:
    """The repository's own arm, loaded once."""
    return load_params()
