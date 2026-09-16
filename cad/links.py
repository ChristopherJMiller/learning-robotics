"""The physical links, as code.

build123d is the source of truth for geometry, and geometry is the source of
truth for mass properties. That chain is the whole point: every link in
``config/arm.toml`` currently carries ``provenance = "estimate"``, and those
numbers were guessed. A simulator fed guessed inertias is confidently wrong,
which is worse than being obviously wrong.

Dimensions are driven by ``arm.params`` so the CAD and the kinematics cannot
disagree about how long a link is. Note the distinction from
``docs/03-kinematics.md``: ``upper_arm_m`` is the distance between **joint
axes**, not the length of the printed part. The part is longer, because it has
to carry bearing bosses past each axis.

Run ``just cad-props`` to compute mass properties, and ``just cad-export`` for
STEP and STL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from build123d import Box, Cylinder, Part, Pos, Rot

from arm.params import ArmParams, default_params

# Printed cross-section. Walls thick enough to be stiff in bending without
# becoming a solid block of plastic.
SECTION_WIDTH_M = 0.030
SECTION_HEIGHT_M = 0.022
WALL_M = 0.0025

# Bearing boss at each joint axis: the part has to extend past the axis.
BOSS_RADIUS_M = 0.013
BOSS_LENGTH_M = 0.016
BORE_RADIUS_M = 0.004

# Servo bodies, treated as solid blocks of their datasheet mass rather than
# modelled in detail. Their mass is known exactly; their shape is not
# interesting, and guessing at it would add error rather than remove it.
XL430_SIZE_M = (0.0285, 0.0465, 0.034)
XL330_SIZE_M = (0.020, 0.034, 0.023)


@dataclass(frozen=True)
class MassProperties:
    """What the simulator actually needs from a shape."""

    volume_m3: float
    mass_kg: float
    com_m: np.ndarray
    inertia_kgm2: np.ndarray

    @property
    def diagonal_inertia_kgm2(self) -> np.ndarray:
        """URDF and MJCF here carry diagonal inertia only."""
        return np.diag(self.inertia_kgm2).copy()

    @property
    def off_diagonal_ratio(self) -> float:
        """How much inertia the diagonal approximation is throwing away.

        If this is not small, the link is far from being aligned with its
        principal axes and a diagonal inertia is a poor description of it.
        """
        diagonal = np.abs(np.diag(self.inertia_kgm2))
        off = np.abs(self.inertia_kgm2 - np.diag(np.diag(self.inertia_kgm2)))
        return float(off.max() / max(diagonal.max(), 1e-18))


def mass_properties(shape, density_kgm3: float) -> MassProperties:
    """Volume, centre of mass and inertia tensor, straight from OpenCASCADE.

    OCCT returns the inertia matrix **about the centre of mass** and for unit
    density, so it scales linearly with density and needs no parallel-axis
    shift. That is exactly the convention URDF expects, which is convenient
    enough to be worth a test rather than a comment
    (``tests/test_cad.py::test_occt_inertia_is_about_the_centre_of_mass``).
    """
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, properties)

    volume = float(properties.Mass())  # "Mass" of a volume property is volume
    centre = properties.CentreOfMass()
    matrix = properties.MatrixOfInertia()

    unit_density_inertia = np.array(
        [[matrix.Value(row, col) for col in (1, 2, 3)] for row in (1, 2, 3)]
    )

    return MassProperties(
        volume_m3=volume,
        mass_kg=volume * density_kgm3,
        com_m=np.array([centre.X(), centre.Y(), centre.Z()]),
        inertia_kgm2=unit_density_inertia * density_kgm3,
    )


def _beam(length_m: float) -> Part:
    """A hollow rectangular beam along +x, centred on the x axis.

    Hollow because a printed part is: the cavity is left empty here and the
    slicer's infill is accounted for separately, via
    ``material.infill_fraction``.
    """
    outer = Pos(length_m / 2, 0, 0) * Box(length_m, SECTION_WIDTH_M, SECTION_HEIGHT_M)
    cavity = Pos(length_m / 2, 0, 0) * Box(
        length_m + 0.001,
        SECTION_WIDTH_M - 2 * WALL_M,
        SECTION_HEIGHT_M - 2 * WALL_M,
    )
    return outer - cavity


def _boss(at_x: float) -> Part:
    """A bearing boss straddling a joint axis, bored through."""
    axis = Rot(90, 0, 0) * Cylinder(BOSS_RADIUS_M, BOSS_LENGTH_M)
    bore = Rot(90, 0, 0) * Cylinder(BORE_RADIUS_M, BOSS_LENGTH_M + 0.002)
    return Pos(at_x, 0, 0) * (axis - bore)


def upper_arm(params: ArmParams | None = None) -> Part:
    """Shoulder axis to elbow axis. Origin at the shoulder axis, extends +x."""
    params = params or default_params()
    _, length, _ = params.link_lengths_m
    return _beam(length) + _boss(0.0) + _boss(length)


def forearm(params: ArmParams | None = None) -> Part:
    """Elbow axis to the tool point. Origin at the elbow axis, extends +x."""
    params = params or default_params()
    _, _, length = params.link_lengths_m
    return _beam(length) + _boss(0.0)


def base(params: ArmParams | None = None) -> Part:
    """Pedestal from the mounting plate up to the shoulder axis."""
    params = params or default_params()
    height, _, _ = params.link_lengths_m

    column = Pos(0, 0, height / 2) * Box(SECTION_WIDTH_M, SECTION_WIDTH_M, height)
    cavity = Pos(0, 0, height / 2) * Box(
        SECTION_WIDTH_M - 2 * WALL_M, SECTION_WIDTH_M - 2 * WALL_M, height + 0.001
    )
    plate = Pos(0, 0, 0.002) * Box(0.070, 0.070, 0.004)
    return (column - cavity) + plate


LINKS = {"base": base, "upper_arm": upper_arm, "forearm": forearm}

# Which servo each link carries, and where its body sits in that link's frame.
# A link carries the actuator that drives the *next* joint.
SERVO_LOAD_M = {
    "base": ("xl430", XL430_SIZE_M, (0.0, 0.0, 0.030)),
    "upper_arm": ("xl330", XL330_SIZE_M, None),  # placed at the elbow axis
    "forearm": (None, None, None),
}


def combine(parts: list[MassProperties]) -> MassProperties:
    """Composite body: total mass, combined centre of mass, shifted inertia.

    Each part's inertia is given about its own centre of mass, so combining
    them needs the parallel axis theorem to move each one to the shared centre:

        I_total = sum_i [ I_i + m_i * (|d_i|^2 * E - d_i d_i^T) ]

    where ``d_i`` is the offset from part i's centre to the combined centre.
    Simply adding inertia tensors -- the tempting shortcut -- is wrong unless
    every part happens to share a centre of mass.
    """
    total_mass = sum(p.mass_kg for p in parts)
    if total_mass <= 0.0:
        raise ValueError("composite has no mass")

    centre = sum(p.mass_kg * p.com_m for p in parts) / total_mass

    inertia = np.zeros((3, 3))
    for part in parts:
        offset = part.com_m - centre
        inertia += part.inertia_kgm2 + part.mass_kg * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )

    return MassProperties(
        volume_m3=sum(p.volume_m3 for p in parts),
        mass_kg=total_mass,
        com_m=centre,
        inertia_kgm2=inertia,
    )


def _solid_envelope(name: str, params: ArmParams) -> Part:
    """The same outline without the cavity, for measuring how much infill fits."""
    length_base, length_upper, length_fore = params.link_lengths_m
    if name == "base":
        return Pos(0, 0, length_base / 2) * Box(SECTION_WIDTH_M, SECTION_WIDTH_M, length_base)
    length = length_upper if name == "upper_arm" else length_fore
    return Pos(length / 2, 0, 0) * Box(length, SECTION_WIDTH_M, SECTION_HEIGHT_M)


def _servo_load(name: str, params: ArmParams) -> MassProperties | None:
    """The actuator a link carries, as a uniform block of its datasheet mass.

    Modelled as a box rather than in detail: the mass is known exactly from the
    datasheet, the shape is not interesting, and inventing geometry for it would
    add error rather than remove it.
    """
    _, length_upper, _ = params.link_lengths_m
    placement = {
        "base": ("xl430", XL430_SIZE_M, np.array([0.0, 0.0, params.link_lengths_m[0]])),
        "upper_arm": ("xl330", XL330_SIZE_M, np.array([length_upper, 0.0, 0.0])),
        "forearm": (None, None, None),
    }[name]

    actuator_key, size, position = placement
    if actuator_key is None:
        return None

    mass = params.actuators[actuator_key].mass_kg
    width, depth, height = size
    # Uniform box inertia about its own centre.
    inertia = (mass / 12.0) * np.diag(
        [
            depth**2 + height**2,
            width**2 + height**2,
            width**2 + depth**2,
        ]
    )
    return MassProperties(
        volume_m3=width * depth * height,
        mass_kg=mass,
        com_m=position,
        inertia_kgm2=inertia,
    )


def link_properties(name: str, params: ArmParams | None = None) -> MassProperties:
    """Everything a link is made of: printed shell, infill, and its servo.

    Reported in the link's own frame, with the origin at its proximal joint
    axis -- which is the frame URDF's ``<inertial><origin>`` is expressed in.
    """
    params = params or default_params()
    density = params.material.density_kgm3

    shell = mass_properties(LINKS[name](params), density)

    # The cavity the shell encloses, filled at the slicer's infill fraction.
    envelope = mass_properties(_solid_envelope(name, params), density)
    cavity_volume = max(envelope.volume_m3 - shell.volume_m3, 0.0)
    infill = MassProperties(
        volume_m3=cavity_volume,
        mass_kg=cavity_volume * density * params.material.infill_fraction,
        com_m=envelope.com_m,
        inertia_kgm2=envelope.inertia_kgm2
        * params.material.infill_fraction
        * (cavity_volume / max(envelope.volume_m3, 1e-18)),
    )

    parts = [shell, infill]
    servo = _servo_load(name, params)
    if servo is not None:
        parts.append(servo)
    return combine(parts)
