"""The boundary between control and whatever is actually moving.

Everything above this line -- kinematics, trajectories, controllers -- must work
identically against a simulator and against real servos. That is the promise
made in ``docs/adr/0001``: swapping Dynamixel for moteus, or simulation for
hardware, should touch only an implementation of :class:`ArmBackend`.

The interface is deliberately torque-first. Position-commanded servos hide the
dynamics; commanding torque is what makes gravity compensation and impedance
control expressible at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class ArmState:
    """One synchronous snapshot of the arm.

    All three arrays are sampled at the same instant. Mixing samples from
    different instants is a timestamp bug, which is the other half of the
    "most robotics bugs are frame or timestamp bugs" pairing -- hence the
    explicit ``t``.
    """

    t: float
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray

    def __post_init__(self) -> None:
        for name in ("q", "dq", "tau"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must be a 3-vector, got {value.shape}")
            object.__setattr__(self, name, value)


@runtime_checkable
class ArmBackend(Protocol):
    """A thing that can be commanded and read.

    Implemented by :class:`arm.sim.MujocoArm` today, and by a Dynamixel driver
    when the hardware exists.
    """

    @property
    def dt(self) -> float:
        """Seconds advanced by one :meth:`step`."""
        ...

    def read(self) -> ArmState: ...

    def write_torque(self, tau: np.ndarray) -> np.ndarray:
        """Command joint torques; return what was actually applied.

        Implementations clamp to the actuator limits, and **return the clamped
        value** so the caller can see the difference. That difference is not a
        detail: an integral term that cannot tell "the arm is slow to respond"
        from "the actuator has nothing left to give" will wind up every time it
        saturates. Returning the applied torque is what makes back-calculation
        anti-windup possible at all.
        """
        ...

    def step(self) -> None:
        """Advance one control period."""
        ...

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None: ...
