"""MuJoCo-backed implementation of :class:`arm.hardware.ArmBackend`."""

from __future__ import annotations

import mujoco
import numpy as np

from arm.hardware import ArmState
from arm.models import MJCF_PATH
from arm.params import ArmParams, default_params


class MujocoArm:
    """The arm in simulation, commanded exactly as the real one will be.

    Physics runs at ``sim.timestep_s``; :meth:`step` advances one *control*
    period, which is slower. That ratio is not incidental -- a Dynamixel bus
    sustains a few hundred Hz while the physics needs finer steps to stay
    stable, and pretending the controller runs at physics rate would make
    simulation results optimistic in a way hardware would not forgive.
    """

    def __init__(
        self,
        params: ArmParams | None = None,
        *,
        mjcf_path=None,
        from_params: bool = False,
    ) -> None:
        self.params = params or default_params()

        # By default the committed MJCF is loaded, so what runs is the artefact
        # under version control and covered by the up-to-date test. Passing
        # from_params compiles the given parameters directly instead, which is
        # how experiments explore variants without touching the baseline.
        if from_params:
            from arm.models import MESH_DIR, build_mjcf

            # Compiled from a string, so there is no file for a relative
            # meshdir to be relative to; give MuJoCo an absolute one.
            self.model = mujoco.MjModel.from_xml_string(
                build_mjcf(self.params, meshdir=str(MESH_DIR))
            )
        else:
            self.model = mujoco.MjModel.from_xml_path(str(mjcf_path or MJCF_PATH))
        self.data = mujoco.MjData(self.model)

        self._site_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self._torque_limit = np.array(
            [
                self.params.actuators[joint.actuator].continuous_torque_nm
                for joint in self.params.joints
            ]
        )

        # params validates that this ratio is a whole number, so the realised
        # control rate is exactly the configured one.
        physics_dt = self.model.opt.timestep
        self._substeps = max(1, round(1.0 / (self.params.sim.control_hz * physics_dt)))
        self._commanded = np.zeros(3)
        self._refused = np.zeros(3)

    # -- ArmBackend ----------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._substeps * self.model.opt.timestep

    def read(self) -> ArmState:
        return ArmState(
            t=float(self.data.time),
            q=self.data.qpos.copy(),
            dq=self.data.qvel.copy(),
            tau=self._commanded.copy(),
        )

    def write_torque(self, tau: np.ndarray) -> np.ndarray:
        """Command torque, clamped to what the chosen actuators can hold.

        Clamping here rather than in the controller is deliberate: a controller
        that silently exceeds the hardware is a controller that works only in
        simulation. The clamped value is returned so the caller can compute the
        refused torque and unwind its integrator accordingly.
        """
        requested = np.asarray(tau, dtype=float)
        self._commanded = np.clip(requested, -self._torque_limit, self._torque_limit)
        self._refused = requested - self._commanded
        self.data.ctrl[:] = self._commanded
        return self._commanded.copy()

    @property
    def refused_torque(self) -> np.ndarray:
        """``commanded - applied`` from the last write; nonzero only if saturated."""
        return self._refused.copy()

    @property
    def is_saturated(self) -> bool:
        return bool(np.any(np.abs(self._refused) > 1e-12))

    def step(self) -> None:
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.asarray(q, dtype=float)
        self.data.qvel[:] = np.zeros(3) if dq is None else np.asarray(dq, dtype=float)
        self._commanded = np.zeros(3)
        self._refused = np.zeros(3)
        mujoco.mj_forward(self.model, self.data)

    # -- extras --------------------------------------------------------------

    @property
    def ee_position(self) -> np.ndarray:
        """Tip position according to MuJoCo, independent of arm.kinematics."""
        return self.data.site_xpos[self._site_ee].copy()

    @property
    def torque_limit(self) -> np.ndarray:
        return self._torque_limit.copy()

    def launch_viewer(self):
        """Open MuJoCo's own interactive viewer on this simulation.

        Complementary to Rerun rather than a replacement. Rerun shows what the
        *controller* believes -- frames, targets, error traces, on a scrubbable
        timeline. This shows what the *physics* is doing: contact points,
        constraint forces, the floor, actuator forces, and the solver's own
        diagnostics. The floor-penetration bug would have been obvious here in
        seconds.

        Returns a passive viewer handle; call ``sync()`` after each step and use
        it as a context manager.
        """
        import mujoco.viewer

        return mujoco.viewer.launch_passive(self.model, self.data)

    def bias_torque(self) -> np.ndarray:
        """MuJoCo's own gravity + Coriolis term, for cross-checking Pinocchio."""
        return self.data.qfrc_bias.copy()
