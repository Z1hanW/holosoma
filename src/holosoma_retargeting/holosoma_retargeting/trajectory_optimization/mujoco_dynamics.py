"""MuJoCo dynamics linearization for dynamics-level TrajOpt."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .builder import LinearDynamics


@dataclass(frozen=True)
class MujocoNominalTrajectory:
    """Nominal MuJoCo states and controls around which TrajOpt is linearized."""

    qpos: np.ndarray
    qvel: np.ndarray
    controls: np.ndarray
    activations: np.ndarray | None = None

    def validate(self, model: mujoco.MjModel) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        qvel = np.asarray(self.qvel, dtype=np.float64)
        controls = np.asarray(self.controls, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError(f"qpos must have shape (T, {model.nq})")
        frame_count = qpos.shape[0]
        if frame_count < 2:
            raise ValueError("nominal trajectory must contain at least two frames")
        if qvel.shape != (frame_count, model.nv):
            raise ValueError(f"qvel must have shape {(frame_count, model.nv)}")
        if controls.shape != (frame_count - 1, model.nu):
            raise ValueError(f"controls must have shape {(frame_count - 1, model.nu)}")
        if model.na:
            if self.activations is None:
                raise ValueError(f"activations are required for model.na={model.na}")
            activations = np.asarray(self.activations, dtype=np.float64)
            if activations.shape != (frame_count, model.na):
                raise ValueError(
                    f"activations must have shape {(frame_count, model.na)}"
                )
        elif self.activations is not None:
            activations = np.asarray(self.activations, dtype=np.float64)
            if activations.shape != (frame_count, 0):
                raise ValueError("activations must be None or have shape (T, 0)")
        arrays = [qpos, qvel, controls]
        if self.activations is not None:
            arrays.append(np.asarray(self.activations))
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("nominal trajectory must contain only finite values")


@dataclass(frozen=True)
class MujocoDynamicsLinearization:
    dynamics: LinearDynamics
    defect_norms: np.ndarray


class MujocoDynamicsLinearizer:
    """Build tangent-space affine dynamics with ``mjd_transitionFD``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        epsilon: float = 1e-6,
        centered: bool = True,
    ) -> None:
        if epsilon <= 0.0 or not np.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        self.model = model
        self.epsilon = float(epsilon)
        self.centered = bool(centered)

    @property
    def state_dimension(self) -> int:
        return 2 * self.model.nv + self.model.na

    def _set_state(
        self,
        data: mujoco.MjData,
        trajectory: MujocoNominalTrajectory,
        frame: int,
    ) -> None:
        data.qpos[:] = trajectory.qpos[frame]
        data.qvel[:] = trajectory.qvel[frame]
        if self.model.na:
            data.act[:] = trajectory.activations[frame]
        if frame < len(trajectory.controls):
            data.ctrl[:] = trajectory.controls[frame]
        mujoco.mj_forward(self.model, data)

    def linearize(
        self,
        trajectory: MujocoNominalTrajectory,
    ) -> MujocoDynamicsLinearization:
        trajectory.validate(self.model)
        frame_count = len(trajectory.qpos)
        state_dimension = self.state_dimension
        transition = np.empty(
            (frame_count - 1, state_dimension, state_dimension),
            dtype=np.float64,
        )
        control = np.empty(
            (frame_count - 1, state_dimension, self.model.nu),
            dtype=np.float64,
        )
        offset = np.empty((frame_count - 1, state_dimension), dtype=np.float64)
        derivative_data = mujoco.MjData(self.model)
        rollout_data = mujoco.MjData(self.model)

        for frame in range(frame_count - 1):
            self._set_state(derivative_data, trajectory, frame)
            mujoco.mjd_transitionFD(
                self.model,
                derivative_data,
                self.epsilon,
                int(self.centered),
                transition[frame],
                control[frame],
                None,
                None,
            )

            self._set_state(rollout_data, trajectory, frame)
            mujoco.mj_step(self.model, rollout_data)
            qpos_defect = np.empty(self.model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(
                self.model,
                qpos_defect,
                1.0,
                trajectory.qpos[frame + 1],
                rollout_data.qpos,
            )
            offset[frame, : self.model.nv] = qpos_defect
            offset[frame, self.model.nv : 2 * self.model.nv] = (
                rollout_data.qvel - trajectory.qvel[frame + 1]
            )
            if self.model.na:
                offset[frame, 2 * self.model.nv :] = (
                    rollout_data.act - trajectory.activations[frame + 1]
                )

        if not (
            np.isfinite(transition).all()
            and np.isfinite(control).all()
            and np.isfinite(offset).all()
        ):
            raise RuntimeError("MuJoCo dynamics linearization produced non-finite values")
        return MujocoDynamicsLinearization(
            dynamics=LinearDynamics(
                transition=transition,
                control=control,
                offset=offset,
            ),
            defect_norms=np.linalg.norm(offset, axis=1),
        )

    def apply_deltas(
        self,
        trajectory: MujocoNominalTrajectory,
        state_deltas: np.ndarray,
        control_deltas: np.ndarray,
    ) -> MujocoNominalTrajectory:
        """Retract tangent-state and control deltas onto a new nominal trajectory."""

        trajectory.validate(self.model)
        state_deltas = np.asarray(state_deltas, dtype=np.float64)
        control_deltas = np.asarray(control_deltas, dtype=np.float64)
        expected_state_shape = (len(trajectory.qpos), self.state_dimension)
        if state_deltas.shape != expected_state_shape:
            raise ValueError(f"state_deltas must have shape {expected_state_shape}")
        if control_deltas.shape != trajectory.controls.shape:
            raise ValueError(
                f"control_deltas must have shape {trajectory.controls.shape}"
            )
        qpos = np.asarray(trajectory.qpos, dtype=np.float64).copy()
        qvel = np.asarray(trajectory.qvel, dtype=np.float64).copy()
        for frame in range(len(qpos)):
            mujoco.mj_integratePos(
                self.model,
                qpos[frame],
                state_deltas[frame, : self.model.nv],
                1.0,
            )
            qvel[frame] += state_deltas[
                frame, self.model.nv : 2 * self.model.nv
            ]
        activations = None
        if self.model.na:
            activations = np.asarray(trajectory.activations, dtype=np.float64).copy()
            activations += state_deltas[:, 2 * self.model.nv :]
        return MujocoNominalTrajectory(
            qpos=qpos,
            qvel=qvel,
            controls=np.asarray(trajectory.controls, dtype=np.float64)
            + control_deltas,
            activations=activations,
        )
