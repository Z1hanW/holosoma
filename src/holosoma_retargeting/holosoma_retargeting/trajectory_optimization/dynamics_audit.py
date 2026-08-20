"""Inverse-dynamics and one-step rollout audits for retargeted trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .object_pose import decode_object_poses


@dataclass(frozen=True)
class MujocoTrajectoryDynamicsAudit:
    fps: float
    qvel: np.ndarray
    qacc: np.ndarray
    controls: np.ndarray
    generalized_force_residual_norm: np.ndarray
    root_force_residual_norm: np.ndarray
    control_violation: np.ndarray
    rollout_qpos_defect_norm: np.ndarray
    rollout_qvel_defect_norm: np.ndarray

    def summary(self) -> dict[str, float | int | str]:
        return {
            "fps_assumption": self.fps,
            "control_mapping": "fixed-gain direct motor least squares",
            "max_abs_control": float(np.max(np.abs(self.controls), initial=0.0)),
            "control_violating_values": int(
                np.count_nonzero(self.control_violation > 0.0)
            ),
            "max_control_violation": float(
                np.max(self.control_violation, initial=0.0)
            ),
            "mean_generalized_force_residual": float(
                np.mean(self.generalized_force_residual_norm)
            ),
            "max_generalized_force_residual": float(
                np.max(self.generalized_force_residual_norm, initial=0.0)
            ),
            "mean_root_force_residual": float(
                np.mean(self.root_force_residual_norm)
            ),
            "max_root_force_residual": float(
                np.max(self.root_force_residual_norm, initial=0.0)
            ),
            "mean_rollout_qpos_defect": float(
                np.mean(self.rollout_qpos_defect_norm)
            ),
            "max_rollout_qpos_defect": float(
                np.max(self.rollout_qpos_defect_norm, initial=0.0)
            ),
            "mean_rollout_qvel_defect": float(
                np.mean(self.rollout_qvel_defect_norm)
            ),
            "max_rollout_qvel_defect": float(
                np.max(self.rollout_qvel_defect_norm, initial=0.0)
            ),
        }


class MujocoTrajectoryDynamicsAuditor:
    """Estimate controls and rollout defects for a sampled qpos trajectory."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        fps: float,
        object_geom_name: str = "trajopt_object_geom",
    ) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be positive and finite")
        self.model = model
        self.fps = float(fps)
        object_geom_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                object_geom_name,
            )
        )
        self.object_mocap_id = -1
        if object_geom_id >= 0:
            object_body_id = int(model.geom_bodyid[object_geom_id])
            self.object_mocap_id = int(model.body_mocapid[object_body_id])
        if model.nu:
            fixed_gain = np.all(
                model.actuator_gaintype
                == mujoco.mjtGain.mjGAIN_FIXED
            )
            no_bias = np.all(
                model.actuator_biastype
                == mujoco.mjtBias.mjBIAS_NONE
            )
            unit_gain = np.allclose(model.actuator_gainprm[:, 0], 1.0)
            if not (fixed_gain and no_bias and unit_gain):
                raise ValueError(
                    "dynamics audit currently requires unit fixed-gain "
                    "actuators without bias"
                )

    def _kinematics(
        self,
        qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = 1.0 / self.fps
        frame_count = len(qpos)
        qvel = np.empty((frame_count, self.model.nv), dtype=np.float64)
        difference = np.empty(self.model.nv, dtype=np.float64)
        for frame in range(frame_count):
            if frame == 0:
                left, right, interval = 0, 1, dt
            elif frame == frame_count - 1:
                left, right, interval = frame_count - 2, frame_count - 1, dt
            else:
                left, right, interval = frame - 1, frame + 1, 2.0 * dt
            mujoco.mj_differentiatePos(
                self.model,
                difference,
                interval,
                qpos[left],
                qpos[right],
            )
            qvel[frame] = difference
        qacc = np.gradient(qvel, dt, axis=0, edge_order=1)
        return qvel, qacc

    def _actuator_moment_matrix(
        self,
        data: mujoco.MjData,
    ) -> np.ndarray:
        moment = np.zeros((self.model.nu, self.model.nv), dtype=np.float64)
        values = np.asarray(data.actuator_moment, dtype=np.float64).reshape(-1)
        row_addresses = np.asarray(data.moment_rowadr, dtype=np.int32)
        row_nonzeros = np.asarray(data.moment_rownnz, dtype=np.int32)
        column_indices = np.asarray(data.moment_colind, dtype=np.int32)
        for actuator in range(self.model.nu):
            start = int(row_addresses[actuator])
            count = int(row_nonzeros[actuator])
            columns = column_indices[start : start + count]
            moment[actuator, columns] = values[start : start + count]
        return moment

    def audit(
        self,
        qpos: np.ndarray,
        object_poses: np.ndarray | None = None,
        *,
        quaternion_order: str = "xyzw",
        pose_layout: str = "auto",
    ) -> MujocoTrajectoryDynamicsAudit:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(f"qpos must have shape (T, {self.model.nq})")
        if len(qpos) < 2:
            raise ValueError("qpos must contain at least two frames")
        transforms = None
        if object_poses is not None:
            if self.object_mocap_id < 0:
                raise ValueError("object_poses require a mocap object geom")
            transforms = decode_object_poses(
                object_poses,
                quaternion_order=quaternion_order,
                pose_layout=pose_layout,
            )
            if len(transforms.positions) != len(qpos):
                raise ValueError("object_poses frame count disagrees with qpos")

        qvel, qacc = self._kinematics(qpos)
        dt = 1.0 / self.fps
        controls = np.zeros((len(qpos), self.model.nu), dtype=np.float64)
        force_residual = np.empty(len(qpos), dtype=np.float64)
        root_residual = np.empty(len(qpos), dtype=np.float64)
        inverse_data = mujoco.MjData(self.model)
        original_inverse_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = dt
        try:
            for frame in range(len(qpos)):
                inverse_data.qpos[:] = qpos[frame]
                inverse_data.qvel[:] = qvel[frame]
                inverse_data.qacc[:] = qacc[frame]
                if transforms is not None:
                    inverse_data.mocap_pos[self.object_mocap_id] = (
                        transforms.positions[frame]
                    )
                    inverse_data.mocap_quat[self.object_mocap_id] = (
                        transforms.quaternions_wxyz[frame]
                    )
                mujoco.mj_inverse(self.model, inverse_data)
                moment = self._actuator_moment_matrix(inverse_data)
                if self.model.nu:
                    control, *_ = np.linalg.lstsq(
                        moment.T,
                        np.asarray(
                            inverse_data.qfrc_inverse
                        ).reshape(-1),
                        rcond=None,
                    )
                    controls[frame] = control
                    residual = (
                        inverse_data.qfrc_inverse
                        - moment.T @ control
                    )
                else:
                    residual = inverse_data.qfrc_inverse.copy()
                force_residual[frame] = np.linalg.norm(residual)
                root_residual[frame] = np.linalg.norm(residual[:6])
        finally:
            self.model.opt.timestep = original_inverse_timestep

        control_lower = np.full(self.model.nu, -np.inf)
        control_upper = np.full(self.model.nu, np.inf)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        control_lower[limited] = self.model.actuator_ctrlrange[limited, 0]
        control_upper[limited] = self.model.actuator_ctrlrange[limited, 1]
        control_violation = np.maximum(
            np.maximum(control_lower[None, :] - controls, 0.0),
            np.maximum(controls - control_upper[None, :], 0.0),
        )

        original_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = dt
        rollout_qpos_defect = np.empty(len(qpos) - 1, dtype=np.float64)
        rollout_qvel_defect = np.empty(len(qpos) - 1, dtype=np.float64)
        rollout_data = mujoco.MjData(self.model)
        qpos_difference = np.empty(self.model.nv, dtype=np.float64)
        try:
            for frame in range(len(qpos) - 1):
                rollout_data.qpos[:] = qpos[frame]
                rollout_data.qvel[:] = qvel[frame]
                rollout_data.ctrl[:] = controls[frame]
                if transforms is not None:
                    rollout_data.mocap_pos[self.object_mocap_id] = (
                        transforms.positions[frame]
                    )
                    rollout_data.mocap_quat[self.object_mocap_id] = (
                        transforms.quaternions_wxyz[frame]
                    )
                mujoco.mj_step(self.model, rollout_data)
                mujoco.mj_differentiatePos(
                    self.model,
                    qpos_difference,
                    1.0,
                    qpos[frame + 1],
                    rollout_data.qpos,
                )
                rollout_qpos_defect[frame] = np.linalg.norm(qpos_difference)
                rollout_qvel_defect[frame] = np.linalg.norm(
                    rollout_data.qvel - qvel[frame + 1]
                )
        finally:
            self.model.opt.timestep = original_timestep
        return MujocoTrajectoryDynamicsAudit(
            fps=self.fps,
            qvel=qvel,
            qacc=qacc,
            controls=controls,
            generalized_force_residual_norm=force_residual,
            root_force_residual_norm=root_residual,
            control_violation=control_violation,
            rollout_qpos_defect_norm=rollout_qpos_defect,
            rollout_qvel_defect_norm=rollout_qvel_defect,
        )
