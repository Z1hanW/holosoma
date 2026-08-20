"""Object-frame MuJoCo kinematic linearization for interaction TrajOpt."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .object_pose import decode_object_poses


@dataclass(frozen=True)
class ObjectFrameKinematicLinearization:
    state_jacobian: np.ndarray
    scale_jacobian: np.ndarray
    target: np.ndarray
    robot_points_object: np.ndarray
    human_points_object: np.ndarray


class MujocoObjectFrameKinematics:
    """Linearize mapped robot points relative to a moving object."""

    _BODY_ALIASES = {
        "left_rubber_hand_link": (
            "left_sphere_hand_link",
            "left_hand_sphere_link",
            "left_hand_link",
        ),
        "right_rubber_hand_link": (
            "right_sphere_hand_link",
            "right_hand_sphere_link",
            "right_hand_link",
        ),
    }

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: list[str],
        body_mapping: dict[str, str],
    ) -> None:
        if not joint_names:
            raise ValueError("joint_names must not be empty")
        self.model = model
        self.joint_names = list(joint_names)
        self.body_names = [
            self._resolve_body(body_mapping[name]) for name in self.joint_names
        ]
        self.body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )
                for body_name in self.body_names
            ],
            dtype=np.int32,
        )
        if np.any(self.body_ids < 0):
            raise ValueError("failed to resolve every mapped MuJoCo body")
        try:
            self.pelvis_index = self.joint_names.index("Pelvis")
        except ValueError as exc:
            raise ValueError("joint_names must contain Pelvis") from exc

    def _resolve_body(self, requested: str) -> str:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            requested,
        )
        if body_id >= 0:
            return requested
        for alias in self._BODY_ALIASES.get(requested, ()):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                alias,
            )
            if body_id >= 0:
                return alias
        raise ValueError(f"MuJoCo model has no body matching {requested!r}")

    def mapped_points(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(f"qpos must have shape (T, {self.model.nq})")
        data = mujoco.MjData(self.model)
        points = np.empty((len(qpos), len(self.body_ids), 3), dtype=np.float64)
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(self.model, data)
            points[frame] = data.xpos[self.body_ids]
        return points

    def align_seed_root(
        self,
        qpos: np.ndarray,
        human_points_world: np.ndarray,
    ) -> np.ndarray:
        """Translate a free-root seed so mapped pelvis matches the target pelvis."""

        qpos = np.asarray(qpos, dtype=np.float64).copy()
        human_points_world = np.asarray(human_points_world, dtype=np.float64)
        expected_human_shape = (len(qpos), len(self.joint_names), 3)
        if human_points_world.shape != expected_human_shape:
            raise ValueError(
                f"human_points_world must have shape {expected_human_shape}"
            )
        if self.model.njnt == 0 or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("align_seed_root requires the first joint to be free")
        data = mujoco.MjData(self.model)
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(self.model, data)
            delta = (
                human_points_world[frame, self.pelvis_index]
                - data.xpos[self.body_ids[self.pelvis_index]]
            )
            qpos[frame, :3] += delta
        return qpos

    def linearize(
        self,
        qpos: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        *,
        quaternion_order: str = "xyzw",
        pose_layout: str = "auto",
    ) -> ObjectFrameKinematicLinearization:
        qpos = np.asarray(qpos, dtype=np.float64)
        human_points_world = np.asarray(human_points_world, dtype=np.float64)
        frame_count = len(qpos)
        joint_count = len(self.joint_names)
        if qpos.shape != (frame_count, self.model.nq):
            raise ValueError(f"qpos must have shape (T, {self.model.nq})")
        if human_points_world.shape != (frame_count, joint_count, 3):
            raise ValueError(
                f"human_points_world must have shape {(frame_count, joint_count, 3)}"
            )
        object_transforms = decode_object_poses(
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        object_positions = object_transforms.positions
        object_rotations = object_transforms.rotations
        if len(object_positions) != frame_count:
            raise ValueError("object_poses frame count disagrees with qpos")

        row_count = 3 * joint_count
        state_jacobian = np.empty(
            (frame_count, row_count, self.model.nv),
            dtype=np.float64,
        )
        scale_jacobian = np.zeros(
            (frame_count, row_count, 3),
            dtype=np.float64,
        )
        target = np.empty((frame_count, row_count), dtype=np.float64)
        robot_points_object = np.empty(
            (frame_count, joint_count, 3),
            dtype=np.float64,
        )
        human_points_object = np.empty_like(robot_points_object)
        data = mujoco.MjData(self.model)
        jacobian_rotation = np.empty((3, self.model.nv), dtype=np.float64)

        for frame in range(frame_count):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(self.model, data)
            world_to_object = object_rotations[frame].T
            human_points_object[frame] = (
                human_points_world[frame] - object_positions[frame]
            ) @ object_rotations[frame]
            pelvis = human_points_object[frame, self.pelvis_index]
            for joint, body_id in enumerate(self.body_ids):
                jacobian_position = np.empty(
                    (3, self.model.nv),
                    dtype=np.float64,
                )
                mujoco.mj_jacBody(
                    self.model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    int(body_id),
                )
                row_slice = slice(3 * joint, 3 * (joint + 1))
                state_jacobian[frame, row_slice] = (
                    world_to_object @ jacobian_position
                )
                robot_point = world_to_object @ (
                    data.xpos[body_id] - object_positions[frame]
                )
                robot_points_object[frame, joint] = robot_point
                human_offset = human_points_object[frame, joint] - pelvis
                scale_jacobian[
                    frame,
                    np.arange(3 * joint, 3 * (joint + 1)),
                    np.arange(3),
                ] = -human_offset
                target[frame, row_slice] = pelvis - robot_point

        arrays = (
            state_jacobian,
            scale_jacobian,
            target,
            robot_points_object,
            human_points_object,
        )
        if any(not np.isfinite(array).all() for array in arrays):
            raise RuntimeError("kinematic linearization produced non-finite values")
        return ObjectFrameKinematicLinearization(
            state_jacobian=state_jacobian,
            scale_jacobian=scale_jacobian,
            target=target,
            robot_points_object=robot_points_object,
            human_points_object=human_points_object,
        )

    def retract(self, qpos: np.ndarray, tangent_deltas: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        tangent_deltas = np.asarray(tangent_deltas, dtype=np.float64)
        if tangent_deltas.shape != (len(qpos), self.model.nv):
            raise ValueError(
                f"tangent_deltas must have shape {(len(qpos), self.model.nv)}"
            )
        for frame in range(len(qpos)):
            mujoco.mj_integratePos(
                self.model,
                qpos[frame],
                tangent_deltas[frame],
                1.0,
            )
        return qpos

    def object_frame_residual(
        self,
        qpos: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        frame_scales: np.ndarray,
        *,
        quaternion_order: str = "xyzw",
        pose_layout: str = "auto",
    ) -> np.ndarray:
        robot_points_world = self.mapped_points(qpos)
        object_transforms = decode_object_poses(
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        object_positions = object_transforms.positions
        object_rotations = object_transforms.rotations
        human_points_world = np.asarray(human_points_world, dtype=np.float64)
        frame_scales = np.asarray(frame_scales, dtype=np.float64)
        if frame_scales.shape != (len(qpos), 3):
            raise ValueError(f"frame_scales must have shape {(len(qpos), 3)}")
        residual = np.empty(
            (len(qpos), len(self.joint_names), 3),
            dtype=np.float64,
        )
        for frame in range(len(qpos)):
            robot_object = (
                robot_points_world[frame] - object_positions[frame]
            ) @ object_rotations[frame]
            human_object = (
                human_points_world[frame] - object_positions[frame]
            ) @ object_rotations[frame]
            pelvis = human_object[self.pelvis_index]
            scaled_target = pelvis + (
                human_object - pelvis
            ) * frame_scales[frame]
            residual[frame] = robot_object - scaled_target
        return residual

    def object_frame_error(
        self,
        qpos: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        frame_scales: np.ndarray,
        *,
        quaternion_order: str = "xyzw",
        pose_layout: str = "auto",
    ) -> np.ndarray:
        residual = self.object_frame_residual(
            qpos,
            human_points_world,
            object_poses,
            frame_scales,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        return np.linalg.norm(residual, axis=2)
