"""Trajectory-wide MuJoCo object and ground collision linearization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .object_pose import decode_object_poses


@dataclass(frozen=True)
class TrajectoryCollisionAudit:
    ground_minimum_distance: np.ndarray
    object_minimum_distance: np.ndarray
    ground_limiting_geoms: tuple[str, ...]
    object_limiting_geoms: tuple[str, ...]

    def summary(self, minimum_distance: float) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for kind, distances in (
            ("ground", self.ground_minimum_distance),
            ("object", self.object_minimum_distance),
        ):
            finite = distances[np.isfinite(distances)]
            result[f"{kind}_minimum_distance_m"] = (
                float(np.min(finite)) if len(finite) else float("inf")
            )
            result[f"{kind}_violating_frames"] = int(
                np.count_nonzero(distances < minimum_distance)
            )
            result[f"{kind}_max_violation_m"] = (
                float(np.max(np.maximum(minimum_distance - finite, 0.0)))
                if len(finite)
                else 0.0
            )
        return result


@dataclass(frozen=True)
class TrajectoryCollisionLinearization:
    frames: np.ndarray
    jacobians: np.ndarray
    distances: np.ndarray
    kinds: tuple[str, ...]
    robot_geoms: tuple[str, ...]


def build_mocap_object_model(
    model_path: str | Path,
    object_mesh_path: str | Path,
    *,
    object_body_name: str = "trajopt_object",
    object_geom_name: str = "trajopt_object_geom",
) -> mujoco.MjModel:
    """Compile a robot model with a pose-controlled collision mesh."""

    model_path = Path(model_path).expanduser().resolve()
    object_mesh_path = Path(object_mesh_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not object_mesh_path.is_file():
        raise FileNotFoundError(object_mesh_path)
    spec = mujoco.MjSpec.from_file(str(model_path))
    mesh_name = f"{object_geom_name}_mesh"
    spec.add_mesh(name=mesh_name, file=str(object_mesh_path))
    body = spec.worldbody.add_body(name=object_body_name, mocap=True)
    body.add_geom(
        name=object_geom_name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        contype=1,
        conaffinity=1,
        mass=0.1,
    )
    return spec.compile()


class MujocoTrajectoryCollision:
    """Evaluate and linearize robot collision distances over a trajectory."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        object_geom_name: str = "trajopt_object_geom",
        ground_geom_name: str = "ground",
    ) -> None:
        self.model = model
        self.object_geom_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                object_geom_name,
            )
        )
        if self.object_geom_id < 0:
            raise ValueError(f"model has no object geom named {object_geom_name!r}")
        object_body_id = int(model.geom_bodyid[self.object_geom_id])
        self.object_mocap_id = int(model.body_mocapid[object_body_id])
        if self.object_mocap_id < 0:
            raise ValueError("object collision geom must belong to a mocap body")
        self.ground_geom_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                ground_geom_name,
            )
        )
        if self.ground_geom_id < 0:
            raise ValueError(f"model has no ground geom named {ground_geom_name!r}")
        excluded = {self.object_geom_id, self.ground_geom_id}
        self.robot_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in range(model.ngeom)
                if geom_id not in excluded
                and (
                    model.geom_contype[geom_id] != 0
                    or model.geom_conaffinity[geom_id] != 0
                )
            ],
            dtype=np.int32,
        )
        if not len(self.robot_geom_ids):
            raise ValueError("model has no collision-enabled robot geoms")
        self._geom_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
            or f"geom_{geom_id}"
            for geom_id in range(model.ngeom)
        )

    def _set_frame(
        self,
        data: mujoco.MjData,
        qpos: np.ndarray,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> None:
        data.qpos[:] = qpos
        data.mocap_pos[self.object_mocap_id] = position
        data.mocap_quat[self.object_mocap_id] = quaternion_wxyz
        mujoco.mj_forward(self.model, data)

    def audit(
        self,
        qpos: np.ndarray,
        object_poses: np.ndarray,
        *,
        quaternion_order: str,
        pose_layout: str = "auto",
        distance_limit: float = 0.2,
    ) -> TrajectoryCollisionAudit:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(f"qpos must have shape (T, {self.model.nq})")
        transforms = decode_object_poses(
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        if len(transforms.positions) != len(qpos):
            raise ValueError("object_poses frame count disagrees with qpos")
        if not np.isfinite(distance_limit) or distance_limit <= 0.0:
            raise ValueError("distance_limit must be positive and finite")

        ground_distances = np.empty(len(qpos), dtype=np.float64)
        object_distances = np.empty(len(qpos), dtype=np.float64)
        ground_names: list[str] = []
        object_names: list[str] = []
        data = mujoco.MjData(self.model)
        fromto = np.empty(6, dtype=np.float64)
        for frame in range(len(qpos)):
            self._set_frame(
                data,
                qpos[frame],
                transforms.positions[frame],
                transforms.quaternions_wxyz[frame],
            )
            frame_ground = []
            frame_object = []
            for geom_id in self.robot_geom_ids:
                geom_id = int(geom_id)
                frame_ground.append(
                    float(
                        mujoco.mj_geomDistance(
                            self.model,
                            data,
                            geom_id,
                            self.ground_geom_id,
                            distance_limit,
                            fromto,
                        )
                    )
                )
                frame_object.append(
                    float(
                        mujoco.mj_geomDistance(
                            self.model,
                            data,
                            geom_id,
                            self.object_geom_id,
                            distance_limit,
                            fromto,
                        )
                    )
                )
            ground_index = int(np.argmin(frame_ground))
            object_index = int(np.argmin(frame_object))
            ground_distances[frame] = frame_ground[ground_index]
            object_distances[frame] = frame_object[object_index]
            ground_names.append(
                self._geom_names[int(self.robot_geom_ids[ground_index])]
            )
            object_names.append(
                self._geom_names[int(self.robot_geom_ids[object_index])]
            )
        return TrajectoryCollisionAudit(
            ground_minimum_distance=ground_distances,
            object_minimum_distance=object_distances,
            ground_limiting_geoms=tuple(ground_names),
            object_limiting_geoms=tuple(object_names),
        )

    def linearize(
        self,
        qpos: np.ndarray,
        object_poses: np.ndarray,
        *,
        quaternion_order: str,
        pose_layout: str = "auto",
        activation_distance: float = 0.05,
        include_ground: bool = True,
        include_object: bool = True,
    ) -> TrajectoryCollisionLinearization:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != self.model.nq:
            raise ValueError(f"qpos must have shape (T, {self.model.nq})")
        if not np.isfinite(activation_distance) or activation_distance <= 0.0:
            raise ValueError("activation_distance must be positive and finite")
        transforms = decode_object_poses(
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        if len(transforms.positions) != len(qpos):
            raise ValueError("object_poses frame count disagrees with qpos")

        obstacles = []
        if include_ground:
            obstacles.append(("ground", self.ground_geom_id))
        if include_object:
            obstacles.append(("object", self.object_geom_id))
        frames: list[int] = []
        jacobians: list[np.ndarray] = []
        distances: list[float] = []
        kinds: list[str] = []
        robot_names: list[str] = []
        data = mujoco.MjData(self.model)
        fromto = np.empty(6, dtype=np.float64)
        jacobian_position = np.empty((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.empty((3, self.model.nv), dtype=np.float64)

        for frame in range(len(qpos)):
            self._set_frame(
                data,
                qpos[frame],
                transforms.positions[frame],
                transforms.quaternions_wxyz[frame],
            )
            for geom_id_value in self.robot_geom_ids:
                geom_id = int(geom_id_value)
                robot_body_id = int(self.model.geom_bodyid[geom_id])
                for kind, obstacle_id in obstacles:
                    fromto[:] = 0.0
                    distance = float(
                        mujoco.mj_geomDistance(
                            self.model,
                            data,
                            geom_id,
                            obstacle_id,
                            activation_distance,
                            fromto,
                        )
                    )
                    if distance >= activation_distance:
                        continue
                    difference = fromto[:3] - fromto[3:]
                    difference_norm = float(np.linalg.norm(difference))
                    if difference_norm > 1e-12:
                        normal = np.sign(distance) * difference / difference_norm
                    elif kind == "ground":
                        normal = np.asarray([0.0, 0.0, 1.0])
                    else:
                        continue
                    mujoco.mj_jac(
                        self.model,
                        data,
                        jacobian_position,
                        jacobian_rotation,
                        fromto[:3],
                        robot_body_id,
                    )
                    jacobian = normal @ jacobian_position
                    if np.linalg.norm(jacobian) < 1e-12:
                        continue
                    frames.append(frame)
                    jacobians.append(jacobian.copy())
                    distances.append(distance)
                    kinds.append(kind)
                    robot_names.append(self._geom_names[geom_id])

        row_count = len(frames)
        return TrajectoryCollisionLinearization(
            frames=np.asarray(frames, dtype=np.int32),
            jacobians=(
                np.asarray(jacobians, dtype=np.float64)
                if row_count
                else np.empty((0, self.model.nv), dtype=np.float64)
            ),
            distances=np.asarray(distances, dtype=np.float64),
            kinds=tuple(kinds),
            robot_geoms=tuple(robot_names),
        )
