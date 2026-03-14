import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.misc import get_index_of_a_in_b
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_apply,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class PinocchioRobot:
    def __init__(self, robot_cfg: RobotConfig, urdf_text: str):
        # create pinocchio robot
        xml_text = self._create_xml_from_urdf(urdf_text)
        self.robot_model = pin.buildModelFromXML(xml_text, pin.JointModelFreeFlyer())
        self.robot_data = self.robot_model.createData()

        # get joint names in pinocchio robot and real robot
        joint_names_in_real_robot = robot_cfg.dof_names
        joint_names_in_pinocchio_robot = [
            name for name in self.robot_model.names if name not in ["universe", "root_joint"]
        ]
        assert len(joint_names_in_pinocchio_robot) == len(joint_names_in_real_robot), (
            "The number of joints in the pinocchio robot and the real robot are not the same"
        )
        self.real2pinocchio_index = get_index_of_a_in_b(joint_names_in_pinocchio_robot, joint_names_in_real_robot)

        # get ref body frame id in pinocchio robot
        self.ref_body_frame_id = self.robot_model.getFrameId(robot_cfg.motion["body_name_ref"][0])

    def fk_and_get_ref_body_pose_in_world(self, configuration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )
        position = ref_body_pose_in_world.translation

        return np.array(position, dtype=np.float32), np.array(quaternion.coeffs(), dtype=np.float32)

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        _, quat_xyzw = self.fk_and_get_ref_body_pose_in_world(configuration)
        return np.expand_dims(quat_xyzw, axis=0)  # xyzw, (1, 4)

    @staticmethod
    def _create_xml_from_urdf(urdf_text: str) -> str:
        """Strip visuals/collisions from URDF text and return XML text."""
        root = ElementTree.fromstring(urdf_text)

        def _is_visual_or_collision(tag: str) -> bool:
            # Handle optional XML namespaces by only checking the suffix after '}'.
            return tag.split("}")[-1] in {"visual", "collision"}

        for parent in root.iter():
            for child in list(parent):
                if _is_visual_or_collision(child.tag):
                    parent.remove(child)

        xml_text = ElementTree.tostring(root, encoding="unicode")
        if not xml_text.lstrip().startswith("<?xml"):
            xml_text = '<?xml version="1.0"?>\n' + xml_text
        return xml_text


class MotionData:
    def __init__(self, motion_path: Path, robot_dof_names: list[str], body_name_ref: str):
        if motion_path.suffix.lower() != ".npz":
            raise ValueError(f"Only .npz motion files are supported in inference: {motion_path}")

        with np.load(motion_path, allow_pickle=True) as data:
            body_names = self._decode_names(data["body_names"])
            joint_names = self._decode_names(data["joint_names"])

            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            if joint_pos.shape[1] == len(joint_names) + 7:
                joint_pos = joint_pos[:, 7:]
            elif joint_pos.shape[1] != len(joint_names):
                raise ValueError(
                    f"Unexpected joint_pos shape {joint_pos.shape} for {motion_path}; "
                    f"expected {len(joint_names)} or {len(joint_names) + 7} columns."
                )

            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            if joint_vel.shape[1] == len(joint_names) + 6:
                joint_vel = joint_vel[:, 6:]
            elif joint_vel.shape[1] != len(joint_names):
                raise ValueError(
                    f"Unexpected joint_vel shape {joint_vel.shape} for {motion_path}; "
                    f"expected {len(joint_names)} or {len(joint_names) + 6} columns."
                )

            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
            object_quat_w = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None
            has_object_size_field = any(
                key in data for key in ("object_size", "object_sizes", "object_scale", "object_scales")
            )
            object_size = self._extract_object_size(data, length=int(body_pos_w.shape[0]))

        joint_indices = get_index_of_a_in_b(robot_dof_names, joint_names)
        self.joint_pos = joint_pos[:, joint_indices]
        self.joint_vel = joint_vel[:, joint_indices]
        self.frame_count = self.joint_pos.shape[0]

        if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
            raise ValueError(f"Unexpected body_quat_w shape {body_quat_w.shape} in {motion_path}")

        self.ref_body_index = body_names.index(body_name_ref)
        self.ref_pos_w = body_pos_w[:, self.ref_body_index, :]
        self.ref_quat_w = body_quat_w[:, self.ref_body_index, :]
        self.root_quat_w = body_quat_w[:, 0, :]
        self.root_pos_w = body_pos_w[:, 0, :]
        self.has_object = object_pos_w is not None and object_quat_w is not None
        if self.has_object:
            self.object_pos_w = np.asarray(object_pos_w, dtype=np.float32)
            self.object_quat_w = np.asarray(object_quat_w, dtype=np.float32)
            self.object_size = np.asarray(object_size, dtype=np.float32)
            if not has_object_size_field:
                logger.warning(
                    "Motion file '{}' has object pose but no object_size/object_scale field; "
                    "obj_target_pose_size_b will use fallback size [1,1,1].",
                    motion_path,
                )
        else:
            self.object_pos_w = None
            self.object_quat_w = None
            self.object_size = None

    @staticmethod
    def _decode_names(arr: np.ndarray) -> list[str]:
        names = arr.tolist()
        decoded: list[str] = []
        for name in names:
            if isinstance(name, bytes):
                decoded.append(name.decode("utf-8"))
            else:
                decoded.append(str(name))
        return decoded

    @staticmethod
    def _extract_object_size(data: dict, length: int) -> np.ndarray | None:
        size_keys = ("object_size", "object_sizes", "object_scale", "object_scales")
        for key in size_keys:
            if key not in data:
                continue
            raw = np.asarray(data[key], dtype=np.float32)
            if raw.ndim == 1:
                if raw.shape[0] != 3:
                    continue
                return np.repeat(raw.reshape(1, 3), length, axis=0)
            if raw.ndim == 2:
                if raw.shape[0] == length and raw.shape[1] == 3:
                    return raw
                if raw.shape[0] == 3 and raw.shape[1] == length:
                    return raw.T
            if raw.ndim == 3 and raw.shape[-1] == 3:
                flattened = raw.reshape(-1, 3)
                if flattened.shape[0] >= length:
                    return flattened[:length]
        return np.ones((length, 3), dtype=np.float32)

    def apply_transition(
        self,
        start_state: dict[str, np.ndarray],
        target_state: dict[str, np.ndarray],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
    ) -> None:
        if num_steps <= 0:
            return

        alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.size == 0:
            return

        def _lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            view = alphas.reshape(-1, *([1] * a.ndim))
            return a + view * (b - a)

        segment_joint_pos = _lerp(start_state["joint_pos"], target_state["joint_pos"])
        segment_joint_vel = _lerp(start_state["joint_vel"], target_state["joint_vel"])
        segment_root_pos = _lerp(start_state["root_pos"], target_state["root_pos"])
        segment_ref_pos = _lerp(start_state["ref_pos"], target_state["ref_pos"])
        segment_root_quat = _slerp_quat_wxyz(start_state["root_quat"], target_state["root_quat"], alphas)
        segment_ref_quat = _slerp_quat_wxyz(start_state["ref_quat"], target_state["ref_quat"], alphas)
        segment_object_pos = None
        segment_object_quat = None
        segment_object_size = None
        if self.has_object:
            segment_object_pos = _lerp(start_state["object_pos"], target_state["object_pos"])
            segment_object_quat = _slerp_quat_wxyz(start_state["object_quat"], target_state["object_quat"], alphas)
            segment_object_size = _lerp(start_state["object_size"], target_state["object_size"])

        if prepend:
            self.joint_pos = np.concatenate([segment_joint_pos, self.joint_pos], axis=0)
            self.joint_vel = np.concatenate([segment_joint_vel, self.joint_vel], axis=0)
            self.root_pos_w = np.concatenate([segment_root_pos, self.root_pos_w], axis=0)
            self.root_quat_w = np.concatenate([segment_root_quat, self.root_quat_w], axis=0)
            self.ref_pos_w = np.concatenate([segment_ref_pos, self.ref_pos_w], axis=0)
            self.ref_quat_w = np.concatenate([segment_ref_quat, self.ref_quat_w], axis=0)
            if self.has_object:
                assert self.object_pos_w is not None and self.object_quat_w is not None and self.object_size is not None
                self.object_pos_w = np.concatenate([segment_object_pos, self.object_pos_w], axis=0)
                self.object_quat_w = np.concatenate([segment_object_quat, self.object_quat_w], axis=0)
                self.object_size = np.concatenate([segment_object_size, self.object_size], axis=0)
        else:
            self.joint_pos = np.concatenate([self.joint_pos, segment_joint_pos], axis=0)
            self.joint_vel = np.concatenate([self.joint_vel, segment_joint_vel], axis=0)
            self.root_pos_w = np.concatenate([self.root_pos_w, segment_root_pos], axis=0)
            self.root_quat_w = np.concatenate([self.root_quat_w, segment_root_quat], axis=0)
            self.ref_pos_w = np.concatenate([self.ref_pos_w, segment_ref_pos], axis=0)
            self.ref_quat_w = np.concatenate([self.ref_quat_w, segment_ref_quat], axis=0)
            if self.has_object:
                assert self.object_pos_w is not None and self.object_quat_w is not None and self.object_size is not None
                self.object_pos_w = np.concatenate([self.object_pos_w, segment_object_pos], axis=0)
                self.object_quat_w = np.concatenate([self.object_quat_w, segment_object_quat], axis=0)
                self.object_size = np.concatenate([self.object_size, segment_object_size], axis=0)

        self.frame_count = int(self.joint_pos.shape[0])


def _yaw_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    qx = quat[..., 0]
    qy = quat[..., 1]
    qz = quat[..., 2]
    qw = quat[..., 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros_like(quat)
    quat_yaw[..., 2] = np.sin(yaw / 2)
    quat_yaw[..., 3] = np.cos(yaw / 2)
    norm = np.linalg.norm(quat_yaw, axis=-1, keepdims=True)
    return np.divide(quat_yaw, norm, out=quat_yaw, where=norm > 0)


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate([-quat[..., :3], quat[..., 3:4]], axis=-1)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([x, y, z, w], axis=-1)


def _quat_apply_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    t = np.cross(xyz, vec) * 2.0
    return vec + w * t + np.cross(xyz, t)


def _matrix_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat_wxyz = xyzw_to_wxyz(quat.reshape(-1, 4))
    mats = matrix_from_quat(quat_wxyz).reshape(quat.shape[:-1] + (3, 3))
    return mats


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.divide(quat, norm, out=quat, where=norm > 0)


def _slerp_quat_wxyz(start: np.ndarray, end: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Slerp between two wxyz quaternions for a sequence of alphas in [0, 1]."""
    start = _normalize_quat_wxyz(np.asarray(start, dtype=np.float32).reshape(4))
    end = _normalize_quat_wxyz(np.asarray(end, dtype=np.float32).reshape(4))
    alphas = np.asarray(alphas, dtype=np.float32).reshape(-1)
    if alphas.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot

    if dot > 0.9995:
        blended = start[None, :] + (end - start)[None, :] * alphas[:, None]
        return _normalize_quat_wxyz(blended)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alphas
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, None] * start[None, :]) + (s1[:, None] * end[None, :])


class MotionFutureTargetPoseProvider:
    def __init__(
        self,
        motion_file: str,
        body_names_to_track: list[str],
        num_future_steps: int,
        target_pose_type: str,
        dt: float,
    ) -> None:
        self.motion_file = motion_file
        self.body_names_to_track = body_names_to_track
        self.num_future_steps = int(num_future_steps)
        self.target_pose_type = target_pose_type
        self.dt = float(dt)
        self.include_time = target_pose_type == "max-coords-future-rel-with-time"
        self.body_names, self.body_pos_w, self.body_quat_w, self.object_pos_w = self._load_motion_npz(motion_file)
        self.time_step_total = int(self.body_pos_w.shape[0])
        self.tracked_body_indexes = self._resolve_body_indexes(self.body_names, body_names_to_track)
        self.num_bodies = len(self.tracked_body_indexes)
        self.obs_dim = self.num_future_steps * (self.num_bodies * 18 + (1 if self.include_time else 0))
        self._step_obs_dim = self.num_bodies * 18 + (1 if self.include_time else 0)

        # Caches from the latest get_future_target_poses() call for debug visualization.
        self._last_obs_flat: np.ndarray | None = None
        self._last_future_steps: np.ndarray | None = None
        self._last_reference_root_pos: np.ndarray | None = None
        self._last_heading_quat: np.ndarray | None = None
        self._last_object_pos_world: np.ndarray | None = None

    @staticmethod
    def _resolve_body_indexes(body_names: list[str], tracked_names: list[str]) -> list[int]:
        indexes = []
        for name in tracked_names:
            if name not in body_names:
                raise ValueError(f"Body name '{name}' not found in motion data")
            indexes.append(body_names.index(name))
        return indexes

    @staticmethod
    def _load_motion_npz(path: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
        motion_path = Path(path)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        with np.load(motion_path, allow_pickle=True) as data:
            body_names = data["body_names"].tolist()
            body_names = [bn.decode("utf-8") if isinstance(bn, (bytes, bytearray)) else bn for bn in body_names]
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
        body_quat_w = body_quat_w[:, :, [1, 2, 3, 0]]  # wxyz -> xyzw
        return body_names, body_pos_w, body_quat_w, object_pos_w

    def get_future_target_poses(self, time_step: int) -> np.ndarray:
        if self.num_future_steps <= 0 or self.num_bodies == 0:
            return np.zeros((1, 0), dtype=np.float32)
        step = int(np.clip(time_step, 0, self.time_step_total - 1))
        time_offsets = np.arange(1, self.num_future_steps + 1, dtype=np.int64)
        future_steps = np.minimum(step + time_offsets, self.time_step_total - 1)
        times = (future_steps - step).astype(np.float32) * self.dt

        target_body_pos = self.body_pos_w[future_steps][:, self.tracked_body_indexes, :]
        target_body_rot = self.body_quat_w[future_steps][:, self.tracked_body_indexes, :]
        current_body_pos = self.body_pos_w[step, self.tracked_body_indexes, :]
        current_body_rot = self.body_quat_w[step, self.tracked_body_indexes, :]

        reference_body_pos = np.roll(target_body_pos, shift=1, axis=0)
        reference_body_pos[0] = current_body_pos
        reference_body_rot = np.roll(target_body_rot, shift=1, axis=0)
        reference_body_rot[0] = current_body_rot

        reference_root_pos = reference_body_pos[:, 0, :]
        reference_root_rot = reference_body_rot[:, 0, :]
        heading_quat = _yaw_quat_xyzw(reference_root_rot)
        heading_inv = _quat_conjugate_xyzw(heading_quat)
        heading_inv = np.repeat(heading_inv[:, None, :], self.num_bodies, axis=1)

        target_rel_body_pos = target_body_pos - reference_body_pos
        target_body_pos_rel_root = target_body_pos - reference_root_pos[:, None, :]

        rel_body_pos = _quat_apply_xyzw(
            heading_inv.reshape(-1, 4), target_rel_body_pos.reshape(-1, 3)
        ).reshape(self.num_future_steps, self.num_bodies * 3)
        body_pos = _quat_apply_xyzw(
            heading_inv.reshape(-1, 4), target_body_pos_rel_root.reshape(-1, 3)
        ).reshape(self.num_future_steps, self.num_bodies * 3)

        rel_body_rot = _quat_mul_xyzw(_quat_conjugate_xyzw(reference_body_rot), target_body_rot)
        body_rot = _quat_mul_xyzw(heading_inv, target_body_rot)

        rel_body_rot_mat = _matrix_from_quat_xyzw(rel_body_rot.reshape(-1, 4))
        rel_body_rot_obs = rel_body_rot_mat[:, :, :2].reshape(self.num_future_steps, self.num_bodies * 6)
        body_rot_mat = _matrix_from_quat_xyzw(body_rot.reshape(-1, 4))
        body_rot_obs = body_rot_mat[:, :, :2].reshape(self.num_future_steps, self.num_bodies * 6)

        obs = np.concatenate((rel_body_pos, body_pos, rel_body_rot_obs, body_rot_obs), axis=-1)
        if self.include_time:
            obs = np.concatenate((obs, times[:, None]), axis=-1)
        obs_flat = obs.reshape(1, -1).astype(np.float32, copy=False)

        # Cache the exact term used in observation assembly, plus world-frame references
        # needed to render trajectories in MuJoCo GUI.
        self._last_obs_flat = obs_flat
        self._last_future_steps = future_steps.astype(np.int64, copy=False)
        self._last_reference_root_pos = reference_root_pos.astype(np.float32, copy=False)
        self._last_heading_quat = heading_quat.astype(np.float32, copy=False)
        if self.object_pos_w is not None:
            self._last_object_pos_world = self.object_pos_w[future_steps].astype(np.float32, copy=False)
        else:
            self._last_object_pos_world = None
        return obs_flat

    def get_tracking_viz_payload(
        self,
        *,
        max_future_steps: int = 10,
        obs_flat_override: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Build debug payload using the exact future-target observation term fed to policy."""
        if (
            self._last_obs_flat is None
            or self._last_future_steps is None
            or self._last_reference_root_pos is None
            or self._last_heading_quat is None
            or self._step_obs_dim <= 0
        ):
            return None

        obs_flat = self._last_obs_flat if obs_flat_override is None else np.asarray(obs_flat_override, dtype=np.float32)
        if obs_flat.ndim == 1:
            obs_flat = obs_flat.reshape(1, -1)
        if obs_flat.shape[0] != 1:
            return None

        full_dim = int(self.num_future_steps * self._step_obs_dim)
        if obs_flat.shape[1] < full_dim:
            obs_flat = self._last_obs_flat
        else:
            # History flattening order places latest sample at the end.
            obs_flat = obs_flat[:, -full_dim:]

        k = min(int(max_future_steps), int(self.num_future_steps), int(self._last_future_steps.shape[0]))
        if k <= 0:
            return None

        obs_steps = obs_flat.reshape(self.num_future_steps, self._step_obs_dim)[:k]
        # Exact per-step position slice from the term consumed by actor_obs:
        # [rel_body_pos(3B), body_pos(3B), rel_body_rot6d(6B), body_rot6d(6B), (time)]
        body_pos_local = obs_steps[:, self.num_bodies * 3 : self.num_bodies * 6].reshape(k, self.num_bodies, 3)

        heading_quat = self._last_heading_quat[:k]
        ref_root_pos = self._last_reference_root_pos[:k]
        heading_quat_expanded = np.repeat(heading_quat[:, None, :], self.num_bodies, axis=1)
        world_offsets = _quat_apply_xyzw(
            heading_quat_expanded.reshape(-1, 4), body_pos_local.reshape(-1, 3)
        ).reshape(k, self.num_bodies, 3)
        keypoints_world = world_offsets + ref_root_pos[:, None, :]

        payload: dict[str, Any] = {
            "source": "motion_future_target_poses",
            "num_future_steps": int(k),
            "num_tracked_bodies": int(self.num_bodies),
            "tracked_body_names": list(self.body_names_to_track),
            "future_motion_steps": self._last_future_steps[:k].astype(np.int64).tolist(),
            "step_obs_dim": int(self._step_obs_dim),
            "future_obs_flat": obs_steps.reshape(-1).astype(np.float32).tolist(),
            "keypoints_world": keypoints_world.astype(np.float32).tolist(),
        }
        if self._last_object_pos_world is not None:
            payload["object_pos_world"] = self._last_object_pos_world[:k].astype(np.float32).tolist()
        return payload


class MotionRawFutureTrajectoryProvider:
    """Fallback trajectory source when policy does not consume motion_future_target_poses."""

    def __init__(self, motion_file: str, body_names_to_track: list[str]) -> None:
        self.motion_file = motion_file
        self.body_names_to_track = list(body_names_to_track)
        self.body_names, self.body_pos_w, _, self.object_pos_w = MotionFutureTargetPoseProvider._load_motion_npz(motion_file)
        self.tracked_body_indexes = MotionFutureTargetPoseProvider._resolve_body_indexes(
            self.body_names, self.body_names_to_track
        )
        self.num_bodies = len(self.tracked_body_indexes)
        self.time_step_total = int(self.body_pos_w.shape[0])

    def get_tracking_viz_payload(self, *, time_step: int, max_future_steps: int = 10) -> dict[str, Any] | None:
        if self.num_bodies <= 0 or self.time_step_total <= 0:
            return None
        step = int(np.clip(time_step, 0, self.time_step_total - 1))
        k = max(1, int(max_future_steps))
        offsets = np.arange(1, k + 1, dtype=np.int64)
        future_steps = np.minimum(step + offsets, self.time_step_total - 1)
        keypoints_world = self.body_pos_w[future_steps][:, self.tracked_body_indexes, :]
        payload: dict[str, Any] = {
            "source": "motion_file_world_fallback",
            "is_exact_policy_input": False,
            "num_future_steps": int(future_steps.shape[0]),
            "num_tracked_bodies": int(self.num_bodies),
            "tracked_body_names": list(self.body_names_to_track),
            "future_motion_steps": future_steps.astype(np.int64).tolist(),
            "keypoints_world": keypoints_world.astype(np.float32).tolist(),
        }
        if self.object_pos_w is not None:
            payload["object_pos_world"] = self.object_pos_w[future_steps].astype(np.float32).tolist()
        return payload


class _ZeroFutureTargetPoseProvider:
    def __init__(self, obs_dim: int) -> None:
        self.obs_dim = int(obs_dim)

    def get_future_target_poses(self, time_step: int) -> np.ndarray:  # noqa: ARG002 - signature match
        return np.zeros((1, self.obs_dim), dtype=np.float32)


OBJECT_OBS_DIMS = {
    "obj_target_pose_size_b": 12,
    "obj_pos_b": 3,
    "obj_ori_b": 6,
    "obj_lin_vel_b": 3,
}


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None
        self.ref_pos_xyz_t = None

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub()
        self.clock_sub.start()
        self._last_clock_reading: int | None = None

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._motion_future_target_pose_provider = None
        self._onnx_metadata: dict | None = None
        self._onnx_obs_dim: int | None = None

        obs_terms = {term for terms in config.observation.obs_dict.values() for term in terms}
        self._uses_videomimic = any(
            term in obs_terms
            for term in (
                "torso_real",
                "torso_xy_rel",
                "torso_yaw_rel",
                "target_joints",
                "target_root_roll",
                "target_root_pitch",
            )
        )
        self._uses_motion_command = any(
            term in obs_terms for term in ("motion_command", "motion_ref_ori_b", "motion_future_target_poses")
        )
        self._uses_object_obs = any(term in obs_terms for term in OBJECT_OBS_DIMS)
        self._motion_data: MotionData | None = None
        self._motion_cfg: dict | None = None
        self._motion_dof_names: list[str] | None = None
        self._motion_align_quat_wxyz: np.ndarray | None = None
        self._motion_align_pos: np.ndarray | None = None
        self._obs_input_name: str | None = None
        self._time_step_input_name: str | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._motion_output_names: set[str] = set()
        self._motion_alignment_enabled = False
        self._object_state_sub = None
        self._prefer_sim_ref_from_object_state = str(
            os.getenv("HOLOSOMA_PREFER_SIM_REF_FROM_OBJECT_STATE", "1")
        ).lower() in {"1", "true", "yes", "on"}
        self._object_state_missing_warned = False
        self._has_object_state_sample = False
        self._auto_start_wait_for_object_state = False
        self._object_sync_diag_logged = False
        self._policy_action_scale_arr: np.ndarray | None = None
        self._onnx_timestep_offset: int = 0
        self._onnx_timestep_offset_aligned: bool = False
        # Disabled by default to preserve stable MuJoCo rollout behavior.
        # Set HOLOSOMA_ONNX_ALIGN_MAX_STEPS>0 to re-enable startup phase search.
        self._onnx_timestep_search_max_steps: int = max(0, int(os.getenv("HOLOSOMA_ONNX_ALIGN_MAX_STEPS", "0")))
        self._onnx_timestep_align_pose_tolerance: float = 5e-3
        self._onnx_unclamp_time_step: bool = str(os.getenv("HOLOSOMA_ONNX_UNCLAMP_TIME_STEP", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._onnx_offset_applies_to_motion_index: bool = str(
            os.getenv("HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX", "1")
        ).lower() in {"1", "true", "yes", "on"}
        self._tracking_viz_pub = None
        self._tracking_viz_pub_enabled = False
        self._tracking_viz_pub_port = int(getattr(config.task, "tracking_viz_pub_port", 5560))
        self._tracking_viz_pub_every_n = max(1, int(getattr(config.task, "tracking_viz_pub_every_n", 1)))
        self._tracking_viz_pub_tick = 0
        self._tracking_viz_fallback_provider: MotionRawFutureTrajectoryProvider | None = None
        self._tracking_viz_no_source_warned = False
        self._object_sync_diag_counter: int = 0
        self._object_sync_diag_every_ticks: int = max(1, int(config.task.rl_rate))

        self._joystick_goal_enabled = bool(config.task.use_joystick_goal)
        self._joystick_goal_scale = float(config.task.joystick_goal_scale)
        self._joystick_yaw_scale = float(config.task.joystick_yaw_scale)
        self._auto_start_stage: str | None = None
        self._auto_start_hold_ticks = 0
        self._auto_start_max_wait_ticks = 0
        self._auto_start_tick_count = 0
        self._auto_start_pose_tolerance = float(config.task.auto_start_stiff_pose_tolerance)

        super().__init__(config)

        if self._joystick_goal_enabled:
            self.use_joystick = True
            self.stand_command[0, 0] = 1.0

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        auto_start_enabled = bool(
            getattr(config.task, "auto_start_policy", False) or getattr(config.task, "auto_start_motion_clip", False)
        )
        if auto_start_enabled:
            logger.info("Auto-start enabled; skipping stiff hold confirmation prompt.")
        elif sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    def _get_ref_body_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_pos, ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(configuration)
        ref_pos = np.expand_dims(ref_pos, axis=0)
        return ref_pos, xyzw_to_wxyz(np.expand_dims(ref_ori_xyzw, axis=0))

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        _, ref_quat_wxyz = self._get_ref_body_pose_in_world(robot_state_data)
        return ref_quat_wxyz

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)
        self._onnx_metadata = metadata
        self._onnx_obs_dim = self._get_onnx_obs_dim()
        metadata_action_scale = self._extract_action_scale_from_metadata(metadata)
        if metadata_action_scale is not None and abs(float(self.policy_action_scale) - metadata_action_scale) > 1e-6:
            logger.warning(
                "Overriding task.policy_action_scale from {} to {} based on ONNX metadata robot.control.action_scale.",
                self.policy_action_scale,
                metadata_action_scale,
            )
            self.policy_action_scale = float(metadata_action_scale)
        self._configure_action_scale_from_metadata(metadata)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self._maybe_enable_object_observations(metadata, model_path)

        if self._uses_videomimic and not self._joystick_goal_enabled and self._motion_data is None:
            self._load_motion_data_from_metadata(metadata, model_path)
            self._apply_default_pose_transitions()
        self._maybe_enable_motion_future_target_poses(metadata, model_path)
        self._maybe_setup_tracking_viz_fallback_provider(metadata, model_path)
        self._setup_tracking_viz_publisher()
        self._log_actor_obs_alignment(metadata)

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported ONNX inputs: {self.onnx_input_names}")

        self._time_step_input_name = "time_step" if "time_step" in self.onnx_input_names else None

        if "actions" in self.onnx_output_names:
            self._action_output_name = "actions"
        elif "action" in self.onnx_output_names:
            self._action_output_name = "action"
        else:
            self._action_output_name = self.onnx_output_names[0]

        self._motion_output_names = set(self.onnx_output_names)
        required_motion_outputs = {"joint_pos", "joint_vel", "ref_quat_xyzw"}
        if self._uses_motion_command and not required_motion_outputs.issubset(self._motion_output_names):
            raise ValueError(
                "Motion outputs missing from ONNX; expected joint_pos, joint_vel, ref_quat_xyzw. "
                f"Available: {self.onnx_output_names}"
            )

        self._onnx_output_fetch = [self._action_output_name]
        if self._uses_motion_command:
            self._onnx_output_fetch += ["joint_pos", "joint_vel", "ref_quat_xyzw"]
            if "ref_pos_xyz" in self._motion_output_names:
                self._onnx_output_fetch.append("ref_pos_xyz")

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(self._onnx_output_fetch, input_feed)
            return dict(zip(self._onnx_output_fetch, output))

        self.policy = policy_act

        if self._uses_motion_command:
            time_step = np.zeros((1, 1), dtype=np.float32)

            obs_dim = self._onnx_obs_dim
            if obs_dim is None:
                group_dims: list[int] = []
                for group_name in self.actor_obs_group_order:
                    group_template = self.obs_buf_dict.get(group_name)
                    if group_template is None:
                        raise ValueError(f"Observation group '{group_name}' must be configured for WBT policy.")
                    group_dims.append(int(group_template.shape[1]))
                obs = np.zeros((1, sum(group_dims)), dtype=np.float32)
            else:
                obs = np.zeros((1, obs_dim), dtype=np.float32)

            input_feed = {self._obs_input_name: obs}
            if self._time_step_input_name:
                input_feed[self._time_step_input_name] = time_step
            outputs = self.policy(input_feed)
            joint_pos = outputs["joint_pos"]
            joint_vel = outputs["joint_vel"]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.ref_quat_xyzw_t = outputs["ref_quat_xyzw"]
            self.ref_pos_xyz_t = outputs.get("ref_pos_xyz")
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
        elif self._uses_videomimic and self._motion_data is not None:
            joint_pos = self._motion_data.joint_pos[:1]
            joint_vel = self._motion_data.joint_vel[:1]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.motion_command_0 = self.motion_command_t.copy()
            ref_quat_wxyz = self._motion_data.ref_quat_w[:1]
            self.ref_quat_xyzw_t = wxyz_to_xyzw(ref_quat_wxyz)
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self.ref_pos_xyz_t = self._motion_data.ref_pos_w[:1]
        elif self._joystick_goal_enabled and self.motion_command_0 is None:
            joint_pos = self.default_dof_angles.reshape(1, -1).astype(np.float32, copy=False)
            joint_vel = np.zeros_like(joint_pos)
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_t = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self.ref_pos_xyz_t = np.zeros((1, 3), dtype=np.float32)

        if self._uses_motion_command:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_align_onnx_timestep_offset_to_current_pose(robot_state_data)

    def _get_onnx_obs_dim(self) -> int | None:
        inputs = self.onnx_policy_session.get_inputs()
        for inp in inputs:
            if inp.name in {"obs", "actor_obs"}:
                shape = inp.shape
                if len(shape) > 1 and isinstance(shape[1], int):
                    return int(shape[1])
        if inputs:
            shape = inputs[0].shape
            if len(shape) > 1 and isinstance(shape[1], int):
                return int(shape[1])
        return None

    def _build_zero_actor_obs(self) -> np.ndarray:
        obs_dim = self._onnx_obs_dim
        if obs_dim is None:
            group_dims: list[int] = []
            for group_name in self.actor_obs_group_order:
                group_template = self.obs_buf_dict.get(group_name)
                if group_template is None:
                    raise ValueError(f"Observation group '{group_name}' must be configured for WBT policy.")
                group_dims.append(int(group_template.shape[1]))
            obs_dim = int(sum(group_dims))
        return np.zeros((1, int(obs_dim)), dtype=np.float32)

    def _query_motion_outputs_at(self, onnx_timestep: int) -> dict[str, np.ndarray] | None:
        if self._obs_input_name is None:
            return None
        if "joint_pos" not in self.onnx_output_names or "joint_vel" not in self.onnx_output_names:
            return None

        obs = self._build_zero_actor_obs()
        input_feed = {self._obs_input_name: obs}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[int(onnx_timestep)]], dtype=np.float32)

        fetch_names = ["joint_pos", "joint_vel"]
        if "ref_quat_xyzw" in self.onnx_output_names:
            fetch_names.append("ref_quat_xyzw")
        if "ref_pos_xyz" in self.onnx_output_names:
            fetch_names.append("ref_pos_xyz")

        outputs = self.onnx_policy_session.run(fetch_names, input_feed)
        return dict(zip(fetch_names, outputs))

    def _maybe_align_onnx_timestep_offset_to_current_pose(self, robot_state_data: np.ndarray | None) -> None:
        if self._onnx_timestep_offset_aligned:
            return

        if not self._uses_motion_command or self._time_step_input_name is None:
            return
        if self._onnx_timestep_search_max_steps <= 0:
            self._onnx_timestep_offset = 0
            self._onnx_timestep_offset_aligned = True
            logger.info("ONNX time_step startup alignment disabled; using offset 0.")
            return
        if robot_state_data is None or robot_state_data.shape[1] < 7 + self.num_dofs:
            return

        target_q = np.asarray(robot_state_data[:, 7 : 7 + self.num_dofs], dtype=np.float32)
        if target_q.shape != (1, self.num_dofs):
            return

        query0 = self._query_motion_outputs_at(0)
        if query0 is None:
            return

        def _err_from_query(query: dict[str, np.ndarray]) -> float:
            return float(np.max(np.abs(np.asarray(query["joint_pos"], dtype=np.float32) - target_q)))

        best_t = 0
        best_query = query0
        err_t0 = _err_from_query(query0)
        best_err = err_t0

        if err_t0 > self._onnx_timestep_align_pose_tolerance:
            max_steps = self._onnx_timestep_search_max_steps
            for t in range(1, max_steps + 1):
                query = self._query_motion_outputs_at(t)
                if query is None:
                    break
                err = _err_from_query(query)
                if err < best_err:
                    best_err = err
                    best_t = t
                    best_query = query
                    if best_err <= self._onnx_timestep_align_pose_tolerance:
                        break

            if best_t != 0:
                logger.warning(
                    "Aligned ONNX time_step offset to current init pose: +{} (joint max err {:.4f} -> {:.4f} rad).",
                    best_t,
                    err_t0,
                    best_err,
                )
            elif best_err > self._onnx_timestep_align_pose_tolerance:
                logger.warning(
                    "ONNX startup pose misalignment remains after search (best joint max err {:.4f} rad within 0..{}).",
                    best_err,
                    self._onnx_timestep_search_max_steps,
                )

        self._onnx_timestep_offset = int(best_t)
        if best_query is not None and "joint_pos" in best_query and "joint_vel" in best_query:
            joint_pos = np.asarray(best_query["joint_pos"], dtype=np.float32)
            joint_vel = np.asarray(best_query["joint_vel"], dtype=np.float32)
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.motion_command_0 = self.motion_command_t.copy()
            if "ref_quat_xyzw" in best_query:
                self.ref_quat_xyzw_t = np.asarray(best_query["ref_quat_xyzw"], dtype=np.float32)
                self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            if "ref_pos_xyz" in best_query:
                self.ref_pos_xyz_t = np.asarray(best_query["ref_pos_xyz"], dtype=np.float32)
        self._onnx_timestep_offset_aligned = True

    @staticmethod
    def _find_repo_root(start: Path) -> Path:
        for parent in [start, *start.parents]:
            if (parent / "src" / "holosoma").exists():
                return parent
        return start

    def _resolve_motion_file(self, motion_file: str, model_path: str | Path | None = None) -> str | None:
        if not motion_file:
            return None
        motion_path = Path(motion_file).expanduser()
        if motion_path.is_file():
            return str(motion_path.resolve())
        if motion_file.startswith("holosoma/data"):
            suffix = motion_file[13:].lstrip("/")
            try:
                from importlib.resources import files

                candidate = files("holosoma.data") / suffix
                if candidate.exists():
                    return str(candidate)
            except Exception:
                pass
        if model_path:
            resolved_model_path = Path(model_path).expanduser().resolve()
            if resolved_model_path.is_file():
                candidate = resolved_model_path.parent / motion_file
                if candidate.is_file():
                    return str(candidate)
            else:
                candidate = resolved_model_path / motion_file
                if candidate.is_file():
                    return str(candidate)
        repo_root = self._find_repo_root(Path(__file__).resolve())
        candidate = repo_root / motion_file
        if candidate.is_file():
            return str(candidate)
        if motion_file.startswith("holosoma/"):
            candidate = repo_root / "src" / motion_file
            if candidate.is_file():
                return str(candidate)
        candidate = repo_root / "src" / motion_file
        if candidate.is_file():
            return str(candidate)
        logger.warning("Motion file not found: {}", motion_file)
        return None

    def _extract_motion_config(self, metadata: dict) -> dict | None:
        motion_cfg = metadata.get("motion_config") if metadata else None
        if isinstance(motion_cfg, dict):
            return motion_cfg
        exp_cfg = metadata.get("experiment_config") if metadata else None
        if not isinstance(exp_cfg, dict):
            return None
        motion_cfg = (
            exp_cfg.get("command", {})
            .get("setup_terms", {})
            .get("motion_command", {})
            .get("params", {})
            .get("motion_config", {})
        )
        return motion_cfg if isinstance(motion_cfg, dict) else None

    @staticmethod
    def _extract_action_scale_from_metadata(metadata: dict) -> float | None:
        if not isinstance(metadata, dict):
            return None
        exp_cfg = metadata.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        robot_cfg = exp_cfg.get("robot")
        if not isinstance(robot_cfg, dict):
            return None
        control_cfg = robot_cfg.get("control")
        if not isinstance(control_cfg, dict):
            return None
        value = control_cfg.get("action_scale")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _configure_action_scale_from_metadata(self, metadata: dict) -> None:
        self._policy_action_scale_arr = None
        disable_effort_over_kp = str(os.getenv("HOLOSOMA_DISABLE_EFFORT_OVER_KP_ACTION_SCALE", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_effort_over_kp:
            logger.info(
                "Using scalar policy_action_scale={} (effort/kp scaling explicitly disabled by env).",
                self.policy_action_scale,
            )
            return

        scalar = float(self.policy_action_scale)
        if not isinstance(metadata, dict):
            logger.info("Using scalar policy_action_scale={} (metadata unavailable).", self.policy_action_scale)
            return

        exp_cfg = metadata.get("experiment_config")
        robot_cfg = exp_cfg.get("robot") if isinstance(exp_cfg, dict) else None
        control_cfg = robot_cfg.get("control") if isinstance(robot_cfg, dict) else None
        effort_limits = robot_cfg.get("dof_effort_limit_list") if isinstance(robot_cfg, dict) else None
        use_effort_over_kp = bool(
            isinstance(control_cfg, dict) and control_cfg.get("action_scales_by_effort_limit_over_p_gain", False)
        )
        kp_values = metadata.get("kp")
        if (
            use_effort_over_kp
            and isinstance(effort_limits, list)
            and len(effort_limits) == self.num_dofs
            and isinstance(kp_values, list)
            and len(kp_values) == self.num_dofs
        ):
            effort = np.asarray(effort_limits, dtype=np.float32).reshape(1, -1)
            kp = np.asarray(kp_values, dtype=np.float32).reshape(1, -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_arr = np.where(np.abs(kp) > 1e-8, scalar * effort / kp, 0.0).astype(np.float32)
            logger.info(
                "Using per-joint action scales from metadata (effort/kp), range=[{:.4f}, {:.4f}].",
                float(np.min(scale_arr)),
                float(np.max(scale_arr)),
            )
            self._policy_action_scale_arr = scale_arr
            return

        if use_effort_over_kp:
            logger.warning(
                "Metadata requests effort/kp action scaling, but required kp/effort arrays are missing or invalid; "
                "falling back to scalar policy_action_scale={}.",
                self.policy_action_scale,
            )
        else:
            logger.info("Using scalar policy_action_scale={} (metadata requests scalar scaling).", self.policy_action_scale)

    def _load_motion_data_from_metadata(self, metadata: dict, model_path: str | Path) -> None:
        motion_cfg = self._extract_motion_config(metadata)
        if not motion_cfg:
            raise ValueError("Motion config missing from ONNX metadata; cannot build VideoMimic observations.")

        metadata_motion_file = motion_cfg.get("motion_file")
        override_motion_file = self.config.task.motion_future_target_poses_motion_file
        if override_motion_file and metadata_motion_file and str(override_motion_file) != str(metadata_motion_file):
            logger.warning(
                "Overriding checkpoint motion source from '{}' to '{}'. "
                "This can misalign object goal terms (obj_target_pose_size_b / object target pose).",
                metadata_motion_file,
                override_motion_file,
            )

        motion_file = override_motion_file or metadata_motion_file
        if not motion_file:
            raise ValueError("motion_config.motion_file missing from ONNX metadata.")

        motion_path = self._resolve_motion_file(str(motion_file), model_path)
        if motion_path is None:
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        body_name_ref = motion_cfg.get("body_name_ref", ["torso_link"])
        if isinstance(body_name_ref, list) and body_name_ref:
            ref_name = body_name_ref[0]
        else:
            ref_name = "torso_link"
        if hasattr(self, "pinocchio_robot"):
            try:
                self.pinocchio_robot.ref_body_frame_id = self.pinocchio_robot.robot_model.getFrameId(ref_name)
            except Exception as exc:
                logger.warning("Failed to set Pinocchio ref body '{}': {}", ref_name, exc)

        robot_dof_names = metadata.get("dof_names") or list(self.config.robot.dof_names)
        self._motion_dof_names = list(robot_dof_names)
        self._motion_data = MotionData(Path(motion_path), list(robot_dof_names), ref_name)
        self._motion_cfg = motion_cfg
        self._motion_alignment_enabled = bool(motion_cfg.get("align_motion_to_init_yaw", False))

    def _extract_init_state_from_metadata(self) -> dict | None:
        if not self._onnx_metadata:
            return None
        exp_cfg = self._onnx_metadata.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        robot_cfg = exp_cfg.get("robot")
        if not isinstance(robot_cfg, dict):
            return None
        init_state = robot_cfg.get("init_state")
        return init_state if isinstance(init_state, dict) else None

    def _build_default_pose_state(self, *, use_motion_end: bool) -> dict[str, np.ndarray] | None:
        if self._motion_data is None:
            return None
        init_state = self._extract_init_state_from_metadata()
        if init_state is None:
            logger.warning("Init state missing from ONNX metadata; skipping default pose transition.")
            return None

        dof_names = self._motion_dof_names or list(self.config.robot.dof_names)
        default_dof = np.array(self.config.robot.default_dof_angles, dtype=np.float32)
        default_joint_angles = init_state.get("default_joint_angles")
        if isinstance(default_joint_angles, dict) and len(default_dof) == len(dof_names):
            for i, name in enumerate(dof_names):
                if name in default_joint_angles:
                    default_dof[i] = float(default_joint_angles[name])

        motion_idx = -1 if use_motion_end else 0
        motion_root_pos = self._motion_data.root_pos_w[motion_idx]
        motion_root_quat = self._motion_data.root_quat_w[motion_idx]
        motion_yaw = self._quat_yaw(motion_root_quat)

        init_pos = np.array(init_state.get("pos", [0.0, 0.0, motion_root_pos[2]]), dtype=np.float32)
        init_rot_xyzw = np.array(init_state.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32).reshape(1, 4)
        init_rot_wxyz = xyzw_to_wxyz(init_rot_xyzw)[0]
        init_roll, init_pitch, _ = quat_to_rpy(init_rot_wxyz)

        default_root_pos = np.array(
            [motion_root_pos[0], motion_root_pos[1], init_pos[2]],
            dtype=np.float32,
        )
        default_root_quat = rpy_to_quat((init_roll, init_pitch, motion_yaw)).astype(np.float32)

        root_quat_xyzw = wxyz_to_xyzw(default_root_quat.reshape(1, 4))[0]
        dof_pos_pin = default_dof[self.pinocchio_robot.real2pinocchio_index]
        configuration = np.concatenate([default_root_pos, root_quat_xyzw, dof_pos_pin], axis=0)
        ref_pos, ref_quat_xyzw = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(configuration)
        ref_quat_wxyz = xyzw_to_wxyz(ref_quat_xyzw.reshape(1, 4))[0]

        state = {
            "joint_pos": default_dof,
            "joint_vel": np.zeros_like(default_dof),
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "ref_pos": ref_pos.astype(np.float32, copy=False),
            "ref_quat": ref_quat_wxyz.astype(np.float32, copy=False),
        }
        if self._motion_data.has_object and self._motion_data.object_pos_w is not None:
            state["object_pos"] = self._motion_data.object_pos_w[motion_idx].astype(np.float32, copy=False)
            state["object_quat"] = self._motion_data.object_quat_w[motion_idx].astype(np.float32, copy=False)
            state["object_size"] = self._motion_data.object_size[motion_idx].astype(np.float32, copy=False)
        return state

    def _build_motion_state(self, motion_idx: int) -> dict[str, np.ndarray] | None:
        if self._motion_data is None:
            return None
        idx = int(np.clip(motion_idx, -self._motion_data.frame_count, self._motion_data.frame_count - 1))
        state = {
            "joint_pos": self._motion_data.joint_pos[idx],
            "joint_vel": self._motion_data.joint_vel[idx],
            "root_pos": self._motion_data.root_pos_w[idx],
            "root_quat": self._motion_data.root_quat_w[idx],
            "ref_pos": self._motion_data.ref_pos_w[idx],
            "ref_quat": self._motion_data.ref_quat_w[idx],
        }
        if self._motion_data.has_object and self._motion_data.object_pos_w is not None:
            state["object_pos"] = self._motion_data.object_pos_w[idx]
            state["object_quat"] = self._motion_data.object_quat_w[idx]
            state["object_size"] = self._motion_data.object_size[idx]
        return state

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        if self._motion_data is None or self._motion_cfg is None:
            return

        enabled_key = "enable_default_pose_prepend" if prepend else "enable_default_pose_append"
        duration_key = "default_pose_prepend_duration_s" if prepend else "default_pose_append_duration_s"
        enabled = bool(self._motion_cfg.get(enabled_key, False))
        duration = float(self._motion_cfg.get(duration_key, 0.0) or 0.0)
        if not enabled or duration <= 0.0:
            return

        dt = 1.0 / float(self.config.task.rl_rate)
        num_steps = round(duration / dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)
        motion_state = self._build_motion_state(0 if prepend else -1)
        if default_state is None or motion_state is None:
            return

        start_state = default_state if prepend else motion_state
        target_state = motion_state if prepend else default_state
        drop_first, drop_last = (False, True) if prepend else (True, False)
        self._motion_data.apply_transition(start_state, target_state, num_steps, prepend, drop_first, drop_last)

    def _apply_default_pose_transitions(self) -> None:
        if self._motion_data is None or self._motion_cfg is None:
            return
        self._maybe_add_default_pose_transition(prepend=True)
        self._maybe_add_default_pose_transition(prepend=False)

    @staticmethod
    def _extract_actor_obs_terms_from_metadata(metadata: dict) -> list[str] | None:
        if not isinstance(metadata, dict):
            return None
        exp_cfg = metadata.get("experiment_config")
        if not isinstance(exp_cfg, dict):
            return None
        obs_cfg = exp_cfg.get("observation")
        if not isinstance(obs_cfg, dict):
            return None
        groups = obs_cfg.get("groups")
        if not isinstance(groups, dict):
            return None
        actor_group = groups.get("actor_obs")
        if not isinstance(actor_group, dict):
            return None
        terms = actor_group.get("terms")
        if not isinstance(terms, dict):
            return None
        return list(terms.keys())

    def _enable_actor_obs_terms(self, actor_terms: list[str], extra_dims: dict[str, int]) -> None:
        obs_dict = {group: list(terms) for group, terms in self.obs_dict.items()}
        obs_dims = dict(self.obs_dims)
        obs_scales = dict(self.obs_scales)
        history_length_dict = dict(self.history_length_dict)

        unknown_terms = [term for term in actor_terms if term not in obs_dims and term not in extra_dims]
        if unknown_terms:
            logger.warning("Dropping unsupported actor obs terms from metadata: {}", unknown_terms)
        actor_terms_filtered = [term for term in actor_terms if term not in unknown_terms]
        for term in actor_terms_filtered:
            if term in extra_dims:
                obs_dims[term] = int(extra_dims[term])
                obs_scales.setdefault(term, 1.0)

        obs_dict["actor_obs"] = actor_terms_filtered
        history_length_dict.setdefault("actor_obs", 1)
        self._reset_obs_config(
            ObservationConfig(
                obs_dict=obs_dict,
                obs_dims=obs_dims,
                obs_scales=obs_scales,
                history_length_dict=history_length_dict,
            )
        )

    def _maybe_start_object_state_subscriber(self, force: bool = False) -> None:
        if self._object_state_sub is not None:
            return
        if not force and not bool(getattr(self.config.task, "object_state_sub_enabled", False)):
            return
        try:
            from holosoma_inference.utils.object_state import ObjectStateSub  # noqa: PLC0415
        except Exception as exc:
            logger.warning("Object-state subscriber unavailable: {}", exc)
            return

        port = int(getattr(self.config.task, "object_state_sub_port", 5557))
        try:
            sub = ObjectStateSub(port=port)
            sub.start()
            self._object_state_sub = sub
            logger.info("Object-state subscriber enabled on port {}.", port)
        except Exception as exc:
            logger.warning("Failed to start object-state subscriber on port {}: {}", port, exc)

    def _setup_tracking_viz_publisher(self) -> None:
        if self._tracking_viz_pub is not None:
            return
        if not bool(getattr(self.config.task, "tracking_viz_pub_enabled", False)):
            return
        try:
            from holosoma_inference.utils.tracking_viz import TrackingVizPub  # noqa: PLC0415
        except Exception as exc:
            logger.warning("Tracking-viz publisher unavailable: {}", exc)
            return
        pub = TrackingVizPub(port=self._tracking_viz_pub_port)
        pub.start()
        if not pub.enabled:
            logger.warning("Tracking-viz publisher failed to start on port {}.", self._tracking_viz_pub_port)
            return
        self._tracking_viz_pub = pub
        self._tracking_viz_pub_enabled = True
        self._tracking_viz_pub_tick = 0
        logger.info(
            "Tracking-viz publisher enabled on port {} (every {} ticks, future_steps={}).",
            self._tracking_viz_pub_port,
            self._tracking_viz_pub_every_n,
            int(getattr(self.config.task, "tracking_viz_future_steps", 10)),
        )
        if self._motion_future_target_pose_provider is not None:
            logger.info("Tracking-viz source: motion_future_target_poses (exact policy input).")
        elif self._tracking_viz_fallback_provider is not None:
            logger.warning("Tracking-viz source: motion_file_world_fallback (debug fallback, not exact policy input).")
        else:
            logger.warning(
                "Tracking-viz is enabled but no trajectory source is available; "
                "no keypoint/object overlay will be published."
            )

    def _extract_motion_future_target_obs_from_actor(self, actor_obs: np.ndarray | None) -> np.ndarray | None:
        if actor_obs is None:
            return None
        actor_obs = np.asarray(actor_obs, dtype=np.float32)
        if actor_obs.ndim == 1:
            actor_obs = actor_obs.reshape(1, -1)
        if actor_obs.ndim != 2 or actor_obs.shape[0] != 1:
            return None
        offset = 0
        for group_name in self.actor_obs_group_order:
            terms = self.obs_terms_sorted.get(group_name, [])
            history_len = max(1, int(self.history_length_dict.get(group_name, 1)))
            for term in terms:
                term_dim = int(self.obs_dims.get(term, 0)) * history_len
                if term == "motion_future_target_poses":
                    if term_dim <= 0 or offset + term_dim > actor_obs.shape[1]:
                        return None
                    return actor_obs[:, offset : offset + term_dim].copy()
                offset += term_dim
        return None

    def _publish_tracking_viz_payload(self, actor_obs: np.ndarray | None = None) -> None:
        if not self._tracking_viz_pub_enabled or self._tracking_viz_pub is None:
            return

        self._tracking_viz_pub_tick += 1
        if (self._tracking_viz_pub_tick - 1) % self._tracking_viz_pub_every_n != 0:
            return

        payload = None
        max_future_steps = int(getattr(self.config.task, "tracking_viz_future_steps", 10))
        if self._motion_future_target_pose_provider is not None:
            obs_group = self._extract_motion_future_target_obs_from_actor(actor_obs)
            payload = self._motion_future_target_pose_provider.get_tracking_viz_payload(
                max_future_steps=max_future_steps,
                obs_flat_override=obs_group,
            )
        elif self._tracking_viz_fallback_provider is not None:
            payload = self._tracking_viz_fallback_provider.get_tracking_viz_payload(
                time_step=int(self._get_policy_motion_timestep()),
                max_future_steps=max_future_steps,
            )
        elif not self._tracking_viz_no_source_warned:
            logger.warning(
                "Tracking-viz enabled but no source available for model {} (obs_dim={}).",
                Path(self.active_model_path).name if self.active_model_path else "unknown",
                self._onnx_obs_dim,
            )
            self._tracking_viz_no_source_warned = True
            return

        if payload is None:
            return

        payload["motion_timestep"] = int(self.motion_timestep)
        payload["onnx_timestep_raw"] = int(self._get_raw_policy_motion_timestep())
        payload["onnx_timestep"] = int(self._get_policy_motion_timestep())
        payload["use_sim_time"] = bool(self.use_sim_time)
        payload["policy_clip_progressing"] = bool(self.motion_clip_progressing)
        self._tracking_viz_pub.publish(payload)

    def _close_tracking_viz_publisher(self) -> None:
        if self._tracking_viz_pub is None:
            return
        try:
            self._tracking_viz_pub.close()
        except Exception:
            pass
        self._tracking_viz_pub = None
        self._tracking_viz_pub_enabled = False

    def _maybe_enable_object_observations(self, metadata: dict, model_path: str) -> None:
        actor_terms = self._extract_actor_obs_terms_from_metadata(metadata)
        if actor_terms:
            object_terms_in_meta = [term for term in actor_terms if term in OBJECT_OBS_DIMS]
            if object_terms_in_meta:
                self._enable_actor_obs_terms(actor_terms, OBJECT_OBS_DIMS)
                logger.info("Enabled object-aware actor observations from ONNX metadata: {}", object_terms_in_meta)

        actor_obs_terms = list(self.obs_dict.get("actor_obs", []))
        self._uses_object_obs = any(term in OBJECT_OBS_DIMS for term in actor_obs_terms)
        if not self._uses_object_obs:
            return

        if self._motion_data is None:
            try:
                self._load_motion_data_from_metadata(metadata, model_path)
                self._apply_default_pose_transitions()
            except Exception as exc:
                logger.warning("Object target observation setup failed; motion target fallback will be zeros: {}", exc)

        needs_live_object_state = any(term in actor_obs_terms for term in ("obj_pos_b", "obj_ori_b", "obj_lin_vel_b"))
        self._maybe_start_object_state_subscriber(force=needs_live_object_state)

    def _infer_motion_future_target_poses_dim(self, metadata: dict) -> int | None:
        motion_cfg = self._extract_motion_config(metadata)
        if not isinstance(motion_cfg, dict):
            return None
        body_names_to_track = motion_cfg.get("body_names_to_track") or []
        body_names_to_track = [
            name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name) for name in body_names_to_track
        ]
        num_future_steps = int(motion_cfg.get("num_future_steps", 0))
        target_pose_type = motion_cfg.get("target_pose_type")
        if not body_names_to_track or num_future_steps <= 0 or not target_pose_type:
            return None
        include_time = target_pose_type == "max-coords-future-rel-with-time"
        num_bodies = len(body_names_to_track)
        return num_future_steps * (num_bodies * 18 + (1 if include_time else 0))

    def _build_motion_future_target_pose_provider(
        self, metadata: dict, model_path: str | None
    ) -> MotionFutureTargetPoseProvider | None:
        motion_cfg = self._extract_motion_config(metadata)
        if not isinstance(motion_cfg, dict):
            return None

        motion_file = self.config.task.motion_future_target_poses_motion_file or motion_cfg.get("motion_file")
        motion_file = self._resolve_motion_file(motion_file, model_path) if motion_file else None
        if motion_file is None:
            return None

        body_names_to_track = motion_cfg.get("body_names_to_track") or []
        body_names_to_track = [
            name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name) for name in body_names_to_track
        ]
        num_future_steps = int(motion_cfg.get("num_future_steps", 0))
        target_pose_type = motion_cfg.get("target_pose_type")
        if not body_names_to_track or num_future_steps <= 0 or not target_pose_type:
            return None

        dt = 1.0 / float(self.config.task.rl_rate)
        try:
            return MotionFutureTargetPoseProvider(
                motion_file=motion_file,
                body_names_to_track=list(body_names_to_track),
                num_future_steps=num_future_steps,
                target_pose_type=str(target_pose_type),
                dt=dt,
            )
        except Exception as exc:
            logger.warning("Failed to build motion future target poses provider: {}", exc)
            return None

    def _maybe_setup_tracking_viz_fallback_provider(self, metadata: dict, model_path: str | None) -> None:
        self._tracking_viz_fallback_provider = None
        self._tracking_viz_no_source_warned = False
        if not bool(getattr(self.config.task, "tracking_viz_pub_enabled", False)):
            return
        if self._motion_future_target_pose_provider is not None:
            return

        motion_cfg = self._extract_motion_config(metadata)
        if not isinstance(motion_cfg, dict):
            return

        body_names_to_track = motion_cfg.get("body_names_to_track") or []
        body_names_to_track = [
            name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name) for name in body_names_to_track
        ]
        if not body_names_to_track:
            return

        motion_file = self.config.task.motion_future_target_poses_motion_file or motion_cfg.get("motion_file")
        motion_file = self._resolve_motion_file(motion_file, model_path) if motion_file else None
        if motion_file is None:
            return

        try:
            self._tracking_viz_fallback_provider = MotionRawFutureTrajectoryProvider(
                motion_file=motion_file,
                body_names_to_track=list(body_names_to_track),
            )
            logger.warning(
                "Tracking-viz fallback enabled from motion file world trajectories; "
                "this model does not expose motion_future_target_poses."
            )
        except Exception as exc:
            logger.warning("Tracking-viz fallback setup failed: {}", exc)

    def _maybe_enable_motion_future_target_poses(self, metadata: dict, model_path: str) -> None:
        actor_terms_from_metadata = self._extract_actor_obs_terms_from_metadata(metadata) or []
        if any(term in OBJECT_OBS_DIMS for term in actor_terms_from_metadata):
            # Object-aware policies often add 21+ actor dims; do not reinterpret as future-target poses.
            metadata_motion_cfg = self._extract_motion_config(metadata)
            has_future_targets = bool(metadata_motion_cfg and int(metadata_motion_cfg.get("num_future_steps", 0)) > 0)
            if not has_future_targets and "motion_future_target_poses" not in self.obs_dict:
                return

        base_dim = None
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is not None:
            base_dim = int(actor_obs_template.shape[1])

        extra_dim = None
        if self._onnx_obs_dim is not None and base_dim is not None:
            extra_dim = self._onnx_obs_dim - base_dim
        if "actor_obs_target" in self.obs_dict:
            extra_dim = None

        has_future_group = "motion_future_target_poses" in self.obs_dict
        if extra_dim is not None and extra_dim <= 0 and not has_future_group:
            if self.config.task.include_motion_future_target_poses:
                logger.warning(
                    "ONNX obs dim ({}) does not exceed actor_obs dim ({}); skipping motion_future_target_poses.",
                    self._onnx_obs_dim,
                    base_dim,
                )
            return

        metadata_obs_dim = self._infer_motion_future_target_poses_dim(metadata)
        should_enable = (
            self.config.task.include_motion_future_target_poses
            or (extra_dim is not None and extra_dim > 0)
            or metadata_obs_dim is not None
        )
        if not should_enable:
            return

        if not self.config.task.include_motion_future_target_poses and extra_dim is not None and extra_dim > 0:
            logger.info(
                "ONNX expects {} extra obs dims; auto-enabling motion_future_target_poses.",
                extra_dim,
            )
        elif not self.config.task.include_motion_future_target_poses and metadata_obs_dim is not None:
            logger.info("Metadata indicates motion_future_target_poses; auto-enabling.")

        if has_future_group:
            self.actor_obs_group_order = self._build_actor_obs_group_order()

        provider = self._motion_future_target_pose_provider
        if provider is None:
            provider = self._build_motion_future_target_pose_provider(metadata, model_path)

        obs_dim = None
        if provider is not None:
            obs_dim = provider.obs_dim
        elif "motion_future_target_poses" in self.obs_dims:
            obs_dim = int(self.obs_dims["motion_future_target_poses"])
        elif self.config.task.motion_future_target_poses_dim is not None:
            obs_dim = int(self.config.task.motion_future_target_poses_dim)
        elif extra_dim is not None and extra_dim > 0:
            obs_dim = extra_dim
        elif metadata_obs_dim is not None:
            obs_dim = int(metadata_obs_dim)

        if obs_dim is None or obs_dim <= 0:
            logger.warning(
                "Cannot enable motion_future_target_poses; provide metadata or --task.motion-future-target-poses-dim."
            )
            return

        if provider is None:
            provider = _ZeroFutureTargetPoseProvider(obs_dim)
            logger.warning(
                "Using zero-filled motion_future_target_poses for {} (dim={}).", Path(model_path).name, obs_dim
            )

        self._motion_future_target_pose_provider = provider
        self._enable_motion_future_target_poses(obs_dim)

        if extra_dim is not None and obs_dim != extra_dim:
            logger.warning(
                "motion_future_target_poses dim ({}) does not match ONNX input delta ({}).", obs_dim, extra_dim
            )

    def _log_actor_obs_alignment(self, metadata: dict) -> None:
        actor_terms_meta = self._extract_actor_obs_terms_from_metadata(metadata) or []
        actor_terms_runtime = list(self.obs_dict.get("actor_obs", []))
        meta_dim = int(sum(int(self.obs_dims.get(term, 0)) for term in actor_terms_meta))
        runtime_dim = int(sum(int(self.obs_dims.get(term, 0)) for term in actor_terms_runtime))
        logger.info("Actor obs terms (metadata): {}", actor_terms_meta)
        logger.info("Actor obs terms (runtime):  {}", actor_terms_runtime)
        logger.info(
            "Actor obs alignment: order_match={}, meta_dim={}, runtime_dim={}, onnx_obs_dim={}",
            actor_terms_meta == actor_terms_runtime if actor_terms_meta else "n/a",
            meta_dim,
            runtime_dim,
            self._onnx_obs_dim,
        )

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "obs_input_name": self._obs_input_name,
                "time_step_input_name": self._time_step_input_name,
                "action_output_name": self._action_output_name,
                "onnx_output_fetch": list(self._onnx_output_fetch),
                "motion_output_names": set(self._motion_output_names),
                "onnx_metadata": self._onnx_metadata,
                "onnx_obs_dim": self._onnx_obs_dim,
                "policy_action_scale": float(self.policy_action_scale),
                "policy_action_scale_arr": None
                if self._policy_action_scale_arr is None
                else self._policy_action_scale_arr.copy(),
                "motion_data": self._motion_data,
                "motion_cfg": self._motion_cfg,
                "motion_alignment_enabled": self._motion_alignment_enabled,
                "motion_future_target_pose_provider": self._motion_future_target_pose_provider,
                "uses_object_obs": self._uses_object_obs,
                "object_state_sub": self._object_state_sub,
                "object_state_missing_warned": self._object_state_missing_warned,
                "has_object_state_sample": self._has_object_state_sample,
                "auto_start_wait_for_object_state": self._auto_start_wait_for_object_state,
                "onnx_timestep_offset": int(self._onnx_timestep_offset),
                "onnx_timestep_offset_aligned": bool(self._onnx_timestep_offset_aligned),
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self._obs_input_name = state.get("obs_input_name")
        self._time_step_input_name = state.get("time_step_input_name")
        self._action_output_name = state.get("action_output_name")
        self._onnx_output_fetch = list(state.get("onnx_output_fetch", []))
        self._motion_output_names = set(state.get("motion_output_names", set()))
        self._onnx_metadata = state.get("onnx_metadata")
        self._onnx_obs_dim = state.get("onnx_obs_dim")
        self.policy_action_scale = float(state.get("policy_action_scale", self.policy_action_scale))
        scale_arr = state.get("policy_action_scale_arr")
        self._policy_action_scale_arr = None if scale_arr is None else np.asarray(scale_arr, dtype=np.float32).copy()
        self._motion_data = state.get("motion_data")
        self._motion_cfg = state.get("motion_cfg")
        self._motion_alignment_enabled = bool(state.get("motion_alignment_enabled", False))
        self._motion_future_target_pose_provider = state.get("motion_future_target_pose_provider")
        self._uses_object_obs = bool(state.get("uses_object_obs", False))
        self._object_state_sub = state.get("object_state_sub")
        self._object_state_missing_warned = bool(state.get("object_state_missing_warned", False))
        self._has_object_state_sample = bool(state.get("has_object_state_sample", False))
        self._auto_start_wait_for_object_state = bool(state.get("auto_start_wait_for_object_state", False))
        self._onnx_timestep_offset = int(state.get("onnx_timestep_offset", 0))
        self._onnx_timestep_offset_aligned = bool(state.get("onnx_timestep_offset_aligned", False))
        self._object_sync_diag_logged = False
        self._object_sync_diag_counter = 0
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._object_state_missing_warned = False
        self._has_object_state_sample = False
        self._auto_start_wait_for_object_state = False
        self._object_sync_diag_logged = False
        self._object_sync_diag_counter = 0
        self._onnx_timestep_offset = 0
        self._onnx_timestep_offset_aligned = False
        self._tracking_viz_pub_tick = 0

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def _get_raw_policy_motion_timestep(self) -> int:
        step = int(self.motion_timestep)
        if self._uses_motion_command and self._time_step_input_name is not None:
            step += int(self._onnx_timestep_offset)
        return max(0, step)

    def _get_policy_motion_timestep(self) -> int:
        step = self._get_raw_policy_motion_timestep()
        if not self._onnx_unclamp_time_step and self._motion_data is not None and self._motion_data.frame_count > 0:
            step = min(step, self._motion_data.frame_count - 1)
        return step

    def _get_motion_index(self) -> int:
        if self._onnx_offset_applies_to_motion_index:
            return self._get_policy_motion_timestep()
        step = max(0, int(self.motion_timestep))
        if self._motion_data is not None and self._motion_data.frame_count > 0:
            step = min(step, self._motion_data.frame_count - 1)
        return step

    def _maybe_update_motion_alignment(self, robot_state_data) -> None:
        if not self._motion_alignment_enabled or self._motion_data is None:
            return
        if self._motion_align_quat_wxyz is not None:
            return
        motion_root_quat_wxyz = self._motion_data.root_quat_w[:1]
        motion_yaw = self._quat_yaw(motion_root_quat_wxyz)
        robot_yaw = self._quat_yaw(robot_state_data[:, 3:7])
        yaw_delta = robot_yaw - motion_yaw
        align_quat = rpy_to_quat((0.0, 0.0, yaw_delta)).reshape(1, 4).astype(np.float32)
        motion_root_pos = self._motion_data.root_pos_w[:1]
        aligned_root_pos = quat_apply(align_quat, motion_root_pos)
        robot_root_pos = robot_state_data[:, :3]
        self._motion_align_quat_wxyz = align_quat
        self._motion_align_pos = robot_root_pos - aligned_root_pos

    def _apply_motion_alignment_pos(self, pos: np.ndarray) -> np.ndarray:
        if self._motion_align_quat_wxyz is None or self._motion_align_pos is None:
            return pos
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)
        aligned = quat_apply(self._motion_align_quat_wxyz, pos)
        return aligned + self._motion_align_pos

    def _apply_motion_alignment_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        if self._motion_align_quat_wxyz is None:
            return quat_wxyz
        if quat_wxyz.ndim == 1:
            quat_wxyz = quat_wxyz.reshape(1, -1)
        return quat_mul(self._motion_align_quat_wxyz, quat_wxyz)

    def _calc_heading_quat_inv(self, quat_wxyz: np.ndarray) -> np.ndarray:
        yaw = self._quat_yaw(quat_wxyz)
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw)).reshape(1, 4)
        return yaw_quat.astype(np.float32)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _get_joystick_goal_obs(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_joystick:
            return np.zeros((1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        lin_cmd = np.clip(self.lin_vel_command[0], -1.0, 1.0)
        yaw_cmd = float(np.clip(self.ang_vel_command[0, 0], -1.0, 1.0))
        torso_xy_rel = (lin_cmd * self._joystick_goal_scale).reshape(1, 2).astype(np.float32, copy=False)
        torso_yaw_rel = np.array([[yaw_cmd * self._joystick_yaw_scale]], dtype=np.float32)
        return torso_xy_rel, torso_yaw_rel

    def _get_videomimic_obs_buffer_dict(self, robot_state_data):
        if self._motion_data is None and not self._joystick_goal_enabled:
            raise ValueError("Motion data is required for VideoMimic observations.")

        if self._joystick_goal_enabled:
            torso_xy_rel, torso_yaw_rel = self._get_joystick_goal_obs()
        else:
            self._maybe_update_motion_alignment(robot_state_data)
            idx = self._get_motion_index()

        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        dof_vel = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        projected_gravity = quat_rotate_inverse(base_quat, np.array([[0.0, 0.0, -1.0]], dtype=np.float32))
        torso_real = np.concatenate(
            [base_ang_vel, projected_gravity, dof_pos, dof_vel, self.last_policy_action], axis=1
        )

        if self._joystick_goal_enabled:
            target_joints = np.zeros((1, self.num_dofs), dtype=np.float32)
            target_root_roll = np.zeros((1, 1), dtype=np.float32)
            target_root_pitch = np.zeros((1, 1), dtype=np.float32)
        else:
            motion_ref_pos_w = self._motion_data.ref_pos_w[idx : idx + 1]
            motion_ref_quat_w = self._motion_data.ref_quat_w[idx : idx + 1]
            motion_root_quat_w = self._motion_data.root_quat_w[idx : idx + 1]
            motion_joint_pos = self._motion_data.joint_pos[idx : idx + 1]

            if self._motion_align_quat_wxyz is not None:
                motion_ref_pos_w = self._apply_motion_alignment_pos(motion_ref_pos_w)
                motion_ref_quat_w = self._apply_motion_alignment_quat(motion_ref_quat_w)
                motion_root_quat_w = self._apply_motion_alignment_quat(motion_root_quat_w)

            robot_ref_pos_w, robot_ref_quat_w = self._get_ref_body_pose_in_world(robot_state_data)
            rel_pos_w = motion_ref_pos_w - robot_ref_pos_w
            heading_inv = self._calc_heading_quat_inv(robot_ref_quat_w)
            rel_pos_b = quat_apply(heading_inv, rel_pos_w)
            torso_xy_rel = rel_pos_b[:, :2]

            target_heading = self._quat_yaw(motion_ref_quat_w)
            robot_heading = self._quat_yaw(robot_ref_quat_w)
            torso_yaw_rel = np.array([[self._normalize_angle(target_heading - robot_heading)]], dtype=np.float32)

            target_joints = motion_joint_pos - self.default_dof_angles
            roll, pitch, _ = quat_to_rpy(motion_root_quat_w.reshape(-1, 4)[0])
            target_root_roll = np.array([[self._normalize_angle(roll)]], dtype=np.float32)
            target_root_pitch = np.array([[self._normalize_angle(pitch)]], dtype=np.float32)

        return {
            "torso_real": torso_real,
            "torso_xy_rel": torso_xy_rel,
            "torso_yaw_rel": torso_yaw_rel,
            "target_joints": target_joints,
            "target_root_roll": target_root_roll,
            "target_root_pitch": target_root_pitch,
        }

    @staticmethod
    def _pose_to_ref_frame(
        ref_pos_w: np.ndarray,
        ref_quat_wxyz: np.ndarray,
        pos_w: np.ndarray,
        quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel_pos_w = pos_w - ref_pos_w
        ref_quat_inv = quat_inverse(ref_quat_wxyz)
        pos_b = quat_apply(ref_quat_inv, rel_pos_w).astype(np.float32, copy=False)
        ori_b = quat_mul(ref_quat_inv, quat_wxyz).astype(np.float32, copy=False)
        ori_mat = matrix_from_quat(ori_b)
        ori_6d = ori_mat[..., :2].reshape(1, -1).astype(np.float32, copy=False)
        return pos_b, ori_6d

    def _get_live_object_state(self) -> dict[str, np.ndarray] | None:
        if self._object_state_sub is None:
            return None
        payload = self._object_state_sub.get_state()
        if not isinstance(payload, dict) or not bool(payload.get("has_object", True)):
            return None
        try:
            object_pos_w = np.asarray(payload["object_pos_w"], dtype=np.float32).reshape(1, 3)
            object_quat_wxyz = np.asarray(payload["object_quat_wxyz"], dtype=np.float32).reshape(1, 4)
            object_lin_vel_w = np.asarray(payload.get("object_lin_vel_w", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(
                1, 3
            )
            robot_ref_pos_raw = payload.get("robot_ref_pos_w")
            robot_ref_quat_raw = payload.get("robot_ref_quat_wxyz")
        except Exception:
            return None
        if not self._has_object_state_sample:
            logger.info(
                "Received first object-state sample (sim_time={}).",
                payload.get("sim_time", "n/a"),
            )
        self._has_object_state_sample = True
        robot_ref_pos_w = None
        robot_ref_quat_wxyz = None
        try:
            if robot_ref_pos_raw is not None:
                robot_ref_pos_w = np.asarray(robot_ref_pos_raw, dtype=np.float32).reshape(1, 3)
            if robot_ref_quat_raw is not None:
                robot_ref_quat_wxyz = np.asarray(robot_ref_quat_raw, dtype=np.float32).reshape(1, 4)
        except Exception:
            robot_ref_pos_w = None
            robot_ref_quat_wxyz = None
        return {
            "object_pos_w": object_pos_w,
            "object_quat_wxyz": object_quat_wxyz,
            "object_lin_vel_w": object_lin_vel_w,
            "robot_ref_pos_w": robot_ref_pos_w,
            "robot_ref_quat_wxyz": robot_ref_quat_wxyz,
        }

    def _get_target_object_from_motion(self, robot_state_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero_pos = np.zeros((1, 3), dtype=np.float32)
        zero_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        one_size = np.ones((1, 3), dtype=np.float32)
        if self._motion_data is None or not self._motion_data.has_object:
            return zero_pos, zero_quat, one_size

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()
        assert self._motion_data.object_pos_w is not None
        assert self._motion_data.object_quat_w is not None
        assert self._motion_data.object_size is not None

        target_pos_w = self._motion_data.object_pos_w[idx : idx + 1]
        target_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1]
        target_size = self._motion_data.object_size[idx : idx + 1]
        if self._motion_align_quat_wxyz is not None:
            target_pos_w = self._apply_motion_alignment_pos(target_pos_w)
            target_quat_wxyz = self._apply_motion_alignment_quat(target_quat_wxyz)
        return (
            target_pos_w.astype(np.float32, copy=False),
            target_quat_wxyz.astype(np.float32, copy=False),
            target_size.astype(np.float32, copy=False),
        )

    def get_current_obs_buffer_dict(self, robot_state_data):
        if self._uses_videomimic:
            current_obs_buffer_dict = self._get_videomimic_obs_buffer_dict(robot_state_data)
            if self._motion_future_target_pose_provider is not None:
                current_obs_buffer_dict["motion_future_target_poses"] = (
                    self._motion_future_target_pose_provider.get_future_target_poses(self.motion_timestep)
                )
            return current_obs_buffer_dict

        current_obs_buffer_dict = {}
        robot_ref_pos_w, robot_ref_quat_w = self._get_ref_body_pose_in_world(robot_state_data)
        object_state = None
        if self._object_state_sub is not None and (self._uses_object_obs or self._prefer_sim_ref_from_object_state):
            object_state = self._get_live_object_state()
            if object_state is not None:
                sim_ref_pos_w = object_state.get("robot_ref_pos_w")
                sim_ref_quat_wxyz = object_state.get("robot_ref_quat_wxyz")
                if (
                    self._prefer_sim_ref_from_object_state
                    and sim_ref_pos_w is not None
                    and sim_ref_quat_wxyz is not None
                ):
                    # In MuJoCo sim2sim, prefer simulator-measured ref body frame to match training-frame conventions.
                    robot_ref_pos_w = sim_ref_pos_w
                    robot_ref_quat_w = sim_ref_quat_wxyz

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = robot_ref_quat_w.copy()  # wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        if self._uses_object_obs:
            if object_state is None and self._object_state_sub is not None:
                object_state = self._get_live_object_state()
                if object_state is not None:
                    sim_ref_pos_w = object_state.get("robot_ref_pos_w")
                    sim_ref_quat_wxyz = object_state.get("robot_ref_quat_wxyz")
                    if (
                        self._prefer_sim_ref_from_object_state
                        and sim_ref_pos_w is not None
                        and sim_ref_quat_wxyz is not None
                    ):
                        robot_ref_pos_w = sim_ref_pos_w
                        robot_ref_quat_w = sim_ref_quat_wxyz
            if object_state is None:
                if not self._object_state_missing_warned:
                    logger.warning("Object-state stream unavailable; object observation terms will be zero-filled.")
                    self._object_state_missing_warned = True
                object_pos_b = np.zeros((1, 3), dtype=np.float32)
                object_ori_6d = np.zeros((1, 6), dtype=np.float32)
                object_lin_vel_b = np.zeros((1, 3), dtype=np.float32)
            else:
                object_pos_w = object_state["object_pos_w"]
                object_quat_wxyz = object_state["object_quat_wxyz"]
                object_lin_vel_w = object_state["object_lin_vel_w"]
                object_pos_b, object_ori_6d = self._pose_to_ref_frame(
                    robot_ref_pos_w, robot_ref_quat_w, object_pos_w, object_quat_wxyz
                )
                object_lin_vel_b = quat_apply(quat_inverse(robot_ref_quat_w), object_lin_vel_w).astype(
                    np.float32, copy=False
                )
                self._object_state_missing_warned = False

            target_pos_w, target_quat_wxyz, target_size = self._get_target_object_from_motion(robot_state_data)
            target_pos_b, target_ori_6d = self._pose_to_ref_frame(
                robot_ref_pos_w, robot_ref_quat_w, target_pos_w, target_quat_wxyz
            )
            target_pose_size_b = np.concatenate([target_pos_b, target_ori_6d, target_size], axis=1).astype(
                np.float32, copy=False
            )

            if (
                object_state is not None
                and not self._object_sync_diag_logged
                and object_pos_w.shape == target_pos_w.shape
                and object_quat_wxyz.shape == target_quat_wxyz.shape
            ):
                def _quat_angle_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
                    a = np.asarray(q_a, dtype=np.float64).reshape(-1)
                    b = np.asarray(q_b, dtype=np.float64).reshape(-1)
                    na = float(np.linalg.norm(a))
                    nb = float(np.linalg.norm(b))
                    if na < 1e-12 or nb < 1e-12:
                        return float("nan")
                    a = a / na
                    b = b / nb
                    dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
                    return float(2.0 * np.degrees(np.arccos(dot)))

                quat_alt_xyzw_to_wxyz = np.array(
                    [
                        target_quat_wxyz[0, 3],
                        target_quat_wxyz[0, 0],
                        target_quat_wxyz[0, 1],
                        target_quat_wxyz[0, 2],
                    ],
                    dtype=np.float32,
                ).reshape(1, 4)
                world_pos_err = float(np.linalg.norm((object_pos_w - target_pos_w).reshape(-1)))
                quat_err_wxyz = _quat_angle_error_deg(object_quat_wxyz[0], target_quat_wxyz[0])
                quat_err_alt = _quat_angle_error_deg(object_quat_wxyz[0], quat_alt_xyzw_to_wxyz[0])
                logger.info(
                    "Object sync diag (first sample): motion_timestep={} world_pos_err={:.4f} m, "
                    "quat_err_if_motion_wxyz={:.2f} deg, quat_err_if_motion_xyzw={:.2f} deg, "
                    "live_quat_wxyz={}, target_quat_raw={}",
                    int(self.motion_timestep),
                    world_pos_err,
                    quat_err_wxyz,
                    quat_err_alt,
                    np.asarray(object_quat_wxyz[0], dtype=np.float64).tolist(),
                    np.asarray(target_quat_wxyz[0], dtype=np.float64).tolist(),
                )
                self._object_sync_diag_logged = True

            if object_state is not None:
                self._object_sync_diag_counter += 1
                if self._object_sync_diag_counter % self._object_sync_diag_every_ticks == 0:
                    def _quat_angle_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
                        a = np.asarray(q_a, dtype=np.float64).reshape(-1)
                        b = np.asarray(q_b, dtype=np.float64).reshape(-1)
                        na = float(np.linalg.norm(a))
                        nb = float(np.linalg.norm(b))
                        if na < 1e-12 or nb < 1e-12:
                            return float("nan")
                        a = a / na
                        b = b / nb
                        dot = float(np.clip(np.abs(np.dot(a, b)), -1.0, 1.0))
                        return float(2.0 * np.degrees(np.arccos(dot)))

                    world_pos_err_live = float(np.linalg.norm((object_pos_w - target_pos_w).reshape(-1)))
                    world_quat_err_live = _quat_angle_error_deg(object_quat_wxyz[0], target_quat_wxyz[0])
                    obj_pos_b_err = float(np.linalg.norm((object_pos_b - target_pos_b).reshape(-1)))
                    onnx_timestep_raw = int(self._get_raw_policy_motion_timestep())
                    onnx_timestep = int(self._get_policy_motion_timestep())
                    motion_idx = int(self._get_motion_index())
                    logger.info(
                        "Object tracking status: motion_timestep={}, onnx_timestep_raw={}, onnx_timestep={}, "
                        "motion_idx={}, world_pos_err={:.4f} m, world_quat_err={:.2f} deg, obj_pos_b_err={:.4f} m",
                        int(self.motion_timestep),
                        onnx_timestep_raw,
                        onnx_timestep,
                        motion_idx,
                        world_pos_err_live,
                        world_quat_err_live,
                        obj_pos_b_err,
                    )

            current_obs_buffer_dict["obj_pos_b"] = object_pos_b
            current_obs_buffer_dict["obj_ori_b"] = object_ori_6d
            current_obs_buffer_dict["obj_lin_vel_b"] = object_lin_vel_b
            current_obs_buffer_dict["obj_target_pose_size_b"] = target_pose_size_b

        if self._motion_future_target_pose_provider is not None:
            current_obs_buffer_dict["motion_future_target_poses"] = (
                self._motion_future_target_pose_provider.get_future_target_poses(self._get_policy_motion_timestep())
            )

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)
        self._publish_tracking_viz_payload(actor_obs=obs.get("actor_obs"))
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._time_step_input_name:
            onnx_timestep = int(self._get_policy_motion_timestep())
            input_feed[self._time_step_input_name] = np.array([[onnx_timestep]], dtype=np.float32)
        outputs = self.policy(input_feed)
        policy_action = outputs[self._action_output_name]

        if self._uses_motion_command:
            joint_pos = outputs.get("joint_pos")
            joint_vel = outputs.get("joint_vel")
            if joint_pos is None or joint_vel is None:
                raise ValueError("Motion outputs missing during inference.")
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.ref_quat_xyzw_t = outputs.get("ref_quat_xyzw", self.ref_quat_xyzw_t)
            self.ref_pos_xyz_t = outputs.get("ref_pos_xyz", self.ref_pos_xyz_t)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        if self._policy_action_scale_arr is not None:
            self.scaled_policy_action = policy_action * self._policy_action_scale_arr
        else:
            self.scaled_policy_action = policy_action * self.policy_action_scale

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1
        return self.scaled_policy_action

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        if not self._onnx_timestep_offset_aligned:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_align_onnx_timestep_offset_to_current_pose(robot_state_data)
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        if self._motion_alignment_enabled:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(robot_state_data)

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
            offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
            self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
            self.motion_start_timestep = current_clock - offset_ms
        self._last_clock_reading = current_clock
        elapsed_ms = current_clock - self.motion_start_timestep
        if self.motion_timestep == 0 and int(elapsed_ms // self.timestep_interval_ms) > 1:
            self.logger.warning(
                "Still at the beginning but the clock jumped ahead: elapsed_ms={elapsed_ms}, self.timestep_interval_ms="
                "{timestep_interval_ms}, self.motion_timestep={motion_timestep}. "
                "Re-anchoring to the current timestamp so the motion always starts from frame 0.",
                elapsed_ms=elapsed_ms,
                timestep_interval_ms=self.timestep_interval_ms,
                motion_timestep=self.motion_timestep,
            )
            # Still at the beginning but the clock jumped ahead (e.g., due to waiting before start).
            # Re-anchor to the current timestamp so the motion always starts from frame 0.
            self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            self.motion_timestep = 0
            return
        previous_motion_timestep = self.motion_timestep
        self.motion_timestep = int(elapsed_ms // self.timestep_interval_ms)
        if self.motion_timestep != previous_motion_timestep:
            self.logger.info(
                "Motion timestep advanced from {previous_motion_timestep} to {motion_timestep}",
                previous_motion_timestep=previous_motion_timestep,
                motion_timestep=self.motion_timestep,
            )

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._has_object_state_sample = False
        self._auto_start_wait_for_object_state = False
        self._object_sync_diag_logged = False

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        if self._motion_alignment_enabled:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(robot_state_data)
        self.logger.info(colored("Starting motion clip", "blue"))

    def _should_auto_start_policy_immediately(self) -> bool:
        # For WBT auto-start, we want stiff-hold first, then policy+clip.
        return not bool(getattr(self.config.task, "auto_start_motion_clip", False))

    def _set_stiff_hold_target_for_autostart(self) -> None:
        state = self.interface.get_low_state()
        self._maybe_align_onnx_timestep_offset_to_current_pose(state)

        # Prefer current simulator state so stiff-hold keeps the exact initialized start pose.
        if state is not None and state.shape[1] >= 7 + self.num_dofs:
            target_q = np.asarray(state[:, 7 : 7 + self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self.logger.info("Auto-start stiff target locked to current initialized robot pose.")
                return

        # Fallback: use motion-start command.
        if self.motion_command_0 is not None:
            target_q = np.asarray(self.motion_command_0[:, : self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self.logger.warning("Auto-start stiff target fallback to ONNX motion start command.")

    def _stiff_hold_pose_error(self) -> float | None:
        if self.last_robot_state_data is None:
            return None
        dof_pos = self.last_robot_state_data[:, 7 : 7 + self.num_dofs]
        if dof_pos.shape != self._stiff_hold_q.shape:
            return None
        return float(np.max(np.abs(dof_pos - self._stiff_hold_q)))

    def _maybe_auto_start_rollout(self):
        if not getattr(self.config.task, "auto_start_motion_clip", False):
            return
        # Stage 1: hold stiff at motion-start pose before enabling policy rollout.
        self._set_stiff_hold_target_for_autostart()
        self._stiff_hold_active = True
        self.use_policy_action = False
        self.get_ready_state = False
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        hold_sec = max(0.0, float(getattr(self.config.task, "auto_start_stiff_hold_sec", 0.0)))
        max_wait_sec = max(hold_sec, float(getattr(self.config.task, "auto_start_stiff_max_wait_sec", hold_sec)))
        self._auto_start_hold_ticks = max(1, int(round(hold_sec * self.rl_rate)))
        self._auto_start_max_wait_ticks = max(self._auto_start_hold_ticks, int(round(max_wait_sec * self.rl_rate)))
        self._auto_start_tick_count = 0
        self._auto_start_stage = "stiff_hold"
        self._auto_start_wait_for_object_state = bool(self._uses_object_obs and self._object_state_sub is not None)
        self.logger.info(
            "Auto-start enabled: stiff-hold min {:.2f}s ({} ticks), max {:.2f}s ({} ticks), then start policy + clip.",
            hold_sec,
            self._auto_start_hold_ticks,
            max_wait_sec,
            self._auto_start_max_wait_ticks,
        )

    def _maybe_advance_auto_start_state(self) -> None:
        if self._auto_start_stage != "stiff_hold":
            return
        self._auto_start_tick_count += 1
        err = self._stiff_hold_pose_error()
        at_target = err is not None and err <= self._auto_start_pose_tolerance
        min_hold_done = self._auto_start_tick_count >= self._auto_start_hold_ticks
        max_wait_reached = self._auto_start_tick_count >= self._auto_start_max_wait_ticks
        if self._auto_start_wait_for_object_state and not self._has_object_state_sample:
            # Poll subscriber during stiff-hold so we can gate rollout on the first object-state sample.
            self._get_live_object_state()
        progress_every = max(1, int(self.rl_rate // 2))
        if self._auto_start_tick_count % progress_every == 0:
            logger.info(
                "Auto-start stiff-hold progress: tick {}/{}, max_wait={}, max_dof_err={}",
                self._auto_start_tick_count,
                self._auto_start_hold_ticks,
                self._auto_start_max_wait_ticks,
                "n/a" if err is None else f"{err:.4f}",
            )
        if not min_hold_done:
            return

        if self._auto_start_wait_for_object_state and not self._has_object_state_sample:
            if not max_wait_reached:
                if self._auto_start_tick_count % progress_every == 0:
                    self.logger.info("Auto-start waiting for first object-state sample before rollout.")
                return
            self.logger.warning(
                "Auto-start object-state wait timed out at {} ticks; proceeding without object-state sample.",
                self._auto_start_tick_count,
            )

        if at_target:
            self.logger.info(
                "Auto-start stiff-hold reached target (max_dof_err={:.4f} rad <= {:.4f} rad).",
                err,
                self._auto_start_pose_tolerance,
            )
        elif not max_wait_reached:
            return
        else:
            self.logger.warning(
                "Auto-start stiff-hold max wait reached ({} ticks, max_dof_err={}); proceeding anyway.",
                self._auto_start_tick_count,
                "n/a" if err is None else f"{err:.4f}",
            )

        self._auto_start_stage = None
        self._handle_start_policy()
        self._handle_start_motion_clip()

    def policy_action(self):
        super().policy_action()
        if not getattr(self, "_last_policy_cycle_has_state", False):
            return
        self._maybe_advance_auto_start_state()

    def _on_run_exit(self) -> None:
        self._close_tracking_viz_publisher()
        if self._object_state_sub is not None:
            try:
                self._object_state_sub.close()
            except Exception:
                pass
            self._object_state_sub = None
        try:
            self.clock_sub.close()
        except Exception:
            pass

    def handle_keyboard_button(self, keycode):
        """Add new keyboard button to start and end the motion clips"""
        if keycode == "s":
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        else:
            super().handle_keyboard_button(keycode)

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses for WBT-specific controls."""
        if cur_key == "start":
            # Start playing motion clip
            self._handle_start_motion_clip()
        else:
            # Delegate all other buttons to base class
            super().handle_joystick_button(cur_key)
        super()._print_control_status()

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
