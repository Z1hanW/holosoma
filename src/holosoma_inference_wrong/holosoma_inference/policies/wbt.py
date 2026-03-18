import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
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
from holosoma_inference.utils.perception_obs import PerceptionObsSub
from holosoma_inference.utils.sim_control import SimControlPush
from holosoma_inference.utils.sim_state import SimStateSub


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
    _OBJECT_SIZE_KEYS = (
        "object_size",
        "box_size",
        "object_scale",
        "box_scale",
    )

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
            self.object_size = self._extract_object_size_np(data, joint_pos.shape[0], source=str(motion_path))

        joint_indices = get_index_of_a_in_b(robot_dof_names, joint_names)
        self.joint_pos = joint_pos[:, joint_indices]
        self.joint_vel = joint_vel[:, joint_indices]
        self.frame_count = self.joint_pos.shape[0]

        if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
            raise ValueError(f"Unexpected body_quat_w shape {body_quat_w.shape} in {motion_path}")

        self.ref_body_index = body_names.index(body_name_ref)
        self.ref_pos_w = body_pos_w[:, self.ref_body_index, :]
        self.ref_quat_w = body_quat_w[:, self.ref_body_index, :]
        self.root_body_index = self._resolve_root_body_index(body_names)
        self.root_quat_w = body_quat_w[:, self.root_body_index, :]
        self.root_pos_w = body_pos_w[:, self.root_body_index, :]
        self.has_object = object_pos_w is not None and object_quat_w is not None
        self.object_pos_w = object_pos_w
        self.object_quat_w = object_quat_w

    @classmethod
    def _normalize_object_size_array(cls, raw: np.ndarray, length: int, *, source: str) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 0:
            return np.full((length, 3), float(arr), dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] == 1:
                return np.full((length, 3), float(arr[0]), dtype=np.float32)
            if arr.shape[0] == 3:
                return np.repeat(arr.reshape(1, 3), repeats=length, axis=0)
            if arr.shape[0] == length:
                return np.repeat(arr.reshape(length, 1), repeats=3, axis=1)
        if arr.ndim == 2:
            if arr.shape == (1, 3):
                return np.repeat(arr, repeats=length, axis=0)
            if arr.shape == (length, 1):
                return np.repeat(arr, repeats=3, axis=1)
            if arr.shape == (length, 3):
                return arr
        raise ValueError(
            f"Unsupported object-size shape {arr.shape} in {source}; "
            "expected scalar, (3,), (T,), (T,3), (1,3), or (T,1)."
        )

    @classmethod
    def _extract_object_size_np(cls, data: dict, length: int, *, source: str) -> np.ndarray:
        for key in cls._OBJECT_SIZE_KEYS:
            if key in data:
                raw = np.asarray(data[key], dtype=np.float32)
                return cls._normalize_object_size_array(raw, length, source=f"{source}:{key}")
        return np.ones((length, 3), dtype=np.float32)

    @staticmethod
    def _resolve_root_body_index(body_names: list[str]) -> int:
        for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
            if candidate in body_names:
                return body_names.index(candidate)
        for idx, name in enumerate(body_names):
            if name.lower() != "world":
                return idx
        return 0

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
            if "object_pos" not in start_state or "object_pos" not in target_state:
                raise KeyError("Object motion transition requested without object_pos in transition state.")
            if "object_quat" not in start_state or "object_quat" not in target_state:
                raise KeyError("Object motion transition requested without object_quat in transition state.")
            if "object_size" not in start_state or "object_size" not in target_state:
                raise KeyError("Object motion transition requested without object_size in transition state.")
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
                assert segment_object_pos is not None and segment_object_quat is not None and segment_object_size is not None
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
                assert segment_object_pos is not None and segment_object_quat is not None and segment_object_size is not None
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
        self.body_names, self.body_pos_w, self.body_quat_w = self._load_motion_npz(motion_file)
        self.time_step_total = int(self.body_pos_w.shape[0])
        self.tracked_body_indexes = self._resolve_body_indexes(self.body_names, body_names_to_track)
        self.num_bodies = len(self.tracked_body_indexes)
        self.obs_dim = self.num_future_steps * (self.num_bodies * 18 + (1 if self.include_time else 0))

    @staticmethod
    def _resolve_body_indexes(body_names: list[str], tracked_names: list[str]) -> list[int]:
        indexes = []
        for name in tracked_names:
            if name not in body_names:
                raise ValueError(f"Body name '{name}' not found in motion data")
            indexes.append(body_names.index(name))
        return indexes

    @staticmethod
    def _load_motion_npz(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        motion_path = Path(path)
        if not motion_path.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
        with np.load(motion_path, allow_pickle=True) as data:
            body_names = data["body_names"].tolist()
            body_names = [bn.decode("utf-8") if isinstance(bn, (bytes, bytearray)) else bn for bn in body_names]
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_quat_w = body_quat_w[:, :, [1, 2, 3, 0]]  # wxyz -> xyzw
        return body_names, body_pos_w, body_quat_w

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
        return obs.reshape(1, -1).astype(np.float32, copy=False)


class _ZeroFutureTargetPoseProvider:
    def __init__(self, obs_dim: int) -> None:
        self.obs_dim = int(obs_dim)

    def get_future_target_poses(self, time_step: int) -> np.ndarray:  # noqa: ARG002 - signature match
        return np.zeros((1, self.obs_dim), dtype=np.float32)


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
        self.clock_sub = ClockSub(port=config.task.sim_clock_port)
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
        self._latest_sim_state: dict | None = None
        self._sim_state_sub: SimStateSub | None = None
        self._latest_sim_perception: np.ndarray | None = None
        self._sim_perception_sub: PerceptionObsSub | None = None
        self._warned_missing_sim_perception = False
        self._logged_sim_perception_receive = False
        self._sim_perception_msg_count = 0
        self._last_sim_perception_time_ms: int | None = None
        self._last_logged_sim_perception_count = 0

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
        self._uses_object_distill = any(
            term in obs_terms
            for term in (
                "sparse_target_root_trajectory_command",
                "obj_current_pose_size_b",
                "obj_goal_pose_size_b",
            )
        )
        self._uses_object_generalist = any(
            term in obs_terms
            for term in (
                "obj_target_pose_size_b",
                "obj_pos_b",
                "obj_ori_b",
                "obj_lin_vel_b",
                "obj_ang_vel_b",
            )
        ) and not self._uses_object_distill
        self._uses_legacy_object_obs = (
            all(term in obs_terms for term in ("obj_target_pose_size_b", "obj_pos_b", "obj_ori_b"))
            and "obj_lin_vel_b" not in obs_terms
            and "obj_ang_vel_b" not in obs_terms
            and not self._uses_object_distill
        )
        self._motion_data: MotionData | None = None
        self._motion_cfg: dict | None = None
        self._motion_dof_names: list[str] | None = None
        self._motion_align_quat_wxyz: np.ndarray | None = None
        self._motion_align_pos: np.ndarray | None = None
        self._obs_input_name: str | None = None
        self._time_step_input_name: str | None = None
        self._perception_input_name: str | None = None
        self._perception_obs_dim: int | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._motion_output_names: set[str] = set()
        self._motion_alignment_enabled = False
        self._pending_motion_restart_after_policy_start = False
        self._reset_restart_ready_at = 0.0
        self._awaiting_sim_reset_time_jump = False
        self._pre_reset_sim_time_ms: int | None = None
        self._sim_control_pub: SimControlPush | None = None
        self._onnx_timestep_offset: int = 0
        self._onnx_timestep_offset_aligned: bool = False
        self._onnx_timestep_search_max_steps: int = max(0, int(os.getenv("HOLOSOMA_ONNX_ALIGN_MAX_STEPS", "0")))
        self._onnx_timestep_align_pose_tolerance: float = float(os.getenv("HOLOSOMA_ONNX_ALIGN_POSE_TOL", "5e-3"))
        self._onnx_unclamp_time_step: bool = str(os.getenv("HOLOSOMA_ONNX_UNCLAMP_TIME_STEP", "0")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._onnx_offset_applies_to_motion_index: bool = str(
            os.getenv("HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX", "1")
        ).lower() in {"1", "true", "yes", "on"}
        self._last_motion_output_timestep: int | None = None

        self._joystick_goal_enabled = bool(config.task.use_joystick_goal)
        self._joystick_goal_scale = float(config.task.joystick_goal_scale)
        self._joystick_yaw_scale = float(config.task.joystick_yaw_scale)
        self._auto_start_stage: str | None = None
        self._auto_start_hold_ticks = 0
        self._auto_start_max_wait_ticks = 0
        self._auto_start_tick_count = 0
        self._auto_start_pose_tolerance = float(getattr(config.task, "auto_start_stiff_pose_tolerance", 0.12))
        self._logged_root_reference_clip_start = False
        self._logged_sim_ref_from_sim_state = False
        self._logged_object_contact_details = False

        super().__init__(config)

        if self.config.task.use_sim_state:
            self._sim_state_sub = SimStateSub(port=self.config.task.sim_state_port)
            self._sim_state_sub.start()
            self._sim_control_pub = SimControlPush(port=self.config.task.sim_control_port)
            self._sim_control_pub.start()
        if self.config.task.use_sim_perception:
            self._sim_perception_sub = PerceptionObsSub(port=self.config.task.sim_perception_port)
            self._sim_perception_sub.start()

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
        elif not auto_start_enabled:
            _show_warning()

        if self.config.task.auto_start_motion:
            self._handle_start_motion_clip()

    def _get_ref_body_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        if bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False)):
            sim_ref_state = self._get_sim_ref_state()
            if sim_ref_state is not None:
                if not self._logged_sim_ref_from_sim_state:
                    logger.info("Using simulator-measured ref-body pose from split sim-state when available.")
                    self._logged_sim_ref_from_sim_state = True
                return sim_ref_state[:, :3], xyzw_to_wxyz(sim_ref_state[:, 3:7])

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

    def _should_use_root_reference_at_clip_start(self) -> bool:
        if not bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)):
            return False
        use_root = int(self._get_motion_index()) == 0
        if use_root and not self._logged_root_reference_clip_start:
            logger.info("Using robot root as observation reference at clip start to match training step-0 semantics.")
            self._logged_root_reference_clip_start = True
        return use_root

    def _get_observation_reference_pose_in_world(self, robot_state_data) -> tuple[np.ndarray, np.ndarray]:
        if self._should_use_root_reference_at_clip_start():
            root_pos = np.asarray(robot_state_data[:, :3], dtype=np.float32)
            root_quat_wxyz = np.asarray(robot_state_data[:, 3:7], dtype=np.float32)
            return root_pos, root_quat_wxyz
        return self._get_ref_body_pose_in_world(robot_state_data)

    def _get_observation_reference_orientation_in_world(self, robot_state_data) -> np.ndarray:
        if self._should_use_root_reference_at_clip_start():
            return np.asarray(robot_state_data[:, 3:7], dtype=np.float32)
        return self._get_ref_body_orientation_in_world(robot_state_data)

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

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])
        if (self._uses_videomimic or self._uses_object_distill or self._uses_object_generalist) and not self._joystick_goal_enabled:
            self._load_motion_data_from_metadata(metadata, model_path)
            if self._uses_videomimic or self.config.task.apply_training_motion_transitions:
                self._apply_default_pose_transitions()
        self._maybe_enable_motion_future_target_poses(metadata, model_path)

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        self._set_policy_action_scales_from_metadata(metadata)

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported ONNX inputs: {self.onnx_input_names}")

        self._time_step_input_name = "time_step" if "time_step" in self.onnx_input_names else None
        self._perception_input_name = "perception_obs" if "perception_obs" in self.onnx_input_names else None
        self._perception_obs_dim = self._get_onnx_input_dim(self._perception_input_name)

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
            if self._perception_input_name:
                input_feed[self._perception_input_name] = self._get_perception_obs_input()
            outputs = self.policy(input_feed)
            joint_pos = outputs["joint_pos"]
            joint_vel = outputs["joint_vel"]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.ref_quat_xyzw_t = outputs["ref_quat_xyzw"]
            self.ref_pos_xyz_t = outputs.get("ref_pos_xyz")
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
        elif (self._uses_videomimic or self._uses_object_distill or self._uses_object_generalist) and self._motion_data is not None:
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
            robot_state_data = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
            if robot_state_data is not None:
                self._maybe_align_onnx_timestep_offset_to_current_pose(robot_state_data)

    def _get_onnx_input_dim(self, input_name: str | None) -> int | None:
        if input_name is None:
            return None
        inputs = self.onnx_policy_session.get_inputs()
        for inp in inputs:
            if inp.name == input_name:
                shape = inp.shape
                if len(shape) > 1 and isinstance(shape[1], int):
                    return int(shape[1])
        return None

    def _get_onnx_obs_dim(self) -> int | None:
        obs_dim = self._get_onnx_input_dim("obs")
        if obs_dim is not None:
            return obs_dim
        obs_dim = self._get_onnx_input_dim("actor_obs")
        if obs_dim is not None:
            return obs_dim
        inputs = self.onnx_policy_session.get_inputs()
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

        input_feed = {self._obs_input_name: self._build_zero_actor_obs()}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[int(onnx_timestep)]], dtype=np.float32)
        if self._perception_input_name:
            input_feed[self._perception_input_name] = self._get_perception_obs_input()

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
            for onnx_timestep in range(1, self._onnx_timestep_search_max_steps + 1):
                query = self._query_motion_outputs_at(onnx_timestep)
                if query is None:
                    break
                err = _err_from_query(query)
                if err < best_err:
                    best_err = err
                    best_t = onnx_timestep
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

    def _load_motion_data_from_metadata(self, metadata: dict, model_path: str | Path) -> None:
        motion_cfg = self._extract_motion_config(metadata)
        if not motion_cfg:
            raise ValueError("Motion config missing from ONNX metadata; cannot build VideoMimic observations.")

        motion_file = self.config.task.motion_file or motion_cfg.get("motion_file")
        if not motion_file:
            raise ValueError("motion_config.motion_file missing from ONNX metadata.")

        motion_path = self._resolve_motion_file(str(motion_file), model_path)
        if motion_path is None:
            raise FileNotFoundError(f"Motion file not found: {motion_file}")
        resolved_motion_path = Path(motion_path)
        if resolved_motion_path.is_dir():
            raise ValueError(
                f"Motion path '{resolved_motion_path}' resolves to a directory. "
                "Pass --task.motion-file with a single .npz clip for split sim2sim inference."
            )

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
        self._motion_data = MotionData(resolved_motion_path, list(robot_dof_names), ref_name)
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
        if self._motion_data.has_object:
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
        if self._motion_data.has_object:
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

    def _maybe_enable_motion_future_target_poses(self, metadata: dict, model_path: str) -> None:
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

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "obs_input_name": self._obs_input_name,
                "time_step_input_name": self._time_step_input_name,
                "perception_input_name": self._perception_input_name,
                "perception_obs_dim": self._perception_obs_dim,
                "action_output_name": self._action_output_name,
                "onnx_output_fetch": list(self._onnx_output_fetch),
                "motion_output_names": set(self._motion_output_names),
                "onnx_metadata": self._onnx_metadata,
                "onnx_obs_dim": self._onnx_obs_dim,
                "motion_data": self._motion_data,
                "motion_cfg": self._motion_cfg,
                "motion_alignment_enabled": self._motion_alignment_enabled,
                "motion_future_target_pose_provider": self._motion_future_target_pose_provider,
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
        self._perception_input_name = state.get("perception_input_name")
        self._perception_obs_dim = state.get("perception_obs_dim")
        self._action_output_name = state.get("action_output_name")
        self._onnx_output_fetch = list(state.get("onnx_output_fetch", []))
        self._motion_output_names = set(state.get("motion_output_names", set()))
        self._onnx_metadata = state.get("onnx_metadata")
        self._onnx_obs_dim = state.get("onnx_obs_dim")
        self._motion_data = state.get("motion_data")
        self._motion_cfg = state.get("motion_cfg")
        self._motion_alignment_enabled = bool(state.get("motion_alignment_enabled", False))
        self._motion_future_target_pose_provider = state.get("motion_future_target_pose_provider")
        self._onnx_timestep_offset = int(state.get("onnx_timestep_offset", 0))
        self._onnx_timestep_offset_aligned = bool(state.get("onnx_timestep_offset_aligned", False))
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._logged_sim_ref_from_sim_state = False
        self._logged_object_contact_details = False

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
        self._onnx_timestep_offset = 0
        self._onnx_timestep_offset_aligned = False
        self._logged_sim_ref_from_sim_state = False
        self._logged_object_contact_details = False

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
        if self._motion_data is None:
            return 0
        if self._onnx_offset_applies_to_motion_index:
            return self._get_policy_motion_timestep()
        idx = max(0, int(self.motion_timestep))
        return min(idx, self._motion_data.frame_count - 1)

    def _refresh_motion_outputs_for_current_timestep(self) -> None:
        if not self._uses_motion_command or self._time_step_input_name is None:
            return

        motion_step = self._get_policy_motion_timestep()
        if self._last_motion_output_timestep == motion_step and self.motion_command_t is not None:
            return

        query = self._query_motion_outputs_at(motion_step)
        if query is None:
            return

        joint_pos = query.get("joint_pos")
        joint_vel = query.get("joint_vel")
        if joint_pos is None or joint_vel is None:
            return

        self.motion_command_t = np.concatenate(
            [np.asarray(joint_pos, dtype=np.float32), np.asarray(joint_vel, dtype=np.float32)],
            axis=1,
        )

        ref_quat_xyzw = query.get("ref_quat_xyzw")
        if ref_quat_xyzw is not None:
            self.ref_quat_xyzw_t = np.asarray(ref_quat_xyzw, dtype=np.float32)

        ref_pos_xyz = query.get("ref_pos_xyz")
        if ref_pos_xyz is not None:
            self.ref_pos_xyz_t = np.asarray(ref_pos_xyz, dtype=np.float32)

        self._last_motion_output_timestep = int(motion_step)

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

            robot_ref_pos_w, robot_ref_quat_w = self._get_observation_reference_pose_in_world(robot_state_data)
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

    def _get_latest_sim_state(self) -> dict | None:
        if self._sim_state_sub is None:
            return self._latest_sim_state
        state = self._sim_state_sub.get_state()
        if state is not None:
            if not self._logged_object_contact_details:
                contact_bodies = state.get("object_robot_contact_bodies")
                contact_geoms = state.get("object_robot_contact_geoms")
                if contact_bodies or contact_geoms:
                    logger.info(
                        "Received object contact details from split sim-state: bodies={} geoms={}",
                        contact_bodies or [],
                        contact_geoms or [],
                    )
                    self._logged_object_contact_details = True
            self._latest_sim_state = state
        return self._latest_sim_state

    def _has_valid_robot_state(self, robot_state_data) -> bool:
        if self.config.task.use_sim_state and self._get_latest_sim_state() is None:
            return False
        return super()._has_valid_robot_state(robot_state_data)

    def _get_sim_root_state(self) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        root_state = state.get("robot_root_state")
        if root_state is None:
            return None
        root_state_np = np.asarray(root_state, dtype=np.float32).reshape(1, -1)
        if root_state_np.shape[1] < 13:
            return None
        return root_state_np[:, :13]

    def _get_sim_ref_state(self) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        ref_state = state.get("robot_ref_state")
        if ref_state is None:
            return None
        ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(1, -1)
        if ref_state_np.shape[1] < 13:
            return None
        return ref_state_np[:, :13]

    def _get_sim_actor_state(self, actor_name: str) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        actors = state.get("actors")
        if not isinstance(actors, dict) or not actors:
            return None
        actor_state = actors.get(actor_name)
        if actor_state is None and len(actors) == 1:
            actor_state = next(iter(actors.values()))
        if actor_state is None:
            return None
        actor_state_np = np.asarray(actor_state, dtype=np.float32).reshape(1, -1)
        if actor_state_np.shape[1] < 13:
            return None
        return actor_state_np[:, :13]

    def _get_latest_sim_time_ms(self) -> int | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        sim_time_ms = state.get("sim_time_ms")
        if sim_time_ms is None:
            return None
        try:
            return int(sim_time_ms)
        except (TypeError, ValueError):
            return None

    def _get_latest_sim_perception_payload(self) -> dict | None:
        if self._sim_perception_sub is None:
            return None
        payload = self._sim_perception_sub.get_payload()
        if payload is not None:
            perception_obs = payload.get("perception_obs")
            if perception_obs is not None:
                perception_arr = np.asarray(perception_obs, dtype=np.float32).reshape(1, -1)
                if self._perception_obs_dim is not None and perception_arr.shape[1] != self._perception_obs_dim:
                    raise ValueError(
                        "Perception observation dimension mismatch: "
                        f"expected {self._perception_obs_dim}, got {perception_arr.shape[1]}"
                    )
                self._latest_sim_perception = perception_arr
                sim_time_ms = payload.get("sim_time_ms")
                if sim_time_ms != self._last_sim_perception_time_ms:
                    self._sim_perception_msg_count += 1
                    self._last_sim_perception_time_ms = sim_time_ms
                if not self._logged_sim_perception_receive:
                    logger.info(
                        "Received split sim perception obs: dim={} sim_time_ms={}",
                        perception_arr.shape[1],
                        sim_time_ms,
                    )
                    self._logged_sim_perception_receive = True
                elif (
                    self._sim_perception_msg_count % 25 == 0
                    and self._sim_perception_msg_count != self._last_logged_sim_perception_count
                    and sim_time_ms is not None
                ):
                    logger.info(
                        "Received {} split sim perception obs messages; latest sim_time_ms={}",
                        self._sim_perception_msg_count,
                        sim_time_ms,
                    )
                    self._last_logged_sim_perception_count = self._sim_perception_msg_count
        return payload

    def _get_perception_obs_input(self) -> np.ndarray:
        self._get_latest_sim_perception_payload()
        if self._latest_sim_perception is not None:
            return self._latest_sim_perception.astype(np.float32, copy=False)

        dim = self._perception_obs_dim
        if dim is None:
            dim = self.obs_dims.get("perception")
        if dim is None:
            raise ValueError("Perception observations requested but perception dimension is unknown.")

        if not self._warned_missing_sim_perception and self._sim_perception_sub is not None:
            logger.warning("No split sim perception observation received yet; using zeros until the first frame arrives.")
            self._warned_missing_sim_perception = True
        return np.zeros((1, int(dim)), dtype=np.float32)

    def _augment_robot_state_with_sim_state(self, robot_state_data: np.ndarray | None) -> np.ndarray | None:
        if robot_state_data is None:
            return None
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is None:
            return robot_state_data
        augmented = np.array(robot_state_data, dtype=np.float32, copy=True)
        augmented[:, :3] = sim_root_state[:, :3]
        augmented[:, 3:7] = xyzw_to_wxyz(sim_root_state[:, 3:7])
        return augmented

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

    def _get_live_object_state_from_sim_state(self) -> dict[str, np.ndarray] | None:
        actor_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        if actor_state is None:
            return None

        object_pos_w = actor_state[:, :3].astype(np.float32, copy=False)
        object_quat_wxyz = xyzw_to_wxyz(actor_state[:, 3:7]).astype(np.float32, copy=False)
        object_lin_vel_w = actor_state[:, 7:10].astype(np.float32, copy=False)
        sim_ref_state = self._get_sim_ref_state()
        robot_ref_pos_w = None
        robot_ref_quat_wxyz = None
        if sim_ref_state is not None:
            robot_ref_pos_w = sim_ref_state[:, :3].astype(np.float32, copy=False)
            robot_ref_quat_wxyz = xyzw_to_wxyz(sim_ref_state[:, 3:7]).astype(np.float32, copy=False)

        return {
            "object_pos_w": object_pos_w,
            "object_quat_wxyz": object_quat_wxyz,
            "object_lin_vel_w": object_lin_vel_w,
            "robot_ref_pos_w": robot_ref_pos_w,
            "robot_ref_quat_wxyz": robot_ref_quat_wxyz,
        }

    def _pose_in_robot_ref_frame(
        self,
        robot_ref_pos_w: np.ndarray,
        robot_ref_quat_wxyz: np.ndarray,
        target_pos_w: np.ndarray,
        target_quat_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel_pos_w = target_pos_w - robot_ref_pos_w
        rel_pos_b = quat_apply(quat_inverse(robot_ref_quat_wxyz), rel_pos_w)
        rel_quat_b = subtract_frame_transforms(robot_ref_quat_wxyz, target_quat_wxyz)
        return rel_pos_b.astype(np.float32, copy=False), rel_quat_b.astype(np.float32, copy=False)

    def _get_object_distill_obs_buffer_dict(self, robot_state_data: np.ndarray):
        if self._motion_data is None:
            raise ValueError("Motion data is required for object-distill observations.")
        if not self._motion_data.has_object:
            raise ValueError("Object-distill observations require a motion clip with object pose data.")

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()

        motion_root_pos_w = self._motion_data.root_pos_w[idx : idx + 1]
        motion_root_quat_wxyz = self._motion_data.root_quat_w[idx : idx + 1]
        if self._motion_align_quat_wxyz is not None:
            motion_root_pos_w = self._apply_motion_alignment_pos(motion_root_pos_w)
            motion_root_quat_wxyz = self._apply_motion_alignment_quat(motion_root_quat_wxyz)

        robot_root_pos_w = robot_state_data[:, :3]
        robot_root_quat_wxyz = robot_state_data[:, 3:7]
        rel_root_pos_w = motion_root_pos_w - robot_root_pos_w
        heading_inv = self._calc_heading_quat_inv(robot_root_quat_wxyz)
        rel_root_pos_b = quat_apply(heading_inv, rel_root_pos_w)

        target_heading = self._quat_yaw(motion_root_quat_wxyz)
        robot_heading = self._quat_yaw(robot_root_quat_wxyz)
        rel_root_yaw = np.array([[self._normalize_angle(target_heading - robot_heading)]], dtype=np.float32)

        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_observation_reference_pose_in_world(robot_state_data)

        sim_object_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        if sim_object_state is not None:
            current_object_pos_w = sim_object_state[:, :3]
            current_object_quat_wxyz = xyzw_to_wxyz(sim_object_state[:, 3:7])
        else:
            current_object_pos_w = self._motion_data.object_pos_w[idx : idx + 1]
            current_object_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1]

        goal_object_pos_w = self._motion_data.object_pos_w[-1:].copy()
        goal_object_quat_wxyz = self._motion_data.object_quat_w[-1:].copy()
        if self._motion_align_quat_wxyz is not None:
            goal_object_pos_w = self._apply_motion_alignment_pos(goal_object_pos_w)
            goal_object_quat_wxyz = self._apply_motion_alignment_quat(goal_object_quat_wxyz)

        obj_current_pos_b, obj_current_quat_b = self._pose_in_robot_ref_frame(
            robot_ref_pos_w,
            robot_ref_quat_wxyz,
            current_object_pos_w,
            current_object_quat_wxyz,
        )
        obj_goal_pos_b, obj_goal_quat_b = self._pose_in_robot_ref_frame(
            robot_ref_pos_w,
            robot_ref_quat_wxyz,
            goal_object_pos_w,
            goal_object_quat_wxyz,
        )

        obj_current_rot6d = matrix_from_quat(obj_current_quat_b)[..., :2].reshape(1, -1)
        obj_goal_rot6d = matrix_from_quat(obj_goal_quat_b)[..., :2].reshape(1, -1)
        obj_current_size = self._motion_data.object_size[idx : idx + 1].astype(np.float32, copy=False)
        obj_goal_size = self._motion_data.object_size[-1:].astype(np.float32, copy=False)

        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            base_lin_vel = quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 7:10])
            base_ang_vel = quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 10:13])
        else:
            base_lin_vel = np.zeros((1, 3), dtype=np.float32)
            base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        return {
            "sparse_target_root_trajectory_command": np.concatenate(
                [rel_root_pos_b[:, :2], rel_root_yaw], axis=1
            ).astype(np.float32, copy=False),
            "base_lin_vel": base_lin_vel.astype(np.float32, copy=False),
            "base_ang_vel": base_ang_vel.astype(np.float32, copy=False),
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
            "obj_current_pose_size_b": np.concatenate(
                [obj_current_pos_b, obj_current_rot6d, obj_current_size], axis=1
            ).astype(np.float32, copy=False),
            "obj_goal_pose_size_b": np.concatenate(
                [obj_goal_pos_b, obj_goal_rot6d, obj_goal_size], axis=1
            ).astype(np.float32, copy=False),
        }

    def _get_object_generalist_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        if self._motion_data is None:
            raise ValueError("Motion data is required for object-generalist observations.")
        if not self._motion_data.has_object:
            raise ValueError("Object-generalist observations require a motion clip with object pose data.")

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()

        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_observation_reference_pose_in_world(robot_state_data)

        motion_object_pos_w = self._motion_data.object_pos_w[idx : idx + 1].copy()
        motion_object_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1].copy()
        if self._motion_align_quat_wxyz is not None:
            motion_object_pos_w = self._apply_motion_alignment_pos(motion_object_pos_w)
            motion_object_quat_wxyz = self._apply_motion_alignment_quat(motion_object_quat_wxyz)

        sim_object_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        if sim_object_state is not None:
            current_object_pos_w = sim_object_state[:, :3]
            current_object_quat_wxyz = xyzw_to_wxyz(sim_object_state[:, 3:7])
            current_object_lin_vel_w = sim_object_state[:, 7:10]
            current_object_ang_vel_w = sim_object_state[:, 10:13]
        else:
            current_object_pos_w = self._motion_data.object_pos_w[idx : idx + 1]
            current_object_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1]
            current_object_lin_vel_w = np.zeros((1, 3), dtype=np.float32)
            current_object_ang_vel_w = np.zeros((1, 3), dtype=np.float32)

        obj_target_pos_b, obj_target_quat_b = self._pose_in_robot_ref_frame(
            robot_ref_pos_w,
            robot_ref_quat_wxyz,
            motion_object_pos_w,
            motion_object_quat_wxyz,
        )
        obj_pos_b, obj_quat_b = self._pose_in_robot_ref_frame(
            robot_ref_pos_w,
            robot_ref_quat_wxyz,
            current_object_pos_w,
            current_object_quat_wxyz,
        )

        obj_target_rot6d = matrix_from_quat(obj_target_quat_b)[..., :2].reshape(1, -1)
        obj_rot6d = matrix_from_quat(obj_quat_b)[..., :2].reshape(1, -1)
        obj_lin_vel_b = quat_apply(
            quat_inverse(robot_ref_quat_wxyz),
            current_object_lin_vel_w - robot_ref_pos_w,
        )
        obj_ang_vel_b = quat_rotate_inverse(robot_ref_quat_wxyz, current_object_ang_vel_w)
        object_size = self._motion_data.object_size[idx : idx + 1].astype(np.float32, copy=False)

        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            base_ang_vel = quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 10:13])
        else:
            base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        return {
            "motion_command": self.motion_command_t,
            "motion_ref_ori_b": self._get_motion_ref_ori_b(robot_state_data),
            "base_ang_vel": base_ang_vel.astype(np.float32, copy=False),
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
            "obj_target_pose_size_b": np.concatenate(
                [obj_target_pos_b, obj_target_rot6d, object_size], axis=1
            ).astype(np.float32, copy=False),
            "obj_pos_b": obj_pos_b.astype(np.float32, copy=False),
            "obj_ori_b": obj_rot6d.astype(np.float32, copy=False),
            "obj_lin_vel_b": obj_lin_vel_b.astype(np.float32, copy=False),
            "obj_ang_vel_b": obj_ang_vel_b.astype(np.float32, copy=False),
        }

    def _get_motion_ref_ori_b(self, robot_state_data: np.ndarray) -> np.ndarray:
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        robot_ref_ori = self._get_observation_reference_orientation_in_world(robot_state_data)
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        return motion_ref_ori_b[..., :2].reshape(1, -1)

    def _maybe_attach_perception_obs(self, current_obs_buffer_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if "perception_obs" not in self.obs_dict:
            return current_obs_buffer_dict
        current_obs_buffer_dict["perception"] = self._get_perception_obs_input()
        return current_obs_buffer_dict

    def _get_legacy_object_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        current_obs_buffer_dict = {
            "motion_command": self.motion_command_t,
            "motion_ref_ori_b": self._get_motion_ref_ori_b(robot_state_data),
            "base_ang_vel": robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6],
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
        }

        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_observation_reference_pose_in_world(robot_state_data)
        object_state = self._get_live_object_state_from_sim_state()
        if object_state is not None:
            sim_ref_pos_w = object_state.get("robot_ref_pos_w")
            sim_ref_quat_wxyz = object_state.get("robot_ref_quat_wxyz")
            if (
                bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False))
                and sim_ref_pos_w is not None
                and sim_ref_quat_wxyz is not None
            ):
                robot_ref_pos_w = sim_ref_pos_w
                robot_ref_quat_wxyz = sim_ref_quat_wxyz

        if object_state is None:
            object_pos_b = np.zeros((1, 3), dtype=np.float32)
            object_ori_6d = np.zeros((1, 6), dtype=np.float32)
        else:
            object_pos_b, object_ori_6d = self._pose_to_ref_frame(
                robot_ref_pos_w,
                robot_ref_quat_wxyz,
                object_state["object_pos_w"],
                object_state["object_quat_wxyz"],
            )

        if self._motion_data is None or not self._motion_data.has_object:
            target_pose_size_b = np.zeros((1, 12), dtype=np.float32)
        else:
            self._maybe_update_motion_alignment(robot_state_data)
            idx = self._get_motion_index()
            target_pos_w = self._motion_data.object_pos_w[idx : idx + 1].copy()
            target_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1].copy()
            target_size = self._motion_data.object_size[idx : idx + 1].astype(np.float32, copy=False)
            if self._motion_align_quat_wxyz is not None:
                target_pos_w = self._apply_motion_alignment_pos(target_pos_w)
                target_quat_wxyz = self._apply_motion_alignment_quat(target_quat_wxyz)
            target_pos_b, target_ori_6d = self._pose_to_ref_frame(
                robot_ref_pos_w,
                robot_ref_quat_wxyz,
                target_pos_w.astype(np.float32, copy=False),
                target_quat_wxyz.astype(np.float32, copy=False),
            )
            target_pose_size_b = np.concatenate([target_pos_b, target_ori_6d, target_size], axis=1).astype(
                np.float32, copy=False
            )

        current_obs_buffer_dict["obj_target_pose_size_b"] = target_pose_size_b
        current_obs_buffer_dict["obj_pos_b"] = object_pos_b.astype(np.float32, copy=False)
        current_obs_buffer_dict["obj_ori_b"] = object_ori_6d.astype(np.float32, copy=False)
        return current_obs_buffer_dict

    def get_current_obs_buffer_dict(self, robot_state_data):
        robot_state_data = self._augment_robot_state_with_sim_state(robot_state_data)
        if robot_state_data is None:
            raise ValueError("Robot state is required for WBT observations.")

        self._refresh_motion_outputs_for_current_timestep()

        if self._uses_object_distill:
            current_obs_buffer_dict = self._get_object_distill_obs_buffer_dict(robot_state_data)
            return self._maybe_attach_perception_obs(current_obs_buffer_dict)

        if self._uses_object_generalist:
            current_obs_buffer_dict = self._get_object_generalist_obs_buffer_dict(robot_state_data)
            return self._maybe_attach_perception_obs(current_obs_buffer_dict)

        if self._uses_legacy_object_obs:
            current_obs_buffer_dict = self._get_legacy_object_obs_buffer_dict(robot_state_data)
            return self._maybe_attach_perception_obs(current_obs_buffer_dict)

        if self._uses_videomimic:
            current_obs_buffer_dict = self._get_videomimic_obs_buffer_dict(robot_state_data)
            if self._motion_future_target_pose_provider is not None:
                current_obs_buffer_dict["motion_future_target_poses"] = (
                    self._motion_future_target_pose_provider.get_future_target_poses(self._get_policy_motion_timestep())
                )
            return self._maybe_attach_perception_obs(current_obs_buffer_dict)

        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        current_obs_buffer_dict["motion_ref_ori_b"] = self._get_motion_ref_ori_b(robot_state_data)

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

        if self._motion_future_target_pose_provider is not None:
            current_obs_buffer_dict["motion_future_target_poses"] = (
                self._motion_future_target_pose_provider.get_future_target_poses(self._get_policy_motion_timestep())
            )

        return self._maybe_attach_perception_obs(current_obs_buffer_dict)

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[self._get_policy_motion_timestep()]], dtype=np.float32)
        if self._perception_input_name:
            if "perception_obs" not in obs:
                raise KeyError("Perception input required by ONNX but observation group 'perception_obs' is missing.")
            input_feed[self._perception_input_name] = obs["perception_obs"]
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
            self._last_motion_output_timestep = self._get_policy_motion_timestep()

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        self.scaled_policy_action = policy_action * self.policy_action_scales

        # update motion timestep
        if self.motion_clip_progressing:
            if self.use_sim_time:
                self._update_clock()
            else:
                self.motion_timestep += 1
            self._maybe_reset_on_motion_end()
        return self.scaled_policy_action

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if getattr(self, "_pending_noninteractive_policy_start", False):
            return None
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _get_viser_state_data(self, robot_state_data):
        """Prefer MuJoCo sim-state for Viser display and include object pose when available."""
        viser_state = self._augment_robot_state_with_sim_state(robot_state_data)
        payload: dict[str, np.ndarray] = {
            "robot_state_data": viser_state if viser_state is not None else robot_state_data,
        }

        object_state = self._get_live_object_state_from_sim_state()
        if object_state is None and self._motion_data is not None and self._motion_data.has_object:
            idx = self._get_motion_index()
            object_state = {
                "object_pos_w": self._motion_data.object_pos_w[idx : idx + 1].astype(np.float32, copy=False),
                "object_quat_wxyz": self._motion_data.object_quat_w[idx : idx + 1].astype(np.float32, copy=False),
            }

        if object_state is not None:
            object_pos_w = object_state.get("object_pos_w")
            object_quat_wxyz = object_state.get("object_quat_wxyz")
            if object_pos_w is not None and object_quat_wxyz is not None:
                payload["object_pos_w"] = object_pos_w
                payload["object_quat_wxyz"] = object_quat_wxyz
        return payload

    def _handle_start_policy(self):
        if not self._onnx_timestep_offset_aligned:
            robot_state_data = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
            if robot_state_data is not None:
                self._maybe_align_onnx_timestep_offset_to_current_pose(robot_state_data)
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        if self.ref_quat_xyzw_0 is not None:
            self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        if self._motion_alignment_enabled:
            robot_state_data = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(robot_state_data)

    def _can_finish_pending_policy_start(self, robot_state_data: np.ndarray) -> bool:  # noqa: ARG002
        if time.monotonic() < self._reset_restart_ready_at:
            return False
        if not self._awaiting_sim_reset_time_jump:
            return True

        sim_time_ms = self._get_latest_sim_time_ms()
        if sim_time_ms is None:
            return False
        if self._pre_reset_sim_time_ms is not None and sim_time_ms >= self._pre_reset_sim_time_ms:
            return False

        self._awaiting_sim_reset_time_jump = False
        self._pre_reset_sim_time_ms = None
        return True

    def _after_auto_start_policy(self) -> None:
        if self._pending_motion_restart_after_policy_start:
            self._pending_motion_restart_after_policy_start = False
            self._handle_start_motion_clip()

    def _handle_viser_reset_request(self) -> None:
        self._request_sim_reset_and_restart(reason="manual")

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
        self._reset_rollout_buffers()
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
        self._onnx_timestep_offset = 0
        self._onnx_timestep_offset_aligned = False
        self._last_motion_output_timestep = None
        self._logged_sim_ref_from_sim_state = False
        self._logged_object_contact_details = False

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        self._last_motion_output_timestep = None
        if self._motion_alignment_enabled:
            robot_state_data = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(robot_state_data)
        self.logger.info(colored("Starting motion clip", "blue"))

    def _request_sim_reset_and_restart(self, reason: str) -> None:
        self._pre_reset_sim_time_ms = self._get_latest_sim_time_ms()
        self._awaiting_sim_reset_time_jump = self._pre_reset_sim_time_ms is not None
        if self._sim_control_pub is not None:
            self._sim_control_pub.request_reset(reason)
        self._handle_stop_policy()
        self._pending_noninteractive_policy_start = True
        self._pending_motion_restart_after_policy_start = True
        self._reset_restart_ready_at = time.monotonic() + float(self.config.task.sim_reset_restart_delay_sec)
        self.logger.info("Requested simulator reset and motion restart ({})", reason)

    def _auto_reset_on_motion_end_enabled(self) -> bool:
        if self._viser_viewer is not None:
            getter = getattr(self._viser_viewer, "auto_reset_on_motion_end_enabled", None)
            if callable(getter):
                return bool(getter())
        return bool(self.config.viser.auto_reset_on_motion_end)

    def _maybe_reset_on_motion_end(self) -> None:
        if self._motion_data is None or not self.motion_clip_progressing:
            return
        if self.motion_timestep < self._motion_data.frame_count - 1:
            return
        if not self._auto_reset_on_motion_end_enabled():
            return
        self.logger.info("Motion clip reached the final frame; triggering reset")
        self._request_sim_reset_and_restart(reason="motion_end")

    def _should_auto_start_policy_immediately(self) -> bool:
        return not bool(getattr(self.config.task, "auto_start_motion_clip", False))

    def _set_stiff_hold_target_for_autostart(self) -> None:
        state = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
        if state is not None:
            self._maybe_align_onnx_timestep_offset_to_current_pose(state)

        if state is not None and state.shape[1] >= 7 + self.num_dofs:
            target_q = np.asarray(state[:, 7 : 7 + self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self.logger.info("Auto-start stiff target locked to current initialized robot pose.")
                return

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

    def _maybe_auto_start_rollout(self) -> None:
        if not getattr(self.config.task, "auto_start_motion_clip", False):
            return

        self._set_stiff_hold_target_for_autostart()
        self._stiff_hold_active = True
        self.use_policy_action = False
        self.get_ready_state = False
        self._pending_noninteractive_policy_start = False
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        hold_sec = max(0.0, float(getattr(self.config.task, "auto_start_stiff_hold_sec", 0.0)))
        max_wait_sec = max(hold_sec, float(getattr(self.config.task, "auto_start_stiff_max_wait_sec", hold_sec)))
        self._auto_start_hold_ticks = max(0, int(round(hold_sec * self.rl_rate)))
        self._auto_start_max_wait_ticks = max(self._auto_start_hold_ticks, int(round(max_wait_sec * self.rl_rate)))
        self._auto_start_tick_count = 0
        if self._auto_start_hold_ticks == 0 and self._auto_start_max_wait_ticks == 0:
            self._auto_start_stage = None
            self._stiff_hold_active = False
            self.logger.info("Auto-start enabled: zero stiff-hold requested, starting policy + clip immediately.")
            self._handle_start_policy()
            self._handle_start_motion_clip()
            return

        self._auto_start_stage = "stiff_hold"
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
        have_sim_state = not self.config.task.use_sim_state or self._get_sim_root_state() is not None

        progress_every = max(1, int(self.rl_rate // 2))
        if self._auto_start_tick_count % progress_every == 0:
            logger.info(
                "Auto-start stiff-hold progress: tick {}/{}, max_wait={}, max_dof_err={}, have_sim_state={}",
                self._auto_start_tick_count,
                self._auto_start_hold_ticks,
                self._auto_start_max_wait_ticks,
                "n/a" if err is None else f"{err:.4f}",
                have_sim_state,
            )

        if not min_hold_done:
            return
        if not have_sim_state and not max_wait_reached:
            return
        if not at_target and not max_wait_reached:
            return

        if not have_sim_state:
            self.logger.warning(
                "Auto-start proceeding without sim_state after max wait ({} ticks).",
                self._auto_start_tick_count,
            )
        if at_target:
            self.logger.info(
                "Auto-start stiff-hold reached target (max_dof_err={:.4f} rad <= {:.4f} rad).",
                err,
                self._auto_start_pose_tolerance,
            )
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
        self._maybe_advance_auto_start_state()

    def _on_run_exit(self) -> None:
        if self._sim_state_sub is not None:
            try:
                self._sim_state_sub.close()
            except Exception:
                pass
            self._sim_state_sub = None
        if self._sim_perception_sub is not None:
            try:
                self._sim_perception_sub.close()
            except Exception:
                pass
            self._sim_perception_sub = None
        if self._sim_control_pub is not None:
            try:
                self._sim_control_pub.close()
            except Exception:
                pass
            self._sim_control_pub = None
        try:
            self.clock_sub.close()
        except Exception:
            pass
        super()._on_run_exit()

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
        robot_state_data = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
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
