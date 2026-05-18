import json
import os
import sys
import threading
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
    quat_to_rpy,
    quat_rotate_inverse,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holosoma_inference.utils.policy_overlay import PolicyOverlayPub
from holosoma_inference.utils.sim_control import ManualRootCommandSub
from holosoma_inference.utils.sim_state import SimStateSub


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


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
        self.frame_name_to_id = {frame.name: idx for idx, frame in enumerate(self.robot_model.frames)}

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

    def fk_and_get_body_positions_in_world(self, configuration: np.ndarray, body_names: list[str]) -> np.ndarray:
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)
        root_pos = np.asarray(configuration[:3], dtype=np.float32)
        positions = np.zeros((len(body_names), 3), dtype=np.float32)
        for idx, body_name in enumerate(body_names):
            frame_id = self._resolve_body_frame_id(str(body_name))
            if frame_id is None:
                positions[idx] = root_pos
                continue
            positions[idx] = np.asarray(self.robot_data.oMf[frame_id].translation, dtype=np.float32)
        return positions

    def _resolve_body_frame_id(self, body_name: str) -> int | None:
        if body_name == "world":
            return None
        for candidate in (body_name, FAKE_BODY_NAME_ALIASES.get(body_name, "")):
            if candidate and candidate in self.frame_name_to_id:
                return int(self.frame_name_to_id[candidate])
        return None

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
        self.motion_path = motion_path
        self.body_names = tuple(body_names)
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


def _first_sustained_true_index(mask: np.ndarray, consecutive_steps: int) -> int | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    count = 0
    needed = max(int(consecutive_steps), 1)
    for idx, value in enumerate(mask):
        count = count + 1 if bool(value) else 0
        if count >= needed:
            return idx - needed + 1
    return None


def _first_sustained_true_index_from(mask: np.ndarray, consecutive_steps: int, start_idx: int) -> int | None:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if start_idx <= 0:
        return _first_sustained_true_index(mask, consecutive_steps)
    if start_idx >= mask.size:
        return None
    relative_idx = _first_sustained_true_index(mask[start_idx:], consecutive_steps)
    return None if relative_idx is None else int(start_idx + relative_idx)


def _smooth_1d_edge_padded(values: np.ndarray, window_steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    window_steps = max(int(window_steps), 1)
    if values.size == 0 or window_steps <= 1:
        return values
    left_pad = window_steps // 2
    right_pad = window_steps - 1 - left_pad
    padded = np.concatenate(
        [
            np.repeat(values[:1], left_pad),
            values,
            np.repeat(values[-1:], right_pad),
        ]
    )
    kernel = np.full((window_steps,), 1.0 / float(window_steps), dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32, copy=False)


def _extract_motion_cfg_from_metadata(metadata: dict[str, object]) -> dict | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    motion_cfg = (
        experiment_config.get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )
    return motion_cfg if isinstance(motion_cfg, dict) else None


def _extract_robot_init_state_from_metadata(metadata: dict[str, object]) -> dict | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    robot_cfg = experiment_config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        return None
    init_state = robot_cfg.get("init_state")
    return init_state if isinstance(init_state, dict) else None


def _extract_control_dt_from_metadata(metadata: dict[str, object]) -> float | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    sim_cfg = experiment_config.get("simulator", {}).get("config", {}).get("sim", {})
    if not isinstance(sim_cfg, dict):
        return None
    fps = float(sim_cfg.get("fps", 0.0) or 0.0)
    control_decimation = float(sim_cfg.get("control_decimation", 0.0) or 0.0)
    if fps <= 0.0 or control_decimation <= 0.0:
        return None
    return control_decimation / fps


def _normalize_quat_wxyz_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.divide(quat, norm, out=quat, where=norm > 0)


def _slerp_quat_wxyz_np(start: np.ndarray, end: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    start = _normalize_quat_wxyz_np(np.asarray(start, dtype=np.float32).reshape(4))
    end = _normalize_quat_wxyz_np(np.asarray(end, dtype=np.float32).reshape(4))
    alphas = np.asarray(alphas, dtype=np.float32).reshape(-1)
    if alphas.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot

    if dot > 0.9995:
        blended = start[None, :] + (end - start)[None, :] * alphas[:, None]
        return _normalize_quat_wxyz_np(blended)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alphas
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, None] * start[None, :]) + (s1[:, None] * end[None, :])


def _apply_transition_segment_np(
    motion: dict[str, np.ndarray],
    *,
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

    segments = {
        "joint_pos": _lerp(start_state["joint_pos"], target_state["joint_pos"]),
        "joint_vel": _lerp(start_state["joint_vel"], target_state["joint_vel"]),
        "root_pos_w": _lerp(start_state["root_pos"], target_state["root_pos"]),
        "ref_pos_w": _lerp(start_state["ref_pos"], target_state["ref_pos"]),
        "root_quat_w": _slerp_quat_wxyz_np(start_state["root_quat"], target_state["root_quat"], alphas),
        "ref_quat_w": _slerp_quat_wxyz_np(start_state["ref_quat"], target_state["ref_quat"], alphas),
    }
    if "object_pos" in start_state and "object_pos" in target_state:
        segments["object_pos_w"] = _lerp(start_state["object_pos"], target_state["object_pos"])
        segments["object_quat_w"] = _slerp_quat_wxyz_np(start_state["object_quat"], target_state["object_quat"], alphas)
        segments["object_size"] = _lerp(start_state["object_size"], target_state["object_size"])

    for key, segment in segments.items():
        if prepend:
            motion[key] = np.concatenate([segment, motion[key]], axis=0)
        else:
            motion[key] = np.concatenate([motion[key], segment], axis=0)


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
        self._contact_aware_carry_window: tuple[int, int] | None = None

        # Calculate timestep interval from rl_rate (e.g., 50Hz = 20ms intervals)
        self.timestep_interval_ms = 1000.0 / config.task.rl_rate

        # Initialize clock subscriber for synchronization
        self.clock_sub = ClockSub(port=config.task.sim_clock_port)
        self.clock_sub.start()
        self._last_clock_reading: int | None = None
        self._last_policy_control_clock_ms: int | None = None
        self._sim_time_control_schedule_ms = self._load_sim_time_control_schedule()
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms: int | None = None

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._latest_sim_state: dict | None = None
        self._sim_state_sub: SimStateSub | None = None
        self._manual_sparse_root_command_sub: ManualRootCommandSub | None = None
        self._manual_sparse_root_command_log_key: tuple[bool, str] | None = None
        self._manual_drop_button_log_value: float | None = None
        self._keyboard_sparse_root_command_enabled = _truthy_env("HOLOSOMA_KEYBOARD_ROOT_COMMAND")
        self._keyboard_sparse_root_command_mode = os.environ.get("HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE", "manual").strip().lower()
        try:
            self._keyboard_sparse_root_command_value = float(
                os.environ.get("HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE", "0.5")
            )
        except ValueError:
            self._keyboard_sparse_root_command_value = 0.5
        self._keyboard_sparse_root_command_value = abs(float(self._keyboard_sparse_root_command_value))
        try:
            keyboard_yaw_value_env = os.environ.get("HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_VALUE")
            if keyboard_yaw_value_env is not None:
                self._keyboard_sparse_root_command_yaw_value = float(keyboard_yaw_value_env)
            else:
                self._keyboard_sparse_root_command_yaw_value = float(
                    np.deg2rad(float(os.environ.get("HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_DEGREES", "17")))
                )
        except ValueError:
            self._keyboard_sparse_root_command_yaw_value = float(np.deg2rad(17.0))
        self._keyboard_sparse_root_command_yaw_value = abs(
            float(self._keyboard_sparse_root_command_yaw_value)
        )
        self._keyboard_sparse_root_pressed_keys: set[str] = set()
        self._keyboard_sparse_root_lock = threading.Lock()
        self._keyboard_sparse_root_last_command: tuple[float, float, float] | None = None
        self._last_sparse_motion_command: list[float] | None = None
        self._last_sparse_effective_command: list[float] | None = None
        self._last_sparse_manual_command: list[float] | None = None
        self._last_sparse_command_source = "auto"
        self._last_sparse_command_mode = "motion"
        self._last_sparse_manual_enabled = False
        self._logged_root_reference_clip_start = False
        self._logged_sim_ref_from_sim_state = False
        self._auto_start_motion_clip_pending = False
        self._auto_start_motion_clip_hold_start_time: float | None = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        self._disable_motion_end_sim_reset = (
            _truthy_env("HOLOSOMA_DISABLE_AUTO_RESET")
            or _truthy_env("HOLOSOMA_DISABLE_MOTION_END_RESET")
            or _truthy_env("HOLOSOMA_DISABLE_CLIP_END_RESET")
        )

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
        self._uses_sparse_root_command_contact_aware = (
            "sparse_target_root_trajectory_command_contact_aware" in obs_terms
        )
        self._uses_sparse_root_command = (
            "sparse_target_root_trajectory_command" in obs_terms
            or self._uses_sparse_root_command_contact_aware
        )
        self._uses_object_mocap_distill = "obj_current_pose_size_b" in obs_terms
        self._uses_object_generalist = any(
            term in obs_terms
            for term in (
                "obj_target_pose_size_b",
                "obj_pos_b",
                "obj_ori_b",
                "obj_lin_vel_b",
                "obj_ang_vel_b",
            )
        )
        self._uses_legacy_object_obs = (
            all(term in obs_terms for term in ("obj_target_pose_size_b", "obj_pos_b", "obj_ori_b"))
            and "obj_lin_vel_b" not in obs_terms
            and "obj_ang_vel_b" not in obs_terms
        )
        self._motion_data: MotionData | None = None
        self._motion_cfg: dict | None = None
        self._motion_align_quat_wxyz: np.ndarray | None = None
        self._motion_align_pos: np.ndarray | None = None
        self._onnx_obs_dim: int | None = None
        self._obs_input_name: str | None = None
        self._time_step_input_name: str | None = None
        self._perception_obs_input_name: str | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._motion_output_names: set[str] = set()
        self._motion_alignment_enabled = False
        try:
            self._motion_index_offset = int(os.environ.get("HOLOSOMA_POLICY_MOTION_INDEX_OFFSET", "0") or "0")
        except ValueError:
            self._motion_index_offset = 0
        self._force_motion_alignment = _truthy_env("HOLOSOMA_FORCE_MOTION_ALIGNMENT")
        self._skip_stiff_prompt = _truthy_env("HOLOSOMA_SKIP_STIFF_PROMPT")
        self._target_object_state_assist = _truthy_env("HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST")
        self._logged_target_object_state_assist = False
        self._target_robot_root_state_assist = _truthy_env("HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST")
        self._logged_target_robot_root_state_assist = False
        self._target_robot_dof_state_assist = _truthy_env("HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST")
        self._logged_target_robot_dof_state_assist = False
        self._use_motion_command_as_q_target = _truthy_env("HOLOSOMA_USE_MOTION_COMMAND_AS_Q_TARGET")
        self._logged_motion_command_q_target = False
        self._use_motion_data_as_q_target = _truthy_env("HOLOSOMA_USE_MOTION_DATA_AS_Q_TARGET")
        self._logged_motion_data_q_target = False
        self._prefill_obs_history_on_motion_start = (
            os.environ.get("HOLOSOMA_PREFILL_OBS_HISTORY_ON_MOTION_START", "0").lower()
            in {"1", "true", "yes", "on"}
        )
        self._logged_motion_start_history_prefill = False
        policy_overlay_port_raw = os.environ.get(
            "HOLOSOMA_POLICY_OVERLAY_PORT",
            os.environ.get("POLICY_OVERLAY_PORT", ""),
        ).strip()
        try:
            self._policy_overlay_port = int(policy_overlay_port_raw) if policy_overlay_port_raw else 0
        except ValueError:
            self._policy_overlay_port = 0
        self._policy_overlay_pub: PolicyOverlayPub | None = None
        self._motion_body_names: tuple[str, ...] = ()
        self._policy_debug_path = Path(os.environ["HOLOSOMA_POLICY_DEBUG_INPUT_PATH"]) if os.environ.get("HOLOSOMA_POLICY_DEBUG_INPUT_PATH") else None
        self._policy_debug_limit = int(os.environ.get("HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT", "12"))
        self._policy_debug_include_values = str(os.environ.get("HOLOSOMA_POLICY_DEBUG_INCLUDE_VALUES", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._policy_debug_count = 0
        self._policy_debug_initialized = False
        self._perception_obs_file_path = (
            Path(os.environ["HOLOSOMA_POLICY_PERCEPTION_OBS_FILE"]).expanduser()
            if os.environ.get("HOLOSOMA_POLICY_PERCEPTION_OBS_FILE")
            else None
        )
        self._perception_obs_file_key = os.environ.get("HOLOSOMA_POLICY_PERCEPTION_OBS_FILE_KEY", "perception_obs")
        self._perception_obs_file_values: np.ndarray | None = None
        self._logged_perception_obs_file = False
        self._policy_action_file_path = (
            Path(os.environ["HOLOSOMA_POLICY_ACTION_FILE"]).expanduser()
            if os.environ.get("HOLOSOMA_POLICY_ACTION_FILE")
            else None
        )
        self._policy_action_file_key = os.environ.get("HOLOSOMA_POLICY_ACTION_FILE_KEY", "actions")
        self._policy_action_file_values: np.ndarray | None = None
        self._logged_policy_action_file = False

        super().__init__(config)
        if self._policy_overlay_port > 0:
            self._policy_overlay_pub = PolicyOverlayPub(port=self._policy_overlay_port)
            self._policy_overlay_pub.start()

        if self._keyboard_sparse_root_command_enabled:
            logger.info(
                "Keyboard sparse root command enabled: w/s=x, a/d=y, q/e=yaw, xy_value={:.3f}, yaw={:.3f} rad ({:.1f} deg), mode={}",
                self._keyboard_sparse_root_command_value,
                self._keyboard_sparse_root_command_yaw_value,
                float(np.rad2deg(self._keyboard_sparse_root_command_yaw_value)),
                self._keyboard_sparse_root_command_mode,
            )
        if self._motion_index_offset != 0:
            logger.info("Using motion sequence index offset: {}", self._motion_index_offset)

        if self.config.task.use_sim_state:
            self._sim_state_sub = SimStateSub(port=self.config.task.sim_state_port)
            self._sim_state_sub.start()

        if self.config.task.use_external_sparse_root_command:
            self._manual_sparse_root_command_sub = ManualRootCommandSub(
                port=self.config.task.sparse_root_command_port,
            )
            self._manual_sparse_root_command_sub.start()

        if self.use_policy_action:
            self._handle_start_policy()

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

        if self._skip_stiff_prompt:
            logger.info("Skipping stiff hold confirmation prompt via HOLOSOMA_SKIP_STIFF_PROMPT.")
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

        if self.config.task.auto_start_motion:
            self._handle_start_motion_clip()
        elif self.config.task.auto_start_motion_clip:
            self._auto_start_motion_clip_pending = True

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

    @staticmethod
    def _extract_motion_config(metadata: dict) -> dict | None:
        motion_cfg = metadata.get("motion_config")
        if isinstance(motion_cfg, dict):
            return motion_cfg

        exp_cfg = metadata.get("experiment_config")
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
    def _find_repo_root(start: Path) -> Path:
        for parent in [start, *start.parents]:
            if (parent / "src" / "holosoma").exists():
                return parent
        return start

    @classmethod
    def _resolve_motion_file(cls, motion_file: str, onnx_path: Path) -> Path | None:
        motion_path = Path(motion_file).expanduser()
        if motion_path.is_file():
            return motion_path

        candidate = onnx_path.parent / motion_file
        if candidate.is_file():
            return candidate

        repo_root = cls._find_repo_root(Path(__file__).resolve())
        candidate = repo_root / motion_file
        if candidate.is_file():
            return candidate

        if motion_file.startswith("holosoma/"):
            candidate = repo_root / "src" / "holosoma" / motion_file
            if candidate.is_file():
                return candidate

        candidate = repo_root / "src" / motion_file
        if candidate.is_file():
            return candidate

        return None

    def _load_motion_data_from_metadata(self, metadata: dict, onnx_path: Path) -> None:
        motion_cfg = self._extract_motion_config(metadata)
        if not motion_cfg and not self.config.task.motion_file:
            raise ValueError("Motion config missing from ONNX metadata; cannot build VideoMimic observations.")

        motion_file = self.config.task.motion_file or motion_cfg.get("motion_file")
        if not motion_file:
            raise ValueError("motion_config.motion_file missing from ONNX metadata.")

        motion_path = self._resolve_motion_file(str(motion_file), onnx_path)
        if motion_path is None:
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        body_name_ref = motion_cfg.get("body_name_ref", ["torso_link"])
        if isinstance(body_name_ref, list) and body_name_ref:
            ref_name = body_name_ref[0]
        else:
            ref_name = "torso_link"

        robot_dof_names = metadata.get("dof_names") or list(self.config.robot.dof_names)
        self._motion_data = MotionData(motion_path, list(robot_dof_names), ref_name)
        self._motion_body_names = tuple(self._motion_data.body_names)
        self._maybe_apply_training_motion_transitions_to_motion_data(metadata, ref_name)
        self._motion_cfg = motion_cfg or {}
        self._contact_aware_carry_window = None
        alignment_from_metadata = bool((motion_cfg or {}).get("align_motion_to_init_yaw", False))
        self._motion_alignment_enabled = bool(alignment_from_metadata or self._force_motion_alignment)
        if self._motion_alignment_enabled and not alignment_from_metadata and self._force_motion_alignment:
            logger.info("Forcing runtime motion alignment for split sim2sim inference.")

    def _maybe_apply_training_motion_transitions_to_motion_data(self, metadata: dict, ref_name: str) -> None:
        if self._motion_data is None or not bool(self.config.task.apply_training_motion_transitions):
            return

        motion_cfg = _extract_motion_cfg_from_metadata(metadata)
        init_state = _extract_robot_init_state_from_metadata(metadata)
        control_dt = _extract_control_dt_from_metadata(metadata)
        if not isinstance(motion_cfg, dict) or not isinstance(init_state, dict) or control_dt is None or control_dt <= 0.0:
            return

        needs_prepend = bool(motion_cfg.get("enable_default_pose_prepend", False))
        needs_append = bool(motion_cfg.get("enable_default_pose_append", False))
        if not needs_prepend and not needs_append:
            return

        motion_data = self._motion_data
        robot_dof_names = list(metadata.get("dof_names") or self.config.robot.dof_names)
        default_dof = np.zeros((len(robot_dof_names),), dtype=np.float32)
        default_joint_angles = init_state.get("default_joint_angles")
        if isinstance(default_joint_angles, dict):
            for i, name in enumerate(robot_dof_names):
                if name in default_joint_angles:
                    default_dof[i] = float(default_joint_angles[name])
        else:
            default_dof = motion_data.joint_pos[0].astype(np.float32, copy=True)

        def _build_default_state(use_motion_end: bool) -> dict[str, np.ndarray]:
            motion_idx = -1 if use_motion_end else 0
            motion_root_pos = motion_data.root_pos_w[motion_idx]
            motion_root_quat = motion_data.root_quat_w[motion_idx]
            _, _, motion_yaw = quat_to_rpy(motion_root_quat)

            init_pos = np.asarray(init_state.get("pos", [0.0, 0.0, motion_root_pos[2]]), dtype=np.float32)
            init_rot_xyzw = np.asarray(init_state.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32).reshape(1, 4)
            init_rot_wxyz = xyzw_to_wxyz(init_rot_xyzw)[0]
            init_roll, init_pitch, _ = quat_to_rpy(init_rot_wxyz)

            default_root_pos = np.asarray([motion_root_pos[0], motion_root_pos[1], init_pos[2]], dtype=np.float32)
            default_root_quat = rpy_to_quat((float(init_roll), float(init_pitch), float(motion_yaw))).astype(np.float32)

            root_quat_xyzw = wxyz_to_xyzw(default_root_quat.reshape(1, 4))[0]
            dof_pos_pin = default_dof[self.pinocchio_robot.real2pinocchio_index]
            configuration = np.concatenate([default_root_pos, root_quat_xyzw, dof_pos_pin], axis=0)
            ref_pos, ref_quat_xyzw = self.pinocchio_robot.fk_and_get_ref_body_pose_in_world(configuration)
            state = {
                "joint_pos": default_dof.astype(np.float32, copy=True),
                "joint_vel": np.zeros_like(default_dof, dtype=np.float32),
                "root_pos": default_root_pos.astype(np.float32, copy=False),
                "root_quat": default_root_quat.astype(np.float32, copy=False),
                "ref_pos": ref_pos.astype(np.float32, copy=False),
                "ref_quat": xyzw_to_wxyz(ref_quat_xyzw.reshape(1, 4))[0].astype(np.float32, copy=False),
            }
            if motion_data.has_object:
                state["object_pos"] = motion_data.object_pos_w[motion_idx].astype(np.float32, copy=False)
                state["object_quat"] = motion_data.object_quat_w[motion_idx].astype(np.float32, copy=False)
                state["object_size"] = motion_data.object_size[motion_idx].astype(np.float32, copy=False)
            return state

        def _motion_state(idx: int) -> dict[str, np.ndarray]:
            state = {
                "joint_pos": motion_data.joint_pos[idx].astype(np.float32, copy=False),
                "joint_vel": motion_data.joint_vel[idx].astype(np.float32, copy=False),
                "root_pos": motion_data.root_pos_w[idx].astype(np.float32, copy=False),
                "root_quat": motion_data.root_quat_w[idx].astype(np.float32, copy=False),
                "ref_pos": motion_data.ref_pos_w[idx].astype(np.float32, copy=False),
                "ref_quat": motion_data.ref_quat_w[idx].astype(np.float32, copy=False),
            }
            if motion_data.has_object:
                state["object_pos"] = motion_data.object_pos_w[idx].astype(np.float32, copy=False)
                state["object_quat"] = motion_data.object_quat_w[idx].astype(np.float32, copy=False)
                state["object_size"] = motion_data.object_size[idx].astype(np.float32, copy=False)
            return state

        motion = {
            "joint_pos": motion_data.joint_pos.astype(np.float32, copy=True),
            "joint_vel": motion_data.joint_vel.astype(np.float32, copy=True),
            "root_pos_w": motion_data.root_pos_w.astype(np.float32, copy=True),
            "root_quat_w": motion_data.root_quat_w.astype(np.float32, copy=True),
            "ref_pos_w": motion_data.ref_pos_w.astype(np.float32, copy=True),
            "ref_quat_w": motion_data.ref_quat_w.astype(np.float32, copy=True),
        }
        if motion_data.has_object:
            motion["object_pos_w"] = motion_data.object_pos_w.astype(np.float32, copy=True)
            motion["object_quat_w"] = motion_data.object_quat_w.astype(np.float32, copy=True)
            motion["object_size"] = motion_data.object_size.astype(np.float32, copy=True)

        if needs_prepend:
            prepend_duration = float(motion_cfg.get("default_pose_prepend_duration_s", 0.0) or 0.0)
            prepend_steps = round(prepend_duration / control_dt)
            if prepend_steps > 1:
                _apply_transition_segment_np(
                    motion,
                    start_state=_build_default_state(use_motion_end=False),
                    target_state=_motion_state(0),
                    num_steps=prepend_steps,
                    prepend=True,
                    drop_first=False,
                    drop_last=True,
                )

        if needs_append:
            append_duration = float(motion_cfg.get("default_pose_append_duration_s", 0.0) or 0.0)
            append_steps = round(append_duration / control_dt)
            if append_steps > 1:
                _apply_transition_segment_np(
                    motion,
                    start_state=_motion_state(-1),
                    target_state=_build_default_state(use_motion_end=True),
                    num_steps=append_steps,
                    prepend=False,
                    drop_first=True,
                    drop_last=False,
                )

        motion_data.joint_pos = motion["joint_pos"]
        motion_data.joint_vel = motion["joint_vel"]
        motion_data.root_pos_w = motion["root_pos_w"]
        motion_data.root_quat_w = motion["root_quat_w"]
        motion_data.ref_pos_w = motion["ref_pos_w"]
        motion_data.ref_quat_w = motion["ref_quat_w"]
        if motion_data.has_object:
            motion_data.object_pos_w = motion["object_pos_w"]
            motion_data.object_quat_w = motion["object_quat_w"]
            motion_data.object_size = motion["object_size"]
        motion_data.frame_count = motion_data.joint_pos.shape[0]
        logger.info(
            "Applied training motion transitions to inference motion data for '{}': frame_count={}",
            ref_name,
            motion_data.frame_count,
        )

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
        self._maybe_force_sparse_depth_distill_obs_config()
        self._maybe_force_legacy_object_history_obs_config()

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        # Keep WBT rollout aligned with training-time action scaling semantics.
        self._set_policy_action_scales_from_metadata(metadata)

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        if (
            self._uses_videomimic
            or self._uses_object_mocap_distill
            or self._uses_object_generalist
            or self._uses_legacy_object_obs
            or self._uses_sparse_root_command
        ):
            self._load_motion_data_from_metadata(metadata, Path(model_path))

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported ONNX inputs: {self.onnx_input_names}")

        self._time_step_input_name = "time_step" if "time_step" in self.onnx_input_names else None
        self._perception_obs_input_name = "perception_obs" if "perception_obs" in self.onnx_input_names else None

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
            obs = self._build_zero_actor_obs()
            input_feed = {self._obs_input_name: obs}
            if self._time_step_input_name:
                input_feed[self._time_step_input_name] = time_step
            if self._perception_obs_input_name:
                perception_dim = self._get_onnx_input_dim(self._perception_obs_input_name)
                if perception_dim is None:
                    raise ValueError("Unable to infer perception_obs input dimension from ONNX.")
                input_feed[self._perception_obs_input_name] = np.zeros((1, perception_dim), dtype=np.float32)
            outputs = self.policy(input_feed)
            joint_pos = outputs["joint_pos"]
            joint_vel = outputs["joint_vel"]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.ref_quat_xyzw_t = outputs["ref_quat_xyzw"]
            self.ref_pos_xyz_t = outputs.get("ref_pos_xyz")
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
        elif (
            self._uses_videomimic
            or self._uses_object_mocap_distill
            or self._uses_object_generalist
            or self._uses_legacy_object_obs
            or self._uses_sparse_root_command
        ) and self._motion_data is not None:
            joint_pos = self._motion_data.joint_pos[:1]
            joint_vel = self._motion_data.joint_vel[:1]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.motion_command_0 = self.motion_command_t.copy()
            ref_quat_wxyz = self._motion_data.ref_quat_w[:1]
            self.ref_quat_xyzw_t = wxyz_to_xyzw(ref_quat_wxyz)
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self.ref_pos_xyz_t = self._motion_data.ref_pos_w[:1]

    def _get_onnx_input_dim(self, input_name: str | None) -> int | None:
        if input_name is None:
            return None
        for inp in self.onnx_policy_session.get_inputs():
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

    def _maybe_force_sparse_depth_distill_obs_config(self) -> None:
        if not self._uses_sparse_root_command or self._onnx_obs_dim != 308:
            return

        configured_dim = 0
        for group, terms in self.obs_dict.items():
            history_len = int(self.history_length_dict.get(group, 1))
            configured_dim += sum(int(self.obs_dims[term]) for term in terms) * history_len
        if configured_dim == 308:
            return

        logger.warning(
            "Overriding sparse depth-distill observation config from {} dims to ONNX-aligned 308 dims.",
            configured_dim,
        )
        object.__setattr__(self.config, "observation", ObservationConfig(
            obs_dict={
                "actor_obs_root": ["sparse_target_root_trajectory_command"],
                "actor_obs_proprio_no_linvel": ["base_ang_vel", "dof_pos", "dof_vel"],
            },
            obs_dims={
                "sparse_target_root_trajectory_command": 3,
                "base_ang_vel": 3,
                "dof_pos": self.num_dofs,
                "dof_vel": self.num_dofs,
            },
            obs_scales={
                "sparse_target_root_trajectory_command": 1.0,
                "base_ang_vel": 1.0,
                "dof_pos": 1.0,
                "dof_vel": 1.0,
            },
            history_length_dict={
                "actor_obs_root": 1,
                "actor_obs_proprio_no_linvel": 5,
            },
        ))
        self._init_obs_config()

    def _maybe_force_legacy_object_history_obs_config(self) -> None:
        if not self._uses_legacy_object_obs or self._onnx_obs_dim is None:
            return

        actor_terms = list(self.config.observation.obs_dict.get("actor_obs", []))
        frame_dim = sum(int(self.config.observation.obs_dims[term]) for term in actor_terms)
        if frame_dim <= 0 or self._onnx_obs_dim % frame_dim != 0:
            return

        expected_history = int(self._onnx_obs_dim // frame_dim)
        current_history = int(self.config.observation.history_length_dict.get("actor_obs", 1))
        if expected_history <= 1 or current_history == expected_history:
            return

        logger.warning(
            "Overriding legacy object observation history from {} to {} to match ONNX obs dim {}.",
            current_history,
            expected_history,
            self._onnx_obs_dim,
        )
        history_lengths = dict(self.config.observation.history_length_dict)
        history_lengths["actor_obs"] = expected_history
        object.__setattr__(self.config, "observation", ObservationConfig(
            obs_dict=dict(self.config.observation.obs_dict),
            obs_dims=dict(self.config.observation.obs_dims),
            obs_scales=dict(self.config.observation.obs_scales),
            history_length_dict=history_lengths,
        ))
        self._init_obs_config()

    def _build_zero_actor_obs(self) -> np.ndarray:
        obs_dim = self._onnx_obs_dim
        if obs_dim is None:
            obs_dim = int(sum(int(template.shape[1]) for template in self.obs_buf_dict.values()))
        return np.zeros((1, int(obs_dim)), dtype=np.float32)

    def _sync_motion_outputs_from_onnx(self, motion_index: int) -> None:
        """Update motion observation targets before constructing actor observations."""
        if not self._uses_motion_command or self._time_step_input_name is None:
            return
        if not {"joint_pos", "joint_vel", "ref_quat_xyzw"}.issubset(self._motion_output_names):
            return

        fetch_names = ["joint_pos", "joint_vel", "ref_quat_xyzw"]
        if "ref_pos_xyz" in self._motion_output_names:
            fetch_names.append("ref_pos_xyz")

        input_feed = {
            self._obs_input_name: self._build_zero_actor_obs(),
            self._time_step_input_name: np.array([[int(motion_index)]], dtype=np.float32),
        }
        if self._perception_obs_input_name:
            perception_dim = self._get_onnx_input_dim(self._perception_obs_input_name)
            if perception_dim is None:
                raise ValueError("Unable to infer perception_obs input dimension from ONNX.")
            input_feed[self._perception_obs_input_name] = np.zeros((1, perception_dim), dtype=np.float32)

        outputs = dict(zip(fetch_names, self.onnx_policy_session.run(fetch_names, input_feed)))
        self.motion_command_t = np.concatenate([outputs["joint_pos"], outputs["joint_vel"]], axis=1)
        self.ref_quat_xyzw_t = outputs["ref_quat_xyzw"]
        self.ref_pos_xyz_t = outputs.get("ref_pos_xyz", self.ref_pos_xyz_t)

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False

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

    def _get_motion_index(self) -> int:
        if self._motion_data is None:
            return 0
        idx = int(self.motion_timestep) + int(self._motion_index_offset)
        if idx < 0:
            return 0
        return min(idx, self._motion_data.frame_count - 1)

    def _get_file_perception_obs(self, expected_dim: int) -> np.ndarray | None:
        if self._perception_obs_file_path is None:
            return None
        if self._perception_obs_file_values is None:
            path = self._perception_obs_file_path
            if path.suffix.lower() == ".npz":
                with np.load(path) as data:
                    if self._perception_obs_file_key not in data.files:
                        raise KeyError(
                            f"{path} does not contain perception obs key "
                            f"{self._perception_obs_file_key!r}; available={data.files}"
                        )
                    values = np.asarray(data[self._perception_obs_file_key], dtype=np.float32)
            else:
                values = np.asarray(np.load(path), dtype=np.float32)
            values = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
            if values.shape[1] != int(expected_dim):
                raise ValueError(
                    f"Perception obs file dim mismatch: got {values.shape[1]}, expected {int(expected_dim)}"
                )
            self._perception_obs_file_values = values.astype(np.float32, copy=False)
            if not self._logged_perception_obs_file:
                logger.info(
                    "Using file-backed perception_obs from {} key={} frames={} dim={}",
                    path,
                    self._perception_obs_file_key,
                    self._perception_obs_file_values.shape[0],
                    self._perception_obs_file_values.shape[1],
                )
                self._logged_perception_obs_file = True

        index_mode = os.environ.get("HOLOSOMA_POLICY_PERCEPTION_OBS_FILE_INDEX", "motion_timestep").strip().lower()
        if index_mode == "motion_index":
            frame_idx = self._get_motion_index()
        elif index_mode in {"count", "policy_count"}:
            frame_idx = int(self._policy_debug_count)
        else:
            frame_idx = int(self.motion_timestep)
        frame_idx = max(0, min(int(frame_idx), int(self._perception_obs_file_values.shape[0]) - 1))
        return self._perception_obs_file_values[frame_idx : frame_idx + 1].copy()

    def _get_file_policy_action(self) -> np.ndarray | None:
        if self._policy_action_file_path is None:
            return None
        if self._policy_action_file_values is None:
            path = self._policy_action_file_path
            if path.suffix.lower() == ".npz":
                with np.load(path) as data:
                    if self._policy_action_file_key not in data.files:
                        raise KeyError(
                            f"{path} does not contain action key {self._policy_action_file_key!r}; "
                            f"available={data.files}"
                        )
                    values = np.asarray(data[self._policy_action_file_key], dtype=np.float32)
            else:
                values = np.asarray(np.load(path), dtype=np.float32)
            values = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
            if values.shape[1] != int(self.num_dofs):
                raise ValueError(f"Policy action file dim mismatch: got {values.shape[1]}, expected {self.num_dofs}")
            self._policy_action_file_values = values.astype(np.float32, copy=False)
            if not self._logged_policy_action_file:
                logger.info(
                    "Using file-backed raw policy actions from {} key={} frames={} dim={}",
                    path,
                    self._policy_action_file_key,
                    self._policy_action_file_values.shape[0],
                    self._policy_action_file_values.shape[1],
                )
                self._logged_policy_action_file = True

        index_mode = os.environ.get("HOLOSOMA_POLICY_ACTION_FILE_INDEX", "motion_timestep").strip().lower()
        if index_mode == "motion_index":
            frame_idx = self._get_motion_index()
        elif index_mode in {"count", "policy_count"}:
            frame_idx = int(self._policy_debug_count)
        else:
            frame_idx = int(self.motion_timestep)
        frame_idx = max(0, min(int(frame_idx), int(self._policy_action_file_values.shape[0]) - 1))
        return self._policy_action_file_values[frame_idx : frame_idx + 1].copy()

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

    def _get_current_motion_target_root_pose(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        if self._motion_data is None:
            return None
        idx = self._get_motion_index()
        root_pos_w = self._motion_data.root_pos_w[idx : idx + 1].copy()
        root_quat_wxyz = self._motion_data.root_quat_w[idx : idx + 1].copy()
        if self._motion_align_quat_wxyz is not None:
            root_pos_w = self._apply_motion_alignment_pos(root_pos_w)
            root_quat_wxyz = self._apply_motion_alignment_quat(root_quat_wxyz)
        return root_pos_w, root_quat_wxyz, idx

    def _get_current_motion_target_body_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
        if self._motion_data is None or not self._motion_body_names:
            return None
        root_pose = self._get_current_motion_target_root_pose()
        if root_pose is None:
            return None
        root_pos_w, root_quat_wxyz, idx = root_pose
        joint_pos = self._motion_data.joint_pos[idx].astype(np.float32, copy=False)
        dof_pos_in_pinocchio = joint_pos[self.pinocchio_robot.real2pinocchio_index]
        root_quat_xyzw = wxyz_to_xyzw(root_quat_wxyz)[0]
        configuration = np.concatenate([root_pos_w[0], root_quat_xyzw, dof_pos_in_pinocchio], axis=0)
        body_pos_w = self.pinocchio_robot.fk_and_get_body_positions_in_world(
            configuration,
            list(self._motion_body_names),
        )
        return body_pos_w, root_pos_w, root_quat_wxyz, idx

    def _publish_policy_overlay(self) -> None:
        pub = self._policy_overlay_pub
        if pub is None:
            return
        target = self._get_current_motion_target_body_positions()
        if target is None or self._motion_data is None:
            payload: dict[str, object] = {"clip_active": bool(self.motion_clip_progressing)}
            payload.update(self._sparse_root_command_overlay_fields())
            pub.publish(payload)
            return

        body_pos_w, root_pos_w, root_quat_wxyz, idx = target
        payload: dict[str, object] = {
            "clip_active": bool(self.motion_clip_progressing),
            "motion_timestep": int(self.motion_timestep),
            "frame_idx": int(idx),
            "motion_path": str(self._motion_data.motion_path),
            "body_names": list(self._motion_body_names),
            "body_pos_w": body_pos_w.tolist(),
            "root_pos_w": root_pos_w.reshape(-1).tolist(),
            "root_quat_wxyz": root_quat_wxyz.reshape(-1).tolist(),
        }
        payload.update(self._sparse_root_command_overlay_fields())
        self._maybe_publish_target_robot_root_state_assist(root_pos_w, root_quat_wxyz)
        self._maybe_publish_target_robot_dof_state_assist(idx)
        if self._motion_data.has_object and self._motion_data.object_pos_w is not None and self._motion_data.object_quat_w is not None:
            object_pos_w = self._motion_data.object_pos_w[idx : idx + 1].copy()
            object_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1].copy()
            if self._motion_align_quat_wxyz is not None:
                object_pos_w = self._apply_motion_alignment_pos(object_pos_w)
                object_quat_wxyz = self._apply_motion_alignment_quat(object_quat_wxyz)
            self._maybe_publish_target_object_state_assist(object_pos_w, object_quat_wxyz)
            payload["object_pos_w"] = object_pos_w.reshape(-1).tolist()
            payload["object_quat_wxyz"] = object_quat_wxyz.reshape(-1).tolist()
        pub.publish(payload)

    def _command3(self, command: np.ndarray) -> list[float]:
        return np.asarray(command, dtype=np.float32).reshape(-1)[:3].astype(float).tolist()

    def _record_sparse_root_command(
        self,
        motion_command: np.ndarray,
        effective_command: np.ndarray,
        *,
        source: str,
        mode: str,
        manual_enabled: bool,
        manual_command: np.ndarray | None = None,
    ) -> None:
        self._last_sparse_motion_command = self._command3(motion_command)
        self._last_sparse_effective_command = self._command3(effective_command)
        self._last_sparse_manual_command = self._command3(manual_command) if manual_command is not None else None
        self._last_sparse_command_source = str(source)
        self._last_sparse_command_mode = str(mode)
        self._last_sparse_manual_enabled = bool(manual_enabled)

    def _sparse_root_command_overlay_fields(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "sparse_command_source": self._last_sparse_command_source,
            "sparse_command_mode": self._last_sparse_command_mode,
            "sparse_manual_enabled": bool(self._last_sparse_manual_enabled),
        }
        if self._last_sparse_motion_command is not None:
            payload["sparse_motion_command"] = self._last_sparse_motion_command
        if self._last_sparse_effective_command is not None:
            payload["sparse_effective_command"] = self._last_sparse_effective_command
        if self._last_sparse_manual_command is not None:
            payload["sparse_manual_command"] = self._last_sparse_manual_command
        return payload

    def _maybe_publish_target_object_state_assist(
        self,
        object_pos_w: np.ndarray,
        object_quat_wxyz: np.ndarray,
    ) -> None:
        if not self._target_object_state_assist:
            return
        publisher = getattr(self.interface, "publish_actor_state", None)
        if publisher is None:
            return
        object_pos = np.asarray(object_pos_w, dtype=np.float32).reshape(1, 3)
        object_quat_xyzw = wxyz_to_xyzw(np.asarray(object_quat_wxyz, dtype=np.float32).reshape(1, 4))
        object_state = np.concatenate(
            [
                object_pos,
                object_quat_xyzw.astype(np.float32, copy=False),
                np.zeros((1, 6), dtype=np.float32),
            ],
            axis=1,
        )
        publisher(self.config.task.sim_object_name, object_state[0])
        if not self._logged_target_object_state_assist:
            logger.info("Publishing target object state assist to MuJoCo actor '{}'.", self.config.task.sim_object_name)
            self._logged_target_object_state_assist = True

    def _maybe_publish_target_robot_root_state_assist(
        self,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
    ) -> None:
        if not self._target_robot_root_state_assist:
            return
        publisher = getattr(self.interface, "publish_robot_root_state", None)
        if publisher is None:
            return
        root_pos = np.asarray(root_pos_w, dtype=np.float32).reshape(1, 3)
        root_quat_xyzw = wxyz_to_xyzw(np.asarray(root_quat_wxyz, dtype=np.float32).reshape(1, 4))
        root_state = np.concatenate(
            [
                root_pos,
                root_quat_xyzw.astype(np.float32, copy=False),
                np.zeros((1, 6), dtype=np.float32),
            ],
            axis=1,
        )
        publisher(root_state[0])
        if not self._logged_target_robot_root_state_assist:
            logger.info("Publishing target robot root state assist to MuJoCo.")
            self._logged_target_robot_root_state_assist = True

    def _maybe_publish_target_robot_dof_state_assist(self, idx: int) -> None:
        if not self._target_robot_dof_state_assist or self._motion_data is None:
            return
        publisher = getattr(self.interface, "publish_robot_dof_state", None)
        if publisher is None:
            return
        joint_pos = self._motion_data.joint_pos[idx].astype(np.float32, copy=False)
        joint_vel = self._motion_data.joint_vel[idx].astype(np.float32, copy=False)
        dof_state = np.stack([joint_pos, joint_vel], axis=1)
        publisher(dof_state)
        if not self._logged_target_robot_dof_state_assist:
            logger.info("Publishing target robot dof state assist to MuJoCo.")
            self._logged_target_robot_dof_state_assist = True

    def _calc_heading_quat_inv(self, quat_wxyz: np.ndarray) -> np.ndarray:
        yaw = self._quat_yaw(quat_wxyz)
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw)).reshape(1, 4)
        return yaw_quat.astype(np.float32)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _get_latest_sim_state(self) -> dict | None:
        if self._sim_state_sub is None:
            return self._latest_sim_state
        state = self._sim_state_sub.get_state()
        if state is not None:
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

    def _get_motion_ref_ori_b(self, robot_state_data: np.ndarray) -> np.ndarray:
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        robot_ref_ori = self._get_observation_reference_orientation_in_world(robot_state_data)
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        return motion_ref_ori_b[..., :2].reshape(1, -1)

    def _get_keyboard_sparse_root_command(self) -> tuple[str, np.ndarray] | None:
        if not self._keyboard_sparse_root_command_enabled:
            return None

        value = self._keyboard_sparse_root_command_value
        yaw_value = self._keyboard_sparse_root_command_yaw_value
        with self._keyboard_sparse_root_lock:
            pressed = set(self._keyboard_sparse_root_pressed_keys)

        x = (float("w" in pressed) - float("s" in pressed)) * value
        y = (float("a" in pressed) - float("d" in pressed)) * value
        yaw = (float("q" in pressed) - float("e" in pressed)) * yaw_value
        command_tuple = (float(x), float(y), float(yaw))
        if command_tuple != self._keyboard_sparse_root_last_command:
            logger.info(
                "Keyboard sparse root command: x={:.3f} y={:.3f} yaw={:.3f}",
                command_tuple[0],
                command_tuple[1],
                command_tuple[2],
            )
            self._keyboard_sparse_root_last_command = command_tuple

        return self._keyboard_sparse_root_command_mode, np.asarray([command_tuple], dtype=np.float32)

    def _apply_sparse_root_command(
        self,
        motion_command: np.ndarray,
        manual_command: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        if mode in {"offset", "add", "motion_plus_manual", "motion+manual"}:
            command = np.array(motion_command, dtype=np.float32, copy=True)
            command[:, :2] += manual_command[:, :2]
            command[:, 2] = np.asarray(
                [self._normalize_angle(float(value)) for value in command[:, 2] + manual_command[:, 2]],
                dtype=np.float32,
            )
            return command

        manual_command[:, 2] = np.asarray(
            [self._normalize_angle(float(value)) for value in manual_command[:, 2]],
            dtype=np.float32,
        )
        return manual_command

    def _apply_external_sparse_root_command(self, motion_command: np.ndarray) -> np.ndarray:
        keyboard_command = self._get_keyboard_sparse_root_command()
        if keyboard_command is not None:
            mode, manual_command = keyboard_command
            effective_command = self._apply_sparse_root_command(motion_command, manual_command, mode)
            self._record_sparse_root_command(
                motion_command,
                effective_command,
                source="manual_keyboard",
                mode=mode,
                manual_enabled=True,
                manual_command=manual_command,
            )
            return effective_command

        sub = self._manual_sparse_root_command_sub
        if sub is None:
            self._record_sparse_root_command(
                motion_command,
                motion_command,
                source="auto",
                mode="motion",
                manual_enabled=False,
            )
            return motion_command

        payload = sub.get_payload()
        enabled = bool(payload.get("enabled", False)) if isinstance(payload, dict) else False
        mode = str(payload.get("mode", "manual")).strip().lower() if isinstance(payload, dict) else "manual"
        log_key = (enabled, mode)
        if log_key != self._manual_sparse_root_command_log_key:
            logger.info("External sparse root command: enabled={} mode={}", enabled, mode)
            self._manual_sparse_root_command_log_key = log_key
        if not enabled:
            self._record_sparse_root_command(
                motion_command,
                motion_command,
                source="auto",
                mode="motion",
                manual_enabled=False,
            )
            return motion_command

        command_raw = payload.get("command") if isinstance(payload, dict) else None
        try:
            manual_command = np.asarray(command_raw, dtype=np.float32).reshape(1, -1)[:, :3]
        except (TypeError, ValueError):
            logger.warning("Ignoring malformed external sparse root command: {}", command_raw)
            self._record_sparse_root_command(
                motion_command,
                motion_command,
                source="auto",
                mode="motion",
                manual_enabled=False,
            )
            return motion_command
        if manual_command.shape[1] != 3:
            logger.warning("Ignoring external sparse root command with dim {}", manual_command.shape[1])
            self._record_sparse_root_command(
                motion_command,
                motion_command,
                source="auto",
                mode="motion",
                manual_enabled=False,
            )
            return motion_command
        manual_command = np.nan_to_num(manual_command, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32,
            copy=False,
        )
        effective_command = self._apply_sparse_root_command(motion_command, manual_command, mode)
        self._record_sparse_root_command(
            motion_command,
            effective_command,
            source="manual",
            mode=mode,
            manual_enabled=True,
            manual_command=manual_command,
        )
        return effective_command

    def _get_sparse_target_root_trajectory_command(self, robot_state_data: np.ndarray) -> np.ndarray:
        if self._motion_data is None:
            raise ValueError("Motion data is required for sparse root trajectory observations.")

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()

        motion_root_pos_w = self._motion_data.root_pos_w[idx : idx + 1].copy()
        motion_root_quat_wxyz = self._motion_data.root_quat_w[idx : idx + 1].copy()
        if self._motion_align_quat_wxyz is not None:
            motion_root_pos_w = self._apply_motion_alignment_pos(motion_root_pos_w)
            motion_root_quat_wxyz = self._apply_motion_alignment_quat(motion_root_quat_wxyz)

        robot_root_pos_w = np.asarray(robot_state_data[:, :3], dtype=np.float32)
        robot_root_quat_wxyz = np.asarray(robot_state_data[:, 3:7], dtype=np.float32)
        rel_pos_w = motion_root_pos_w - robot_root_pos_w
        rel_pos_b = quat_apply(self._calc_heading_quat_inv(robot_root_quat_wxyz), rel_pos_w)
        rel_xy = rel_pos_b[:, :2]

        target_heading = self._quat_yaw(motion_root_quat_wxyz)
        robot_heading = self._quat_yaw(robot_root_quat_wxyz)
        rel_yaw = np.array([[self._normalize_angle(target_heading - robot_heading)]], dtype=np.float32)
        motion_command = np.concatenate([rel_xy, rel_yaw], axis=1).astype(np.float32, copy=False)
        return self._apply_external_sparse_root_command(motion_command)

    def _get_contact_aware_carry_window(self) -> tuple[int, int]:
        if self._contact_aware_carry_window is not None:
            return self._contact_aware_carry_window
        if self._motion_data is None or not self._motion_data.has_object or self._motion_data.object_pos_w is None:
            end = 0 if self._motion_data is None else int(self._motion_data.frame_count)
            self._contact_aware_carry_window = (0, end)
            return self._contact_aware_carry_window

        cfg = self._motion_cfg or {}
        mode = str(cfg.get("contact_aware_carry_window_mode", "rel_z")).strip().lower().replace("-", "_")
        consecutive_steps = 5
        total_steps = int(self._motion_data.frame_count)
        if total_steps <= 0:
            self._contact_aware_carry_window = (0, 0)
            return self._contact_aware_carry_window

        if mode == "peak_height":
            alpha = max(0.0, min(float(cfg.get("contact_aware_peak_height_alpha", 0.91)), 1.0))
            smoothing_steps = int(cfg.get("contact_aware_peak_height_smoothing_steps", 5))
            height = _smooth_1d_edge_padded(self._motion_data.object_pos_w[:, 2], smoothing_steps)
            threshold = float(np.min(height) + max(float(np.max(height) - np.min(height)), 0.0) * alpha)
            high_mask = height >= threshold
            carry_start = _first_sustained_true_index(high_mask, consecutive_steps)
            if carry_start is None:
                high_indices = np.flatnonzero(high_mask)
                carry_start = int(high_indices[0]) if high_indices.size else int(np.argmax(height))
            peak_step = int(np.argmax(height))
            carry_end = _first_sustained_true_index_from(
                ~high_mask,
                consecutive_steps,
                start_idx=min(peak_step + 1, total_steps),
            )
            if carry_end is None:
                carry_end = total_steps
        else:
            rel_z = self._motion_data.object_pos_w[:, 2] - self._motion_data.root_pos_w[:, 2]
            z_min = float(np.min(rel_z))
            z_range = max(float(np.max(rel_z) - z_min), 0.0)
            threshold = z_min + max(0.10, z_range * 0.35)
            lifted_mask = rel_z >= threshold
            carry_start = _first_sustained_true_index(lifted_mask, consecutive_steps)
            if carry_start is None:
                lifted_indices = np.flatnonzero(lifted_mask)
                carry_start = int(lifted_indices[0]) if lifted_indices.size else int(np.argmax(rel_z))
            lowered_mask = rel_z < threshold
            carry_end = _first_sustained_true_index_from(
                lowered_mask,
                consecutive_steps,
                start_idx=min(int(carry_start) + 1, total_steps),
            )
            if carry_end is None:
                carry_end = total_steps

        carry_start = max(0, min(int(carry_start), total_steps))
        carry_end = max(carry_start, min(int(carry_end), total_steps))
        self._contact_aware_carry_window = (carry_start, carry_end)
        logger.info("Contact-aware sparse root command active window: [{}, {}) mode={}", carry_start, carry_end, mode)
        return self._contact_aware_carry_window

    def _get_sparse_target_root_trajectory_command_contact_aware(
        self,
        robot_state_data: np.ndarray,
        base_command: np.ndarray | None = None,
    ) -> np.ndarray:
        if base_command is None:
            base_command = self._get_sparse_target_root_trajectory_command(robot_state_data)
        if self._last_sparse_manual_enabled or self._motion_data is None or not self._motion_data.has_object:
            return base_command
        carry_start, carry_end = self._get_contact_aware_carry_window()
        if carry_start <= self._get_motion_index() < carry_end:
            return base_command
        zero_command = np.zeros_like(base_command, dtype=np.float32)
        self._record_sparse_root_command(
            base_command,
            zero_command,
            source="auto_contact_aware",
            mode="motion",
            manual_enabled=False,
        )
        return zero_command

    def _get_external_drop_button_override(self) -> np.ndarray | None:
        sub = self._manual_sparse_root_command_sub
        if sub is None:
            return None
        payload = sub.get_payload()
        if not isinstance(payload, dict) or "drop_button" not in payload:
            return None
        raw_value = payload.get("drop_button")
        try:
            drop_value = float(np.asarray(raw_value, dtype=np.float32).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            logger.warning("Ignoring malformed external drop button value: {}", raw_value)
            return None
        drop_value = 1.0 if drop_value >= 0.5 else 0.0
        if self._manual_drop_button_log_value != drop_value:
            logger.info("External drop button override: {}", int(drop_value))
            self._manual_drop_button_log_value = drop_value
        return np.array([[drop_value]], dtype=np.float32)

    def _get_drop_button(self) -> np.ndarray:
        external_drop_button = self._get_external_drop_button_override()
        if external_drop_button is not None:
            return external_drop_button
        if self._motion_data is None or not self._motion_data.has_object:
            return np.zeros((1, 1), dtype=np.float32)
        _, carry_end = self._get_contact_aware_carry_window()
        return np.array([[1.0 if self._get_motion_index() >= carry_end else 0.0]], dtype=np.float32)

    def _get_depth_distill_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        base_lin_vel = robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3]
        base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        sparse_command = self._get_sparse_target_root_trajectory_command(robot_state_data)
        contact_aware_sparse_command = (
            self._get_sparse_target_root_trajectory_command_contact_aware(robot_state_data, sparse_command)
            if self._uses_sparse_root_command_contact_aware
            else sparse_command
        )
        return {
            "sparse_target_root_trajectory_command": sparse_command,
            "sparse_target_root_trajectory_command_contact_aware": contact_aware_sparse_command,
            "drop_button": self._get_drop_button(),
            "base_lin_vel": base_lin_vel.astype(np.float32, copy=False),
            "base_ang_vel": base_ang_vel.astype(np.float32, copy=False),
            "dof_pos": (robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles).astype(
                np.float32,
                copy=False,
            ),
            "dof_vel": robot_state_data[
                :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
            ].astype(np.float32, copy=False),
            "actions": self.last_policy_action.astype(np.float32, copy=False),
        }

    def _get_object_mocap_distill_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        if self._motion_data is None:
            raise ValueError("Motion data is required for mocap object-distill observations.")
        if not self._motion_data.has_object:
            raise ValueError("Mocap object-distill observations require a motion clip with object pose data.")

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()

        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_observation_reference_pose_in_world(robot_state_data)
        sim_object_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        if sim_object_state is not None:
            current_object_pos_w = sim_object_state[:, :3]
            current_object_quat_wxyz = xyzw_to_wxyz(sim_object_state[:, 3:7])
        else:
            current_object_pos_w = self._motion_data.object_pos_w[idx : idx + 1]
            current_object_quat_wxyz = self._motion_data.object_quat_w[idx : idx + 1]

        obj_current_pos_b, obj_current_quat_b = self._pose_in_robot_ref_frame(
            robot_ref_pos_w,
            robot_ref_quat_wxyz,
            current_object_pos_w,
            current_object_quat_wxyz,
        )
        obj_current_rot6d = matrix_from_quat(obj_current_quat_b)[..., :2].reshape(1, -1)
        obj_current_size = self._motion_data.object_size[idx : idx + 1].astype(np.float32, copy=False)

        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            base_ang_vel = quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 10:13])
        else:
            base_ang_vel = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        return {
            "sparse_target_root_trajectory_command": self._get_sparse_target_root_trajectory_command(robot_state_data),
            "base_ang_vel": base_ang_vel.astype(np.float32, copy=False),
            "dof_pos": (robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles).astype(
                np.float32,
                copy=False,
            ),
            "dof_vel": robot_state_data[
                :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
            ].astype(np.float32, copy=False),
            "actions": self.last_policy_action.astype(np.float32, copy=False),
            "obj_current_pose_size_b": np.concatenate(
                [obj_current_pos_b, obj_current_rot6d, obj_current_size], axis=1
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

    def _get_legacy_object_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        obs = self._get_object_generalist_obs_buffer_dict(robot_state_data)
        obs.pop("obj_lin_vel_b", None)
        obs.pop("obj_ang_vel_b", None)
        return obs

    def _get_videomimic_obs_buffer_dict(self, robot_state_data):
        if self._motion_data is None:
            raise ValueError("Motion data is required for VideoMimic observations.")

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

    def get_current_obs_buffer_dict(self, robot_state_data):
        robot_state_data = self._augment_robot_state_with_sim_state(robot_state_data)
        if self._uses_videomimic:
            return self._get_videomimic_obs_buffer_dict(robot_state_data)
        if self._uses_object_mocap_distill:
            return self._get_object_mocap_distill_obs_buffer_dict(robot_state_data)
        if self._uses_object_generalist:
            return self._get_object_generalist_obs_buffer_dict(robot_state_data)
        if self._uses_legacy_object_obs:
            return self._get_legacy_object_obs_buffer_dict(robot_state_data)
        if self._uses_sparse_root_command:
            return self._get_depth_distill_obs_buffer_dict(robot_state_data)

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

        return current_obs_buffer_dict

    def _publish_waiting_policy_overlay(self, robot_state_data: np.ndarray) -> None:
        if not self._uses_sparse_root_command:
            return
        try:
            robot_state_data = self._augment_robot_state_with_sim_state(robot_state_data)
            self._get_sparse_target_root_trajectory_command(robot_state_data)
            self._publish_policy_overlay()
            self._logged_waiting_overlay_error = False
        except Exception as exc:
            if not getattr(self, "_logged_waiting_overlay_error", False):
                self.logger.warning("Unable to publish waiting sparse-root command overlay: {}", exc)
                self._logged_waiting_overlay_error = True

    def rl_inference(self, robot_state_data):
        self._maybe_start_pending_auto_motion_clip(robot_state_data)

        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None
            self._last_policy_control_clock_ms = None
            self._sim_time_control_schedule_index = 0
            self._last_policy_control_target_clock_ms = None
        elif self._should_skip_sim_time_control_tick():
            self._skip_next_lowcmd_publish = (
                os.environ.get("HOLOSOMA_POLICY_SUPPRESS_DUP_SIM_TIME_LOWCMD", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            return self.scaled_policy_action.copy()
        elif self.use_sim_time and not self._sim_time_control_schedule_ms:
            self._update_clock()
        self._skip_next_lowcmd_publish = False

        motion_index = self._get_motion_index()
        self._sync_motion_outputs_from_onnx(motion_index)
        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._time_step_input_name:
            action_time_step = motion_index if self._uses_motion_command else 0
            input_feed[self._time_step_input_name] = np.array([[action_time_step]], dtype=np.float32)
        perception_obs = None
        if self._perception_obs_input_name:
            perception_dim = self._get_onnx_input_dim(self._perception_obs_input_name)
            perception_obs = self._get_file_perception_obs(perception_dim)
            if perception_obs is None:
                perception_target_sim_time_ms = None
                if os.environ.get("HOLOSOMA_POLICY_ALIGN_PERCEPTION_TO_SIM_STATE", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    get_sim_time_ms = getattr(self.interface, "get_sim_time_ms", None)
                    if callable(get_sim_time_ms):
                        perception_target_sim_time_ms = get_sim_time_ms()
                        try:
                            perception_target_sim_time_ms += float(
                                os.environ.get("HOLOSOMA_POLICY_PERCEPTION_TARGET_OFFSET_MS", "0") or "0"
                            )
                        except ValueError:
                            pass
                perception_obs = self._get_split_perception_obs(
                    perception_dim,
                    target_sim_time_ms=perception_target_sim_time_ms,
                )
            input_feed[self._perception_obs_input_name] = perception_obs
        outputs = self.policy(input_feed)
        policy_action = outputs[self._action_output_name]
        action_override = self._get_file_policy_action()
        if action_override is not None:
            policy_action = action_override

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
        self.scaled_policy_action = policy_action * self.policy_action_scales
        if self._use_motion_command_as_q_target and self._uses_motion_command:
            target_joint_pos = np.asarray(self.motion_command_t[:, : self.num_dofs], dtype=np.float32)
            self.scaled_policy_action = target_joint_pos - self.default_dof_angles
            if not self._logged_motion_command_q_target:
                logger.info("Using motion_command joint_pos directly as MuJoCo q_target for diagnostic rollout.")
                self._logged_motion_command_q_target = True
        if self._use_motion_data_as_q_target and self._motion_data is not None:
            motion_index = self._get_motion_index()
            target_joint_pos = self._motion_data.joint_pos[motion_index : motion_index + 1].astype(np.float32, copy=False)
            self.scaled_policy_action = target_joint_pos - self.default_dof_angles
            if not self._logged_motion_data_q_target:
                logger.info("Using motion .npz joint_pos directly as MuJoCo q_target for diagnostic rollout.")
                self._logged_motion_data_q_target = True
        self._maybe_debug_policy_io(robot_state_data, obs["actor_obs"], perception_obs, policy_action)
        self._publish_policy_overlay()

        # update motion timestep
        if self.motion_clip_progressing:
            if not self.use_sim_time:
                self.motion_timestep += 1
            self._maybe_restart_sim_at_motion_end()
        return self.scaled_policy_action

    @staticmethod
    def _policy_debug_stats(values: np.ndarray, *, max_values: int = 8) -> dict:
        arr = np.asarray(values, dtype=np.float32)
        flat = arr.reshape(-1)
        finite = np.isfinite(flat)
        finite_vals = flat[finite]
        stats = {
            "shape": list(arr.shape),
            "finite": int(finite.sum()),
            "count": int(flat.size),
            "nonzero": int(np.count_nonzero(np.abs(flat[finite]) > 1.0e-7)) if finite_vals.size else 0,
            "first": flat[:max_values].astype(float).tolist(),
        }
        if finite_vals.size:
            stats.update(
                {
                    "min": float(finite_vals.min()),
                    "max": float(finite_vals.max()),
                    "mean": float(finite_vals.mean()),
                    "std": float(finite_vals.std()),
                    "p01": float(np.percentile(finite_vals, 1)),
                    "p50": float(np.percentile(finite_vals, 50)),
                    "p99": float(np.percentile(finite_vals, 99)),
                }
            )
        return stats

    @staticmethod
    def _policy_debug_depth_stats(values: np.ndarray | None) -> dict | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        stats = WholeBodyTrackingPolicy._policy_debug_stats(arr)
        if arr.size == 58 * 87:
            image = arr.reshape(58, 87)
            finite_image = np.where(np.isfinite(image), image, np.nan)
            min_idx = np.nanargmin(finite_image)
            max_idx = np.nanargmax(finite_image)
            row_mean = np.nanmean(finite_image, axis=1)
            stats.update(
                {
                    "image_shape": [58, 87],
                    "min_rc": [int(min_idx // 87), int(min_idx % 87)],
                    "max_rc": [int(max_idx // 87), int(max_idx % 87)],
                    "row_mean_argmin": int(np.nanargmin(row_mean)),
                    "row_mean_argmax": int(np.nanargmax(row_mean)),
                    "top_row_mean": float(np.nanmean(finite_image[0])),
                    "center_row_mean": float(np.nanmean(finite_image[29])),
                    "bottom_row_mean": float(np.nanmean(finite_image[-1])),
                    "center_value": float(finite_image[29, 43]),
                }
            )
        return stats

    def _policy_debug_torque_stats(
        self,
        *,
        q_actual: np.ndarray,
        dq_actual: np.ndarray,
        q_target: np.ndarray,
    ) -> tuple[dict, dict]:
        joint2motor = np.asarray(self.robot_config.joint2motor, dtype=np.int64)
        motor_kp = np.asarray(self.robot_config.motor_kp, dtype=np.float32)
        motor_kd = np.asarray(self.robot_config.motor_kd, dtype=np.float32)
        motor_effort = np.asarray(self.robot_config.motor_effort_limit, dtype=np.float32)

        joint_kp = motor_kp[joint2motor].reshape(1, -1)
        joint_kd = motor_kd[joint2motor].reshape(1, -1)
        joint_effort = motor_effort[joint2motor].reshape(1, -1)

        unclipped_tau = joint_kp * (q_target - q_actual) - joint_kd * dq_actual
        clipped_tau = np.clip(unclipped_tau, -joint_effort, joint_effort)
        sat_ratio = np.abs(clipped_tau) / np.maximum(joint_effort, 1.0e-6)

        top_idx = np.argsort(np.abs(unclipped_tau.reshape(-1)))[::-1][:8]
        stats = {
            "estimated_pd_tau_unclipped": self._policy_debug_stats(unclipped_tau),
            "estimated_pd_tau_clipped": self._policy_debug_stats(clipped_tau),
            "estimated_pd_tau_sat_ratio": self._policy_debug_stats(sat_ratio),
            "estimated_pd_tau_saturated_joint_count": int(np.count_nonzero(np.abs(unclipped_tau) >= joint_effort - 1.0e-5)),
        }
        top = {
            "estimated_pd_tau_top": [
                {
                    "joint": self.dof_names[int(idx)],
                    "q_error": float((q_target - q_actual).reshape(-1)[idx]),
                    "dq_actual": float(dq_actual.reshape(-1)[idx]),
                    "kp": float(joint_kp.reshape(-1)[idx]),
                    "kd": float(joint_kd.reshape(-1)[idx]),
                    "effort_limit": float(joint_effort.reshape(-1)[idx]),
                    "tau_unclipped": float(unclipped_tau.reshape(-1)[idx]),
                    "tau_clipped": float(clipped_tau.reshape(-1)[idx]),
                    "sat_ratio": float(sat_ratio.reshape(-1)[idx]),
                }
                for idx in top_idx
            ]
        }
        return stats, top

    def _maybe_debug_policy_io(
        self,
        robot_state_data: np.ndarray,
        actor_obs: np.ndarray,
        perception_obs: np.ndarray | None,
        policy_action: np.ndarray,
    ) -> None:
        if self._policy_debug_path is None or self._policy_debug_count >= self._policy_debug_limit:
            return
        if not self._policy_debug_initialized:
            self._policy_debug_path.parent.mkdir(parents=True, exist_ok=True)
            self._policy_debug_path.write_text("")
            self._policy_debug_initialized = True

        q_actual = np.asarray(robot_state_data[:, 7 : 7 + self.num_dofs], dtype=np.float32)
        dq_actual = np.asarray(
            robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            dtype=np.float32,
        )
        q_target = self.default_dof_angles.reshape(1, -1).astype(np.float32) + self.scaled_policy_action.astype(
            np.float32
        )
        q_target_clipped = q_target.copy()
        if self._clip_joint_targets and self.q_min_arr is not None and self.q_max_arr is not None:
            np.clip(q_target_clipped[0], self.q_min_arr, self.q_max_arr, out=q_target_clipped[0])
        q_error = (q_target_clipped - q_actual).reshape(-1)
        top_idx = np.argsort(np.abs(q_error))[::-1][:8]
        torque_stats, torque_top = self._policy_debug_torque_stats(
            q_actual=q_actual,
            dq_actual=dq_actual,
            q_target=q_target_clipped,
        )

        current_obs = getattr(self, "_last_current_obs_buffer_dict", {})
        record = {
            "count": int(self._policy_debug_count),
            "motion_timestep": int(self.motion_timestep),
            "motion_index": int(self._get_motion_index()),
            "clock_ms": int(self.clock_sub.get_clock()) if self.use_sim_time else None,
            "control_target_clock_ms": (
                int(self._last_policy_control_target_clock_ms)
                if self._last_policy_control_target_clock_ms is not None
                else None
            ),
            "sim_time_ms": (
                float(self.interface.get_sim_time_ms())
                if callable(getattr(self.interface, "get_sim_time_ms", None))
                and self.interface.get_sim_time_ms() is not None
                else None
            ),
            "control_schedule_index": int(self._sim_time_control_schedule_index),
            "motion_clip_progressing": bool(self.motion_clip_progressing),
            "actor_obs": self._policy_debug_stats(actor_obs),
            "perception_obs": self._policy_debug_depth_stats(perception_obs),
            "policy_action_raw": self._policy_debug_stats(policy_action),
            "policy_action_scaled": self._policy_debug_stats(self.scaled_policy_action),
            "q_actual_first": q_actual.reshape(-1)[:8].astype(float).tolist(),
            "dq_actual_first": dq_actual.reshape(-1)[:8].astype(float).tolist(),
            "q_target_first": q_target_clipped.reshape(-1)[:8].astype(float).tolist(),
            "q_target_minus_actual": self._policy_debug_stats(q_error),
            "q_error_top": [
                {
                    "joint": self.dof_names[int(idx)],
                    "actual": float(q_actual.reshape(-1)[idx]),
                    "target": float(q_target_clipped.reshape(-1)[idx]),
                    "error": float(q_error[idx]),
                    "raw_action": float(policy_action.reshape(-1)[idx]),
                    "scaled_action": float(self.scaled_policy_action.reshape(-1)[idx]),
                }
                for idx in top_idx
            ],
            "robot_root": np.asarray(robot_state_data[:, :7], dtype=np.float32).reshape(-1).astype(float).tolist(),
        }
        record.update(torque_stats)
        record.update(torque_top)
        if self._motion_data is not None:
            idx = self._get_motion_index()
            record["motion_root"] = self._motion_data.root_pos_w[idx].astype(float).tolist()
            record["motion_q_first"] = self._motion_data.joint_pos[idx, :8].astype(float).tolist()
        for key in ("sparse_target_root_trajectory_command", "base_ang_vel", "dof_pos", "dof_vel"):
            if key in current_obs:
                record[key] = self._policy_debug_stats(current_obs[key])
        if self._policy_debug_include_values:
            record["actor_obs_values"] = np.asarray(actor_obs, dtype=np.float32).reshape(-1).astype(float).tolist()
            record["perception_obs_values"] = (
                np.asarray(perception_obs, dtype=np.float32).reshape(-1).astype(float).tolist()
                if perception_obs is not None
                else None
            )
            record["policy_action_raw_values"] = (
                np.asarray(policy_action, dtype=np.float32).reshape(-1).astype(float).tolist()
            )
            record["policy_action_scaled_values"] = (
                np.asarray(self.scaled_policy_action, dtype=np.float32).reshape(-1).astype(float).tolist()
            )
            record["q_actual_values"] = q_actual.reshape(-1).astype(float).tolist()
            record["dq_actual_values"] = dq_actual.reshape(-1).astype(float).tolist()
            record["q_target_values"] = q_target_clipped.reshape(-1).astype(float).tolist()

        with self._policy_debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._policy_debug_count += 1

    def _maybe_restart_sim_at_motion_end(self) -> None:
        if not bool(getattr(self.config.task, "restart_sim_on_motion_end", False)):
            return
        if self._motion_data is None or self._motion_data.frame_count <= 0:
            return
        if self._motion_end_reset_requested:
            return
        if int(self.motion_timestep) < self._motion_data.frame_count - 1:
            return

        if self._disable_motion_end_sim_reset:
            self.motion_timestep = min(int(self.motion_timestep), max(self._motion_data.frame_count - 1, 0))
            self._motion_end_reset_requested = True
            self.logger.info("Motion clip reached the end; automatic simulator reset is disabled.")
            return

        self._motion_end_reset_requested = True
        sim_control_pub = getattr(self.interface, "_sim_control_pub", None)
        if sim_control_pub is not None and hasattr(sim_control_pub, "request_reset"):
            sim_control_pub.request_reset("motion_end")
            self.logger.info("Motion clip reached the end; requested simulator reset and restarting clip.")
        else:
            self.logger.warning("Motion clip reached the end, but simulator reset channel is unavailable.")
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        self._handle_start_motion_clip()
        self._motion_end_reset_requested = False

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

    def _after_auto_start_policy(self) -> None:
        if self._auto_start_motion_clip_pending:
            self._auto_start_motion_clip_hold_start_time = None

    def _maybe_start_pending_auto_motion_clip(self, robot_state_data: np.ndarray) -> None:
        if not self._auto_start_motion_clip_pending or not self.use_policy_action or self.motion_clip_progressing:
            return

        hold_sec = max(0.0, float(getattr(self.config.task, "auto_start_stiff_hold_sec", 0.0) or 0.0))
        max_wait_sec = max(0.0, float(getattr(self.config.task, "auto_start_stiff_max_wait_sec", 0.0) or 0.0))
        pose_tolerance = max(
            0.0,
            float(getattr(self.config.task, "auto_start_stiff_pose_tolerance", 0.0) or 0.0),
        )
        now = time.perf_counter()
        if self._auto_start_motion_clip_hold_start_time is None:
            self._auto_start_motion_clip_hold_start_time = now
            self._auto_start_motion_clip_last_log_time = 0.0
            self.logger.info("Policy auto-started; holding motion frame 0 before starting motion clip.")

        elapsed = now - self._auto_start_motion_clip_hold_start_time
        motion_index = self._get_motion_index()
        dof_err = None
        if self._motion_data is not None:
            target = self._motion_data.joint_pos[motion_index : motion_index + 1]
            current = robot_state_data[:, 7 : 7 + self.num_dofs]
            dof_err = float(np.max(np.abs(current - target)))

        waited_long_enough = elapsed >= hold_sec
        pose_ready = dof_err is None or dof_err <= pose_tolerance
        timed_out = max_wait_sec > 0.0 and elapsed >= max_wait_sec
        if waited_long_enough and (pose_ready or timed_out or max_wait_sec <= 0.0):
            self._auto_start_motion_clip_pending = False
            self._auto_start_motion_clip_hold_start_time = None
            self.logger.info(
                "Starting auto motion clip after policy warmup: elapsed={:.2f}s dof_err={}",
                elapsed,
                "n/a" if dof_err is None else f"{dof_err:.4f}",
            )
            self._handle_start_motion_clip()
        elif now - self._auto_start_motion_clip_last_log_time >= 1.0:
            self._auto_start_motion_clip_last_log_time = now
            self.logger.info(
                "Waiting before auto motion clip: elapsed={:.2f}s dof_err={} target_hold={:.2f}s max_wait={:.2f}s",
                elapsed,
                "n/a" if dof_err is None else f"{dof_err:.4f}",
                hold_sec,
                max_wait_sec,
            )

    def _load_sim_time_control_schedule(self) -> list[int]:
        """Load an optional debug schedule of simulator millisecond ticks for policy inference."""
        path_raw = os.environ.get("HOLOSOMA_POLICY_CONTROL_SCHEDULE_MS_FILE", "").strip()
        if not path_raw:
            return []
        try:
            values = json.loads(Path(path_raw).expanduser().read_text())
            schedule = [int(round(float(value))) for value in values]
        except Exception as exc:
            logger.warning("Failed to load policy control schedule '{}': {}", path_raw, exc)
            return []
        schedule = [value for value in schedule if value >= 0]
        if not schedule:
            logger.warning("Ignoring empty policy control schedule '{}'", path_raw)
            return []
        logger.info("Loaded {} sim-time policy control schedule ticks from {}", len(schedule), path_raw)
        return schedule

    def _should_skip_sim_time_control_tick(self) -> bool:
        """Gate ONNX inference on simulator time when MuJoCo runs slower than wall clock."""
        if not self.use_sim_time:
            return False

        current_clock = int(self.clock_sub.get_clock())
        if self._sim_time_control_schedule_ms:
            get_sim_time_ms = getattr(self.interface, "get_sim_time_ms", None)
            if callable(get_sim_time_ms):
                try:
                    sim_time_ms = get_sim_time_ms()
                    if sim_time_ms is not None:
                        current_clock = int(round(float(sim_time_ms)))
                except (TypeError, ValueError):
                    pass
            index = min(self._sim_time_control_schedule_index, len(self._sim_time_control_schedule_ms) - 1)
            target_clock = int(self._sim_time_control_schedule_ms[index])
            if current_clock < target_clock:
                return True
            motion_timestep = int(self._sim_time_control_schedule_index)
            if self._disable_motion_end_sim_reset and self._motion_data is not None:
                motion_timestep = min(motion_timestep, max(self._motion_data.frame_count - 1, 0))
            self.motion_timestep = motion_timestep
            self._last_policy_control_target_clock_ms = target_clock
            self._last_policy_control_clock_ms = current_clock
            self._last_clock_reading = current_clock
            if self.motion_start_timestep is None:
                self.motion_start_timestep = current_clock - int(round(motion_timestep * self.timestep_interval_ms))
            self._sim_time_control_schedule_index += 1
            return False

        if self._last_policy_control_clock_ms is None:
            self._last_policy_control_clock_ms = current_clock
            return False

        if current_clock < self._last_policy_control_clock_ms:
            self._last_policy_control_clock_ms = current_clock
            return False

        interval_ms = max(1, int(round(self.timestep_interval_ms)))
        elapsed_ms = current_clock - self._last_policy_control_clock_ms
        tolerance_ms = int(os.environ.get("HOLOSOMA_POLICY_SIM_TIME_TOLERANCE_MS", "1") or "1")
        if elapsed_ms + max(0, tolerance_ms) < interval_ms:
            return True

        completed_intervals = max(1, elapsed_ms // interval_ms)
        self._last_policy_control_clock_ms += completed_intervals * interval_ms
        return False

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)
        if self._motion_alignment_enabled:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(self._augment_robot_state_with_sim_state(robot_state_data))

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        current_clock = self.clock_sub.get_clock()
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            if bool(getattr(self.config.task, "restart_motion_on_clock_reset", False)):
                self.logger.warning("Clock sync returned earlier timestamp; restarting motion clip from frame 0.")
                self._handle_start_motion_clip()
                current_clock = self.clock_sub.get_clock()
            else:
                # Simulator clock jumped backwards (e.g., reset). Re-anchor start time while preserving progress.
                offset_ms = round(self.motion_timestep * self.timestep_interval_ms)
                self.logger.warning("Clock sync returned earlier timestamp; adjusting motion timing anchor.")
                self.motion_start_timestep = current_clock - offset_ms
        if self.motion_start_timestep is None:
            self.motion_start_timestep = current_clock
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
        if self._disable_motion_end_sim_reset and self._motion_data is not None:
            self.motion_timestep = min(self.motion_timestep, max(self._motion_data.frame_count - 1, 0))
        if self.motion_timestep != previous_motion_timestep and self.motion_timestep % 50 == 0:
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
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._motion_end_reset_requested = False

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self._logged_root_reference_clip_start = False
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        if self._motion_alignment_enabled:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(self._augment_robot_state_with_sim_state(robot_state_data))
        if self._prefill_obs_history_on_motion_start:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None and self._has_valid_robot_state(robot_state_data):
                motion_index = self._get_motion_index()
                self._sync_motion_outputs_from_onnx(motion_index)
                self._prefill_obs_history(robot_state_data)
                if not self._logged_motion_start_history_prefill:
                    self.logger.info("Prefilled observation history at motion start with frame {}.", motion_index)
                    self._logged_motion_start_history_prefill = True
        self.logger.info(colored("Starting motion clip", "blue"))

    def handle_keyboard_button(self, keycode):
        """Add new keyboard button to start and end the motion clips"""
        key = str(keycode).lower()
        if self._keyboard_sparse_root_command_enabled and key in {"w", "s", "a", "d", "q", "e"}:
            with self._keyboard_sparse_root_lock:
                self._keyboard_sparse_root_pressed_keys.add(key)
            return
        if key in {"space", " ", "s"}:
            self.clock_sub.reset_origin()
            self._handle_start_motion_clip()
        else:
            super().handle_keyboard_button(keycode)

    def handle_keyboard_release(self, keycode):
        key = str(keycode).lower()
        if self._keyboard_sparse_root_command_enabled and key in {"w", "s", "a", "d", "q", "e"}:
            with self._keyboard_sparse_root_lock:
                self._keyboard_sparse_root_pressed_keys.discard(key)
            return
        super().handle_keyboard_release(keycode)

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
