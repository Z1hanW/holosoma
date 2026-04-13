import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored
from holosoma.utils.object_pose_correction import (
    apply_omomo_largebox_center_offset_wxyz_np,
    apply_omomo_largebox_ground_contact_wxyz_np,
    apply_omomo_largebox_primitive_local_alignment_wxyz_np,
    apply_omomo_largebox_zup_correction_wxyz_np,
    get_omomo_largebox_primitive_extents_xyz_np,
    is_omomo_largebox_clip,
)

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
    quat_to_rpy,
    quat_rotate_inverse,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from holosoma_inference.utils.perception_obs import PerceptionObsSub
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
            object_name = str(np.asarray(data["object_name"]).item()).strip() if "object_name" in data else ""
            object_urdf_path = (
                str(np.asarray(data["object_urdf_path"]).item()).strip() if "object_urdf_path" in data else ""
            )
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
        if self.has_object and is_omomo_largebox_clip(
            motion_path.stem,
            object_name=object_name,
            object_urdf_path=object_urdf_path,
        ):
            object_quat_w = apply_omomo_largebox_zup_correction_wxyz_np(object_quat_w)
            object_quat_w = apply_omomo_largebox_primitive_local_alignment_wxyz_np(object_quat_w)
            object_pos_w = apply_omomo_largebox_center_offset_wxyz_np(object_pos_w, object_quat_w)
            object_pos_w = apply_omomo_largebox_ground_contact_wxyz_np(object_pos_w, object_quat_w)
            self.object_size = np.repeat(
                get_omomo_largebox_primitive_extents_xyz_np().reshape(1, 3),
                repeats=self.frame_count,
                axis=0,
            )
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
        self._last_motion_output_timestep: int | None = None

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
        self._latest_sim_state: dict | None = None
        self._sim_state_sub: SimStateSub | None = None
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._logged_sim_ref_from_sim_state = False
        self._auto_start_motion_clip_pending = False
        self._logged_first_policy_step_debug = False
        self._auto_start_stage: str | None = None
        self._auto_start_hold_ticks = 0
        self._auto_start_max_wait_ticks = 0
        self._auto_start_tick_count = 0
        self._training_freeze_zero_prob = 0.0
        self._training_freeze_zero_extra_holds = 0
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self._auto_start_pose_tolerance = float(getattr(config.task, "auto_start_stiff_pose_tolerance", 0.12))
        self._auto_start_rearm_requested = False
        self._auto_start_force_motion_start_pose = False
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._suppress_root_reference_at_clip_start = False
        self._warm_autostart_obs_history = os.getenv("HOLOSOMA_WARM_AUTOSTART_OBS_HISTORY", "1") != "0"
        self._freeze_autostart_obs_snapshot = os.getenv("HOLOSOMA_FREEZE_AUTOSTART_OBS_SNAPSHOT", "1") != "0"
        self._dryrun_autostart_policy_history = os.getenv("HOLOSOMA_DRYRUN_AUTOSTART_POLICY_HISTORY", "1") != "0"
        self._autostart_policy_history_prime_steps_override = (
            os.getenv("HOLOSOMA_AUTOSTART_POLICY_DRYRUN_STEPS", "").strip()
        )
        self._auto_start_history_snapshot: dict[str, dict[str, np.ndarray]] | None = None

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
        self._uses_sparse_root_distill = "sparse_target_root_trajectory_command" in obs_terms
        if self._uses_sparse_root_distill:
            self._uses_motion_command = True
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
        self._perception_input_name: str | None = None
        self._perception_input_dim: int | None = None
        self._time_step_input_name: str | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._motion_output_names: set[str] = set()
        self._motion_alignment_enabled = False
        self._perception_obs_sub: PerceptionObsSub | None = None
        self._last_perception_obs: np.ndarray | None = None
        self._logged_waiting_for_perception_obs = False

        super().__init__(config)

        if self.config.task.use_sim_state:
            self._sim_state_sub = SimStateSub(port=self.config.task.sim_state_port)
            self._sim_state_sub.start()

        if self._perception_input_name is not None:
            if not bool(getattr(self.config.task, "use_split_perception_obs", False)):
                raise ValueError(
                    "Model expects 'perception_obs'; enable --task.use-split-perception-obs for split sim2sim inference."
                )
            self._perception_obs_sub = PerceptionObsSub(port=self.config.task.perception_obs_port)
            self._perception_obs_sub.start()

        if self.use_policy_action:
            self._handle_start_policy()

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if bool(getattr(self.config.task, "use_zmq_lowcmd", False)) and self.motion_command_0 is not None:
            # Split sim resets into the clip start pose; holding that pose keeps the carried
            # object aligned with the robot before the rollout actually begins.
            self._stiff_hold_q = self.motion_command_0[:, : self.num_dofs].astype(np.float32, copy=True)

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

        if self.config.task.auto_start_motion:
            self._handle_start_motion_clip()
        elif self.config.task.auto_start_motion_clip:
            self._auto_start_motion_clip_pending = False

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
        if self._suppress_root_reference_at_clip_start:
            return False
        use_root = self._remaining_root_reference_clip_start_obs > 0 and int(self._get_motion_index()) == 0
        if use_root and not self._logged_root_reference_clip_start:
            logger.info("Using robot root as observation reference at clip start to match training step-0 semantics.")
            self._logged_root_reference_clip_start = True
        return use_root

    def _consume_root_reference_at_clip_start(self) -> None:
        if self._remaining_root_reference_clip_start_obs <= 0:
            return
        if int(self._get_motion_index()) != 0:
            self._remaining_root_reference_clip_start_obs = 0
            return
        self._remaining_root_reference_clip_start_obs = 0

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

    def _should_auto_start_policy_immediately(self) -> bool:
        return not bool(getattr(self.config.task, "auto_start_motion_clip", False))

    def _set_stiff_hold_target_for_autostart(self) -> None:
        if self._auto_start_force_motion_start_pose and self.motion_command_0 is not None:
            target_q = np.asarray(self.motion_command_0[:, : self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self._auto_start_force_motion_start_pose = False
                self.logger.info("Auto-start stiff target forced to motion start pose after simulator reset.")
                return

        state = self._augment_robot_state_with_sim_state(self.interface.get_low_state())
        if state is not None and state.shape[1] >= 7 + self.num_dofs:
            target_q = np.asarray(state[:, 7 : 7 + self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self._auto_start_force_motion_start_pose = False
                self.logger.info("Auto-start stiff target locked to current initialized robot pose.")
                return

        if self.motion_command_0 is not None:
            target_q = np.asarray(self.motion_command_0[:, : self.num_dofs], dtype=np.float32)
            if target_q.shape == self._stiff_hold_q.shape:
                self._stiff_hold_q = target_q.copy()
                self._auto_start_force_motion_start_pose = False
                self.logger.warning("Auto-start stiff target fallback to ONNX motion start command.")

    def _stiff_hold_pose_error(self) -> float | None:
        state = self.interface.get_low_state()
        if not self._has_valid_robot_state(state):
            return None
        dof_pos = state[:, 7 : 7 + self.num_dofs]
        if dof_pos.shape != self._stiff_hold_q.shape:
            return None
        return float(np.max(np.abs(dof_pos - self._stiff_hold_q)))

    def _warm_auto_start_observation_history(self, robot_state_data: np.ndarray) -> None:
        if not self._warm_autostart_obs_history:
            return
        if self._dryrun_autostart_policy_history:
            return
        if self._obs_input_name is None:
            return
        try:
            self.last_policy_action.fill(0.0)
            if self._freeze_autostart_obs_snapshot:
                if self._auto_start_history_snapshot is None:
                    current_obs_buffer_dict = self.get_current_obs_buffer_dict(
                        self._augment_robot_state_with_sim_state(robot_state_data)
                    )
                    self._auto_start_history_snapshot = self.parse_current_obs_dict(current_obs_buffer_dict)
                snapshot = {
                    group: {term: value.copy() for term, value in term_dict.items()}
                    for group, term_dict in self._auto_start_history_snapshot.items()
                }
                self._update_obs_history(snapshot)
            else:
                self._prepare_group_observations(self._augment_robot_state_with_sim_state(robot_state_data))
        except Exception as exc:
            if not hasattr(self, "_logged_auto_start_history_warmup_error"):
                self.logger.warning("Failed to warm auto-start observation history: {}", exc)
                self._logged_auto_start_history_warmup_error = True

    def _get_autostart_policy_history_prime_steps(self) -> int:
        override = self._autostart_policy_history_prime_steps_override
        if override:
            try:
                return max(0, int(override))
            except ValueError:
                if not hasattr(self, "_logged_invalid_autostart_policy_history_prime_steps", False):
                    self.logger.warning(
                        "Ignoring invalid HOLOSOMA_AUTOSTART_POLICY_DRYRUN_STEPS={!r}",
                        override,
                    )
                    self._logged_invalid_autostart_policy_history_prime_steps = True
                return 0
        history_len = int(self.history_length_dict.get("actor_obs", 1))
        return max(0, history_len - 1)

    def _prime_auto_start_policy_history(self, robot_state_data: np.ndarray) -> bool:
        if not self._dryrun_autostart_policy_history:
            return False
        if not self._warm_autostart_obs_history:
            return False
        if self._obs_input_name is None or self._action_output_name is None:
            return False

        prime_steps = self._get_autostart_policy_history_prime_steps()
        if prime_steps <= 0:
            return False

        augmented_state = self._augment_robot_state_with_sim_state(robot_state_data)
        if augmented_state is None:
            return False

        perception_obs: np.ndarray | None = None
        if self._perception_input_name is not None:
            perception_obs = self._get_split_perception_obs()
            if perception_obs is None:
                if not hasattr(self, "_logged_auto_start_history_prime_waiting_for_perception_obs"):
                    self.logger.info("Skipping auto-start policy-history priming until split perception obs is available.")
                    self._logged_auto_start_history_prime_waiting_for_perception_obs = True
                return False

        self._reset_observation_history_state()
        self._auto_start_history_snapshot = None
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_motion_output_timestep = None
        if self.motion_command_0 is not None:
            self.motion_command_t = self.motion_command_0.copy()
        if self.ref_quat_xyzw_0 is not None:
            self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self._refresh_motion_outputs_for_current_timestep()
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = (
            1 if bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)) else 0
        )

        seed_obs = self.prepare_obs_for_rl(augmented_state)
        input_feed = {self._obs_input_name: seed_obs["actor_obs"]}
        if self._perception_input_name is not None and perception_obs is not None:
            input_feed[self._perception_input_name] = perception_obs
        if self._time_step_input_name is not None:
            input_feed[self._time_step_input_name] = np.array([[0]], dtype=np.float32)

        outputs = self.policy(input_feed)
        seed_action = np.clip(outputs[self._action_output_name], -100, 100)
        if self._uses_motion_command and not self._should_source_motion_outputs_from_motion_data():
            joint_pos = outputs.get("joint_pos")
            joint_vel = outputs.get("joint_vel")
            if joint_pos is not None and joint_vel is not None:
                self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
                self.ref_quat_xyzw_t = outputs.get("ref_quat_xyzw", self.ref_quat_xyzw_t)
                self.ref_pos_xyz_t = outputs.get("ref_pos_xyz", self.ref_pos_xyz_t)

        self.last_policy_action = seed_action.copy()
        self.scaled_policy_action = seed_action * self.policy_action_scales
        self._consume_root_reference_at_clip_start()

        for _ in range(max(0, prime_steps - 1)):
            self.prepare_obs_for_rl(augmented_state)

        self._preserve_obs_history_on_next_motion_start = True
        self._preserve_root_reference_state_on_next_motion_start = True
        self.logger.info(
            "Primed auto-start policy history with seeded action over {} actor steps at motion timestep 0.",
            prime_steps,
        )
        return True

    def _maybe_auto_start_rollout(self) -> None:
        if not getattr(self.config.task, "auto_start_motion_clip", False):
            return

        self._reset_observation_history_state()
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._suppress_root_reference_at_clip_start = False
        self._auto_start_history_snapshot = None
        self._auto_start_motion_clip_pending = False
        self._pending_noninteractive_policy_start = False
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_motion_output_timestep = 0
        if self.motion_command_0 is not None:
            self.motion_command_t = self.motion_command_0.copy()
        if self.ref_quat_xyzw_0 is not None:
            self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self._set_stiff_hold_target_for_autostart()
        self._stiff_hold_active = True
        self.use_policy_action = False
        self.get_ready_state = False
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

        state = self.interface.get_low_state()
        if not self._has_valid_robot_state(state):
            return

        self._warm_auto_start_observation_history(state)
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

        self._preserve_obs_history_on_next_motion_start = self._warm_autostart_obs_history and (
            not self._dryrun_autostart_policy_history
        )
        self._preserve_root_reference_state_on_next_motion_start = False
        self._suppress_root_reference_at_clip_start = False
        self._auto_start_history_snapshot = None
        self._auto_start_stage = None
        self._handle_start_policy()
        self._prime_auto_start_policy_history(state)
        self._handle_start_motion_clip()

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
        self._maybe_apply_training_motion_transitions_to_motion_data(metadata, ref_name)
        self._motion_cfg = motion_cfg or {}
        self._motion_alignment_enabled = bool((motion_cfg or {}).get("align_motion_to_init_yaw", False))
        freeze_prob_raw = (motion_cfg or {}).get("freeze_at_timestep_zero_prob", 0.0)
        try:
            freeze_prob = float(freeze_prob_raw or 0.0)
        except (TypeError, ValueError):
            freeze_prob = 0.0
        self._training_freeze_zero_prob = min(max(freeze_prob, 0.0), 0.999)
        freeze_holds_override = os.environ.get("HOLOSOMA_TRAINING_FREEZE_ZERO_EXTRA_HOLDS")
        if freeze_holds_override not in (None, ""):
            try:
                self._training_freeze_zero_extra_holds = max(0, int(freeze_holds_override))
            except ValueError:
                self._training_freeze_zero_extra_holds = 0
                logger.warning(
                    "Ignoring invalid HOLOSOMA_TRAINING_FREEZE_ZERO_EXTRA_HOLDS={!r}",
                    freeze_holds_override,
                )
            else:
                logger.info(
                    "Overriding training-like timestep-0 extra holds to {} via environment.",
                    self._training_freeze_zero_extra_holds,
                )
        elif self._training_freeze_zero_prob > 0.0:
            # Training can keep timestep 0 for multiple actor steps; use the geometric expectation
            # as a deterministic approximation for sim2sim verification.
            self._training_freeze_zero_extra_holds = int(
                min(200, round(self._training_freeze_zero_prob / max(1e-6, 1.0 - self._training_freeze_zero_prob)))
            )
        else:
            self._training_freeze_zero_extra_holds = 0
        self._training_freeze_zero_remaining_holds = 0

    def _should_source_motion_outputs_from_motion_data(self) -> bool:
        return bool(
            self._uses_motion_command
            and self._motion_data is not None
            and bool(getattr(self.config.task, "apply_training_motion_transitions", False))
        )

    def _get_motion_outputs_from_motion_data(self, motion_timestep: int) -> dict[str, np.ndarray] | None:
        if self._motion_data is None:
            return None

        idx = max(0, min(int(motion_timestep), self._motion_data.frame_count - 1))
        return {
            "joint_pos": self._motion_data.joint_pos[idx : idx + 1].astype(np.float32, copy=False),
            "joint_vel": self._motion_data.joint_vel[idx : idx + 1].astype(np.float32, copy=False),
            "ref_quat_xyzw": wxyz_to_xyzw(self._motion_data.ref_quat_w[idx : idx + 1]).astype(np.float32, copy=False),
            "ref_pos_xyz": self._motion_data.ref_pos_w[idx : idx + 1].astype(np.float32, copy=False),
        }

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
            or self._uses_object_generalist
            or self._uses_legacy_object_obs
            or self._uses_sparse_root_distill
            or bool(getattr(self.config.task, "apply_training_motion_transitions", False))
        ):
            self._load_motion_data_from_metadata(metadata, Path(model_path))

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported ONNX inputs: {self.onnx_input_names}")

        self._perception_input_name = "perception_obs" if "perception_obs" in self.onnx_input_names else None
        self._perception_input_dim = self._get_onnx_input_dim(self._perception_input_name)
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
            if self._should_source_motion_outputs_from_motion_data():
                outputs = self._get_motion_outputs_from_motion_data(0)
                if outputs is None:
                    raise ValueError("Motion data was expected for training-aligned motion outputs but is unavailable.")
            else:
                time_step = np.zeros((1, 1), dtype=np.float32)
                obs = self._build_zero_actor_obs()
                input_feed = {self._obs_input_name: obs}
                if self._perception_input_name:
                    input_feed[self._perception_input_name] = self._build_zero_perception_obs()
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
            self._last_motion_output_timestep = 0
        elif (
            self._uses_videomimic
            or self._uses_object_generalist
            or self._uses_legacy_object_obs
            or self._uses_sparse_root_distill
        ) and self._motion_data is not None:
            joint_pos = self._motion_data.joint_pos[:1]
            joint_vel = self._motion_data.joint_vel[:1]
            self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
            self.motion_command_0 = self.motion_command_t.copy()
            ref_quat_wxyz = self._motion_data.ref_quat_w[:1]
            self.ref_quat_xyzw_t = wxyz_to_xyzw(ref_quat_wxyz)
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self.ref_pos_xyz_t = self._motion_data.ref_pos_w[:1]
            self._last_motion_output_timestep = 0

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

    def _build_zero_actor_obs(self) -> np.ndarray:
        obs_dim = self._onnx_obs_dim
        if obs_dim is None:
            obs_dim = int(sum(int(template.shape[1]) for template in self.obs_buf_dict.values()))
        return np.zeros((1, int(obs_dim)), dtype=np.float32)

    def _build_zero_perception_obs(self) -> np.ndarray:
        return np.zeros((1, int(self._perception_input_dim or 0)), dtype=np.float32)

    def _query_motion_outputs_at(self, motion_timestep: int) -> dict[str, np.ndarray] | None:
        if self._should_source_motion_outputs_from_motion_data():
            return self._get_motion_outputs_from_motion_data(motion_timestep)

        if self._obs_input_name is None or "joint_pos" not in self.onnx_output_names or "joint_vel" not in self.onnx_output_names:
            return None

        input_feed = {self._obs_input_name: self._build_zero_actor_obs()}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[int(motion_timestep)]], dtype=np.float32)
        if self._perception_input_name:
            input_feed[self._perception_input_name] = self._build_zero_perception_obs()

        fetch_names = ["joint_pos", "joint_vel"]
        if "ref_quat_xyzw" in self.onnx_output_names:
            fetch_names.append("ref_quat_xyzw")
        if "ref_pos_xyz" in self.onnx_output_names:
            fetch_names.append("ref_pos_xyz")

        outputs = self.onnx_policy_session.run(fetch_names, input_feed)
        return dict(zip(fetch_names, outputs))

    def _refresh_motion_outputs_for_current_timestep(self) -> None:
        if not self._uses_motion_command or self._time_step_input_name is None:
            return

        motion_timestep = self._get_motion_index()
        if self._last_motion_output_timestep == motion_timestep and self.motion_command_t is not None:
            return

        outputs = self._query_motion_outputs_at(motion_timestep)
        if outputs is None:
            return

        joint_pos = outputs.get("joint_pos")
        joint_vel = outputs.get("joint_vel")
        if joint_pos is None or joint_vel is None:
            return

        self.motion_command_t = np.concatenate(
            [np.asarray(joint_pos, dtype=np.float32), np.asarray(joint_vel, dtype=np.float32)],
            axis=1,
        )
        ref_quat_xyzw = outputs.get("ref_quat_xyzw")
        if ref_quat_xyzw is not None:
            self.ref_quat_xyzw_t = np.asarray(ref_quat_xyzw, dtype=np.float32)
        ref_pos_xyz = outputs.get("ref_pos_xyz")
        if ref_pos_xyz is not None:
            self.ref_pos_xyz_t = np.asarray(ref_pos_xyz, dtype=np.float32)
        self._last_motion_output_timestep = int(motion_timestep)

    @staticmethod
    def _preview_array(values: np.ndarray | None, count: int = 6) -> list[float] | None:
        if values is None:
            return None
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []
        return np.round(arr[:count], 4).tolist()

    def _build_first_step_actor_obs_with_overrides(
        self,
        robot_state_data: np.ndarray,
        *,
        use_root_reference_at_clip_start: bool,
        prefer_sim_ref_from_sim_state: bool,
        motion_timestep_override: int | None = None,
    ) -> np.ndarray:
        old_use_root_reference = bool(getattr(self.config.task, "use_root_reference_at_clip_start", False))
        old_prefer_sim_ref = bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False))
        old_motion_timestep = int(self.motion_timestep)
        old_motion_command_t = None if self.motion_command_t is None else self.motion_command_t.copy()
        old_ref_quat_xyzw_t = None if self.ref_quat_xyzw_t is None else self.ref_quat_xyzw_t.copy()
        old_ref_pos_xyz_t = None if self.ref_pos_xyz_t is None else self.ref_pos_xyz_t.copy()
        old_last_motion_output_timestep = self._last_motion_output_timestep
        try:
            object.__setattr__(self.config.task, "use_root_reference_at_clip_start", use_root_reference_at_clip_start)
            object.__setattr__(self.config.task, "prefer_sim_ref_from_sim_state", prefer_sim_ref_from_sim_state)
            if motion_timestep_override is not None:
                self.motion_timestep = int(motion_timestep_override)
                self._last_motion_output_timestep = None
                self._refresh_motion_outputs_for_current_timestep()
            current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        finally:
            object.__setattr__(self.config.task, "use_root_reference_at_clip_start", old_use_root_reference)
            object.__setattr__(self.config.task, "prefer_sim_ref_from_sim_state", old_prefer_sim_ref)
            self.motion_timestep = old_motion_timestep
            self.motion_command_t = old_motion_command_t
            self.ref_quat_xyzw_t = old_ref_quat_xyzw_t
            self.ref_pos_xyz_t = old_ref_pos_xyz_t
            self._last_motion_output_timestep = old_last_motion_output_timestep

        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        group_outputs: dict[str, np.ndarray] = {}
        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []
            for term in self.obs_terms_sorted[group]:
                obs = np.asarray(term_dict[term], dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                history = [np.zeros_like(obs) for _ in range(max(history_len - 1, 0))] + [obs]
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))
            group_outputs[group] = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )
        return self._assemble_actor_obs(group_outputs)

    def _log_first_policy_step_debug(
        self,
        robot_state_data: np.ndarray,
        obs: dict[str, np.ndarray],
        policy_action: np.ndarray,
    ) -> None:
        if self._logged_first_policy_step_debug:
            return

        actor_obs = obs.get("actor_obs")
        input_terms = self.get_current_obs_buffer_dict(robot_state_data)
        q_target = (
            np.asarray(self.default_dof_angles, dtype=np.float32).reshape(1, -1)
            + np.asarray(policy_action, dtype=np.float32) * np.asarray(self.policy_action_scales, dtype=np.float32)
        )
        sim_root_state = self._get_sim_root_state()
        sim_ref_state = self._get_sim_ref_state()
        sim_object_state = self._get_sim_actor_state(self.config.task.sim_object_name)

        self.logger.info(
            "First active policy step debug: timestep={} actor_obs_dim={} dof_pos[:6]={} motion_q[:6]={} "
            "base_ang_vel={}",
            int(self.motion_timestep),
            0 if actor_obs is None else int(actor_obs.shape[1]),
            self._preview_array(input_terms.get("dof_pos"), count=6),
            self._preview_array(self.motion_command_t[:, : self.num_dofs] if self.motion_command_t is not None else None),
            self._preview_array(input_terms.get("base_ang_vel"), count=3),
        )
        self.logger.info(
            "First active policy step debug: motion_ref_ori_b[:6]={} actions_in[:6]={} "
            "obj_target_pose_size_b[:12]={} obj_pos_b[:3]={} obj_ori_b[:6]={}",
            self._preview_array(input_terms.get("motion_ref_ori_b")),
            self._preview_array(input_terms.get("actions")),
            self._preview_array(input_terms.get("obj_target_pose_size_b"), count=12),
            self._preview_array(input_terms.get("obj_pos_b"), count=3),
            self._preview_array(input_terms.get("obj_ori_b")),
        )
        self.logger.info(
            "First active policy step debug: policy_action[:6]={} scaled_action[:6]={} q_target[:6]={}",
            self._preview_array(policy_action),
            self._preview_array(np.asarray(policy_action, dtype=np.float32) * np.asarray(self.policy_action_scales, dtype=np.float32)),
            self._preview_array(q_target),
        )
        self.logger.info(
            "First active policy step debug: sim_root_pos={} sim_root_quat_xyzw={} sim_ref_pos={} sim_ref_quat_xyzw={} "
            "sim_object_pos={} sim_object_quat_xyzw={}",
            self._preview_array(sim_root_state[:, :3] if sim_root_state is not None else None, count=3),
            self._preview_array(sim_root_state[:, 3:7] if sim_root_state is not None else None, count=4),
            self._preview_array(sim_ref_state[:, :3] if sim_ref_state is not None else None, count=3),
            self._preview_array(sim_ref_state[:, 3:7] if sim_ref_state is not None else None, count=4),
            self._preview_array(sim_object_state[:, :3] if sim_object_state is not None else None, count=3),
            self._preview_array(sim_object_state[:, 3:7] if sim_object_state is not None else None, count=4),
        )

        if int(self.motion_timestep) == 0 and self._obs_input_name is not None:
            alt_cases = [
                (
                    "no_root_ref_start_t0",
                    False,
                    bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False)),
                    0,
                ),
                (
                    "no_sim_ref_override_t0",
                    bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)),
                    False,
                    0,
                ),
                ("ref_body_only_t0", False, False, 0),
                (
                    "training_like_t1",
                    bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)),
                    bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False)),
                    1,
                ),
                ("ref_body_only_t1", False, False, 1),
            ]
            for label, use_root_reference, prefer_sim_ref, motion_timestep_override in alt_cases:
                alt_obs = self._build_first_step_actor_obs_with_overrides(
                    robot_state_data,
                    use_root_reference_at_clip_start=use_root_reference,
                    prefer_sim_ref_from_sim_state=prefer_sim_ref,
                    motion_timestep_override=motion_timestep_override,
                )
                alt_input_feed = {self._obs_input_name: alt_obs}
                if self._perception_input_name:
                    perception_obs = self._get_split_perception_obs()
                    if perception_obs is not None:
                        alt_input_feed[self._perception_input_name] = perception_obs
                if self._time_step_input_name:
                    alt_input_feed[self._time_step_input_name] = np.array([[motion_timestep_override]], dtype=np.float32)
                alt_outputs = self.policy(alt_input_feed)
                alt_policy_action = alt_outputs[self._action_output_name]
                alt_scaled_action = np.asarray(alt_policy_action, dtype=np.float32) * np.asarray(
                    self.policy_action_scales, dtype=np.float32
                )
                alt_q_target = np.asarray(self.default_dof_angles, dtype=np.float32).reshape(1, -1) + alt_scaled_action
                self.logger.info(
                    "First active policy step counterfactual [{}]: policy_action[:6]={} scaled_action[:6]={} q_target[:6]={}",
                    label,
                    self._preview_array(alt_policy_action),
                    self._preview_array(alt_scaled_action),
                    self._preview_array(alt_q_target),
                )
        self._logged_first_policy_step_debug = True

    def _get_split_perception_obs(self) -> np.ndarray | None:
        if self._perception_input_name is None:
            return None
        if self._perception_obs_sub is None:
            return self._last_perception_obs

        payload = self._perception_obs_sub.get_payload()
        if payload is None:
            return self._last_perception_obs

        raw_obs = payload.get("perception_obs")
        if raw_obs is None:
            return self._last_perception_obs

        perception_obs = np.asarray(raw_obs, dtype=np.float32).reshape(1, -1)
        expected_dim = self._perception_input_dim
        if expected_dim is not None and perception_obs.shape[1] != expected_dim:
            raise ValueError(
                f"Split perception_obs dim mismatch: expected {expected_dim}, got {perception_obs.shape[1]}"
            )

        self._last_perception_obs = perception_obs
        self._logged_waiting_for_perception_obs = False
        return perception_obs

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
        self._reset_observation_history_state()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._logged_first_policy_step_debug = False
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_root_reference_state_on_next_motion_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self._reset_observation_history_state()
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._logged_first_policy_step_debug = False
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_root_reference_state_on_next_motion_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None

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
        idx = int(self.motion_timestep)
        if idx < 0:
            return 0
        return min(idx, self._motion_data.frame_count - 1)

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

    def _get_base_lin_vel_obs(self, robot_state_data: np.ndarray) -> np.ndarray:
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            return quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 7:10]).astype(np.float32, copy=False)
        return robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3].astype(np.float32, copy=False)

    def _get_base_ang_vel_obs(self, robot_state_data: np.ndarray) -> np.ndarray:
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            return quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 10:13]).astype(np.float32, copy=False)
        return robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6].astype(np.float32, copy=False)

    def _get_motion_ref_ori_b(self, robot_state_data: np.ndarray) -> np.ndarray:
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        robot_ref_ori = self._get_observation_reference_orientation_in_world(robot_state_data)
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)
        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        return motion_ref_ori_b[..., :2].reshape(1, -1)

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

        return {
            "motion_command": self.motion_command_t,
            "motion_ref_ori_b": self._get_motion_ref_ori_b(robot_state_data),
            "base_ang_vel": self._get_base_ang_vel_obs(robot_state_data),
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
        current_obs_buffer_dict = {
            "motion_command": self.motion_command_t,
            "motion_ref_ori_b": self._get_motion_ref_ori_b(robot_state_data),
            "base_ang_vel": self._get_base_ang_vel_obs(robot_state_data),
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
        }

        robot_ref_pos_w, robot_ref_quat_wxyz = self._get_observation_reference_pose_in_world(robot_state_data)
        sim_object_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        sim_ref_state = self._get_sim_ref_state()
        use_root_reference = self._should_use_root_reference_at_clip_start()
        if (
            (not use_root_reference)
            and bool(getattr(self.config.task, "prefer_sim_ref_from_sim_state", False))
            and sim_ref_state is not None
            and sim_ref_state.shape[1] >= 7
        ):
            robot_ref_pos_w = sim_ref_state[:, :3].astype(np.float32, copy=False)
            robot_ref_quat_wxyz = xyzw_to_wxyz(sim_ref_state[:, 3:7]).astype(np.float32, copy=False)

        if sim_object_state is None:
            object_pos_b = np.zeros((1, 3), dtype=np.float32)
            object_ori_6d = np.zeros((1, 6), dtype=np.float32)
        else:
            current_object_pos_w = sim_object_state[:, :3].astype(np.float32, copy=False)
            current_object_quat_wxyz = xyzw_to_wxyz(sim_object_state[:, 3:7]).astype(np.float32, copy=False)
            object_pos_b, object_quat_b = self._pose_in_robot_ref_frame(
                robot_ref_pos_w,
                robot_ref_quat_wxyz,
                current_object_pos_w,
                current_object_quat_wxyz,
            )
            object_ori_6d = matrix_from_quat(object_quat_b)[..., :2].reshape(1, -1).astype(np.float32, copy=False)

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
            target_pos_b, target_quat_b = self._pose_in_robot_ref_frame(
                robot_ref_pos_w,
                robot_ref_quat_wxyz,
                target_pos_w.astype(np.float32, copy=False),
                target_quat_wxyz.astype(np.float32, copy=False),
            )
            target_ori_6d = matrix_from_quat(target_quat_b)[..., :2].reshape(1, -1)
            target_pose_size_b = np.concatenate([target_pos_b, target_ori_6d, target_size], axis=1).astype(
                np.float32, copy=False
            )

        current_obs_buffer_dict["obj_target_pose_size_b"] = target_pose_size_b
        current_obs_buffer_dict["obj_pos_b"] = object_pos_b.astype(np.float32, copy=False)
        current_obs_buffer_dict["obj_ori_b"] = object_ori_6d.astype(np.float32, copy=False)
        return current_obs_buffer_dict

    def _get_sparse_root_distill_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        if self._motion_data is None:
            raise ValueError("Motion data is required for sparse-root distill observations.")

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
        heading_inv = self._calc_heading_quat_inv(robot_root_quat_wxyz)
        rel_pos_b = quat_apply(heading_inv, rel_pos_w)
        target_heading = self._quat_yaw(motion_root_quat_wxyz)
        robot_heading = self._quat_yaw(robot_root_quat_wxyz)
        rel_yaw = np.array([[self._normalize_angle(target_heading - robot_heading)]], dtype=np.float32)

        return {
            "sparse_target_root_trajectory_command": np.concatenate([rel_pos_b[:, :2], rel_yaw], axis=1).astype(
                np.float32, copy=False
            ),
            "base_lin_vel": self._get_base_lin_vel_obs(robot_state_data),
            "base_ang_vel": self._get_base_ang_vel_obs(robot_state_data),
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
        }

    def _get_videomimic_obs_buffer_dict(self, robot_state_data):
        if self._motion_data is None:
            raise ValueError("Motion data is required for VideoMimic observations.")

        self._maybe_update_motion_alignment(robot_state_data)
        idx = self._get_motion_index()

        base_quat = robot_state_data[:, 3:7]
        base_ang_vel = self._get_base_ang_vel_obs(robot_state_data)
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
        self._refresh_motion_outputs_for_current_timestep()
        if self._uses_videomimic:
            return self._get_videomimic_obs_buffer_dict(robot_state_data)
        if self._uses_sparse_root_distill:
            return self._get_sparse_root_distill_obs_buffer_dict(robot_state_data)
        if self._uses_object_generalist:
            return self._get_object_generalist_obs_buffer_dict(robot_state_data)
        if self._uses_legacy_object_obs:
            return self._get_legacy_object_obs_buffer_dict(robot_state_data)

        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        current_obs_buffer_dict["motion_ref_ori_b"] = self._get_motion_ref_ori_b(robot_state_data)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = self._get_base_ang_vel_obs(robot_state_data)

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        if self._auto_start_motion_clip_pending and self.use_policy_action and not self.motion_clip_progressing:
            self._auto_start_motion_clip_pending = False
            self._handle_start_motion_clip()

        if (
            self.motion_clip_progressing
            and self.use_sim_time
            and self._last_clock_reading is not None
            and bool(getattr(self.config.task, "restart_motion_on_clock_reset", False))
        ):
            current_clock = self.clock_sub.get_clock()
            if current_clock < self._last_clock_reading:
                if bool(getattr(self.config.task, "auto_start_motion_clip", False)):
                    self.logger.warning(
                        "Clock sync returned earlier timestamp before inference; skipping actor step and re-arming auto-start stiff-hold."
                    )
                    self._rearm_auto_start_after_clock_reset()
                    return self.scaled_policy_action
                self.logger.warning("Clock sync returned earlier timestamp before inference; restarting motion clip from frame 0.")
                self._handle_start_motion_clip()

        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None
            hold_q = self.motion_command_0[:, : self.num_dofs].astype(np.float32, copy=False)
            hold_offset = hold_q - self.default_dof_angles.reshape(1, -1)
            self.last_policy_action.fill(0.0)
            self.scaled_policy_action = hold_offset.astype(np.float32, copy=True)
            return self.scaled_policy_action

        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._perception_input_name:
            perception_obs = self._get_split_perception_obs()
            if perception_obs is None:
                if not self._logged_waiting_for_perception_obs:
                    self.logger.info("Waiting for first split perception obs; holding zero action.")
                    self._logged_waiting_for_perception_obs = True
                self.last_policy_action.fill(0.0)
                self.scaled_policy_action.fill(0.0)
                return self.scaled_policy_action
            input_feed[self._perception_input_name] = perception_obs
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[self.motion_timestep]], dtype=np.float32)
        self._consume_root_reference_at_clip_start()
        outputs = self.policy(input_feed)
        policy_action = outputs[self._action_output_name]

        self._log_first_policy_step_debug(robot_state_data, obs, policy_action)

        if self._uses_motion_command and not self._should_source_motion_outputs_from_motion_data():
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
        hold_q = self._stiff_hold_q.copy()
        return {
            "q": hold_q,
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _rearm_auto_start_after_clock_reset(self) -> None:
        self._reset_observation_history_state()
        self.last_policy_action.fill(0.0)
        self._auto_start_force_motion_start_pose = True
        if self.motion_command_0 is not None:
            hold_q = self.motion_command_0[:, : self.num_dofs].astype(np.float32, copy=False)
            hold_offset = hold_q - self.default_dof_angles.reshape(1, -1)
            self.scaled_policy_action = hold_offset.astype(np.float32, copy=True)
            self.motion_command_t = self.motion_command_0.copy()
        if self.ref_quat_xyzw_0 is not None:
            self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self._auto_start_rearm_requested = True
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._last_motion_output_timestep = 0
        self._logged_first_policy_step_debug = False
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_root_reference_state_on_next_motion_start = False
        self._logged_sim_ref_from_sim_state = False
        self._suppress_root_reference_at_clip_start = False

    def _can_finish_pending_policy_start(self, robot_state_data: np.ndarray) -> bool:  # noqa: ARG002
        if self._perception_input_name is None:
            return True
        if self._get_split_perception_obs() is not None:
            return True
        if not self._logged_waiting_for_perception_obs:
            self.logger.info("Waiting for split perception obs before enabling policy actions.")
            self._logged_waiting_for_perception_obs = True
        return False

    def _after_auto_start_policy(self) -> None:
        if self._auto_start_motion_clip_pending:
            self._auto_start_motion_clip_pending = False
            self._handle_start_motion_clip()

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
        if self._training_freeze_zero_remaining_holds > 0:
            self._training_freeze_zero_remaining_holds -= 1
            if not self._logged_training_freeze_zero_alignment:
                self.logger.info(
                    "Applying training-like timestep-0 freeze: prob={:.2f}, extra_holds={}",
                    self._training_freeze_zero_prob,
                    self._training_freeze_zero_extra_holds,
                )
                self._logged_training_freeze_zero_alignment = True
            if self._training_freeze_zero_remaining_holds == 0:
                self.motion_timestep = 1
                self.motion_start_timestep = current_clock - int(round(self.timestep_interval_ms))
                self._last_clock_reading = current_clock
                self.logger.info("Released training-like timestep-0 freeze; next observation will use motion timestep 1.")
            else:
                self.motion_timestep = 0
                self.motion_start_timestep = None
                self._last_clock_reading = None
            return
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            if bool(getattr(self.config.task, "restart_motion_on_clock_reset", False)):
                if bool(getattr(self.config.task, "auto_start_motion_clip", False)):
                    self.logger.warning(
                        "Clock sync returned earlier timestamp; re-arming auto-start stiff-hold before restarting the motion clip."
                    )
                    self._rearm_auto_start_after_clock_reset()
                    return

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
        self._reset_observation_history_state()
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._suppress_root_reference_at_clip_start = False
        self._auto_start_history_snapshot = None
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is not None and robot_state_data.shape[1] >= 7 + self.num_dofs:
            self._stiff_hold_q = robot_state_data[:, 7 : 7 + self.num_dofs].astype(np.float32, copy=True)
        self.logger.info("Actions set to hold current pose")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1 if bool(getattr(self.config.task, "use_zmq_lowcmd", False)) else 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_motion_output_timestep = 0
        self._last_clock_reading = None
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_root_reference_state_on_next_motion_start = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        preserve_root_reference_state = False
        if self._preserve_obs_history_on_next_motion_start:
            self._preserve_obs_history_on_next_motion_start = False
            preserve_root_reference_state = self._preserve_root_reference_state_on_next_motion_start
        else:
            self._reset_observation_history_state()
        self._preserve_root_reference_state_on_next_motion_start = False
        self._auto_start_history_snapshot = None
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_motion_output_timestep = None
        if self.motion_command_0 is not None:
            self.motion_command_t = self.motion_command_0.copy()
        if self.ref_quat_xyzw_0 is not None:
            self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self._refresh_motion_outputs_for_current_timestep()
        self._last_clock_reading = None
        self._training_freeze_zero_remaining_holds = self._training_freeze_zero_extra_holds
        self._logged_training_freeze_zero_alignment = False
        if not preserve_root_reference_state:
            self._logged_root_reference_clip_start = False
            self._remaining_root_reference_clip_start_obs = (
                1 if bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)) else 0
            )
        self._logged_first_policy_step_debug = False
        if self._motion_alignment_enabled:
            robot_state_data = self.interface.get_low_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(self._augment_robot_state_with_sim_state(robot_state_data))
        self.logger.info(colored("Starting motion clip", "blue"))

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

    def policy_action(self):
        if (
            self.motion_clip_progressing
            and self.use_sim_time
            and self._last_clock_reading is not None
            and bool(getattr(self.config.task, "restart_motion_on_clock_reset", False))
            and bool(getattr(self.config.task, "auto_start_motion_clip", False))
        ):
            current_clock = self.clock_sub.get_clock()
            if current_clock < self._last_clock_reading:
                self.logger.warning(
                    "Clock sync returned earlier timestamp before control step; re-arming auto-start stiff-hold immediately."
                )
                self._rearm_auto_start_after_clock_reset()
                self._auto_start_rearm_requested = False
                self._maybe_auto_start_rollout()
        super().policy_action()
        if self._auto_start_rearm_requested:
            self._auto_start_rearm_requested = False
            self._maybe_auto_start_rollout()
        self._maybe_advance_auto_start_state()
