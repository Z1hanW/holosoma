import json
import sys
from multiprocessing import shared_memory
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

    def fk_and_get_ref_body_orientation_in_world(self, configuration: np.ndarray) -> np.ndarray:
        # forward kinematics
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        # get ref body pose in world
        ref_body_pose_in_world = self.robot_data.oMf[self.ref_body_frame_id]
        quaternion = pin.Quaternion(ref_body_pose_in_world.rotation)  # (4, )

        return np.expand_dims(quaternion.coeffs(), axis=0)  # xyzw, (1, 4)

    def fk_ref_body_pose_with_root_identity(self, dof_pos_in_real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root_pos = np.zeros(3, dtype=np.float64)
        root_ori_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        dof_pos_in_pinocchio = dof_pos_in_real.reshape(-1)[self.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)
        pin.framesForwardKinematics(self.robot_model, self.robot_data, configuration)

        ref_body_pose = self.robot_data.oMf[self.ref_body_frame_id]
        ref_pos = np.asarray(ref_body_pose.translation, dtype=np.float32).reshape(1, 3)
        ref_quat_xyzw = np.asarray(pin.Quaternion(ref_body_pose.rotation).coeffs(), dtype=np.float32).reshape(1, 4)
        return ref_pos, xyzw_to_wxyz(ref_quat_xyzw)

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


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        # initialize timestep
        self.motion_timestep = 0
        self.motion_clip_progressing = False
        self.motion_start_timestep = None
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_ref_pos_xyz_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None
        self.motion_ref_pos_xyz_0 = None
        self._depth_img_shm = None
        self._depth_img_array = None
        self._motion_root_pos_w = None
        self._motion_root_quat_wxyz = None
        self._motion_joint_pos = None
        self._motion_joint_vel = None
        self._motion_object_pos_w = None
        self._contact_aware_carry_window = None

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

        super().__init__(config)

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

        if sys.stdin.isatty():
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

    def _get_ref_body_orientation_in_world(self, robot_state_data):
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

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])
        self._load_sparse_root_motion_file()

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw
        input_feed = self._make_initial_input_feed()
        init_output_names = [
            name for name in ("joint_pos", "joint_vel", "ref_quat_xyzw", "ref_pos_xyz") if name in self.onnx_output_names
        ]
        outputs = self.onnx_policy_session.run(init_output_names, input_feed)
        output_map = dict(zip(init_output_names, outputs, strict=True))

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate([output_map["joint_pos"], output_map["joint_vel"]], axis=1)
        self.ref_quat_xyzw_t = output_map["ref_quat_xyzw"]
        self.motion_ref_pos_xyz_t = output_map.get("ref_pos_xyz")
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
        self.motion_ref_pos_xyz_0 = None if self.motion_ref_pos_xyz_t is None else self.motion_ref_pos_xyz_t.copy()
        if self._motion_joint_pos is not None and self._motion_joint_vel is not None:
            self.motion_command_t = np.concatenate(
                [self._motion_joint_pos[:1], self._motion_joint_vel[:1]], axis=1
            )
            self.motion_command_0 = self.motion_command_t.copy()

        def policy_act(input_feed):
            policy_output_names = [
                name
                for name in ("actions", "joint_pos", "joint_vel", "ref_quat_xyzw", "ref_pos_xyz")
                if name in self.onnx_output_names
            ]
            output = self.onnx_policy_session.run(policy_output_names, input_feed)
            output_map = dict(zip(policy_output_names, output, strict=True))
            action = output_map["actions"]
            motion_command = np.concatenate([output_map["joint_pos"], output_map["joint_vel"]], axis=1)
            ref_quat_xyzw = output_map["ref_quat_xyzw"]
            ref_pos_xyz = output_map.get("ref_pos_xyz")
            return action, motion_command, ref_quat_xyzw, ref_pos_xyz

        self.policy = policy_act

    def _make_initial_input_feed(self) -> dict[str, np.ndarray]:
        """Create zero inputs for WBT model bootstrap outputs."""
        input_feed: dict[str, np.ndarray] = {}
        for input_meta in self.onnx_policy_session.get_inputs():
            name = input_meta.name
            if name == "obs":
                input_feed[name] = self._initial_actor_obs()
            elif name == "time_step":
                input_feed[name] = np.zeros((1, 1), dtype=np.float32)
            elif name == "perception_obs":
                input_feed[name] = self._zeros_for_input(input_meta)
            else:
                input_feed[name] = self._zeros_for_input(input_meta)
        return input_feed

    def _initial_actor_obs(self) -> np.ndarray:
        if "actor_obs" in self.obs_buf_dict:
            return self.obs_buf_dict["actor_obs"].copy()

        parts = [self.obs_buf_dict[group].copy() for group in self._actor_obs_group_names()]
        if not parts:
            raise ValueError("WBT policy needs actor_obs or split actor observation groups.")
        return np.concatenate(parts, axis=1)

    def _actor_obs_group_names(self) -> list[str]:
        return [group for group in self.obs_dict if group.startswith("actor_obs") and group != "actor_obs"]

    @staticmethod
    def _zeros_for_input(input_meta) -> np.ndarray:
        shape = []
        for dim in input_meta.shape:
            shape.append(dim if isinstance(dim, int) and dim > 0 else 1)
        return np.zeros(tuple(shape), dtype=np.float32)

    def _load_sparse_root_motion_file(self) -> None:
        motion_file = str(getattr(self.config.task, "motion_file", "") or "")
        if not motion_file:
            return

        path = self._resolve_motion_file_path(motion_file)
        with np.load(path, allow_pickle=True) as data:
            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()
            joint_indexes = [joint_names.index(name) for name in self.dof_names]
            root_index = body_names.index("pelvis") if "pelvis" in body_names else 0

            self._motion_root_pos_w = np.asarray(data["body_pos_w"][:, root_index, :], dtype=np.float32)
            self._motion_root_quat_wxyz = np.asarray(data["body_quat_w"][:, root_index, :], dtype=np.float32)
            self._motion_joint_pos = np.asarray(data["joint_pos"][:, 7:], dtype=np.float32)[:, joint_indexes]
            self._motion_joint_vel = np.asarray(data["joint_vel"][:, 6:], dtype=np.float32)[:, joint_indexes]
            if "object_pos_w" in data:
                self._motion_object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32)

        logger.info("[WBT] Loaded sparse-root motion file: {} ({} frames)", path, self._motion_root_pos_w.shape[0])

    @staticmethod
    def _resolve_motion_file_path(motion_file: str) -> Path:
        path = Path(motion_file).expanduser()
        if path.is_file():
            return path

        for base in (Path.cwd(), *Path(__file__).resolve().parents):
            candidate = base / motion_file
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    def _motion_frame_index(self) -> int:
        if self._motion_root_pos_w is None:
            return 0
        return min(max(int(self.motion_timestep), 0), self._motion_root_pos_w.shape[0] - 1)

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "motion_ref_pos_xyz_0": None
                if self.motion_ref_pos_xyz_0 is None
                else self.motion_ref_pos_xyz_0.copy(),
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        self.motion_ref_pos_xyz_0 = (
            None if state["motion_ref_pos_xyz_0"] is None else state["motion_ref_pos_xyz_0"].copy()
        )
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_ref_pos_xyz_t = (
            None if self.motion_ref_pos_xyz_0 is None else self.motion_ref_pos_xyz_0.copy()
        )
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_clock_reading = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target_dof_pos = self._stiff_hold_q

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t
        sparse_root_command = self._get_sparse_target_root_trajectory_command(robot_state_data)
        current_obs_buffer_dict["sparse_target_root_trajectory_command"] = sparse_root_command
        current_obs_buffer_dict["sparse_target_root_trajectory_command_contact_aware"] = (
            self._get_sparse_target_root_trajectory_command_contact_aware(sparse_root_command)
        )

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_lin_vel
        current_obs_buffer_dict["base_lin_vel"] = robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3]

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
        current_obs_buffer_dict["cam_depth"] = self._get_depth_image_obs()

        return current_obs_buffer_dict

    def _get_sparse_target_root_trajectory_command(self, robot_state_data) -> np.ndarray:
        target_pos, target_quat_wxyz = self._get_target_root_pose()
        if target_pos is None or target_quat_wxyz is None:
            return np.zeros((1, 3), dtype=np.float32)

        robot_root_pos = robot_state_data[:, :3]
        delta_xy_world = target_pos[:, :2] - robot_root_pos[:, :2]

        robot_yaw = quat_to_rpy(robot_state_data[0, 3:7])[2]
        target_yaw = quat_to_rpy(target_quat_wxyz[0])[2]

        c = np.cos(-robot_yaw)
        s = np.sin(-robot_yaw)
        delta_x_body = c * delta_xy_world[:, 0] - s * delta_xy_world[:, 1]
        delta_y_body = s * delta_xy_world[:, 0] + c * delta_xy_world[:, 1]
        yaw_error = (target_yaw - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        return np.array([[delta_x_body[0], delta_y_body[0], yaw_error]], dtype=np.float32)

    def _get_sparse_target_root_trajectory_command_contact_aware(self, base_command: np.ndarray) -> np.ndarray:
        if self._motion_object_pos_w is None or self._motion_root_pos_w is None:
            return base_command
        carry_start, carry_end = self._get_contact_aware_carry_window()
        idx = self._motion_frame_index()
        if carry_start <= idx < carry_end:
            return base_command
        return np.zeros_like(base_command)

    def _get_contact_aware_carry_window(self) -> tuple[int, int]:
        if self._contact_aware_carry_window is not None:
            return self._contact_aware_carry_window
        if self._motion_object_pos_w is None or self._motion_root_pos_w is None:
            self._contact_aware_carry_window = (0, 0)
            return self._contact_aware_carry_window

        rel_z = self._motion_object_pos_w[:, 2] - self._motion_root_pos_w[:, 2]
        z_min = float(np.min(rel_z))
        z_range = max(float(np.max(rel_z) - z_min), 0.0)
        threshold = z_min + max(0.10, z_range * 0.35)
        lifted_mask = rel_z >= threshold

        carry_start = self._first_sustained_true_index(lifted_mask, 5)
        if carry_start is None:
            lifted_indices = np.flatnonzero(lifted_mask)
            carry_start = int(lifted_indices[0]) if lifted_indices.size else int(np.argmax(rel_z))

        carry_end = self._first_sustained_true_index_from(~lifted_mask, 5, min(int(carry_start) + 1, rel_z.shape[0]))
        if carry_end is None:
            carry_end = rel_z.shape[0]

        self._contact_aware_carry_window = (int(carry_start), int(carry_end))
        logger.info(
            "[WBT] Contact-aware sparse root active window: [{}, {})",
            self._contact_aware_carry_window[0],
            self._contact_aware_carry_window[1],
        )
        return self._contact_aware_carry_window

    @staticmethod
    def _first_sustained_true_index(mask: np.ndarray, steps: int) -> int | None:
        return WholeBodyTrackingPolicy._first_sustained_true_index_from(mask, steps, 0)

    @staticmethod
    def _first_sustained_true_index_from(mask: np.ndarray, steps: int, start_idx: int) -> int | None:
        if steps <= 1:
            indices = np.flatnonzero(mask[start_idx:])
            return int(indices[0] + start_idx) if indices.size else None

        run = 0
        run_start = start_idx
        for idx in range(start_idx, mask.shape[0]):
            if bool(mask[idx]):
                if run == 0:
                    run_start = idx
                run += 1
                if run >= steps:
                    return run_start
            else:
                run = 0
        return None

    def _get_target_root_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._motion_root_pos_w is not None and self._motion_root_quat_wxyz is not None:
            idx = self._motion_frame_index()
            return self._motion_root_pos_w[idx : idx + 1], self._motion_root_quat_wxyz[idx : idx + 1]

        if self.motion_ref_pos_xyz_t is None or self.ref_quat_xyzw_t is None or self.motion_command_t is None:
            return None, None

        ref_rel_pos, ref_rel_quat_wxyz = self.pinocchio_robot.fk_ref_body_pose_with_root_identity(
            self.motion_command_t[:, : self.num_dofs]
        )
        target_ref_pos = self.motion_ref_pos_xyz_t[:, :3].astype(np.float32, copy=False)
        target_ref_quat_wxyz = xyzw_to_wxyz(self.ref_quat_xyzw_t).astype(np.float32, copy=False)
        root_quat_wxyz = quat_mul(target_ref_quat_wxyz, quat_inverse(ref_rel_quat_wxyz)).astype(
            np.float32, copy=False
        )
        root_pos = target_ref_pos - quat_apply(root_quat_wxyz, ref_rel_pos)
        return root_pos.astype(np.float32, copy=False), root_quat_wxyz

    def _get_depth_image_obs(self) -> np.ndarray:
        if "cam_depth" not in self.obs_dims:
            return np.zeros((1, 0), dtype=np.float32)

        camera_config = self.config.camera
        if camera_config is None:
            raise ValueError("perception_obs requires a camera configuration.")

        expected_shape = (
            len(camera_config.poses),
            1,
            camera_config.props.resized_height,
            camera_config.props.resized_width,
        )

        if self._depth_img_array is None:
            try:
                self._depth_img_shm = shared_memory.SharedMemory(name="depth_img_shm")
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "perception_obs requires shared memory 'depth_img_shm'. Start the MuJoCo image server first."
                ) from exc
            self._depth_img_array = np.ndarray(expected_shape, dtype=np.float32, buffer=self._depth_img_shm.buf)
            logger.info("[WBT] Depth shared memory attached: shape={}", expected_shape)

        flattened = self._depth_img_array.copy().reshape(1, -1).astype(np.float32, copy=False)
        expected_dim = self.obs_dims["cam_depth"]
        if flattened.shape[1] != expected_dim:
            raise ValueError(
                f"Depth observation shape mismatch: got {flattened.shape[1]} values, expected {expected_dim}."
            )
        return flattened

    def prepare_obs_for_rl(self, robot_state_data):
        group_outputs = self._prepare_group_observations(robot_state_data)

        if "actor_obs" in group_outputs:
            actor_obs = group_outputs["actor_obs"].astype(np.float32, copy=False)
        else:
            parts = [group_outputs[group].astype(np.float32, copy=False) for group in self._actor_obs_group_names()]
            if not parts:
                raise KeyError("WBT policy needs actor_obs or split actor observation groups.")
            actor_obs = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

        obs = {"obs": actor_obs, "actor_obs": actor_obs}
        if "perception_obs" in group_outputs:
            obs["perception_obs"] = group_outputs["perception_obs"].astype(np.float32, copy=False)
        return obs

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the start while waiting to trigger the clip.
            self.motion_timestep = 0
            self.motion_start_timestep = None
            self._last_clock_reading = None

        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {}
        for name in self.onnx_input_names:
            if name == "time_step":
                input_feed[name] = np.array([[self.motion_timestep]], dtype=np.float32)
            elif name == "obs":
                input_feed[name] = obs["obs"]
            else:
                input_feed[name] = obs[name]
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t, self.motion_ref_pos_xyz_t = self.policy(input_feed)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
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
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

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
        self.motion_yaw_offset = 0.0

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._last_clock_reading = None
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
