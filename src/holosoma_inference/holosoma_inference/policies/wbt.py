import json
import sys
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
        self._motion_data: MotionData | None = None
        self._motion_cfg: dict | None = None
        self._motion_align_quat_wxyz: np.ndarray | None = None
        self._motion_align_pos: np.ndarray | None = None
        self._obs_input_name: str | None = None
        self._time_step_input_name: str | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._motion_output_names: set[str] = set()
        self._motion_alignment_enabled = False

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
        if self._uses_videomimic:
            self._load_motion_data_from_metadata(metadata, model_path)
        self._maybe_enable_motion_future_target_poses(metadata, model_path)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

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

        motion_file = motion_cfg.get("motion_file")
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

        robot_dof_names = metadata.get("dof_names") or list(self.config.robot.dof_names)
        self._motion_data = MotionData(Path(motion_path), list(robot_dof_names), ref_name)
        self._motion_cfg = motion_cfg
        self._motion_alignment_enabled = bool(motion_cfg.get("align_motion_to_init_yaw", False))

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
                "action_output_name": self._action_output_name,
                "onnx_output_fetch": list(self._onnx_output_fetch),
                "motion_output_names": set(self._motion_output_names),
                "onnx_metadata": self._onnx_metadata,
                "onnx_obs_dim": self._onnx_obs_dim,
                "motion_data": self._motion_data,
                "motion_cfg": self._motion_cfg,
                "motion_alignment_enabled": self._motion_alignment_enabled,
                "motion_future_target_pose_provider": self._motion_future_target_pose_provider,
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
        self._motion_data = state.get("motion_data")
        self._motion_cfg = state.get("motion_cfg")
        self._motion_alignment_enabled = bool(state.get("motion_alignment_enabled", False))
        self._motion_future_target_pose_provider = state.get("motion_future_target_pose_provider")
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

    def get_current_obs_buffer_dict(self, robot_state_data):
        if self._uses_videomimic:
            current_obs_buffer_dict = self._get_videomimic_obs_buffer_dict(robot_state_data)
            if self._motion_future_target_pose_provider is not None:
                current_obs_buffer_dict["motion_future_target_poses"] = (
                    self._motion_future_target_pose_provider.get_future_target_poses(self.motion_timestep)
                )
            return current_obs_buffer_dict

        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
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

        if self._motion_future_target_pose_provider is not None:
            current_obs_buffer_dict["motion_future_target_poses"] = (
                self._motion_future_target_pose_provider.get_future_target_poses(self.motion_timestep)
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
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[self.motion_timestep]], dtype=np.float32)
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
