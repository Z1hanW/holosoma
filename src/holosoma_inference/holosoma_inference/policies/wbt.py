import hashlib
import io
import json
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pinocchio as pin
from defusedxml import ElementTree
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.button_window_contract import (
    KINEMATIC_LIFT_CONSECUTIVE_STEPS,
    KINEMATIC_LIFT_HEIGHT_THRESHOLD,
    KINEMATIC_LIFT_RATIO_THRESHOLD,
    embedded_button_window_contract_from_metadata,
    kinematic_lift_window_from_rel_z_np,
    validated_contact_aware_button_window_mode,
)
from holosoma_inference.utils.contact_sidecar_contract import (
    embedded_contact_sidecar_contract_from_metadata,
    policy_requires_contact_window,
)
from holosoma_inference.utils.embedded_motion_timeline import (
    embedded_motion_timeline_contract_from_metadata,
    embedded_motion_tensors_sha256,
    read_stable_regular_file_bytes,
)
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
from holosoma_inference.utils.policy_contract import (
    actor_perception_input_name_from_metadata,
    effective_motion_transition_settings_from_metadata,
    perception_observation_contract_sha256_from_metadata,
    validate_onnx_policy_contract,
)
from holosoma_inference.utils.sim_control import ManualRootCommandSub
from holosoma_inference.utils.sim_state import SimStateSub


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


_ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS_ENV = (
    "HOLOSOMA_ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS"
)
_PRECOMPUTED_ROOT_COMMAND_KEY = "policy_command_xy_yaw"
_PRECOMPUTED_ROOT_COMMAND_PHASE_KEY = "policy_command_phase"


def _normalized_sparse_root_command_mode(motion_config: Mapping[str, object]) -> str:
    raw_mode = motion_config.get(
        "contact_aware_sparse_root_command_mode",
        "tracking_error",
    )
    if not isinstance(raw_mode, str):
        raise ValueError(
            "motion_config.contact_aware_sparse_root_command_mode must be a string, "
            f"got {raw_mode!r}."
        )
    mode = raw_mode.strip().lower().replace("-", "_")
    if mode in {"tracking", "default", "robot_tracking_error"}:
        return "tracking_error"
    return mode


def _validated_zero_root_command_when_drop_active(
    motion_config: Mapping[str, object],
) -> bool:
    value = motion_config.get("zero_root_command_when_drop_active", False)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "motion_config.zero_root_command_when_drop_active must be a boolean, "
            f"got {value!r}."
        )
    return bool(value)


def _pickup_step_and_threshold_from_rel_z_np(
    rel_z: np.ndarray,
) -> tuple[int, np.float32]:
    """Mirror MotionCommand's float32 clip pickup detector exactly."""

    values = np.asarray(rel_z, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Precomputed command pickup detection requires a non-empty rank-1 rel-z trace."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Precomputed command pickup rel-z trace contains non-finite values.")
    z_min = np.min(values).astype(np.float32)
    z_range = np.maximum(
        np.max(values).astype(np.float32) - z_min,
        np.float32(0.0),
    ).astype(np.float32)
    threshold = (
        z_min
        + np.maximum(
            np.float32(KINEMATIC_LIFT_HEIGHT_THRESHOLD),
            z_range * np.float32(KINEMATIC_LIFT_RATIO_THRESHOLD),
        ).astype(np.float32)
    ).astype(np.float32)
    lifted = values >= threshold
    run_length = 0
    pickup_step: int | None = None
    for idx, flag in enumerate(lifted.tolist()):
        run_length = run_length + 1 if flag else 0
        if run_length >= KINEMATIC_LIFT_CONSECUTIVE_STEPS:
            pickup_step = idx - KINEMATIC_LIFT_CONSECUTIVE_STEPS + 1
            break
    if pickup_step is None:
        lifted_indices = np.flatnonzero(lifted)
        pickup_step = (
            int(lifted_indices[0])
            if lifted_indices.size
            else int(np.argmax(values))
        )
    return pickup_step, threshold


def _validated_runtime_motion_transition_settings(
    metadata: Mapping[str, object],
    *,
    apply_training_motion_transitions: bool,
) -> dict[str, object]:
    """Validate the artifact timeline and reject a silently raw WBT rollout."""

    if type(apply_training_motion_transitions) is not bool:
        raise ValueError("task.apply_training_motion_transitions must be boolean.")
    settings = effective_motion_transition_settings_from_metadata(metadata)
    has_applied_transition = any(
        bool(settings[phase_name]["applied"])
        for phase_name in ("prepend", "append")
    )
    if has_applied_transition and not apply_training_motion_transitions:
        message = (
            "This WBT artifact was trained with an authenticated effective motion transition, but "
            "task.apply_training_motion_transitions=False would execute a non-equivalent raw timeline."
        )
        if not _truthy_env(_ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS_ENV):
            raise RuntimeError(
                message
                + f" Set {_ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS_ENV}=1 only for an "
                "explicitly non-scientific diagnostic rollout."
            )
        logger.warning(
            "{} {}=1 explicitly permits this non-equivalent diagnostic rollout.",
            message,
            _ALLOW_UNAPPLIED_TRAINING_MOTION_TRANSITIONS_ENV,
        )
    return settings


def _map_source_window_to_materialized_timeline(
    window: tuple[int, int],
    *,
    source_semantics: str,
    prepend_steps: int,
) -> tuple[int, int]:
    """Map a training motion-time window onto a materialized runtime prepend.

    Global training holds ``time_steps == 0`` throughout the runtime prepend.
    Therefore a source window beginning at zero includes the whole prefix;
    positive starts are shifted after it.
    """

    start, end = (int(window[0]), int(window[1]))
    prepend_steps = int(prepend_steps)
    if source_semantics != "global_multi_clip_runtime" or prepend_steps <= 0:
        return start, end
    return (0 if start == 0 else prepend_steps + start, prepend_steps + end)


def _infer_contact_export_clip_id(directory_name: str) -> str:
    """Strip an exporter ordering prefix without corrupting normal clip IDs."""

    normalized = str(directory_name).strip()
    prefix, separator, suffix = normalized.partition("_")
    if separator and prefix.isdecimal() and suffix.strip():
        return suffix.strip()
    return normalized


def _resolve_contact_export_clip_id(directory_name: str, active_clip_ids: set[str]) -> str:
    normalized = str(directory_name).strip()
    if normalized in active_clip_ids:
        return normalized
    return _infer_contact_export_clip_id(normalized)


_CONTACT_WINDOW_OBSERVATION_TERMS = frozenset(
    {
        "sparse_target_root_trajectory_command_contact_aware",
        "drop_button",
        "pickup_button",
    }
)

_MAX_CONTACT_AWARE_SMOOTHING_STEPS = 4096


def _validated_contact_aware_carry_window_config(
    motion_cfg: Mapping[str, object],
) -> tuple[str, float, int]:
    """Validate the serialized training values again at their point of use."""

    mode_raw = motion_cfg.get("contact_aware_carry_window_mode", "rel_z")
    if not isinstance(mode_raw, str) or mode_raw not in {"rel_z", "peak_height"}:
        raise ValueError(
            "motion_config.contact_aware_carry_window_mode must be exactly "
            f"'rel_z' or 'peak_height', got {mode_raw!r}."
        )

    alpha_raw = motion_cfg.get("contact_aware_peak_height_alpha", 0.91)
    if (
        isinstance(alpha_raw, (bool, np.bool_))
        or not isinstance(alpha_raw, Real)
        or not math.isfinite(float(alpha_raw))
        or not 0.0 <= float(alpha_raw) <= 1.0
    ):
        raise ValueError(
            "motion_config.contact_aware_peak_height_alpha must be a finite real number "
            f"in [0, 1], got {alpha_raw!r}."
        )

    smoothing_raw = motion_cfg.get("contact_aware_peak_height_smoothing_steps", 5)
    if (
        isinstance(smoothing_raw, (bool, np.bool_))
        or not isinstance(smoothing_raw, Integral)
        or not 1 <= int(smoothing_raw) <= _MAX_CONTACT_AWARE_SMOOTHING_STEPS
    ):
        raise ValueError(
            "motion_config.contact_aware_peak_height_smoothing_steps must be an integer in "
            f"[1, {_MAX_CONTACT_AWARE_SMOOTHING_STEPS}], got {smoothing_raw!r}."
        )

    return mode_raw, float(alpha_raw), int(smoothing_raw)


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
    )
    _OBJECT_SCALE_KEYS = ("object_scale", "box_scale")

    def __init__(
        self,
        motion_path: Path,
        robot_dof_names: list[str],
        body_name_ref: str,
        *,
        motion_payload: bytes | None = None,
        expected_source_sha256: str | None = None,
    ):
        if motion_path.suffix.lower() != ".npz":
            raise ValueError(f"Only .npz motion files are supported in inference: {motion_path}")
        if motion_payload is None:
            motion_payload = read_stable_regular_file_bytes(
                motion_path,
                label="Inference motion source",
            )
        elif not isinstance(motion_payload, bytes) or not motion_payload:
            raise ValueError("motion_payload must be non-empty immutable bytes when provided.")
        source_sha256 = hashlib.sha256(motion_payload).hexdigest()
        if expected_source_sha256 is not None:
            if (
                not isinstance(expected_source_sha256, str)
                or len(expected_source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_source_sha256)
            ):
                raise ValueError("expected_source_sha256 must be a lowercase SHA-256 digest.")
            if source_sha256 != expected_source_sha256:
                raise ValueError(
                    "External motion source SHA-256 does not match patched ONNX provenance: "
                    f"expected={expected_source_sha256}, actual={source_sha256}, path={motion_path}."
                )

        try:
            archive = np.load(io.BytesIO(motion_payload), allow_pickle=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Motion file {motion_path} must be a non-pickled NumPy NPZ archive."
            ) from exc
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError(f"Motion file {motion_path} must be a NumPy NPZ archive.")
        with archive as data:
            required_keys = {
                "body_names",
                "joint_names",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            }
            missing_keys = sorted(required_keys.difference(data.files))
            if missing_keys:
                raise ValueError(f"Motion file {motion_path} is missing required fields: {missing_keys}.")
            if "fps" not in data:
                raise ValueError(f"Motion file {motion_path} is missing required scalar fps metadata.")
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size != 1:
                raise ValueError(
                    f"Motion file {motion_path} fps metadata must contain exactly one value, "
                    f"got shape {np.asarray(data['fps']).shape}."
                )
            if fps_values.dtype.kind not in {"i", "u", "f"}:
                raise ValueError(
                    f"Motion file {motion_path} fps metadata must be a real numeric scalar, "
                    f"got dtype {fps_values.dtype}."
                )
            try:
                fps = float(fps_values[0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Motion file {motion_path} fps metadata must be numeric.") from exc
            if not math.isfinite(fps) or fps <= 0.0:
                raise ValueError(
                    f"Motion file {motion_path} fps metadata must be finite and positive, got {fps!r}."
                )

            try:
                body_names_raw = data["body_names"]
                joint_names_raw = data["joint_names"]
            except ValueError as exc:
                raise ValueError(
                    f"Motion file {motion_path} contains pickled/object name arrays; "
                    "use fixed-width Unicode or bytes arrays."
                ) from exc
            body_names = self._decode_names(
                body_names_raw,
                field="body_names",
                source=motion_path,
            )
            joint_names = self._decode_names(
                joint_names_raw,
                field="joint_names",
                source=motion_path,
            )

            joint_pos = self._load_float_array(data, "joint_pos", source=motion_path)
            if joint_pos.ndim != 2:
                raise ValueError(f"Unexpected joint_pos shape {joint_pos.shape} in {motion_path}; expected rank 2.")
            if joint_pos.shape[1] == len(joint_names) + 7:
                joint_pos = joint_pos[:, 7:]
            elif joint_pos.shape[1] != len(joint_names):
                raise ValueError(
                    f"Unexpected joint_pos shape {joint_pos.shape} for {motion_path}; "
                    f"expected {len(joint_names)} or {len(joint_names) + 7} columns."
                )

            joint_vel = self._load_float_array(data, "joint_vel", source=motion_path)
            if joint_vel.ndim != 2:
                raise ValueError(f"Unexpected joint_vel shape {joint_vel.shape} in {motion_path}; expected rank 2.")
            if joint_vel.shape[1] == len(joint_names) + 6:
                joint_vel = joint_vel[:, 6:]
            elif joint_vel.shape[1] != len(joint_names):
                raise ValueError(
                    f"Unexpected joint_vel shape {joint_vel.shape} for {motion_path}; "
                    f"expected {len(joint_names)} or {len(joint_names) + 6} columns."
                )

            body_pos_w = self._load_float_array(data, "body_pos_w", source=motion_path)
            body_quat_w = self._load_float_array(data, "body_quat_w", source=motion_path)
            object_pos_w = (
                self._load_float_array(data, "object_pos_w", source=motion_path)
                if "object_pos_w" in data
                else None
            )
            object_quat_w = (
                self._load_float_array(data, "object_quat_w", source=motion_path)
                if "object_quat_w" in data
                else None
            )
            object_size = (
                self._extract_object_size_np(data, joint_pos.shape[0], source=str(motion_path))
                if object_pos_w is not None
                else np.ones((joint_pos.shape[0], 3), dtype=np.float32)
            )
            precomputed_root_command = self._extract_precomputed_root_command_np(
                data,
                int(joint_pos.shape[0]),
                source=motion_path,
            )

        frame_count = int(joint_pos.shape[0])
        if frame_count <= 0:
            raise ValueError(f"Motion file {motion_path} must contain at least one frame.")
        if joint_vel.shape[0] != frame_count:
            raise ValueError(
                f"Motion frame-count mismatch in {motion_path}: joint_pos has {frame_count} frames but "
                f"joint_vel has {joint_vel.shape[0]}."
            )
        expected_body_pos_shape = (frame_count, len(body_names), 3)
        if body_pos_w.shape != expected_body_pos_shape:
            raise ValueError(
                f"Unexpected body_pos_w shape {body_pos_w.shape} in {motion_path}; "
                f"expected {expected_body_pos_shape}."
            )
        expected_body_quat_shape = (frame_count, len(body_names), 4)
        if body_quat_w.shape != expected_body_quat_shape:
            raise ValueError(
                f"Unexpected body_quat_w shape {body_quat_w.shape} in {motion_path}; "
                f"expected {expected_body_quat_shape}."
            )
        if (object_pos_w is None) != (object_quat_w is None):
            raise ValueError(
                f"Motion file {motion_path} must provide object_pos_w and object_quat_w together."
            )
        if object_pos_w is not None:
            if object_pos_w.shape != (frame_count, 3):
                raise ValueError(
                    f"Unexpected object_pos_w shape {object_pos_w.shape} in {motion_path}; "
                    f"expected {(frame_count, 3)}."
                )
            if object_quat_w.shape != (frame_count, 4):
                raise ValueError(
                    f"Unexpected object_quat_w shape {object_quat_w.shape} in {motion_path}; "
                    f"expected {(frame_count, 4)}."
                )
        arrays_to_validate = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "object_size": object_size,
        }
        if object_pos_w is not None:
            arrays_to_validate["object_pos_w"] = object_pos_w
            arrays_to_validate["object_quat_w"] = object_quat_w
        for field, values in arrays_to_validate.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Motion field {field} in {motion_path} contains non-finite values.")
        if np.any(object_size <= 0.0):
            raise ValueError(f"Motion object_size in {motion_path} must contain strictly positive extents.")
        for field, quaternions in (
            ("body_quat_w", body_quat_w),
            ("object_quat_w", object_quat_w),
        ):
            if quaternions is None:
                continue
            norms = np.linalg.norm(quaternions, axis=-1)
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-3):
                raise ValueError(
                    f"Motion field {field} in {motion_path} must contain unit WXYZ quaternions."
                )

        if len(set(robot_dof_names)) != len(robot_dof_names):
            raise ValueError("Runtime robot DOF names must be unique when loading motion data.")
        missing_dofs = [name for name in robot_dof_names if name not in joint_names]
        if missing_dofs:
            raise ValueError(f"Motion file {motion_path} is missing runtime robot DOFs: {missing_dofs}.")
        if body_name_ref not in body_names:
            raise ValueError(f"Reference body {body_name_ref!r} is absent from motion file {motion_path}.")

        joint_indices = get_index_of_a_in_b(robot_dof_names, joint_names)
        self.motion_path = motion_path
        self.source_sha256 = source_sha256
        self.source_size = len(motion_payload)
        self.fps = fps
        self.body_names = tuple(body_names)
        self.joint_pos = joint_pos[:, joint_indices]
        self.joint_vel = joint_vel[:, joint_indices]
        self.source_frame_count = frame_count
        self.frame_count = frame_count
        self.object_size = object_size

        self.ref_body_index = body_names.index(body_name_ref)
        self.ref_pos_w = body_pos_w[:, self.ref_body_index, :]
        self.ref_quat_w = body_quat_w[:, self.ref_body_index, :]
        self.root_body_index = self._resolve_root_body_index(body_names)
        self.root_quat_w = body_quat_w[:, self.root_body_index, :]
        self.root_pos_w = body_pos_w[:, self.root_body_index, :]
        self.has_object = object_pos_w is not None and object_quat_w is not None
        self.object_pos_w = object_pos_w
        self.object_quat_w = object_quat_w
        self.precomputed_root_command = (
            None if precomputed_root_command is None else precomputed_root_command[0]
        )
        self.precomputed_root_command_phase = (
            None if precomputed_root_command is None else precomputed_root_command[1]
        )
        self.has_precomputed_root_command = precomputed_root_command is not None

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
                raw = np.asarray(data[key])
                if raw.dtype.kind != "f":
                    raise ValueError(
                        f"Motion field {key} in {source} must use a real floating dtype, got {raw.dtype}."
                    )
                return cls._normalize_object_size_array(raw, length, source=f"{source}:{key}")
        scale_keys = [key for key in cls._OBJECT_SCALE_KEYS if key in data]
        if scale_keys:
            raise ValueError(
                f"Motion file {source} provides mesh scale field(s) {scale_keys} but no physical "
                "object_size/box_size extents; scale and size are not interchangeable."
            )
        return np.ones((length, 3), dtype=np.float32)

    @staticmethod
    def _extract_precomputed_root_command_np(
        data: object,
        length: int,
        *,
        source: Path,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        command_present = _PRECOMPUTED_ROOT_COMMAND_KEY in data
        phase_present = _PRECOMPUTED_ROOT_COMMAND_PHASE_KEY in data
        if command_present != phase_present:
            raise ValueError(
                f"Motion file {source} must contain both {_PRECOMPUTED_ROOT_COMMAND_KEY!r} "
                f"and {_PRECOMPUTED_ROOT_COMMAND_PHASE_KEY!r}."
            )
        if not command_present:
            return None

        command = np.asarray(data[_PRECOMPUTED_ROOT_COMMAND_KEY])
        phase = np.asarray(data[_PRECOMPUTED_ROOT_COMMAND_PHASE_KEY])
        if command.dtype.kind != "f" or command.shape != (length, 3):
            raise ValueError(
                f"Motion field {_PRECOMPUTED_ROOT_COMMAND_KEY} in {source} must be a floating "
                f"array with shape ({length}, 3), got {command.dtype} {command.shape}."
            )
        if phase.dtype.kind not in "iu" or phase.shape != (length,):
            raise ValueError(
                f"Motion field {_PRECOMPUTED_ROOT_COMMAND_PHASE_KEY} in {source} must be an integer "
                f"array with shape ({length},), got {phase.dtype} {phase.shape}."
            )
        if not np.all(np.isfinite(command)):
            raise ValueError(f"Precomputed root command in {source} contains non-finite values.")

        phase_i64 = phase.astype(np.int64, copy=False)
        valid_phase = np.isin(phase_i64, (0, 1, 2))
        if not np.all(valid_phase):
            raise ValueError(
                f"Precomputed root command phase in {source} contains invalid values: "
                f"{np.unique(phase_i64[~valid_phase]).tolist()}"
            )
        zero_phase = phase_i64 == 0
        forward_phase = phase_i64 == 1
        yaw_phase = phase_i64 == 2
        if np.any(command[:, 1] != 0.0):
            raise ValueError(f"Precomputed turn-then-forward command in {source} must keep dy exactly zero.")
        if np.any((command[:, 0] != 0.0) & (command[:, 2] != 0.0)):
            raise ValueError(f"Precomputed turn-then-forward command in {source} couples dx and dyaw.")
        if np.any(command[zero_phase] != 0.0):
            raise ValueError(f"Zero-phase precomputed command rows in {source} must be zero.")
        if np.any(command[forward_phase, 0] <= 0.0) or np.any(command[forward_phase, 2] != 0.0):
            raise ValueError(f"Forward-phase precomputed command rows in {source} are inconsistent.")
        if np.any(command[yaw_phase, 0] != 0.0) or np.any(command[yaw_phase, 2] == 0.0):
            raise ValueError(f"Yaw-phase precomputed command rows in {source} are inconsistent.")
        if np.any(command[:, 0] < 0.0) or np.any(command[:, 0] > 10.0):
            raise ValueError(f"Precomputed forward command in {source} must lie in [0, 10] metres.")
        if np.any(np.abs(command[:, 2]) > math.pi):
            raise ValueError(f"Precomputed yaw command in {source} must lie in [-pi, pi].")
        return (
            command.astype(np.float32, copy=False),
            phase_i64.astype(np.uint8, copy=False),
        )

    @staticmethod
    def _load_float_array(data, field: str, *, source: Path) -> np.ndarray:
        raw = np.asarray(data[field])
        if raw.dtype.kind != "f":
            raise ValueError(
                f"Motion field {field} in {source} must use a real floating dtype, got {raw.dtype}."
            )
        return raw.astype(np.float32, copy=False)

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
    def _decode_names(arr: np.ndarray, *, field: str, source: Path) -> list[str]:
        arr = np.asarray(arr)
        if arr.dtype.kind not in {"U", "S"}:
            raise ValueError(
                f"Motion field {field} in {source} must use a Unicode/bytes string dtype, got {arr.dtype}."
            )
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"Motion field {field} in {source} must be a non-empty rank-1 name array.")
        names = arr.tolist()
        decoded: list[str] = []
        for name in names:
            if isinstance(name, bytes):
                try:
                    decoded_name = name.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Motion field {field} in {source} contains a non-UTF-8 name."
                    ) from exc
            else:
                decoded_name = str(name)
            if not decoded_name or decoded_name != decoded_name.strip() or "\x00" in decoded_name:
                raise ValueError(
                    f"Motion field {field} in {source} contains an empty, padded, or NUL name."
                )
            decoded.append(decoded_name)
        if len(set(decoded)) != len(decoded):
            raise ValueError(f"Motion field {field} in {source} contains duplicate names.")
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


_CONTACT_INTERVAL_PRIMARY_REGION_GROUPS = (
    ("left_wrist", "right_wrist"),
    (
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
        "torso",
    ),
)
_CONTACT_STAGE_RELEASE_LEAD_STEPS = 30
_CONTACT_INTERVAL_FALLBACK_FILES = {
    "left_wrist": "left_wrist_contact_interval_steps.npy",
    "right_wrist": "right_wrist_contact_interval_steps.npy",
    "left_elbow": "left_elbow_contact_interval_steps.npy",
    "right_elbow": "right_elbow_contact_interval_steps.npy",
    "left_wrist_roll": "left_wrist_roll_contact_interval_steps.npy",
    "right_wrist_roll": "right_wrist_roll_contact_interval_steps.npy",
    "left_wrist_pitch": "left_wrist_pitch_contact_interval_steps.npy",
    "right_wrist_pitch": "right_wrist_pitch_contact_interval_steps.npy",
    "torso": "torso_contact_interval_steps.npy",
}
_CONTACT_INTERVAL_REGION_ALIASES = {
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
}


def _normalize_contact_interval(raw_interval) -> tuple[int, int] | None:
    if isinstance(raw_interval, (list, tuple)):
        if len(raw_interval) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in raw_interval
        ):
            return None
    try:
        values = np.asarray(raw_interval).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size != 2 or values.dtype.kind not in {"i", "u"}:
        return None
    start, end = int(values[0]), int(values[1])
    if start < 0 or end <= start:
        return None
    return start, end


def _select_primary_contact_interval(intervals_by_region: dict[str, object]) -> tuple[int, int] | None:
    """Mirror the training-side union of all recognized carry regions."""

    normalized: dict[str, tuple[int, int]] = {}
    for raw_name, raw_interval in intervals_by_region.items():
        name = _CONTACT_INTERVAL_REGION_ALIASES.get(str(raw_name).strip(), str(raw_name).strip())
        interval = _normalize_contact_interval(raw_interval)
        if name and interval is not None:
            normalized[name] = interval

    carry_intervals = [
        normalized[name]
        for region_group in _CONTACT_INTERVAL_PRIMARY_REGION_GROUPS
        for name in region_group
        if name in normalized
    ]
    if carry_intervals:
        return (
            min(interval[0] for interval in carry_intervals),
            max(interval[1] for interval in carry_intervals),
        )
    if normalized:
        return (
            min(interval[0] for interval in normalized.values()),
            max(interval[1] for interval in normalized.values()),
        )
    return None


def _convert_contact_interval_timebase(
    interval: tuple[int, int],
    *,
    metadata: Mapping[str, object] | None,
    motion_fps: float | None,
) -> tuple[int, int]:
    """Convert exported contact steps to the active motion timebase.

    This intentionally mirrors the training-side helper without importing the
    training package into the standalone inference runtime.
    """

    if not metadata:
        return int(interval[0]), int(interval[1])
    raw_source_fps = metadata.get("contact_interval_fps", metadata.get("fps"))
    if raw_source_fps is None:
        return int(interval[0]), int(interval[1])
    if (
        isinstance(raw_source_fps, (bool, np.bool_))
        or not isinstance(raw_source_fps, Real)
        or isinstance(motion_fps, (bool, np.bool_))
        or not isinstance(motion_fps, Real)
    ):
        raise ValueError(
            f"Contact interval FPS metadata must be real numeric values: source={raw_source_fps!r}, "
            f"motion={motion_fps!r}."
        )
    try:
        source_fps = float(raw_source_fps)
        target_fps = float(motion_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Contact interval FPS metadata must be numeric: source={raw_source_fps!r}, "
            f"motion={motion_fps!r}."
        ) from exc
    if (
        not math.isfinite(source_fps)
        or source_fps <= 0.0
        or not math.isfinite(target_fps)
        or target_fps <= 0.0
    ):
        raise ValueError(
            f"Contact interval FPS values must be finite and positive: source={source_fps}, "
            f"motion={target_fps}."
        )
    start_step, end_step = int(interval[0]), int(interval[1])
    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1.0e-9):
        return start_step, end_step

    scale = target_fps / source_fps
    converted_start = int(math.ceil(start_step * scale - 1.0e-9))
    converted_end = int(math.ceil(end_step * scale - 1.0e-9))
    if converted_end <= converted_start:
        raise ValueError(
            "Contact interval became empty after FPS conversion: "
            f"interval={interval}, source_fps={source_fps}, motion_fps={target_fps}, "
            f"converted={(converted_start, converted_end)}."
        )
    return converted_start, converted_end


def _load_contact_interval_from_dir(clip_dir: Path) -> tuple[int, int] | None:
    intervals_by_region: dict[str, object] = {}
    json_path = clip_dir / "contact_intervals.json"
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Ignoring invalid contact interval file '{}': {}", json_path, exc)
        else:
            if isinstance(payload, dict):
                intervals_by_region.update(payload)

    if not intervals_by_region:
        for region_name, file_name in _CONTACT_INTERVAL_FALLBACK_FILES.items():
            interval_path = clip_dir / file_name
            if not interval_path.is_file():
                continue
            try:
                intervals_by_region[region_name] = np.load(
                    interval_path,
                    allow_pickle=False,
                )
            except Exception as exc:
                logger.warning("Ignoring invalid contact interval file '{}': {}", interval_path, exc)
    return _select_primary_contact_interval(intervals_by_region)


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
    simulator = experiment_config.get("simulator")
    if simulator is None:
        return None
    if not isinstance(simulator, dict):
        raise ValueError("experiment_config.simulator must be a mapping.")
    simulator_config = simulator.get("config")
    if simulator_config is None:
        return None
    if not isinstance(simulator_config, dict):
        raise ValueError("experiment_config.simulator.config must be a mapping.")
    sim_cfg = simulator_config.get("sim")
    if sim_cfg is None:
        return None
    if not isinstance(sim_cfg, dict):
        raise ValueError("experiment_config.simulator.config.sim must be a mapping.")

    has_fps = "fps" in sim_cfg
    has_decimation = "control_decimation" in sim_cfg
    if not has_fps and not has_decimation:
        return None
    if has_fps != has_decimation:
        raise ValueError("Serialized simulator timebase must declare both fps and control_decimation.")

    raw_fps = sim_cfg["fps"]
    raw_decimation = sim_cfg["control_decimation"]
    if isinstance(raw_fps, (bool, np.bool_)) or isinstance(raw_decimation, (bool, np.bool_)):
        raise ValueError("Serialized simulator fps and control_decimation must be real numeric values.")
    fps = float(raw_fps)
    control_decimation = float(raw_decimation)
    if (
        not math.isfinite(fps)
        or not math.isfinite(control_decimation)
        or fps <= 0.0
        or control_decimation <= 0.0
    ):
        raise ValueError("Serialized simulator fps and control_decimation must be finite and positive.")
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
    if "precomputed_root_command" in motion:
        segments["precomputed_root_command"] = np.zeros(
            (alphas.size, 3),
            dtype=np.float32,
        )
        segments["precomputed_root_command_phase"] = np.zeros(
            (alphas.size,),
            dtype=np.uint8,
        )

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
        self._contact_aware_carry_window: tuple[int, int] | None = None
        self._contact_aware_contact_window: tuple[int, int] | None = None
        self._contact_aware_button_window: tuple[int, int] | None = None
        self._motion_transition_prepend_steps = 0
        self._precomputed_turn_then_forward_enabled = False
        self._runtime_pickup_latched = False
        self._runtime_pickup_consecutive_counter = 0
        self._runtime_pickup_threshold_rel_z: np.float32 | None = None
        self._runtime_reference_pickup_step: int | None = None
        self._runtime_pickup_last_tick: tuple[int | None, float] | None = None
        self._runtime_pickup_episode_generation: int | None = None

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
        self._manual_pickup_button_log_value: float | None = None
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
        self._remaining_root_reference_clip_start_obs = 0
        self._logged_sim_ref_from_sim_state = False
        self._auto_start_motion_clip_pending = False
        self._auto_start_motion_clip_hold_start_time: float | None = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        self._motion_end_reset_episode_generation: int | None = None
        self._disable_motion_end_sim_reset = (
            _truthy_env("HOLOSOMA_DISABLE_AUTO_RESET")
            or _truthy_env("HOLOSOMA_DISABLE_MOTION_END_RESET")
            or _truthy_env("HOLOSOMA_DISABLE_CLIP_END_RESET")
        )
        self._training_freeze_zero_prob = 0.0
        self._training_freeze_zero_extra_holds = 0
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self._logged_first_policy_step_debug = False
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._suppress_root_reference_at_clip_start = False
        self._warm_autostart_obs_history = os.environ.get("HOLOSOMA_WARM_AUTOSTART_OBS_HISTORY", "1") != "0"
        self._dryrun_autostart_policy_history = os.environ.get(
            "HOLOSOMA_DRYRUN_AUTOSTART_POLICY_HISTORY", "0"
        ) != "0"
        self._autostart_policy_history_prime_steps_override = os.environ.get(
            "HOLOSOMA_AUTOSTART_POLICY_DRYRUN_STEPS", ""
        ).strip()
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
        self._uses_sparse_root_command_contact_aware = (
            "sparse_target_root_trajectory_command_contact_aware" in obs_terms
        )
        self._uses_contact_window_observation = bool(
            obs_terms.intersection(_CONTACT_WINDOW_OBSERVATION_TERMS)
        )
        self._uses_sparse_root_command = (
            "sparse_target_root_trajectory_command" in obs_terms
            or self._uses_sparse_root_command_contact_aware
        )
        self._uses_object_mocap_distill = "obj_current_pose_size_b" in obs_terms
        current_object_terms = {
            "obj_size",
            "obj_target_ori_b",
            "obj_target_pos_b",
            "obj_pos_b",
            "obj_ori_b",
        }
        legacy_object_terms = {"obj_target_pose_size_b", "obj_pos_b", "obj_ori_b"}
        velocity_object_terms = legacy_object_terms | {"obj_lin_vel_b", "obj_ang_vel_b"}
        self._uses_current_object_obs = current_object_terms.issubset(obs_terms)
        self._uses_velocity_object_obs = velocity_object_terms.issubset(obs_terms)
        self._uses_legacy_object_obs = (
            legacy_object_terms.issubset(obs_terms)
            and not {"obj_lin_vel_b", "obj_ang_vel_b"}.intersection(obs_terms)
        )
        self._uses_object_generalist = self._uses_current_object_obs or self._uses_velocity_object_obs
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
        self._embedded_motion_frame_count: int | None = None
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

        if self.config.task.use_sim_state and not callable(
            getattr(self.interface, "get_latest_sim_state_snapshot", None)
        ):
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
        if self._suppress_root_reference_at_clip_start:
            return False
        use_root = self._remaining_root_reference_clip_start_obs > 0 and int(self._get_motion_index()) == 0
        if use_root and not self._logged_root_reference_clip_start:
            logger.info("Using robot root as observation reference at clip start to match training step-0 semantics.")
            self._logged_root_reference_clip_start = True
        return use_root

    def _consume_root_reference_at_clip_start(self) -> None:
        """Consume the one observation for which training uses root as ref body."""
        if self._remaining_root_reference_clip_start_obs <= 0:
            return
        # If the clock advanced before an actor observation could be built,
        # the special step-0 state is no longer reproducible.
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

    def _get_autostart_policy_history_prime_steps(self) -> int:
        override = self._autostart_policy_history_prime_steps_override
        if override:
            try:
                return max(0, int(override))
            except ValueError:
                if not getattr(self, "_logged_invalid_autostart_policy_history_prime_steps", False):
                    self.logger.warning(
                        "Ignoring invalid HOLOSOMA_AUTOSTART_POLICY_DRYRUN_STEPS={!r}",
                        override,
                    )
                    self._logged_invalid_autostart_policy_history_prime_steps = True
                return 0

        actor_history_lengths = [
            int(history_length)
            for group_name, history_length in self.history_length_dict.items()
            if str(group_name).startswith("actor_obs")
        ]
        history_len = max(actor_history_lengths, default=1)
        return max(0, history_len - 1)

    def _prime_auto_start_policy_history(self, robot_state_data: np.ndarray) -> bool:
        """Run an explicitly requested, non-equivalent diagnostic warmup.

        Training history contains only states reached by real simulator steps
        and actions that were actually applied.  Repeated policy calls on one
        frozen state cannot reproduce that contract, so this legacy diagnostic
        is disabled by default and must never be used for scientific parity.
        """
        if not self._dryrun_autostart_policy_history or not self._warm_autostart_obs_history:
            return False
        if self._obs_input_name is None or self._action_output_name is None:
            return False

        prime_steps = self._get_autostart_policy_history_prime_steps()
        if prime_steps <= 0:
            return False

        augmented_state = self._augment_robot_state_with_sim_state(robot_state_data)
        if augmented_state is None:
            return False

        perception_input_name = getattr(
            self,
            "_perception_obs_input_name",
            getattr(self, "_perception_input_name", None),
        )
        perception_obs: np.ndarray | None = None
        if perception_input_name is not None:
            perception_dim = self._get_onnx_input_dim(perception_input_name)
            try:
                perception_obs = self._get_split_perception_obs(
                    perception_dim,
                    target_sim_time_ms=self._get_control_tick_sim_time_ms(),
                    target_episode_generation=self._get_control_tick_episode_generation(),
                )
            except RuntimeError as exc:
                if not getattr(self, "_logged_auto_start_history_prime_waiting_for_perception_obs", False):
                    self.logger.info("Skipping auto-start history priming until perception is available: {}", exc)
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

        def run_prime_policy(actor_obs: np.ndarray) -> None:
            input_feed = {self._obs_input_name: actor_obs}
            if perception_input_name is not None and perception_obs is not None:
                input_feed[perception_input_name] = perception_obs
            if self._time_step_input_name is not None:
                input_feed[self._time_step_input_name] = np.array([[0]], dtype=np.float32)

            outputs = self.policy(input_feed)
            self._update_policy_action_state(
                outputs[self._action_output_name],
                label=f"ONNX output {self._action_output_name!r} during history priming",
            )
            if self._uses_motion_command and not self._should_source_motion_outputs_from_motion_data():
                joint_pos = outputs.get("joint_pos")
                joint_vel = outputs.get("joint_vel")
                if joint_pos is not None and joint_vel is not None:
                    self.motion_command_t = np.concatenate([joint_pos, joint_vel], axis=1)
                    self.ref_quat_xyzw_t = outputs.get("ref_quat_xyzw", self.ref_quat_xyzw_t)
                    self.ref_pos_xyz_t = outputs.get("ref_pos_xyz", self.ref_pos_xyz_t)

        seed_obs = self.prepare_obs_for_rl(augmented_state)
        run_prime_policy(seed_obs["actor_obs"])

        self._consume_root_reference_at_clip_start()

        for _ in range(max(0, prime_steps - 1)):
            primed_obs = self.prepare_obs_for_rl(augmented_state)
            run_prime_policy(primed_obs["actor_obs"])

        self._preserve_obs_history_on_next_motion_start = True
        self._preserve_root_reference_state_on_next_motion_start = True
        self.logger.info(
            "Ran non-equivalent diagnostic auto-start history warmup over {} unpublished actor steps "
            "at motion timestep 0.",
            prime_steps,
        )
        return True

    @staticmethod
    def _extract_motion_config(metadata: dict) -> dict | None:
        top_level_present = "motion_config" in metadata
        top_level_cfg = metadata.get("motion_config")
        exp_cfg = metadata.get("experiment_config")
        command = exp_cfg.get("command") if isinstance(exp_cfg, dict) else None
        setup_terms = command.get("setup_terms") if isinstance(command, dict) else None
        motion_command = (
            setup_terms.get("motion_command")
            if isinstance(setup_terms, dict)
            else None
        )
        params = motion_command.get("params") if isinstance(motion_command, dict) else None
        nested_present = isinstance(params, dict) and "motion_config" in params
        nested_cfg = params.get("motion_config") if nested_present else None

        if nested_present:
            if not isinstance(nested_cfg, dict):
                raise ValueError(
                    "experiment_config motion_config must be a mapping."
                )
            if top_level_present:
                if not isinstance(top_level_cfg, dict):
                    raise ValueError(
                        "Top-level and experiment_config motion_config metadata disagree."
                    )
                canonical_top = json.dumps(
                    top_level_cfg,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                canonical_nested = json.dumps(
                    nested_cfg,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                if canonical_top != canonical_nested:
                    raise ValueError(
                        "Top-level and experiment_config motion_config metadata disagree; "
                        "the serialized experiment_config is canonical."
                    )
            return nested_cfg

        if not top_level_present:
            return None
        if not isinstance(top_level_cfg, dict):
            raise ValueError("Legacy top-level motion_config metadata must be a mapping.")
        return top_level_cfg

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
        embedded_timeline_contract = embedded_motion_timeline_contract_from_metadata(metadata)
        expected_source_sha256 = (
            embedded_timeline_contract["source_motion_sha256"]
            if embedded_timeline_contract is not None
            else None
        )
        self._motion_data = MotionData(
            motion_path,
            list(robot_dof_names),
            ref_name,
            expected_source_sha256=expected_source_sha256,
        )
        self._motion_body_names = tuple(self._motion_data.body_names)
        self._motion_transition_prepend_steps = self._maybe_apply_training_motion_transitions_to_motion_data(
            metadata, ref_name
        )
        if embedded_timeline_contract is not None:
            has_effective_transition = bool(
                int(embedded_timeline_contract["effective_prepend_steps"])
                or int(embedded_timeline_contract["effective_append_steps"])
            )
            runtime_materialization = (
                "effective_training_timeline"
                if self.config.task.apply_training_motion_transitions
                or not has_effective_transition
                else "raw_unsafe_diagnostic"
            )
            if runtime_materialization == embedded_timeline_contract["materialization"]:
                runtime_tensors_sha256, runtime_frame_count = embedded_motion_tensors_sha256(
                    {
                        "joint_pos": self._motion_data.joint_pos,
                        "joint_vel": self._motion_data.joint_vel,
                        "ref_pos_xyz": self._motion_data.ref_pos_w,
                        "ref_quat_xyzw": self._motion_data.ref_quat_w[:, [1, 2, 3, 0]],
                    }
                )
                if (
                    runtime_tensors_sha256
                    != embedded_timeline_contract["embedded_tensors_sha256"]
                    or runtime_frame_count
                    != int(embedded_timeline_contract["embedded_frame_count"])
                ):
                    raise RuntimeError(
                        "External motion materialization does not reproduce the digest-bound "
                        "timeline embedded in the ONNX artifact."
                    )
            else:
                logger.warning(
                    "External motion runtime materialization {} intentionally differs from ONNX "
                    "artifact materialization {}; this is permitted only by the explicit unsafe "
                    "diagnostic overrides already validated for this rollout.",
                    runtime_materialization,
                    embedded_timeline_contract["materialization"],
                )
        self._motion_cfg = motion_cfg or {}
        self._configure_precomputed_turn_then_forward_runtime()
        self._contact_aware_carry_window = None
        self._contact_aware_button_window = self._load_contact_aware_button_window(onnx_path)
        if validated_contact_aware_button_window_mode(self._motion_cfg) == "contact_interval":
            self._contact_aware_contact_window = self._contact_aware_button_window
        else:
            # Preserve the sidecar independently for rel-z root release
            # capping.  Kinematic button labels must never become that cap.
            self._contact_aware_contact_window = self._load_contact_interval_window(
                onnx_path
            )
        freeze_prob_raw = self._motion_cfg.get("freeze_at_timestep_zero_prob", 0.0)
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
        elif self._training_freeze_zero_prob > 0.0:
            # Deployment is deterministic; use the geometric expectation of
            # training's Bernoulli timestep-0 hold distribution.
            self._training_freeze_zero_extra_holds = int(
                min(
                    200,
                    round(
                        self._training_freeze_zero_prob
                        / max(1.0e-6, 1.0 - self._training_freeze_zero_prob)
                    ),
                )
            )
        else:
            self._training_freeze_zero_extra_holds = 0
        self._training_freeze_zero_remaining_holds = 0
        alignment_from_metadata = bool((motion_cfg or {}).get("align_motion_to_init_yaw", False))
        self._motion_alignment_enabled = bool(alignment_from_metadata or self._force_motion_alignment)
        if self._motion_alignment_enabled and not alignment_from_metadata and self._force_motion_alignment:
            logger.info("Forcing runtime motion alignment for split sim2sim inference.")

    def _reset_per_model_motion_state_for_setup(self) -> None:
        """Clear motion state that must never leak between preloaded policy slots."""

        self._motion_output_names = set(self.onnx_output_names)
        self._motion_data = None
        self._motion_cfg = None
        self._effective_motion_transition_settings = None
        self._embedded_motion_frame_count = None
        self._motion_body_names = ()
        self._motion_transition_prepend_steps = 0
        self._contact_aware_carry_window = None
        self._contact_aware_contact_window = None
        self._contact_aware_button_window = None
        self._training_freeze_zero_prob = 0.0
        self._training_freeze_zero_extra_holds = 0
        self._motion_alignment_enabled = False
        self._precomputed_turn_then_forward_enabled = False
        self._runtime_pickup_threshold_rel_z = None
        self._runtime_reference_pickup_step = None
        self._reset_runtime_pickup_latch()
        self.motion_command_0 = None
        self.motion_command_t = None
        self.ref_quat_xyzw_0 = None
        self.ref_quat_xyzw_t = None
        self.ref_pos_xyz_t = None

    def _configure_precomputed_turn_then_forward_runtime(self) -> None:
        motion_cfg = self._motion_cfg or {}
        mode = _normalized_sparse_root_command_mode(motion_cfg)
        self._precomputed_turn_then_forward_enabled = (
            mode == "precomputed_turn_then_forward"
        )
        if not self._precomputed_turn_then_forward_enabled:
            self._runtime_pickup_threshold_rel_z = None
            self._runtime_reference_pickup_step = None
            self._reset_runtime_pickup_latch()
            return

        conflicting_modes = {
            "pure_rl_policy_command_after_lift": bool(
                motion_cfg.get("pure_rl_policy_command_after_lift_enabled", False)
            ),
            "hybrid_stage2": bool(motion_cfg.get("hybrid_stage2_enabled", False)),
            "hybrid_velocity": bool(motion_cfg.get("hybrid_velocity_enabled", False)),
        }
        conflicts = [name for name, enabled in conflicting_modes.items() if enabled]
        if conflicts:
            raise ValueError(
                "precomputed_turn_then_forward is an exclusive actor-command mode; "
                f"disable {conflicts}."
            )
        motion = self._motion_data
        if motion is None or not motion.has_object:
            raise ValueError(
                "precomputed_turn_then_forward deployment requires external motion data "
                "with object_pos_w/object_quat_w."
            )
        if not motion.has_precomputed_root_command:
            raise ValueError(
                "precomputed_turn_then_forward deployment requires both "
                "policy_command_xy_yaw and policy_command_phase in the selected motion NPZ."
            )
        if motion.precomputed_root_command.shape != (motion.frame_count, 3):
            raise ValueError(
                "Materialized precomputed root command does not match the runtime motion timeline: "
                f"command={motion.precomputed_root_command.shape}, frames={motion.frame_count}."
            )
        if motion.precomputed_root_command_phase.shape != (motion.frame_count,):
            raise ValueError(
                "Materialized precomputed root command phase does not match the runtime motion timeline: "
                f"phase={motion.precomputed_root_command_phase.shape}, frames={motion.frame_count}."
            )

        settings = self._effective_motion_transition_settings or {}
        source_semantics = str(settings.get("source_semantics", "single_clip_static"))
        source_offset = (
            int(self._motion_transition_prepend_steps)
            if source_semantics == "global_multi_clip_runtime"
            else 0
        )
        source_end = (
            source_offset + int(motion.source_frame_count)
            if source_semantics == "global_multi_clip_runtime"
            else int(motion.frame_count)
        )
        if not 0 <= source_offset < source_end <= int(motion.frame_count):
            raise ValueError(
                "Authenticated transition does not leave a valid source trace for runtime pickup "
                f"detection: source=[{source_offset}, {source_end}), frames={motion.frame_count}."
            )
        rel_z = (
            motion.object_pos_w[source_offset:source_end, 2]
            - motion.root_pos_w[source_offset:source_end, 2]
        ).astype(np.float32, copy=False)
        source_pickup_step, pickup_threshold = _pickup_step_and_threshold_from_rel_z_np(
            rel_z
        )
        runtime_pickup_step = (
            0
            if source_semantics == "global_multi_clip_runtime" and source_pickup_step == 0
            else source_offset + source_pickup_step
        )
        self._runtime_pickup_threshold_rel_z = pickup_threshold
        self._runtime_reference_pickup_step = int(runtime_pickup_step)
        self._reset_runtime_pickup_latch()
        phase_counts = np.bincount(
            motion.precomputed_root_command_phase.astype(np.int64, copy=False),
            minlength=3,
        )
        logger.info(
            "Enabled deployment-equivalent precomputed turn-then-forward command: "
            "zero_frames={} forward_frames={} yaw_frames={} pickup_threshold_rel_z={:.6f} "
            "pickup_consecutive_steps={} runtime_pickup_latch=True.",
            int(phase_counts[0]),
            int(phase_counts[1]),
            int(phase_counts[2]),
            float(pickup_threshold),
            KINEMATIC_LIFT_CONSECUTIVE_STEPS,
        )

    def _has_embedded_motion_outputs(self) -> bool:
        required_motion_outputs = {"joint_pos", "joint_vel", "ref_quat_xyzw"}
        output_names = set(getattr(self, "_motion_output_names", ()))
        return required_motion_outputs.issubset(output_names)

    def _active_motion_frame_count(self) -> int | None:
        """Return the validated timeline length used by this runtime policy.

        External MotionData takes precedence because an explicitly diagnostic
        transition override may intentionally materialize a timeline different
        from the graph.  A provenance-bearing self-contained graph can use its
        authenticated embedded frame count.  Legacy combined graphs have no
        safe length provenance here and retain their historical behavior.
        """

        motion_data = getattr(self, "_motion_data", None)
        if motion_data is not None:
            frame_count = int(getattr(motion_data, "frame_count", 0) or 0)
            return frame_count if frame_count > 0 else None
        embedded_frame_count = getattr(self, "_embedded_motion_frame_count", None)
        if embedded_frame_count is None or not self._has_embedded_motion_outputs():
            return None
        frame_count = int(embedded_frame_count)
        return frame_count if frame_count > 0 else None

    def _policy_requires_external_motion_data(self) -> bool:
        return bool(self._uses_motion_command and not self._has_embedded_motion_outputs())

    def _will_apply_authenticated_motion_transition(self) -> bool:
        """Return whether this runtime will materialize an authenticated transition.

        ``apply_training_motion_transitions`` is enabled for canonical WBT
        deployment so artifacts with an applied transition reproduce their
        training timeline.  That default must not turn an explicitly inactive
        contract into an external motion-file dependency, however.
        """

        if not bool(
            getattr(self.config.task, "apply_training_motion_transitions", False)
        ):
            return False
        settings = getattr(self, "_effective_motion_transition_settings", None)
        if not isinstance(settings, Mapping):
            return False
        return any(
            isinstance(settings.get(phase_name), Mapping)
            and settings[phase_name].get("applied") is True
            for phase_name in ("prepend", "append")
        )

    def _policy_requires_motion_data_for_setup(self) -> bool:
        """Identify real external motion consumers for the active artifact."""

        return bool(
            self._uses_videomimic
            or self._uses_object_mocap_distill
            or self._uses_object_generalist
            or self._uses_legacy_object_obs
            or self._uses_sparse_root_command
            # ``drop_button``/``pickup_button`` can be selected without a
            # sparse-root term. Their authenticated sidecar window is still
            # indexed on the external motion timeline, so treating a complete
            # embedded-output graph as self-contained would silently replace
            # the training button signal with an empty/fallback window.
            or bool(getattr(self, "_uses_contact_window_observation", False))
            or self._will_apply_authenticated_motion_transition()
            or self._policy_requires_external_motion_data()
        )

    def _should_source_motion_outputs_from_motion_data(self) -> bool:
        return bool(
            self._uses_motion_command
            and self._motion_data is not None
            and (
                self._will_apply_authenticated_motion_transition()
                or not self._has_embedded_motion_outputs()
            )
        )

    def _get_motion_outputs_from_motion_data(self, motion_timestep: int) -> dict[str, np.ndarray] | None:
        if self._motion_data is None or self._motion_data.frame_count <= 0:
            return None
        idx = max(0, min(int(motion_timestep), self._motion_data.frame_count - 1))
        return {
            "joint_pos": self._motion_data.joint_pos[idx : idx + 1].astype(np.float32, copy=False),
            "joint_vel": self._motion_data.joint_vel[idx : idx + 1].astype(np.float32, copy=False),
            "ref_quat_xyzw": wxyz_to_xyzw(self._motion_data.ref_quat_w[idx : idx + 1]).astype(
                np.float32,
                copy=False,
            ),
            "ref_pos_xyz": self._motion_data.ref_pos_w[idx : idx + 1].astype(np.float32, copy=False),
        }

    def _validate_runtime_motion_timebase(self, metadata: dict) -> None:
        """Require the runtime, training config, and selected motion to advance at one shared rate."""

        if isinstance(self.rl_rate, (bool, np.bool_)):
            raise ValueError(f"Inference rl_rate must be a finite positive frequency, got {self.rl_rate!r}.")
        try:
            runtime_fps = float(self.rl_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Inference rl_rate must be a finite positive frequency, got {self.rl_rate!r}."
            ) from exc
        if not math.isfinite(runtime_fps) or runtime_fps <= 0.0:
            raise ValueError(f"Inference rl_rate must be finite and positive, got {runtime_fps!r}.")

        if self._motion_data is not None:
            motion_fps = float(self._motion_data.fps)
            if not math.isclose(motion_fps, runtime_fps, rel_tol=1.0e-6, abs_tol=1.0e-6):
                raise ValueError(
                    "Motion FPS must match the inference policy control frequency because runtime advances "
                    "one motion frame per control step: "
                    f"motion.fps={motion_fps}, rl_rate={runtime_fps}. Resample the motion or use the "
                    "training control frequency."
                )

        try:
            training_control_dt = _extract_control_dt_from_metadata(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Serialized simulator control timebase is malformed.") from exc
        if training_control_dt is not None:
            training_fps = 1.0 / float(training_control_dt)
            if not math.isclose(training_fps, runtime_fps, rel_tol=1.0e-6, abs_tol=1.0e-6):
                raise ValueError(
                    "Inference rl_rate does not match the serialized training control frequency: "
                    f"rl_rate={runtime_fps}, training_control_fps={training_fps}."
                )

    def _maybe_apply_training_motion_transitions_to_motion_data(self, metadata: dict, ref_name: str) -> int:
        apply_transitions = self.config.task.apply_training_motion_transitions
        transition_settings = _validated_runtime_motion_transition_settings(
            metadata,
            apply_training_motion_transitions=apply_transitions,
        )
        if self._motion_data is None or not apply_transitions:
            return 0
        init_state = _extract_robot_init_state_from_metadata(metadata)
        source_semantics = str(transition_settings["source_semantics"])
        prepend_contract = transition_settings["prepend"]
        append_contract = transition_settings["append"]
        needs_prepend = bool(prepend_contract["applied"])
        needs_append = bool(append_contract["applied"])
        if not needs_prepend and not needs_append:
            return 0
        if not isinstance(init_state, dict):
            raise ValueError(
                "Applied motion transitions require serialized robot init_state metadata."
            )

        motion_data = self._motion_data
        has_precomputed_root_command = bool(
            getattr(motion_data, "has_precomputed_root_command", False)
        )
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
        if has_precomputed_root_command:
            motion["precomputed_root_command"] = motion_data.precomputed_root_command.astype(
                np.float32,
                copy=True,
            )
            motion["precomputed_root_command_phase"] = (
                motion_data.precomputed_root_command_phase.astype(np.uint8, copy=True)
            )

        applied_prepend_steps = 0
        if needs_prepend:
            prepend_steps = int(prepend_contract["steps"])
            _apply_transition_segment_np(
                motion,
                start_state=_build_default_state(use_motion_end=False),
                target_state=_motion_state(0),
                num_steps=prepend_steps,
                prepend=True,
                drop_first=False,
                drop_last=True,
            )
            applied_prepend_steps = prepend_steps

        if needs_append:
            append_steps = int(append_contract["steps"])
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
        if has_precomputed_root_command:
            motion_data.precomputed_root_command = motion["precomputed_root_command"]
            motion_data.precomputed_root_command_phase = motion[
                "precomputed_root_command_phase"
            ]
        motion_data.frame_count = motion_data.joint_pos.shape[0]
        logger.info(
            "Applied authenticated training motion transitions to inference motion data for '{}': "
            "source_semantics={} prepend_steps={} append_steps={} frame_count={}",
            ref_name,
            source_semantics,
            applied_prepend_steps,
            int(append_contract["steps"]),
            motion_data.frame_count,
        )
        return applied_prepend_steps

    @classmethod
    def _resolve_contact_interval_root(cls, raw_root: str, onnx_path: Path) -> Path | None:
        candidates = [Path(raw_root).expanduser()]
        candidates.append(onnx_path.parent / raw_root)
        repo_root = cls._find_repo_root(Path(__file__).resolve())
        candidates.extend((repo_root / raw_root, repo_root / "src" / raw_root))
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        return None

    def _materialize_contact_button_window(
        self,
        *,
        interval: tuple[int, int],
        clip_metadata: Mapping[str, object],
        clip_id: str,
        provenance_source: str,
    ) -> tuple[int, int]:
        """Map one validated raw sidecar interval onto the runtime timeline."""

        cfg = self._motion_cfg or {}
        raw_start, raw_end = interval
        start, end = _convert_contact_interval_timebase(
            interval,
            metadata=clip_metadata,
            motion_fps=getattr(self._motion_data, "fps", None),
        )
        compensated_in_training = bool(
            cfg.get("contact_interval_runtime_prepend_compensation", False)
        )
        prepend_steps = int(self._motion_transition_prepend_steps)
        source_semantics = str(
            (self._effective_motion_transition_settings or {}).get(
                "source_semantics",
                "single_clip_static",
            )
        )
        if compensated_in_training and source_semantics == "global_multi_clip_runtime":
            training_window = (
                max(0, int(start) - prepend_steps),
                int(end) - prepend_steps,
            )
        else:
            training_window = (int(start), int(end))
        window = _map_source_window_to_materialized_timeline(
            training_window,
            source_semantics=source_semantics,
            prepend_steps=prepend_steps,
        )
        if compensated_in_training:
            declared_source_frame_count = getattr(
                self._motion_data,
                "source_frame_count",
                None,
            )
            if declared_source_frame_count is None:
                source_frame_count = int(getattr(self._motion_data, "frame_count", 0))
                if source_semantics == "global_multi_clip_runtime":
                    source_frame_count -= prepend_steps
            else:
                source_frame_count = int(declared_source_frame_count)
            if not (0 <= training_window[0] < training_window[1] <= source_frame_count):
                raise ValueError(
                    "Runtime-prepend-compensated training contact interval is outside the "
                    "active inference motion range after source-time conversion: "
                    f"clip={clip_id!r}, interval={training_window}, "
                    f"source_frame_count={source_frame_count}."
                )
            active_frame_count = int(getattr(self._motion_data, "frame_count", 0))
            if not (0 <= window[0] < window[1] <= active_frame_count):
                raise ValueError(
                    "Runtime-prepend-compensated training contact interval is outside the active "
                    "inference motion range: "
                    f"clip={clip_id!r}, interval={window}, frame_count={active_frame_count}."
                )
        logger.info(
            "Using {} contact sidecar for policy buttons: clip={} raw=[{}, {}) "
            "motion_timebase=[{}, {}) effective=[{}, {}) "
            "runtime_prepend_compensated_in_training={}.",
            provenance_source,
            clip_id,
            raw_start,
            raw_end,
            start,
            end,
            window[0],
            window[1],
            compensated_in_training,
        )
        return window

    def _embedded_contact_button_window(self) -> tuple[int, int] | None:
        metadata = getattr(self, "_onnx_metadata", {})
        contract = embedded_contact_sidecar_contract_from_metadata(metadata)
        if contract is None:
            return None
        if self._motion_data is None:
            raise RuntimeError(
                "Embedded contact-sidecar provenance requires the selected external motion data."
            )
        clip_id = self._motion_data.motion_path.stem
        if contract["clip_id"] != clip_id:
            raise RuntimeError(
                "Embedded contact-sidecar clip does not match the active inference motion: "
                f"contract={contract['clip_id']!r}, active={clip_id!r}."
            )
        if contract["source_motion_sha256"] != self._motion_data.source_sha256:
            raise RuntimeError(
                "Embedded contact-sidecar provenance is bound to different motion bytes."
            )
        if contract["source_motion_size"] != int(self._motion_data.source_size):
            raise RuntimeError(
                "Embedded contact-sidecar provenance is bound to a different motion byte size."
            )
        if contract["source_frame_count"] != int(
            getattr(self._motion_data, "source_frame_count", self._motion_data.frame_count)
        ):
            raise RuntimeError(
                "Embedded contact-sidecar provenance is bound to a different source frame count."
            )
        if not math.isclose(
            float(contract["motion_fps"]),
            float(self._motion_data.fps),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "Embedded contact-sidecar provenance is bound to a different motion FPS."
            )
        fps_key = contract["contact_interval_fps_key"]
        clip_metadata = (
            {}
            if fps_key is None
            else {
                fps_key: (
                    None
                    if contract["contact_interval_fps"] is None
                    else float(contract["contact_interval_fps"])
                )
            }
        )
        raw_interval = contract["selected_raw_interval"]
        return self._materialize_contact_button_window(
            interval=(int(raw_interval[0]), int(raw_interval[1])),
            clip_metadata=clip_metadata,
            clip_id=clip_id,
            provenance_source="digest-bound embedded",
        )

    def _load_contact_interval_window(self, onnx_path: Path) -> tuple[int, int] | None:
        """Load the legacy/exported sidecar window independently of button mode."""
        # Match training: contact-window banks are not configured for
        # robot-only motions, even when a generic contact-aware observation
        # preset or sampling flag is present.
        if self._motion_data is None or not self._motion_data.has_object:
            return None
        cfg = self._motion_cfg or {}
        for flag_name in (
            "use_adaptive_timesteps_sampler",
            "uniform_t1_window_sampling_enabled",
        ):
            flag_value = cfg.get(flag_name, False)
            if not isinstance(flag_value, bool):
                raise ValueError(
                    f"motion_config.{flag_name} must be boolean, got {flag_value!r}."
                )
        runtime_requires_contact_window = bool(
            bool(getattr(self, "_uses_contact_window_observation", False))
            or bool(cfg.get("use_adaptive_timesteps_sampler", False))
            or bool(cfg.get("uniform_t1_window_sampling_enabled", False))
        )
        metadata = getattr(self, "_onnx_metadata", {})
        if isinstance(metadata, Mapping) and "experiment_config" in metadata:
            metadata_requires_contact_window = policy_requires_contact_window(metadata)
            if metadata_requires_contact_window != runtime_requires_contact_window:
                raise RuntimeError(
                    "Runtime and serialized policy disagree on whether a contact sidecar is required."
                )
            requires_contact_window = metadata_requires_contact_window
        else:
            requires_contact_window = runtime_requires_contact_window
        if not requires_contact_window:
            return None
        embedded_window = self._embedded_contact_button_window()
        if embedded_window is not None:
            # The selected values and their source-file digests are already in
            # the ONNX payload.  Do not reopen a mutable full bank at runtime.
            return embedded_window
        training_provenance = (
            metadata.get("training_provenance")
            if isinstance(metadata, Mapping)
            else None
        )
        embedded_timeline = embedded_motion_timeline_contract_from_metadata(metadata)
        if (
            embedded_timeline is not None
            and isinstance(training_provenance, Mapping)
            and isinstance(training_provenance.get("contact_sidecar_manifest_sha256"), str)
        ):
            raise RuntimeError(
                "Digest-provenanced patched policy uses contact-aware observations but is missing "
                "its embedded active contact-sidecar contract; re-run patch_motion_onnx with the "
                "verified full contact and motion banks."
            )
        configured_root = str(cfg.get("adaptive_sampling_contact_interval_root") or "").strip()
        environment_root = os.environ.get("HOLOSOMA_CONTACT_INTERVAL_ROOT", "").strip()
        # Treat an empty environment variable as "no override".  Otherwise a
        # shell-exported empty value silently erases the serialized training
        # root and changes contact-aware observation semantics at inference.
        raw_root = environment_root or configured_root
        if not raw_root:
            return None
        logger.warning(
            "Using legacy external contact-sidecar compatibility path for clip {}; selected bytes "
            "are not per-clip digest-bound in this artifact and this rollout is diagnostic.",
            self._motion_data.motion_path.stem,
        )
        contact_root = self._resolve_contact_interval_root(raw_root, onnx_path)
        if contact_root is None:
            message = (
                f"Training contact interval root {raw_root!r} is unavailable; "
                "using a kinematic fallback would change the student observation contract."
            )
            if not _truthy_env("HOLOSOMA_ALLOW_CONTACT_WINDOW_FALLBACK"):
                raise FileNotFoundError(
                    message
                    + " Provide HOLOSOMA_CONTACT_INTERVAL_ROOT or explicitly set "
                    "HOLOSOMA_ALLOW_CONTACT_WINDOW_FALLBACK=1 for a non-equivalent diagnostic rollout."
                )
            logger.warning("{} Explicit fallback override is enabled.", message)
            return None

        clip_id = self._motion_data.motion_path.stem
        matching_clip_dirs: list[tuple[Path, dict[str, object]]] = []
        for clip_dir in sorted(contact_root.iterdir()):
            if not clip_dir.is_dir():
                continue
            exported_clip_id = ""
            clip_metadata: dict[str, object] = {}
            metadata_path = clip_dir / "metadata.json"
            if metadata_path.is_file():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    inferred_clip_id = _resolve_contact_export_clip_id(
                        clip_dir.name,
                        {clip_id},
                    )
                    if inferred_clip_id == clip_id:
                        raise RuntimeError(
                            "Invalid training contact metadata for the active inference clip: "
                            f"{metadata_path}: {exc}"
                        ) from exc
                    continue
                if not isinstance(payload, dict):
                    inferred_clip_id = _resolve_contact_export_clip_id(
                        clip_dir.name,
                        {clip_id},
                    )
                    if inferred_clip_id == clip_id:
                        raise RuntimeError(
                            "Training contact metadata for the active inference clip must be a JSON object: "
                            f"{metadata_path}"
                        )
                    continue
                if isinstance(payload, dict):
                    clip_metadata = payload
                    exported_clip_id = str(payload.get("clip_id") or "").strip()
            if not exported_clip_id:
                exported_clip_id = _resolve_contact_export_clip_id(clip_dir.name, {clip_id})
            if exported_clip_id == clip_id:
                matching_clip_dirs.append((clip_dir, clip_metadata))

        if len(matching_clip_dirs) > 1:
            raise RuntimeError(
                "Multiple training contact directories match the active inference clip; "
                f"clip={clip_id!r}, directories={[str(path) for path, _ in matching_clip_dirs]}."
            )
        if matching_clip_dirs:
            clip_dir, clip_metadata = matching_clip_dirs[0]
            interval = _load_contact_interval_from_dir(clip_dir)
            if interval is not None:
                return self._materialize_contact_button_window(
                    interval=interval,
                    clip_metadata=clip_metadata,
                    clip_id=clip_id,
                    provenance_source="legacy external",
                )

        message = (
            f"No training contact interval matched clip {clip_id!r} under {str(contact_root)!r}; "
            "using a kinematic fallback would change the student observation contract."
        )
        if not _truthy_env("HOLOSOMA_ALLOW_CONTACT_WINDOW_FALLBACK"):
            raise RuntimeError(
                message
                + " Supply the exact sidecar bank or explicitly set "
                "HOLOSOMA_ALLOW_CONTACT_WINDOW_FALLBACK=1 for a non-equivalent diagnostic rollout."
            )
        logger.warning("{} Explicit fallback override is enabled.", message)
        return None

    def _load_kinematic_button_window(self) -> tuple[int, int]:
        """Load and independently recompute the digest-bound kinematic window."""

        metadata = getattr(self, "_onnx_metadata", {})
        contract = embedded_button_window_contract_from_metadata(metadata)
        if contract is None:
            raise RuntimeError(
                "Kinematic-button policies require a digest-bound integer button-window "
                "contract; legacy/unpatched ONNX artifacts cannot authenticate the source "
                "motion or pickup/drop transition frames. Re-run patch_motion_onnx."
            )
        if (
            self._motion_data is None
            or not self._motion_data.has_object
            or self._motion_data.object_pos_w is None
        ):
            raise RuntimeError(
                "A digest-bound kinematic button-window contract requires an active "
                "motion with object_pos_w; the pickup/drop transitions cannot be verified."
            )

        motion_data = self._motion_data
        settings = self._effective_motion_transition_settings or {}
        source_semantics = str(settings.get("source_semantics", "single_clip_static"))
        prepend_steps = int(self._motion_transition_prepend_steps)
        # Both runtime-hold materialization and single-clip static splicing put
        # the raw source after the realized prefix.  Their button semantics
        # differ only when deriving the effective window below.
        source_offset = max(prepend_steps, 0)
        source_frame_count = int(
            getattr(motion_data, "source_frame_count", motion_data.frame_count)
        )
        source_end = source_offset + source_frame_count
        if source_end > int(motion_data.frame_count):
            raise RuntimeError(
                "Materialized inference timeline is too short for its declared source motion."
            )
        source_rel_z = np.asarray(
            motion_data.object_pos_w[source_offset:source_end, 2]
            - motion_data.root_pos_w[source_offset:source_end, 2],
            dtype=np.float32,
        )
        source_window = kinematic_lift_window_from_rel_z_np(source_rel_z)
        materialized_append_steps = (
            int(motion_data.frame_count) - source_end
        )
        if materialized_append_steps < 0:
            raise RuntimeError(
                "Materialized inference timeline has a negative append length."
            )
        if source_semantics == "single_clip_static":
            materialized_rel_z = np.asarray(
                motion_data.object_pos_w[:, 2] - motion_data.root_pos_w[:, 2],
                dtype=np.float32,
            )
            materialized_window = kinematic_lift_window_from_rel_z_np(
                materialized_rel_z
            )
        else:
            # Runtime-hold semantics keep the source t1 decision active over
            # the prefix.  In particular t1==0 remains 0 rather than +prepend.
            materialized_window = _map_source_window_to_materialized_timeline(
                source_window,
                source_semantics=source_semantics,
                prepend_steps=prepend_steps,
            )

        clip_id = motion_data.motion_path.stem
        expected_scalars = {
            "clip_id": clip_id,
            "source_motion_sha256": motion_data.source_sha256,
            "source_motion_size": int(motion_data.source_size),
            "source_frame_count": source_frame_count,
        }
        for key, expected in expected_scalars.items():
            if contract[key] != expected:
                raise RuntimeError(
                    f"Embedded button-window {key} does not match active motion: "
                    f"contract={contract[key]!r}, active={expected!r}."
                )
        if not math.isclose(
            float(contract["motion_fps"]),
            float(motion_data.fps),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "Embedded button-window contract is bound to a different motion FPS."
            )
        if contract["motion_transition_contract_sha256"] != settings.get(
            "contract_sha256"
        ):
            raise RuntimeError(
                "Embedded button-window contract is bound to different motion-transition metadata."
            )
        if contract["source_semantics"] != source_semantics:
            raise RuntimeError(
                "Embedded button-window source semantics do not match inference metadata."
            )
        if int(contract["effective_prepend_steps"]) != prepend_steps:
            raise RuntimeError(
                "Embedded button-window runtime prepend does not match the materialized motion."
            )
        if int(contract["effective_append_steps"]) != materialized_append_steps:
            raise RuntimeError(
                "Embedded button-window static append does not match the materialized motion."
            )
        if tuple(int(value) for value in contract["source_window"]) != source_window:
            raise RuntimeError(
                "Embedded button-window integers do not match the authenticated source motion."
            )
        if tuple(int(value) for value in contract["materialized_window"]) != materialized_window:
            raise RuntimeError(
                "Embedded button-window materialized integers do not match runtime prepend mapping."
            )
        logger.info(
            "Using digest-bound kinematic policy-button window: clip={} source=[{}, {}) "
            "effective=[{}, {}) prepend_steps={}.",
            clip_id,
            source_window[0],
            source_window[1],
            materialized_window[0],
            materialized_window[1],
            prepend_steps,
        )
        return materialized_window

    def _load_contact_aware_button_window(
        self,
        onnx_path: Path,
    ) -> tuple[int, int] | None:
        """Dispatch button labels without coupling them to root carry mode."""

        mode = validated_contact_aware_button_window_mode(self._motion_cfg or {})
        metadata = getattr(self, "_onnx_metadata", {})
        embedded_contract = embedded_button_window_contract_from_metadata(metadata)
        if mode == "contact_interval":
            if embedded_contract is not None:
                raise RuntimeError(
                    "Legacy contact_interval metadata contains a stale kinematic button-window contract."
                )
            return self._load_contact_interval_window(onnx_path)
        return self._load_kinematic_button_window()

    def setup_policy(self, model_path):
        self._perception_contract_sha256 = None
        self.onnx_policy_session, metadata = self._load_onnx_session_and_metadata(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        self._reset_per_model_motion_state_for_setup()

        self._onnx_metadata = metadata
        self._configure_policy_recurrent_state(metadata)
        embedded_timeline_contract = embedded_motion_timeline_contract_from_metadata(
            metadata
        )
        self._embedded_motion_frame_count = (
            None
            if embedded_timeline_contract is None
            else int(embedded_timeline_contract["embedded_frame_count"])
        )
        self._effective_motion_transition_settings = (
            _validated_runtime_motion_transition_settings(
                metadata,
                apply_training_motion_transitions=(
                    self.config.task.apply_training_motion_transitions
                ),
            )
        )
        self._onnx_obs_dim = self._get_onnx_obs_dim()
        has_policy_contract = validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={inp.name: inp.shape for inp in self.onnx_policy_session.get_inputs()},
            output_shapes={out.name: out.shape for out in self.onnx_policy_session.get_outputs()},
            input_types={inp.name: inp.type for inp in self.onnx_policy_session.get_inputs()},
            output_types={out.name: out.type for out in self.onnx_policy_session.get_outputs()},
            observation=self.config.observation,
            runtime_dof_names=self.dof_names,
            runtime_default_dof_angles=self.default_dof_angles,
            runtime_motor_effort_limits=self.robot_config.motor_effort_limit,
            runtime_joint2motor=self.robot_config.joint2motor,
        )
        self._perception_contract_sha256 = self._effective_perception_contract_sha256(metadata)
        self._has_policy_contract = bool(has_policy_contract)
        if not has_policy_contract:
            self._maybe_force_sparse_depth_distill_obs_config()

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = self._joint_values_to_motor_order(metadata["kp"], "KP") if "kp" in metadata else None
        self.onnx_kd = self._joint_values_to_motor_order(metadata["kd"], "KD") if "kd" in metadata else None

        # Keep WBT rollout aligned with training-time action scaling semantics.
        self._set_policy_action_scales_from_metadata(metadata)

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        required_motion_outputs = {"joint_pos", "joint_vel", "ref_quat_xyzw"}
        if self._policy_requires_motion_data_for_setup():
            self._load_motion_data_from_metadata(metadata, Path(model_path))
        self._validate_runtime_motion_timebase(metadata)

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported ONNX inputs: {self.onnx_input_names}")

        self._time_step_input_name = "time_step" if "time_step" in self.onnx_input_names else None
        self._perception_obs_input_name = actor_perception_input_name_from_metadata(metadata)

        if "actions" in self.onnx_output_names:
            self._action_output_name = "actions"
        elif "action" in self.onnx_output_names:
            self._action_output_name = "action"
        else:
            raise ValueError(f"Unsupported ONNX outputs: {self.onnx_output_names}")

        has_embedded_motion_outputs = required_motion_outputs.issubset(self._motion_output_names)
        source_motion_outputs_from_data = self._should_source_motion_outputs_from_motion_data()
        if self._uses_motion_command and not (
            source_motion_outputs_from_data or has_embedded_motion_outputs
        ):
            raise ValueError(
                "Action-only ONNX policies with motion-command observations require the exact selected "
                "motion data at runtime; alternatively, a legacy combined graph must expose joint_pos, "
                "joint_vel, and ref_quat_xyzw. "
                f"Available ONNX outputs: {self.onnx_output_names}"
            )

        self._onnx_output_fetch = [self._action_output_name]
        if self._uses_motion_command and not source_motion_outputs_from_data:
            self._onnx_output_fetch += ["joint_pos", "joint_vel", "ref_quat_xyzw"]
            if "ref_pos_xyz" in self._motion_output_names:
                self._onnx_output_fetch.append("ref_pos_xyz")

        def policy_act(input_feed):
            return self._run_policy_onnx(input_feed, self._onnx_output_fetch)

        self.policy = policy_act

        if self._uses_motion_command:
            if self._should_source_motion_outputs_from_motion_data():
                outputs = self._get_motion_outputs_from_motion_data(0)
                if outputs is None:
                    raise ValueError("Training-aligned motion data is unavailable for motion outputs.")
            else:
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
            joint_pos = self._require_finite_array(
                outputs["joint_pos"],
                label="initial motion joint position output",
            )
            joint_vel = self._require_finite_array(
                outputs["joint_vel"],
                label="initial motion joint velocity output",
            )
            self.motion_command_t = self._require_finite_array(
                np.concatenate([joint_pos, joint_vel], axis=1),
                label="initial motion command output",
            )
            self.ref_quat_xyzw_t = self._require_finite_array(
                outputs["ref_quat_xyzw"],
                label="initial reference quaternion output",
            )
            ref_pos_xyz = outputs.get("ref_pos_xyz")
            self.ref_pos_xyz_t = (
                None
                if ref_pos_xyz is None
                else self._require_finite_array(
                    ref_pos_xyz,
                    label="initial reference position output",
                )
            )
            self.motion_command_0 = self.motion_command_t.copy()
            self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()
            self._last_motion_output_timestep = 0
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
            clip_observations=self.observation_clip,
        ))
        self._init_obs_config()

    def _build_zero_actor_obs(self) -> np.ndarray:
        obs_dim = self._onnx_obs_dim
        if obs_dim is None:
            obs_dim = int(sum(int(template.shape[1]) for template in self.obs_buf_dict.values()))
        return np.zeros((1, int(obs_dim)), dtype=np.float32)

    def _build_zero_perception_obs(self, input_name: str | None = None) -> np.ndarray:
        input_name = input_name or getattr(
            self,
            "_perception_obs_input_name",
            getattr(self, "_perception_input_name", None),
        )
        input_dim = self._get_onnx_input_dim(input_name)
        if input_dim is None:
            raise ValueError(f"Unable to infer {input_name!r} input dimension from ONNX.")
        return np.zeros((1, input_dim), dtype=np.float32)

    def _query_motion_outputs_at(self, motion_timestep: int) -> dict[str, np.ndarray] | None:
        """Return the motion targets that training exposes at a given clock step."""
        if self._should_source_motion_outputs_from_motion_data():
            return self._get_motion_outputs_from_motion_data(motion_timestep)

        if (
            self._obs_input_name is None
            or "joint_pos" not in self.onnx_output_names
            or "joint_vel" not in self.onnx_output_names
        ):
            return None

        input_feed = {self._obs_input_name: self._build_zero_actor_obs()}
        if self._time_step_input_name:
            input_feed[self._time_step_input_name] = np.array([[int(motion_timestep)]], dtype=np.float32)
        perception_input_name = getattr(
            self,
            "_perception_obs_input_name",
            getattr(self, "_perception_input_name", None),
        )
        if perception_input_name:
            input_feed[perception_input_name] = self._build_zero_perception_obs(perception_input_name)

        fetch_names = ["joint_pos", "joint_vel"]
        if "ref_quat_xyzw" in self.onnx_output_names:
            fetch_names.append("ref_quat_xyzw")
        if "ref_pos_xyz" in self.onnx_output_names:
            fetch_names.append("ref_pos_xyz")
        outputs = self.onnx_policy_session.run(fetch_names, input_feed)
        return dict(zip(fetch_names, outputs))

    def _refresh_motion_outputs_for_current_timestep(self) -> None:
        if not self._uses_motion_command:
            return
        if not self._should_source_motion_outputs_from_motion_data() and self._time_step_input_name is None:
            return

        motion_timestep = self._get_motion_index()
        if getattr(self, "_last_motion_output_timestep", None) == motion_timestep and self.motion_command_t is not None:
            return

        outputs = self._query_motion_outputs_at(motion_timestep)
        if outputs is None or outputs.get("joint_pos") is None or outputs.get("joint_vel") is None:
            return
        self.motion_command_t = np.concatenate(
            [
                np.asarray(outputs["joint_pos"], dtype=np.float32),
                np.asarray(outputs["joint_vel"], dtype=np.float32),
            ],
            axis=1,
        )
        if outputs.get("ref_quat_xyzw") is not None:
            self.ref_quat_xyzw_t = np.asarray(outputs["ref_quat_xyzw"], dtype=np.float32)
        if outputs.get("ref_pos_xyz") is not None:
            self.ref_pos_xyz_t = np.asarray(outputs["ref_pos_xyz"], dtype=np.float32)
        self._last_motion_output_timestep = int(motion_timestep)

    def _sync_motion_outputs_from_onnx(self, motion_index: int) -> None:
        """Compatibility entry point; source targets from the active training contract."""
        self._last_motion_output_timestep = None
        old_motion_timestep = self.motion_timestep
        try:
            # Callers pass the already clamped active motion index.
            self.motion_timestep = int(motion_index) - int(getattr(self, "_motion_index_offset", 0))
            self._refresh_motion_outputs_for_current_timestep()
        finally:
            self.motion_timestep = old_motion_timestep

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "has_policy_contract": bool(getattr(self, "_has_policy_contract", False)),
                "onnx_obs_dim": self._onnx_obs_dim,
                "obs_input_name": self._obs_input_name,
                "time_step_input_name": self._time_step_input_name,
                "perception_obs_input_name": self._perception_obs_input_name,
                "action_output_name": self._action_output_name,
                "onnx_output_fetch": list(self._onnx_output_fetch),
                "motion_output_names": set(self._motion_output_names),
                "embedded_motion_frame_count": getattr(
                    self,
                    "_embedded_motion_frame_count",
                    None,
                ),
                "pinocchio_robot": self.pinocchio_robot,
                "motion_data": self._motion_data,
                "motion_cfg": None if self._motion_cfg is None else dict(self._motion_cfg),
                "motion_body_names": tuple(self._motion_body_names),
                "motion_transition_prepend_steps": int(self._motion_transition_prepend_steps),
                "effective_motion_transition_settings": (
                    None
                    if getattr(self, "_effective_motion_transition_settings", None) is None
                    else dict(self._effective_motion_transition_settings)
                ),
                "contact_aware_carry_window": self._contact_aware_carry_window,
                "contact_aware_contact_window": getattr(
                    self,
                    "_contact_aware_contact_window",
                    None,
                ),
                "contact_aware_button_window": self._contact_aware_button_window,
                "training_freeze_zero_prob": float(self._training_freeze_zero_prob),
                "training_freeze_zero_extra_holds": int(self._training_freeze_zero_extra_holds),
                "motion_alignment_enabled": bool(self._motion_alignment_enabled),
                "motion_command_0": None if self.motion_command_0 is None else self.motion_command_0.copy(),
                "ref_quat_xyzw_0": None if self.ref_quat_xyzw_0 is None else self.ref_quat_xyzw_0.copy(),
                "ref_pos_xyz_0": None if self.ref_pos_xyz_t is None else self.ref_pos_xyz_t.copy(),
            }
        )
        return state

    def _validate_policy_state_collection(self, states: list[dict]) -> None:
        super()._validate_policy_state_collection(states)
        if len(states) > 1 and not all(state["has_policy_contract"] for state in states):
            raise ValueError(
                "WBT multi-policy switching requires complete serialized contracts for every ONNX model; "
                "legacy models can mutate shared observation state and cannot be mixed safely."
            )

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self._has_policy_contract = state["has_policy_contract"]
        self._onnx_obs_dim = state["onnx_obs_dim"]
        self._obs_input_name = state["obs_input_name"]
        self._time_step_input_name = state["time_step_input_name"]
        self._perception_obs_input_name = state["perception_obs_input_name"]
        self._action_output_name = state["action_output_name"]
        self._onnx_output_fetch = list(state["onnx_output_fetch"])
        self._motion_output_names = set(state["motion_output_names"])
        self._embedded_motion_frame_count = state["embedded_motion_frame_count"]
        self.pinocchio_robot = state["pinocchio_robot"]
        self._motion_data = state["motion_data"]
        self._motion_cfg = None if state["motion_cfg"] is None else dict(state["motion_cfg"])
        self._motion_body_names = tuple(state["motion_body_names"])
        self._motion_transition_prepend_steps = state["motion_transition_prepend_steps"]
        self._effective_motion_transition_settings = state[
            "effective_motion_transition_settings"
        ]
        self._contact_aware_carry_window = state["contact_aware_carry_window"]
        self._contact_aware_contact_window = state[
            "contact_aware_contact_window"
        ]
        self._contact_aware_button_window = state["contact_aware_button_window"]
        self._training_freeze_zero_prob = state["training_freeze_zero_prob"]
        self._training_freeze_zero_extra_holds = state["training_freeze_zero_extra_holds"]
        self._motion_alignment_enabled = state["motion_alignment_enabled"]
        self._reset_observation_history_state()
        self.motion_command_0 = (
            None if state["motion_command_0"] is None else state["motion_command_0"].copy()
        )
        self.ref_quat_xyzw_0 = (
            None if state["ref_quat_xyzw_0"] is None else state["ref_quat_xyzw_0"].copy()
        )
        self.ref_pos_xyz_t = None if state["ref_pos_xyz_0"] is None else state["ref_pos_xyz_0"].copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_motion_output_timestep = 0
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._training_freeze_zero_remaining_holds = 0
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        self._motion_end_reset_episode_generation = None

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self._reset_observation_history_state()
        self.motion_command_t = None if self.motion_command_0 is None else self.motion_command_0.copy()
        self.ref_quat_xyzw_t = None if self.ref_quat_xyzw_0 is None else self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None
        self._last_motion_output_timestep = 0
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self._training_freeze_zero_remaining_holds = 0
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        self._motion_end_reset_episode_generation = None
        self._reset_runtime_pickup_latch()

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
        idx = int(self.motion_timestep) + int(getattr(self, "_motion_index_offset", 0))
        if idx < 0:
            return 0
        frame_count = self._active_motion_frame_count()
        # Legacy ONNX motion-command policies may carry their trajectory
        # internally without authenticated frame-count provenance. Preserve
        # their historical unbounded time_step and let the graph clamp it.
        if frame_count is None:
            return idx
        return min(idx, frame_count - 1)

    def _get_file_perception_obs(self, expected_dim: int) -> np.ndarray | None:
        if self._perception_obs_file_path is None:
            return None
        if self._perception_obs_file_values is None:
            path = self._perception_obs_file_path
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    if self._perception_obs_file_key not in data.files:
                        raise KeyError(
                            f"{path} does not contain perception obs key "
                            f"{self._perception_obs_file_key!r}; available={data.files}"
                        )
                    values = np.asarray(data[self._perception_obs_file_key], dtype=np.float32)
            else:
                values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
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
                with np.load(path, allow_pickle=False) as data:
                    if self._policy_action_file_key not in data.files:
                        raise KeyError(
                            f"{path} does not contain action key {self._policy_action_file_key!r}; "
                            f"available={data.files}"
                        )
                    values = np.asarray(data[self._policy_action_file_key], dtype=np.float32)
            else:
                values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
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
            payload: dict[str, object] = {
                "clip_active": bool(self.motion_clip_progressing),
                "policy_step_count": int(self._policy_debug_count),
            }
            payload.update(self._sparse_root_command_overlay_fields())
            pub.publish(payload)
            return

        body_pos_w, root_pos_w, root_quat_wxyz, idx = target
        payload: dict[str, object] = {
            "clip_active": bool(self.motion_clip_progressing),
            "policy_step_count": int(self._policy_debug_count),
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

    @staticmethod
    def _sim_state_payload(snapshot) -> Mapping | None:
        if snapshot is None:
            return None
        payload = getattr(snapshot, "payload", snapshot)
        return payload if isinstance(payload, Mapping) else None

    def _pin_control_tick_state(self, robot_state_data) -> None:
        super()._pin_control_tick_state(robot_state_data)
        snapshot = getattr(self, "_control_tick_sim_state_snapshot", None)
        if snapshot is None and self._sim_state_sub is not None:
            # A non-ZMQ-lowcmd fallback still polls exactly once per control
            # tick. Every observation helper below reads this pinned mapping.
            snapshot = self._sim_state_sub.get_state()
            self._control_tick_sim_state_snapshot = snapshot
        payload = self._sim_state_payload(snapshot)
        if payload is not None:
            self._latest_sim_state = payload

    def _get_latest_sim_state(self) -> Mapping | None:
        if bool(getattr(self, "_control_tick_state_pinned", False)):
            return self._sim_state_payload(
                getattr(self, "_control_tick_sim_state_snapshot", None)
            )

        get_snapshot = getattr(
            getattr(self, "interface", None),
            "get_latest_sim_state_snapshot",
            None,
        )
        if callable(get_snapshot):
            payload = self._sim_state_payload(get_snapshot())
            if payload is not None:
                self._latest_sim_state = payload
            return payload
        if self._sim_state_sub is not None:
            state = self._sim_state_sub.get_state()
            if state is not None:
                self._latest_sim_state = state
        return self._latest_sim_state

    def _get_control_tick_sim_time_ms(self) -> float | None:
        snapshot = getattr(self, "_control_tick_sim_state_snapshot", None)
        sim_time_ms = getattr(snapshot, "sim_time_ms", None)
        if sim_time_ms is None:
            payload = self._sim_state_payload(snapshot)
            if payload is not None:
                sim_time_ms = payload.get("sim_time_ms")
        if sim_time_ms is None:
            return None
        sim_time_ms = float(sim_time_ms)
        if not np.isfinite(sim_time_ms) or sim_time_ms < 0.0:
            raise ValueError(
                f"Pinned simulator sim_time_ms must be finite and non-negative, got {sim_time_ms!r}."
            )
        return sim_time_ms

    def _get_control_tick_episode_generation(self) -> int | None:
        snapshot = getattr(self, "_control_tick_sim_state_snapshot", None)
        episode_generation = getattr(snapshot, "episode_generation", None)
        if episode_generation is None:
            payload = self._sim_state_payload(snapshot)
            if payload is not None:
                episode_generation = payload.get("episode_generation")
        if episode_generation is None:
            return None
        if (
            isinstance(episode_generation, bool)
            or not isinstance(episode_generation, (int, np.integer))
            or int(episode_generation) < 0
            or int(episode_generation) > (1 << 63) - 1
        ):
            raise ValueError(
                "Pinned simulator episode_generation must be a non-negative integer within "
                f"the transport range, got {episode_generation!r}."
            )
        return int(episode_generation)

    def _get_control_clock_ms(self) -> int:
        pinned_sim_time_ms = self._get_control_tick_sim_time_ms()
        if pinned_sim_time_ms is not None:
            return int(round(pinned_sim_time_ms))
        return int(self.clock_sub.get_clock())

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
            raise ValueError(
                f"Simulator robot_root_state must contain at least 13 values, got {root_state_np.shape[1]}."
            )
        return self._require_finite_array(
            root_state_np[:, :13],
            label="simulator robot_root_state",
        )

    def _get_sim_ref_state(self) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        ref_state = state.get("robot_ref_state")
        if ref_state is None:
            return None
        ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(1, -1)
        if ref_state_np.shape[1] < 13:
            raise ValueError(
                f"Simulator robot_ref_state must contain at least 13 values, got {ref_state_np.shape[1]}."
            )
        return self._require_finite_array(
            ref_state_np[:, :13],
            label="simulator robot_ref_state",
        )

    def _get_sim_actor_state(self, actor_name: str) -> np.ndarray | None:
        state = self._get_latest_sim_state()
        if not state:
            return None
        actors = state.get("actors")
        if not isinstance(actors, Mapping) or not actors:
            return None
        actor_state = actors.get(actor_name)
        if actor_state is None and len(actors) == 1:
            actor_state = next(iter(actors.values()))
        if actor_state is None:
            return None
        actor_state_np = np.asarray(actor_state, dtype=np.float32).reshape(1, -1)
        if actor_state_np.shape[1] < 13:
            raise ValueError(
                f"Simulator actor {actor_name!r} state must contain at least 13 values, "
                f"got {actor_state_np.shape[1]}."
            )
        return self._require_finite_array(
            actor_state_np[:, :13],
            label=f"simulator actor {actor_name!r} state",
        )

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
        """Return base linear velocity in the body frame used during training."""
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            return quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 7:10]).astype(
                np.float32,
                copy=False,
            )
        return robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3].astype(
            np.float32,
            copy=False,
        )

    def _get_base_ang_vel_obs(self, robot_state_data: np.ndarray) -> np.ndarray:
        """Return base angular velocity in the body frame used during training."""
        sim_root_state = self._get_sim_root_state()
        if sim_root_state is not None:
            root_quat_wxyz = xyzw_to_wxyz(sim_root_state[:, 3:7])
            return quat_rotate_inverse(root_quat_wxyz, sim_root_state[:, 10:13]).astype(
                np.float32,
                copy=False,
            )
        return robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6].astype(
            np.float32,
            copy=False,
        )

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

        sub = getattr(self, "_manual_sparse_root_command_sub", None)
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

    def _reset_runtime_pickup_latch(self) -> None:
        self._runtime_pickup_latched = False
        self._runtime_pickup_consecutive_counter = 0
        self._runtime_pickup_last_tick = None
        self._runtime_pickup_episode_generation = None
        pickup_step = getattr(self, "_runtime_reference_pickup_step", None)
        if (
            bool(getattr(self, "_precomputed_turn_then_forward_enabled", False))
            and pickup_step is not None
            and getattr(self, "_motion_data", None) is not None
            and self._get_motion_index() >= int(pickup_step)
        ):
            # Training's reset path treats a deliberately late-started clip as
            # already picked. Canonical deployment starts at frame zero.
            self._runtime_pickup_latched = True
            self._runtime_pickup_consecutive_counter = (
                KINEMATIC_LIFT_CONSECUTIVE_STEPS
            )

    def _update_runtime_pickup_latch(self, robot_state_data: np.ndarray) -> None:
        if not self._precomputed_turn_then_forward_enabled:
            return
        threshold = self._runtime_pickup_threshold_rel_z
        if threshold is None:
            raise RuntimeError(
                "Precomputed turn-then-forward deployment is missing its clip pickup threshold."
            )

        sim_time_ms = self._get_control_tick_sim_time_ms()
        if sim_time_ms is None:
            raise RuntimeError(
                "Precomputed turn-then-forward pickup latch requires an authenticated simulator "
                "tick timestamp so one physics step cannot be counted more than once."
            )
        episode_generation = self._get_control_tick_episode_generation()
        if (
            self._runtime_pickup_episode_generation is not None
            and episode_generation is not None
            and episode_generation != self._runtime_pickup_episode_generation
        ):
            self._reset_runtime_pickup_latch()
        self._runtime_pickup_episode_generation = episode_generation
        tick = (episode_generation, float(sim_time_ms))
        if tick == self._runtime_pickup_last_tick:
            return
        self._runtime_pickup_last_tick = tick
        if self._runtime_pickup_latched:
            return

        object_state = self._get_sim_actor_state(self.config.task.sim_object_name)
        if object_state is None:
            raise RuntimeError(
                "Precomputed turn-then-forward pickup latch requires the live simulator object "
                "state; reference-motion object pose is not a valid fallback."
            )
        robot_root_pos_w = np.asarray(robot_state_data[:, :3], dtype=np.float32)
        current_rel_z = np.asarray(
            object_state[:, 2] - robot_root_pos_w[:, 2],
            dtype=np.float32,
        )
        if current_rel_z.shape != (1,) or not np.all(np.isfinite(current_rel_z)):
            raise RuntimeError(
                "Precomputed turn-then-forward pickup latch received an invalid simulator rel-z."
            )
        if bool(current_rel_z[0] >= threshold):
            self._runtime_pickup_consecutive_counter += 1
        else:
            self._runtime_pickup_consecutive_counter = 0
        if (
            self._runtime_pickup_consecutive_counter
            >= KINEMATIC_LIFT_CONSECUTIVE_STEPS
        ):
            self._runtime_pickup_latched = True

    def _get_precomputed_turn_then_forward_command(
        self,
        robot_state_data: np.ndarray,
    ) -> np.ndarray:
        motion = self._motion_data
        if (
            not self._precomputed_turn_then_forward_enabled
            or motion is None
            or not motion.has_precomputed_root_command
        ):
            raise RuntimeError(
                "Precomputed turn-then-forward command was requested without its runtime contract."
            )
        self._update_runtime_pickup_latch(robot_state_data)
        idx = self._get_motion_index()
        raw_command = motion.precomputed_root_command[idx : idx + 1].astype(
            np.float32,
            copy=True,
        )
        motion_command = (
            raw_command
            if self._runtime_pickup_latched
            else np.zeros_like(raw_command, dtype=np.float32)
        )
        effective_command = self._apply_external_sparse_root_command(motion_command)
        if not self._last_sparse_manual_enabled:
            self._record_sparse_root_command(
                motion_command,
                effective_command,
                source="auto_precomputed_turn_then_forward",
                mode="motion",
                manual_enabled=False,
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

    def _get_rolling_reference_delta_command(self) -> np.ndarray:
        """Mirror training's per-step rolling reference-to-reference delta."""

        motion = self._motion_data
        if motion is None or not motion.has_object:
            raise ValueError(
                "rolling_reference_delta requires authenticated object motion data."
            )
        motion_cfg = self._motion_cfg or {}
        raw_lookahead = motion_cfg.get(
            "contact_aware_sparse_root_segment_steps",
            30,
        )
        if (
            isinstance(raw_lookahead, (bool, np.bool_))
            or not isinstance(raw_lookahead, (int, np.integer))
            or int(raw_lookahead) < 1
        ):
            raise ValueError(
                "motion_config.contact_aware_sparse_root_segment_steps must be a "
                f"positive integer, got {raw_lookahead!r}."
            )
        raw_yaw_threshold_deg = motion_cfg.get(
            "contact_aware_sparse_root_zero_yaw_threshold_deg",
            0.0,
        )
        if (
            isinstance(raw_yaw_threshold_deg, (bool, np.bool_))
            or not isinstance(raw_yaw_threshold_deg, (int, float, np.integer, np.floating))
            or not np.isfinite(float(raw_yaw_threshold_deg))
            or not 0.0 <= float(raw_yaw_threshold_deg) <= 180.0
        ):
            raise ValueError(
                "motion_config.contact_aware_sparse_root_zero_yaw_threshold_deg "
                f"must be a finite real in [0, 180], got {raw_yaw_threshold_deg!r}."
            )

        idx = self._get_motion_index()
        lookahead = int(raw_lookahead)
        endpoint = idx + lookahead
        carry_start, carry_end = self._get_contact_aware_carry_window()
        valid = (
            carry_start <= idx < carry_end
            and endpoint < carry_end
            and endpoint < int(motion.frame_count)
        )
        if not valid:
            command = np.zeros((1, 3), dtype=np.float32)
            return self._apply_external_sparse_root_command(command)

        start_pos_w = motion.root_pos_w[idx : idx + 1]
        endpoint_pos_w = motion.root_pos_w[endpoint : endpoint + 1]
        start_quat_wxyz = motion.root_quat_w[idx : idx + 1]
        endpoint_quat_wxyz = motion.root_quat_w[endpoint : endpoint + 1]
        rel_pos_w = endpoint_pos_w - start_pos_w
        rel_pos_b = quat_apply(
            self._calc_heading_quat_inv(start_quat_wxyz),
            rel_pos_w,
        )
        rel_yaw_value = self._normalize_angle(
            self._quat_yaw(endpoint_quat_wxyz)
            - self._quat_yaw(start_quat_wxyz)
        )
        yaw_threshold_rad = np.deg2rad(
            np.float32(float(raw_yaw_threshold_deg))
        )
        if abs(float(rel_yaw_value)) <= float(yaw_threshold_rad):
            rel_yaw_value = 0.0
        rel_yaw = np.asarray([[rel_yaw_value]], dtype=np.float32)
        command = np.concatenate([rel_pos_b[:, :2], rel_yaw], axis=1).astype(
            np.float32,
            copy=False,
        )
        if not np.all(np.isfinite(command)):
            raise RuntimeError(
                "rolling_reference_delta produced a non-finite actor command."
            )
        return self._apply_external_sparse_root_command(command)

    def _get_contact_aware_carry_window(self) -> tuple[int, int]:
        if self._contact_aware_carry_window is not None:
            return self._contact_aware_carry_window
        if self._motion_data is None or not self._motion_data.has_object or self._motion_data.object_pos_w is None:
            end = 0 if self._motion_data is None else int(self._motion_data.frame_count)
            self._contact_aware_carry_window = (0, end)
            return self._contact_aware_carry_window

        cfg = self._motion_cfg or {}
        mode, alpha, smoothing_steps = _validated_contact_aware_carry_window_config(cfg)
        consecutive_steps = 5
        total_steps = int(self._motion_data.frame_count)
        if total_steps <= 0:
            self._contact_aware_carry_window = (0, 0)
            return self._contact_aware_carry_window

        source_semantics = str(
            (getattr(self, "_effective_motion_transition_settings", None) or {}).get(
                "source_semantics",
                "single_clip_static",
            )
        )
        prepend_steps = int(getattr(self, "_motion_transition_prepend_steps", 0) or 0)
        source_offset = (
            prepend_steps
            if source_semantics == "global_multi_clip_runtime" and prepend_steps > 0
            else 0
        )
        source_total_steps = total_steps - source_offset
        if source_total_steps <= 0:
            raise ValueError(
                "Materialized runtime prepend leaves no source motion frames for the "
                "contact-aware carry-window contract."
            )
        source_object_pos_w = self._motion_data.object_pos_w[source_offset:]
        source_root_pos_w = self._motion_data.root_pos_w[source_offset:]

        if mode == "peak_height":
            height = _smooth_1d_edge_padded(source_object_pos_w[:, 2], smoothing_steps)
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
                start_idx=min(peak_step + 1, source_total_steps),
            )
            if carry_end is None:
                carry_end = source_total_steps

        else:
            rel_z = source_object_pos_w[:, 2] - source_root_pos_w[:, 2]
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
                start_idx=min(int(carry_start) + 1, source_total_steps),
            )
            if carry_end is None:
                carry_end = source_total_steps

        carry_start = max(0, min(int(carry_start), source_total_steps))
        carry_end = max(carry_start, min(int(carry_end), source_total_steps))
        carry_start, carry_end = _map_source_window_to_materialized_timeline(
            (carry_start, carry_end),
            source_semantics=source_semantics,
            prepend_steps=source_offset,
        )

        # Training's rel-z contact-aware root command uses the exported t2
        # interval to stop before release. The sidecar window has already been
        # mapped onto the materialized clock, so cap only after the kinematic
        # source window receives the same mapping.
        contact_window = getattr(
            self,
            "_contact_aware_contact_window",
            getattr(self, "_contact_aware_button_window", None),
        )
        if mode == "rel_z" and contact_window is not None:
            _, contact_t2 = contact_window
            release_start = max(
                0,
                min(int(contact_t2) - _CONTACT_STAGE_RELEASE_LEAD_STEPS, total_steps),
            )
            carry_end = min(carry_end, release_start)
        carry_end = max(carry_start, carry_end)
        self._contact_aware_carry_window = (carry_start, carry_end)
        logger.info("Contact-aware sparse root command active window: [{}, {}) mode={}", carry_start, carry_end, mode)
        return self._contact_aware_carry_window

    def _get_sparse_target_root_trajectory_command_contact_aware(
        self,
        robot_state_data: np.ndarray,
        base_command: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._precomputed_turn_then_forward_enabled:
            return self._get_precomputed_turn_then_forward_command(robot_state_data)
        if _normalized_sparse_root_command_mode(
            self._motion_cfg or {}
        ) == "rolling_reference_delta":
            return self._get_rolling_reference_delta_command()
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

    def _get_external_button_override(self, button_name: str) -> np.ndarray | None:
        """Return one strict binary external button override, if published.

        The sparse-root publisher is also the authenticated manual-button
        transport for split rollouts.  Valid scalar values are thresholded at
        0.5; malformed, non-scalar, or non-finite payloads are ignored so they
        cannot silently inject a different command bit.
        """

        if button_name not in {"pickup", "drop"}:
            raise ValueError(f"Unsupported external policy button: {button_name!r}")
        sub = getattr(self, "_manual_sparse_root_command_sub", None)
        if sub is None:
            return None
        payload = sub.get_payload()
        payload_key = f"{button_name}_button"
        if not isinstance(payload, dict) or payload_key not in payload:
            return None
        raw_value = payload.get(payload_key)
        try:
            values = np.asarray(raw_value, dtype=np.float32).reshape(-1)
            if values.size != 1:
                raise ValueError(f"expected one scalar, got {values.size}")
            button_value = float(values[0])
            if not np.isfinite(button_value):
                raise ValueError("value must be finite")
        except (TypeError, ValueError, IndexError):
            logger.warning(
                "Ignoring malformed external {} button value: {}",
                button_name,
                raw_value,
            )
            return None
        button_value = 1.0 if button_value >= 0.5 else 0.0
        log_attr = f"_manual_{button_name}_button_log_value"
        if getattr(self, log_attr, None) != button_value:
            logger.info("External {} button override: {}", button_name, int(button_value))
            setattr(self, log_attr, button_value)
        return np.array([[button_value]], dtype=np.float32)

    def _get_external_pickup_button_override(self) -> np.ndarray | None:
        return self._get_external_button_override("pickup")

    def _get_external_drop_button_override(self) -> np.ndarray | None:
        return self._get_external_button_override("drop")

    def _get_drop_button(self) -> np.ndarray:
        external_drop_button = self._get_external_drop_button_override()
        if external_drop_button is not None:
            return external_drop_button
        if self._motion_data is None or not self._motion_data.has_object:
            return np.zeros((1, 1), dtype=np.float32)
        _, carry_end = self._get_contact_aware_button_window()
        return np.array([[1.0 if self._get_motion_index() >= carry_end else 0.0]], dtype=np.float32)

    def _get_pickup_button(self) -> np.ndarray:
        external_pickup_button = self._get_external_pickup_button_override()
        if external_pickup_button is not None:
            return external_pickup_button
        if self._motion_data is None or not self._motion_data.has_object:
            return np.zeros((1, 1), dtype=np.float32)
        carry_start, _ = self._get_contact_aware_button_window()
        return np.array([[1.0 if self._get_motion_index() < carry_start else 0.0]], dtype=np.float32)

    def _apply_drop_exclusive_root_command(
        self,
        command: np.ndarray,
        drop_button: np.ndarray,
    ) -> np.ndarray:
        """Mirror the training-side final actor command/drop gate exactly."""

        if not _validated_zero_root_command_when_drop_active(
            self._motion_cfg or {}
        ):
            return command
        command_array = np.asarray(command)
        drop_array = np.asarray(drop_button)
        if command_array.shape != (1, 3):
            raise RuntimeError(
                "Drop-exclusive root command must have shape (1, 3), "
                f"got {command_array.shape}."
            )
        if drop_array.shape != (1, 1):
            raise RuntimeError(
                "Drop-exclusive button must have shape (1, 1), "
                f"got {drop_array.shape}."
            )
        if not np.issubdtype(command_array.dtype, np.floating):
            raise RuntimeError(
                "Drop-exclusive root command must use a floating dtype, "
                f"got {command_array.dtype}."
            )
        if not np.all(np.isfinite(command_array)) or not np.all(
            np.isfinite(drop_array)
        ):
            raise RuntimeError(
                "Drop-exclusive actor command and button must contain only finite values."
            )
        return np.where(
            drop_array >= np.float32(0.5),
            np.zeros_like(command_array),
            command_array,
        ).astype(np.float32, copy=False)

    def _get_contact_aware_button_window(self) -> tuple[int, int]:
        if self._contact_aware_button_window is not None:
            return self._contact_aware_button_window
        return self._get_contact_aware_carry_window()

    def _get_depth_distill_obs_buffer_dict(self, robot_state_data: np.ndarray) -> dict[str, np.ndarray]:
        sparse_command = self._get_sparse_target_root_trajectory_command(robot_state_data)
        contact_aware_sparse_command = (
            self._get_sparse_target_root_trajectory_command_contact_aware(robot_state_data, sparse_command)
            if self._uses_sparse_root_command_contact_aware
            else sparse_command
        )
        drop_button = self._get_drop_button()
        sparse_command = self._apply_drop_exclusive_root_command(
            sparse_command,
            drop_button,
        )
        contact_aware_sparse_command = self._apply_drop_exclusive_root_command(
            contact_aware_sparse_command,
            drop_button,
        )
        return {
            "sparse_target_root_trajectory_command": sparse_command,
            "sparse_target_root_trajectory_command_contact_aware": contact_aware_sparse_command,
            "pickup_button": self._get_pickup_button(),
            "drop_button": drop_button,
            "base_lin_vel": self._get_base_lin_vel_obs(robot_state_data),
            "base_ang_vel": self._get_base_ang_vel_obs(robot_state_data),
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

        return {
            "sparse_target_root_trajectory_command": self._get_sparse_target_root_trajectory_command(robot_state_data),
            "base_ang_vel": self._get_base_ang_vel_obs(robot_state_data),
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
        if sim_object_state is None:
            raise RuntimeError(
                "Object-policy observations require a valid current object state from the simulator bridge. "
                "Substituting the motion target would silently collapse obj_pos_b/obj_ori_b tracking error."
            )
        current_object_pos_w = sim_object_state[:, :3]
        current_object_quat_wxyz = xyzw_to_wxyz(sim_object_state[:, 3:7])
        current_object_lin_vel_w = sim_object_state[:, 7:10]
        current_object_ang_vel_w = sim_object_state[:, 10:13]

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
            current_object_lin_vel_w,
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
            # Current object-training checkpoints serialize these target
            # components as three independently sorted terms.  They have the
            # same aggregate width as ``obj_target_pose_size_b`` but a
            # different flattened layout, so all four representations remain
            # explicit and the selected metadata-backed preset decides which
            # keys enter the policy input.
            "obj_size": object_size,
            "obj_target_ori_b": obj_target_rot6d.astype(np.float32, copy=False),
            "obj_target_pos_b": obj_target_pos_b.astype(np.float32, copy=False),
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
        self._maybe_complete_motion_end_reset()

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
        consumed_motion_index = motion_index
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
                perception_target_episode_generation = (
                    self._get_control_tick_episode_generation()
                )
                if os.environ.get("HOLOSOMA_POLICY_ALIGN_PERCEPTION_TO_SIM_STATE", "1").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    perception_target_sim_time_ms = self._get_control_tick_sim_time_ms()
                    if perception_target_sim_time_ms is not None:
                        try:
                            perception_target_sim_time_ms += float(
                                os.environ.get("HOLOSOMA_POLICY_PERCEPTION_TARGET_OFFSET_MS", "0") or "0"
                            )
                        except ValueError:
                            pass
                perception_obs = self._get_split_perception_obs(
                    perception_dim,
                    target_sim_time_ms=perception_target_sim_time_ms,
                    target_episode_generation=perception_target_episode_generation,
                )
            input_feed[self._perception_obs_input_name] = perception_obs
        self._consume_root_reference_at_clip_start()
        outputs = self.policy(input_feed)
        policy_action = self._require_finite_array(
            outputs[self._action_output_name],
            label=f"ONNX output {self._action_output_name!r}",
        )
        action_override = self._get_file_policy_action()
        if action_override is not None:
            policy_action = self._require_finite_array(
                action_override,
                label="policy action file override",
            )

        if self._uses_motion_command and not self._should_source_motion_outputs_from_motion_data():
            joint_pos = outputs.get("joint_pos")
            joint_vel = outputs.get("joint_vel")
            if joint_pos is None or joint_vel is None:
                raise ValueError("Motion outputs missing during inference.")
            self.motion_command_t = self._require_finite_array(
                np.concatenate([joint_pos, joint_vel], axis=1),
                label="ONNX motion command output",
            )
            self.ref_quat_xyzw_t = self._require_finite_array(
                outputs.get("ref_quat_xyzw", self.ref_quat_xyzw_t),
                label="ONNX reference quaternion output",
            )
            ref_pos_xyz = outputs.get("ref_pos_xyz", self.ref_pos_xyz_t)
            self.ref_pos_xyz_t = (
                None
                if ref_pos_xyz is None
                else self._require_finite_array(
                    ref_pos_xyz,
                    label="ONNX reference position output",
                )
            )

        policy_action = self._update_policy_action_state(
            policy_action,
            label="policy action selected for control",
        )
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
        self.scaled_policy_action = self._require_finite_array(
            self.scaled_policy_action,
            label="final scaled policy action",
        )
        self._maybe_debug_policy_io(robot_state_data, obs["actor_obs"], perception_obs, policy_action)
        self._publish_policy_overlay()

        # Preserve the action produced for the consumed final frame even when
        # the restart path clears internal action history immediately after it.
        control_action = self.scaled_policy_action.copy()
        self._advance_motion_after_policy_step(consumed_motion_index)
        return control_action

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
            "clock_ms": self._get_control_clock_ms() if self.use_sim_time else None,
            "control_target_clock_ms": (
                int(self._last_policy_control_target_clock_ms)
                if self._last_policy_control_target_clock_ms is not None
                else None
            ),
            "sim_time_ms": self._get_control_tick_sim_time_ms(),
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
        for key in (
            "sparse_target_root_trajectory_command",
            "sparse_target_root_trajectory_command_contact_aware",
            "pickup_button",
            "drop_button",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
        ):
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

    def _advance_motion_after_policy_step(self, consumed_motion_index: int) -> None:
        if not self.motion_clip_progressing:
            return
        if not self.use_sim_time:
            self.motion_timestep += 1
        self._maybe_restart_sim_at_motion_end(
            consumed_motion_index=consumed_motion_index,
        )

    def _hold_at_motion_end(self) -> None:
        frame_count = self._active_motion_frame_count()
        if frame_count is None:
            return
        last_index = max(frame_count - 1, 0)
        offset = int(getattr(self, "_motion_index_offset", 0))
        self.motion_timestep = max(last_index - offset, 0)

    def _maybe_complete_motion_end_reset(self) -> None:
        if not self._motion_end_reset_requested:
            return
        requested_generation = self._motion_end_reset_episode_generation
        if requested_generation is None:
            return
        current_generation = self._get_control_tick_episode_generation()
        if current_generation is None or current_generation == requested_generation:
            return
        if current_generation < requested_generation:
            raise RuntimeError(
                "Simulator episode_generation regressed while awaiting motion-end reset: "
                f"requested_from={requested_generation}, current={current_generation}."
            )
        self.logger.info(
            "Observed simulator reset acknowledgement: episode_generation {} -> {}; restarting clip.",
            requested_generation,
            current_generation,
        )
        self._handle_start_motion_clip()

    def _maybe_restart_sim_at_motion_end(self, *, consumed_motion_index: int) -> None:
        if not bool(getattr(self.config.task, "restart_sim_on_motion_end", False)):
            return
        frame_count = self._active_motion_frame_count()
        if frame_count is None:
            return
        if self._motion_end_reset_requested:
            self._hold_at_motion_end()
            return
        if int(consumed_motion_index) < frame_count - 1:
            return

        if self._disable_motion_end_sim_reset:
            self._hold_at_motion_end()
            self._motion_end_reset_requested = True
            self._motion_end_reset_episode_generation = None
            self.logger.info("Motion clip reached the end; automatic simulator reset is disabled.")
            return

        self._motion_end_reset_requested = True
        self._hold_at_motion_end()
        episode_generation = self._get_control_tick_episode_generation()
        self._motion_end_reset_episode_generation = episode_generation
        sim_control_pub = getattr(self.interface, "_sim_control_pub", None)
        if episode_generation is None:
            self.logger.error(
                "Motion clip reached the end without a pinned simulator episode_generation; "
                "holding the final frame instead of replaying against unreset physics."
            )
            return
        if sim_control_pub is None or not hasattr(sim_control_pub, "request_reset"):
            self.logger.error(
                "Motion clip reached the end, but the simulator reset channel is unavailable; "
                "holding the final frame instead of replaying against unreset physics."
            )
            return
        reset_sent = sim_control_pub.request_reset("motion_end")
        if reset_sent is not True:
            self.logger.error(
                "Motion-end simulator reset could not be published; holding the final frame "
                "instead of replaying against unreset physics."
            )
            return
        self.logger.info(
            "Motion clip reached the end; requested simulator reset and awaiting an "
            "episode_generation acknowledgement before restarting the clip."
        )

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
            prime_steps = self._get_autostart_policy_history_prime_steps()
            priming_required = (
                self._dryrun_autostart_policy_history
                and self._warm_autostart_obs_history
                and prime_steps > 0
            )
            if priming_required and not self._prime_auto_start_policy_history(robot_state_data):
                return
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

        current_clock = self._get_control_clock_ms()
        if self._sim_time_control_schedule_ms:
            index = min(self._sim_time_control_schedule_index, len(self._sim_time_control_schedule_ms) - 1)
            target_clock = int(self._sim_time_control_schedule_ms[index])
            if current_clock < target_clock:
                return True
            motion_timestep = int(self._sim_time_control_schedule_index)
            frame_count = self._active_motion_frame_count()
            if self._disable_motion_end_sim_reset and frame_count is not None:
                motion_timestep = min(motion_timestep, max(frame_count - 1, 0))
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
            robot_state_data = self._get_control_tick_robot_state()
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(self._augment_robot_state_with_sim_state(robot_state_data))

    def _update_clock(self):
        # Use synchronized clock with motion-relative timing
        current_clock = self._get_control_clock_ms()
        if self._training_freeze_zero_remaining_holds > 0:
            self._training_freeze_zero_remaining_holds -= 1
            if not self._logged_training_freeze_zero_alignment:
                self.logger.info(
                    "Applying training-like timestep-0 hold: prob={:.3f}, deterministic_holds={}.",
                    self._training_freeze_zero_prob,
                    self._training_freeze_zero_extra_holds,
                )
                self._logged_training_freeze_zero_alignment = True
            if self._training_freeze_zero_remaining_holds == 0:
                self.motion_timestep = 1
                self.motion_start_timestep = current_clock - int(round(self.timestep_interval_ms))
            else:
                self.motion_timestep = 0
                self.motion_start_timestep = current_clock
            self._last_clock_reading = current_clock
            return
        if self.motion_start_timestep is None:
            # Motion just started; anchor to the first received clock tick.
            self.motion_start_timestep = current_clock
        elif self._last_clock_reading is not None and current_clock < self._last_clock_reading:
            if bool(getattr(self.config.task, "restart_motion_on_clock_reset", False)):
                self.logger.warning("Clock sync returned earlier timestamp; restarting motion clip from frame 0.")
                self._handle_start_motion_clip()
                current_clock = self._get_control_clock_ms()
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
        frame_count = self._active_motion_frame_count()
        if self._disable_motion_end_sim_reset and frame_count is not None:
            self.motion_timestep = min(self.motion_timestep, max(frame_count - 1, 0))
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
        self._reset_observation_history_state()
        self._preserve_obs_history_on_next_motion_start = False
        self._preserve_root_reference_state_on_next_motion_start = False
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.motion_timestep = 0
        self.motion_start_timestep = None  # Reset motion start time
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self._last_motion_output_timestep = 0
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self.robot_yaw_offset = 0.0
        self._logged_root_reference_clip_start = False
        self._remaining_root_reference_clip_start_obs = 0
        self._training_freeze_zero_remaining_holds = 0
        self._logged_training_freeze_zero_alignment = False
        self._logged_sim_ref_from_sim_state = False
        self._motion_align_quat_wxyz = None
        self._motion_align_pos = None
        self._motion_end_reset_requested = False
        self._motion_end_reset_episode_generation = None
        self._reset_runtime_pickup_latch()

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.clock_sub.reset_origin()
        preserve_root_reference_state = False
        if getattr(self, "_preserve_obs_history_on_next_motion_start", False):
            self._preserve_obs_history_on_next_motion_start = False
            preserve_root_reference_state = getattr(
                self,
                "_preserve_root_reference_state_on_next_motion_start",
                False,
            )
        else:
            self._reset_observation_history_state()
        self._preserve_root_reference_state_on_next_motion_start = False
        self._auto_start_history_snapshot = None
        self.motion_clip_progressing = True
        # Capture motion-specific start timestep for policy-level timing control
        self.motion_start_timestep = None  # will be set in rl_inference
        self.motion_timestep = 0  # Reset to start from beginning of motion
        self._reset_runtime_pickup_latch()
        self._last_motion_output_timestep = None
        if self.motion_command_0 is not None:
            self.motion_command_t = self.motion_command_0.copy()
        if self.ref_quat_xyzw_0 is not None:
            self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self._refresh_motion_outputs_for_current_timestep()
        self._last_clock_reading = None
        self._last_policy_control_clock_ms = None
        self._sim_time_control_schedule_index = 0
        self._last_policy_control_target_clock_ms = None
        self._training_freeze_zero_remaining_holds = getattr(self, "_training_freeze_zero_extra_holds", 0)
        self._logged_training_freeze_zero_alignment = False
        if not preserve_root_reference_state:
            self._logged_root_reference_clip_start = False
            self._remaining_root_reference_clip_start_obs = (
                1 if bool(getattr(self.config.task, "use_root_reference_at_clip_start", False)) else 0
            )
        self._logged_first_policy_step_debug = False
        self._auto_start_motion_clip_hold_start_time = None
        self._auto_start_motion_clip_last_log_time = 0.0
        self._motion_end_reset_requested = False
        self._motion_end_reset_episode_generation = None
        motion_start_robot_state = (
            self._get_control_tick_robot_state()
            if self._motion_alignment_enabled
            or getattr(self, "_prefill_obs_history_on_motion_start", False)
            else None
        )
        if self._motion_alignment_enabled:
            robot_state_data = motion_start_robot_state
            if robot_state_data is not None:
                self._maybe_update_motion_alignment(self._augment_robot_state_with_sim_state(robot_state_data))
        if getattr(self, "_prefill_obs_history_on_motion_start", False):
            robot_state_data = motion_start_robot_state
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
        robot_state_data = self._augment_robot_state_with_sim_state(
            self._get_control_tick_robot_state()
        )
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
