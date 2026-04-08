from __future__ import annotations

import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any, List

import numpy as np
import smart_open
import torch
import torch.nn.functional as F
from loguru import logger

from holosoma.config_types.command import (
    CleanNoisyClipCurriculumConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
    SparseObjectGoalConfig,
)
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.clip_sampling import build_prefix_mask, piecewise_constant_schedule_value, project_group_weights
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.object_geometry import load_urdf_geometry_extents
from holosoma.utils.rotations import (
    get_euler_xyz,
    normalize_angle,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quaternion_to_matrix,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType

_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD = 0.10
_RUNTIME_PICKUP_CONSECUTIVE_STEPS = 5
_CLIP_PICKUP_LIFT_RATIO_THRESHOLD = 0.35
_CONTACT_PRIOR_REGION_NAMES = ("left_palm", "right_palm", "arms", "torso")
_CONTACT_PRIOR_REGION_FORCE_BODY_NAMES = {
    "left_palm": ("left_wrist_yaw_link",),
    "right_palm": ("right_wrist_yaw_link",),
    "arms": (
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
    ),
    "torso": ("torso_link",),
}
_CONTACT_PRIOR_REGION_POSITION_BODY_NAMES = {
    "left_palm": ("left_wrist_yaw_link",),
    "right_palm": ("right_wrist_yaw_link",),
    "arms": ("left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    "torso": ("torso_link",),
}
_CONTACT_PRIOR_PHASE_COUNT = 2
_CONTACT_PRIOR_FORCE_THRESHOLD = 1.0
_CONTACT_PRIOR_OBJECT_POS_ERROR_THRESHOLD = 0.20
_CONTACT_PRIOR_OBJECT_ROT_ERROR_THRESHOLD = 0.80
_CONTACT_PRIOR_BODY_POS_ERROR_THRESHOLD = 0.35
_CONTACT_PRIOR_CONFIDENCE_WARMUP_SAMPLES = 2048.0
_OBJECT_CONTACT_PROXY_DISTANCE_THRESHOLD = 0.08


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first_col = F.normalize(rot6d[..., 0:3], dim=-1)
    second_col_raw = rot6d[..., 3:6]
    second_col = F.normalize(
        second_col_raw - torch.sum(first_col * second_col_raw, dim=-1, keepdim=True) * first_col,
        dim=-1,
    )
    third_col = torch.cross(first_col, second_col, dim=-1)
    return torch.stack((first_col, second_col, third_col), dim=-1)


def _first_sustained_true_index(mask: torch.Tensor, consecutive_steps: int) -> int | None:
    """Return the earliest index where `mask` stays true for `consecutive_steps` frames."""
    if mask.numel() == 0:
        return None
    if consecutive_steps <= 1:
        true_indices = torch.nonzero(mask, as_tuple=False)
        if true_indices.numel() == 0:
            return None
        return int(true_indices[0, 0].item())

    run_length = 0
    for idx, flag in enumerate(mask.detach().cpu().tolist()):
        run_length = run_length + 1 if flag else 0
        if run_length >= consecutive_steps:
            return idx - consecutive_steps + 1
    return None


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    _OBJECT_SIZE_KEYS = (
        "object_size",
        "box_size",
        "object_scale",
        "box_scale",
    )

    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        motion_clip_id: int | None = None,
        motion_clip_name: str | None = None,
        object_size_scale: list[float] | None = None,
    ):
        self._robot_body_names = list(robot_body_names)
        self._robot_joint_names = list(robot_joint_names)
        self._object_size_scale = self._normalize_object_size_scale(object_size_scale)

        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)
        motion_path = Path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        self.clip_ids: list[str] = []
        self.clip_object_names: list[str] = []
        self.clip_object_urdf_paths: list[str] = []
        self.clip_offsets = torch.zeros(0, dtype=torch.long, device=device)
        self.clip_lengths = torch.zeros(0, dtype=torch.long, device=device)
        self.num_clips = 0
        self.motion_clip_id = motion_clip_id
        self.motion_clip_name = motion_clip_name
        if motion_path.is_dir():
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz_dir(
                motion_path,
                device,
                motion_clip_id=motion_clip_id,
                motion_clip_name=motion_clip_name,
            )
        elif motion_file.endswith((".h5", ".hdf5")):
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_h5(
                motion_file,
                device,
                motion_clip_id=motion_clip_id,
                motion_clip_name=motion_clip_name,
            )
        else:
            body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]
        self._apply_object_size_scale()

    @staticmethod
    def _normalize_object_size_scale(raw_scale: list[float] | None) -> np.ndarray | None:
        if raw_scale is None:
            return None
        arr = np.asarray(raw_scale, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size == 1:
            value = float(arr[0])
            return np.array([value, value, value], dtype=np.float32)
        if arr.size == 3:
            return arr.astype(np.float32, copy=False)
        raise ValueError(
            "MotionConfig.object_size_scale must have length 1 or 3. "
            f"Got shape {arr.shape} from value {raw_scale!r}."
        )

    def _apply_object_size_scale(self) -> None:
        if self._object_size_scale is None or not hasattr(self, "_object_size"):
            return
        if self._object_size.numel() == 0:
            return
        scale = torch.tensor(self._object_size_scale, dtype=self._object_size.dtype, device=self._object_size.device)
        self._object_size = self._object_size * scale.view(1, 3)

    @classmethod
    def _normalize_object_size_array(cls, raw: np.ndarray, length: int, *, source: str) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 0:
            scalar = float(arr)
            return np.full((length, 3), scalar, dtype=np.float32)

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
            if arr.shape == (3, 1):
                row = arr.reshape(1, 3)
                return np.repeat(row, repeats=length, axis=0)
            if arr.shape == (length, 1):
                return np.repeat(arr, repeats=3, axis=1)
            if arr.shape == (3, length):
                return arr.transpose(1, 0)
            if arr.shape == (length, 3):
                return arr

        raise ValueError(
            f"Unsupported object-size shape {arr.shape} in {source}; "
            "expected scalar, (3,), (T,), (T,3), (1,3), or (T,1)."
        )

    @classmethod
    def _extract_object_size_np(cls, data: Any, length: int, *, source: str) -> np.ndarray:
        for key in cls._OBJECT_SIZE_KEYS:
            if key in data:
                raw = np.asarray(data[key], dtype=np.float32)
                return cls._normalize_object_size_array(raw, length, source=f"{source}:{key}")
        return np.ones((length, 3), dtype=np.float32)

    @staticmethod
    def _scalar_str(value: Any) -> str:
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.shape == ():
            item = arr.item()
        else:
            item = arr.reshape(-1)[0]
            if hasattr(item, "item"):
                item = item.item()
        return str(item).strip()

    @classmethod
    def _load_clip_object_metadata_map(cls, motion_dir: Path) -> dict[str, dict[str, str]]:
        candidate_files = (
            motion_dir / "_clip_object_urdf_map.json",
            motion_dir / "clip_object_urdf_map.json",
        )
        for path in candidate_files:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse clip-object metadata map '{}': {}", path, exc)
                return {}

            if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
                payload = payload["clips"]
            if not isinstance(payload, dict):
                logger.warning("Invalid clip-object metadata map format in '{}': expected dict.", path)
                return {}

            normalized: dict[str, dict[str, str]] = {}
            for clip_id, entry in payload.items():
                if not isinstance(clip_id, str):
                    continue
                if isinstance(entry, str):
                    normalized[clip_id] = {"object_name": "", "object_urdf_path": entry.strip()}
                elif isinstance(entry, dict):
                    obj_name = str(entry.get("object_name", "")).strip()
                    obj_urdf = str(entry.get("object_urdf_path", "")).strip()
                    normalized[clip_id] = {"object_name": obj_name, "object_urdf_path": obj_urdf}
            logger.info("Loaded clip-object metadata map '{}' ({} entries).", path, len(normalized))
            return normalized
        return {}

    @classmethod
    def _extract_object_clip_metadata(
        cls,
        *,
        data: Any,
        clip_id: str,
        clip_map: dict[str, dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        object_name = cls._scalar_str(data["object_name"]) if "object_name" in data else ""
        object_urdf_path = cls._scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""

        if clip_map is not None and clip_id in clip_map:
            mapped = clip_map[clip_id]
            if not object_name:
                object_name = mapped.get("object_name", "").strip()
            if not object_urdf_path:
                object_urdf_path = mapped.get("object_urdf_path", "").strip()

        if not object_name and object_urdf_path:
            object_name = Path(object_urdf_path).stem
        if not object_name:
            object_name = "object"
        return object_name, object_urdf_path

    @staticmethod
    def _resolve_motion_object_urdf_path(raw_path: str, *, base_dir: Path) -> str:
        path_str = str(raw_path).strip()
        if not path_str:
            return ""
        candidate = Path(path_str)
        if not candidate.is_absolute() and not path_str.startswith("holosoma/data"):
            candidate = (base_dir / path_str).resolve()
            return str(candidate)
        return str(Path(resolve_data_file_path(path_str)).resolve())

    @staticmethod
    def _format_motion_file_issues(issues: list[tuple[Path, str]], limit: int = 5) -> str:
        if not issues:
            return ""
        sample = "; ".join(f"{path}: {reason}" for path, reason in issues[:limit])
        remaining = len(issues) - min(len(issues), limit)
        if remaining > 0:
            sample = f"{sample}; +{remaining} more"
        return sample

    @staticmethod
    def _filter_valid_npz_archives(files: list[Path]) -> tuple[list[Path], list[tuple[Path, str]]]:
        valid_files: list[Path] = []
        invalid_files: list[tuple[Path, str]] = []
        for path in files:
            try:
                if zipfile.is_zipfile(path):
                    valid_files.append(path)
                else:
                    invalid_files.append((path, "not a valid .npz zip archive"))
            except OSError as exc:
                invalid_files.append((path, f"{type(exc).__name__}: {exc}"))
        return valid_files, invalid_files

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _resolve_body_subset_indexes(
        self,
        body_names_clip: list[str],
        *,
        source: str,
    ) -> tuple[list[str], np.ndarray]:
        """Select body indexes in clip order according to configured robot body names.

        This makes multi-clip loading robust when clips include extra scene bodies whose
        count varies per clip.
        """
        name_to_idx: dict[str, int] = {}
        duplicates: list[str] = []
        for idx, name in enumerate(body_names_clip):
            if name in name_to_idx:
                duplicates.append(name)
                continue
            name_to_idx[name] = idx
        if duplicates:
            dup_sorted = sorted(set(duplicates))
            raise ValueError(f"Duplicate body names in {source}: {dup_sorted}")

        missing = [name for name in self._robot_body_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Missing robot body names in {source}: {missing}")

        body_indexes = np.array([name_to_idx[name] for name in self._robot_body_names], dtype=np.int64)
        return list(self._robot_body_names), body_indexes

    def _set_clip_metadata(
        self,
        clip_ids: list[str],
        offsets: np.ndarray,
        lengths: np.ndarray,
        device: str,
        clip_object_names: list[str] | None = None,
        clip_object_urdf_paths: list[str] | None = None,
    ) -> None:
        self.clip_ids = clip_ids
        if clip_object_names is None:
            clip_object_names = [""] * len(clip_ids)
        if clip_object_urdf_paths is None:
            clip_object_urdf_paths = [""] * len(clip_ids)
        if len(clip_object_names) != len(clip_ids):
            raise ValueError("clip_object_names length must match clip_ids length")
        if len(clip_object_urdf_paths) != len(clip_ids):
            raise ValueError("clip_object_urdf_paths length must match clip_ids length")
        self.clip_object_names = clip_object_names
        self.clip_object_urdf_paths = clip_object_urdf_paths
        self.clip_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.num_clips = len(clip_ids)

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        clip_id = Path(motion_file).stem
        clip_object_names: list[str] | None = None
        clip_object_urdfs: list[str] | None = None
        try:
            with smart_open.open(motion_file, "rb") as f, np.load(f, allow_pickle=True) as data:
                self.fps = data["fps"]

                body_names = data["body_names"].tolist()
                joint_names = data["joint_names"].tolist()

                # The first 7 joints_pos are [xyz, wxyz] of the pelvis, omit them from the joint_pos
                # The first 6 joints_vel are [vel_xyz, vel_wxyz] of the pelvis, omit them from the joint_vel
                # We'll use the pelvis position and quaternion from body_pos_w[:, 0] and body_quat_w[:, 0] directly.
                self._joint_pos = torch.tensor(data["joint_pos"][:, 7:], dtype=torch.float32, device=device)
                self._joint_vel = torch.tensor(data["joint_vel"][:, 6:], dtype=torch.float32, device=device)
                assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

                self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
                assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

                # NOTE: wxyz after loading from npz
                body_quat_w_wxyz = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # This is wxyz
                self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

                self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
                self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

                # add object pos and quat
                self.has_object = "object_pos_w" in data
                if self.has_object:
                    length = int(self._joint_pos.shape[0])
                    # NOTE: wxyz after loading from npz
                    self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                    object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                    self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                    self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
                    object_size = self._extract_object_size_np(data, length, source=motion_file)
                    self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
                    obj_name, obj_urdf = self._extract_object_clip_metadata(data=data, clip_id=clip_id, clip_map=None)
                    clip_object_names = [obj_name]
                    clip_object_urdfs = [obj_urdf]
                else:
                    self._object_pos_w = torch.zeros(0, 3, device=device)
                    self._object_quat_w = torch.zeros(0, 4, device=device)
                    self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                    self._object_size = torch.zeros(0, 3, device=device)
        except (AssertionError, KeyError, zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
            raise zipfile.BadZipFile(f"Failed to load motion npz '{motion_file}': {exc}") from exc
        length = int(self._joint_pos.shape[0])
        self._set_clip_metadata(
            [clip_id],
            np.array([0]),
            np.array([length]),
            device,
            clip_object_names=clip_object_names,
            clip_object_urdf_paths=clip_object_urdfs,
        )
        return body_names, joint_names

    def _load_data_from_motion_npz_dir(
        self,
        motion_dir: Path,
        device: str,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> tuple[list[str], list[str]]:
        clip_object_map = self._load_clip_object_metadata_map(motion_dir)
        files = sorted(motion_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in motion directory: {motion_dir}")

        if motion_clip_name is not None:
            matches = [path for path in files if path.stem == motion_clip_name]
            if not matches:
                raise ValueError(f"Clip name '{motion_clip_name}' not found in {motion_dir}")
            files = matches
        elif motion_clip_id is not None:
            clip_idx = int(motion_clip_id)
            if clip_idx < 0 or clip_idx >= len(files):
                raise IndexError(f"Clip index {clip_idx} out of range for {motion_dir}")
            files = [files[clip_idx]]

        files, invalid_archives = self._filter_valid_npz_archives(files)
        if invalid_archives:
            issue_summary = self._format_motion_file_issues(invalid_archives)
            if not files:
                raise zipfile.BadZipFile(
                    f"No valid motion clips remain in {motion_dir}. Invalid clips: {issue_summary}"
                )
            logger.warning(
                "Skipping {} invalid motion clips in '{}'. Examples: {}",
                len(invalid_archives),
                motion_dir,
                issue_summary,
            )

        if len(files) == 1:
            body_names, joint_names = self._load_data_from_motion_npz(str(files[0]), device)
            clip_entry = clip_object_map.get(files[0].stem, {})
            if self.has_object and clip_entry:
                mapped_name = str(clip_entry.get("object_name", "")).strip()
                mapped_urdf = str(clip_entry.get("object_urdf_path", "")).strip()
                if self.clip_object_names:
                    if mapped_name and (not self.clip_object_names[0] or self.clip_object_names[0] == "object"):
                        self.clip_object_names[0] = mapped_name
                if self.clip_object_urdf_paths:
                    if mapped_urdf and not self.clip_object_urdf_paths[0]:
                        self.clip_object_urdf_paths[0] = mapped_urdf
            return body_names, joint_names

        required_keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
            "fps",
        )
        object_keys = ("object_pos_w", "object_quat_w", "object_lin_vel_w")

        joint_names: list[str] = []
        body_names: list[str] = []
        fps_ref: float | None = None
        has_object: bool | None = None

        clip_ids: list[str] = []
        offsets: list[int] = []
        lengths: list[int] = []
        offset = 0

        joint_pos_list: list[np.ndarray] = []
        joint_vel_list: list[np.ndarray] = []
        body_pos_list: list[np.ndarray] = []
        body_quat_list: list[np.ndarray] = []
        body_lin_vel_list: list[np.ndarray] = []
        body_ang_vel_list: list[np.ndarray] = []
        object_pos_list: list[np.ndarray] = []
        object_quat_list: list[np.ndarray] = []
        object_lin_vel_list: list[np.ndarray] = []
        object_size_list: list[np.ndarray] = []

        clip_object_names: list[str] = []
        clip_object_urdfs: list[str] = []
        late_load_failures: list[tuple[Path, str]] = []

        for file_path in files:
            try:
                data_file = np.load(file_path, allow_pickle=True)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                late_load_failures.append((file_path, f"{type(exc).__name__}: {exc}"))
                continue

            with data_file as data:
                try:
                    missing = [key for key in required_keys if key not in data]
                    if missing:
                        raise KeyError(f"Missing keys in {file_path}: {missing}")

                    clip_has_object = "object_pos_w" in data
                    if clip_has_object:
                        for key in object_keys:
                            if key not in data:
                                raise KeyError(f"Missing object key '{key}' in {file_path}")
                    if has_object is None:
                        has_object = clip_has_object
                    elif has_object != clip_has_object:
                        raise ValueError("Object fields are inconsistent across clips.")

                    joint_names_clip = self._decode_h5_strings(np.asarray(data["joint_names"]))
                    body_names_clip_raw = self._decode_h5_strings(np.asarray(data["body_names"]))
                    body_names_clip, body_indexes_clip = self._resolve_body_subset_indexes(
                        body_names_clip_raw,
                        source=str(file_path),
                    )
                    if not joint_names:
                        joint_names = joint_names_clip
                    elif joint_names_clip != joint_names:
                        raise ValueError(f"Joint names mismatch in {file_path}")
                    if not body_names:
                        body_names = body_names_clip
                    elif body_names_clip != body_names:
                        raise ValueError(f"Body names mismatch in {file_path}")

                    fps_arr = np.array(data["fps"]).reshape(-1)
                    fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0
                    if fps_ref is None:
                        fps_ref = fps
                    elif abs(fps_ref - fps) > 1e-6:
                        raise ValueError(f"FPS mismatch in {file_path}: {fps} != {fps_ref}")

                    joint_pos = np.asarray(data["joint_pos"])
                    length = int(joint_pos.shape[0])

                    clip_ids.append(file_path.stem)
                    offsets.append(offset)
                    lengths.append(length)
                    offset += length

                    joint_pos_list.append(joint_pos)
                    joint_vel_list.append(np.asarray(data["joint_vel"]))
                    body_pos = np.asarray(data["body_pos_w"])
                    body_quat = np.asarray(data["body_quat_w"])
                    body_lin_vel = np.asarray(data["body_lin_vel_w"])
                    body_ang_vel = np.asarray(data["body_ang_vel_w"])
                    expected_bodies = len(body_names_clip_raw)
                    for key, arr in (
                        ("body_pos_w", body_pos),
                        ("body_quat_w", body_quat),
                        ("body_lin_vel_w", body_lin_vel),
                        ("body_ang_vel_w", body_ang_vel),
                    ):
                        if arr.shape[1] != expected_bodies:
                            raise ValueError(
                                f"{key} body dimension mismatch in {file_path}: "
                                f"{arr.shape[1]} != {expected_bodies}"
                            )

                    body_pos_list.append(body_pos[:, body_indexes_clip])
                    body_quat_list.append(body_quat[:, body_indexes_clip])
                    body_lin_vel_list.append(body_lin_vel[:, body_indexes_clip])
                    body_ang_vel_list.append(body_ang_vel[:, body_indexes_clip])

                    if clip_has_object:
                        object_pos_list.append(np.asarray(data["object_pos_w"]))
                        object_quat_list.append(np.asarray(data["object_quat_w"]))
                        object_lin_vel_list.append(np.asarray(data["object_lin_vel_w"]))
                        object_size_list.append(
                            self._extract_object_size_np(data, length, source=str(file_path))
                        )
                        obj_name, obj_urdf = self._extract_object_clip_metadata(
                            data=data,
                            clip_id=file_path.stem,
                            clip_map=clip_object_map,
                        )
                        clip_object_names.append(obj_name)
                        clip_object_urdfs.append(obj_urdf)
                    else:
                        clip_object_names.append("")
                        clip_object_urdfs.append("")
                except (AssertionError, KeyError, ValueError) as exc:
                    late_load_failures.append((file_path, f"{type(exc).__name__}: {exc}"))
                    continue

        if late_load_failures:
            issue_summary = self._format_motion_file_issues(late_load_failures)
            if not clip_ids:
                raise zipfile.BadZipFile(
                    f"Failed to load any motion clips from {motion_dir}. Examples: {issue_summary}"
                )
            logger.warning(
                "Skipped {} motion clips that failed to open in '{}'. Examples: {}",
                len(late_load_failures),
                motion_dir,
                issue_summary,
            )

        self.fps = float(fps_ref) if fps_ref is not None else 30.0
        self._set_clip_metadata(
            clip_ids,
            np.array(offsets),
            np.array(lengths),
            device,
            clip_object_names=clip_object_names,
            clip_object_urdf_paths=clip_object_urdfs,
        )

        joint_pos = np.concatenate(joint_pos_list, axis=0)
        joint_vel = np.concatenate(joint_vel_list, axis=0)
        body_pos_w = np.concatenate(body_pos_list, axis=0)
        body_quat_w = np.concatenate(body_quat_list, axis=0)
        body_lin_vel_w = np.concatenate(body_lin_vel_list, axis=0)
        body_ang_vel_w = np.concatenate(body_ang_vel_list, axis=0)

        self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
        self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
        assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

        self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

        body_quat_w_wxyz = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]

        self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

        self.has_object = bool(has_object)
        if self.has_object:
            object_pos_w = np.concatenate(object_pos_list, axis=0)
            object_quat_w = np.concatenate(object_quat_list, axis=0)
            object_lin_vel_w = np.concatenate(object_lin_vel_list, axis=0)
            object_size = np.concatenate(object_size_list, axis=0)

            self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
            object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
            self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
            self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
            self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)
            self._object_size = torch.zeros(0, 3, device=device)

        return body_names, joint_names

    @staticmethod
    def _decode_h5_strings(values: np.ndarray) -> list[str]:
        decoded: list[str] = []
        for item in values:
            if isinstance(item, (bytes, np.bytes_)):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    @staticmethod
    def _finite_diff(data: np.ndarray, fps: float) -> np.ndarray:
        if data.shape[0] == 1:
            return np.zeros_like(data)
        vel = (data[1:] - data[:-1]) * fps
        return np.concatenate([vel, vel[-1:]], axis=0)

    @staticmethod
    def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
        out = q.copy()
        out[..., :3] *= -1.0
        return out

    @staticmethod
    def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = np.split(a, 4, axis=-1)
        bx, by, bz, bw = np.split(b, 4, axis=-1)
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        return np.concatenate([x, y, z, w], axis=-1)

    @staticmethod
    def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qvec = q[..., :3]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (q[..., 3:4] * uv + uuv)

    @staticmethod
    def _angular_velocity_xyzw(quats: np.ndarray, fps: float) -> np.ndarray:
        if quats.shape[0] == 1:
            return np.zeros(quats.shape[:-1] + (3,), dtype=quats.dtype)
        q0 = quats[:-1]
        q1 = quats[1:]
        dq = MotionLoader._quat_mul_xyzw(q1, MotionLoader._quat_conjugate_xyzw(q0))
        dq = dq / np.linalg.norm(dq, axis=-1, keepdims=True)
        w = np.clip(dq[..., 3], -1.0, 1.0)
        v = dq[..., :3]
        sin_half = np.linalg.norm(v, axis=-1)
        angle = 2.0 * np.arctan2(sin_half, w)
        small = sin_half < 1e-8
        axis = np.zeros_like(v)
        axis[~small] = v[~small] / sin_half[~small][..., None]
        omega = axis * (angle[..., None] * fps)
        omega[small] = 2.0 * v[small] * fps
        return np.concatenate([omega, omega[-1:]], axis=0)

    @staticmethod
    def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
        return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)

    @staticmethod
    def _infer_link_frame(link_names: list[str], link_pos: np.ndarray, root_pos: np.ndarray) -> str:
        for pelvis_name in ("pelvis", "pelvis_link"):
            if pelvis_name in link_names:
                idx = link_names.index(pelvis_name)
                diff = np.linalg.norm(link_pos[:, idx] - root_pos, axis=-1)
                if np.median(diff) < 1e-3:
                    return "world"
                return "local"
        return "world"

    @staticmethod
    def _normalize_link_name(name: str) -> str:
        if name.endswith(".STL") or name.endswith(".stl"):
            return name[:-4]
        return name

    def _get_h5_attr_or_dataset(self, h5f: Any, name: str) -> np.ndarray | None:
        if name in h5f.attrs:
            return np.asarray(h5f.attrs[name])
        if f"/{name}" in h5f.attrs:
            return np.asarray(h5f.attrs[f"/{name}"])
        if name in h5f:
            return np.asarray(h5f[name])
        if f"/{name}" in h5f:
            return np.asarray(h5f[f"/{name}"])
        return None

    def _resolve_h5_clip_metadata_values(
        self,
        h5f: Any,
        *,
        clip_ids: list[str],
        selected_clip_indices: list[int],
        field_names: tuple[str, ...],
    ) -> list[str]:
        containers = []
        if "clips" in h5f:
            containers.append(h5f["clips"])
        if "meta" in h5f:
            containers.append(h5f["meta"])
        containers.append(h5f)

        raw_values = None
        for container in containers:
            for field_name in field_names:
                raw_values = self._get_h5_attr_or_dataset(container, field_name)
                if raw_values is not None:
                    break
            if raw_values is not None:
                break

        if raw_values is not None:
            arr = np.asarray(raw_values)
            if arr.shape == ():
                return [self._scalar_str(arr)] * len(selected_clip_indices)
            flat = arr.reshape(-1)
            if flat.shape[0] >= max(selected_clip_indices, default=0) + 1:
                return [self._scalar_str(flat[idx]) for idx in selected_clip_indices]

        clips_group = h5f["clips"] if "clips" in h5f else None
        if clips_group is not None:
            nested_values: list[str] = []
            for clip_idx in selected_clip_indices:
                clip_id = clip_ids[clip_idx]
                clip_group = clips_group.get(clip_id, None)
                if clip_group is None:
                    return []
                clip_value = None
                for field_name in field_names:
                    clip_value = self._get_h5_attr_or_dataset(clip_group, field_name)
                    if clip_value is not None:
                        break
                if clip_value is None:
                    return []
                nested_values.append(self._scalar_str(clip_value))
            return nested_values

        return []

    def _resolve_h5_clip_object_metadata(
        self,
        h5f: Any,
        *,
        motion_file: str,
        clip_ids: list[str],
        selected_clip_indices: list[int],
    ) -> tuple[list[str], list[str]]:
        raw_object_names = self._resolve_h5_clip_metadata_values(
            h5f,
            clip_ids=clip_ids,
            selected_clip_indices=selected_clip_indices,
            field_names=("object_name", "object_names"),
        )
        raw_object_urdfs = self._resolve_h5_clip_metadata_values(
            h5f,
            clip_ids=clip_ids,
            selected_clip_indices=selected_clip_indices,
            field_names=("object_urdf_path", "object_urdf_paths"),
        )

        base_dir = Path(motion_file).parent
        clip_object_names: list[str] = []
        clip_object_urdfs: list[str] = []
        for local_idx, clip_idx in enumerate(selected_clip_indices):
            clip_name = raw_object_names[local_idx].strip() if local_idx < len(raw_object_names) else ""
            clip_urdf = raw_object_urdfs[local_idx].strip() if local_idx < len(raw_object_urdfs) else ""
            if clip_urdf:
                clip_urdf = self._resolve_motion_object_urdf_path(clip_urdf, base_dir=base_dir)
            if not clip_name and clip_urdf:
                clip_name = Path(clip_urdf).stem
            if not clip_name:
                clip_name = "object"
            clip_object_names.append(clip_name)
            clip_object_urdfs.append(clip_urdf)
        return clip_object_names, clip_object_urdfs

    def _load_data_from_motion_h5(
        self,
        motion_file: str,
        device: str,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> tuple[list[str], list[str]]:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("h5py is required to load HDF5 motion files.") from exc

        with h5py.File(motion_file, "r") as h5f:
            if "meta" not in h5f or "data" not in h5f:
                return self._load_data_from_motion_h5_videomimic(h5f, motion_file, device)

            meta = h5f["meta"]
            data = h5f["data"]

            joint_names = self._decode_h5_strings(np.asarray(meta["joint_names"]))
            body_names = self._decode_h5_strings(np.asarray(meta["body_names"]))

            clips = h5f["clips"] if "clips" in h5f else None
            clip_ids: list[str] = []
            offsets = None
            lengths = None
            clip_fps = None
            selected_clip_idx: int | None = None
            selected_clip_indices: list[int] = [0]
            if clips is not None:
                clip_ids = self._decode_h5_strings(np.asarray(clips["clip_ids"]))
                offsets = np.asarray(clips["offsets"], dtype=np.int64)
                lengths = np.asarray(clips["lengths"], dtype=np.int64)
                if "clip_fps" in clips:
                    clip_fps = np.asarray(clips["clip_fps"], dtype=np.float32)

            load_all = motion_clip_id is None and motion_clip_name is None
            if clips is None:
                if not load_all:
                    raise ValueError("motion_clip_id/name provided but HDF5 motion file has no /clips group.")
                start = 0
                length = int(data["joint_pos"].shape[0])
                fps_val = np.asarray(meta["fps"])
                clip_id = Path(motion_file).stem
                clip_ids = [clip_id]
                selected_clip_indices = [0]
            elif load_all:
                start = 0
                length = int(data["joint_pos"].shape[0])
                fps_val = np.asarray(meta["fps"])
                if clip_fps is not None:
                    if not np.allclose(clip_fps, float(np.array(fps_val).reshape(-1)[0])):
                            raise ValueError("clip_fps must be consistent across clips for multi-clip loading.")
                assert offsets is not None and lengths is not None
                selected_clip_indices = list(range(len(clip_ids)))
            else:
                if motion_clip_name is not None:
                    if motion_clip_name not in clip_ids:
                        raise ValueError(f"Clip name '{motion_clip_name}' not found in HDF5 motion file.")
                    clip_idx = clip_ids.index(motion_clip_name)
                else:
                    clip_idx = int(motion_clip_id)

                assert offsets is not None and lengths is not None
                if clip_idx < 0 or clip_idx >= len(lengths):
                    raise IndexError(f"Clip index {clip_idx} out of range for HDF5 motion file.")
                selected_clip_idx = clip_idx
                selected_clip_indices = [clip_idx]
                start = int(offsets[clip_idx])
                length = int(lengths[clip_idx])
                fps_val = clip_fps[clip_idx] if clip_fps is not None else np.asarray(meta["fps"])

            clip_object_names, clip_object_urdfs = self._resolve_h5_clip_object_metadata(
                h5f,
                motion_file=motion_file,
                clip_ids=clip_ids,
                selected_clip_indices=selected_clip_indices,
            )
            if clips is None:
                self._set_clip_metadata(
                    clip_ids,
                    np.array([0]),
                    np.array([length]),
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )
            elif load_all:
                assert offsets is not None and lengths is not None
                self._set_clip_metadata(
                    clip_ids,
                    offsets,
                    lengths,
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )
            else:
                self._set_clip_metadata(
                    [clip_ids[selected_clip_idx]],
                    np.array([0]),
                    np.array([length]),
                    device,
                    clip_object_names=clip_object_names,
                    clip_object_urdf_paths=clip_object_urdfs,
                )

            fps_arr = np.array(fps_val).reshape(-1)
            self.fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0

            end = start + length
            joint_pos = np.asarray(data["joint_pos"][start:end])
            joint_vel = np.asarray(data["joint_vel"][start:end])
            body_pos_w = np.asarray(data["body_pos_w"][start:end])
            body_quat_w = np.asarray(data["body_quat_w"][start:end])
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"][start:end])
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"][start:end])

            self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
            assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

            self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
            assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

            body_quat_w_wxyz = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

            self.has_object = "object_pos_w" in data
            if self.has_object:
                object_pos_w = np.asarray(data["object_pos_w"][start:end])
                object_quat_w = np.asarray(data["object_quat_w"][start:end])
                object_lin_vel_w = np.asarray(data["object_lin_vel_w"][start:end])
                object_size = None
                for key in self._OBJECT_SIZE_KEYS:
                    if key not in data:
                        continue
                    raw_size = np.asarray(data[key], dtype=np.float32)
                    # Support clip-wise object size annotations: shape (num_clips, 3) or (num_clips,).
                    if (
                        clips is not None
                        and lengths is not None
                        and raw_size.ndim in (1, 2)
                        and raw_size.shape[0] == len(lengths)
                    ):
                        if selected_clip_idx is not None:
                            raw_size = raw_size[selected_clip_idx]
                            object_size = self._normalize_object_size_array(
                                raw_size, length, source=f"{motion_file}:{key}"
                            )
                            break
                        if load_all:
                            per_clip_sizes = []
                            for clip_i, clip_len in enumerate(lengths):
                                clip_size = self._normalize_object_size_array(
                                    raw_size[clip_i], int(clip_len), source=f"{motion_file}:{key}"
                                )
                                per_clip_sizes.append(clip_size)
                            object_size = np.concatenate(per_clip_sizes, axis=0)
                            break
                    # Most common format stores size per frame for the full bank.
                    if raw_size.ndim >= 1 and raw_size.shape[0] >= end:
                        raw_size = raw_size[start:end]
                    object_size = self._normalize_object_size_array(
                        raw_size, length, source=f"{motion_file}:{key}"
                    )
                    break
                if object_size is None:
                    object_size = np.ones((length, 3), dtype=np.float32)

                self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
                self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
                self._object_size = torch.tensor(object_size, dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                self._object_size = torch.zeros(0, 3, device=device)

        return body_names, joint_names

    def _load_data_from_motion_h5_videomimic(
        self,
        h5f: Any,
        motion_file: str,
        device: str,
    ) -> tuple[list[str], list[str]]:
        required = ("root_pos", "root_quat", "joints", "link_pos", "link_quat")
        missing = [key for key in required if key not in h5f]
        if missing:
            raise KeyError(f"Missing keys in VideoMimic HDF5 file: {missing}")

        root_pos = np.asarray(h5f["root_pos"], dtype=np.float32)
        root_quat_xyzw = np.asarray(h5f["root_quat"], dtype=np.float32)
        joints = np.asarray(h5f["joints"], dtype=np.float32)
        link_pos = np.asarray(h5f["link_pos"], dtype=np.float32)
        link_quat_xyzw = np.asarray(h5f["link_quat"], dtype=np.float32)

        joint_names_raw = self._get_h5_attr_or_dataset(h5f, "joint_names")
        link_names_raw = self._get_h5_attr_or_dataset(h5f, "link_names")
        if joint_names_raw is None or link_names_raw is None:
            raise ValueError("VideoMimic HDF5 file must provide joint_names and link_names.")
        joint_names = self._decode_h5_strings(np.asarray(joint_names_raw))
        link_names = self._decode_h5_strings(np.asarray(link_names_raw))
        link_names = [self._normalize_link_name(name) for name in link_names]

        fps_raw = self._get_h5_attr_or_dataset(h5f, "fps")
        fps_arr = np.array(fps_raw).reshape(-1) if fps_raw is not None else np.array([30.0], dtype=np.float32)
        self.fps = float(fps_arr[0]) if fps_arr.size > 0 else 30.0

        num_frames = int(root_pos.shape[0])
        if joints.shape[0] != num_frames:
            raise ValueError("VideoMimic HDF5 joint length does not match root_pos length.")

        if self._robot_joint_names:
            missing_joints = [name for name in self._robot_joint_names if name not in joint_names]
            if missing_joints:
                zeros = np.zeros((num_frames, len(missing_joints)), dtype=joints.dtype)
                joints = np.concatenate([joints, zeros], axis=1)
                joint_names.extend(missing_joints)
                logger.warning("Missing joints in VideoMimic HDF5, padded with zeros: {}", missing_joints)

        # VideoMimic uses link_pos/link_quat in the env/world frame. Keep them as-is.
        link_pos_w = link_pos
        link_quat_w = link_quat_xyzw

        body_names = list(self._robot_body_names)
        num_bodies = len(body_names)
        body_pos_w = np.broadcast_to(root_pos[:, None, :], (num_frames, num_bodies, 3)).copy()
        body_quat_w = np.broadcast_to(root_quat_xyzw[:, None, :], (num_frames, num_bodies, 4)).copy()

        link_name_map = {name: i for i, name in enumerate(link_names)}
        for body_idx, body_name in enumerate(body_names):
            link_idx = link_name_map.get(body_name)
            if link_idx is None:
                continue
            body_pos_w[:, body_idx] = link_pos_w[:, link_idx]
            body_quat_w[:, body_idx] = link_quat_w[:, link_idx]

        body_lin_vel_w = self._finite_diff(body_pos_w, self.fps)
        body_ang_vel_w = self._angular_velocity_xyzw(body_quat_w, self.fps)

        root_lin_vel = self._finite_diff(root_pos, self.fps)
        root_ang_vel = self._angular_velocity_xyzw(root_quat_xyzw, self.fps)
        dof_vel = self._finite_diff(joints, self.fps)

        joint_pos = np.concatenate([root_pos, self._xyzw_to_wxyz(root_quat_xyzw), joints], axis=-1)
        joint_vel = np.concatenate([root_lin_vel, root_ang_vel, dof_vel], axis=-1)

        self._joint_pos = torch.tensor(joint_pos[:, 7:], dtype=torch.float32, device=device)
        self._joint_vel = torch.tensor(joint_vel[:, 6:], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

        self.has_object = False
        self._object_pos_w = torch.zeros(0, 3, device=device)
        self._object_quat_w = torch.zeros(0, 4, device=device)
        self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        self._object_size = torch.zeros(0, 3, device=device)

        clip_id = Path(motion_file).stem
        self._set_clip_metadata([clip_id], np.array([0]), np.array([num_frames]), device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_size(self) -> torch.Tensor:
        return self._object_size[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_size", "_object_size"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        if self.num_clips == 1:
            device = self.clip_lengths.device if self.clip_lengths.numel() > 0 else self._joint_pos.device
            self.clip_lengths = torch.tensor([self.time_step_total], dtype=torch.long, device=device)
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int | None,
        device: str,
        env_fps: int,
        clip_lengths: torch.Tensor | None = None,
        bin_size_s: float = 1.0,
        kernel_size: int = 3,
        decay_lambda: float = 0.001,
        kernel_lambda: float = 0.8,
    ):
        # TODO: think better about the decay_lambda, will 0.001 be too small?
        self.device = device
        # fps of the rl environment
        self.env_fps = env_fps

        if clip_lengths is not None:
            clip_lengths = torch.as_tensor(clip_lengths, dtype=torch.long, device=self.device).reshape(-1)
            if clip_lengths.numel() == 0:
                raise ValueError("clip_lengths must contain at least one clip.")
            self.clip_lengths = torch.clamp(clip_lengths, min=1)
        else:
            if motion_time_step_total is None:
                raise ValueError("motion_time_step_total must be provided when clip_lengths is None.")
            total_steps = max(int(motion_time_step_total), 1)
            self.clip_lengths = torch.tensor([total_steps], dtype=torch.long, device=self.device)

        self.num_clips = int(self.clip_lengths.numel())
        # Keep the longest clip length for backwards-compatible stats/debugging.
        self.motion_time_step_total = int(self.clip_lengths.max().item())

        # size of the bin in seconds
        self.bin_size_s = bin_size_s
        # size of the kernel for smoothing the sampling probabilities
        self.kernel_size = kernel_size
        self.kernel_lambda = kernel_lambda
        # exponential decay when updating the failure counts over training steps.

        self.decay_lambda = decay_lambda

        clip_duration_s = self.clip_lengths.to(dtype=torch.float32) / float(self.env_fps)
        self.num_bins_per_clip = torch.clamp(torch.ceil(clip_duration_s / self.bin_size_s).long(), min=1)
        self.max_num_bins = int(self.num_bins_per_clip.max().item())
        # Maintain the old attribute for single-clip callers and debug metrics.
        self.num_bins = self.max_num_bins

        # initialize exponential 1d decay kernel, used for smoothing the failure counts over time.
        assert self.kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel = torch.tensor(
            [self.kernel_lambda ** abs(i) for i in range((-self.kernel_size + 1) // 2, (self.kernel_size + 1) // 2)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        shape = (self.num_clips, self.max_num_bins)
        self.current_bin_failed_count = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.bin_failed_count = torch.zeros(shape, dtype=torch.float32, device=self.device)

    def _resolve_clip_ids(self, clip_ids: torch.Tensor | None, count: int) -> torch.Tensor:
        if clip_ids is None:
            if self.num_clips != 1:
                raise ValueError("clip_ids must be provided for multi-clip adaptive timestep sampling.")
            return torch.zeros((count,), dtype=torch.long, device=self.device)
        clip_ids = torch.as_tensor(clip_ids, dtype=torch.long, device=self.device).reshape(-1)
        if clip_ids.numel() != count:
            raise ValueError(f"Expected {count} clip ids, got {clip_ids.numel()}.")
        if torch.any(clip_ids < 0) or torch.any(clip_ids >= self.num_clips):
            raise ValueError(f"clip_ids must be in [0, {self.num_clips}).")
        return clip_ids

    def _sampling_probabilities_for_clip(self, clip_id: int) -> torch.Tensor:
        valid_bins = int(self.num_bins_per_clip[clip_id].item())
        sampling_probabilities = self.bin_failed_count[clip_id, :valid_bins] + 1e-6
        sampling_probabilities = F.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = F.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities += 0.01
        return sampling_probabilities / sampling_probabilities.sum()

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor, clip_ids: torch.Tensor | None = None):
        """Update the current bin failed count with terminated time steps."""
        failed_at_time_step = torch.as_tensor(failed_at_time_step, dtype=torch.float32, device=self.device).reshape(-1)
        if failed_at_time_step.numel() == 0:
            self.current_bin_failed_count.zero_()
            return

        clip_ids = self._resolve_clip_ids(clip_ids, failed_at_time_step.numel())
        clip_lengths = self.clip_lengths[clip_ids].to(dtype=torch.float32)
        num_bins = self.num_bins_per_clip[clip_ids]
        failed_bin = torch.floor(failed_at_time_step / clip_lengths * num_bins.to(dtype=torch.float32)).long()
        failed_bin = torch.clamp(failed_bin, min=0)
        failed_bin = torch.minimum(failed_bin, num_bins - 1)
        flat_ids = clip_ids * self.max_num_bins + failed_bin
        counts = torch.bincount(flat_ids, minlength=self.num_clips * self.max_num_bins).to(dtype=torch.float32)
        self.current_bin_failed_count[:] = counts.view(self.num_clips, self.max_num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.decay_lambda * self.current_bin_failed_count) + (
            1 - self.decay_lambda
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        if self.num_clips != 1:
            raise RuntimeError("sampling_probabilities is only defined for single-clip adaptive timestep sampling.")
        return self._sampling_probabilities_for_clip(0)

    def sample(self, clip_ids_or_num_samples: torch.Tensor | int) -> torch.Tensor:
        if isinstance(clip_ids_or_num_samples, int):
            clip_ids = self._resolve_clip_ids(None, clip_ids_or_num_samples)
        else:
            clip_ids = self._resolve_clip_ids(clip_ids_or_num_samples, int(clip_ids_or_num_samples.numel()))

        phases = torch.zeros((clip_ids.numel(),), dtype=torch.float32, device=self.device)
        if clip_ids.numel() == 0:
            return phases

        unique_clip_ids, inverse = torch.unique(clip_ids, return_inverse=True)
        for local_idx, clip_id_tensor in enumerate(unique_clip_ids):
            clip_id = int(clip_id_tensor.item())
            env_mask = inverse == local_idx
            num_samples = int(env_mask.sum().item())
            sampled_bins = torch.multinomial(self._sampling_probabilities_for_clip(clip_id), num_samples, replacement=True)
            num_bins = float(self.num_bins_per_clip[clip_id].item())
            phases[env_mask] = (sampled_bins.to(dtype=torch.float32) + torch.rand(num_samples, device=self.device)) / num_bins
        return phases

    def get_stats(self):
        # Metrics
        entropies: list[torch.Tensor] = []
        top1_probs: list[torch.Tensor] = []
        top1_bins: list[torch.Tensor] = []
        for clip_id in range(self.num_clips):
            prob = self._sampling_probabilities_for_clip(clip_id)
            if prob.numel() <= 1:
                entropies.append(torch.zeros((), device=self.device, dtype=torch.float32))
                top1_probs.append(torch.ones((), device=self.device, dtype=torch.float32))
                top1_bins.append(torch.zeros((), device=self.device, dtype=torch.float32))
                continue
            H = -(prob * (prob + 1e-12).log()).sum()
            H_norm = H / np.log(float(prob.numel()))
            pmax, imax = prob.max(dim=0)
            entropies.append(H_norm.to(dtype=torch.float32))
            top1_probs.append(pmax.to(dtype=torch.float32))
            top1_bins.append(imax.to(dtype=torch.float32) / float(prob.numel()))

        self.metrics["sampling_entropy"] = torch.stack(entropies).mean()
        self.metrics["sampling_top1_prob"] = torch.stack(top1_probs).mean()
        self.metrics["sampling_top1_bin"] = torch.stack(top1_bins).mean()


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose
        self._clip_terrain_offsets: torch.Tensor | None = None
        self._clip_terrain_offsets_by_row: torch.Tensor | None = None
        self._terrain_row_ids: torch.Tensor | None = None
        self._terrain_row_stride: float = 0.0
        self._terrain_row_count: int = 0
        self._forced_clip_idx: int | None = None
        self.manual_control_enabled = False
        self.manual_xy_rel: torch.Tensor | None = None
        self.manual_yaw_rel: torch.Tensor | None = None
        self.manual_object_reset_enabled = False
        self.manual_object_reset_pos_offset_w: torch.Tensor | None = None
        self.manual_object_reset_rpy_offset: torch.Tensor | None = None
        self.manual_goal_enabled = False
        self.manual_goal_object_pos_w: torch.Tensor | None = None
        self.manual_goal_object_rot6d_w: torch.Tensor | None = None
        self.manual_goal_override_enabled = False
        self.manual_goal_xy_rel: torch.Tensor | None = None
        self.manual_goal_yaw_rel: torch.Tensor | None = None
        self.base_goal_object_pos_w: torch.Tensor | None = None
        self.base_goal_object_rot6d_w: torch.Tensor | None = None
        self.base_goal_is_external: torch.Tensor | None = None
        self.manual_goal_is_external: torch.Tensor | None = None
        self.clip_goal_object_pos_w: torch.Tensor | None = None
        self.clip_goal_object_rot6d_w: torch.Tensor | None = None
        self._sparse_goal_cfg: SparseObjectGoalConfig | None = None
        self._sparse_goal_curriculum_enabled = False
        self._sparse_goal_reset_counter = 0
        self._command_only_env_prob = 0.0
        self._command_only_env_fraction_last_reset = 0.0
        self._sparse_goal_external_prob = 0.0
        self._sparse_goal_external_fraction_last_reset = 0.0
        self.command_only_env_mask: torch.Tensor | None = None
        self._training_iteration: int | None = None
        self._training_total_iterations: int | None = None
        self._clean_noisy_clip_curriculum_cfg: CleanNoisyClipCurriculumConfig | None = None
        self._clean_noisy_clip_curriculum_enabled = False
        self._clean_clip_mask: torch.Tensor | None = None
        self._noisy_clip_mask: torch.Tensor | None = None
        self.pickup_anchor_set: torch.Tensor | None = None
        self.pickup_anchor_root_pos_w: torch.Tensor | None = None
        self.pickup_anchor_root_quat_w: torch.Tensor | None = None
        self.pickup_object_rel_z_baseline: torch.Tensor | None = None
        self.pickup_consecutive_counter: torch.Tensor | None = None
        self._multi_object_enabled = False
        self._sim_object_names: list[str] = []
        self._clip_object_ids: torch.Tensor | None = None
        self._object_indices_matrix: torch.Tensor | None = None
        self._fixed_clip_ids: torch.Tensor | None = None
        self.object_name: str = "object"
        self.object_indices_in_simulator: torch.Tensor | None = None
        self._debug_representative_clip_ids: torch.Tensor | None = None
        self._contact_prior_available = False
        self._contact_prior_force_body_names_by_region: dict[str, list[str]] = {}
        self._contact_prior_position_body_names_by_region: dict[str, list[str]] = {}
        self._contact_prior_position_body_indices_by_region: dict[str, torch.Tensor] = {}
        self._contact_prior_total_count: torch.Tensor | None = None
        self._contact_prior_contact_sum: torch.Tensor | None = None
        self._contact_prior_force_mean: torch.Tensor | None = None
        self._contact_prior_force_count: torch.Tensor | None = None
        self._contact_prior_position_mean: torch.Tensor | None = None
        self._contact_prior_position_count: torch.Tensor | None = None
        self._object_contact_body_indices_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._runtime_default_pose_prepend_enabled = False
        self._runtime_default_pose_prepend_steps = 0
        self._runtime_default_pose_prepend_active: torch.Tensor | None = None
        self._runtime_default_pose_prepend_step: torch.Tensor | None = None
        self._runtime_default_pose_prepend_defaults: dict[str, torch.Tensor] = {}

    def set_forced_clip(self, clip_idx: int | None) -> None:
        """Force a specific clip index for resets (None clears the override)."""
        if clip_idx is None:
            self._forced_clip_idx = None
            return
        if clip_idx < 0 or clip_idx >= self.motion.num_clips:
            raise ValueError(f"clip_idx {clip_idx} out of range for {self.motion.num_clips} clips.")
        self._forced_clip_idx = int(clip_idx)

    def set_training_iteration(self, iteration: int, *, total_iterations: int | None = None) -> None:
        """Expose the current PPO iteration so command curriculum can follow the training schedule exactly."""
        self._training_iteration = int(iteration)
        self._training_total_iterations = None if total_iterations is None else int(total_iterations)
        self._refresh_current_clip_sampling_weights()

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device
        self.manual_control_enabled = False
        self.manual_xy_rel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.manual_yaw_rel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.manual_object_reset_enabled = False
        self.manual_object_reset_pos_offset_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.manual_object_reset_rpy_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.manual_goal_enabled = False
        self.manual_goal_object_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        self.manual_goal_object_rot6d_w = identity_rot6d.unsqueeze(0).repeat(self.num_envs, 1)
        self.manual_goal_override_enabled = False
        self.manual_goal_xy_rel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self.manual_goal_yaw_rel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.base_goal_object_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.base_goal_object_rot6d_w = identity_rot6d.unsqueeze(0).repeat(self.num_envs, 1)
        self.base_goal_is_external = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.manual_goal_is_external = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.clip_goal_object_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.clip_goal_object_rot6d_w = identity_rot6d.unsqueeze(0).repeat(self.num_envs, 1)
        self._sparse_goal_cfg = self.motion_cfg.sparse_object_goal
        self._sparse_goal_curriculum_enabled = bool(self._sparse_goal_cfg.enabled)
        self._sparse_goal_reset_counter = 0
        self._command_only_env_prob = 0.0
        self._command_only_env_fraction_last_reset = 0.0
        self._sparse_goal_external_prob = 0.0
        self._sparse_goal_external_fraction_last_reset = 0.0
        self.command_only_env_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._training_iteration = 0
        self._training_total_iterations = None
        self._clean_noisy_clip_curriculum_cfg = self.motion_cfg.clean_noisy_clip_curriculum
        self._clean_noisy_clip_curriculum_enabled = bool(self._clean_noisy_clip_curriculum_cfg.enabled)
        self._clean_clip_mask = None
        self._noisy_clip_mask = None
        self.pickup_anchor_set = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.pickup_anchor_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.pickup_anchor_root_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.pickup_anchor_root_quat_w[:, 3] = 1.0
        self.pickup_object_rel_z_baseline = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.pickup_consecutive_counter = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        init_state = self._env.robot_config.init_state
        reset_to_default_pose_env = os.environ.get("HOLOSOMA_RESET_TO_DEFAULT_POSE")
        if reset_to_default_pose_env is None:
            reset_to_default_pose_env = os.environ.get("HOLOSOMA_DEFAULT_POSE_INIT", "0")
        self._reset_to_default_pose = reset_to_default_pose_env.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if self._reset_to_default_pose:
            start_probs = [float(self.motion_cfg.start_at_timestep_zero_prob)]
            if self.motion_cfg.start_at_timestep_zero_prob_end is not None:
                start_probs.append(float(self.motion_cfg.start_at_timestep_zero_prob_end))
            if any(prob < 0.999 for prob in start_probs):
                logger.warning(
                    "reset_to_default_pose=True applies to every reset, including non-zero motion starts. "
                    "This can make random clip starts much harder than runtime prepend alone."
                )
        self._init_root_pos = torch.tensor(init_state.pos, dtype=torch.float32, device=self.device)
        self._init_root_rot = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device)
        self._init_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        self._init_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)
        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, init_yaw = get_euler_xyz(init_root_quat, w_last=True)
        self._init_root_yaw = init_yaw.squeeze(0)

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        self.motion: MotionLoader = MotionLoader(
            self.motion_cfg.motion_file,
            robot_body_names_alias,
            robot_joint_names,
            device=self.device,
            motion_clip_id=self.motion_cfg.motion_clip_id,
            motion_clip_name=self.motion_cfg.motion_clip_name,
            object_size_scale=self.motion_cfg.object_size_scale,
        )
        self.multi_clip = self.motion.num_clips > 1
        if self.multi_clip:
            logger.info("Multi-clip motion bank detected ({} clips).", self.motion.num_clips)

        self._configure_motion_terrain_pairs()

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            simulator_type = self._env.simulator.get_simulator_type()
            assert simulator_type in {SimulatorType.ISAACSIM, SimulatorType.MUJOCO}, (
                f"Object carry motions currently support IsaacSim or MuJoCo, got {simulator_type}."
            )
            self._configure_simulator_object_mapping()
            self._configure_fixed_env_clip_assignment()
            self._configure_debug_representative_clips()
        elif self._sparse_goal_curriculum_enabled:
            logger.warning("Sparse object-goal curriculum requested but motion has no object; disabling curriculum.")
            self._sparse_goal_curriculum_enabled = False
            self.object_indices_in_simulator = None
        self._configure_runtime_default_pose_prepend()
        self._configure_contact_prior_regions()

        # 4. get the adaptive timesteps sampler
        self.use_adaptive_timesteps_sampler = self.motion_cfg.use_adaptive_timesteps_sampler
        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total,
                self.device,
                int(1 / (self._env.dt)),
                clip_lengths=self.motion.clip_lengths,
            )
            if self.multi_clip:
                logger.info(
                    "Per-clip adaptive timestep sampling enabled for multi-clip motion bank ({} clips).",
                    self.motion.num_clips,
                )

        # 5. clip sampling configuration
        self.clip_weighting_strategy = self.motion_cfg.clip_weighting_strategy
        self.min_weight_factor = self.motion_cfg.min_weight_factor
        self.max_weight_factor = self.motion_cfg.max_weight_factor
        self._clip_sampling_weights: torch.Tensor | None = None
        self._raw_clip_sampling_weights: torch.Tensor | None = None
        self._base_clip_weights: torch.Tensor | None = None
        self._clip_success_counts: torch.Tensor | None = None
        self._clip_total_counts: torch.Tensor | None = None

        # 6. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self._configure_target_pose_settings()
        self._init_clip_sampling()
        self.init_buffers()

        # 7. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    @staticmethod
    def _normalize_path_key(path: str) -> str:
        if not path:
            return ""
        try:
            return str(Path(path).resolve())
        except Exception:
            return path

    def _resolve_sim_object_name(
        self,
        *,
        clip_id: str,
        clip_object_name: str,
        clip_object_urdf: str,
        sim_names: list[str],
        sim_name_by_urdf: dict[str, str],
        sim_name_by_stem: dict[str, str],
    ) -> str:
        normalized_urdf = self._normalize_path_key(clip_object_urdf)
        if normalized_urdf and normalized_urdf in sim_name_by_urdf:
            return sim_name_by_urdf[normalized_urdf]

        if normalized_urdf:
            stem = Path(normalized_urdf).stem.lower()
            if stem in sim_name_by_stem:
                return sim_name_by_stem[stem]

        key = clip_object_name.strip().lower()
        if key:
            if key in sim_name_by_stem:
                return sim_name_by_stem[key]
            for name in sim_names:
                name_lc = name.lower()
                if key == name_lc or name_lc.endswith(f"_{key}") or name_lc.endswith(key):
                    return name

        logger.warning(
            "No simulator object matched clip '{}' (object_name='{}', object_urdf='{}'); "
            "fallback to '{}'.",
            clip_id,
            clip_object_name,
            clip_object_urdf,
            sim_names[0],
        )
        return sim_names[0]

    def _configure_simulator_object_mapping(self) -> None:
        sim = self._env.simulator
        rigid_objects = getattr(getattr(sim, "scene", None), "rigid_objects", {})

        object_urdf_by_name_raw = getattr(sim, "_object_urdf_by_name", {})
        object_urdf_by_name: dict[str, str] = (
            dict(object_urdf_by_name_raw) if isinstance(object_urdf_by_name_raw, dict) else {}
        )

        sim_object_names: list[str] = [name for name in object_urdf_by_name.keys() if name != "usd_scene_objects"]
        if not sim_object_names and hasattr(rigid_objects, "keys"):
            sim_object_names = [name for name in rigid_objects.keys() if name != "usd_scene_objects"]
        if not sim_object_names:
            sim_object_names = ["object"]

        self._sim_object_names = list(dict.fromkeys(sim_object_names))
        self._clip_object_ids = torch.zeros(self.motion.num_clips, dtype=torch.long, device=self.device)

        if len(self._sim_object_names) == 1:
            self.object_name = self._sim_object_names[0]
            self.object_indices_in_simulator = sim.get_actor_indices(self.object_name, env_ids=None)
            self._multi_object_enabled = False
            self._object_indices_matrix = None
            env_object_urdf_paths = getattr(sim, "_env_object_urdf_paths", None)
            if isinstance(env_object_urdf_paths, list) and env_object_urdf_paths:
                unique_env_object_count = len({self._normalize_path_key(path) for path in env_object_urdf_paths if path})
                logger.info(
                    "Using single simulator object slot '{}' with {} env-specific object assignment(s) across {} clips.",
                    self.object_name,
                    unique_env_object_count,
                    self.motion.num_clips,
                )
            else:
                logger.info("Using single object '{}' for all {} clips.", self.object_name, self.motion.num_clips)
            return

        sim_name_by_urdf: dict[str, str] = {}
        sim_name_by_stem: dict[str, str] = {}
        for name in self._sim_object_names:
            sim_name_by_stem[name.lower()] = name
            urdf_path = object_urdf_by_name.get(name, "")
            normalized = self._normalize_path_key(urdf_path)
            if normalized:
                sim_name_by_urdf[normalized] = name
                sim_name_by_stem[Path(normalized).stem.lower()] = name

        clip_object_names = self.motion.clip_object_names
        clip_object_urdfs = self.motion.clip_object_urdf_paths
        if len(clip_object_names) != self.motion.num_clips:
            clip_object_names = [""] * self.motion.num_clips
        if len(clip_object_urdfs) != self.motion.num_clips:
            clip_object_urdfs = [""] * self.motion.num_clips

        clip_object_ids: list[int] = []
        for clip_idx, clip_id in enumerate(self.motion.clip_ids):
            resolved_name = self._resolve_sim_object_name(
                clip_id=clip_id,
                clip_object_name=clip_object_names[clip_idx],
                clip_object_urdf=clip_object_urdfs[clip_idx],
                sim_names=self._sim_object_names,
                sim_name_by_urdf=sim_name_by_urdf,
                sim_name_by_stem=sim_name_by_stem,
            )
            clip_object_ids.append(self._sim_object_names.index(resolved_name))

        self._clip_object_ids = torch.tensor(clip_object_ids, dtype=torch.long, device=self.device)
        object_indices = [sim.get_actor_indices(name, env_ids=None) for name in self._sim_object_names]
        self._object_indices_matrix = torch.stack(object_indices, dim=0)
        self.object_name = self._sim_object_names[0]
        self.object_indices_in_simulator = self._object_indices_matrix[0]
        self._multi_object_enabled = True
        logger.info(
            "Configured multi-object mapping: {} simulator objects for {} clips.",
            len(self._sim_object_names),
            self.motion.num_clips,
        )

    def _configure_debug_representative_clips(self) -> None:
        self._debug_representative_clip_ids = None
        debug_mode = bool(getattr(self._env.training_config, "debug", False))
        toy_mode = bool(getattr(self._env.training_config, "toy_mode", False))
        if not (debug_mode or toy_mode):
            return
        if not self.multi_clip or not self.motion.has_object:
            return

        clip_object_names = self.motion.clip_object_names
        clip_object_urdfs = self.motion.clip_object_urdf_paths
        representative_ids: list[int] = []
        seen_keys: set[str] = set()
        for clip_idx in range(self.motion.num_clips):
            obj_name = clip_object_names[clip_idx].strip() if clip_idx < len(clip_object_names) else ""
            obj_urdf = clip_object_urdfs[clip_idx].strip() if clip_idx < len(clip_object_urdfs) else ""
            normalized_urdf = self._normalize_path_key(obj_urdf)
            if normalized_urdf:
                key = f"urdf::{normalized_urdf}"
            elif obj_name:
                key = f"name::{obj_name.lower()}"
            else:
                key = "unknown"

            if key in seen_keys:
                continue
            seen_keys.add(key)
            representative_ids.append(clip_idx)

        if not representative_ids:
            representative_ids = [0]

        self._debug_representative_clip_ids = torch.tensor(representative_ids, dtype=torch.long, device=self.device)
        logger.info(
            "Debug/Toy mode: using {} representative clips (one per URDF/object key) over {} total clips.",
            len(representative_ids),
            self.motion.num_clips,
        )

    def _configure_fixed_env_clip_assignment(self) -> None:
        self._fixed_clip_ids = None
        if not self.motion.has_object:
            return

        env_object_urdf_paths = getattr(self._env.simulator, "_env_object_urdf_paths", None)
        if not isinstance(env_object_urdf_paths, list) or not env_object_urdf_paths:
            return
        if len(env_object_urdf_paths) != self.num_envs:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires one simulator object URDF per env. "
                f"Got {len(env_object_urdf_paths)} entries for {self.num_envs} envs."
            )

        clip_object_urdfs = self.motion.clip_object_urdf_paths
        if len(clip_object_urdfs) != self.motion.num_clips:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires clip object URDF metadata for every clip. "
                f"Motion bank exposed {len(clip_object_urdfs)} URDF entries for {self.motion.num_clips} clips."
            )

        clip_ids_by_urdf: dict[str, list[int]] = {}
        missing_clip_urdf_ids: list[str] = []
        for clip_idx, clip_urdf in enumerate(clip_object_urdfs):
            normalized_urdf = self._normalize_path_key(clip_urdf)
            if not normalized_urdf:
                missing_clip_urdf_ids.append(self.motion.clip_ids[clip_idx])
                continue
            clip_ids_by_urdf.setdefault(normalized_urdf, []).append(clip_idx)

        if missing_clip_urdf_ids:
            raise RuntimeError(
                "Fixed env-to-clip assignment requires object URDF metadata on every clip. "
                f"Missing clip metadata for {len(missing_clip_urdf_ids)} clip(s): {missing_clip_urdf_ids[:8]}"
            )

        fixed_clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        seen_counts_by_urdf: dict[str, int] = {}
        unmatched_env_ids: list[int] = []
        for env_id, env_object_urdf in enumerate(env_object_urdf_paths):
            normalized_urdf = self._normalize_path_key(env_object_urdf)
            clip_candidates = clip_ids_by_urdf.get(normalized_urdf)
            if not clip_candidates:
                unmatched_env_ids.append(env_id)
                continue
            seen_count = seen_counts_by_urdf.get(normalized_urdf, 0)
            fixed_clip_ids[env_id] = int(clip_candidates[seen_count % len(clip_candidates)])
            seen_counts_by_urdf[normalized_urdf] = seen_count + 1

        if unmatched_env_ids:
            sample_env_ids = unmatched_env_ids[:8]
            sample_urdfs = [env_object_urdf_paths[idx] for idx in sample_env_ids]
            raise RuntimeError(
                "Fixed env-to-clip assignment requires every env object URDF to appear in the motion bank. "
                f"Unmatched env count={len(unmatched_env_ids)} sample env ids={sample_env_ids} "
                f"sample urdfs={sample_urdfs}"
            )

        self._fixed_clip_ids = fixed_clip_ids
        assigned_unique_clip_count = int(torch.unique(fixed_clip_ids).numel())
        clip_groups_with_multiple_clips = sum(1 for clip_ids in clip_ids_by_urdf.values() if len(clip_ids) > 1)
        if clip_groups_with_multiple_clips > 0:
            logger.info(
                "Configured fixed env-to-clip assignment across {} envs using {} URDF groups and {} active clips. "
                "URDF groups with multiple clips are assigned round-robin across envs.",
                self.num_envs,
                len(clip_ids_by_urdf),
                assigned_unique_clip_count,
            )
        else:
            logger.info(
                "Configured fixed env-to-clip assignment across {} envs and {} active clips.",
                self.num_envs,
                assigned_unique_clip_count,
            )

    def _configure_contact_prior_regions(self) -> None:
        self._contact_prior_available = False
        self._contact_prior_force_body_names_by_region = {region: [] for region in _CONTACT_PRIOR_REGION_NAMES}
        self._contact_prior_position_body_names_by_region = {region: [] for region in _CONTACT_PRIOR_REGION_NAMES}
        self._contact_prior_position_body_indices_by_region = {
            region: torch.zeros((0,), device=self.device, dtype=torch.long) for region in _CONTACT_PRIOR_REGION_NAMES
        }
        if not self.motion.has_object:
            return

        getter = getattr(self._env.simulator, "get_object_contact_force_history", None)
        if getter is None:
            logger.warning("Online contact prior disabled: simulator does not expose object-only contact force history.")
            return

        all_body_names = list(self._env.simulator.body_names)  # type: ignore[attr-defined]
        body_name_to_index = {name: idx for idx, name in enumerate(all_body_names)}
        self._contact_prior_available = True

        for region_name in _CONTACT_PRIOR_REGION_NAMES:
            force_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_FORCE_BODY_NAMES[region_name]
                if body_name in body_name_to_index
            ]
            position_names = [
                body_name
                for body_name in _CONTACT_PRIOR_REGION_POSITION_BODY_NAMES[region_name]
                if body_name in body_name_to_index
            ]
            self._contact_prior_force_body_names_by_region[region_name] = force_names
            self._contact_prior_position_body_names_by_region[region_name] = position_names
            position_indices = [body_name_to_index[body_name] for body_name in position_names]
            self._contact_prior_position_body_indices_by_region[region_name] = torch.tensor(
                position_indices,
                dtype=torch.long,
                device=self.device,
            )
            if not force_names or not position_names:
                logger.warning(
                    "Contact prior region '{}' is partially unavailable. force_bodies={} position_bodies={}",
                    region_name,
                    force_names,
                    position_names,
                )

    def _get_active_object_indices(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.object_indices_in_simulator is None:
            raise RuntimeError(
                "Simulator object indices are not configured. "
                "Use motion clips with object data and enable robot object assets, "
                "or switch to a non-object experiment."
            )
        env_ids_tensor = self._ensure_index_tensor(env_ids)
        if not self._multi_object_enabled or self._clip_object_ids is None or self._object_indices_matrix is None:
            if env_ids is None:
                return self.object_indices_in_simulator
            return self.object_indices_in_simulator[env_ids_tensor]
        active_object_ids = self._clip_object_ids[self.clip_ids[env_ids_tensor]]
        return self._object_indices_matrix[active_object_ids, env_ids_tensor]

    def _set_simulator_object_states(self, env_ids: torch.Tensor, active_states: torch.Tensor) -> None:
        if not self._multi_object_enabled or self._clip_object_ids is None:
            self._env.simulator.set_actor_states([self.object_name], env_ids, active_states)
            return

        active_object_ids = self._clip_object_ids[self.clip_ids[env_ids]]
        all_states: list[torch.Tensor] = []
        for object_id, _ in enumerate(self._sim_object_names):
            states = torch.zeros((env_ids.numel(), 13), device=self.device, dtype=torch.float32)
            states[:, 2] = -100.0 - 5.0 * float(object_id)
            states[:, 6] = 1.0
            active_mask = active_object_ids == object_id
            if active_mask.any():
                states[active_mask] = active_states[active_mask]
            all_states.append(states)
        stacked_states = torch.cat(all_states, dim=0)
        self._env.simulator.set_actor_states(self._sim_object_names, env_ids, stacked_states)

    def _apply_manual_object_reset_overrides(
        self,
        obj_pos_w: torch.Tensor,
        obj_quat_w: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.manual_object_reset_enabled:
            return obj_pos_w, obj_quat_w
        if self.manual_object_reset_pos_offset_w is not None:
            obj_pos_w = obj_pos_w + self.manual_object_reset_pos_offset_w[env_ids]
        if self.manual_object_reset_rpy_offset is not None:
            rpy = self.manual_object_reset_rpy_offset[env_ids]
            delta_quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            obj_quat_w = quat_mul(delta_quat, obj_quat_w, w_last=True)
        return obj_pos_w, obj_quat_w

    @staticmethod
    def contact_prior_region_names() -> tuple[str, ...]:
        return _CONTACT_PRIOR_REGION_NAMES

    def _current_contact_prior_phase_ids(self) -> torch.Tensor:
        if self.pickup_anchor_set is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        return self.pickup_anchor_set.to(dtype=torch.long)

    def _object_contact_force_history_by_names(self, body_names: list[str]) -> torch.Tensor:
        return self.get_body_object_contact_force_history(body_names)

    def _object_contact_body_indices_by_names(self, body_names: list[str]) -> torch.Tensor:
        key = tuple(body_names)
        cached = self._object_contact_body_indices_cache.get(key)
        if cached is not None:
            return cached

        simulator_body_names = list(getattr(self._env.simulator, "body_names", []))
        missing = [name for name in body_names if name not in simulator_body_names]
        if missing:
            raise ValueError(f"Requested object-contact bodies {missing} are not available in simulator bodies.")

        indices = torch.tensor(
            [simulator_body_names.index(name) for name in body_names],
            device=self.device,
            dtype=torch.long,
        )
        self._object_contact_body_indices_cache[key] = indices
        return indices

    def _object_contact_proximity_mask_by_indices(
        self,
        body_indices: torch.Tensor,
        *,
        distance_threshold: float = _OBJECT_CONTACT_PROXY_DISTANCE_THRESHOLD,
    ) -> torch.Tensor:
        if body_indices.numel() == 0 or not self.motion.has_object:
            return torch.zeros((self.num_envs, body_indices.numel()), device=self.device, dtype=torch.bool)

        half_extents = 0.5 * torch.clamp(self.object_size, min=1.0e-4)
        body_pos_obj = self._body_positions_in_object_frame(body_indices)
        signed_outside = torch.abs(body_pos_obj) - half_extents.unsqueeze(1)
        outside = torch.clamp(signed_outside, min=0.0)
        outside_dist = torch.linalg.norm(outside, dim=-1)
        return outside_dist <= float(distance_threshold)

    def _proxy_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        if not body_names:
            return torch.zeros((self.num_envs, 1, 0, 3), device=self.device, dtype=torch.float32)

        raw_history = getattr(self._env.simulator, "contact_forces_history", None)
        if raw_history is None:
            return torch.zeros((self.num_envs, 1, len(body_names), 3), device=self.device, dtype=torch.float32)

        body_indices = self._object_contact_body_indices_by_names(body_names)
        body_force_history = raw_history[:, :, body_indices, :].to(dtype=torch.float32)
        proximity_mask = self._object_contact_proximity_mask_by_indices(body_indices).to(dtype=body_force_history.dtype)
        return body_force_history * proximity_mask.unsqueeze(1).unsqueeze(-1)

    def get_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        if not body_names:
            return torch.zeros((self.num_envs, 1, 0, 3), device=self.device, dtype=torch.float32)

        proxy_history = self._proxy_body_object_contact_force_history(body_names)

        getter = getattr(self._env.simulator, "get_object_contact_force_history", None)
        if getter is None:
            return proxy_history

        try:
            filtered_history = getter(body_names).to(dtype=torch.float32)
        except Exception:
            return proxy_history

        if filtered_history.shape[1] != proxy_history.shape[1]:
            if filtered_history.shape[1] == 1:
                filtered_history = filtered_history.expand(-1, proxy_history.shape[1], -1, -1)
            elif proxy_history.shape[1] == 1:
                proxy_history = proxy_history.expand(-1, filtered_history.shape[1], -1, -1)
            else:
                history_len = min(filtered_history.shape[1], proxy_history.shape[1])
                filtered_history = filtered_history[:, :history_len]
                proxy_history = proxy_history[:, :history_len]

        filtered_norm = torch.linalg.norm(filtered_history, dim=-1)
        proxy_norm = torch.linalg.norm(proxy_history, dim=-1)
        use_proxy = proxy_norm > filtered_norm
        return torch.where(use_proxy.unsqueeze(-1), proxy_history, filtered_history)

    def _body_positions_in_object_frame(self, body_indices: torch.Tensor) -> torch.Tensor:
        if body_indices.numel() == 0:
            return torch.zeros((self.num_envs, 0, 3), device=self.device, dtype=torch.float32)
        body_pos_w = self._env.simulator._rigid_body_pos[:, body_indices, :]
        object_pos_w = self.simulator_object_pos_w[:, None, :]
        object_quat_inv = quat_inverse(self.simulator_object_quat_w, w_last=True)[:, None, :].expand(
            -1, body_indices.numel(), -1
        )
        return quat_apply(object_quat_inv, body_pos_w - object_pos_w, w_last=True)

    def get_current_contact_prior_region_measurements(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        current_force = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        current_contact = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.bool)
        current_position = torch.zeros((self.num_envs, num_regions, 3), device=self.device, dtype=torch.float32)
        if not self.motion.has_object or not self._contact_prior_available:
            return current_force, current_contact, current_position

        for region_idx, region_name in enumerate(_CONTACT_PRIOR_REGION_NAMES):
            force_body_names = self._contact_prior_force_body_names_by_region.get(region_name, [])
            if force_body_names:
                force_history = self._object_contact_force_history_by_names(force_body_names)
                per_body_force = torch.max(torch.norm(force_history, dim=-1), dim=1)[0]
                region_force = torch.max(per_body_force, dim=1)[0]
                current_force[:, region_idx] = region_force
                current_contact[:, region_idx] = region_force > _CONTACT_PRIOR_FORCE_THRESHOLD

            position_body_indices = self._contact_prior_position_body_indices_by_region.get(region_name)
            if position_body_indices is None or position_body_indices.numel() == 0:
                continue

            position_body_names = self._contact_prior_position_body_names_by_region.get(region_name, [])
            relative_positions = self._body_positions_in_object_frame(position_body_indices)
            if position_body_names:
                position_force_history = self._object_contact_force_history_by_names(position_body_names)
                position_force_weights = torch.max(torch.norm(position_force_history, dim=-1), dim=1)[0]
            else:
                position_force_weights = torch.zeros(
                    (self.num_envs, position_body_indices.numel()),
                    device=self.device,
                    dtype=torch.float32,
                )

            uniform_weights = torch.full_like(position_force_weights, 1.0 / float(position_body_indices.numel()))
            weight_denom = position_force_weights.sum(dim=1, keepdim=True)
            normalized_weights = torch.where(
                weight_denom > 1.0e-6,
                position_force_weights / weight_denom.clamp_min(1.0e-6),
                uniform_weights,
            )
            current_position[:, region_idx] = torch.sum(relative_positions * normalized_weights.unsqueeze(-1), dim=1)

        return current_force, current_contact, current_position

    def _default_pose_reset_targets(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dof_pos = self._env.default_dof_pos[env_ids].clone()
        dof_vel = torch.zeros_like(dof_pos)

        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_pos[:, 2] = self._init_root_pos[2]

        init_root_quat = self._init_root_rot.unsqueeze(0).expand(env_ids.numel(), -1)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)
        _, _, motion_yaw = get_euler_xyz(self.body_quat_w[env_ids, 0], w_last=True)
        root_rot = quat_from_euler_xyz(init_roll, init_pitch, motion_yaw)

        root_lin_vel = self._init_root_lin_vel.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        root_ang_vel = self._init_root_ang_vel.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        return dof_pos, dof_vel, root_pos, root_rot, root_lin_vel, root_ang_vel

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        debug_tile_layout = os.environ.get("HOLOSOMA_DEBUG_TILE_LAYOUT", "0").lower() in ("1", "true", "yes", "on")
        use_fixed_tile_layout = (
            debug_tile_layout
            and self.multi_clip
            and self._fixed_clip_ids is None
            and self.motion_cfg.pair_terrain_with_motion
            and self._terrain_row_ids is not None
            and self._terrain_row_count > 0
            and self.motion.num_clips > 0
        )

        if use_fixed_tile_layout:
            row_count = max(1, int(self._terrain_row_count))
            tile_capacity = row_count * int(self.motion.num_clips)
            tile_ids = torch.remainder(env_ids, tile_capacity)
            self.clip_ids[env_ids] = torch.div(tile_ids, row_count, rounding_mode="floor")
            self._terrain_row_ids[env_ids] = torch.remainder(tile_ids, row_count)
        else:
            if self._forced_clip_idx is not None:
                self.clip_ids[env_ids] = int(self._forced_clip_idx)
            elif self._fixed_clip_ids is not None:
                self.clip_ids[env_ids] = self._fixed_clip_ids[env_ids]
            elif self._debug_representative_clip_ids is not None and self._debug_representative_clip_ids.numel() > 0:
                reps = self._debug_representative_clip_ids
                self.clip_ids[env_ids] = reps[env_ids % reps.numel()]
            elif self.multi_clip:
                self._update_clip_success_stats(env_ids)
                if self._env.is_evaluating:
                    self.clip_ids[env_ids] = 0
                else:
                    if self._clip_sampling_weights is None:
                        self.clip_ids[env_ids] = torch.randint(
                            0, self.motion.num_clips, (env_ids.numel(),), device=self.device
                        )
                    else:
                        self.clip_ids[env_ids] = torch.multinomial(
                            self._clip_sampling_weights, env_ids.numel(), replacement=True
                        )
            else:
                self.clip_ids[env_ids] = 0

            if self._terrain_row_ids is not None:
                if self._env.is_evaluating or self._terrain_row_count <= 1:
                    self._terrain_row_ids[env_ids] = 0
                else:
                    self._terrain_row_ids[env_ids] = torch.randint(
                        0, self._terrain_row_count, (env_ids.numel(),), device=self.device
                    )

        # 0. Sample the time steps
        if self.use_adaptive_timesteps_sampler:
            phase = self.adaptive_timesteps_sampler.sample(self.clip_ids[env_ids])
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        if self._env.is_evaluating:
            phase = torch.zeros_like(phase)

        clip_lengths = self._current_clip_lengths(env_ids)
        start_margin = self._min_start_margin_steps()
        valid_starts = torch.clamp(clip_lengths - start_margin, min=1)
        self.time_steps[env_ids] = (phase * valid_starts).long()

        # Handle start_at_timestep_zero_prob.
        base_prob = self._current_start_at_timestep_zero_prob()
        if base_prob > 0.0:
            probs = torch.full((env_ids.numel(),), base_prob, device=self.device, dtype=torch.float32)
            probs = torch.clamp(probs, 0.0, 1.0)
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < probs, torch.zeros_like(subset), subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        max_valid = torch.clamp(clip_lengths - 2, min=0)
        self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], max_valid)

        if self.motion_cfg.align_motion_to_init_yaw:
            self._update_motion_alignment(env_ids)
        if self.manual_goal_is_external is not None:
            self.manual_goal_is_external[env_ids] = False
        if self.command_only_env_mask is not None:
            self.command_only_env_mask[env_ids] = False
        self._update_sparse_object_goals_on_reset(env_ids, clip_lengths)
        self._clear_runtime_default_pose_prepend(env_ids)

        # 1. Get the reference root/body poses
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_rot = self.body_quat_w[env_ids, 0].clone()  # xyzw
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()
        runtime_prepend_mask = self._runtime_default_pose_prepend_reset_mask(env_ids)

        if self._reset_to_default_pose:
            dof_pos, dof_vel, root_pos, root_rot, root_lin_vel, root_ang_vel = self._default_pose_reset_targets(env_ids)
        elif torch.any(runtime_prepend_mask):
            prepend_env_ids = env_ids[runtime_prepend_mask]
            prepend_targets = self._default_pose_reset_targets(prepend_env_ids)
            dof_pos[runtime_prepend_mask] = prepend_targets[0]
            dof_vel[runtime_prepend_mask] = prepend_targets[1]
            root_pos[runtime_prepend_mask] = prepend_targets[2]
            root_rot[runtime_prepend_mask] = prepend_targets[3]
            root_lin_vel[runtime_prepend_mask] = prepend_targets[4]
            root_ang_vel[runtime_prepend_mask] = prepend_targets[5]

        # 2. Adding noise
        reset_noise_scale = torch.ones((env_ids.numel(), 1), device=self.device, dtype=torch.float32)
        reset_noise_scale_3 = reset_noise_scale.expand(-1, 3)

        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise * reset_noise_scale
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0) * reset_noise_scale_3  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (
            (torch.rand((len(env_ids), 3), device=self.device) - 0.5)
            * 2
            * root_rot_noise_rpy.unsqueeze(0)
            * reset_noise_scale_3
        )
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0) * reset_noise_scale_3  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0) * reset_noise_scale_3  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel
        self._reset_pickup_anchor_state(env_ids, root_pos_w=target_root_pos, root_quat_w=target_root_rot)

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (1, 3)
            target_obj_pos = obj_pos + (
                (torch.rand(obj_pos.shape, device=self.device) - 0.5)
                * 2
                * obj_pos_noise
                * reset_noise_scale_3
            )
            target_obj_pos, target_obj_ori = self._apply_manual_object_reset_overrides(
                target_obj_pos, obj_ori, env_ids
            )

            object_states = torch.cat(
                [target_obj_pos, target_obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 13), xyzw
            # 4.3 set active object states; inactive objects are parked away for multi-URDF banks.
            self._set_simulator_object_states(env_ids, object_states)
            self._reset_pickup_anchor_state(
                env_ids,
                root_pos_w=target_root_pos,
                root_quat_w=target_root_rot,
                object_pos_w=target_obj_pos,
            )
            self._update_manual_goal_override(env_ids)

        if torch.any(runtime_prepend_mask):
            self._activate_runtime_default_pose_prepend(env_ids[runtime_prepend_mask])

        self._update_future_target_poses()

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)
        if (
            self._runtime_default_pose_prepend_enabled
            and self._runtime_default_pose_prepend_active is not None
            and self._runtime_default_pose_prepend_step is not None
        ):
            active_mask = self._runtime_default_pose_prepend_active
            if torch.any(active_mask):
                advance_mask = advance_mask & ~active_mask
                last_step_mask = active_mask & (
                    self._runtime_default_pose_prepend_step >= (self._runtime_default_pose_prepend_steps - 1)
                )
                keep_warmup_mask = active_mask & ~last_step_mask
                self._runtime_default_pose_prepend_step[keep_warmup_mask] += 1
                self._runtime_default_pose_prepend_active[last_step_mask] = False

        # Handle freeze_at_timestep_zero_prob: for envs at timestep 0, randomly decide whether to advance
        freeze_prob = self._current_freeze_at_timestep_zero_prob()
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == 0
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()
        max_steps = self._current_clip_lengths() - 1
        self.time_steps = torch.minimum(self.time_steps, max_steps)

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.update_bin_failed_count()

        self._update_future_target_poses()
        self._update_pickup_anchor_state()
        self._update_contact_prior_state()
        self._update_manual_goal_override()

    def _current_clip_lengths(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        return self.motion.clip_lengths[clip_ids]

    def _get_motion_indices(self, steps: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.motion.num_clips <= 1:
            return steps
        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        offsets = self.motion.clip_offsets[clip_ids]
        if steps.ndim > offsets.ndim:
            offsets = offsets.view(-1, *([1] * (steps.ndim - 1)))
        return offsets + steps

    def _clear_runtime_default_pose_prepend(self, env_ids: torch.Tensor) -> None:
        if (
            not self._runtime_default_pose_prepend_enabled
            or self._runtime_default_pose_prepend_active is None
            or self._runtime_default_pose_prepend_step is None
        ):
            return
        self._runtime_default_pose_prepend_active[env_ids] = False
        self._runtime_default_pose_prepend_step[env_ids] = 0

    def _runtime_default_pose_prepend_reset_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._runtime_default_pose_prepend_enabled:
            return torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.bool)
        return self.time_steps[env_ids] == 0

    def _activate_runtime_default_pose_prepend(self, env_ids: torch.Tensor) -> None:
        if (
            env_ids.numel() == 0
            or not self._runtime_default_pose_prepend_enabled
            or self._runtime_default_pose_prepend_active is None
            or self._runtime_default_pose_prepend_step is None
        ):
            return
        self._runtime_default_pose_prepend_active[env_ids] = True
        self._runtime_default_pose_prepend_step[env_ids] = 0

    def get_runtime_default_pose_prepend_mask(self) -> torch.Tensor:
        if self._runtime_default_pose_prepend_active is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return self._runtime_default_pose_prepend_active

    def _runtime_default_pose_prepend_active_env_ids(self) -> torch.Tensor:
        if not self._runtime_default_pose_prepend_enabled or self._runtime_default_pose_prepend_active is None:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        return torch.nonzero(self._runtime_default_pose_prepend_active, as_tuple=False).flatten()

    def _runtime_default_pose_prepend_alpha(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert self._runtime_default_pose_prepend_step is not None
        alpha = self._runtime_default_pose_prepend_step[env_ids].to(dtype=torch.float32)
        return alpha / float(self._runtime_default_pose_prepend_steps)

    def _blend_runtime_default_pose_prepend_lerp(self, current: torch.Tensor, key: str) -> torch.Tensor:
        env_ids = self._runtime_default_pose_prepend_active_env_ids()
        if env_ids.numel() == 0:
            return current
        defaults = self._runtime_default_pose_prepend_defaults.get(key)
        if defaults is None:
            return current
        clip_ids = self.clip_ids[env_ids]
        alpha = self._runtime_default_pose_prepend_alpha(env_ids)
        alpha_view = alpha.view(-1, *([1] * (current.ndim - 1)))
        blended = current.clone()
        blended[env_ids] = defaults[clip_ids] + alpha_view * (current[env_ids] - defaults[clip_ids])
        return blended

    def _blend_runtime_default_pose_prepend_quat(self, current: torch.Tensor, key: str) -> torch.Tensor:
        env_ids = self._runtime_default_pose_prepend_active_env_ids()
        if env_ids.numel() == 0:
            return current
        defaults = self._runtime_default_pose_prepend_defaults.get(key)
        if defaults is None:
            return current
        clip_ids = self.clip_ids[env_ids]
        start = defaults[clip_ids]
        end = current[env_ids]
        alpha = self._runtime_default_pose_prepend_alpha(env_ids)

        if current.ndim == 2:
            blended_env = slerp(start, end, alpha.unsqueeze(-1))
        elif current.ndim == 3:
            alpha_flat = alpha.unsqueeze(1).expand(-1, start.shape[1]).reshape(-1, 1)
            blended_env = slerp(start.reshape(-1, 4), end.reshape(-1, 4), alpha_flat).view_as(start)
        else:
            raise ValueError(f"Unsupported quaternion tensor rank {current.ndim}.")

        blended = current.clone()
        blended[env_ids] = blended_env
        return blended

    def _raw_motion_joint_pos(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        joint_pos = self.motion.joint_pos[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(joint_pos, "joint_pos")

    def _raw_motion_joint_vel(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        joint_vel = self.motion.joint_vel[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(joint_vel, "joint_vel")

    def _raw_motion_body_pos_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        body_pos = self.motion.body_pos_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_pos, "body_pos")

    def _raw_motion_body_quat_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        body_quat = self.motion.body_quat_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_quat(body_quat, "body_quat")

    def _raw_motion_body_lin_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        body_lin_vel = self.motion.body_lin_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_lin_vel, "body_lin_vel")

    def _raw_motion_body_ang_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        body_ang_vel = self.motion.body_ang_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(body_ang_vel, "body_ang_vel")

    def _raw_motion_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        motion_idx = self._get_motion_indices(self.time_steps)
        object_pos = self.motion.object_pos_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(object_pos, "object_pos")

    def _raw_motion_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        motion_idx = self._get_motion_indices(self.time_steps)
        object_quat = self.motion.object_quat_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_quat(object_quat, "object_quat")

    def _raw_motion_object_lin_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        motion_idx = self._get_motion_indices(self.time_steps)
        object_lin_vel = self.motion.object_lin_vel_w[motion_idx]
        return self._blend_runtime_default_pose_prepend_lerp(object_lin_vel, "object_lin_vel")

    @property
    def current_clip_lengths(self) -> torch.Tensor:
        return self._current_clip_lengths()

    def motion_end_mask(self) -> torch.Tensor:
        clip_lengths = self._current_clip_lengths()
        return self.time_steps >= (clip_lengths - 2)

    def _min_start_margin_steps(self) -> int:
        """Ensure enough frames for stepping + future target poses."""
        return max(2, int(self.num_future_steps))

    def _valid_start_counts(self) -> torch.Tensor:
        margin = self._min_start_margin_steps()
        valid = self.motion.clip_lengths - margin
        valid = torch.clamp(valid, min=1)
        return valid.to(dtype=torch.float32)

    def _configure_clean_noisy_clip_curriculum(self) -> None:
        if not self.multi_clip:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        cfg = self._clean_noisy_clip_curriculum_cfg
        if cfg is None or not cfg.enabled:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        clean_mask = build_prefix_mask(self.motion.clip_ids, cfg.clean_clip_name_prefixes).to(device=self.device)
        noisy_mask = ~clean_mask
        if not torch.any(clean_mask):
            logger.warning(
                "clean_noisy_clip_curriculum is enabled but no clips matched clean prefixes {}. Disabling it.",
                cfg.clean_clip_name_prefixes,
            )
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return
        if not torch.any(noisy_mask):
            logger.warning(
                "clean_noisy_clip_curriculum is enabled but all clips matched clean prefixes {}. Disabling it.",
                cfg.clean_clip_name_prefixes,
            )
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            return

        if len(cfg.stage_start_iterations) != len(cfg.clean_group_probabilities):
            raise ValueError(
                "clean_noisy_clip_curriculum.stage_start_iterations and clean_group_probabilities "
                f"must have the same length, got {len(cfg.stage_start_iterations)} and "
                f"{len(cfg.clean_group_probabilities)}."
            )
        if not cfg.stage_start_iterations:
            raise ValueError("clean_noisy_clip_curriculum requires at least one schedule stage.")
        if any(value < 0.0 or value > 1.0 for value in cfg.clean_group_probabilities):
            raise ValueError(
                "clean_noisy_clip_curriculum.clean_group_probabilities must stay in [0, 1], "
                f"got {cfg.clean_group_probabilities}."
            )

        self._clean_clip_mask = clean_mask
        self._noisy_clip_mask = noisy_mask
        logger.info(
            "Enabled clean/noisy clip curriculum: {} clean clips, {} noisy clips, stages={} probs={}.",
            int(clean_mask.sum().item()),
            int(noisy_mask.sum().item()),
            list(cfg.stage_start_iterations),
            [float(value) for value in cfg.clean_group_probabilities],
        )

    def _current_clean_group_probability(self) -> float | None:
        cfg = self._clean_noisy_clip_curriculum_cfg
        if not self._clean_noisy_clip_curriculum_enabled or cfg is None:
            return None
        return piecewise_constant_schedule_value(
            self._training_iteration,
            cfg.stage_start_iterations,
            cfg.clean_group_probabilities,
        )

    def _refresh_current_clip_sampling_weights(self) -> None:
        if not self.multi_clip:
            return

        if self._raw_clip_sampling_weights is None:
            return

        weights = self._raw_clip_sampling_weights
        if self._clean_noisy_clip_curriculum_enabled and self._clean_clip_mask is not None:
            clean_prob = self._current_clean_group_probability()
            if clean_prob is not None:
                weights = project_group_weights(
                    weights,
                    clean_mask=self._clean_clip_mask,
                    clean_group_probability=clean_prob,
                )
        total = torch.sum(weights)
        if torch.isfinite(total) and total.item() > 0.0:
            self._clip_sampling_weights = weights / total
        else:
            self._clip_sampling_weights = None

    def get_clean_noisy_clip_curriculum_log_state(self) -> dict[str, float]:
        """Return scalar clean/noisy curriculum metrics for training logs."""
        clean_prob = self._current_clean_group_probability()
        if clean_prob is None or self._clean_clip_mask is None or self._clip_sampling_weights is None:
            return {}
        clean_weight = float(self._clip_sampling_weights[self._clean_clip_mask].sum().item())
        return {
            "clean_clip_target_prob": float(clean_prob),
            "clean_clip_sample_weight": clean_weight,
            "noisy_clip_sample_weight": max(0.0, 1.0 - clean_weight),
        }

    def _init_clip_sampling(self) -> None:
        if not self.multi_clip:
            return
        if self._fixed_clip_ids is not None:
            self._clean_noisy_clip_curriculum_enabled = False
            self._clean_clip_mask = None
            self._noisy_clip_mask = None
            logger.info(
                "Fixed env-to-clip assignment is active; bypassing clip-level weighting curricula. "
                "Only within-clip timestep curriculum remains enabled."
            )
            return
        self._configure_clean_noisy_clip_curriculum()
        strategy = self.clip_weighting_strategy
        if strategy == "uniform_step":
            weights = self._valid_start_counts()
        elif strategy in ("uniform_clip", "success_rate_adaptive"):
            weights = torch.ones(self.motion.num_clips, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown clip_weighting_strategy '{strategy}'.")

        weights = weights / weights.sum()
        self._raw_clip_sampling_weights = weights

        if strategy == "success_rate_adaptive":
            self._base_clip_weights = weights.clone()
            self._clip_success_counts = torch.zeros(self.motion.num_clips, device=self.device)
            self._clip_total_counts = torch.zeros(self.motion.num_clips, device=self.device)
        self._refresh_current_clip_sampling_weights()

    def _update_clip_success_stats(self, env_ids: torch.Tensor) -> None:
        if not self.multi_clip or self.clip_weighting_strategy != "success_rate_adaptive":
            return
        if self._env.is_evaluating:
            return
        if self._clip_success_counts is None or self._clip_total_counts is None:
            return
        if env_ids.numel() == 0:
            return

        episode_lengths = self._env.episode_length_buf[env_ids]
        valid_mask = episode_lengths > 0
        if not torch.any(valid_mask):
            return

        valid_env_ids = env_ids[valid_mask]
        clip_ids = self.clip_ids[valid_env_ids]
        successes = self.motion_end_mask()[valid_env_ids].to(dtype=torch.float32)

        ones = torch.ones_like(successes)
        self._clip_total_counts.index_add_(0, clip_ids, ones)
        self._clip_success_counts.index_add_(0, clip_ids, successes)
        self._refresh_adaptive_clip_weights()

    def _refresh_adaptive_clip_weights(self) -> None:
        if self.clip_weighting_strategy != "success_rate_adaptive":
            return
        if self._clip_total_counts is None or self._clip_success_counts is None:
            return
        if self._base_clip_weights is None:
            return

        total = self._clip_total_counts
        success = self._clip_success_counts
        valid_mask = total > 0

        inv_success = torch.ones_like(total)
        if torch.any(valid_mask):
            success_rates = torch.zeros_like(total)
            success_rates[valid_mask] = success[valid_mask] / total[valid_mask]
            inv_success[valid_mask] = 1.0 / (success_rates[valid_mask] + 0.05)
            mean_inv = inv_success[valid_mask].mean()
            if mean_inv > 1e-6:
                inv_success = inv_success / mean_inv

        factors = torch.clamp(inv_success, self.min_weight_factor, self.max_weight_factor)
        weights = self._base_clip_weights * factors
        if weights.sum() > 1e-9:
            self._raw_clip_sampling_weights = weights / weights.sum()
        else:
            self._raw_clip_sampling_weights = self._base_clip_weights.clone()
        self._refresh_current_clip_sampling_weights()

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _goal_vec3(self, values: list[float], *, name: str) -> torch.Tensor:
        if len(values) != 3:
            raise ValueError(f"{name} must provide exactly 3 values, got {len(values)}")
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _iteration_curriculum_progress(self, start_iter: int | None, end_iter: int | None) -> float | None:
        if start_iter is None or end_iter is None or self._training_iteration is None:
            return None
        if self._training_iteration < start_iter:
            return 0.0
        if end_iter <= start_iter:
            return 1.0
        return min(max(float(self._training_iteration - start_iter) / float(end_iter - start_iter), 0.0), 1.0)

    def _iteration_schedule_value(
        self,
        start_value: float,
        end_value: float,
        *,
        start_iter: int | None,
        end_iter: int | None,
    ) -> float | None:
        if start_iter is None or end_iter is None or self._training_iteration is None:
            return None
        if self._training_iteration < start_iter:
            return 0.0
        if end_iter <= start_iter:
            return float(end_value)
        alpha = self._iteration_curriculum_progress(start_iter, end_iter)
        if alpha is None:
            return None
        return float(start_value + (end_value - start_value) * alpha)

    def _scheduled_reset_prob(
        self,
        start_value: float,
        *,
        end_value: float | None,
        start_iter: int | None,
        end_iter: int | None,
    ) -> float:
        start_value = self._clamp01(float(start_value))
        if end_value is None or start_iter is None or end_iter is None:
            return start_value

        end_value = self._clamp01(float(end_value))
        if self._env.is_evaluating:
            return end_value

        alpha = self._iteration_curriculum_progress(start_iter, end_iter)
        if alpha is None:
            return start_value
        return self._clamp01(start_value + (end_value - start_value) * alpha)

    def _current_start_at_timestep_zero_prob(self) -> float:
        return self._scheduled_reset_prob(
            float(self.motion_cfg.start_at_timestep_zero_prob),
            end_value=self.motion_cfg.start_at_timestep_zero_prob_end,
            start_iter=self.motion_cfg.start_at_timestep_zero_prob_start_iter,
            end_iter=self.motion_cfg.start_at_timestep_zero_prob_end_iter,
        )

    def _current_freeze_at_timestep_zero_prob(self) -> float:
        return self._scheduled_reset_prob(
            float(self.motion_cfg.freeze_at_timestep_zero_prob),
            end_value=self.motion_cfg.freeze_at_timestep_zero_prob_end,
            start_iter=self.motion_cfg.freeze_at_timestep_zero_prob_start_iter,
            end_iter=self.motion_cfg.freeze_at_timestep_zero_prob_end_iter,
        )

    def _sparse_goal_curriculum_progress(self) -> float:
        if self._sparse_goal_cfg is None:
            return 1.0
        progress = self._iteration_curriculum_progress(
            self._sparse_goal_cfg.external_goal_range_start_iter,
            self._sparse_goal_cfg.external_goal_range_end_iter,
        )
        if progress is not None:
            return progress
        ramp_cfg = self._sparse_goal_cfg.external_goal_range_ramp_resets
        if ramp_cfg is None:
            ramp = max(0, int(self._sparse_goal_cfg.external_goal_prob_ramp_resets))
        else:
            ramp = max(0, int(ramp_cfg))
        if ramp <= 0:
            return 1.0
        return min(float(self._sparse_goal_reset_counter) / float(ramp), 1.0)

    def _carry_extension_curriculum_progress(self) -> float:
        if self._sparse_goal_cfg is None:
            return 1.0
        progress = self._iteration_curriculum_progress(
            self._sparse_goal_cfg.carry_extension_range_start_iter,
            self._sparse_goal_cfg.carry_extension_range_end_iter,
        )
        if progress is not None:
            return progress
        ramp_cfg = self._sparse_goal_cfg.carry_extension_range_ramp_resets
        if ramp_cfg is None:
            ramp_cfg = self._sparse_goal_cfg.carry_extension_prob_ramp_resets
        if ramp_cfg is None:
            ramp_cfg = self._sparse_goal_cfg.external_goal_prob_ramp_resets
        ramp = max(0, int(ramp_cfg))
        if ramp <= 0:
            return 1.0
        return min(float(self._sparse_goal_reset_counter) / float(ramp), 1.0)

    def _command_only_env_curriculum_progress(self) -> float:
        if self._sparse_goal_cfg is None:
            return 1.0
        progress = self._iteration_curriculum_progress(
            self._sparse_goal_cfg.command_only_env_prob_start_iter,
            self._sparse_goal_cfg.command_only_env_prob_end_iter,
        )
        if progress is not None:
            return progress
        ramp_cfg = self._sparse_goal_cfg.command_only_env_prob_ramp_resets
        if ramp_cfg is None:
            ramp_cfg = self._sparse_goal_cfg.external_goal_prob_ramp_resets
        ramp = max(0, int(ramp_cfg))
        if ramp <= 0:
            return 1.0
        return min(float(self._sparse_goal_reset_counter) / float(ramp), 1.0)

    def _goal_vec3_interp(
        self,
        end_values: list[float],
        *,
        name: str,
        start_values: list[float] | None = None,
        alpha: float | None = None,
    ) -> torch.Tensor:
        end_tensor = self._goal_vec3(end_values, name=name)
        if start_values is None:
            return end_tensor
        start_tensor = self._goal_vec3(start_values, name=f"{name}_start")
        mix = float(self._clamp01(self._sparse_goal_curriculum_progress() if alpha is None else alpha))
        return start_tensor + (end_tensor - start_tensor) * mix

    def _get_clip_pickup_steps_by_clip(self) -> torch.Tensor:
        cache_name = (
            "_clip_pickup_steps_by_clip_"
            f"h{_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD:.4f}_"
            f"r{_CLIP_PICKUP_LIFT_RATIO_THRESHOLD:.4f}_"
            f"c{_RUNTIME_PICKUP_CONSECUTIVE_STEPS:d}"
        ).replace(".", "p")
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached

        pickup_steps_by_clip = torch.zeros((self.motion.num_clips,), device=self.device, dtype=torch.long)
        if not self.motion.has_object:
            setattr(self, cache_name, pickup_steps_by_clip)
            return pickup_steps_by_clip

        clip_offsets = self.motion.clip_offsets
        clip_lengths = self.motion.clip_lengths
        root_pos_w = self.motion.body_pos_w[:, 0]
        object_pos_w = self.motion.object_pos_w

        for clip_idx in range(self.motion.num_clips):
            clip_start = int(clip_offsets[clip_idx].item())
            clip_length = int(clip_lengths[clip_idx].item())
            if clip_length <= 0:
                continue

            clip_end = clip_start + clip_length
            clip_rel_z = object_pos_w[clip_start:clip_end, 2] - root_pos_w[clip_start:clip_end, 2]
            z_min = clip_rel_z.min()
            z_range = torch.clamp(clip_rel_z.max() - z_min, min=0.0)
            pickup_threshold = z_min + torch.maximum(
                z_min.new_tensor(float(_RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD)),
                z_range * float(_CLIP_PICKUP_LIFT_RATIO_THRESHOLD),
            )

            lifted_mask = clip_rel_z >= pickup_threshold
            pickup_step = _first_sustained_true_index(lifted_mask, _RUNTIME_PICKUP_CONSECUTIVE_STEPS)
            if pickup_step is None:
                lifted_indices = torch.nonzero(lifted_mask, as_tuple=False)
                if lifted_indices.numel() > 0:
                    pickup_step = int(lifted_indices[0, 0].item())
                else:
                    pickup_step = int(torch.argmax(clip_rel_z).item())
            pickup_steps_by_clip[clip_idx] = pickup_step

        setattr(self, cache_name, pickup_steps_by_clip)
        return pickup_steps_by_clip

    def _reset_pickup_anchor_state(
        self,
        env_ids: torch.Tensor,
        *,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
        object_pos_w: torch.Tensor | None = None,
    ) -> None:
        if (
            self.pickup_anchor_set is None
            or self.pickup_anchor_root_pos_w is None
            or self.pickup_anchor_root_quat_w is None
            or self.pickup_object_rel_z_baseline is None
            or self.pickup_consecutive_counter is None
        ):
            return

        self.pickup_anchor_set[env_ids] = False
        self.pickup_consecutive_counter[env_ids] = 0
        self.pickup_anchor_root_pos_w[env_ids] = 0.0
        self.pickup_anchor_root_quat_w[env_ids] = 0.0
        self.pickup_anchor_root_quat_w[env_ids, 3] = 1.0
        self.pickup_object_rel_z_baseline[env_ids] = 0.0

        if root_pos_w is None or root_quat_w is None or object_pos_w is None:
            return
        self.pickup_anchor_root_pos_w[env_ids] = root_pos_w
        self.pickup_anchor_root_quat_w[env_ids] = root_quat_w
        self.pickup_object_rel_z_baseline[env_ids] = object_pos_w[:, 2] - root_pos_w[:, 2]

        # If reset starts after the clip's pickup phase, treat the object as already
        # picked at reset time and anchor the sparse goal to the current reset root.
        clip_pickup_steps = self._get_clip_pickup_steps_by_clip()[self.clip_ids[env_ids]]
        already_picked_mask = self.time_steps[env_ids] >= clip_pickup_steps
        if not torch.any(already_picked_mask):
            return

        prime_env_ids = env_ids[already_picked_mask]
        self.pickup_anchor_set[prime_env_ids] = True
        self.pickup_consecutive_counter[prime_env_ids] = _RUNTIME_PICKUP_CONSECUTIVE_STEPS
        self.pickup_anchor_root_pos_w[prime_env_ids] = root_pos_w[already_picked_mask]
        self.pickup_anchor_root_quat_w[prime_env_ids] = root_quat_w[already_picked_mask]
        self._apply_manual_goal_world_from_command(
            prime_env_ids,
            anchor_pos_w=root_pos_w[already_picked_mask],
            anchor_quat_w=root_quat_w[already_picked_mask],
        )

    def _update_pickup_anchor_state(self) -> None:
        if (
            not self.motion.has_object
            or self.pickup_anchor_set is None
            or self.pickup_anchor_root_pos_w is None
            or self.pickup_anchor_root_quat_w is None
            or self.pickup_object_rel_z_baseline is None
            or self.pickup_consecutive_counter is None
        ):
            return

        current_rel_z = self.simulator_object_pos_w[:, 2] - self.robot_root_pos_w[:, 2]
        lifted = (current_rel_z - self.pickup_object_rel_z_baseline) >= _RUNTIME_PICKUP_LIFT_HEIGHT_THRESHOLD
        self.pickup_consecutive_counter = torch.where(
            lifted,
            self.pickup_consecutive_counter + 1,
            torch.zeros_like(self.pickup_consecutive_counter),
        )
        newly_picked = (~self.pickup_anchor_set) & (
            self.pickup_consecutive_counter >= _RUNTIME_PICKUP_CONSECUTIVE_STEPS
        )
        if not newly_picked.any():
            return

        self.pickup_anchor_set[newly_picked] = True
        self.pickup_anchor_root_pos_w[newly_picked] = self.robot_root_pos_w[newly_picked]
        self.pickup_anchor_root_quat_w[newly_picked] = self.robot_root_quat_w[newly_picked]
        self._apply_manual_goal_world_from_command(
            torch.nonzero(newly_picked, as_tuple=False).view(-1),
            anchor_pos_w=self.robot_root_pos_w[newly_picked],
            anchor_quat_w=self.robot_root_quat_w[newly_picked],
        )

    def _update_contact_prior_state(self) -> None:
        if (
            not self._sparse_goal_curriculum_enabled
            or not self.motion.has_object
            or not self._contact_prior_available
            or self.command_only_env_mask is None
            or self._contact_prior_total_count is None
            or self._contact_prior_contact_sum is None
            or self._contact_prior_force_mean is None
            or self._contact_prior_force_count is None
            or self._contact_prior_position_mean is None
            or self._contact_prior_position_count is None
        ):
            return

        source_mask = ~self.command_only_env_mask
        source_mask &= self._env.episode_length_buf > 1
        if not torch.any(source_mask):
            return

        body_pos_error = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        object_pos_error = torch.norm(self.object_pos_w - self.simulator_object_pos_w, dim=-1)
        object_rot_error = quat_error_magnitude(self.object_quat_w, self.simulator_object_quat_w)
        stable_mask = (
            source_mask
            & (body_pos_error <= _CONTACT_PRIOR_BODY_POS_ERROR_THRESHOLD)
            & (object_pos_error <= _CONTACT_PRIOR_OBJECT_POS_ERROR_THRESHOLD)
            & (object_rot_error <= _CONTACT_PRIOR_OBJECT_ROT_ERROR_THRESHOLD)
        )
        if not torch.any(stable_mask):
            return

        current_force, current_contact, current_position = self.get_current_contact_prior_region_measurements()
        stable_clip_ids = self.clip_ids[stable_mask]
        stable_phase_ids = self._current_contact_prior_phase_ids()[stable_mask]
        stable_contact = current_contact[stable_mask]
        stable_force = current_force[stable_mask]
        stable_position = current_position[stable_mask]
        clip_phase_pairs = torch.unique(torch.stack((stable_clip_ids, stable_phase_ids), dim=1), dim=0)

        for clip_id, phase_id in clip_phase_pairs.tolist():
            pair_mask = (stable_clip_ids == clip_id) & (stable_phase_ids == phase_id)
            if not torch.any(pair_mask):
                continue

            pair_contact = stable_contact[pair_mask]
            pair_force = stable_force[pair_mask]
            pair_position = stable_position[pair_mask]
            pair_count = float(pair_mask.sum().item())
            self._contact_prior_total_count[clip_id, phase_id] += pair_count
            self._contact_prior_contact_sum[clip_id, phase_id] += pair_contact.to(dtype=torch.float32).sum(dim=0)

            for region_idx in range(len(_CONTACT_PRIOR_REGION_NAMES)):
                region_contact_mask = pair_contact[:, region_idx]
                region_contact_count = float(region_contact_mask.to(dtype=torch.float32).sum().item())
                if region_contact_count <= 0.0:
                    continue

                batch_force_mean = pair_force[region_contact_mask, region_idx].mean()
                prev_force_count = self._contact_prior_force_count[clip_id, phase_id, region_idx]
                new_force_count = prev_force_count + region_contact_count
                self._contact_prior_force_mean[clip_id, phase_id, region_idx] = (
                    self._contact_prior_force_mean[clip_id, phase_id, region_idx] * prev_force_count
                    + batch_force_mean * region_contact_count
                ) / new_force_count.clamp_min(1.0)
                self._contact_prior_force_count[clip_id, phase_id, region_idx] = new_force_count

                batch_position_mean = pair_position[region_contact_mask, region_idx].mean(dim=0)
                prev_position_count = self._contact_prior_position_count[clip_id, phase_id, region_idx]
                new_position_count = prev_position_count + region_contact_count
                self._contact_prior_position_mean[clip_id, phase_id, region_idx] = (
                    self._contact_prior_position_mean[clip_id, phase_id, region_idx] * prev_position_count
                    + batch_position_mean * region_contact_count
                ) / new_position_count.clamp_min(1.0)
                self._contact_prior_position_count[clip_id, phase_id, region_idx] = new_position_count

    def get_contact_prior_targets(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        occupancy = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        force = torch.zeros((self.num_envs, num_regions), device=self.device, dtype=torch.float32)
        position = torch.zeros((self.num_envs, num_regions, 3), device=self.device, dtype=torch.float32)
        confidence = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        valid_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if (
            self._contact_prior_total_count is None
            or self._contact_prior_contact_sum is None
            or self._contact_prior_force_mean is None
            or self._contact_prior_position_mean is None
        ):
            return occupancy, force, position, confidence, valid_mask

        phase_ids = self._current_contact_prior_phase_ids()
        total_count = self._contact_prior_total_count[self.clip_ids, phase_ids]
        observed_contact_count = self._contact_prior_contact_sum[self.clip_ids, phase_ids].sum(dim=-1)
        # A prior should only be considered valid after we have actually observed at least one
        # supported body-object contact for this clip/phase. Otherwise confidence can rise from
        # stable co-tracking samples while all contact targets remain identically zero.
        valid_mask = observed_contact_count > 0.0
        if torch.any(valid_mask):
            occupancy[valid_mask] = self._contact_prior_contact_sum[self.clip_ids[valid_mask], phase_ids[valid_mask]] / (
                total_count[valid_mask].unsqueeze(-1).clamp_min(1.0)
            )
            force[valid_mask] = self._contact_prior_force_mean[self.clip_ids[valid_mask], phase_ids[valid_mask]]
            position[valid_mask] = self._contact_prior_position_mean[self.clip_ids[valid_mask], phase_ids[valid_mask]]
            confidence[valid_mask] = torch.clamp(
                total_count[valid_mask] / float(_CONTACT_PRIOR_CONFIDENCE_WARMUP_SAMPLES),
                min=0.0,
                max=1.0,
            )
        return occupancy, force, position, confidence, valid_mask

    def get_contact_prior_region_targets(
        self,
        region_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if region_name not in _CONTACT_PRIOR_REGION_NAMES:
            raise ValueError(f"Unknown contact prior region '{region_name}'.")
        region_idx = _CONTACT_PRIOR_REGION_NAMES.index(region_name)
        occupancy, force, position, confidence, valid_mask = self.get_contact_prior_targets()
        return (
            occupancy[:, region_idx].unsqueeze(-1),
            force[:, region_idx].unsqueeze(-1),
            position[:, region_idx, :],
            confidence.unsqueeze(-1),
            valid_mask.unsqueeze(-1).to(dtype=torch.float32),
        )

    def _apply_motion_alignment_pos_subset(self, pos: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        align_quat = self._align_quat[env_ids]
        align_pos = self._align_pos[env_ids]
        if pos.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, pos.shape[1], -1)
            align_pos = align_pos[:, None, :]
        return quat_apply(align_quat, pos, w_last=True) + align_pos

    def _apply_motion_alignment_quat_subset(self, quat: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        align_quat = self._align_quat[env_ids]
        if quat.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, quat.shape[1], -1)
        return quat_mul(align_quat, quat, w_last=True)

    def _sample_clip_based_object_goal_pose_w(
        self, env_ids: torch.Tensor, clip_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Clip goals represent the final placement target of the active clip.
        # Keep the legacy clip_goal_delta config around for compatibility, but do
        # not use future-step waypoints here anymore.
        final_goal_steps = torch.clamp(clip_lengths - 1, min=0)
        goal_motion_idx = self._get_motion_indices(final_goal_steps, env_ids=env_ids)

        goal_pos_w = self.motion.object_pos_w[goal_motion_idx]
        goal_quat_w = self.motion.object_quat_w[goal_motion_idx]
        if self.motion_cfg.align_motion_to_init_yaw:
            goal_pos_w = self._apply_motion_alignment_pos_subset(goal_pos_w, env_ids)
            goal_quat_w = self._apply_motion_alignment_quat_subset(goal_quat_w, env_ids)
        else:
            goal_pos_w = goal_pos_w + self._get_env_offsets(env_ids)
        return goal_pos_w, goal_quat_w

    def _sample_clip_pickup_anchor_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pickup_steps = self._get_clip_pickup_steps_by_clip()[self.clip_ids[env_ids]]
        pickup_motion_idx = self._get_motion_indices(pickup_steps, env_ids=env_ids)

        anchor_pos_w = self.motion.body_pos_w[pickup_motion_idx, 0]
        anchor_quat_w = self.motion.body_quat_w[pickup_motion_idx, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            anchor_pos_w = self._apply_motion_alignment_pos_subset(anchor_pos_w, env_ids)
            anchor_quat_w = self._apply_motion_alignment_quat_subset(anchor_quat_w, env_ids)
        else:
            anchor_pos_w = anchor_pos_w + self._get_env_offsets(env_ids)
        return anchor_pos_w, anchor_quat_w

    def _goal_command_from_world(
        self,
        goal_pos_w: torch.Tensor,
        goal_quat_w: torch.Tensor,
        *,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_heading_quat = yaw_quat(anchor_quat_w, w_last=True)
        anchor_heading_inv = quat_inverse(anchor_heading_quat, w_last=True)
        goal_pos_heading = quat_apply(anchor_heading_inv, goal_pos_w - anchor_pos_w, w_last=True)
        goal_rot_mat_w = quaternion_to_matrix(goal_quat_w, w_last=True)
        goal_heading = torch.atan2(goal_rot_mat_w[:, 1, 0], goal_rot_mat_w[:, 0, 0])
        _, _, anchor_heading = get_euler_xyz(anchor_heading_quat, w_last=True)
        goal_yaw_rel = normalize_angle(goal_heading - anchor_heading)
        return goal_pos_heading[:, :2], goal_yaw_rel

    def _goal_world_from_command(
        self,
        env_ids: torch.Tensor,
        *,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
        goal_xy_rel: torch.Tensor | None = None,
        goal_yaw_rel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.manual_goal_xy_rel is None or self.manual_goal_yaw_rel is None:
            raise RuntimeError("Manual goal command buffers are not initialized.")
        if goal_xy_rel is None:
            goal_xy_rel = self.manual_goal_xy_rel[env_ids]
        if goal_yaw_rel is None:
            goal_yaw_rel = self.manual_goal_yaw_rel[env_ids, 0]

        anchor_heading_quat = yaw_quat(anchor_quat_w, w_last=True)
        rel_goal_pos = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=torch.float32)
        rel_goal_pos[:, :2] = goal_xy_rel
        goal_pos_w = anchor_pos_w + quat_apply(anchor_heading_quat, rel_goal_pos, w_last=True)
        goal_pos_w[:, 2] = self._ground_resting_object_center_z(env_ids)

        _, _, anchor_heading = get_euler_xyz(anchor_heading_quat, w_last=True)
        goal_heading = anchor_heading + goal_yaw_rel
        zeros = torch.zeros_like(goal_heading)
        goal_quat_w = quat_from_euler_xyz(zeros, zeros, goal_heading)
        goal_rot_mat_w = quaternion_to_matrix(goal_quat_w, w_last=True)
        goal_rot6d_w = goal_rot_mat_w[..., :2].reshape(goal_rot_mat_w.shape[0], 6)
        return goal_pos_w, goal_quat_w, goal_rot6d_w

    def _apply_manual_goal_world_from_command(
        self,
        env_ids: torch.Tensor,
        *,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
    ) -> None:
        # Materialize the fixed pickup-frame command into a world-space goal using
        # the currently latched pickup anchor (or a preview anchor during reset).
        if self.manual_goal_object_pos_w is None or self.manual_goal_object_rot6d_w is None:
            return
        if env_ids.numel() == 0:
            return
        goal_pos_w, _goal_quat_w, goal_rot6d_w = self._goal_world_from_command(
            env_ids,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )
        self.manual_goal_enabled = True
        self.manual_goal_object_pos_w[env_ids] = goal_pos_w
        self.manual_goal_object_rot6d_w[env_ids] = goal_rot6d_w

    def _sample_external_object_goal_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sparse_goal_cfg is None:
            raise RuntimeError("Sparse goal config is not initialized.")

        num_samples = env_ids.numel()
        progress = self._sparse_goal_curriculum_progress()
        pos_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_pos_local_min,
            name="external_goal_pos_local_min",
            start_values=self._sparse_goal_cfg.external_goal_pos_local_min_start,
            alpha=progress,
        )
        pos_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_pos_local_max,
            name="external_goal_pos_local_max",
            start_values=self._sparse_goal_cfg.external_goal_pos_local_max_start,
            alpha=progress,
        )
        pos_lo = torch.minimum(pos_min, pos_max)
        pos_hi = torch.maximum(pos_min, pos_max)
        local_pos = pos_lo.unsqueeze(0) + (pos_hi - pos_lo).unsqueeze(0) * torch.rand(
            (num_samples, 3), device=self.device
        )
        goal_pos_w = self._get_env_offsets(env_ids) + local_pos

        rpy_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_rpy_min,
            name="external_goal_rpy_min",
            start_values=self._sparse_goal_cfg.external_goal_rpy_min_start,
            alpha=progress,
        )
        rpy_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_rpy_max,
            name="external_goal_rpy_max",
            start_values=self._sparse_goal_cfg.external_goal_rpy_max_start,
            alpha=progress,
        )
        rpy_lo = torch.minimum(rpy_min, rpy_max)
        rpy_hi = torch.maximum(rpy_min, rpy_max)
        rpy = rpy_lo.unsqueeze(0) + (rpy_hi - rpy_lo).unsqueeze(0) * torch.rand((num_samples, 3), device=self.device)
        goal_quat_w = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        return goal_pos_w, goal_quat_w

    def _sample_external_goal_command(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sparse_goal_cfg is None:
            raise RuntimeError("Sparse goal config is not initialized.")

        num_samples = env_ids.numel()
        progress = self._sparse_goal_curriculum_progress()
        pos_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_pos_local_min,
            name="external_goal_pos_local_min",
            start_values=self._sparse_goal_cfg.external_goal_pos_local_min_start,
            alpha=progress,
        )
        pos_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_pos_local_max,
            name="external_goal_pos_local_max",
            start_values=self._sparse_goal_cfg.external_goal_pos_local_max_start,
            alpha=progress,
        )
        pos_lo = torch.minimum(pos_min, pos_max)
        pos_hi = torch.maximum(pos_min, pos_max)
        goal_xy_rel = pos_lo[:2].unsqueeze(0) + (pos_hi[:2] - pos_lo[:2]).unsqueeze(0) * torch.rand(
            (num_samples, 2), device=self.device
        )

        rpy_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_rpy_min,
            name="external_goal_rpy_min",
            start_values=self._sparse_goal_cfg.external_goal_rpy_min_start,
            alpha=progress,
        )
        rpy_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.external_goal_rpy_max,
            name="external_goal_rpy_max",
            start_values=self._sparse_goal_cfg.external_goal_rpy_max_start,
            alpha=progress,
        )
        yaw_lo = torch.minimum(rpy_min[2], rpy_max[2])
        yaw_hi = torch.maximum(rpy_min[2], rpy_max[2])
        goal_yaw_rel = yaw_lo + (yaw_hi - yaw_lo) * torch.rand((num_samples,), device=self.device)
        return goal_xy_rel, goal_yaw_rel

    def _sample_carry_extension_object_goal_pose_w(
        self,
        env_ids: torch.Tensor,
        clip_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sparse_goal_cfg is None:
            raise RuntimeError("Sparse goal config is not initialized.")

        num_samples = env_ids.numel()
        base_pos_w, base_quat_w = self._sample_clip_based_object_goal_pose_w(env_ids, clip_lengths)
        progress = self._carry_extension_curriculum_progress()

        pos_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.carry_extension_pos_local_min,
            name="carry_extension_pos_local_min",
            start_values=self._sparse_goal_cfg.carry_extension_pos_local_min_start,
            alpha=progress,
        )
        pos_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.carry_extension_pos_local_max,
            name="carry_extension_pos_local_max",
            start_values=self._sparse_goal_cfg.carry_extension_pos_local_max_start,
            alpha=progress,
        )
        pos_lo = torch.minimum(pos_min, pos_max)
        pos_hi = torch.maximum(pos_min, pos_max)
        local_pos = pos_lo.unsqueeze(0) + (pos_hi - pos_lo).unsqueeze(0) * torch.rand(
            (num_samples, 3), device=self.device
        )
        goal_pos_w = base_pos_w + quat_apply(base_quat_w, local_pos, w_last=True)

        rpy_min = self._goal_vec3_interp(
            self._sparse_goal_cfg.carry_extension_rpy_min,
            name="carry_extension_rpy_min",
            start_values=self._sparse_goal_cfg.carry_extension_rpy_min_start,
            alpha=progress,
        )
        rpy_max = self._goal_vec3_interp(
            self._sparse_goal_cfg.carry_extension_rpy_max,
            name="carry_extension_rpy_max",
            start_values=self._sparse_goal_cfg.carry_extension_rpy_max_start,
            alpha=progress,
        )
        rpy_lo = torch.minimum(rpy_min, rpy_max)
        rpy_hi = torch.maximum(rpy_min, rpy_max)
        rpy = rpy_lo.unsqueeze(0) + (rpy_hi - rpy_lo).unsqueeze(0) * torch.rand((num_samples, 3), device=self.device)
        delta_quat_w = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        goal_quat_w = quat_mul(base_quat_w, delta_quat_w, w_last=True)
        return goal_pos_w, goal_quat_w

    def _current_external_goal_prob(self) -> float:
        if self._sparse_goal_cfg is None:
            return 0.0
        if self._env.is_evaluating:
            if self._sparse_goal_cfg.eval_external_goal_prob is not None:
                return self._clamp01(float(self._sparse_goal_cfg.eval_external_goal_prob))
            return self._clamp01(float(self._sparse_goal_cfg.external_goal_prob_end))

        iter_value = self._iteration_schedule_value(
            float(self._sparse_goal_cfg.external_goal_prob_start),
            float(self._sparse_goal_cfg.external_goal_prob_end),
            start_iter=self._sparse_goal_cfg.external_goal_prob_start_iter,
            end_iter=self._sparse_goal_cfg.external_goal_prob_end_iter,
        )
        if iter_value is not None:
            return self._clamp01(iter_value)

        prob_start = float(self._sparse_goal_cfg.external_goal_prob_start)
        prob_end = float(self._sparse_goal_cfg.external_goal_prob_end)
        ramp_resets = max(0, int(self._sparse_goal_cfg.external_goal_prob_ramp_resets))
        if ramp_resets <= 0:
            alpha = 1.0
        else:
            alpha = min(float(self._sparse_goal_reset_counter) / float(ramp_resets), 1.0)
        return self._clamp01(prob_start + (prob_end - prob_start) * alpha)

    def _current_carry_extension_prob(self) -> float:
        if self._sparse_goal_cfg is None:
            return 0.0
        if self._env.is_evaluating:
            if self._sparse_goal_cfg.eval_carry_extension_prob is not None:
                return self._clamp01(float(self._sparse_goal_cfg.eval_carry_extension_prob))
            return self._clamp01(float(self._sparse_goal_cfg.carry_extension_prob_end))

        iter_value = self._iteration_schedule_value(
            float(self._sparse_goal_cfg.carry_extension_prob_start),
            float(self._sparse_goal_cfg.carry_extension_prob_end),
            start_iter=self._sparse_goal_cfg.carry_extension_prob_start_iter,
            end_iter=self._sparse_goal_cfg.carry_extension_prob_end_iter,
        )
        if iter_value is not None:
            return self._clamp01(iter_value)

        prob_start = float(self._sparse_goal_cfg.carry_extension_prob_start)
        prob_end = float(self._sparse_goal_cfg.carry_extension_prob_end)
        ramp_cfg = self._sparse_goal_cfg.carry_extension_prob_ramp_resets
        if ramp_cfg is None:
            ramp_cfg = self._sparse_goal_cfg.external_goal_prob_ramp_resets
        ramp_resets = max(0, int(ramp_cfg))
        if ramp_resets <= 0:
            alpha = 1.0
        else:
            alpha = min(float(self._sparse_goal_reset_counter) / float(ramp_resets), 1.0)
        return self._clamp01(prob_start + (prob_end - prob_start) * alpha)

    def _current_command_only_env_prob(self) -> float:
        if self._sparse_goal_cfg is None:
            return 0.0
        if self._env.is_evaluating:
            if self._sparse_goal_cfg.eval_command_only_env_prob is not None:
                return self._clamp01(float(self._sparse_goal_cfg.eval_command_only_env_prob))
            return self._clamp01(float(self._sparse_goal_cfg.command_only_env_prob_end))

        iter_value = self._iteration_schedule_value(
            float(self._sparse_goal_cfg.command_only_env_prob_start),
            float(self._sparse_goal_cfg.command_only_env_prob_end),
            start_iter=self._sparse_goal_cfg.command_only_env_prob_start_iter,
            end_iter=self._sparse_goal_cfg.command_only_env_prob_end_iter,
        )
        if iter_value is not None:
            return self._clamp01(iter_value)

        prob_start = float(self._sparse_goal_cfg.command_only_env_prob_start)
        prob_end = float(self._sparse_goal_cfg.command_only_env_prob_end)
        ramp_cfg = self._sparse_goal_cfg.command_only_env_prob_ramp_resets
        if ramp_cfg is None:
            ramp_cfg = self._sparse_goal_cfg.external_goal_prob_ramp_resets
        ramp_resets = max(0, int(ramp_cfg))
        if ramp_resets <= 0:
            alpha = 1.0
        else:
            alpha = min(float(self._sparse_goal_reset_counter) / float(ramp_resets), 1.0)
        return self._clamp01(prob_start + (prob_end - prob_start) * alpha)

    def _update_sparse_object_goals_on_reset(self, env_ids: torch.Tensor, clip_lengths: torch.Tensor) -> None:
        if not self._sparse_goal_curriculum_enabled or self._sparse_goal_cfg is None:
            return
        if env_ids.numel() == 0:
            return
        if not self.motion.has_object:
            return
        if (
            self.manual_goal_object_pos_w is None
            or self.manual_goal_object_rot6d_w is None
            or self.manual_goal_xy_rel is None
            or self.manual_goal_yaw_rel is None
        ):
            return

        self.manual_goal_enabled = True
        clip_goal_pos_w, clip_goal_quat_w = self._sample_clip_based_object_goal_pose_w(env_ids, clip_lengths)
        clip_pickup_anchor_pos_w, clip_pickup_anchor_quat_w = self._sample_clip_pickup_anchor_pose_w(env_ids)
        goal_xy_rel, goal_yaw_rel = self._goal_command_from_world(
            clip_goal_pos_w,
            clip_goal_quat_w,
            anchor_pos_w=clip_pickup_anchor_pos_w,
            anchor_quat_w=clip_pickup_anchor_quat_w,
        )
        # Keep a preview world goal for reset/debug buffers; once pickup is latched,
        # `_apply_manual_goal_world_from_command` rematerializes it from the actual anchor.
        preview_goal_pos_w = clip_goal_pos_w.clone()
        preview_goal_quat_w = clip_goal_quat_w.clone()

        p_command = self._current_command_only_env_prob()
        p_carry = self._current_carry_extension_prob()
        p_ext = self._current_external_goal_prob()
        p_command = self._clamp01(p_command)
        p_carry = min(self._clamp01(p_carry), p_command)
        p_ext = self._clamp01(min(p_ext, max(p_command - p_carry, 0.0)))
        self._command_only_env_prob = p_command
        self._sparse_goal_external_prob = p_carry + p_ext
        goal_selector = torch.rand(env_ids.numel(), device=self.device)
        carry_extension_mask = goal_selector < p_carry
        external_mask = (~carry_extension_mask) & (goal_selector < (p_carry + p_ext))
        command_clip_mask = (~carry_extension_mask) & (~external_mask) & (goal_selector < p_command)
        command_only_mask = carry_extension_mask | external_mask | command_clip_mask
        non_clip_mask = carry_extension_mask | external_mask
        if carry_extension_mask.any():
            carry_pos_w, carry_quat_w = self._sample_carry_extension_object_goal_pose_w(
                env_ids[carry_extension_mask],
                clip_lengths[carry_extension_mask],
            )
            carry_xy_rel, carry_yaw_rel = self._goal_command_from_world(
                carry_pos_w,
                carry_quat_w,
                anchor_pos_w=clip_pickup_anchor_pos_w[carry_extension_mask],
                anchor_quat_w=clip_pickup_anchor_quat_w[carry_extension_mask],
            )
            goal_xy_rel[carry_extension_mask] = carry_xy_rel
            goal_yaw_rel[carry_extension_mask] = carry_yaw_rel
            preview_goal_pos_w[carry_extension_mask] = carry_pos_w
            preview_goal_quat_w[carry_extension_mask] = carry_quat_w
        if external_mask.any():
            ext_xy_rel, ext_yaw_rel = self._sample_external_goal_command(env_ids[external_mask])
            goal_xy_rel[external_mask] = ext_xy_rel
            goal_yaw_rel[external_mask] = ext_yaw_rel
            ext_preview_pos_w, ext_preview_quat_w, _ext_preview_rot6d_w = self._goal_world_from_command(
                env_ids[external_mask],
                anchor_pos_w=clip_pickup_anchor_pos_w[external_mask],
                anchor_quat_w=clip_pickup_anchor_quat_w[external_mask],
                goal_xy_rel=ext_xy_rel,
                goal_yaw_rel=ext_yaw_rel,
            )
            preview_goal_pos_w[external_mask] = ext_preview_pos_w
            preview_goal_quat_w[external_mask] = ext_preview_quat_w

        clip_goal_rot_mat = quaternion_to_matrix(clip_goal_quat_w, w_last=True)
        clip_goal_rot6d_w = clip_goal_rot_mat[..., :2].reshape(clip_goal_rot_mat.shape[0], 6)
        goal_rot_mat = quaternion_to_matrix(preview_goal_quat_w, w_last=True)
        goal_rot6d_w = goal_rot_mat[..., :2].reshape(goal_rot_mat.shape[0], 6)
        if self.clip_goal_object_pos_w is not None:
            self.clip_goal_object_pos_w[env_ids] = clip_goal_pos_w
        if self.clip_goal_object_rot6d_w is not None:
            self.clip_goal_object_rot6d_w[env_ids] = clip_goal_rot6d_w
        if self.base_goal_object_pos_w is not None:
            self.base_goal_object_pos_w[env_ids] = preview_goal_pos_w
        if self.base_goal_object_rot6d_w is not None:
            self.base_goal_object_rot6d_w[env_ids] = goal_rot6d_w
        if self.base_goal_is_external is not None:
            self.base_goal_is_external[env_ids] = non_clip_mask
        self.manual_goal_xy_rel[env_ids] = goal_xy_rel
        self.manual_goal_yaw_rel[env_ids, 0] = goal_yaw_rel
        self.manual_goal_object_pos_w[env_ids] = preview_goal_pos_w
        self.manual_goal_object_rot6d_w[env_ids] = goal_rot6d_w
        if self.manual_goal_is_external is not None:
            self.manual_goal_is_external[env_ids] = non_clip_mask
        if self.command_only_env_mask is not None:
            self.command_only_env_mask[env_ids] = command_only_mask
        self._command_only_env_fraction_last_reset = float(command_only_mask.float().mean().item())
        self._sparse_goal_external_fraction_last_reset = float(non_clip_mask.float().mean().item())

        if not self._env.is_evaluating:
            self._sparse_goal_reset_counter += int(env_ids.numel())

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def get_sparse_goal_external_mask(self) -> torch.Tensor:
        if (
            not self.motion.has_object
            or not self.manual_goal_enabled
            or self.manual_goal_object_pos_w is None
            or self.manual_goal_object_rot6d_w is None
        ):
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        if self.manual_goal_is_external is not None:
            external_mask = self.manual_goal_is_external.clone()
        else:
            external_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        if self.clip_goal_object_pos_w is None or self.clip_goal_object_rot6d_w is None:
            return external_mask

        pos_diff = torch.any(torch.abs(self.manual_goal_object_pos_w - self.clip_goal_object_pos_w) > 1.0e-6, dim=-1)
        rot_diff = torch.any(
            torch.abs(self.manual_goal_object_rot6d_w - self.clip_goal_object_rot6d_w) > 1.0e-6, dim=-1
        )
        return external_mask | pos_diff | rot_diff

    def get_command_only_env_mask(self) -> torch.Tensor:
        if self.command_only_env_mask is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        return self.command_only_env_mask.clone()

    def _manual_goal_anchor_pose_w(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.pickup_anchor_set is None
            or self.pickup_anchor_root_pos_w is None
            or self.pickup_anchor_root_quat_w is None
        ):
            return self.robot_root_pos_w[env_ids], self.robot_root_quat_w[env_ids]

        anchor_pos_w = self.pickup_anchor_root_pos_w[env_ids].clone()
        anchor_quat_w = self.pickup_anchor_root_quat_w[env_ids].clone()
        missing_anchor = ~self.pickup_anchor_set[env_ids]
        if missing_anchor.any():
            anchor_pos_w[missing_anchor] = self.robot_root_pos_w[env_ids][missing_anchor]
            anchor_quat_w[missing_anchor] = self.robot_root_quat_w[env_ids][missing_anchor]
        return anchor_pos_w, anchor_quat_w

    def _ground_resting_object_center_z(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_offsets = self._get_env_offsets(env_ids)
        object_size = self._resolved_object_size_for_env_ids(env_ids)
        if object_size.ndim == 1:
            object_size = object_size.unsqueeze(0)
        return env_offsets[:, 2] + 0.5 * object_size[:, 2]

    def _default_object_urdf_path(self) -> str:
        obj_cfg = getattr(self._env.robot_config, "object", None)
        if obj_cfg is None:
            return ""
        urdf_path = str(getattr(obj_cfg, "object_urdf_path", "") or "").strip()
        if not urdf_path.lower().endswith(".urdf"):
            return ""
        return urdf_path

    def _resolved_object_size_for_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        object_size = self.object_size[env_ids].clone()
        if object_size.numel() == 0:
            return object_size

        clip_object_urdfs = list(getattr(self.motion, "clip_object_urdf_paths", []))
        default_urdf = self._default_object_urdf_path()
        if not clip_object_urdfs and not default_urdf:
            return object_size

        env_ids_cpu = env_ids.detach().cpu().tolist()
        for local_idx, env_id in enumerate(env_ids_cpu):
            try:
                clip_idx = int(self.clip_ids[int(env_id)].item())
            except Exception:
                clip_idx = -1
            object_urdf = ""
            if 0 <= clip_idx < len(clip_object_urdfs):
                object_urdf = str(clip_object_urdfs[clip_idx]).strip()
            if not object_urdf:
                object_urdf = default_urdf
            if not object_urdf:
                continue
            extents = load_urdf_geometry_extents(object_urdf)
            if extents is None:
                continue
            object_size[local_idx] = torch.tensor(extents, device=self.device, dtype=object_size.dtype)
        return object_size

    def _update_manual_goal_override(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.manual_goal_override_enabled:
            return
        if (
            self.manual_goal_object_pos_w is None
            or self.manual_goal_object_rot6d_w is None
            or self.manual_goal_xy_rel is None
            or self.manual_goal_yaw_rel is None
        ):
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        anchor_pos_w, anchor_quat_w = self._manual_goal_anchor_pose_w(env_ids)
        self._apply_manual_goal_world_from_command(
            env_ids,
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )
        if self.manual_goal_is_external is not None:
            self.manual_goal_is_external[env_ids] = True

    def _get_env_offsets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        terrain_state = None
        terrain_manager = getattr(self._env, "terrain_manager", None)
        if terrain_manager is not None:
            try:
                terrain_state = terrain_manager.get_state("locomotion_terrain")
            except Exception:
                terrain_state = None
        base = getattr(terrain_state, "env_origins", None)
        if base is None:
            base = self._env.simulator.scene.env_origins
        if self._clip_terrain_offsets is None or not hasattr(self, "clip_ids"):
            return base if env_ids is None else base[env_ids]

        clip_ids = self.clip_ids if env_ids is None else self.clip_ids[env_ids]
        clip_offsets = self._clip_terrain_offsets[clip_ids]
        if self._terrain_row_ids is not None:
            row_ids = self._terrain_row_ids if env_ids is None else self._terrain_row_ids[env_ids]
            if self._clip_terrain_offsets_by_row is not None:
                clip_offsets = self._clip_terrain_offsets_by_row[row_ids, clip_ids]
            elif self._terrain_row_stride > 0.0:
                row_offsets = torch.zeros_like(clip_offsets)
                row_offsets[:, 1] = row_ids.to(row_offsets.dtype) * self._terrain_row_stride
                clip_offsets = clip_offsets + row_offsets

        if self.motion_cfg.pair_terrain_with_motion:
            return clip_offsets

        if base.device != clip_offsets.device:
            base = base.to(clip_offsets.device)
        return base + clip_offsets

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._raw_motion_joint_pos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._raw_motion_joint_vel()

    @property
    def body_pos_w(self) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w()[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w()[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_lin_vel_w()[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_ang_vel_w()[:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_pos_w(self) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    @property
    def ref_quat_w(self) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def root_pos_w(self) -> torch.Tensor:
        pos = self._raw_motion_body_pos_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    @property
    def root_quat_w(self) -> torch.Tensor:
        quat = self._raw_motion_body_quat_w()[:, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_lin_vel_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        vel = self._raw_motion_body_ang_vel_w()[:, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        pos = self._raw_motion_object_pos_w()
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._get_env_offsets()

    @property
    def object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        quat = self._raw_motion_object_quat_w()
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        vel = self._raw_motion_object_lin_vel_w()
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def object_size(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        motion_idx = self._get_motion_indices(self.time_steps)
        return self.motion.object_size[motion_idx]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        indices = self._get_active_object_indices()
        return self._env.simulator.all_root_states[indices][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
            quat[:, 3] = 1.0
            return quat
        indices = self._get_active_object_indices()
        return self._env.simulator.all_root_states[indices][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        indices = self._get_active_object_indices()
        return self._env.simulator.all_root_states[indices][:, 7:10]

    @property
    def simulator_object_ang_vel_w(self) -> torch.Tensor:
        if not self.motion.has_object:
            return torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        indices = self._get_active_object_indices()
        return self._env.simulator.all_root_states[indices][:, 10:13]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self._terrain_row_ids is not None:
            self._terrain_row_ids.zero_()
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0
        self._align_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._align_quat[:, 3] = 1.0
        self._align_pos = torch.zeros(self.num_envs, 3, device=self.device)
        num_regions = len(_CONTACT_PRIOR_REGION_NAMES)
        self._contact_prior_total_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_contact_sum = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_force_mean = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_force_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_position_mean = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._contact_prior_position_count = torch.zeros(
            (self.motion.num_clips, _CONTACT_PRIOR_PHASE_COUNT, num_regions),
            device=self.device,
            dtype=torch.float32,
        )

        if self.num_future_steps > 0 and self.target_pose_type is not None:
            self.future_target_poses = torch.zeros(
                self.num_envs,
                self.num_future_steps * self.num_obs_per_target_pose,
                device=self.device,
            )

        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

        if self._clip_success_counts is not None:
            self._clip_success_counts.zero_()
        if self._clip_total_counts is not None:
            self._clip_total_counts.zero_()
        if self.clip_weighting_strategy == "success_rate_adaptive" and self._base_clip_weights is not None:
            self._raw_clip_sampling_weights = self._base_clip_weights.clone()
        self._refresh_current_clip_sampling_weights()
        self._sparse_goal_reset_counter = 0
        self._command_only_env_prob = 0.0
        self._command_only_env_fraction_last_reset = 0.0
        self._sparse_goal_external_prob = 0.0
        self._sparse_goal_external_fraction_last_reset = 0.0
        if self.command_only_env_mask is not None:
            self.command_only_env_mask.zero_()
        if self.manual_goal_is_external is not None:
            self.manual_goal_is_external.zero_()
        if self.clip_goal_object_pos_w is not None:
            self.clip_goal_object_pos_w.zero_()
        if self.clip_goal_object_rot6d_w is not None:
            self.clip_goal_object_rot6d_w.zero_()
        if self.base_goal_object_pos_w is not None:
            self.base_goal_object_pos_w.zero_()
        if self.base_goal_object_rot6d_w is not None:
            self.base_goal_object_rot6d_w.zero_()
        if self.base_goal_is_external is not None:
            self.base_goal_is_external.zero_()
        if self.manual_goal_xy_rel is not None:
            self.manual_goal_xy_rel.zero_()
        if self.manual_goal_yaw_rel is not None:
            self.manual_goal_yaw_rel.zero_()
        if self.pickup_anchor_set is not None:
            self.pickup_anchor_set.zero_()
        if self.pickup_anchor_root_pos_w is not None:
            self.pickup_anchor_root_pos_w.zero_()
        if self.pickup_anchor_root_quat_w is not None:
            self.pickup_anchor_root_quat_w.zero_()
            self.pickup_anchor_root_quat_w[:, 3] = 1.0
        if self.pickup_object_rel_z_baseline is not None:
            self.pickup_object_rel_z_baseline.zero_()
        if self.pickup_consecutive_counter is not None:
            self.pickup_consecutive_counter.zero_()
        if self._runtime_default_pose_prepend_active is not None:
            self._runtime_default_pose_prepend_active.zero_()
        if self._runtime_default_pose_prepend_step is not None:
            self._runtime_default_pose_prepend_step.zero_()

    def _update_motion_alignment(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        clip_ids = self.clip_ids[env_ids]
        clip_offsets = self.motion.clip_offsets[clip_ids]
        motion_root_quat = self.motion.body_quat_w[clip_offsets, 0]
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        yaw_delta = self._init_root_yaw - motion_yaw
        zeros = torch.zeros_like(yaw_delta)
        align_quat = quat_from_euler_xyz(zeros, zeros, yaw_delta)
        self._align_quat[env_ids] = align_quat

        motion_root_pos = self.motion.body_pos_w[clip_offsets, 0]
        env_offsets = self._get_env_offsets(env_ids)
        desired_root_pos = env_offsets + self._init_root_pos
        aligned_root_pos = quat_apply(align_quat, motion_root_pos, w_last=True)
        self._align_pos[env_ids] = desired_root_pos - aligned_root_pos

    def _apply_motion_alignment_pos(self, pos: torch.Tensor) -> torch.Tensor:
        align_quat = self._align_quat
        align_pos = self._align_pos
        if pos.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, pos.shape[1], -1)
            align_pos = align_pos[:, None, :]
        return quat_apply(align_quat, pos, w_last=True) + align_pos

    def _apply_motion_alignment_vec(self, vec: torch.Tensor) -> torch.Tensor:
        align_quat = self._align_quat
        if vec.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, vec.shape[1], -1)
        return quat_apply(align_quat, vec, w_last=True)

    def _apply_motion_alignment_quat(self, quat: torch.Tensor) -> torch.Tensor:
        align_quat = self._align_quat
        if quat.ndim == 3:
            align_quat = align_quat[:, None, :].expand(-1, quat.shape[1], -1)
        return quat_mul(align_quat, quat, w_last=True)

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        # Human (robot) tracking metrics.
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        # Object co-tracking metrics (separate from human tracking metrics).
        if self.motion.has_object:
            self.metrics["motion/error_object_ref_pos"] = torch.norm(
                self.object_pos_w - self.simulator_object_pos_w, dim=-1
            )
            self.metrics["motion/error_object_ref_rot"] = quat_error_magnitude(
                self.object_quat_w, self.simulator_object_quat_w
            )
            self.metrics["motion/error_object_ref_lin_vel"] = torch.norm(
                self.object_lin_vel_w - self.simulator_object_lin_vel_w, dim=-1
            )
        else:
            zeros = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            self.metrics["motion/error_object_ref_pos"] = zeros
            self.metrics["motion/error_object_ref_rot"] = zeros
            self.metrics["motion/error_object_ref_lin_vel"] = zeros

        self.metrics["motion/reset_start_at_timestep_zero_prob"] = torch.full(
            (self.num_envs,),
            float(self._current_start_at_timestep_zero_prob()),
            device=self.device,
            dtype=torch.float32,
        )
        self.metrics["motion/reset_freeze_at_timestep_zero_prob"] = torch.full(
            (self.num_envs,),
            float(self._current_freeze_at_timestep_zero_prob()),
            device=self.device,
            dtype=torch.float32,
        )
        clean_prob = self._current_clean_group_probability()
        if clean_prob is not None and self._clean_clip_mask is not None and self._clip_sampling_weights is not None:
            clean_weight = float(self._clip_sampling_weights[self._clean_clip_mask].sum().item())
            self.metrics["motion/clean_clip_target_prob"] = torch.full(
                (self.num_envs,),
                float(clean_prob),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["motion/clean_clip_sample_weight"] = torch.full(
                (self.num_envs,),
                clean_weight,
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["motion/noisy_clip_sample_weight"] = torch.full(
                (self.num_envs,),
                max(0.0, 1.0 - clean_weight),
                device=self.device,
                dtype=torch.float32,
            )

        if self._sparse_goal_curriculum_enabled:
            self.metrics["goal/training_iteration"] = torch.full(
                (self.num_envs,),
                float(self._training_iteration or 0),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["goal/command_only_env_prob"] = torch.full(
                (self.num_envs,),
                float(self._command_only_env_prob),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["goal/command_only_env_fraction_last_reset"] = torch.full(
                (self.num_envs,),
                float(self._command_only_env_fraction_last_reset),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["goal/external_prob"] = torch.full(
                (self.num_envs,),
                float(self._sparse_goal_external_prob),
                device=self.device,
                dtype=torch.float32,
            )
            self.metrics["goal/external_fraction_last_reset"] = torch.full(
                (self.num_envs,),
                float(self._sparse_goal_external_fraction_last_reset),
                device=self.device,
                dtype=torch.float32,
            )
            progress = self._sparse_goal_curriculum_progress()
            self.metrics["goal/external_prob_curriculum_progress"] = torch.full(
                (self.num_envs,),
                float(progress),
                device=self.device,
                dtype=torch.float32,
            )
            command_progress = self._command_only_env_curriculum_progress()
            self.metrics["goal/command_only_env_prob_curriculum_progress"] = torch.full(
                (self.num_envs,),
                float(command_progress),
                device=self.device,
                dtype=torch.float32,
            )
            prior_occupancy, prior_force, _, prior_confidence, prior_valid = self.get_contact_prior_targets()
            self.metrics["goal/contact_prior_confidence"] = prior_confidence
            self.metrics["goal/contact_prior_valid"] = prior_valid.to(dtype=torch.float32)
            for region_idx, region_name in enumerate(_CONTACT_PRIOR_REGION_NAMES):
                metric_name = region_name.replace("left_", "l_").replace("right_", "r_")
                self.metrics[f"goal/contact_prior_{metric_name}_occupancy"] = prior_occupancy[:, region_idx]
                self.metrics[f"goal/contact_prior_{metric_name}_force"] = prior_force[:, region_idx]
            if self._sparse_goal_cfg is not None:
                pos_min = self._goal_vec3_interp(
                    self._sparse_goal_cfg.external_goal_pos_local_min,
                    name="external_goal_pos_local_min",
                    start_values=self._sparse_goal_cfg.external_goal_pos_local_min_start,
                    alpha=progress,
                )
                pos_max = self._goal_vec3_interp(
                    self._sparse_goal_cfg.external_goal_pos_local_max,
                    name="external_goal_pos_local_max",
                    start_values=self._sparse_goal_cfg.external_goal_pos_local_max_start,
                    alpha=progress,
                )
                rpy_min = self._goal_vec3_interp(
                    self._sparse_goal_cfg.external_goal_rpy_min,
                    name="external_goal_rpy_min",
                    start_values=self._sparse_goal_cfg.external_goal_rpy_min_start,
                    alpha=progress,
                )
                rpy_max = self._goal_vec3_interp(
                    self._sparse_goal_cfg.external_goal_rpy_max,
                    name="external_goal_rpy_max",
                    start_values=self._sparse_goal_cfg.external_goal_rpy_max_start,
                    alpha=progress,
                )
                pos_half_extent = 0.5 * torch.abs(pos_max - pos_min)
                yaw_half_extent = 0.5 * abs(float(rpy_max[2] - rpy_min[2]))
                self.metrics["goal/external_pos_range_x_half"] = torch.full(
                    (self.num_envs,),
                    float(pos_half_extent[0]),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.metrics["goal/external_pos_range_y_half"] = torch.full(
                    (self.num_envs,),
                    float(pos_half_extent[1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                self.metrics["goal/external_yaw_range_half"] = torch.full(
                    (self.num_envs,),
                    yaw_half_extent,
                    device=self.device,
                    dtype=torch.float32,
                )

        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _configure_motion_terrain_pairs(self) -> None:
        self._clip_terrain_offsets = None
        self._clip_terrain_offsets_by_row = None
        self._terrain_row_ids = None
        self._terrain_row_stride = 0.0
        self._terrain_row_count = 0
        if not self.motion_cfg.pair_terrain_with_motion:
            return

        terrain_state = self._env.terrain_manager.get_state("locomotion_terrain")
        terrain = getattr(terrain_state, "terrain", None)
        tile_names = getattr(terrain, "obj_tile_names", []) if terrain is not None else []
        tile_offsets = getattr(terrain, "obj_tile_offsets", None) if terrain is not None else None
        tile_stride = getattr(terrain, "obj_tile_stride", None) if terrain is not None else None
        tile_rows = int(getattr(terrain, "obj_tile_rows", 0) or 0) if terrain is not None else 0

        if tile_names and tile_offsets is not None and tile_stride is not None and tile_rows > 0:
            if len(set(tile_names)) != len(tile_names):
                raise ValueError("Duplicate OBJ tile names detected; stems must be unique for pairing.")

            tile_offsets = np.asarray(tile_offsets, dtype=np.float32)
            if tile_offsets.shape[0] != len(tile_names):
                raise ValueError("OBJ tile offsets length does not match tile names.")
            stride = np.asarray(tile_stride, dtype=np.float32).reshape(-1)
            if stride.size < 2:
                raise ValueError("OBJ tile stride must provide at least X/Y spacing.")

            name_to_idx = {name: idx for idx, name in enumerate(tile_names)}
            missing = [clip_id for clip_id in self.motion.clip_ids if clip_id not in name_to_idx]
            if missing:
                raise ValueError(f"Missing terrain OBJ for clips: {missing}")

            clip_offsets = np.stack([tile_offsets[name_to_idx[clip_id]] for clip_id in self.motion.clip_ids], axis=0)
            row_offsets = np.repeat(clip_offsets[None, :, :], repeats=max(1, tile_rows), axis=0)
            if row_offsets.shape[0] > 1:
                row_offsets[:, :, 1] += np.arange(row_offsets.shape[0], dtype=np.float32)[:, None] * float(stride[1])
            self._clip_terrain_offsets = torch.tensor(clip_offsets, device=self.device, dtype=torch.float32)
            self._clip_terrain_offsets_by_row = torch.tensor(row_offsets, device=self.device, dtype=torch.float32)
            self._terrain_row_stride = float(stride[1])
            self._terrain_row_count = max(1, tile_rows)
            self._terrain_row_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            capacity = self._terrain_row_count * len(tile_names)
            if self.num_envs > capacity:
                raise ValueError(
                    "pair_terrain_with_motion requires terrain slots >= envs per rank "
                    f"(got num_envs={self.num_envs}, rows={self._terrain_row_count}, "
                    f"cols={len(tile_names)}, capacity={capacity}). "
                    "Increase terrain num_rows or reduce num_envs."
                )

            unused = [name for name in tile_names if name not in self.motion.clip_ids]
            if unused:
                logger.warning("Unused terrain OBJ tiles (no matching motion clip): {}", unused)

            logger.info("Motion/terrain pairing enabled for {} clips.", len(self.motion.clip_ids))
            return

        origin_grid = None
        if terrain is not None and hasattr(terrain, "env_origin_grid"):
            origin_grid = getattr(terrain, "env_origin_grid")
        elif terrain is not None and hasattr(terrain, "_env_origins"):
            origin_grid = getattr(terrain, "_env_origins")

        if origin_grid is None:
            raise ValueError(
                "pair_terrain_with_motion requires terrain tile metadata or a terrain origin grid. "
                "For OBJ pairing, set --terrain.terrain-term.obj-file-path to named OBJ tiles."
            )

        origin_grid_np = np.asarray(origin_grid, dtype=np.float32)
        if origin_grid_np.ndim != 3 or origin_grid_np.shape[2] < 3:
            raise ValueError(
                "Terrain origin grid must have shape (rows, cols, 3) to pair motion clips with terrain columns."
            )
        if origin_grid_np.shape[2] > 3:
            origin_grid_np = origin_grid_np[:, :, :3]

        num_rows, num_cols, _ = origin_grid_np.shape
        num_clips = len(self.motion.clip_ids)
        if num_cols < num_clips:
            raise ValueError(
                "pair_terrain_with_motion requires terrain columns >= motion clips "
                f"(got num_cols={num_cols}, num_clips={num_clips})."
            )

        clip_offsets_by_row = origin_grid_np[:, :num_clips, :]
        self._clip_terrain_offsets = torch.tensor(clip_offsets_by_row[0], device=self.device, dtype=torch.float32)
        self._clip_terrain_offsets_by_row = torch.tensor(clip_offsets_by_row, device=self.device, dtype=torch.float32)
        self._terrain_row_count = max(1, num_rows)
        self._terrain_row_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        capacity = self._terrain_row_count * num_clips
        if self.num_envs > capacity:
            raise ValueError(
                "pair_terrain_with_motion requires terrain slots >= envs per rank "
                f"(got num_envs={self.num_envs}, rows={self._terrain_row_count}, "
                f"clip_paired_cols={num_clips}, capacity={capacity}). "
                "Increase terrain num_rows or reduce num_envs."
            )

        if num_cols > num_clips:
            logger.warning(
                "Terrain has more columns ({}) than motion clips ({}); extra columns are unused for pairing.",
                num_cols,
                num_clips,
            )

        logger.info(
            "Motion/terrain pairing enabled for {} clips using terrain origin-grid column order.",
            len(self.motion.clip_ids),
        )

    def _configure_target_pose_settings(self) -> None:
        self.num_future_steps = int(self.motion_cfg.num_future_steps)
        self.target_pose_type = self.motion_cfg.target_pose_type
        self.num_obs_per_target_pose = 0
        self.future_target_poses: torch.Tensor | None = None

        if self.num_future_steps <= 0:
            return
        if self.target_pose_type is None:
            raise ValueError("target_pose_type must be set when num_future_steps > 0.")

        include_time = self._target_pose_includes_time(self.target_pose_type)
        num_bodies = len(self.motion_cfg.body_names_to_track)
        self.num_obs_per_target_pose = num_bodies * 18 + (1 if include_time else 0)

    def _target_pose_includes_time(self, target_pose_type: str) -> bool:
        if target_pose_type == "max-coords-future-rel":
            return False
        if target_pose_type == "max-coords-future-rel-with-time":
            return True
        raise ValueError(f"Unknown target_pose_type '{target_pose_type}'.")

    def _update_future_target_poses(self) -> None:
        if self.num_future_steps <= 0 or self.target_pose_type is None:
            return
        if self.future_target_poses is None:
            return
        self.future_target_poses[:] = self._compute_future_target_poses(
            num_future_steps=self.num_future_steps,
            target_pose_type=self.target_pose_type,
        )

    def _compute_future_target_poses(self, num_future_steps: int, target_pose_type: str) -> torch.Tensor:
        include_time = self._target_pose_includes_time(target_pose_type)

        time_offsets = torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
        future_steps = self.time_steps.unsqueeze(1) + time_offsets.unsqueeze(0)
        max_steps = self._current_clip_lengths().unsqueeze(1) - 1
        future_steps = torch.minimum(future_steps, max_steps)

        times = (future_steps - self.time_steps.unsqueeze(1)).to(dtype=torch.float32) * self._env.dt
        future_steps_global = self._get_motion_indices(future_steps)

        target_body_pos = (
            self.motion.body_pos_w[future_steps_global][:, :, self.tracked_body_indexes]
            + self._get_env_offsets()[:, None, None, :]
        )
        target_body_rot = self.motion.body_quat_w[future_steps_global][:, :, self.tracked_body_indexes]

        reference_body_pos = target_body_pos.roll(shifts=1, dims=1)
        reference_body_pos[:, 0] = self.body_pos_w
        reference_body_rot = target_body_rot.roll(shifts=1, dims=1)
        reference_body_rot[:, 0] = self.body_quat_w

        reference_root_pos = reference_body_pos[:, :, 0, :]
        reference_root_rot = reference_body_rot[:, :, 0, :]

        heading_quat = yaw_quat(reference_root_rot, w_last=True)
        heading_inv = quat_inverse(heading_quat, w_last=True)
        heading_inv = heading_inv.unsqueeze(2).expand(-1, -1, target_body_pos.shape[2], -1)

        target_rel_body_pos = target_body_pos - reference_body_pos
        target_body_pos_rel_root = target_body_pos - reference_root_pos.unsqueeze(2)

        flat_heading_inv = heading_inv.reshape(-1, 4)
        flat_rel_body_pos = target_rel_body_pos.reshape(-1, 3)
        flat_body_pos = target_body_pos_rel_root.reshape(-1, 3)

        flat_rel_body_pos = quat_apply(flat_heading_inv, flat_rel_body_pos, w_last=True)
        flat_body_pos = quat_apply(flat_heading_inv, flat_body_pos, w_last=True)

        rel_body_pos = flat_rel_body_pos.reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 3
        )
        body_pos = flat_body_pos.reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 3
        )

        rel_body_rot = quat_mul(
            quat_conjugate(reference_body_rot, w_last=True),
            target_body_rot,
            w_last=True,
        )
        body_rot = quat_mul(heading_inv, target_body_rot, w_last=True)

        rel_body_rot_mat = quaternion_to_matrix(rel_body_rot.reshape(-1, 4), w_last=True)
        body_rot_mat = quaternion_to_matrix(body_rot.reshape(-1, 4), w_last=True)

        rel_body_rot_obs = rel_body_rot_mat[..., :2].reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 6
        )
        body_rot_obs = body_rot_mat[..., :2].reshape(
            self.num_envs, num_future_steps, target_body_pos.shape[2] * 6
        )

        obs = torch.cat((rel_body_pos, body_pos, rel_body_rot_obs, body_rot_obs), dim=-1)

        if include_time:
            obs = torch.cat((obs, times.unsqueeze(-1)), dim=-1)

        return obs.reshape(self.num_envs, -1)

    def get_future_target_poses(
        self, *, num_future_steps: int | None = None, target_pose_type: str | None = None
    ) -> torch.Tensor:
        if num_future_steps is None and target_pose_type is None:
            if self.future_target_poses is None:
                return torch.zeros(self.num_envs, 0, device=self.device)
            return self.future_target_poses

        resolved_steps = self.num_future_steps if num_future_steps is None else num_future_steps
        resolved_type = self.target_pose_type if target_pose_type is None else target_pose_type
        if resolved_steps <= 0 or resolved_type is None:
            return torch.zeros(self.num_envs, 0, device=self.device)
        return self._compute_future_target_poses(resolved_steps, resolved_type)

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        if self.multi_clip:
            if prepend:
                logger.warning("Skipping default pose transitions for multi-clip motion banks.")
            return
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _configure_runtime_default_pose_prepend(self) -> None:
        self._runtime_default_pose_prepend_enabled = False
        self._runtime_default_pose_prepend_steps = 0
        self._runtime_default_pose_prepend_defaults = {}
        self._runtime_default_pose_prepend_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._runtime_default_pose_prepend_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        if not self.multi_clip or not self.motion_cfg.enable_default_pose_prepend:
            return

        duration = self.motion_cfg.default_pose_prepend_duration_s
        if duration <= 0.0:
            return

        if self._env.simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            logger.warning("Runtime default-pose prepend only supports IsaacSim; disabling multi-clip prepend.")
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Runtime default pose prepend duration {}s is too short for dt {}; disabling multi-clip prepend.",
                duration,
                self._env.dt,
            )
            return

        default_states = [
            self._build_default_pose_state_robot_order(int(motion_idx.item()))
            for motion_idx in self.motion.clip_offsets
        ]
        if not default_states:
            return

        self._runtime_default_pose_prepend_defaults = {
            "joint_pos": torch.stack([state["joint_pos"] for state in default_states], dim=0),
            "joint_vel": torch.stack([state["joint_vel"] for state in default_states], dim=0),
            "body_pos": torch.stack([state["body_pos"] for state in default_states], dim=0),
            "body_quat": torch.stack([state["body_quat"] for state in default_states], dim=0),
            "body_lin_vel": torch.stack([state["body_lin_vel"] for state in default_states], dim=0),
            "body_ang_vel": torch.stack([state["body_ang_vel"] for state in default_states], dim=0),
            "object_pos": torch.stack([state["object_pos"] for state in default_states], dim=0),
            "object_quat": torch.stack([state["object_quat"] for state in default_states], dim=0),
            "object_lin_vel": torch.stack([state["object_lin_vel"] for state in default_states], dim=0),
        }
        self._runtime_default_pose_prepend_enabled = True
        self._runtime_default_pose_prepend_steps = num_steps
        logger.info(
            "Using runtime default-pose prepend for multi-clip motion bank ({} clips, {} frames, {}s).",
            self.motion.num_clips,
            num_steps,
            duration,
        )

    def _build_default_pose_state_robot_order(self, motion_idx: int) -> dict[str, torch.Tensor]:
        """Build the robot default standing pose anchored to a specific motion frame."""
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        )
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        if self.motion.has_object:
            object_pos = self.motion.object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion.object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion.object_lin_vel_w[motion_idx].to(self.device)
            object_size = self.motion.object_size[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(4, device=self.device, dtype=torch.float32)
            object_quat[3] = 1.0
            object_lin_vel = torch.zeros(3, device=self.device, dtype=torch.float32)
            object_size = torch.zeros(3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": body_states["pos"],
            "body_quat": body_states["quat"],
            "body_lin_vel": body_states["lin_vel"],
            "body_ang_vel": body_states["ang_vel"],
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
            "object_size": object_size,
        }

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        motion_idx = -1 if use_motion_end else 0
        default_state = self._build_default_pose_state_robot_order(motion_idx)

        return {
            "joint_pos": default_state["joint_pos"].clone(),
            "joint_vel": default_state["joint_vel"],
            "root_pos": default_state["root_pos"],
            "root_quat": default_state["root_quat"],
            "root_lin_vel": default_state["root_lin_vel"],
            "root_ang_vel": default_state["root_ang_vel"],
            "body_pos": self._map_robot_bodies_to_motion_order(default_state["body_pos"]),
            "body_quat": self._map_robot_bodies_to_motion_order(default_state["body_quat"]),
            "body_lin_vel": self._map_robot_bodies_to_motion_order(default_state["body_lin_vel"]),
            "body_ang_vel": self._map_robot_bodies_to_motion_order(default_state["body_ang_vel"]),
            "object_pos": default_state["object_pos"],
            "object_quat": default_state["object_quat"],
            "object_lin_vel": default_state["object_lin_vel"],
            "object_size": default_state["object_size"],
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = self._get_env_offsets()[env_id]

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor


    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
            state["object_size"] = self.motion._object_size[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
            state["object_size"] = default_state["object_size"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)
            segments["object_size"] = _lerp(start["object_size"], target["object_size"], alphas_joint)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
