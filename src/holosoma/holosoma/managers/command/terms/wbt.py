from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import smart_open
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
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


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        motion_clip_id: int | None = None,
        motion_clip_name: str | None = None,
    ):
        self._robot_body_names = list(robot_body_names)
        self._robot_joint_names = list(robot_joint_names)

        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)
        motion_path = Path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        self.clip_ids: list[str] = []
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

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _set_clip_metadata(
        self,
        clip_ids: list[str],
        offsets: np.ndarray,
        lengths: np.ndarray,
        device: str,
    ) -> None:
        self.clip_ids = clip_ids
        self.clip_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.num_clips = len(clip_ids)

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with smart_open.open(motion_file, "rb") as f, np.load(f) as data:
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
                # NOTE: wxyz after loading from npz
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        clip_id = Path(motion_file).stem
        length = int(self._joint_pos.shape[0])
        self._set_clip_metadata([clip_id], np.array([0]), np.array([length]), device)
        return body_names, joint_names

    def _load_data_from_motion_npz_dir(
        self,
        motion_dir: Path,
        device: str,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> tuple[list[str], list[str]]:
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

        if len(files) == 1:
            return self._load_data_from_motion_npz(str(files[0]), device)

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

        for file_path in files:
            with np.load(file_path, allow_pickle=True) as data:
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
                body_names_clip = self._decode_h5_strings(np.asarray(data["body_names"]))
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
                body_pos_list.append(np.asarray(data["body_pos_w"]))
                body_quat_list.append(np.asarray(data["body_quat_w"]))
                body_lin_vel_list.append(np.asarray(data["body_lin_vel_w"]))
                body_ang_vel_list.append(np.asarray(data["body_ang_vel_w"]))

                if clip_has_object:
                    object_pos_list.append(np.asarray(data["object_pos_w"]))
                    object_quat_list.append(np.asarray(data["object_quat_w"]))
                    object_lin_vel_list.append(np.asarray(data["object_lin_vel_w"]))

        self.fps = float(fps_ref) if fps_ref is not None else 30.0
        self._set_clip_metadata(clip_ids, np.array(offsets), np.array(lengths), device)

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

            self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
            object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
            self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
            self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)

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
                self._set_clip_metadata([clip_id], np.array([0]), np.array([length]), device)
            elif load_all:
                start = 0
                length = int(data["joint_pos"].shape[0])
                fps_val = np.asarray(meta["fps"])
                if clip_fps is not None:
                    if not np.allclose(clip_fps, float(np.array(fps_val).reshape(-1)[0])):
                        raise ValueError("clip_fps must be consistent across clips for multi-clip loading.")
                assert offsets is not None and lengths is not None
                self._set_clip_metadata(clip_ids, offsets, lengths, device)
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
                start = int(offsets[clip_idx])
                length = int(lengths[clip_idx])
                fps_val = clip_fps[clip_idx] if clip_fps is not None else np.asarray(meta["fps"])
                self._set_clip_metadata([clip_ids[clip_idx]], np.array([0]), np.array([length]), device)

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

                self._object_pos_w = torch.tensor(object_pos_w, dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(object_quat_w, dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]
                self._object_lin_vel_w = torch.tensor(object_lin_vel_w, dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)

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

        frame_mode = self._infer_link_frame(link_names, link_pos, root_pos)
        link_pos_w = link_pos
        link_quat_w = link_quat_xyzw
        if frame_mode == "local":
            link_pos_w = root_pos[:, None, :] + self._quat_rotate_xyzw(root_quat_xyzw[:, None, :], link_pos)
            link_quat_w = self._quat_mul_xyzw(root_quat_xyzw[:, None, :], link_quat_xyzw)

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
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        bin_size_s: float = 1.0,
        kernel_size: int = 3,
        decay_lambda: float = 0.001,
        kernel_lambda: float = 0.8,
    ):
        # TODO: think better about the decay_lambda, will 0.001 be too small?
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        # size of the bin in seconds
        self.bin_size_s = bin_size_s
        # size of the kernel for smoothing the sampling probabilities
        self.kernel_size = kernel_size
        self.kernel_lambda = kernel_lambda
        # exponential decay when updating the failure counts over training steps.

        self.decay_lambda = decay_lambda

        # number of bins in the motion
        self.num_bins = math.ceil((self.motion_time_step_total / self.env_fps) / self.bin_size_s)

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
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.floor(failed_at_time_step / self.motion_time_step_total * self.num_bins).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.decay_lambda * self.current_bin_failed_count) + (
            1 - self.decay_lambda
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + 1e-6
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities += 0.01
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(self.num_bins)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


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

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        init_state = self._env.robot_config.init_state
        self._init_root_pos = torch.tensor(init_state.pos, dtype=torch.float32, device=self.device)
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
        )
        self.multi_clip = self.motion.num_clips > 1
        if self.multi_clip:
            logger.info("Multi-clip motion bank detected ({} clips). Sampling clips per env reset.", self.motion.num_clips)

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
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )

        # 4. get the adaptive timesteps sampler
        self.use_adaptive_timesteps_sampler = self.motion_cfg.use_adaptive_timesteps_sampler
        if self.multi_clip and self.use_adaptive_timesteps_sampler:
            logger.warning("Adaptive timestep sampling is disabled for multi-clip motion banks.")
            self.use_adaptive_timesteps_sampler = False
        if self.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )

        # 5. clip sampling configuration
        self.clip_weighting_strategy = self.motion_cfg.clip_weighting_strategy
        self.min_weight_factor = self.motion_cfg.min_weight_factor
        self.max_weight_factor = self.motion_cfg.max_weight_factor
        self._clip_sampling_weights: torch.Tensor | None = None
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

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        if self.multi_clip:
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

        # 0. Sample the time steps
        if self.use_adaptive_timesteps_sampler:
            phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        if self._env.is_evaluating:
            phase = torch.zeros_like(phase)

        clip_lengths = self._current_clip_lengths(env_ids)
        start_margin = self._min_start_margin_steps()
        valid_starts = torch.clamp(clip_lengths - start_margin, min=1)
        self.time_steps[env_ids] = (phase * valid_starts).long()

        # Handle start_at_timestep_zero_prob
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = 0
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, torch.zeros_like(subset), subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        max_valid = torch.clamp(clip_lengths - 2, min=0)
        self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids], max_valid)

        if self.motion_cfg.align_motion_to_init_yaw:
            self._update_motion_alignment(env_ids)

        # 1. Get the reference root/body poses
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_rot = self.body_quat_w[env_ids, 0].clone()  # xyzw
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
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
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        pos_noise = torch.zeros_like(root_pos)
        pos_noise[:, :2] = torch.rand(root_pos.shape, device=self.device)[:, :2] * root_pos_noise[:2].unsqueeze(0)
        # z 轴你可以选择保持对称或不动
        pos_noise[:, 2] = (torch.rand(root_pos.shape, device=self.device)[:, 2] - 0.5) * 2 * root_pos_noise[2]
        target_root_pos = root_pos + pos_noise
        
        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

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
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

        self._update_future_target_poses()

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at timestep 0, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
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

    def _init_clip_sampling(self) -> None:
        if not self.multi_clip:
            return
        strategy = self.clip_weighting_strategy
        if strategy == "uniform_step":
            weights = self._valid_start_counts()
        elif strategy in ("uniform_clip", "success_rate_adaptive"):
            weights = torch.ones(self.motion.num_clips, device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown clip_weighting_strategy '{strategy}'.")

        weights = weights / weights.sum()
        self._clip_sampling_weights = weights

        if strategy == "success_rate_adaptive":
            self._base_clip_weights = weights.clone()
            self._clip_success_counts = torch.zeros(self.motion.num_clips, device=self.device)
            self._clip_total_counts = torch.zeros(self.motion.num_clips, device=self.device)

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
            self._clip_sampling_weights = weights / weights.sum()
        else:
            self._clip_sampling_weights = self._base_clip_weights.clone()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        return self.motion.joint_pos[motion_idx]

    @property
    def joint_vel(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        return self.motion.joint_vel[motion_idx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        pos = self.motion.body_pos_w[motion_idx][:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._env.simulator.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        quat = self.motion.body_quat_w[motion_idx][:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        vel = self.motion.body_lin_vel_w[motion_idx][:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        vel = self.motion.body_ang_vel_w[motion_idx][:, self.tracked_body_indexes]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_pos_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        pos = self.motion.body_pos_w[motion_idx, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        quat = self.motion.body_quat_w[motion_idx, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def root_pos_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        pos = self.motion.body_pos_w[motion_idx, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        quat = self.motion.body_quat_w[motion_idx, 0]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        vel = self.motion.body_lin_vel_w[motion_idx, self.ref_body_index]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        vel = self.motion.body_ang_vel_w[motion_idx, self.ref_body_index]
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
        # Applies env origins, but ideally we should rely on the simulator
        motion_idx = self._get_motion_indices(self.time_steps)
        pos = self.motion.object_pos_w[motion_idx]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_pos(pos)
        return pos + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        quat = self.motion.object_quat_w[motion_idx]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_quat(quat)
        return quat

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        motion_idx = self._get_motion_indices(self.time_steps)
        vel = self.motion.object_lin_vel_w[motion_idx]
        if self.motion_cfg.align_motion_to_init_yaw:
            return self._apply_motion_alignment_vec(vel)
        return vel

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
            self._clip_sampling_weights = self._base_clip_weights.clone()

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
        env_origins = self._env.simulator.scene.env_origins[env_ids]
        desired_root_pos = env_origins + self._init_root_pos
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
            + self._env.simulator.scene.env_origins[:, None, None, :]
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

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
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

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
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
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

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
