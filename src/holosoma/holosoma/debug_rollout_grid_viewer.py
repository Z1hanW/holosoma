from __future__ import annotations

import argparse
import hashlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.debug_rollout_viewer import (  # noqa: E402
    _DEFAULT_ROBOT_URDF_PATH,
    _DRAWN_CONTACT_REGION_ORDER,
    _REGION_ORDER,
    _REGION_OVERLAY_STYLE,
    _as_points,
    _decode_scalar,
    _filter_entries,
    _joint_indices_for_viser,
    _load_entries,
    _load_exclude_clip_ids,
    _require_object_overlay_mesh,
    _rgb_tuple,
    _xyzw_to_wxyz,
)
from holosoma.utils.viser_live import _create_viser_urdf_handle, _ensure_viser_api_compat  # noqa: E402
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402


@dataclass
class SlotState:
    entry: Any
    offset: np.ndarray
    original_offset: np.ndarray | None
    ref: dict[str, np.ndarray]
    valid_indices: np.ndarray
    handles: list[Any]
    robot_root: Any | None
    robot_viser: Any | None
    robot_joint_indices: np.ndarray | None
    robot_joint_pos: np.ndarray | None
    robot_body_handle: Any | None
    robot_body_pos: np.ndarray | None
    object_motion_pos: np.ndarray | None
    object_motion_quat_wxyz: np.ndarray | None
    object_handles: list[Any]
    contact_handles: list[Any]
    original_robot_root: Any | None
    original_robot_viser: Any | None
    original_robot_joint_indices: np.ndarray | None
    original_robot_joint_pos: np.ndarray | None
    original_object_motion_pos: np.ndarray | None
    original_object_motion_quat_wxyz: np.ndarray | None
    original_object_handles: list[Any]


def _remove_handles(handles: list[Any]) -> None:
    seen: set[int] = set()
    for handle in handles:
        handle_id = id(handle)
        if handle_id in seen:
            continue
        seen.add(handle_id)
        try:
            handle.remove()
        except Exception:
            pass


def _load_valid_indices(ref: dict[str, np.ndarray]) -> np.ndarray:
    trajectory_length = int(np.asarray(ref.get("trajectory_length", np.asarray(0))).reshape(-1)[0])
    if trajectory_length <= 0:
        lengths = [
            np.asarray(ref.get(key, np.zeros((0,)))).shape[0]
            for key in ("body_pos_local", "object_pos_local", "target_joint_pos", "target_object_pos_local")
        ]
        trajectory_length = max(lengths) if lengths else 0
    valid_steps = np.asarray(ref.get("valid_steps", np.zeros((0,), dtype=np.bool_)), dtype=np.bool_)
    valid_indices = np.flatnonzero(valid_steps)
    if valid_indices.size == 0:
        valid_indices = np.arange(max(1, trajectory_length), dtype=np.int64)
    return valid_indices.astype(np.int64, copy=False)


def _entry_valid_frame_count(entry: Any) -> int:
    with np.load(entry.data_dir / "teacher_rollout_reference.npz", allow_pickle=True) as data:
        ref = {key: np.asarray(data[key]) for key in data.files}
    return int(_load_valid_indices(ref).size)


def _resolve_required_motion_bank_path(entry: Any) -> Path:
    raw_path = str(entry.metadata.get("teacher_rollout_motion_bank_path") or "").strip()
    if not raw_path:
        raise RuntimeError(f"Missing teacher_rollout_motion_bank_path metadata for clip={entry.clip_id!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (entry.data_dir / path).resolve()
    if not path.exists():
        raise RuntimeError(f"Missing teacher rollout motion bank for clip={entry.clip_id!r}: {path}")
    return path


def _load_motion_bank_kinematics(
    entry: Any,
    robot_joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motion_bank_path = _resolve_required_motion_bank_path(entry)
    return _load_motion_bank_kinematics_from_path(motion_bank_path, entry=entry, robot_joint_names=robot_joint_names)


def _load_motion_bank_kinematics_from_path(
    motion_bank_path: Path,
    *,
    entry: Any,
    robot_joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(motion_bank_path, allow_pickle=True) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        object_pos = np.asarray(data["object_pos_w"], dtype=np.float32)
        object_quat_wxyz = np.asarray(data["object_quat_w"], dtype=np.float32)
        motion_joint_names = (
            [_decode_scalar(name) for name in np.asarray(data["joint_names"])]
            if "joint_names" in data.files
            else []
        )
    if joint_pos.ndim != 2 or joint_pos.shape[1] < 7 or joint_pos.shape[0] == 0:
        raise RuntimeError(f"Invalid joint_pos in {motion_bank_path}: shape={joint_pos.shape}")
    if object_pos.ndim != 2 or object_pos.shape[1] != 3 or object_pos.shape[0] == 0:
        raise RuntimeError(f"Invalid object_pos_w in {motion_bank_path}: shape={object_pos.shape}")
    if object_quat_wxyz.ndim != 2 or object_quat_wxyz.shape[1] != 4 or object_quat_wxyz.shape[0] == 0:
        raise RuntimeError(f"Invalid object_quat_w in {motion_bank_path}: shape={object_quat_wxyz.shape}")
    joint_indices, _error = _joint_indices_for_viser(
        viser_joint_names=robot_joint_names,
        motion_joint_names=motion_joint_names,
        num_motion_joints=max(0, int(joint_pos.shape[1]) - 7),
    )
    if joint_indices is None:
        raise RuntimeError(f"Cannot map motion bank joints for clip={entry.clip_id!r}: {_error}")
    return joint_pos, joint_indices, object_pos, object_quat_wxyz


def _resolve_original_motion_bank_path(entry: Any, original_motion_dir: Path) -> Path:
    path = original_motion_dir / f"{entry.clip_id}.npz"
    if not path.is_file():
        raise RuntimeError(f"Missing original motion bank for clip={entry.clip_id!r}: {path}")
    return path


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(), byteorder="little")


def _sample_real_mesh_vertices(entry: Any, mesh: Any, max_points: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise RuntimeError(
            "Real object mesh has no usable vertices for "
            f"clip={entry.clip_id!r}; object_urdf_path={entry.metadata.get('object_urdf_path')!r}"
        )
    max_points = int(max_points)
    if max_points <= 0 or vertices.shape[0] <= max_points:
        return vertices
    rng = np.random.default_rng(_stable_seed(entry.clip_id))
    indices = np.sort(rng.choice(vertices.shape[0], size=max_points, replace=False))
    return vertices[indices]


class RolloutGridViewer:
    def __init__(
        self,
        *,
        data_root: Path,
        vis_root: Path,
        stats_root: Path,
        host: str,
        port: int,
        robot_urdf_path: Path | None,
        group_size: int,
        cols: int,
        spacing: float,
        playback_fps: float,
        autoplay: bool,
        robot_mode: str,
        object_visual: str,
        object_point_count: int,
        original_motion_dir: Path | None,
        success_only: bool,
        strict_success_only: bool,
        solid_only: bool,
        exclude_clip_ids: set[str],
    ) -> None:
        self.data_root = data_root
        self.vis_root = vis_root
        self.stats_root = stats_root
        self.group_size = max(1, int(group_size))
        self.cols = max(1, int(cols))
        self.spacing = float(spacing)
        self.playback_fps = max(1.0, float(playback_fps))
        self.playing = bool(autoplay)
        self.robot_mode = str(robot_mode)
        self.object_visual = str(object_visual)
        self.object_point_count = max(1, int(object_point_count))
        self.original_motion_dir = original_motion_dir.expanduser().resolve() if original_motion_dir else None
        self.frame_float = 0.0
        self.frame_slider_max = 0
        self.slider_syncing = False
        self.group_syncing = False
        self.slots: list[SlotState] = []
        self.scene_handles: list[Any] = []

        self.entries = _filter_entries(
            _load_entries(data_root, vis_root, stats_root),
            data_root=data_root,
            stats_root=stats_root,
            success_only=success_only,
            strict_success_only=strict_success_only,
            solid_only=solid_only,
            exclude_clip_ids=exclude_clip_ids,
        )
        self.groups = [
            self.entries[start : start + self.group_size]
            for start in range(0, len(self.entries), self.group_size)
        ]
        if not self.groups:
            raise FileNotFoundError(f"No rollout clips found under {data_root / 'clips'}")
        self.global_slider_max = max(0, max(_entry_valid_frame_count(entry) for entry in self.entries) - 1)

        self.host = host
        self.port = resolve_viser_port(port)
        self.server = viser.ViserServer(host=host, port=self.port, label="debug_rollout_grid")
        _ensure_viser_api_compat(self.server)
        self.server.scene.add_grid("/grid", width=max(8.0, self.cols * self.spacing), height=8.0)

        self.robot_urdf_path = robot_urdf_path.expanduser().resolve() if robot_urdf_path else None
        self.robot_enabled = self.robot_mode == "urdf" and self.robot_urdf_path is not None and self.robot_urdf_path.exists()
        if self.original_motion_dir is not None:
            if not self.original_motion_dir.is_dir():
                raise FileNotFoundError(f"Original motion dir does not exist: {self.original_motion_dir}")
            missing = [
                entry.clip_id
                for entry in self.entries
                if not (self.original_motion_dir / f"{entry.clip_id}.npz").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Original motion dir is missing clips: "
                    + ", ".join(missing[:10])
                    + (f" ... ({len(missing)} total)" if len(missing) > 10 else "")
                )
            if self.robot_mode != "urdf":
                raise RuntimeError("--original-motion-dir comparison requires --robot-mode urdf")
            if not self.robot_enabled:
                raise RuntimeError("--original-motion-dir comparison requires a valid --robot-urdf")

        with self.server.gui.add_folder("Groups"):
            self.group_dropdown = self.server.gui.add_dropdown(
                "Group",
                options=tuple(self._group_label(idx) for idx in range(len(self.groups))),
                initial_value=self._group_label(0),
            )
            self.info_md = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Playback"):
            self.frame_slider = self.server.gui.add_slider(
                "Valid Frame",
                min=0,
                max=self.global_slider_max,
                step=1,
                initial_value=0,
            )
            self.playing_cb = self.server.gui.add_checkbox("Playing", initial_value=self.playing)
            self.fps_number = self.server.gui.add_number("FPS", initial_value=self.playback_fps, min=1, max=240, step=1)
            self.frame_md = self.server.gui.add_markdown("")

        self.group_dropdown.on_update(self._on_group_update)
        self.frame_slider.on_update(self._on_frame_update)
        self.playing_cb.on_update(self._on_playing_update)

        self.current_group_index = 0
        self.load_group(0)
        threading.Thread(target=self._player_loop, daemon=True).start()

    def _group_label(self, idx: int) -> str:
        entries = self.groups[idx]
        first = entries[0].clip_id
        last = entries[-1].clip_id
        return f"group_{idx + 1:02d} ({len(entries)}): {first} ... {last}"

    def _slot_offset(self, slot_idx: int) -> np.ndarray:
        row = slot_idx // self.cols
        col = slot_idx % self.cols
        rows = max(1, math.ceil(self.group_size / self.cols))
        x = (col - (self.cols - 1) * 0.5) * self.spacing
        y = ((rows - 1) * 0.5 - row) * self.spacing
        return np.asarray([x, y, 0.0], dtype=np.float32)

    def _clear_group(self) -> None:
        for slot in self.slots:
            if slot.robot_viser is not None:
                try:
                    slot.robot_viser.remove()
                except Exception:
                    pass
            if slot.original_robot_viser is not None:
                try:
                    slot.original_robot_viser.remove()
                except Exception:
                    pass
            _remove_handles(slot.handles)
        _remove_handles(self.scene_handles)
        self.slots = []
        self.scene_handles = []

    def _on_group_update(self, _evt) -> None:
        if self.group_syncing:
            return
        value = str(self.group_dropdown.value)
        for idx in range(len(self.groups)):
            if value == self._group_label(idx):
                self.load_group(idx)
                return

    def _on_frame_update(self, _evt) -> None:
        if self.slider_syncing:
            return
        if self.playing:
            return
        self.frame_float = float(self.frame_slider.value)
        self.apply_frame(int(self.frame_slider.value))

    def _on_playing_update(self, _evt) -> None:
        self.playing = bool(self.playing_cb.value)

    def _set_slider_max(self, max_frame: int) -> None:
        max_frame = max(0, int(max_frame))
        self.frame_slider_max = max_frame
        self.slider_syncing = True
        try:
            self.frame_slider.value = 0
        finally:
            self.slider_syncing = False

    def load_group(self, group_index: int) -> None:
        self.current_group_index = int(group_index)
        self._clear_group()
        entries = self.groups[self.current_group_index]
        max_frames = 1

        for slot_idx, entry in enumerate(entries):
            base_offset = self._slot_offset(slot_idx)
            if self.original_motion_dir is not None:
                compare_sep = min(2.4, max(1.6, self.spacing * 0.65))
                offset = base_offset + np.asarray([-0.5 * compare_sep, 0.0, 0.0], dtype=np.float32)
                original_offset = base_offset + np.asarray([0.5 * compare_sep, 0.0, 0.0], dtype=np.float32)
            else:
                offset = base_offset
                original_offset = None
            handles: list[Any] = []
            prefix = f"/group_{self.current_group_index:02d}/slot_{slot_idx:02d}"
            frame = self.server.scene.add_frame(prefix, position=base_offset, show_axes=False)
            handles.append(frame)
            label = self.server.scene.add_label(
                f"{prefix}/label",
                text=f"{slot_idx + 1:02d}: {entry.clip_id}",
                position=base_offset + np.asarray([0.0, 0.0, 1.35], dtype=np.float32),
            )
            handles.append(label)
            if original_offset is not None:
                rollout_label = self.server.scene.add_label(
                    f"{prefix}/rollout_label",
                    text="rollout",
                    position=offset + np.asarray([0.0, 0.0, 1.15], dtype=np.float32),
                )
                original_label = self.server.scene.add_label(
                    f"{prefix}/original_label",
                    text="original",
                    position=original_offset + np.asarray([0.0, 0.0, 1.15], dtype=np.float32),
                )
                handles.extend([rollout_label, original_label])

            with np.load(entry.data_dir / "teacher_rollout_reference.npz", allow_pickle=True) as data:
                ref = {key: np.asarray(data[key]) for key in data.files}
            valid_indices = _load_valid_indices(ref)
            max_frames = max(max_frames, int(valid_indices.size))

            robot_root = None
            robot_viser = None
            robot_joint_pos = None
            robot_joint_indices = None
            robot_body_handle = None
            robot_body_pos = None
            object_motion_pos = None
            object_motion_quat_wxyz = None
            original_robot_root = None
            original_robot_viser = None
            original_robot_joint_pos = None
            original_robot_joint_indices = None
            original_object_motion_pos = None
            original_object_motion_quat_wxyz = None
            if self.robot_mode == "body-points":
                robot_body_pos = np.asarray(ref.get("body_pos_local", np.zeros((0, 0, 3))), dtype=np.float32)
                if robot_body_pos.ndim != 3 or robot_body_pos.shape[2] != 3 or robot_body_pos.shape[0] == 0:
                    raise RuntimeError(f"Missing body_pos_local kinematic trajectory for clip={entry.clip_id!r}")
                first_raw_idx = int(np.clip(valid_indices[0], 0, robot_body_pos.shape[0] - 1))
                body_points = robot_body_pos[first_raw_idx] + offset.reshape(1, 3)
                body_colors = np.tile(np.asarray([40, 220, 255], dtype=np.uint8), (body_points.shape[0], 1))
                robot_body_handle = self.server.scene.add_point_cloud(
                    f"{prefix}/robot_body_points",
                    points=body_points,
                    colors=body_colors,
                    point_size=0.035,
                    point_shape="circle",
                )
                handles.append(robot_body_handle)
            elif self.robot_enabled and self.robot_urdf_path is not None:
                robot_root = self.server.scene.add_frame(f"{prefix}/rollout_robot", show_axes=False)
                handles.append(robot_root)
                robot_viser = _create_viser_urdf_handle(
                    ViserUrdf,
                    self.server,
                    self.robot_urdf_path,
                    root_node_name=f"{prefix}/rollout_robot",
                )
                robot_joint_names = list(robot_viser.get_actuated_joint_names())
                robot_viser.update_cfg(np.zeros((len(robot_joint_names),), dtype=np.float32))
                (
                    robot_joint_pos,
                    robot_joint_indices,
                    object_motion_pos,
                    object_motion_quat_wxyz,
                ) = _load_motion_bank_kinematics(entry, robot_joint_names)
                if self.original_motion_dir is not None and original_offset is not None:
                    original_robot_root = self.server.scene.add_frame(f"{prefix}/original_robot", show_axes=False)
                    handles.append(original_robot_root)
                    original_robot_viser = _create_viser_urdf_handle(
                        ViserUrdf,
                        self.server,
                        self.robot_urdf_path,
                        root_node_name=f"{prefix}/original_robot",
                    )
                    original_robot_joint_names = list(original_robot_viser.get_actuated_joint_names())
                    if original_robot_joint_names != robot_joint_names:
                        raise RuntimeError(
                            "Original/rollout viewer robot joint order mismatch for "
                            f"clip={entry.clip_id!r}"
                        )
                    original_robot_viser.update_cfg(np.zeros((len(original_robot_joint_names),), dtype=np.float32))
                    (
                        original_robot_joint_pos,
                        original_robot_joint_indices,
                        original_object_motion_pos,
                        original_object_motion_quat_wxyz,
                    ) = _load_motion_bank_kinematics_from_path(
                        _resolve_original_motion_bank_path(entry, self.original_motion_dir),
                        entry=entry,
                        robot_joint_names=original_robot_joint_names,
                    )

            object_handles: list[Any] = []
            original_object_handles: list[Any] = []
            contact_handles: list[Any] = []
            object_mesh = _require_object_overlay_mesh(entry)
            if self.object_visual == "mesh":
                vertices = np.asarray(object_mesh.vertices, dtype=np.float32)
                faces = np.asarray(object_mesh.faces, dtype=np.int32)
                handle = self.server.scene.add_mesh_simple(
                    f"{prefix}/object_mesh",
                    vertices=vertices,
                    faces=faces,
                    color=(175, 185, 200),
                    opacity=0.55,
                    side="double",
                )
                if original_offset is not None:
                    original_handle = self.server.scene.add_mesh_simple(
                        f"{prefix}/original_object_mesh",
                        vertices=vertices,
                        faces=faces,
                        color=(170, 105, 255),
                        opacity=0.45,
                        side="double",
                    )
                    handles.append(original_handle)
                    original_object_handles.append(original_handle)
            else:
                points = _sample_real_mesh_vertices(entry, object_mesh, self.object_point_count)
                colors = np.tile(np.asarray([175, 185, 200], dtype=np.uint8), (points.shape[0], 1))
                handle = self.server.scene.add_point_cloud(
                    f"{prefix}/object_surface_points",
                    points=points,
                    colors=colors,
                    point_size=0.006,
                    point_shape="circle",
                )
                if original_offset is not None:
                    original_colors = np.tile(np.asarray([170, 105, 255], dtype=np.uint8), (points.shape[0], 1))
                    original_handle = self.server.scene.add_point_cloud(
                        f"{prefix}/original_object_surface_points",
                        points=points,
                        colors=original_colors,
                        point_size=0.006,
                        point_shape="circle",
                    )
                    handles.append(original_handle)
                    original_object_handles.append(original_handle)
            handles.append(handle)
            object_handles.append(handle)

            for region_name in _DRAWN_CONTACT_REGION_ORDER:
                points = _as_points(entry.data_dir / f"{region_name}_contact_points.npy")
                if points.shape[0] == 0:
                    continue
                style = _REGION_OVERLAY_STYLE[region_name]
                point_handle = self.server.scene.add_point_cloud(
                    f"{prefix}/contact/{region_name}",
                    points=points,
                    colors=_rgb_tuple(region_name),
                    point_size=float(style["scatter_size"]) * 0.0007,
                    point_shape="circle",
                )
                handles.append(point_handle)
                contact_handles.append(point_handle)

            obj_pos = object_motion_pos
            path_indices = np.arange(valid_indices.size)
            if obj_pos is None:
                obj_pos = np.asarray(ref.get("object_pos_local", np.zeros((0, 3))), dtype=np.float32)
                path_indices = valid_indices
            if obj_pos.ndim == 2 and obj_pos.shape[0] >= 2:
                idx = np.clip(path_indices, 0, obj_pos.shape[0] - 1)
                path = obj_pos[idx] + offset.reshape(1, 3)
                if path.shape[0] > 240:
                    sel = np.linspace(0, path.shape[0] - 1, 240).round().astype(np.int64)
                    path = path[sel]
                path_handle = self.server.scene.add_spline_catmull_rom(
                    f"{prefix}/object_path",
                    positions=path,
                    line_width=2.0,
                    color=(40, 220, 255),
                )
                handles.append(path_handle)
            if original_offset is not None and original_object_motion_pos is not None and original_object_motion_pos.shape[0] >= 2:
                idx = np.clip(np.arange(valid_indices.size), 0, original_object_motion_pos.shape[0] - 1)
                path = original_object_motion_pos[idx] + original_offset.reshape(1, 3)
                if path.shape[0] > 240:
                    sel = np.linspace(0, path.shape[0] - 1, 240).round().astype(np.int64)
                    path = path[sel]
                path_handle = self.server.scene.add_spline_catmull_rom(
                    f"{prefix}/original_object_path",
                    positions=path,
                    line_width=2.0,
                    color=(170, 105, 255),
                )
                handles.append(path_handle)

            self.slots.append(
                SlotState(
                    entry=entry,
                    offset=offset,
                    original_offset=original_offset,
                    ref=ref,
                    valid_indices=valid_indices,
                    handles=handles,
                    robot_root=robot_root,
                    robot_viser=robot_viser,
                    robot_joint_indices=robot_joint_indices,
                    robot_joint_pos=robot_joint_pos,
                    robot_body_handle=robot_body_handle,
                    robot_body_pos=robot_body_pos,
                    object_motion_pos=object_motion_pos,
                    object_motion_quat_wxyz=object_motion_quat_wxyz,
                    object_handles=object_handles,
                    contact_handles=contact_handles,
                    original_robot_root=original_robot_root,
                    original_robot_viser=original_robot_viser,
                    original_robot_joint_indices=original_robot_joint_indices,
                    original_robot_joint_pos=original_robot_joint_pos,
                    original_object_motion_pos=original_object_motion_pos,
                    original_object_motion_quat_wxyz=original_object_motion_quat_wxyz,
                    original_object_handles=original_object_handles,
                )
            )

        self._set_slider_max(max_frames - 1)
        self.frame_float = 0.0
        self.info_md.content = (
            f"`{self._group_label(self.current_group_index)}`  "
            f"showing `{len(entries)}` clips; total success clips `{len(self.entries)}`.  "
            f"robot=`{self.robot_mode}: teacher rollout motion bank`, object_visual=`{self.object_visual}`."
            + (
                f"  comparison=`rollout left vs original right from {self.original_motion_dir}`."
                if self.original_motion_dir is not None
                else ""
            )
        )
        self.apply_frame(0)

    def apply_frame(self, slider_index: int) -> None:
        if self.slots:
            first_slot = self.slots[0]
            first_local_idx = int(np.clip(slider_index, 0, first_slot.valid_indices.size - 1))
            first_raw_idx = int(first_slot.valid_indices[first_local_idx])
            self.frame_md.content = (
                f"valid frame `{first_local_idx}` -> teacher rollout raw frame `{first_raw_idx}` "
                f"and motion-bank frame `{first_local_idx}`"
            )
        with self.server.atomic():
            for slot in self.slots:
                local_idx = int(np.clip(slider_index, 0, slot.valid_indices.size - 1))
                raw_idx = int(slot.valid_indices[local_idx])
                if slot.object_motion_pos is not None and slot.object_motion_quat_wxyz is not None:
                    obj_idx = int(
                        np.clip(local_idx, 0, min(slot.object_motion_pos.shape[0], slot.object_motion_quat_wxyz.shape[0]) - 1)
                    )
                    obj_pos = slot.offset + slot.object_motion_pos[obj_idx]
                    obj_wxyz = slot.object_motion_quat_wxyz[obj_idx]
                    for handle in slot.object_handles + slot.contact_handles:
                        handle.position = obj_pos
                        handle.wxyz = obj_wxyz
                else:
                    obj_pos_arr = np.asarray(slot.ref.get("object_pos_local", np.zeros((0, 3))), dtype=np.float32)
                    obj_quat_arr = np.asarray(slot.ref.get("object_quat_w", np.zeros((0, 4))), dtype=np.float32)
                    if obj_pos_arr.ndim == 2 and obj_quat_arr.ndim == 2 and obj_pos_arr.shape[0] and obj_quat_arr.shape[0]:
                        obj_idx = int(np.clip(raw_idx, 0, min(obj_pos_arr.shape[0], obj_quat_arr.shape[0]) - 1))
                        obj_pos = slot.offset + obj_pos_arr[obj_idx]
                        obj_wxyz = _xyzw_to_wxyz(obj_quat_arr[obj_idx])
                        for handle in slot.object_handles + slot.contact_handles:
                            handle.position = obj_pos
                            handle.wxyz = obj_wxyz

                if slot.robot_body_handle is not None and slot.robot_body_pos is not None:
                    body_idx = int(np.clip(raw_idx, 0, slot.robot_body_pos.shape[0] - 1))
                    slot.robot_body_handle.points = slot.robot_body_pos[body_idx] + slot.offset.reshape(1, 3)

                if slot.robot_viser is not None and slot.robot_root is not None:
                    if slot.robot_joint_pos is not None and slot.robot_joint_indices is not None:
                        motion_idx = int(np.clip(local_idx, 0, slot.robot_joint_pos.shape[0] - 1))
                        q = slot.robot_joint_pos[motion_idx]
                        slot.robot_root.position = slot.offset + q[:3]
                        slot.robot_root.wxyz = q[3:7]
                        dof = q[7:]
                        max_joint_index = int(slot.robot_joint_indices.max()) if slot.robot_joint_indices.size else -1
                        if dof.shape[0] > max_joint_index:
                            slot.robot_viser.update_cfg(dof[slot.robot_joint_indices].astype(np.float32, copy=False))

                if (
                    slot.original_offset is not None
                    and slot.original_object_motion_pos is not None
                    and slot.original_object_motion_quat_wxyz is not None
                ):
                    obj_idx = int(
                        np.clip(
                            local_idx,
                            0,
                            min(
                                slot.original_object_motion_pos.shape[0],
                                slot.original_object_motion_quat_wxyz.shape[0],
                            )
                            - 1,
                        )
                    )
                    obj_pos = slot.original_offset + slot.original_object_motion_pos[obj_idx]
                    obj_wxyz = slot.original_object_motion_quat_wxyz[obj_idx]
                    for handle in slot.original_object_handles:
                        handle.position = obj_pos
                        handle.wxyz = obj_wxyz

                if slot.original_robot_viser is not None and slot.original_robot_root is not None:
                    if slot.original_robot_joint_pos is not None and slot.original_robot_joint_indices is not None:
                        motion_idx = int(np.clip(local_idx, 0, slot.original_robot_joint_pos.shape[0] - 1))
                        q = slot.original_robot_joint_pos[motion_idx]
                        original_offset = slot.original_offset if slot.original_offset is not None else slot.offset
                        slot.original_robot_root.position = original_offset + q[:3]
                        slot.original_robot_root.wxyz = q[3:7]
                        dof = q[7:]
                        max_joint_index = (
                            int(slot.original_robot_joint_indices.max())
                            if slot.original_robot_joint_indices.size
                            else -1
                        )
                        if dof.shape[0] > max_joint_index:
                            slot.original_robot_viser.update_cfg(
                                dof[slot.original_robot_joint_indices].astype(np.float32, copy=False)
                            )

    def _player_loop(self) -> None:
        next_tick = time.perf_counter()
        while True:
            if self.playing and self.slots:
                now = time.perf_counter()
                if now >= next_tick:
                    fps = max(1.0, float(self.fps_number.value))
                    next_tick = now + 1.0 / fps
                    max_frame = int(self.frame_slider_max)
                    self.frame_float += 1.0
                    if self.frame_float > max_frame:
                        self.frame_float = 0.0
                    frame_idx = int(self.frame_float)
                    self.apply_frame(frame_idx)
                    self.slider_syncing = True
                    try:
                        self.frame_slider.value = frame_idx
                    finally:
                        self.slider_syncing = False
            time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid viewer for successful teacher rollout clips.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--vis-root", type=Path, default=None)
    parser.add_argument("--stats-root", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7081)
    parser.add_argument("--robot-urdf", type=Path, default=_DEFAULT_ROBOT_URDF_PATH)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--robot-mode", choices=("body-points", "urdf", "none"), default="body-points")
    parser.add_argument("--object-visual", choices=("surface-points", "mesh"), default="surface-points")
    parser.add_argument("--object-point-count", type=int, default=6000)
    parser.add_argument(
        "--original-motion-dir",
        type=Path,
        default=None,
        help="Strict comparison source: directory containing <clip_id>.npz original motion-bank files.",
    )
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--strict-success-only", action="store_true")
    parser.add_argument("--solid-only", action="store_true")
    parser.add_argument("--exclude-clip", action="append", default=[])
    parser.add_argument("--exclude-clips-file", type=Path, default=None)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    vis_root = args.vis_root.expanduser().resolve() if args.vis_root else data_root
    stats_root = args.stats_root.expanduser().resolve() if args.stats_root else data_root
    viewer = RolloutGridViewer(
        data_root=data_root,
        vis_root=vis_root,
        stats_root=stats_root,
        host=args.host,
        port=args.port,
        robot_urdf_path=args.robot_urdf,
        group_size=args.group_size,
        cols=args.cols,
        spacing=args.spacing,
        playback_fps=args.playback_fps,
        autoplay=bool(args.autoplay),
        robot_mode=args.robot_mode,
        object_visual=args.object_visual,
        object_point_count=args.object_point_count,
        original_motion_dir=args.original_motion_dir,
        success_only=not bool(args.include_failures),
        strict_success_only=bool(args.strict_success_only),
        solid_only=bool(args.solid_only),
        exclude_clip_ids=_load_exclude_clip_ids(args),
    )
    print(
        f"[INFO] debug rollout grid viewer ready on http://{args.host}:{viewer.port} "
        f"groups={len(viewer.groups)} entries={len(viewer.entries)}",
        flush=True,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
