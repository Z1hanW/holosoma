from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.utils.viser_live import _create_viser_urdf_handle, _ensure_viser_api_compat  # noqa: E402
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402


_DEFAULT_ROBOT_URDF_PATH = SRC_ROOT / "holosoma" / "data" / "robots" / "g1" / "g1_29dof.urdf"
_ORIGINAL_G1_MESH_COLOR = (150, 80, 255)
_ARM_LINK_REGION_ORDER = (
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
)
_REGION_ORDER = ("left_wrist", "right_wrist", *_ARM_LINK_REGION_ORDER, "torso")
_DRAWN_CONTACT_REGION_ORDER = _REGION_ORDER
_CONTACT_BODY_NAMES = {
    "left_wrist_yaw_link": "left_wrist",
    "right_wrist_yaw_link": "right_wrist",
    "left_elbow_link": "left_elbow",
    "right_elbow_link": "right_elbow",
    "left_wrist_roll_link": "left_wrist_roll",
    "right_wrist_roll_link": "right_wrist_roll",
    "left_wrist_pitch_link": "left_wrist_pitch",
    "right_wrist_pitch_link": "right_wrist_pitch",
    "torso_link": "torso",
}
_PRODUCT_OFFSET = np.asarray([0.0, -1.35, 0.45], dtype=np.float32)
_REGION_OVERLAY_STYLE: dict[str, dict[str, Any]] = {
    "left_wrist": {
        "rgba": np.asarray([255, 140, 0, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "right_wrist": {
        "rgba": np.asarray([255, 220, 0, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "left_elbow": {
        "rgba": np.asarray([255, 105, 180, 255], dtype=np.uint8),
        "scatter_size": 36.0,
    },
    "right_elbow": {
        "rgba": np.asarray([220, 20, 60, 255], dtype=np.uint8),
        "scatter_size": 36.0,
    },
    "left_wrist_roll": {
        "rgba": np.asarray([0, 191, 255, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "right_wrist_roll": {
        "rgba": np.asarray([30, 144, 255, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "left_wrist_pitch": {
        "rgba": np.asarray([50, 205, 50, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "right_wrist_pitch": {
        "rgba": np.asarray([34, 139, 34, 255], dtype=np.uint8),
        "scatter_size": 30.0,
    },
    "torso": {
        "rgba": np.asarray([128, 64, 192, 255], dtype=np.uint8),
        "scatter_size": 82.0,
    },
}


def _quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_vec3(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float32)
    parts = [part for part in str(raw).replace(",", " ").split() if part]
    if len(parts) != 3:
        return np.asarray(default, dtype=np.float32)
    try:
        return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    except Exception:
        return np.asarray(default, dtype=np.float32)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _origin_transform(origin_el: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    if origin_el is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_parse_vec3(origin_el.get("rpy"), (0.0, 0.0, 0.0)))
    transform[:3, 3] = _parse_vec3(origin_el.get("xyz"), (0.0, 0.0, 0.0))
    return transform


def _load_object_overlay_mesh(
    *,
    clip_id: str,
    object_name: str,
    object_urdf_path: str,
):
    resolved_urdf = Path(object_urdf_path).expanduser()
    if not resolved_urdf.exists():
        return None
    resolved_urdf = resolved_urdf.resolve()

    try:
        root = ET.parse(resolved_urdf).getroot()
    except Exception:
        return None

    link = root.find("link")
    if link is None:
        return None

    import trimesh

    def _build_from_geom_tag(geom_tag: str):
        meshes: list["trimesh.Trimesh"] = []
        for geom_parent in link.findall(geom_tag):
            geometry_el = geom_parent.find("geometry")
            if geometry_el is None:
                continue
            mesh_obj = None
            box_el = geometry_el.find("box")
            if box_el is not None:
                size = _parse_vec3(box_el.get("size"), (1.0, 1.0, 1.0))
                mesh_obj = trimesh.creation.box(extents=size.astype(np.float64))
            else:
                mesh_el = geometry_el.find("mesh")
                if mesh_el is None:
                    continue
                filename = str(mesh_el.get("filename", "")).strip()
                if not filename:
                    continue
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh_path = (resolved_urdf.parent / mesh_path).resolve()
                if not mesh_path.exists():
                    continue
                try:
                    loaded = trimesh.load(str(mesh_path), process=False)
                except Exception:
                    continue
                if isinstance(loaded, trimesh.Scene):
                    dumped = loaded.dump(concatenate=True)
                    mesh_obj = dumped if isinstance(dumped, trimesh.Trimesh) else None
                elif isinstance(loaded, trimesh.Trimesh):
                    mesh_obj = loaded
                if mesh_obj is None:
                    continue
                scale = _parse_vec3(mesh_el.get("scale"), (1.0, 1.0, 1.0)).astype(np.float64)
                mesh_obj = mesh_obj.copy()
                mesh_obj.apply_scale(scale)

            if mesh_obj is None:
                continue
            mesh_obj = mesh_obj.copy()
            mesh_obj.apply_transform(_origin_transform(geom_parent.find("origin")).astype(np.float64))
            meshes.append(mesh_obj)

        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    mesh = _build_from_geom_tag("visual")
    if mesh is None:
        mesh = _build_from_geom_tag("collision")
    if mesh is None:
        return None

    return mesh


@dataclass(frozen=True)
class ClipEntry:
    label: str
    clip_id: str
    clip_dir_name: str
    data_dir: Path
    vis_dir: Path
    stats_dir: Path
    metadata: dict[str, Any]


@dataclass
class RobotMotion:
    path: Path | None
    joint_pos: np.ndarray | None
    joint_indices: np.ndarray | None
    frame_mode: str
    status: str


@dataclass
class OriginalMotion:
    path: Path | None
    joint_pos: np.ndarray | None
    joint_indices: np.ndarray | None
    object_pos: np.ndarray | None
    object_quat_wxyz: np.ndarray | None
    status: str


def _decode_scalar(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return quat_xyzw[[3, 0, 1, 2]]


def _xyzw_to_wxyz_array(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
        return np.zeros((0, 4), dtype=np.float32)
    return quat_xyzw[:, [3, 0, 1, 2]]


def _rgb_tuple(region_name: str) -> tuple[int, int, int]:
    rgba = np.asarray(_REGION_OVERLAY_STYLE[region_name]["rgba"], dtype=np.uint8)
    return int(rgba[0]), int(rgba[1]), int(rgba[2])


def _as_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    points = np.asarray(np.load(path), dtype=np.float32)
    return points.reshape(-1, 3)


def _decimate_path(points: np.ndarray, max_points: int = 240) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(np.int64)
    return points[idx]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _truthy_metadata(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_success_clip_ids(root: Path) -> set[str]:
    success_path = root / "success_clips.txt"
    if success_path.exists():
        return {line.strip() for line in success_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    summary_path = root / "summary.csv"
    if summary_path.exists():
        return {
            row["clip_id"]
            for row in _read_csv_rows(summary_path)
            if row.get("clip_id") and _truthy_metadata(row.get("success"))
        }
    return set()


def _is_strict_success(entry: "ClipEntry") -> bool:
    status = str(entry.metadata.get("status", ""))
    return bool(
        status == "success_contact_and_final_position"
        or (
            _truthy_metadata(entry.metadata.get("success"))
            and _truthy_metadata(entry.metadata.get("stable_contact_success"))
            and _truthy_metadata(entry.metadata.get("final_position_success"))
        )
    )


def _entry_object_kind(entry: "ClipEntry") -> str:
    name = str(entry.metadata.get("object_name") or entry.clip_id)
    if name.startswith("box_"):
        return "box"
    if "__any_" in name:
        return name.split("__any_", 1)[1].split("_", 1)[0]
    parts = name.split("_")
    return parts[0] if parts else name


def _filter_entries(
    entries: list["ClipEntry"],
    *,
    data_root: Path,
    stats_root: Path,
    success_only: bool,
    strict_success_only: bool,
    solid_only: bool,
    exclude_clip_ids: set[str] | None,
) -> list["ClipEntry"]:
    if strict_success_only:
        filtered = [entry for entry in entries if _is_strict_success(entry)]
    elif success_only:
        success_ids = _load_success_clip_ids(data_root) or _load_success_clip_ids(stats_root)
        if success_ids:
            filtered = [entry for entry in entries if entry.clip_id in success_ids]
        else:
            filtered = [entry for entry in entries if _truthy_metadata(entry.metadata.get("success"))]
    else:
        filtered = entries

    if solid_only:
        solid_kinds = {"box", "bin", "barrel", "ball"}
        filtered = [entry for entry in filtered if _entry_object_kind(entry) in solid_kinds]

    if exclude_clip_ids:
        filtered = [
            entry
            for entry in filtered
            if entry.clip_id not in exclude_clip_ids and entry.clip_dir_name not in exclude_clip_ids
        ]

    if not filtered:
        mode = "strict successful" if strict_success_only else "successful"
        if solid_only:
            mode = f"{mode} solid"
        raise FileNotFoundError(f"No {mode} rollout clips found under: {data_root / 'clips'}")
    return filtered


def _load_exclude_clip_ids(args: argparse.Namespace) -> set[str]:
    excluded = {str(value).strip() for value in getattr(args, "exclude_clip", []) if str(value).strip()}
    exclude_file = getattr(args, "exclude_clips_file", None)
    if exclude_file:
        path = Path(exclude_file).expanduser()
        if path.exists():
            excluded.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return excluded


def _load_entries(data_root: Path, vis_root: Path, stats_root: Path) -> list[ClipEntry]:
    clips_root = data_root / "clips"
    if not clips_root.is_dir():
        raise FileNotFoundError(f"Missing rollout product directory: {clips_root}")

    entries: list[ClipEntry] = []
    used_labels: set[str] = set()
    for data_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        stats_dir = stats_root / "clips" / data_dir.name
        vis_dir = vis_root / "clips" / data_dir.name
        metadata_path = stats_dir / "metadata.json"
        if not metadata_path.exists():
            metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        clip_id = str(metadata.get("clip_id") or data_dir.name)
        label = clip_id
        if label in used_labels:
            label = f"{clip_id} [{data_dir.name}]"
        used_labels.add(label)
        entries.append(
            ClipEntry(
                label=label,
                clip_id=clip_id,
                clip_dir_name=data_dir.name,
                data_dir=data_dir,
                vis_dir=vis_dir,
                stats_dir=stats_dir,
                metadata=metadata,
            )
        )

    if not entries:
        raise FileNotFoundError(f"No usable rollout clips found under: {clips_root}")

    summary_path = stats_root / "summary.csv"
    if summary_path.exists():
        order = {row["clip_id"]: idx for idx, row in enumerate(_read_csv_rows(summary_path)) if "clip_id" in row}
        entries.sort(key=lambda entry: (order.get(entry.clip_id, 10**9), entry.clip_id, entry.clip_dir_name))
    else:
        entries.sort(key=lambda entry: (entry.clip_id, entry.clip_dir_name))
    return entries


def _resolve_motion_bank_path(entry: ClipEntry) -> Path | None:
    raw_path = str(entry.metadata.get("teacher_rollout_motion_bank_path") or "").strip()
    candidates: list[Path] = []
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                [
                    entry.data_dir / path,
                    entry.stats_dir / path,
                    entry.data_dir.parent.parent / "motion_bank" / path.name,
                ]
            )

    fallback_name = f"{entry.clip_id}.npz"
    candidates.append(entry.data_dir.parent.parent / "motion_bank" / fallback_name)
    candidates.append(entry.data_dir.parent.parent / "motion_bank" / f"{entry.clip_dir_name}.npz")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _resolve_original_motion_path(entry: ClipEntry, original_motion_dir: Path | None) -> Path | None:
    if original_motion_dir is None:
        return None
    candidates = [
        original_motion_dir / f"{entry.clip_id}.npz",
        original_motion_dir / f"{entry.clip_dir_name}.npz",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _read_npz_frame_count(path: Path | None, keys: tuple[str, ...]) -> int:
    if path is None or not path.exists():
        return 0
    try:
        with np.load(path, allow_pickle=True) as data:
            lengths = [
                int(np.asarray(data[key]).shape[0])
                for key in keys
                if key in data.files and np.asarray(data[key]).ndim >= 1
            ]
    except Exception:
        return 0
    return max(lengths) if lengths else 0


def _reference_playback_frame_count(entry: ClipEntry, *, playback_full_motion: bool) -> int:
    ref_path = entry.data_dir / "teacher_rollout_reference.npz"
    if not ref_path.exists():
        return 1
    try:
        with np.load(ref_path, allow_pickle=True) as data:
            trajectory_raw = np.asarray(data["trajectory_length"]) if "trajectory_length" in data.files else np.asarray(0)
            trajectory_length = int(trajectory_raw.reshape(-1)[0])
            if trajectory_length <= 0:
                candidate_lengths = [
                    int(np.asarray(data[key]).shape[0])
                    for key in (
                        "body_pos_local",
                        "object_pos_local",
                        "target_joint_pos",
                        "target_object_pos_local",
                    )
                    if key in data.files and np.asarray(data[key]).ndim >= 1
                ]
                trajectory_length = max(candidate_lengths) if candidate_lengths else 0
            if playback_full_motion:
                return max(1, trajectory_length)
            valid_steps = np.asarray(data["valid_steps"], dtype=np.bool_) if "valid_steps" in data.files else np.zeros((0,), dtype=np.bool_)
            valid_count = int(np.count_nonzero(valid_steps))
            return max(1, valid_count if valid_count > 0 else trajectory_length)
    except Exception:
        return 1


def _initial_playback_frame_count(
    entry: ClipEntry,
    original_motion_dir: Path | None,
    *,
    playback_full_motion: bool,
) -> int:
    if playback_full_motion:
        original_path = _resolve_original_motion_path(entry, original_motion_dir)
        original_count = _read_npz_frame_count(original_path, ("joint_pos", "object_pos_w", "object_quat_w"))
        if original_count > 0:
            return original_count
    return _reference_playback_frame_count(entry, playback_full_motion=playback_full_motion)


def _joint_indices_for_viser(
    *,
    viser_joint_names: list[str],
    motion_joint_names: list[str],
    num_motion_joints: int,
) -> tuple[np.ndarray | None, str | None]:
    if motion_joint_names:
        name_to_motion_idx = {name: idx for idx, name in enumerate(motion_joint_names)}
        missing = [name for name in viser_joint_names if name not in name_to_motion_idx]
        if missing:
            return None, f"motion_bank missing URDF joints: {missing[:6]}"
        return np.asarray([name_to_motion_idx[name] for name in viser_joint_names], dtype=np.int64), None

    if len(viser_joint_names) > num_motion_joints:
        return None, f"motion_bank has {num_motion_joints} joints but URDF has {len(viser_joint_names)}"
    return np.arange(len(viser_joint_names), dtype=np.int64), None


class DebugRolloutViewer:
    def __init__(
        self,
        *,
        data_root: Path,
        vis_root: Path,
        stats_root: Path,
        initial_sequence: str | None,
        host: str,
        port: int,
        robot_urdf_path: Path | None,
        original_motion_dir: Path | None,
        show_original_motion_initial: bool = False,
        show_primitive_box_initial: bool = False,
        autoplay_initial: bool = False,
        loop_initial: bool = True,
        playback_full_motion: bool = False,
        replay_only: bool = False,
        playback_fps: float = 30.0,
        success_only: bool = False,
        strict_success_only: bool = False,
        solid_only: bool = False,
        exclude_clip_ids: set[str] | None = None,
    ) -> None:
        self.data_root = data_root
        self.vis_root = vis_root
        self.stats_root = stats_root
        self.original_motion_dir = original_motion_dir.expanduser().resolve() if original_motion_dir is not None else None
        self.playback_full_motion = bool(playback_full_motion)
        self.replay_only = bool(replay_only)
        self.playback_fps = max(1.0, float(playback_fps))
        self.entries = _filter_entries(
            _load_entries(data_root, vis_root, stats_root),
            data_root=data_root,
            stats_root=stats_root,
            success_only=bool(success_only),
            strict_success_only=bool(strict_success_only),
            solid_only=bool(solid_only),
            exclude_clip_ids=exclude_clip_ids,
        )
        self.entry_by_label = {entry.label: entry for entry in self.entries}
        self.current_entry: ClipEntry | None = None

        self.server = viser.ViserServer(host=host, port=port, label="debug_rollout")
        _ensure_viser_api_compat(self.server)
        self.server.scene.add_grid("/grid", width=6.0, height=6.0, position=(0.0, 0.0, 0.0))

        self.static_handles: list[Any] = []
        self.rollout_handles: list[Any] = []
        self.mesh_handles: list[Any] = []
        self.box_handles: list[Any] = []
        self.point_handles: list[Any] = []
        self.path_handles: list[Any] = []
        self.original_handles: list[Any] = []
        self.original_object_handles: list[Any] = []
        self.original_path_handles: list[Any] = []
        self.dynamic_body_handle: Any | None = None
        self.dynamic_label_handles: list[Any] = []
        self.current_object_handles: list[Any] = []
        self.current_contact_handles: list[Any] = []
        self.robot_root_handle: Any | None = None
        self.robot_viser: ViserUrdf | None = None
        self.robot_joint_names: list[str] = []
        self.original_robot_root_handle: Any | None = None
        self.original_robot_viser: ViserUrdf | None = None
        self.original_robot_joint_names: list[str] = []
        self.robot_motion = RobotMotion(
            path=None,
            joint_pos=None,
            joint_indices=None,
            frame_mode="none",
            status="disabled",
        )
        self.original_motion = OriginalMotion(
            path=None,
            joint_pos=None,
            joint_indices=None,
            object_pos=None,
            object_quat_wxyz=None,
            status="disabled",
        )
        self.robot_load_error: str | None = None
        self.original_robot_load_error: str | None = None

        self.ref: dict[str, np.ndarray] = {}
        self.valid_indices = np.zeros((0,), dtype=np.int64)
        self.body_names: list[str] = []
        self.region_points: dict[str, np.ndarray] = {}
        self.is_programmatic_sequence_update = False
        self.is_programmatic_slider_update = False
        self.playing = bool(autoplay_initial)
        self.frame_float = 0.0
        self._last_slider_sync_time = 0.0
        self._last_frame_md_sync_time = 0.0

        initial_label = self.entries[0].label
        if initial_sequence:
            for entry in self.entries:
                if initial_sequence in {entry.label, entry.clip_id, entry.clip_dir_name}:
                    initial_label = entry.label
                    break
        initial_frame_count = _initial_playback_frame_count(
            self.entry_by_label[initial_label],
            self.original_motion_dir,
            playback_full_motion=self.playback_full_motion,
        )

        with self.server.gui.add_folder("Rollout Products"):
            self.sequence_dropdown = self.server.gui.add_dropdown(
                "Sequence",
                options=tuple(entry.label for entry in self.entries),
                initial_value=initial_label,
            )
            self.reload_button = self.server.gui.add_button("Reload")
            self.info_md = self.server.gui.add_markdown("")
            self.interval_md = self.server.gui.add_markdown("")

        with self.server.gui.add_folder("Display"):
            self.show_robot_cb = self.server.gui.add_checkbox("Training G1", initial_value=robot_urdf_path is not None)
            self.show_product_cb = self.server.gui.add_checkbox("Static Product Overlay", initial_value=not self.replay_only)
            self.show_rollout_cb = self.server.gui.add_checkbox("Rollout View", initial_value=True)
            self.show_original_motion_cb = self.server.gui.add_checkbox(
                "Reference Motion",
                initial_value=show_original_motion_initial,
            )
            self.show_mesh_cb = self.server.gui.add_checkbox("Object Mesh", initial_value=True)
            self.show_box_cb = self.server.gui.add_checkbox(
                "Primitive Box",
                initial_value=show_primitive_box_initial,
            )
            self.show_points_cb = self.server.gui.add_checkbox("Contact Points", initial_value=not self.replay_only)
            self.show_paths_cb = self.server.gui.add_checkbox("Object Path", initial_value=not self.replay_only)
            self.show_body_labels_cb = self.server.gui.add_checkbox("Body Labels", initial_value=False)

        self.frame_slider_label = "Motion Frame" if self.playback_full_motion else "Valid Frame"
        self.frame_slider_max = max(0, int(initial_frame_count) - 1)
        self.playback_folder = self.server.gui.add_folder("Playback")
        with self.playback_folder:
            self.frame_slider = self.server.gui.add_slider(
                self.frame_slider_label,
                min=0,
                max=self.frame_slider_max,
                step=1,
                initial_value=0,
            )
            self.playing_cb = self.server.gui.add_checkbox("Playing", initial_value=bool(autoplay_initial))
            self.play_button = self.server.gui.add_button("Play")
            self.pause_button = self.server.gui.add_button("Pause")
            self.fps_number = self.server.gui.add_number("FPS", initial_value=self.playback_fps, min=1, max=240, step=1)
            self.loop_cb = self.server.gui.add_checkbox("Loop", initial_value=bool(loop_initial))
            self.frame_md = self.server.gui.add_markdown("")

        self._register_callbacks()
        self._setup_robot(robot_urdf_path)
        self.load_entry(initial_label)
        threading.Thread(target=self._player_loop, daemon=True).start()

    def _register_frame_slider_callback(self) -> None:
        @self.frame_slider.on_update
        def _(_evt) -> None:
            if self.is_programmatic_slider_update:
                return
            if self.playing:
                return
            self.frame_float = float(self.frame_slider.value)
            self.apply_frame(int(self.frame_slider.value))

    def _set_frame_slider_max(self, max_frame: int) -> None:
        max_frame = max(0, int(max_frame))
        if max_frame == self.frame_slider_max:
            return

        try:
            self.frame_slider.remove()
        except Exception:
            pass
        self.frame_slider_max = max_frame
        with self.playback_folder:
            self.frame_slider = self.server.gui.add_slider(
                self.frame_slider_label,
                min=0,
                max=self.frame_slider_max,
                step=1,
                initial_value=0,
            )
        self._register_frame_slider_callback()

    def _register_callbacks(self) -> None:
        self._register_frame_slider_callback()

        @self.sequence_dropdown.on_update
        def _(_evt) -> None:
            if self.is_programmatic_sequence_update:
                return
            self.load_entry(str(self.sequence_dropdown.value))

        @self.reload_button.on_click
        def _(_evt) -> None:
            self.load_entry(str(self.sequence_dropdown.value))

        @self.playing_cb.on_update
        def _(_evt) -> None:
            self.playing = bool(self.playing_cb.value)

        @self.play_button.on_click
        def _(_evt) -> None:
            self.playing = True
            self.playing_cb.value = True

        @self.pause_button.on_click
        def _(_evt) -> None:
            self.playing = False
            self.playing_cb.value = False

        def _display_changed(_evt) -> None:
            if bool(self.show_body_labels_cb.value) and not self.replay_only:
                self.apply_frame(int(self.frame_slider.value))
            else:
                self._clear_dynamic_body_frame()
            self.apply_visibility()

        for handle in (
            self.show_robot_cb,
            self.show_product_cb,
            self.show_rollout_cb,
            self.show_original_motion_cb,
            self.show_mesh_cb,
            self.show_box_cb,
            self.show_points_cb,
            self.show_paths_cb,
            self.show_body_labels_cb,
        ):
            handle.on_update(_display_changed)

    def clear_scene(self) -> None:
        handles = (
            self.static_handles
            + self.rollout_handles
            + self.mesh_handles
            + self.box_handles
            + self.point_handles
            + self.path_handles
            + self.original_handles
            + self.original_object_handles
            + self.original_path_handles
            + self.current_object_handles
            + self.current_contact_handles
            + self.dynamic_label_handles
        )
        seen_handles: set[int] = set()
        for handle in handles:
            handle_id = id(handle)
            if handle_id in seen_handles:
                continue
            seen_handles.add(handle_id)
            try:
                handle.remove()
            except Exception:
                pass
        if self.dynamic_body_handle is not None:
            try:
                self.dynamic_body_handle.remove()
            except Exception:
                pass
        self.static_handles = []
        self.rollout_handles = []
        self.mesh_handles = []
        self.box_handles = []
        self.point_handles = []
        self.path_handles = []
        self.original_handles = []
        self.original_object_handles = []
        self.original_path_handles = []
        self.current_object_handles = []
        self.current_contact_handles = []
        self.dynamic_label_handles = []
        self.dynamic_body_handle = None

    def _setup_robot(self, robot_urdf_path: Path | None) -> None:
        if robot_urdf_path is None:
            self.robot_motion = RobotMotion(
                path=None,
                joint_pos=None,
                joint_indices=None,
                frame_mode="none",
                status="disabled",
            )
            return
        robot_urdf_path = robot_urdf_path.expanduser().resolve()
        if not robot_urdf_path.exists():
            self.robot_load_error = f"missing robot URDF: {robot_urdf_path}"
            self.robot_motion = RobotMotion(
                path=None,
                joint_pos=None,
                joint_indices=None,
                frame_mode="none",
                status=self.robot_load_error,
            )
            self.show_robot_cb.value = False
            return

        try:
            self.robot_root_handle = self.server.scene.add_frame("/training_g1", show_axes=False)
            self.robot_viser = _create_viser_urdf_handle(
                ViserUrdf,
                self.server,
                robot_urdf_path,
                root_node_name="/training_g1",
            )
            self.robot_joint_names = list(self.robot_viser.get_actuated_joint_names())
            self.robot_viser.update_cfg(np.zeros((len(self.robot_joint_names),), dtype=np.float32))
        except Exception as exc:
            self.robot_load_error = f"failed to load robot URDF: {exc}"
            self.robot_motion = RobotMotion(
                path=None,
                joint_pos=None,
                joint_indices=None,
                frame_mode="none",
                status=self.robot_load_error,
            )
            self.show_robot_cb.value = False
            return

        self.robot_motion = RobotMotion(
            path=None,
            joint_pos=None,
            joint_indices=np.arange(len(self.robot_joint_names), dtype=np.int64),
            frame_mode="reference_root",
            status=f"loaded {robot_urdf_path.name}",
        )

        try:
            self.original_robot_root_handle = self.server.scene.add_frame("/original_g1", show_axes=False)
            try:
                self.original_robot_viser = ViserUrdf(
                    self.server,
                    urdf_or_path=robot_urdf_path,
                    root_node_name="/original_g1",
                    mesh_color_override=_ORIGINAL_G1_MESH_COLOR,
                )
            except TypeError:
                self.original_robot_viser = ViserUrdf(
                    self.server,
                    urdf_path=robot_urdf_path,
                    root_node_name="/original_g1",
                    mesh_color_override=_ORIGINAL_G1_MESH_COLOR,
                )
            self.original_robot_joint_names = list(self.original_robot_viser.get_actuated_joint_names())
            self.original_robot_viser.update_cfg(np.zeros((len(self.original_robot_joint_names),), dtype=np.float32))
            self.original_robot_root_handle.visible = False
            self.original_robot_viser.show_visual = False
        except Exception as exc:
            self.original_robot_load_error = f"failed to load original robot URDF: {exc}"
            self.original_robot_root_handle = None
            self.original_robot_viser = None
            self.original_robot_joint_names = []

    def _load_robot_motion(self, entry: ClipEntry) -> None:
        if self.robot_viser is None:
            return

        motion_bank_path = _resolve_motion_bank_path(entry)
        if motion_bank_path is None:
            self.robot_motion = RobotMotion(
                path=None,
                joint_pos=None,
                joint_indices=np.arange(len(self.robot_joint_names), dtype=np.int64),
                frame_mode="reference_root",
                status="motion_bank missing; showing default G1 pose at reference root",
            )
            return

        try:
            with np.load(motion_bank_path, allow_pickle=True) as data:
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                motion_joint_names = (
                    [_decode_scalar(name) for name in np.asarray(data["joint_names"])]
                    if "joint_names" in data.files
                    else []
                )
        except Exception as exc:
            self.robot_motion = RobotMotion(
                path=motion_bank_path,
                joint_pos=None,
                joint_indices=np.arange(len(self.robot_joint_names), dtype=np.int64),
                frame_mode="reference_root",
                status=f"failed to load motion_bank ({exc}); showing default G1 pose",
            )
            return

        if joint_pos.ndim != 2 or joint_pos.shape[1] < 7:
            self.robot_motion = RobotMotion(
                path=motion_bank_path,
                joint_pos=None,
                joint_indices=np.arange(len(self.robot_joint_names), dtype=np.int64),
                frame_mode="reference_root",
                status=f"invalid motion_bank joint_pos shape {joint_pos.shape}; showing default G1 pose",
            )
            return

        num_motion_joints = max(0, int(joint_pos.shape[1]) - 7)
        joint_indices, error = _joint_indices_for_viser(
            viser_joint_names=self.robot_joint_names,
            motion_joint_names=motion_joint_names,
            num_motion_joints=num_motion_joints,
        )
        if joint_indices is None:
            self.robot_motion = RobotMotion(
                path=motion_bank_path,
                joint_pos=None,
                joint_indices=np.arange(len(self.robot_joint_names), dtype=np.int64),
                frame_mode="reference_root",
                status=f"{error}; showing default G1 pose",
            )
            return

        trajectory_len = int(np.asarray(self.ref.get("trajectory_length", np.asarray(0))).reshape(-1)[0])
        frame_mode = "raw" if joint_pos.shape[0] == trajectory_len else "valid"
        self.robot_motion = RobotMotion(
            path=motion_bank_path,
            joint_pos=joint_pos,
            joint_indices=joint_indices,
            frame_mode=frame_mode,
            status=f"{motion_bank_path.name} ({joint_pos.shape[0]} frames, {num_motion_joints} joints)",
        )

    def _load_original_motion(self, entry: ClipEntry) -> None:
        target_joint_pos = np.asarray(self.ref.get("target_joint_pos", np.zeros((0, 0))), dtype=np.float32)
        target_object_pos = np.asarray(self.ref.get("target_object_pos_local", np.zeros((0, 3))), dtype=np.float32)
        target_object_quat = np.asarray(self.ref.get("target_object_quat_w", np.zeros((0, 4))), dtype=np.float32)
        if target_joint_pos.ndim == 2 and target_joint_pos.shape[1] >= 7 and target_joint_pos.shape[0] > 0:
            motion_joint_names = [_decode_scalar(name) for name in np.asarray(self.ref.get("joint_names", []))]
            num_motion_joints = max(0, int(target_joint_pos.shape[1]) - 7)
            joint_indices = None
            status = f"MotionCommand target ({target_joint_pos.shape[0]} frames, {num_motion_joints} joints)"
            if self.original_robot_viser is not None:
                joint_indices, error = _joint_indices_for_viser(
                    viser_joint_names=self.original_robot_joint_names,
                    motion_joint_names=motion_joint_names,
                    num_motion_joints=num_motion_joints,
                )
                if joint_indices is None:
                    status = f"MotionCommand target: {error}; object-only target overlay"

            object_pos = target_object_pos if target_object_pos.ndim == 2 and target_object_pos.shape[1] == 3 else None
            object_quat = (
                _xyzw_to_wxyz_array(target_object_quat)
                if target_object_quat.ndim == 2 and target_object_quat.shape[1] == 4
                else None
            )
            self.original_motion = OriginalMotion(
                path=None,
                joint_pos=target_joint_pos,
                joint_indices=joint_indices,
                object_pos=object_pos,
                object_quat_wxyz=object_quat,
                status=status,
            )
            return

        original_motion_path = _resolve_original_motion_path(entry, self.original_motion_dir)
        if original_motion_path is None:
            root_msg = str(self.original_motion_dir) if self.original_motion_dir is not None else "<disabled>"
            self.original_motion = OriginalMotion(
                path=None,
                joint_pos=None,
                joint_indices=None,
                object_pos=None,
                object_quat_wxyz=None,
                status=f"missing under {root_msg}",
            )
            return

        try:
            with np.load(original_motion_path, allow_pickle=True) as data:
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32) if "joint_pos" in data.files else None
                object_pos = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data.files else None
                object_quat = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data.files else None
                motion_joint_names = (
                    [_decode_scalar(name) for name in np.asarray(data["joint_names"])]
                    if "joint_names" in data.files
                    else []
                )
        except Exception as exc:
            self.original_motion = OriginalMotion(
                path=original_motion_path,
                joint_pos=None,
                joint_indices=None,
                object_pos=None,
                object_quat_wxyz=None,
                status=f"failed to load original motion ({exc})",
            )
            return

        joint_indices = None
        if (
            self.original_robot_viser is not None
            and joint_pos is not None
            and joint_pos.ndim == 2
            and joint_pos.shape[1] >= 7
        ):
            num_motion_joints = max(0, int(joint_pos.shape[1]) - 7)
            joint_indices, error = _joint_indices_for_viser(
                viser_joint_names=self.original_robot_joint_names,
                motion_joint_names=motion_joint_names,
                num_motion_joints=num_motion_joints,
            )
            if joint_indices is None:
                status = f"{original_motion_path.name}: {error}; object-only original overlay"
            else:
                status = f"{original_motion_path.name} ({joint_pos.shape[0]} frames, {num_motion_joints} joints)"
        else:
            status = f"{original_motion_path.name}: no valid joint_pos; object-only original overlay"

        if object_pos is not None and (object_pos.ndim != 2 or object_pos.shape[1] != 3):
            object_pos = None
        if object_quat is not None and (object_quat.ndim != 2 or object_quat.shape[1] != 4):
            object_quat = None

        self.original_motion = OriginalMotion(
            path=original_motion_path,
            joint_pos=joint_pos,
            joint_indices=joint_indices,
            object_pos=object_pos,
            object_quat_wxyz=object_quat,
            status=status,
        )

    def load_entry(self, label: str) -> None:
        entry = self.entry_by_label[label]
        self.current_entry = entry
        self.clear_scene()

        ref_path = entry.data_dir / "teacher_rollout_reference.npz"
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing teacher rollout reference: {ref_path}")
        with np.load(ref_path, allow_pickle=True) as data:
            self.ref = {key: np.asarray(data[key]) for key in data.files}

        trajectory_length = int(self.ref.get("trajectory_length", np.asarray(0)).reshape(-1)[0])
        if trajectory_length <= 0:
            candidate_lengths = [
                np.asarray(self.ref.get(key, np.zeros((0,)))).shape[0]
                for key in (
                    "body_pos_local",
                    "object_pos_local",
                    "target_joint_pos",
                    "target_object_pos_local",
                )
            ]
            trajectory_length = max(candidate_lengths) if candidate_lengths else 0

        self.body_names = [_decode_scalar(name) for name in np.asarray(self.ref.get("tracked_body_names", []))]
        self._load_robot_motion(entry)
        self._load_original_motion(entry)

        if self.playback_full_motion:
            playback_lengths = [trajectory_length]
            for arr in (
                self.original_motion.joint_pos,
                self.original_motion.object_pos,
                self.original_motion.object_quat_wxyz,
            ):
                if arr is not None and arr.ndim >= 1 and arr.shape[0] > 0:
                    playback_lengths.append(int(arr.shape[0]))
            playback_length = max(playback_lengths) if playback_lengths else trajectory_length
            self.valid_indices = np.arange(playback_length, dtype=np.int64)
        else:
            valid_steps = np.asarray(self.ref.get("valid_steps", np.zeros((0,), dtype=np.bool_)), dtype=np.bool_)
            self.valid_indices = np.flatnonzero(valid_steps)
            if self.valid_indices.size == 0:
                self.valid_indices = np.arange(trajectory_length, dtype=np.int64)

        if self.replay_only:
            self.region_points = {region_name: np.zeros((0, 3), dtype=np.float32) for region_name in _REGION_ORDER}
        else:
            self.region_points = {
                region_name: _as_points(entry.data_dir / f"{region_name}_contact_points.npy")
                for region_name in _REGION_ORDER
            }
        extents = np.asarray(entry.metadata["primitive_extents_xyz"], dtype=np.float32)

        if not self.replay_only:
            self._add_static_product(entry, extents)
        self._add_rollout_view(entry, extents)
        self._add_original_view(entry, extents)

        self._set_frame_slider_max(max(0, int(self.valid_indices.size - 1)))
        print(
            "[INFO] playback "
            f"sequence={entry.clip_id} "
            f"mode={'full_motion' if self.playback_full_motion else 'valid_steps'} "
            f"frames={int(self.valid_indices.size)} "
            f"slider_max={max(0, int(self.valid_indices.size - 1))} "
            f"replay_only={int(self.replay_only)}",
            flush=True,
        )
        self.is_programmatic_slider_update = True
        self.frame_slider.value = 0
        self.is_programmatic_slider_update = False
        self.frame_float = 0.0
        self.fps_number.value = self.playback_fps
        self._last_slider_sync_time = 0.0
        self._last_frame_md_sync_time = 0.0
        self._update_info()
        self.apply_frame(0)
        self.apply_visibility()

    def load_next_entry(self) -> bool:
        if self.current_entry is None:
            return False
        labels = [entry.label for entry in self.entries]
        try:
            current_index = labels.index(self.current_entry.label)
        except ValueError:
            return False
        next_index = current_index + 1
        if next_index >= len(labels):
            return False

        next_label = labels[next_index]
        self.is_programmatic_sequence_update = True
        try:
            self.sequence_dropdown.value = next_label
        finally:
            self.is_programmatic_sequence_update = False
        self.load_entry(next_label)
        self.frame_float = 0.0
        return True

    def _add_static_product(self, entry: ClipEntry, extents: np.ndarray) -> None:
        product_frame = self.server.scene.add_frame(
            "/product",
            show_axes=True,
            axes_length=0.25,
            axes_radius=0.008,
            position=_PRODUCT_OFFSET,
        )
        self.static_handles.append(product_frame)

        object_mesh = _load_object_overlay_mesh(
            clip_id=entry.clip_id,
            object_name=str(entry.metadata.get("object_name", "")),
            object_urdf_path=str(entry.metadata.get("object_urdf_path", "")),
        )
        if object_mesh is not None:
            mesh = object_mesh.copy()
            mesh.visual.vertex_colors = np.tile(np.asarray([175, 185, 200, 210], dtype=np.uint8), (len(mesh.vertices), 1))
            handle = self.server.scene.add_mesh_trimesh("/product/object_mesh", mesh, position=_PRODUCT_OFFSET)
            self.static_handles.append(handle)
            self.mesh_handles.append(handle)

        box_handle = self.server.scene.add_box(
            "/product/primitive_box",
            color=(150, 150, 170),
            dimensions=extents,
            position=_PRODUCT_OFFSET,
        )
        self.static_handles.append(box_handle)
        self.box_handles.append(box_handle)

        for region_name in _DRAWN_CONTACT_REGION_ORDER:
            points = self.region_points[region_name]
            if points.shape[0] == 0:
                continue
            style = _REGION_OVERLAY_STYLE[region_name]
            handle = self.server.scene.add_point_cloud(
                f"/product/points/{region_name}",
                points=points,
                colors=_rgb_tuple(region_name),
                point_size=float(style["scatter_size"]) * 0.0008,
                point_shape="circle",
                position=_PRODUCT_OFFSET,
            )
            self.static_handles.append(handle)
            self.point_handles.append(handle)

        label = self.server.scene.add_label(
            "/product/label",
            text="primitive-frame contact products",
            position=_PRODUCT_OFFSET + np.asarray([0.0, 0.0, float(extents[2]) * 0.75 + 0.18], dtype=np.float32),
        )
        self.static_handles.append(label)

    def _add_rollout_view(self, entry: ClipEntry, extents: np.ndarray) -> None:
        rollout_frame = self.server.scene.add_frame("/rollout", show_axes=True, axes_length=0.35, axes_radius=0.01)
        self.rollout_handles.append(rollout_frame)

        object_mesh = _load_object_overlay_mesh(
            clip_id=entry.clip_id,
            object_name=str(entry.metadata.get("object_name", "")),
            object_urdf_path=str(entry.metadata.get("object_urdf_path", "")),
        )
        if object_mesh is not None:
            mesh = object_mesh.copy()
            mesh.visual.vertex_colors = np.tile(np.asarray([175, 185, 200, 180], dtype=np.uint8), (len(mesh.vertices), 1))
            handle = self.server.scene.add_mesh_trimesh("/rollout/current_object_mesh", mesh)
            self.current_object_handles.append(handle)
            self.rollout_handles.append(handle)
            self.mesh_handles.append(handle)

        box_handle = self.server.scene.add_box(
            "/rollout/current_primitive_box",
            color=(120, 145, 190),
            dimensions=extents,
        )
        self.current_object_handles.append(box_handle)
        self.rollout_handles.append(box_handle)
        self.box_handles.append(box_handle)

        if not self.replay_only:
            for region_name in _DRAWN_CONTACT_REGION_ORDER:
                points = self.region_points[region_name]
                if points.shape[0] == 0:
                    continue
                style = _REGION_OVERLAY_STYLE[region_name]
                handle = self.server.scene.add_point_cloud(
                    f"/rollout/current_points/{region_name}",
                    points=points,
                    colors=_rgb_tuple(region_name),
                    point_size=float(style["scatter_size"]) * 0.0007,
                    point_shape="circle",
                )
                self.current_contact_handles.append(handle)
                self.rollout_handles.append(handle)
                self.point_handles.append(handle)

        obj_pos = np.asarray(self.ref.get("object_pos_local", np.zeros((0, 3))), dtype=np.float32)
        if obj_pos.ndim == 2 and obj_pos.shape[0] > 0 and self.valid_indices.size:
            path_indices = np.clip(self.valid_indices, 0, obj_pos.shape[0] - 1)
            valid_obj_pos = obj_pos[path_indices]
        else:
            valid_obj_pos = np.zeros((0, 3))
        if not self.replay_only and valid_obj_pos.shape[0] >= 2:
            path = _decimate_path(valid_obj_pos)
            handle = self.server.scene.add_spline_catmull_rom(
                "/rollout/object_path",
                positions=path,
                line_width=3.0,
                color=(40, 220, 255),
            )
            self.rollout_handles.append(handle)
            self.path_handles.append(handle)

    def _add_original_view(self, entry: ClipEntry, extents: np.ndarray) -> None:
        original_frame = self.server.scene.add_frame(
            "/original_motion",
            show_axes=True,
            axes_length=0.30,
            axes_radius=0.008,
            visible=False,
        )
        self.original_handles.append(original_frame)

        object_mesh = _load_object_overlay_mesh(
            clip_id=entry.clip_id,
            object_name=str(entry.metadata.get("object_name", "")),
            object_urdf_path=str(entry.metadata.get("object_urdf_path", "")),
        )
        if object_mesh is not None:
            mesh = object_mesh.copy()
            mesh.visual.vertex_colors = np.tile(
                np.asarray([255, 175, 45, 145], dtype=np.uint8),
                (len(mesh.vertices), 1),
            )
            handle = self.server.scene.add_mesh_trimesh(
                "/original_motion/current_object_mesh",
                mesh,
                visible=False,
            )
            self.original_handles.append(handle)
            self.original_object_handles.append(handle)
            self.mesh_handles.append(handle)

        box_handle = self.server.scene.add_box(
            "/original_motion/current_primitive_box",
            color=(255, 170, 40),
            dimensions=extents,
            visible=False,
        )
        self.original_handles.append(box_handle)
        self.original_object_handles.append(box_handle)
        self.box_handles.append(box_handle)

        object_pos = self.original_motion.object_pos
        if not self.replay_only and object_pos is not None and object_pos.shape[0] >= 2:
            path = _decimate_path(object_pos)
            handle = self.server.scene.add_spline_catmull_rom(
                "/original_motion/object_path",
                positions=path,
                line_width=3.0,
                color=(255, 170, 40),
                visible=False,
            )
            self.original_handles.append(handle)
            self.original_path_handles.append(handle)
            self.path_handles.append(handle)

        label = self.server.scene.add_label(
            "/original_motion/label",
            text="original input motion",
            position=np.asarray([0.0, 0.0, float(extents[2]) * 0.75 + 0.18], dtype=np.float32),
            visible=False,
        )
        self.original_handles.append(label)

    def _update_info(self) -> None:
        if self.current_entry is None:
            return
        entry = self.current_entry
        counts = {region: int(points.shape[0]) for region, points in self.region_points.items()}
        status = str(entry.metadata.get("status", "n/a"))
        n_valid = int(self.valid_indices.size)
        glb_path = entry.vis_dir / "contact_overlay.glb"
        png_path = entry.vis_dir / "contact_overlay.png"
        robot_status = self.robot_motion.status
        if self.robot_load_error:
            robot_status = self.robot_load_error
        original_status = self.original_motion.status
        if self.original_robot_load_error:
            original_status = f"{original_status}; {self.original_robot_load_error}"
        points_summary = ", ".join(f"{region} `{counts.get(region, 0)}`" for region in _REGION_ORDER)
        self.info_md.content = (
            f"clip: `{entry.clip_id}`  \n"
            f"dir: `{entry.clip_dir_name}`  \n"
            f"status: `{status}` | valid frames: `{n_valid}`  \n"
            f"training g1: `{robot_status}`  \n"
            f"reference motion: `{original_status}`  \n"
            f"points: {points_summary}  \n"
            f"glb: `{glb_path}`  \n"
            f"png: `{png_path}`"
        )

        interval_path = entry.data_dir / "contact_intervals.json"
        intervals = json.loads(interval_path.read_text(encoding="utf-8")) if interval_path.exists() else {}
        self.interval_md.content = "contact intervals `[t1, t2]`:  \n" + "  \n".join(
            f"- `{region}`: `{intervals.get(region, [-1, -1])}`" for region in _REGION_ORDER
        )

    def apply_visibility(self) -> None:
        static_ids = {id(handle) for handle in self.static_handles}
        rollout_ids = {id(handle) for handle in self.rollout_handles}
        original_ids = {id(handle) for handle in self.original_handles}

        def _base_visible(handle: Any) -> bool:
            visible = True
            handle_id = id(handle)
            if handle_id in static_ids:
                visible = visible and bool(self.show_product_cb.value)
            if handle_id in rollout_ids:
                visible = visible and bool(self.show_rollout_cb.value)
            if handle_id in original_ids:
                visible = visible and bool(self.show_original_motion_cb.value)
            return visible

        for handle in self.static_handles:
            handle.visible = bool(self.show_product_cb.value)
        for handle in self.rollout_handles:
            handle.visible = bool(self.show_rollout_cb.value)
        for handle in self.original_handles:
            handle.visible = bool(self.show_original_motion_cb.value)
        for handle in self.mesh_handles:
            handle.visible = bool(_base_visible(handle) and self.show_mesh_cb.value)
        for handle in self.box_handles:
            handle.visible = bool(_base_visible(handle) and self.show_box_cb.value)
        for handle in self.point_handles:
            handle.visible = bool(_base_visible(handle) and self.show_points_cb.value)
        for handle in self.path_handles:
            handle.visible = bool(_base_visible(handle) and self.show_paths_cb.value)
        for handle in self.dynamic_label_handles:
            handle.visible = bool(self.show_rollout_cb.value and self.show_body_labels_cb.value)
        if self.dynamic_body_handle is not None:
            self.dynamic_body_handle.visible = bool(self.show_rollout_cb.value)
        robot_visible = bool(self.show_robot_cb.value)
        if self.robot_root_handle is not None:
            self.robot_root_handle.visible = robot_visible
        if self.robot_viser is not None:
            try:
                self.robot_viser.show_visual = robot_visible
            except Exception:
                pass
        original_visible = bool(self.show_original_motion_cb.value)
        if self.original_robot_root_handle is not None:
            self.original_robot_root_handle.visible = original_visible
        if self.original_robot_viser is not None:
            try:
                self.original_robot_viser.show_visual = original_visible
            except Exception:
                pass

    def apply_frame(self, slider_index: int) -> None:
        if self.valid_indices.size == 0:
            return
        slider_index = int(np.clip(slider_index, 0, self.valid_indices.size - 1))
        raw_idx = int(self.valid_indices[slider_index])
        obj_pos_arr = np.asarray(self.ref.get("object_pos_local", np.zeros((0, 3))), dtype=np.float32)
        obj_quat_arr = np.asarray(self.ref.get("object_quat_w", np.zeros((0, 4))), dtype=np.float32)
        if (
            obj_pos_arr.ndim != 2
            or obj_pos_arr.shape[0] == 0
            or obj_quat_arr.ndim != 2
            or obj_quat_arr.shape[0] == 0
        ):
            return
        rollout_idx = int(np.clip(raw_idx, 0, min(obj_pos_arr.shape[0], obj_quat_arr.shape[0]) - 1))
        obj_pos = obj_pos_arr[rollout_idx]
        obj_wxyz = _xyzw_to_wxyz(obj_quat_arr[rollout_idx])

        with self.server.atomic():
            for handle in self.current_object_handles:
                handle.position = obj_pos
                handle.wxyz = obj_wxyz
            for handle in self.current_contact_handles:
                handle.position = obj_pos
                handle.wxyz = obj_wxyz

            self._update_robot_frame(slider_index, raw_idx)
            original_idx = self._update_original_frame(raw_idx)
            if bool(self.show_body_labels_cb.value) and not self.replay_only:
                self._update_body_frame(raw_idx)
            elif self.dynamic_body_handle is not None or self.dynamic_label_handles:
                self._clear_dynamic_body_frame()
            original_text = "n/a" if original_idx is None else str(original_idx)
            rollout_text = str(rollout_idx) if rollout_idx == raw_idx else f"{rollout_idx} (clamped from {raw_idx})"
            now = time.perf_counter()
            if not self.playing or now - self._last_frame_md_sync_time >= 0.10:
                self.frame_md.content = (
                    f"motion frame: `{slider_index}` | teacher rollout step: `{rollout_text}` | "
                    f"reference motion step: `{original_text}`"
                )
                self._last_frame_md_sync_time = now

    def _update_robot_frame(self, slider_index: int, raw_idx: int) -> None:
        if self.robot_viser is None or self.robot_root_handle is None:
            return

        joint_pos = self.robot_motion.joint_pos
        joint_indices = self.robot_motion.joint_indices
        if joint_pos is not None and joint_indices is not None and joint_pos.shape[0] > 0:
            if self.robot_motion.frame_mode == "raw":
                motion_idx = int(np.clip(raw_idx, 0, joint_pos.shape[0] - 1))
            else:
                motion_idx = int(np.clip(slider_index, 0, joint_pos.shape[0] - 1))
            q = joint_pos[motion_idx]
            self.robot_root_handle.position = q[:3]
            self.robot_root_handle.wxyz = q[3:7]
            dof = q[7:]
            max_joint_index = int(joint_indices.max()) if joint_indices.size else -1
            if dof.shape[0] > max_joint_index:
                self.robot_viser.update_cfg(dof[joint_indices].astype(np.float32, copy=False))
            return

        root_pos_arr = np.asarray(self.ref.get("root_pos_local", np.zeros((0, 3))), dtype=np.float32)
        root_quat_arr = np.asarray(self.ref.get("root_quat_w", np.zeros((0, 4))), dtype=np.float32)
        if root_pos_arr.shape[0] == 0 or root_quat_arr.shape[0] == 0:
            return
        fallback_idx = int(np.clip(raw_idx, 0, min(root_pos_arr.shape[0], root_quat_arr.shape[0]) - 1))
        self.robot_root_handle.position = root_pos_arr[fallback_idx]
        self.robot_root_handle.wxyz = _xyzw_to_wxyz(root_quat_arr[fallback_idx])
        self.robot_viser.update_cfg(np.zeros((len(self.robot_joint_names),), dtype=np.float32))

    def _update_original_frame(self, raw_idx: int) -> int | None:
        lengths = [
            arr.shape[0]
            for arr in (
                self.original_motion.object_pos,
                self.original_motion.object_quat_wxyz,
                self.original_motion.joint_pos,
            )
            if arr is not None and arr.ndim >= 1 and arr.shape[0] > 0
        ]
        if not lengths:
            return None
        original_idx = int(np.clip(raw_idx, 0, min(lengths) - 1))

        object_pos = self.original_motion.object_pos
        object_quat = self.original_motion.object_quat_wxyz
        if object_pos is not None and object_quat is not None:
            for handle in self.original_object_handles:
                handle.position = object_pos[original_idx]
                handle.wxyz = object_quat[original_idx]

        if self.original_robot_viser is not None and self.original_robot_root_handle is not None:
            joint_pos = self.original_motion.joint_pos
            joint_indices = self.original_motion.joint_indices
            if joint_pos is not None and joint_indices is not None and joint_pos.shape[0] > 0:
                q = joint_pos[original_idx]
                self.original_robot_root_handle.position = q[:3]
                self.original_robot_root_handle.wxyz = q[3:7]
                dof = q[7:]
                max_joint_index = int(joint_indices.max()) if joint_indices.size else -1
                if dof.shape[0] > max_joint_index:
                    self.original_robot_viser.update_cfg(dof[joint_indices].astype(np.float32, copy=False))

        return original_idx

    def _clear_dynamic_body_frame(self) -> None:
        if self.dynamic_body_handle is not None:
            try:
                self.dynamic_body_handle.remove()
            except Exception:
                pass
            self.dynamic_body_handle = None
        for handle in self.dynamic_label_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.dynamic_label_handles = []

    def _update_body_frame(self, raw_idx: int) -> None:
        body_pos = np.asarray(self.ref.get("body_pos_local", np.zeros((0, 0, 3))), dtype=np.float32)
        if body_pos.ndim != 3 or body_pos.shape[0] == 0 or body_pos.shape[1] == 0:
            return

        self._clear_dynamic_body_frame()

        body_idx = int(np.clip(raw_idx, 0, body_pos.shape[0] - 1))
        all_points = body_pos[body_idx]
        selected_points: list[np.ndarray] = []
        selected_colors: list[tuple[int, int, int]] = []
        selected_names: list[str] = []
        for idx, body_name in enumerate(self.body_names[: all_points.shape[0]]):
            region_name = _CONTACT_BODY_NAMES.get(body_name)
            if region_name is None:
                continue
            selected_points.append(all_points[idx])
            selected_colors.append(_rgb_tuple(region_name))
            selected_names.append(body_name)
        if not selected_points:
            return
        points = np.asarray(selected_points, dtype=np.float32)
        colors = np.asarray(selected_colors, dtype=np.uint8)
        self.dynamic_body_handle = self.server.scene.add_point_cloud(
            "/rollout/current_bodies",
            points=points,
            colors=colors,
            point_size=0.035,
            point_shape="circle",
            visible=bool(self.show_rollout_cb.value),
        )

        if self.show_body_labels_cb.value:
            for idx, body_name in enumerate(selected_names):
                label_handle = self.server.scene.add_label(
                    f"/rollout/body_label/{body_name}",
                    text=body_name,
                    position=points[idx] + np.asarray([0.0, 0.0, 0.04], dtype=np.float32),
                    visible=bool(self.show_rollout_cb.value),
                )
                self.dynamic_label_handles.append(label_handle)

    def _player_loop(self) -> None:
        next_tick = time.perf_counter()
        while True:
            if self.playing and self.valid_indices.size > 0:
                now = time.perf_counter()
                if now >= next_tick:
                    fps = max(1.0, float(self.fps_number.value))
                    next_tick = now + 1.0 / fps
                    self.frame_float += 1.0
                    last = float(self.valid_indices.size - 1)
                    if self.frame_float > last:
                        if self.loop_cb.value:
                            self.frame_float = 0.0
                            self._last_slider_sync_time = 0.0
                        elif self.load_next_entry():
                            next_tick = now + 1.0 / fps
                            continue
                        else:
                            self.frame_float = last
                            self.playing = False
                            self.playing_cb.value = False
                    frame_idx = int(self.frame_float)
                    if now - self._last_slider_sync_time >= 0.10 or frame_idx in {0, int(last)}:
                        self.is_programmatic_slider_update = True
                        self.frame_slider.value = frame_idx
                        self.is_programmatic_slider_update = False
                        self._last_slider_sync_time = now
                    self.apply_frame(frame_idx)
            if self.playing:
                sleep_s = max(0.001, min(0.02, next_tick - time.perf_counter()))
            else:
                sleep_s = 0.02
            time.sleep(sleep_s)

    def run_forever(self) -> None:
        print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
        while True:
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug viewer for exported teacher box-contact rollout products.")
    parser.add_argument("--data-root", default="outputs", help="Directory containing clips/*.npy and rollout refs.")
    parser.add_argument("--vis-root", default="outputs_vis", help="Directory containing clips/contact_overlay.{glb,png}.")
    parser.add_argument("--stats-root", default="outputs_sts", help="Directory containing clips/metadata.json.")
    parser.add_argument(
        "--original-motion-dir",
        default=None,
        help="Original input motion directory, e.g. data/ds_box_data/train_g1_w_obj_prepared.",
    )
    parser.add_argument("--sequence", default=None, help="Initial sequence label, clip id, or clip directory name.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=None, help="Viser port. Defaults to VISER_PORT or Viser default.")
    parser.add_argument(
        "--robot-urdf",
        default=str(_DEFAULT_ROBOT_URDF_PATH),
        help="URDF used to draw the training G1 robot. Defaults to the object-generalist training G1.",
    )
    parser.add_argument("--no-robot", action="store_true", help="Disable the training G1 robot overlay.")
    parser.add_argument(
        "--show-original-motion",
        action="store_true",
        help="Show the original input/reference motion overlay at startup.",
    )
    parser.add_argument(
        "--show-primitive-box",
        action="store_true",
        help="Show the primitive AABB box overlay at startup.",
    )
    parser.add_argument("--autoplay", action="store_true", help="Start playback automatically.")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop the selected clip.")
    parser.add_argument("--fps", type=float, default=30.0, help="Initial playback FPS.")
    parser.add_argument(
        "--playback-full-motion",
        action="store_true",
        help="Use the full rollout/reference trajectory for playback instead of filtering to valid_steps.",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Only replay the rollout/reference objects and G1 meshes; skip contact/path/static debug overlays.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only show clips listed in success_clips.txt or marked success=True.",
    )
    parser.add_argument(
        "--strict-success-only",
        action="store_true",
        help="Only show clips with stable contact and final-position success.",
    )
    parser.add_argument(
        "--solid-only",
        action="store_true",
        help="Only show box/bin/barrel/ball clips.",
    )
    parser.add_argument(
        "--exclude-clip",
        action="append",
        default=[],
        help="Clip id or clip directory name to remove from the dropdown. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-clips-file",
        default=None,
        help="Text file containing one clip id per line to remove from the dropdown.",
    )
    parser.add_argument("--list-only", action="store_true", help="List available sequences and exit.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    vis_root = Path(args.vis_root).expanduser().resolve()
    stats_root = Path(args.stats_root).expanduser().resolve()
    original_motion_dir = (
        Path(args.original_motion_dir).expanduser().resolve() if args.original_motion_dir is not None else None
    )
    robot_urdf_path = None if args.no_robot else Path(args.robot_urdf).expanduser().resolve()
    exclude_clip_ids = _load_exclude_clip_ids(args)
    entries = _filter_entries(
        _load_entries(data_root, vis_root, stats_root),
        data_root=data_root,
        stats_root=stats_root,
        success_only=bool(args.success_only),
        strict_success_only=bool(args.strict_success_only),
        solid_only=bool(args.solid_only),
        exclude_clip_ids=exclude_clip_ids,
    )
    if args.list_only:
        for entry in entries:
            print(entry.label)
        return

    port = int(args.port) if args.port is not None else int(resolve_viser_port())
    viewer = DebugRolloutViewer(
        data_root=data_root,
        vis_root=vis_root,
        stats_root=stats_root,
        initial_sequence=args.sequence,
        host=str(args.host),
        port=port,
        robot_urdf_path=robot_urdf_path,
        original_motion_dir=original_motion_dir,
        show_original_motion_initial=bool(args.show_original_motion),
        show_primitive_box_initial=bool(args.show_primitive_box),
        autoplay_initial=bool(args.autoplay),
        loop_initial=not bool(args.no_loop),
        playback_full_motion=bool(args.playback_full_motion),
        replay_only=bool(args.replay_only),
        playback_fps=float(args.fps),
        success_only=bool(args.success_only),
        strict_success_only=bool(args.strict_success_only),
        solid_only=bool(args.solid_only),
        exclude_clip_ids=exclude_clip_ids,
    )
    viewer.run_forever()


if __name__ == "__main__":
    main()
