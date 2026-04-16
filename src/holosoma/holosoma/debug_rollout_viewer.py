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

from holosoma.utils.object_pose_correction import (  # noqa: E402
    get_omomo_largebox_primitive_fit_local_correction_wxyz_np,
    is_omomo_largebox_clip,
)
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402


_REGION_ORDER = ("left_wrist", "right_wrist", "arm", "torso")
_CONTACT_BODY_NAMES = {
    "left_wrist_yaw_link": "left_wrist",
    "right_wrist_yaw_link": "right_wrist",
    "left_elbow_link": "arm",
    "right_elbow_link": "arm",
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
    "arm": {
        "rgba": np.asarray([255, 165, 0, 255], dtype=np.uint8),
        "scatter_size": 36.0,
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

    if is_omomo_largebox_clip(clip_id, object_name, object_urdf_path):
        correction_wxyz = get_omomo_largebox_primitive_fit_local_correction_wxyz_np().astype(np.float64)
        rot_inv = _quat_to_rotmat_wxyz(correction_wxyz).T
        transformed = mesh.copy()
        transformed.vertices = np.asarray(transformed.vertices, dtype=np.float64) @ rot_inv.T
        mesh = transformed

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
    ) -> None:
        self.data_root = data_root
        self.vis_root = vis_root
        self.stats_root = stats_root
        self.entries = _load_entries(data_root, vis_root, stats_root)
        self.entry_by_label = {entry.label: entry for entry in self.entries}
        self.current_entry: ClipEntry | None = None

        self.server = viser.ViserServer(host=host, port=port, label="debug_rollout")
        self.server.scene.add_grid("/grid", width=6.0, height=6.0, position=(0.0, 0.0, 0.0))

        self.static_handles: list[Any] = []
        self.rollout_handles: list[Any] = []
        self.mesh_handles: list[Any] = []
        self.box_handles: list[Any] = []
        self.point_handles: list[Any] = []
        self.path_handles: list[Any] = []
        self.dynamic_body_handle: Any | None = None
        self.dynamic_label_handles: list[Any] = []
        self.current_object_handles: list[Any] = []
        self.current_contact_handles: list[Any] = []

        self.ref: dict[str, np.ndarray] = {}
        self.valid_indices = np.zeros((0,), dtype=np.int64)
        self.body_names: list[str] = []
        self.region_points: dict[str, np.ndarray] = {}
        self.is_programmatic_slider_update = False
        self.playing = False
        self.frame_float = 0.0

        initial_label = self.entries[0].label
        if initial_sequence:
            for entry in self.entries:
                if initial_sequence in {entry.label, entry.clip_id, entry.clip_dir_name}:
                    initial_label = entry.label
                    break

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
            self.show_product_cb = self.server.gui.add_checkbox("Static Product Overlay", initial_value=True)
            self.show_rollout_cb = self.server.gui.add_checkbox("Rollout View", initial_value=True)
            self.show_mesh_cb = self.server.gui.add_checkbox("Object Mesh", initial_value=True)
            self.show_box_cb = self.server.gui.add_checkbox("Primitive Box", initial_value=True)
            self.show_points_cb = self.server.gui.add_checkbox("Contact Points", initial_value=True)
            self.show_paths_cb = self.server.gui.add_checkbox("Object/Body Paths", initial_value=True)
            self.show_body_labels_cb = self.server.gui.add_checkbox("Body Labels", initial_value=False)

        with self.server.gui.add_folder("Playback"):
            self.frame_slider = self.server.gui.add_slider("Valid Frame", min=0, max=1, step=1, initial_value=0)
            self.play_button = self.server.gui.add_button("Play / Pause")
            self.fps_number = self.server.gui.add_number("FPS", initial_value=30, min=1, max=240, step=1)
            self.loop_cb = self.server.gui.add_checkbox("Loop", initial_value=True)
            self.frame_md = self.server.gui.add_markdown("")

        self._register_callbacks()
        self.load_entry(initial_label)
        threading.Thread(target=self._player_loop, daemon=True).start()

    def _register_callbacks(self) -> None:
        @self.sequence_dropdown.on_update
        def _(_evt) -> None:
            self.load_entry(str(self.sequence_dropdown.value))

        @self.reload_button.on_click
        def _(_evt) -> None:
            self.load_entry(str(self.sequence_dropdown.value))

        @self.frame_slider.on_update
        def _(_evt) -> None:
            if self.is_programmatic_slider_update:
                return
            self.playing = False
            self.frame_float = float(self.frame_slider.value)
            self.apply_frame(int(self.frame_slider.value))

        @self.play_button.on_click
        def _(_evt) -> None:
            self.playing = not self.playing

        for handle in (
            self.show_product_cb,
            self.show_rollout_cb,
            self.show_mesh_cb,
            self.show_box_cb,
            self.show_points_cb,
            self.show_paths_cb,
            self.show_body_labels_cb,
        ):
            handle.on_update(lambda _evt: self.apply_frame(int(self.frame_slider.value)))

    def clear_scene(self) -> None:
        for handle in (
            self.static_handles
            + self.rollout_handles
            + self.mesh_handles
            + self.box_handles
            + self.point_handles
            + self.path_handles
            + self.current_object_handles
            + self.current_contact_handles
            + self.dynamic_label_handles
        ):
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
        self.current_object_handles = []
        self.current_contact_handles = []
        self.dynamic_label_handles = []
        self.dynamic_body_handle = None

    def load_entry(self, label: str) -> None:
        entry = self.entry_by_label[label]
        self.current_entry = entry
        self.clear_scene()

        ref_path = entry.data_dir / "teacher_rollout_reference.npz"
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing teacher rollout reference: {ref_path}")
        with np.load(ref_path, allow_pickle=True) as data:
            self.ref = {key: np.asarray(data[key]) for key in data.files}

        valid_steps = np.asarray(self.ref.get("valid_steps", np.zeros((0,), dtype=np.bool_)), dtype=np.bool_)
        self.valid_indices = np.flatnonzero(valid_steps)
        if self.valid_indices.size == 0:
            self.valid_indices = np.arange(int(self.ref.get("trajectory_length", np.asarray(0)).reshape(-1)[0]))
        self.body_names = [_decode_scalar(name) for name in np.asarray(self.ref.get("tracked_body_names", []))]

        self.region_points = {
            region_name: _as_points(entry.data_dir / f"{region_name}_contact_points.npy")
            for region_name in _REGION_ORDER
        }
        extents = np.asarray(entry.metadata["primitive_extents_xyz"], dtype=np.float32)

        self._add_static_product(entry, extents)
        self._add_rollout_view(entry, extents)

        self.frame_slider.max = max(0, int(self.valid_indices.size - 1))
        self.is_programmatic_slider_update = True
        self.frame_slider.value = 0
        self.is_programmatic_slider_update = False
        self.frame_float = 0.0
        self.fps_number.value = 30
        self._update_info()
        self.apply_frame(0)
        self.apply_visibility()

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

        for region_name in _REGION_ORDER:
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

        for region_name in _REGION_ORDER:
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
        valid_obj_pos = obj_pos[self.valid_indices] if obj_pos.ndim == 2 and self.valid_indices.size else np.zeros((0, 3))
        if valid_obj_pos.shape[0] >= 2:
            path = _decimate_path(valid_obj_pos)
            handle = self.server.scene.add_spline_catmull_rom(
                "/rollout/object_path",
                positions=path,
                line_width=3.0,
                color=(40, 220, 255),
            )
            self.rollout_handles.append(handle)
            self.path_handles.append(handle)

        body_pos = np.asarray(self.ref.get("body_pos_local", np.zeros((0, 0, 3))), dtype=np.float32)
        if body_pos.ndim == 3 and body_pos.shape[0] > 0 and self.body_names:
            body_name_to_idx = {name: idx for idx, name in enumerate(self.body_names)}
            for body_name, region_name in _CONTACT_BODY_NAMES.items():
                idx = body_name_to_idx.get(body_name)
                if idx is None:
                    continue
                positions = body_pos[self.valid_indices, idx, :] if self.valid_indices.size else body_pos[:, idx, :]
                if positions.shape[0] < 2:
                    continue
                handle = self.server.scene.add_spline_catmull_rom(
                    f"/rollout/body_path/{body_name}",
                    positions=_decimate_path(positions),
                    line_width=2.0,
                    color=_rgb_tuple(region_name),
                )
                self.rollout_handles.append(handle)
                self.path_handles.append(handle)

    def _update_info(self) -> None:
        if self.current_entry is None:
            return
        entry = self.current_entry
        counts = {region: int(points.shape[0]) for region, points in self.region_points.items()}
        status = str(entry.metadata.get("status", "n/a"))
        n_valid = int(self.valid_indices.size)
        glb_path = entry.vis_dir / "contact_overlay.glb"
        png_path = entry.vis_dir / "contact_overlay.png"
        self.info_md.content = (
            f"clip: `{entry.clip_id}`  \n"
            f"dir: `{entry.clip_dir_name}`  \n"
            f"status: `{status}` | valid frames: `{n_valid}`  \n"
            f"points: left_wrist `{counts['left_wrist']}`, right_wrist `{counts['right_wrist']}`, "
            f"arm `{counts['arm']}`, torso `{counts['torso']}`  \n"
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

        def _base_visible(handle: Any) -> bool:
            visible = True
            handle_id = id(handle)
            if handle_id in static_ids:
                visible = visible and bool(self.show_product_cb.value)
            if handle_id in rollout_ids:
                visible = visible and bool(self.show_rollout_cb.value)
            return visible

        for handle in self.static_handles:
            handle.visible = bool(self.show_product_cb.value)
        for handle in self.rollout_handles:
            handle.visible = bool(self.show_rollout_cb.value)
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

    def apply_frame(self, slider_index: int) -> None:
        if self.valid_indices.size == 0:
            return
        slider_index = int(np.clip(slider_index, 0, self.valid_indices.size - 1))
        raw_idx = int(self.valid_indices[slider_index])
        obj_pos_arr = np.asarray(self.ref.get("object_pos_local", np.zeros((0, 3))), dtype=np.float32)
        obj_quat_arr = np.asarray(self.ref.get("object_quat_w", np.zeros((0, 4))), dtype=np.float32)
        if obj_pos_arr.shape[0] <= raw_idx or obj_quat_arr.shape[0] <= raw_idx:
            return
        obj_pos = obj_pos_arr[raw_idx]
        obj_wxyz = _xyzw_to_wxyz(obj_quat_arr[raw_idx])

        with self.server.atomic():
            for handle in self.current_object_handles:
                handle.position = obj_pos
                handle.wxyz = obj_wxyz
            for handle in self.current_contact_handles:
                handle.position = obj_pos
                handle.wxyz = obj_wxyz

            self._update_body_frame(raw_idx)
            self.frame_md.content = f"slider frame: `{slider_index}` | raw rollout step: `{raw_idx}`"
            self.apply_visibility()

    def _update_body_frame(self, raw_idx: int) -> None:
        body_pos = np.asarray(self.ref.get("body_pos_local", np.zeros((0, 0, 3))), dtype=np.float32)
        if body_pos.ndim != 3 or body_pos.shape[0] <= raw_idx or body_pos.shape[1] == 0:
            return

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

        points = body_pos[raw_idx]
        colors = np.tile(np.asarray([180, 180, 185], dtype=np.uint8), (points.shape[0], 1))
        for idx, body_name in enumerate(self.body_names[: points.shape[0]]):
            region_name = _CONTACT_BODY_NAMES.get(body_name)
            if region_name is not None:
                colors[idx] = np.asarray(_rgb_tuple(region_name), dtype=np.uint8)
        self.dynamic_body_handle = self.server.scene.add_point_cloud(
            "/rollout/current_bodies",
            points=points,
            colors=colors,
            point_size=0.035,
            point_shape="circle",
            visible=bool(self.show_rollout_cb.value),
        )

        if self.show_body_labels_cb.value:
            for idx, body_name in enumerate(self.body_names[: points.shape[0]]):
                if body_name not in _CONTACT_BODY_NAMES:
                    continue
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
                        else:
                            self.frame_float = last
                            self.playing = False
                    frame_idx = int(self.frame_float)
                    self.is_programmatic_slider_update = True
                    self.frame_slider.value = frame_idx
                    self.is_programmatic_slider_update = False
                    self.apply_frame(frame_idx)
            time.sleep(0.005)

    def run_forever(self) -> None:
        print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
        while True:
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug viewer for exported teacher box-contact rollout products.")
    parser.add_argument("--data-root", default="outputs", help="Directory containing clips/*.npy and rollout refs.")
    parser.add_argument("--vis-root", default="outputs_vis", help="Directory containing clips/contact_overlay.{glb,png}.")
    parser.add_argument("--stats-root", default="outputs_sts", help="Directory containing clips/metadata.json.")
    parser.add_argument("--sequence", default=None, help="Initial sequence label, clip id, or clip directory name.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=None, help="Viser port. Defaults to VISER_PORT or Viser default.")
    parser.add_argument("--list-only", action="store_true", help="List available sequences and exit.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    vis_root = Path(args.vis_root).expanduser().resolve()
    stats_root = Path(args.stats_root).expanduser().resolve()
    entries = _load_entries(data_root, vis_root, stats_root)
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
    )
    viewer.run_forever()


if __name__ == "__main__":
    main()
