from __future__ import annotations

from contextlib import nullcontext
import functools
import hashlib
import inspect
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.object_geometry import load_urdf_geometry_extents
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    matrix_to_quaternion,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    quaternion_to_matrix,
    yaw_quat,
)
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port

# UC Berkeley palette for viewer overrides.
CALIFORNIA_GOLD = (253, 181, 21)

LIGHT_BLUE = CALIFORNIA_GOLD
TERRAIN_GRAY = (70, 70, 70)
GROUND_DARK_GRAY = (45, 45, 45)
OBJECT_MESH_COLOR = CALIFORNIA_GOLD
SIM_VISUAL_MESH_COLOR = (70, 160, 255)
SIM_COLLISION_MESH_COLOR = (255, 120, 70)
SIM_ROBOT_POINTS_COLOR = np.array([70, 190, 120], dtype=np.uint8)
SIM_OBJECT_POINTS_COLOR = np.array(CALIFORNIA_GOLD, dtype=np.uint8)
HEIGHTMAP_MARKER_COLOR = (255, 165, 0)
CAMERA_MARKER_COLOR = (0, 255, 255)
COMMAND_ARROW_COLOR = np.array([255, 165, 0], dtype=np.uint8)
TARGET_BOX_COLOR = CALIFORNIA_GOLD
FUTURE_GOAL_BOX_COLOR = (255, 0, 0)
SENSOR_MARKER_RADIUS = 0.03
SIM_MESH_MODE_OPTIONS = ("visual", "collision", "both", "none")
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def _normalize_viser_image_format(image_format: str, *, faithful_mode: bool = False) -> str:
    normalized = str(image_format).strip().lower()
    if normalized == "auto":
        return "png" if faithful_mode else "jpeg"
    if normalized in {"jpeg", "png"}:
        return normalized
    return "png" if faithful_mode else "jpeg"


def _call_viser_method_compat(method: Any, *args, **kwargs) -> Any:
    normalized_kwargs = dict(kwargs)
    if "format" in normalized_kwargs:
        normalized_kwargs["format"] = _normalize_viser_image_format(normalized_kwargs["format"])

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        params = signature.parameters
        accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        if not accepts_var_kwargs:
            normalized_kwargs = {key: value for key, value in normalized_kwargs.items() if key in params}

    return method(*args, **normalized_kwargs)


_VIRIDIS_LUT = np.array(
    [
        (68, 1, 84),
        (69, 6, 90),
        (70, 12, 95),
        (71, 18, 101),
        (71, 24, 106),
        (72, 29, 111),
        (72, 34, 115),
        (71, 39, 119),
        (71, 44, 123),
        (70, 49, 126),
        (69, 54, 129),
        (67, 59, 131),
        (66, 64, 133),
        (64, 68, 135),
        (62, 73, 137),
        (60, 77, 138),
        (58, 82, 139),
        (56, 86, 139),
        (54, 90, 139),
        (52, 94, 139),
        (50, 98, 139),
        (49, 102, 138),
        (47, 105, 137),
        (46, 109, 136),
        (45, 113, 135),
        (43, 117, 134),
        (42, 120, 133),
        (41, 124, 132),
        (40, 127, 130),
        (40, 131, 129),
        (39, 134, 128),
        (38, 137, 126),
        (38, 140, 125),
        (38, 144, 123),
        (37, 147, 121),
        (37, 150, 120),
        (37, 153, 118),
        (36, 156, 116),
        (36, 159, 114),
        (36, 162, 112),
        (36, 165, 110),
        (36, 168, 108),
        (35, 171, 106),
        (35, 174, 104),
        (35, 177, 102),
        (35, 180, 100),
        (35, 183, 98),
        (35, 186, 96),
        (35, 189, 94),
        (35, 192, 92),
        (35, 194, 90),
        (35, 197, 88),
        (35, 200, 86),
        (35, 203, 84),
        (35, 206, 82),
        (35, 209, 80),
        (35, 211, 78),
        (35, 214, 76),
        (35, 217, 74),
        (35, 220, 72),
        (35, 222, 70),
        (35, 225, 68),
        (35, 228, 66),
        (35, 230, 64),
        (35, 233, 62),
        (35, 235, 60),
        (35, 238, 58),
        (36, 240, 56),
        (36, 243, 54),
        (36, 245, 52),
        (36, 248, 50),
        (36, 250, 48),
        (36, 252, 46),
        (37, 255, 44),
    ],
    dtype=np.uint8,
)


class _ViserSceneCompat:
    """Compat adapter for Viser versions that expose scene APIs on the server directly."""

    def __init__(self, server: Any, scene: Any | None = None) -> None:
        self._server = server
        self._scene = scene

    def add_line_segments(self, *args, **kwargs) -> Any:
        target = self._scene if self._scene is not None and hasattr(self._scene, "add_line_segments") else self._server
        if hasattr(target, "add_line_segments"):
            return _call_viser_method_compat(target.add_line_segments, *args, **kwargs)

        logger.debug("Current Viser version has no add_line_segments(); skipping line-segment overlay.")
        return _NoopViserHandle()

    def __getattr__(self, name: str) -> Any:
        if self._scene is not None and hasattr(self._scene, name):
            attr = getattr(self._scene, name)
        else:
            attr = getattr(self._server, name)

        if callable(attr) and name.startswith("add_"):
            @functools.wraps(attr)
            def _wrapped(*args, **kwargs) -> Any:
                return _call_viser_method_compat(attr, *args, **kwargs)

            return _wrapped

        return attr


class _NoopViserHandle:
    """Small mutable placeholder for optional Viser overlays unsupported by old Viser."""

    def __init__(self) -> None:
        self.visible = False
        self.points = None
        self.colors = None
        self.position = None
        self.wxyz = None

    def remove(self) -> None:
        return None


class _ViserCompatImageHandle:
    """Image handle for old Viser versions whose add_image() handles are immutable."""

    def __init__(
        self,
        server: Any,
        *,
        node_name: str,
        image: np.ndarray,
        render_width: float,
        render_height: float,
        image_format: str,
        jpeg_quality: int | None,
        visible: bool,
    ) -> None:
        self._server = server
        self._node_name = node_name
        self._render_width = float(render_width)
        self._render_height = float(render_height)
        self._format = image_format
        self._jpeg_quality = jpeg_quality
        self._visible = bool(visible)
        self._handle = None
        self._image = np.asarray(image)
        self._recreate()

    def _recreate(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        kwargs = {
            "image": self._image,
            "render_width": self._render_width,
            "render_height": self._render_height,
            "format": self._format,
            "visible": self._visible,
        }
        if self._jpeg_quality is not None:
            kwargs["jpeg_quality"] = self._jpeg_quality
        self._handle = _call_viser_method_compat(self._server.add_image, self._node_name, **kwargs)

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, value: np.ndarray) -> None:
        self._image = np.asarray(value)
        self._recreate()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)
        if self._handle is not None:
            try:
                self._handle.visible = self._visible
            except Exception:
                self._recreate()

    def remove(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None


class _ViserGuiCompat:
    """Compat adapter for Viser versions that use add_gui_* on the server."""

    def __init__(self, server: Any, gui: Any | None = None) -> None:
        self._server = server
        self._gui = gui

    def add_image(self, *args, **kwargs) -> Any:
        if self._gui is not None and hasattr(self._gui, "add_image"):
            return _call_viser_method_compat(self._gui.add_image, *args, **kwargs)
        if hasattr(self._server, "add_gui_image"):
            return _call_viser_method_compat(self._server.add_gui_image, *args, **kwargs)

        image = None
        if args:
            image = args[0]
        if image is None:
            image = kwargs.pop("image", None)
        if image is None:
            raise TypeError("add_image requires an image array")

        label = str(kwargs.pop("label", "Perception Image"))
        image_format = _normalize_viser_image_format(kwargs.pop("format", "jpeg"))
        jpeg_quality = kwargs.pop("jpeg_quality", None)
        visible = bool(kwargs.pop("visible", True))

        render_height = float(kwargs.pop("render_height", 0.30))
        render_width = kwargs.pop("render_width", None)
        if render_width is None:
            try:
                h = int(getattr(image, "shape", [1, 1])[0])
                w = int(getattr(image, "shape", [1, 1])[1])
                aspect = float(w) / float(max(1, h))
                render_width = render_height * aspect
            except Exception:
                render_width = render_height
        render_width = float(render_width)

        node_name = "/viser_gui_compat/" + "".join(ch if ch.isalnum() else "_" for ch in label)
        return _ViserCompatImageHandle(
            self._server,
            node_name=node_name,
            image=image,
            render_width=render_width,
            render_height=render_height,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            visible=visible,
        )

    def __getattr__(self, name: str) -> Any:
        if self._gui is not None and hasattr(self._gui, name):
            return getattr(self._gui, name)
        if name.startswith("add_"):
            legacy_name = f"add_gui_{name[4:]}"
            if hasattr(self._server, legacy_name):
                return getattr(self._server, legacy_name)
        return getattr(self._server, name)


def _ensure_viser_api_compat(server: Any) -> None:
    """Attach scene/gui adapters when running against older Viser APIs."""

    raw_scene = getattr(server, "scene", None)
    if not isinstance(raw_scene, _ViserSceneCompat):
        server.scene = _ViserSceneCompat(server, raw_scene)  # type: ignore[attr-defined]

    raw_gui = getattr(server, "gui", None)
    if not isinstance(raw_gui, _ViserGuiCompat):
        server.gui = _ViserGuiCompat(server, raw_gui)  # type: ignore[attr-defined]


def _create_viser_urdf_handle(
    viser_urdf_cls: Any,
    target: Any,
    urdf_path: str | Path,
    *,
    root_node_name: str,
) -> Any:
    """Create ViserUrdf across API variants."""

    path_obj = Path(urdf_path)
    constructors = (
        lambda: viser_urdf_cls(target, urdf_or_path=path_obj, root_node_name=root_node_name),
        lambda: viser_urdf_cls(target, urdf_path=path_obj, root_node_name=root_node_name),
        lambda: viser_urdf_cls(target, path_obj, root_node_name=root_node_name),
    )
    last_error: Exception | None = None
    for factory in constructors:
        try:
            return factory()
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to construct ViserUrdf handle.")


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    scaled = values * float(len(_VIRIDIS_LUT) - 1)
    idx0 = np.floor(scaled).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(_VIRIDIS_LUT) - 1)
    t = (scaled - idx0)[..., None]
    c0 = _VIRIDIS_LUT[idx0].astype(np.float32)
    c1 = _VIRIDIS_LUT[idx1].astype(np.float32)
    return ((1.0 - t) * c0 + t * c1).astype(np.uint8)


def _valid_depth_stats(depth: np.ndarray, near: float, far: float) -> tuple[float | None, float | None, int]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if not np.any(valid):
        return None, None, 0
    depth_valid = depth[valid]
    return float(depth_valid.min()), float(depth_valid.max()), int(depth_valid.size)


def _depth_crc32(depth: np.ndarray, far: float) -> str:
    depth = np.asarray(depth, dtype=np.float32)
    missing_fill = np.float32(far + 1.0)
    payload = np.nan_to_num(depth, nan=missing_fill, posinf=missing_fill, neginf=missing_fill)
    checksum = zlib.crc32(payload.tobytes()) & 0xFFFFFFFF
    return f"{checksum:08x}"


def _depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)

    depth_clipped = np.clip(depth, near, far)
    depth_valid = depth_clipped[valid]
    min_d = float(depth_valid.min())
    max_d = float(depth_valid.max())
    denom = max(max_d - min_d, 1.0e-6)
    norm = (depth_clipped - min_d) / denom
    norm = np.where(valid, norm, 0.0)
    colored = _apply_colormap(norm)
    colored[~valid] = 0
    return colored


def _depth_to_rgb_fixed_range(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if far <= near + 1.0e-6:
        far = near + 1.0
    depth_clipped = np.clip(depth, near, far)
    norm = (depth_clipped - near) / max(far - near, 1.0e-6)
    norm = np.where(valid, norm, 0.0)
    colored = _apply_colormap(norm)
    colored[~valid] = 0
    return colored


def _normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp(min=1.0e-6)


def _parse_quat_wxyz(raw) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    try:
        vals = [float(v) for v in raw]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    return (vals[0], vals[1], vals[2], vals[3])


def _axis_override(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _frustum_quat_from_camera(cam_quat_xyzw: torch.Tensor) -> torch.Tensor:
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=cam_quat_xyzw.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=cam_quat_xyzw.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=cam_quat_xyzw.device)

    x_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), x_axis.unsqueeze(0), w_last=True).squeeze(0)
    y_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), y_axis.unsqueeze(0), w_last=True).squeeze(0)
    z_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), z_axis.unsqueeze(0), w_last=True).squeeze(0)

    # Viser frustum expects OpenCV: +Z forward, +X right, +Y down.
    # Perception camera uses USD axes (x right, y up, -z forward) when camera_frame_quat is set.
    x_right = _normalize_vec(x_cam)
    y_down = _normalize_vec(-y_cam)
    z_fwd = _normalize_vec(-z_cam)

    rot = torch.stack([x_right, y_down, z_fwd], dim=1)
    quat_wxyz = matrix_to_quaternion(rot)
    return quat_wxyz


def _frustum_quat_from_rays(
    ray_dirs_base: torch.Tensor,
    body_quat_xyzw: torch.Tensor,
    *,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor | None:
    if ray_dirs_base is None or ray_dirs_base.numel() == 0:
        return None

    # Keep math on a single device/dtype to avoid CUDA/CPU mismatches.
    target_device = ray_dirs_base.device
    target_dtype = ray_dirs_base.dtype
    if body_quat_xyzw.device != target_device or body_quat_xyzw.dtype != target_dtype:
        body_quat_xyzw = body_quat_xyzw.to(device=target_device, dtype=target_dtype)

    num_rays = int(ray_dirs_base.shape[0])
    if width and height and num_rays >= (width * height):
        center_v = height // 2
        center_u = width // 2
        idx_center = center_v * width + center_u
        idx_right = center_v * width + min(center_u + 1, width - 1)
    else:
        idx_center = num_rays // 2
        idx_right = min(idx_center + 1, num_rays - 1)

    center_dir = ray_dirs_base[idx_center]
    right_dir = ray_dirs_base[idx_right]

    if body_quat_xyzw.ndim == 1:
        body_quat_xyzw = body_quat_xyzw.unsqueeze(0)
    center_world = quat_apply(body_quat_xyzw, center_dir.unsqueeze(0), w_last=True).squeeze(0)
    right_world = quat_apply(body_quat_xyzw, right_dir.unsqueeze(0), w_last=True).squeeze(0)

    fwd = _normalize_vec(center_world)
    right_proj = right_world - fwd * torch.dot(right_world, fwd)
    if torch.linalg.norm(right_proj) < 1.0e-6:
        fallback_right = torch.tensor([0.0, -1.0, 0.0], device=body_quat_xyzw.device)
        right_proj = quat_apply(body_quat_xyzw, fallback_right.unsqueeze(0), w_last=True).squeeze(0)
        right_proj = right_proj - fwd * torch.dot(right_proj, fwd)
        if torch.linalg.norm(right_proj) < 1.0e-6:
            return None

    right = _normalize_vec(right_proj)
    down = _normalize_vec(torch.cross(fwd, right))

    rot = torch.stack([right, down, fwd], dim=1)
    return matrix_to_quaternion(rot)


def _frustum_quat_from_world_rays(
    ray_dirs_world: torch.Tensor,
    *,
    width: int | None = None,
    height: int | None = None,
) -> torch.Tensor | None:
    if ray_dirs_world is None or ray_dirs_world.numel() == 0:
        return None
    if ray_dirs_world.ndim != 2 or ray_dirs_world.shape[-1] != 3:
        return None

    num_rays = int(ray_dirs_world.shape[0])
    if width and height and num_rays >= (width * height):
        center_v = height // 2
        center_u = width // 2
        idx_center = center_v * width + center_u
        idx_right = center_v * width + min(center_u + 1, width - 1)
    else:
        idx_center = num_rays // 2
        idx_right = min(idx_center + 1, num_rays - 1)

    center_dir = ray_dirs_world[idx_center]
    right_dir = ray_dirs_world[idx_right]

    # Use camera center ray as frustum forward so frustum orientation matches
    # the actual depth rays emitted into the scene.
    fwd = _normalize_vec(center_dir)
    right_proj = right_dir - fwd * torch.dot(right_dir, fwd)
    if torch.linalg.norm(right_proj) < 1.0e-6:
        return None

    right = _normalize_vec(right_proj)
    down = _normalize_vec(torch.cross(fwd, right))
    rot = torch.stack([right, down, fwd], dim=1)
    return matrix_to_quaternion(rot)


def _quat_from_forward_up(forward: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    forward = _normalize_vec(forward)
    up_hint = _normalize_vec(up_hint)
    up_proj = up_hint - torch.dot(up_hint, forward) * forward
    if torch.linalg.norm(up_proj) < 1.0e-4:
        fallback = torch.tensor([0.0, 0.0, 1.0], device=forward.device, dtype=forward.dtype)
        if torch.abs(torch.dot(fallback, forward)) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=forward.device, dtype=forward.dtype)
        up_proj = fallback - torch.dot(fallback, forward) * forward
    z_axis = _normalize_vec(up_proj)
    y_axis = _normalize_vec(torch.cross(z_axis, forward))
    z_axis = _normalize_vec(torch.cross(forward, y_axis))
    rot = torch.stack([forward, y_axis, z_axis], dim=1)
    quat_wxyz = matrix_to_quaternion(rot)
    return quat_wxyz[[1, 2, 3, 0]]


def _make_marker_mesh(color: tuple[int, int, int], radius: float):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def _make_box_mesh(color: tuple[int, int, int], extents: np.ndarray, alpha: int = 120):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None
    extents = np.asarray(extents, dtype=np.float32).reshape(3)
    extents = np.maximum(extents, 1.0e-3)
    mesh = trimesh.creation.box(extents=extents)
    rgba = np.array([color[0], color[1], color[2], alpha], dtype=np.uint8)
    mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
    return mesh


def _set_visual_handle_visible(handle: Any, visible: bool) -> None:
    if handle is None:
        return
    try:
        if hasattr(handle, 'show_visual'):
            handle.show_visual = bool(visible)
        else:
            handle.visible = bool(visible)
    except Exception:
        pass


def _get_visual_handle_visible(handle: Any, default: bool = True) -> bool:
    if handle is None:
        return bool(default)
    try:
        if hasattr(handle, 'show_visual'):
            return bool(handle.show_visual)
        return bool(getattr(handle, 'visible', default))
    except Exception:
        return bool(default)


def _parse_urdf_vec3(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float32)
    parts = [part for part in str(raw).replace(',', ' ').split() if part]
    if len(parts) != 3:
        return np.asarray(default, dtype=np.float32)
    try:
        return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    except Exception:
        return np.asarray(default, dtype=np.float32)


def _urdf_rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _urdf_origin_transform(origin_el: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    if origin_el is None:
        return transform
    transform[:3, :3] = _urdf_rpy_matrix(_parse_urdf_vec3(origin_el.get('rpy'), (0.0, 0.0, 0.0)))
    transform[:3, 3] = _parse_urdf_vec3(origin_el.get('xyz'), (0.0, 0.0, 0.0))
    return transform


def _normalize_mesh_kind(mesh_kind: str | None) -> str:
    normalized = str(mesh_kind or "visual").strip().lower()
    if normalized not in {"visual", "collision"}:
        normalized = "visual"
    return normalized


@functools.lru_cache(maxsize=128)
def _load_combined_urdf_mesh(urdf_path: str, mesh_kind: str = "visual"):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None

    normalized_kind = _normalize_mesh_kind(mesh_kind)
    try:
        urdf_file = Path(urdf_path).expanduser().resolve()
        root = ET.parse(urdf_file).getroot()
    except Exception:
        return None

    meshes: list[Any] = []
    for link in root.findall('link'):
        for element in link.findall(normalized_kind):
            geometry = element.find('geometry')
            if geometry is None:
                continue

            mesh_obj = None
            mesh_tag = geometry.find('mesh')
            if mesh_tag is not None:
                filename = str(mesh_tag.get('filename', '')).strip()
                if not filename:
                    continue
                mesh_file = Path(filename)
                if not mesh_file.is_absolute():
                    mesh_file = (urdf_file.parent / mesh_file).resolve()
                if not mesh_file.exists():
                    continue
                try:
                    loaded = trimesh.load(str(mesh_file), process=False)
                except Exception:
                    continue
                if isinstance(loaded, trimesh.Scene):
                    dumped = loaded.dump(concatenate=True)
                    mesh_obj = dumped if isinstance(dumped, trimesh.Trimesh) else None
                elif isinstance(loaded, trimesh.Trimesh):
                    mesh_obj = loaded
                if mesh_obj is None:
                    continue
                scale = _parse_urdf_vec3(mesh_tag.get('scale'), (1.0, 1.0, 1.0)).astype(np.float64)
                mesh_obj = mesh_obj.copy()
                mesh_obj.apply_scale(scale)
            else:
                box_tag = geometry.find('box')
                cylinder_tag = geometry.find('cylinder')
                sphere_tag = geometry.find('sphere')
                if box_tag is not None:
                    size = _parse_urdf_vec3(box_tag.get('size'), (1.0, 1.0, 1.0)).astype(np.float64)
                    mesh_obj = trimesh.creation.box(extents=size)
                elif cylinder_tag is not None:
                    radius = float(cylinder_tag.get('radius', '0.0'))
                    length = float(cylinder_tag.get('length', '0.0'))
                    if radius <= 0.0 or length <= 0.0:
                        continue
                    mesh_obj = trimesh.creation.cylinder(radius=radius, height=length)
                elif sphere_tag is not None:
                    radius = float(sphere_tag.get('radius', '0.0'))
                    if radius <= 0.0:
                        continue
                    mesh_obj = trimesh.creation.icosphere(radius=radius)
                else:
                    continue

            if mesh_obj is None:
                continue
            mesh_obj = mesh_obj.copy()
            mesh_obj.apply_transform(_urdf_origin_transform(element.find('origin')).astype(np.float64))
            meshes.append(mesh_obj)

    if not meshes:
        return None

    merged = trimesh.util.concatenate(meshes)
    return merged


@functools.lru_cache(maxsize=128)
def _load_combined_urdf_visual_mesh(urdf_path: str):
    return _load_combined_urdf_mesh(urdf_path, "visual")


@functools.lru_cache(maxsize=128)
def _load_combined_urdf_collision_mesh(urdf_path: str):
    return _load_combined_urdf_mesh(urdf_path, "collision")


def _triangulate_face_indices(face_counts, face_indices) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    cursor = 0
    for raw_count in face_counts:
        count = int(raw_count)
        if count < 3:
            cursor += max(count, 0)
            continue
        face = [int(face_indices[cursor + idx]) for idx in range(count)]
        cursor += count
        for idx in range(1, count - 1):
            triangles.append((face[0], face[idx], face[idx + 1]))
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _build_trimesh_from_usd_geom_prim(prim: Any):
    try:
        import trimesh  # type: ignore[import-not-found]
        from pxr import UsdGeom  # type: ignore[import-not-found]
    except Exception:
        return None

    if prim.IsA(UsdGeom.Mesh):
        mesh_geom = UsdGeom.Mesh(prim)
        points = mesh_geom.GetPointsAttr().Get()
        face_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            return None
        vertices = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
        faces = _triangulate_face_indices(face_counts, face_indices)
        if vertices.size == 0 or faces.size == 0:
            return None
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get()
        size = 2.0 if size is None else float(size)
        return trimesh.creation.box(extents=(size, size, size))

    if prim.IsA(UsdGeom.Sphere):
        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get()
        radius = 1.0 if radius is None else float(radius)
        return trimesh.creation.icosphere(radius=radius)

    if prim.IsA(UsdGeom.Cylinder):
        cylinder = UsdGeom.Cylinder(prim)
        radius = cylinder.GetRadiusAttr().Get()
        height = cylinder.GetHeightAttr().Get()
        radius = 1.0 if radius is None else float(radius)
        height = 2.0 if height is None else float(height)
        if radius <= 0.0 or height <= 0.0:
            return None
        return trimesh.creation.cylinder(radius=radius, height=height)

    if prim.IsA(UsdGeom.Capsule):
        capsule = UsdGeom.Capsule(prim)
        radius = capsule.GetRadiusAttr().Get()
        height = capsule.GetHeightAttr().Get()
        radius = 1.0 if radius is None else float(radius)
        height = 2.0 if height is None else float(height)
        if radius <= 0.0 or height <= 0.0:
            return None
        return trimesh.creation.capsule(radius=radius, height=height)

    return None


def _prim_has_collision_hint(prim: Any, root_prim_path: str) -> bool:
    prim_path = str(prim.GetPath())
    rel_path = prim_path[len(root_prim_path) :] if prim_path.startswith(root_prim_path) else prim_path
    rel_path_lc = rel_path.casefold()
    if "collision" in rel_path_lc or "collider" in rel_path_lc:
        return True

    try:
        from pxr import PhysxSchema, UsdPhysics  # type: ignore[import-not-found]
    except Exception:
        PhysxSchema = None  # type: ignore[assignment]
        UsdPhysics = None  # type: ignore[assignment]

    current = prim
    while current is not None and current.IsValid():
        current_path = str(current.GetPath())
        if current_path == root_prim_path:
            break
        current_name = current.GetName().casefold()
        if "collision" in current_name or "collider" in current_name:
            return True
        if UsdPhysics is not None and current.HasAPI(UsdPhysics.CollisionAPI):
            return True
        physx_collision_api = getattr(PhysxSchema, "PhysxCollisionAPI", None) if PhysxSchema is not None else None
        if physx_collision_api is not None and current.HasAPI(physx_collision_api):
            return True
        current = current.GetParent()
    return False


def _transform_mesh_vertices_between_usd_frames(vertices: np.ndarray, source_world_tf: Any, target_inv_world_tf: Any) -> np.ndarray:
    try:
        from pxr import Gf  # type: ignore[import-not-found]
    except Exception:
        return np.asarray(vertices, dtype=np.float32)

    src = np.asarray(vertices, dtype=np.float64)
    dst = np.zeros_like(src)
    for idx, vertex in enumerate(src):
        world_point = source_world_tf.TransformAffine(Gf.Vec3d(float(vertex[0]), float(vertex[1]), float(vertex[2])))
        local_point = target_inv_world_tf.TransformAffine(world_point)
        dst[idx, 0] = float(local_point[0])
        dst[idx, 1] = float(local_point[1])
        dst[idx, 2] = float(local_point[2])
    return dst.astype(np.float32, copy=False)


@functools.lru_cache(maxsize=512)
def _load_combined_live_usd_mesh(root_prim_path: str, mesh_kind: str = "visual"):
    try:
        import trimesh  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except Exception:
        return None

    normalized_kind = _normalize_mesh_kind(mesh_kind)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        return None

    root_world_tf = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    root_inv_world_tf = root_world_tf.GetInverse()

    subtree_roots = []
    candidate_suffixes = ['']
    if normalized_kind == "visual":
        candidate_suffixes.extend(['/baseLink', '/baseLink/visuals', '/visuals', '/Visuals', '/baseLink/Visuals'])
    else:
        candidate_suffixes.extend(
            [
                '/baseLink',
                '/baseLink/collisions',
                '/collisions',
                '/collision',
                '/Collisions',
                '/Collision',
                '/baseLink/Collisions',
                '/baseLink/Collision',
            ]
        )
    for suffix in candidate_suffixes:
        candidate = stage.GetPrimAtPath(root_prim_path + suffix)
        if candidate.IsValid():
            subtree_roots.append(candidate)

    meshes: list[Any] = []
    matched_prim_count = 0
    visited_prim_paths: set[str] = set()
    traverse_predicate = None
    try:
        traverse_predicate = Usd.TraverseInstanceProxies(getattr(Usd, "PrimAllPrimsPredicate", Usd.PrimDefaultPredicate))
    except Exception:
        traverse_predicate = None

    for subtree_root in subtree_roots:
        prim_iter = (
            Usd.PrimRange(subtree_root, traverse_predicate)
            if traverse_predicate is not None
            else Usd.PrimRange(subtree_root)
        )
        for prim in prim_iter:
            if not prim.IsValid():
                continue
            prim_path = str(prim.GetPath())
            if prim_path in visited_prim_paths:
                continue
            visited_prim_paths.add(prim_path)
            collision_like = _prim_has_collision_hint(prim, root_prim_path)
            if normalized_kind == "visual" and collision_like:
                continue
            if normalized_kind == "collision" and not collision_like:
                continue
            mesh_obj = _build_trimesh_from_usd_geom_prim(prim)
            if mesh_obj is None:
                continue
            prim_world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            mesh_obj = mesh_obj.copy()
            mesh_obj.vertices = _transform_mesh_vertices_between_usd_frames(
                mesh_obj.vertices,
                prim_world_tf,
                root_inv_world_tf,
            )
            meshes.append(mesh_obj)
            matched_prim_count += 1

    if not meshes:
        return None

    merged = trimesh.util.concatenate(meshes)
    setattr(merged, '_holosoma_visual_prim_count', matched_prim_count)
    return merged


@functools.lru_cache(maxsize=512)
def _load_combined_live_usd_visual_mesh(root_prim_path: str, mesh_signature: str = ""):
    return _load_combined_live_usd_mesh(root_prim_path, "visual")


@functools.lru_cache(maxsize=512)
def _load_combined_live_usd_collision_mesh(root_prim_path: str, mesh_signature: str = ""):
    return _load_combined_live_usd_mesh(root_prim_path, "collision")


def _import_viser() -> tuple[Any | None, Any | None, str | None]:
    ensure_viser_on_path()
    try:
        import viser  # type: ignore[import-not-found]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]
    except Exception as exc:
        return None, None, str(exc)
    return viser, ViserUrdf, None


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: Any) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


@functools.lru_cache(maxsize=32)
def _load_target_skeleton_edges(
    robot_urdf_path: str,
    tracked_body_names: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    try:
        root = ET.parse(robot_urdf_path).getroot()
    except Exception:
        return ()

    parent_by_child: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent_link = str(parent_elem.get("link", "")).strip()
        child_link = str(child_elem.get("link", "")).strip()
        if parent_link and child_link:
            parent_by_child[child_link] = parent_link

    tracked_idx_by_link: dict[str, int] = {}
    for idx, body_name in enumerate(tracked_body_names):
        link_name = FAKE_BODY_NAME_ALIASES.get(str(body_name), str(body_name))
        if link_name:
            tracked_idx_by_link[link_name] = idx

    edges: list[tuple[int, int]] = []
    for child_idx, body_name in enumerate(tracked_body_names):
        current_link = FAKE_BODY_NAME_ALIASES.get(str(body_name), str(body_name))
        visited: set[str] = set()
        parent_link = parent_by_child.get(current_link)
        while parent_link and parent_link not in visited:
            visited.add(parent_link)
            parent_idx = tracked_idx_by_link.get(parent_link)
            if parent_idx is not None and parent_idx != child_idx:
                edges.append((parent_idx, child_idx))
                break
            parent_link = parent_by_child.get(parent_link)

    deduped: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge in edges:
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        deduped.append(edge)
    return tuple(deduped)


def _resolve_object_urdf_path(robot_config: Any) -> str | None:
    obj_cfg = getattr(robot_config, "object", None)
    if not obj_cfg or not getattr(obj_cfg, "enabled", False):
        return None
    obj_path = getattr(obj_cfg, "object_urdf_path", None)
    if not obj_path:
        return None
    resolved = _resolve_data_path(obj_path)
    resolved_path = Path(resolved)

    if resolved_path.is_dir():
        urdfs = sorted(list(resolved_path.rglob("*.urdf")) + list(resolved_path.rglob("*.URDF")))
        return str(urdfs[0]) if urdfs else None

    if resolved_path.suffix.lower() != ".json":
        return resolved

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse object spec map '{}': {}", resolved_path, exc)
        return None

    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        payload = payload["clips"]
    if not isinstance(payload, dict):
        logger.warning("Invalid object spec map '{}': expected dict payload", resolved_path)
        return None

    for entry in payload.values():
        if isinstance(entry, str):
            urdf_raw = entry.strip()
        elif isinstance(entry, dict):
            urdf_raw = str(entry.get("object_urdf_path", "")).strip()
        else:
            urdf_raw = ""
        if not urdf_raw:
            continue
        try:
            if not Path(urdf_raw).is_absolute() and not urdf_raw.startswith("@holosoma/") and not urdf_raw.startswith(
                "holosoma/"
            ):
                candidate = (resolved_path.parent / urdf_raw).resolve()
            else:
                candidate = Path(_resolve_data_path(urdf_raw))
        except Exception:
            continue
        if candidate.exists() and candidate.suffix.lower() == ".urdf":
            return str(candidate)

    logger.warning("No valid URDF found in object spec map '{}'; disabling Viser object URDF.", resolved_path)
    return None


def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _resolve_obj_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_dir():
        matches = list(path.glob("*.obj")) + list(path.glob("*.OBJ"))
        return sorted(matches)
    if any(char in path_str for char in ("*", "?", "[")):
        import glob

        return sorted(Path(p) for p in glob.glob(path_str))
    return [path] if path.exists() else []


def _load_terrain_mesh(
    obj_path: str | None,
    *,
    obj_metadata_path: str | None,
    num_rows: int | None,
    num_cols: int | None,
    clip_name: str | None = None,
):
    if not obj_path:
        return None

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("Viser terrain disabled (trimesh unavailable): {}", exc)
        return None

    single_clip_only = os.environ.get("VISER_TERRAIN_SINGLE_CLIP_ONLY", "1").lower() not in (
        "0",
        "false",
        "no",
    )

    def _load_mesh(path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump(concatenate=False)
            if not meshes:
                raise ValueError("Loaded terrain scene has no geometries.")
            mesh = max(meshes, key=lambda m: len(getattr(m, "faces", [])) or len(m.vertices))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded terrain is not a trimesh: {type(mesh)}")
        return mesh

    def _tile_single_mesh(base_mesh: trimesh.Trimesh, *, rows: int, cols: int) -> trimesh.Trimesh:
        if rows * cols <= 1:
            return base_mesh
        gap = 1e-4
        stride = (base_mesh.bounds[1] - base_mesh.bounds[0]) + gap
        tiles = []
        for row in range(rows):
            for col in range(cols):
                tile = base_mesh.copy()
                tile.apply_translation([col * stride[0], row * stride[1], 0.0])
                tiles.append(tile)
        return trimesh.util.concatenate(tiles)

    def _tile_multi_obj_mesh(obj_paths: list[Path], *, rows: int) -> trimesh.Trimesh:
        meshes = []
        spans = []
        for path in obj_paths:
            mesh = _load_mesh(path)
            meshes.append(mesh)
            spans.append(mesh.bounds[1] - mesh.bounds[0])
        if not meshes:
            raise ValueError("No terrain OBJ meshes loaded from directory/glob input.")
        if len(meshes) == 1:
            return _tile_single_mesh(meshes[0], rows=rows, cols=1)

        spans_arr = np.vstack(spans)
        gap = 1e-4
        stride = spans_arr.max(axis=0) + gap
        tiles = []
        for col, mesh in enumerate(meshes):
            base_offset = np.array([col * stride[0], 0.0, 0.0], dtype=np.float64)
            for row in range(rows):
                tile = mesh.copy()
                tile.apply_translation(base_offset + np.array([0.0, row * stride[1], 0.0], dtype=np.float64))
                tiles.append(tile)
        return trimesh.util.concatenate(tiles)

    def _canonical_pair_key(raw_name: str | None) -> str | None:
        if raw_name is None:
            return None
        name = str(raw_name).strip()
        if not name:
            return None
        if (name.startswith("b'") and name.endswith("'")) or (name.startswith('b"') and name.endswith('"')):
            name = name[2:-1]
        name = Path(name).name
        lower = name.lower()
        for suffix in (".npz", ".h5", ".hdf5", ".obj"):
            if lower.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name.casefold()

    def _select_obj_path(paths: list[Path], name: str | None) -> Path:
        if len(paths) == 1:
            return paths[0]
        clip_key = _canonical_pair_key(name)
        if clip_key:
            keyed_paths: dict[str, Path] = {}
            for candidate in paths:
                key = _canonical_pair_key(candidate.stem)
                if key is not None and key not in keyed_paths:
                    keyed_paths[key] = candidate
            if clip_key in keyed_paths:
                return keyed_paths[clip_key]
            candidates_preview = ", ".join(path.stem for path in paths[:12])
            raise FileNotFoundError(
                f"No terrain OBJ matching clip '{name}'. Available stems (first 12): {candidates_preview}"
            )
        logger.warning("Terrain OBJ input has multiple meshes; no clip provided, using {}.", paths[0].name)
        return paths[0]

    def _metadata_is_local(path_str: str | None) -> bool:
        if not path_str:
            return True
        try:
            with Path(path_str).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        tile_offsets = np.asarray(payload.get("tile_offsets", []))
        tile_rows = int(payload.get("tile_rows", 1) or 1)
        tile_cols = int(payload.get("tile_cols", 1) or 1)
        tile_count = int(tile_offsets.shape[0]) if tile_offsets.ndim >= 1 else 0
        return tile_count <= 1 and tile_rows <= 1 and tile_cols <= 1

    def _load_obj_metadata(path_str: str | None) -> dict[str, Any] | None:
        if not path_str:
            return None
        try:
            with Path(path_str).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load terrain OBJ metadata '{}': {}", path_str, exc)
            return None
        if not isinstance(payload, dict):
            logger.warning("Ignoring invalid terrain OBJ metadata '{}': expected JSON object.", path_str)
            return None
        return payload

    def _extract_single_tile_mesh(
        base_mesh: trimesh.Trimesh,
        metadata: dict[str, Any],
        selected_clip_name: str | None,
    ) -> trimesh.Trimesh | None:
        clip_key = _canonical_pair_key(selected_clip_name)
        tile_names = list(metadata.get("tile_names", []))
        tile_offsets = np.asarray(metadata.get("tile_offsets", []), dtype=np.float64)
        tile_stride = np.asarray(metadata.get("tile_stride", []), dtype=np.float64).reshape(-1)
        tile_rows = max(1, int(metadata.get("tile_rows", 1) or 1))
        tile_cols = max(1, int(metadata.get("tile_cols", len(tile_names) or 1) or 1))

        if tile_offsets.size == 0 or tile_offsets.ndim != 2 or tile_offsets.shape[1] < 2:
            return None
        if tile_offsets.shape[1] == 2:
            tile_offsets = np.concatenate(
                [tile_offsets, np.zeros((tile_offsets.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
        elif tile_offsets.shape[1] > 3:
            tile_offsets = tile_offsets[:, :3]
        if tile_stride.size < 2:
            return None
        if tile_stride.size == 2:
            tile_stride = np.array([tile_stride[0], tile_stride[1], 0.0], dtype=np.float64)
        elif tile_stride.size > 3:
            tile_stride = tile_stride[:3]

        col_idx = None
        if tile_names:
            keyed_indices: dict[str, int] = {}
            for idx, raw_name in enumerate(tile_names):
                key = _canonical_pair_key(raw_name)
                if key is not None and key not in keyed_indices:
                    keyed_indices[key] = idx
            if clip_key is not None:
                col_idx = keyed_indices.get(clip_key)
                if col_idx is None:
                    logger.warning("Terrain metadata has no tile matching clip '{}'.", selected_clip_name)
                    return None
            elif keyed_indices:
                col_idx = min(keyed_indices.values())
        elif tile_cols == 1:
            col_idx = 0
        else:
            col_idx = 0 if clip_key is None else None
            if col_idx is None:
                return None

        if tile_offsets.shape[0] == tile_rows * tile_cols:
            tile_idx = int(col_idx)
        else:
            tile_idx = int(col_idx)
        if not (0 <= tile_idx < tile_offsets.shape[0]):
            return None

        tile_offset = np.asarray(tile_offsets[tile_idx], dtype=np.float64)
        x_min = float(tile_offset[0])
        y_min = float(tile_offset[1])
        x_max = x_min + float(tile_stride[0])
        y_max = y_min + float(tile_stride[1])
        pad = max(1e-4, float(max(abs(tile_stride[0]), abs(tile_stride[1]))) * 1e-4)

        try:
            face_centroids = np.asarray(base_mesh.triangles_center, dtype=np.float64)
        except Exception:
            return None
        if face_centroids.ndim != 2 or face_centroids.shape[1] < 2:
            return None

        mask = (
            (face_centroids[:, 0] >= (x_min - pad))
            & (face_centroids[:, 0] <= (x_max + pad))
            & (face_centroids[:, 1] >= (y_min - pad))
            & (face_centroids[:, 1] <= (y_max + pad))
        )
        face_indices = np.flatnonzero(mask)
        if face_indices.size == 0:
            logger.warning(
                "No terrain faces found for clip '{}' in metadata tile {}.",
                selected_clip_name,
                tile_idx,
            )
            return None

        tile_mesh = base_mesh.submesh([face_indices], append=True, repair=False)
        if isinstance(tile_mesh, trimesh.Scene):
            meshes = tile_mesh.dump(concatenate=False)
            if not meshes:
                return None
            tile_mesh = max(meshes, key=lambda m: len(getattr(m, "faces", [])) or len(m.vertices))
        if not isinstance(tile_mesh, trimesh.Trimesh):
            return None
        if len(tile_mesh.faces) == 0:
            return None
        tile_mesh = tile_mesh.copy()
        tile_mesh.apply_translation((-tile_offset).tolist())
        return tile_mesh

    resolved_rows = max(1, int(num_rows or 1))
    resolved_cols = max(1, int(num_cols or 1))
    terrain_path = Path(_resolve_data_path(obj_path))
    obj_paths = _resolve_obj_paths(str(terrain_path))
    if not obj_paths:
        return None

    if obj_metadata_path and len(obj_paths) == 1:
        selected_path = obj_paths[0]
        base_mesh = _load_mesh(selected_path)
        metadata_is_local = _metadata_is_local(obj_metadata_path)
        metadata = _load_obj_metadata(obj_metadata_path) if obj_metadata_path else None
        if single_clip_only and not metadata_is_local:
            if metadata is None:
                logger.warning(
                    "Viser terrain single-clip mode requires readable metadata for '{}'; hiding terrain mesh.",
                    obj_metadata_path,
                )
                return None
            tile_mesh = _extract_single_tile_mesh(base_mesh, metadata, clip_name)
            if tile_mesh is not None:
                return tile_mesh, True
            logger.warning(
                "Viser terrain single-clip mode could not isolate a tile from '{}'; hiding terrain mesh.",
                obj_metadata_path,
            )
            return None
        return base_mesh, metadata_is_local
    else:
        if obj_metadata_path and len(obj_paths) != 1:
            logger.warning("OBJ metadata provided with directory/glob input; ignoring metadata and using OBJ selection rules.")
        if len(obj_paths) > 1 and single_clip_only:
            try:
                selected_path = _select_obj_path(obj_paths, clip_name)
            except FileNotFoundError as exc:
                logger.error("{}", exc)
                return None
            return _load_mesh(selected_path), True
        if len(obj_paths) > 1:
            return _tile_multi_obj_mesh(obj_paths, rows=resolved_rows), False
        try:
            selected_path = _select_obj_path(obj_paths, clip_name)
        except FileNotFoundError as exc:
            logger.error("{}", exc)
            return None

    base_mesh = _load_mesh(selected_path)
    if single_clip_only:
        return base_mesh, True
    if resolved_rows * resolved_cols > 1:
        return _tile_single_mesh(base_mesh, rows=resolved_rows, cols=resolved_cols), False
    return base_mesh, True


class ViserLiveViewer:
    def __init__(self, env: Any) -> None:
        self._env = env
        self._enabled = bool(getattr(env.training_config, "enable_viser", False))
        self._faithful_mode = os.environ.get("VISER_FAITHFUL_MODE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._server = None
        self._viser_urdf_cls = None
        mesh_source = os.environ.get("VISER_MESH_SOURCE", "").strip().lower()
        robot_mesh_source = os.environ.get("VISER_ROBOT_MESH_SOURCE", "").strip().lower()
        if mesh_source in {"sim", "simulator", "isaacsim", "usd"}:
            self._mesh_source = "sim"
        elif mesh_source in {"urdf"}:
            self._mesh_source = "urdf"
        else:
            self._mesh_source = "legacy"
        self._sim_meshes_enabled = self._mesh_source == "sim"
        self._load_urdf_visuals = (
            robot_mesh_source in {"urdf", "texture", "textured", "rgb"}
            or (
                False
                if self._sim_meshes_enabled
                else os.environ.get("VISER_LOAD_URDF", "1").lower() in ("1", "true", "yes", "on")
            )
        )
        mesh_mode_default = os.environ.get("VISER_MESH_MODE", "visual").strip().lower()
        if mesh_mode_default not in SIM_MESH_MODE_OPTIONS:
            mesh_mode_default = "visual"
        self._mesh_mode_default = mesh_mode_default
        self._vr = None
        self._vo = None
        self._vo_collision = None
        self._robot_points_handle = None
        self._object_points_handle = None
        self._primary_object_variants: dict[str, Any] = {}
        self._primary_object_collision_variants: dict[str, Any] = {}
        self._active_primary_object_key: str | None = None
        self._active_primary_object_collision_key: str | None = None
        self._object_visual_mesh_vertices_local: dict[str, np.ndarray] = {}
        self._object_collision_mesh_vertices_local: dict[str, np.ndarray] = {}
        self._robot_root = None
        self._object_root = None
        self._secondary_env_ids: list[int] = []
        self._secondary_env_slot: dict[int, int] = {}
        self._secondary_robot_roots: dict[int, Any] = {}
        self._secondary_object_roots: dict[int, Any] = {}
        self._secondary_vr: dict[int, Any] = {}
        self._secondary_vo: dict[int, Any] = {}
        self._secondary_vo_collision: dict[int, Any] = {}
        self._secondary_robot_points_handles: dict[int, Any] = {}
        self._secondary_object_points_handles: dict[int, Any] = {}
        self._secondary_object_visual_key: dict[int, str] = {}
        self._secondary_object_collision_key: dict[int, str] = {}
        self._env_sequence_label_handles: dict[int, Any] = {}
        self._robot_mesh_frames: dict[int, dict[str, Any]] = {}
        self._robot_visual_mesh_handles: dict[int, dict[str, Any]] = {}
        self._robot_collision_mesh_handles: dict[int, dict[str, Any]] = {}
        self._robot_mesh_envs_logged: set[int] = set()
        self._robot_mesh_envs_missing_logged: set[int] = set()
        self._body_name_to_index = {str(name): idx for idx, name in enumerate(getattr(env, "body_names", []))}
        self._viser_multi_env_spacing = 2.5
        self._viser_multi_env_cols = 1
        self._joint_order: np.ndarray | None = None
        self._joint_count = 0
        self._offset: np.ndarray | None = None
        self._last_update = 0.0
        self._show_env_sequence_labels = os.environ.get("VISER_SHOW_ENV_SEQUENCE_LABELS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._env_sequence_label_height = float(os.environ.get("VISER_ENV_SEQUENCE_LABEL_HEIGHT", "1.6"))
        self._env_sequence_label_debug_logged = False
        self._scandots_handle = None
        self._scandots_rays_handle = None
        self._scandots_enabled = False
        self._scandots_point_size = 0.02
        self._scandots_color = np.array([255, 0, 0], dtype=np.uint8)
        self._scandots_warned = False
        self._ray_direction_stats_suffix = ""
        self._strict_camera_rays = os.environ.get("VISER_STRICT_CAMERA_RAYS", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._target_keypoints_handle = None
        self._target_keypoint_skeleton_handle = None
        self._target_skeleton_edges: tuple[tuple[int, int], ...] | None = None
        self._target_keypoints_point_size = 0.03
        self._target_keypoints_line_width = 2.5
        self._target_keypoints_color = np.array([128, 0, 128], dtype=np.uint8)
        self._show_target_keypoints = os.environ.get("VISER_SHOW_TARGET_KEYPOINTS", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._target_object_center_handle = None
        self._target_object_trajectory_handle = None
        self._rollout_root_trajectory_handle = None
        self._rollout_object_trajectory_handle = None
        self._target_object_center_point_size = 0.045
        self._target_object_trajectory_line_width = 2.5
        self._rollout_root_trajectory_line_width = 3.0
        self._rollout_object_trajectory_line_width = 3.0
        self._rollout_trajectory_mesh_width = float(os.environ.get("VISER_ROLLOUT_TRAJECTORY_MESH_WIDTH", "0.06"))
        self._rollout_trajectory_z_offset = float(os.environ.get("VISER_ROLLOUT_TRAJECTORY_Z_OFFSET", "0.08"))
        self._target_object_center_color = np.array([0, 255, 0], dtype=np.uint8)
        self._rollout_root_trajectory_color = np.array([255, 128, 0], dtype=np.uint8)
        self._rollout_object_trajectory_color = np.array([0, 0, 255], dtype=np.uint8)
        self._rollout_root_trajectory_points_w: list[np.ndarray] = []
        self._rollout_root_trajectory_last_clip_idx: int | None = None
        self._rollout_root_trajectory_last_time_step: int | None = None
        self._rollout_object_trajectory_points_w: list[np.ndarray] = []
        self._rollout_object_trajectory_last_clip_idx: int | None = None
        self._rollout_object_trajectory_last_time_step: int | None = None
        self._rollout_trajectory_mesh_logged: set[str] = set()
        self._show_target_object_center = os.environ.get("VISER_SHOW_TARGET_OBJECT_CENTER", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._show_target_object_trajectory = os.environ.get("VISER_SHOW_TARGET_OBJECT_TRAJECTORY", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._show_rollout_root_trajectory = os.environ.get("VISER_SHOW_ROLLOUT_ROOT_TRAJECTORY", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._show_rollout_object_trajectory = os.environ.get("VISER_SHOW_ROLLOUT_OBJECT_TRAJECTORY", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._show_target_box = os.environ.get("VISER_SHOW_TARGET_BOX", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._show_future_goal_box = os.environ.get("VISER_SHOW_FUTURE_GOAL_BOX", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._distill_minimal_ui = os.environ.get("VISER_DISTILL_MINIMAL_UI", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._start_paused = os.environ.get("VISER_START_PAUSED", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._play_restarts_visible_replay = os.environ.get("VISER_PLAY_RESTARTS_VISIBLE_REPLAY", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._reset_restarts_visible_replay = os.environ.get("VISER_RESET_RESTARTS_VISIBLE_REPLAY", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._disable_perception_image_pipeline = os.environ.get("VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._disable_perception_frustum = os.environ.get("VISER_DISABLE_PERCEPTION_FRUSTUM", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._show_perception_frustum_default = os.environ.get("VISER_SHOW_PERCEPTION_FRUSTUM", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._manual_control_default = os.environ.get("VISER_MANUAL_CONTROL_DEFAULT", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._drop_button_default = os.environ.get("VISER_DROP_BUTTON_DEFAULT", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._manual_force_enabled = os.environ.get("VISER_FORCE_MANUAL_CONTROL", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self._enable_object_reset_override = os.environ.get("VISER_ENABLE_OBJECT_RESET_OVERRIDE", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        self._clip_lock_default = os.environ.get("VISER_CLIP_LOCK_DEFAULT", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        self._clip_group_gui_enabled = os.environ.get("VISER_ENABLE_CLIP_GROUP_GUI", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._clip_group_size_override = max(0, int(os.environ.get("VISER_CLIP_GROUP_SIZE", "0") or "0"))
        self._initial_clip_group_index = max(0, int(os.environ.get("VISER_INITIAL_CLIP_GROUP_INDEX", "0") or "0"))
        reset_to_default_pose_env = os.environ.get("HOLOSOMA_RESET_TO_DEFAULT_POSE")
        if reset_to_default_pose_env is None:
            reset_to_default_pose_env = os.environ.get("HOLOSOMA_DEFAULT_POSE_INIT", "0")
        self._reset_to_default_pose = reset_to_default_pose_env.lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._manual_use_hw_joystick = os.environ.get("VISER_MANUAL_USE_HW_JOYSTICK", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        manual_hw_backend = os.environ.get("VISER_MANUAL_HW_BACKEND", "auto").strip().lower()
        if manual_hw_backend not in ("auto", "bridge", "pygame"):
            manual_hw_backend = "auto"
        self._manual_hw_backend = manual_hw_backend
        try:
            self._manual_hw_device = int(
                os.environ.get(
                    "VISER_MANUAL_HW_DEVICE",
                    os.environ.get("JOYSTICK_DEVICE", "0"),
                )
            )
        except Exception:
            self._manual_hw_device = 0
        self._manual_hw_type = os.environ.get("VISER_MANUAL_HW_TYPE", os.environ.get("JOYSTICK_TYPE", "xbox")).strip().lower()
        if self._manual_hw_type not in ("xbox", "switch"):
            self._manual_hw_type = "xbox"
        try:
            self._manual_hw_deadzone = float(os.environ.get("VISER_MANUAL_HW_DEADZONE", "0.08"))
        except Exception:
            self._manual_hw_deadzone = 0.08
        if self._manual_hw_deadzone < 0.0:
            self._manual_hw_deadzone = 0.0
        if self._manual_hw_deadzone > 1.0:
            self._manual_hw_deadzone = 1.0
        self._disable_contact_force_viz = os.environ.get("VISER_DISABLE_CONTACT_FORCE_VIZ", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        perception_transport_format = os.environ.get("VISER_PERCEPTION_IMAGE_FORMAT", "auto")
        self._perception_transport_format = _normalize_viser_image_format(
            perception_transport_format,
            faithful_mode=self._faithful_mode and "VISER_PERCEPTION_IMAGE_FORMAT" not in os.environ,
        )

        perception_jpeg_quality_raw = os.environ.get("VISER_PERCEPTION_JPEG_QUALITY", "90").strip()
        try:
            perception_jpeg_quality = int(perception_jpeg_quality_raw)
        except Exception:
            perception_jpeg_quality = 90
        if perception_jpeg_quality < 1 or perception_jpeg_quality > 100:
            perception_jpeg_quality = 90
        self._perception_jpeg_quality = perception_jpeg_quality

        self._perception_flip_vertical = os.environ.get("VISER_PERCEPTION_FLIP_VERTICAL", "0").lower() not in (
            "0",
            "false",
            "no",
        )
        if self._faithful_mode and "VISER_PERCEPTION_FLIP_VERTICAL" not in os.environ:
            self._perception_flip_vertical = False

        depth_colormap = os.environ.get("VISER_DEPTH_COLORMAP", "dynamic").strip().lower()
        if depth_colormap not in ("dynamic", "fixed"):
            depth_colormap = "dynamic"
        if self._faithful_mode and "VISER_DEPTH_COLORMAP" not in os.environ:
            depth_colormap = "fixed"
        self._depth_colormap = depth_colormap

        perception_image_mode = os.environ.get("VISER_PERCEPTION_IMAGE_MODE", "auto").strip().lower()
        if perception_image_mode not in ("auto", "depth", "rgb"):
            perception_image_mode = "auto"
        if self._faithful_mode and "VISER_PERCEPTION_IMAGE_MODE" not in os.environ:
            perception_image_mode = "depth"
        self._perception_image_mode = perception_image_mode
        perception_depth_source = os.environ.get("VISER_PERCEPTION_DEPTH_SOURCE", "obs").strip().lower()
        if perception_depth_source not in ("obs", "raw"):
            perception_depth_source = "obs"
        self._perception_depth_source = perception_depth_source
        if self._faithful_mode:
            self._strict_camera_rays = True
            self._disable_perception_image_pipeline = False
        self._play_control = None
        self._step_button = None
        self._reset_button = None
        self._default_pose_init_cb = None
        self._step_requested = False
        self._reset_requested = False
        self._reset_visible_requested = False
        self._play_last_value = not self._start_paused
        self._clip_dropdown = None
        self._clip_apply = None
        self._clip_label = None
        self._clip_names: list[str] = []
        self._pending_clip_idx: int | None = None
        self._pending_visible_group_index: int | None = None
        self._pending_control_sync = False
        self._control_sleep_s = 0.02
        self._pending_clip_start: int | None = None
        self._clip_group_prev_btn = None
        self._clip_group_next_btn = None
        self._clip_group_apply_btn = None
        self._clip_group_index_in = None
        self._clip_group_info = None
        self._clip_group_ui_syncing = False
        self._grid_handle = None
        self._terrain_handle = None
        self._ground_handle = None
        self._terrain_clip_name = None
        self._terrain_is_local = False
        self._show_robot_cb = None
        self._show_object_cb = None
        self._show_terrain_cb = None
        self._show_grid_cb = None
        self._show_scandots_cb = None
        self._mesh_mode_dropdown = None
        self._recenter_cb = None
        self._scandots_size_slider = None
        self._contact_force_cb = None
        self._contact_force_scale_slider = None
        self._contact_force_threshold_slider = None
        self._contact_force_handle = None
        self._manual_control_cb = None
        self._manual_root_sync_button = None
        self._manual_root_status = None
        self._manual_forward_cb = None
        self._manual_back_cb = None
        self._manual_left_cb = None
        self._manual_right_cb = None
        self._manual_yaw_left_cb = None
        self._manual_yaw_right_cb = None
        self._manual_lin_scale_slider = None
        self._manual_yaw_scale_slider = None
        self._manual_root_pos_x_slider = None
        self._manual_root_pos_y_slider = None
        self._manual_root_yaw_slider = None
        self._manual_root_cmd_xy = None
        self._manual_root_cmd_yaw = None
        self._drop_button_cb = None
        self._drop_button_reset_button = None
        self._drop_button_status = None
        self._drop_button_gui_enabled = False
        self._object_reset_random_button = None
        self._object_reset_override_cb = None
        self._object_reset_zero_button = None
        self._object_reset_pos_x_slider = None
        self._object_reset_pos_y_slider = None
        self._object_reset_pos_z_slider = None
        self._object_reset_roll_slider = None
        self._object_reset_pitch_slider = None
        self._object_reset_yaw_slider = None
        self._object_reset_status = None
        self._object_ground_status = None
        self._object_ground_status_debug_logged = False
        self._manual_command_arrow_handle = None
        self._manual_command_arrow_height = 0.30
        self._manual_command_arrow_head_ratio = 0.30
        self._manual_command_arrow_head_width_ratio = 0.65
        self._manual_command_arrow_max_len = 1.20
        self._manual_command_arrow_line_width = 5.0
        self._manual_command_yaw_radius = 0.28
        self._manual_command_yaw_arc_angle = 4.5
        self._manual_command_yaw_head_len = 0.10
        self._manual_command_yaw_head_width = 0.08
        self._manual_command_yaw_segments = 20
        self._manual_hw_joystick = None
        self._manual_hw_pygame = None
        self._manual_hw_axes = {"LX": 0, "LY": 1, "RX": 2}
        self._manual_hw_init_attempted = False
        self._manual_hw_warned = False
        self._auto_forward_after_lift_enabled = os.environ.get("VISER_AUTO_FORWARD_AFTER_LIFT", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._auto_forward_after_lift_command = self._parse_float_list_env(
            "VISER_AUTO_FORWARD_AFTER_LIFT_COMMAND",
            [0.5, 0.0, 0.0],
            expected_len=3,
        )
        self._auto_forward_after_lift_rel_z_delta = self._float_env(
            "VISER_AUTO_FORWARD_AFTER_LIFT_REL_Z_DELTA",
            0.10,
        )
        self._auto_forward_after_lift_consecutive_steps = max(
            1,
            int(self._float_env("VISER_AUTO_FORWARD_AFTER_LIFT_CONSECUTIVE_STEPS", 5.0)),
        )
        self._auto_forward_after_lift_duration_s = self._float_env(
            "VISER_AUTO_FORWARD_AFTER_LIFT_DURATION_S",
            8.0,
        )
        self._auto_forward_after_lift_log_period_s = 1.0 / max(
            self._float_env("VISER_AUTO_FORWARD_AFTER_LIFT_LOG_HZ", 20.0),
            1.0,
        )
        self._auto_forward_after_lift_log_path = Path(
            os.environ.get(
                "VISER_AUTO_FORWARD_AFTER_LIFT_LOG_PATH",
                "logs/runtime/viser_auto_forward_after_lift.jsonl",
            )
        ).expanduser()
        self._auto_forward_after_lift_heading_hold_enabled = os.environ.get(
            "VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_HOLD",
            "0",
        ).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._auto_forward_after_lift_heading_kp = self._float_env(
            "VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_KP",
            1.0,
        )
        self._auto_forward_after_lift_heading_kd = self._float_env(
            "VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_KD",
            0.0,
        )
        self._auto_forward_after_lift_heading_yaw_limit = abs(
            self._float_env("VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_YAW_LIMIT", 0.6)
        )
        self._auto_forward_after_lift_heading_deadband = abs(
            self._float_env("VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_DEADBAND", 0.0)
        )
        self._auto_forward_after_lift_heading_xy_compensation_enabled = os.environ.get(
            "VISER_AUTO_FORWARD_AFTER_LIFT_HEADING_XY_COMPENSATION",
            "1",
        ).lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._auto_forward_after_lift_state = "waiting"
        self._auto_forward_after_lift_baseline_rel_z: float | None = None
        self._auto_forward_after_lift_ready_count = 0
        self._auto_forward_after_lift_trigger_time: float | None = None
        self._auto_forward_after_lift_last_log_time = 0.0
        self._auto_forward_after_lift_log_handle = None
        self._auto_forward_after_lift_logged_start = False
        self._auto_forward_after_lift_applied_command: list[float] | None = None
        self._auto_forward_after_lift_last_snapshot_wall_time: float | None = None
        self._auto_forward_after_lift_last_snapshot_yaw: float | None = None
        self._auto_forward_after_lift_heading_target_yaw: float | None = None
        self._auto_forward_after_lift_heading_error: float | None = None
        self._auto_forward_after_lift_effective_command: list[float] | None = None
        self._clip_start_slider = None
        self._clip_lock_cb = None
        self._perception_enabled = False
        self._perception_depth_handle = None
        self._perception_stats = None
        self._perception_show_depth_cb = None
        self._perception_show_frustum_cb = None
        self._perception_show_points_cb = None
        self._perception_depth_source_dropdown = None
        self._perception_show_heightmap_joint_cb = None
        self._perception_show_camera_joint_cb = None
        self._perception_frustum = None
        self._perception_frame = None
        self._heightmap_marker_handle = None
        self._heightmap_marker_is_pc = False
        self._camera_marker_handle = None
        self._camera_marker_is_pc = False
        self._perception_last_shape: tuple[int, int] | None = None
        self._perception_last_mode: str | None = None
        self._perception_last_fov: float | None = None
        self._perception_last_aspect: float | None = None
        self._target_box_handle = None
        self._target_box_last_dimensions: tuple[float, float, float] | None = None
        self._target_box_last_mesh_key: str | None = None
        self._future_goal_box_handle = None
        self._future_goal_box_last_dimensions: tuple[float, float, float] | None = None
        self._future_goal_box_last_mesh_key: str | None = None
        self._scene_prefix = ""
        self._global_root = None
        self._global_frame_wxyz: tuple[float, float, float, float] | None = None
        self._root_body_index: int | None = None
        self._root_body_name: str | None = None
        self._root_pose_debug_logged = False
        self._record_step_debug_logged = False
        self._defer_startup = os.environ.get("VISER_DEFER_INIT", "0").lower() in ("1", "true", "yes", "on")
        self._startup_initialized = False

        if not self._enabled:
            return
        if not _is_rank0():
            self._enabled = False
            return
        if self._defer_startup:
            logger.info("Deferring Viser live viewer startup until first simulator step.")
            return
        self._initialize_server()

    def _initialize_server(self) -> None:
        if not self._enabled or self._startup_initialized:
            return
        self._startup_initialized = True

        env = self._env
        cfg = env.training_config
        self._env_id = int(getattr(cfg, "viser_env_id", 0))
        if self._env_id < 0 or self._env_id >= getattr(env, "num_envs", 1):
            logger.warning("Viser env_id {} out of range; defaulting to 0.", self._env_id)
            self._env_id = 0
        env_count = int(getattr(cfg, "viser_env_count", 1))
        if env_count < 1:
            env_count = 1
        self._viser_multi_env_spacing = float(getattr(cfg, "viser_multi_env_spacing", 2.5))
        try:
            self._viser_multi_env_cols = max(1, int(os.environ.get("VISER_MULTI_ENV_COLS", "1")))
        except Exception:
            self._viser_multi_env_cols = 1
        max_envs = int(getattr(env, "num_envs", 1))
        end_env = min(max_envs, self._env_id + env_count)
        self._secondary_env_ids = list(range(self._env_id + 1, end_env))
        self._secondary_env_slot = {env_id: idx + 1 for idx, env_id in enumerate(self._secondary_env_ids)}
        if self._secondary_env_ids:
            logger.info(
                "Viser multi-env view enabled: primary env {} + secondary envs {} (spacing={} cols={})",
                self._env_id,
                self._secondary_env_ids,
                self._viser_multi_env_spacing,
                self._viser_multi_env_cols,
            )
        self._resolve_root_body_index()

        update_hz = float(getattr(cfg, "viser_update_hz", 30.0))
        sync_to_sim = bool(getattr(cfg, "viser_sync_to_sim", True))
        force_dt = bool(getattr(cfg, "viser_force_dt", True))
        sim_hz = None
        sim_cfg = getattr(env.simulator, "simulator_config", None)
        if sim_cfg is not None and hasattr(sim_cfg, "sim"):
            sim_fps = getattr(sim_cfg.sim, "fps", None)
            sim_decimation = getattr(sim_cfg.sim, "control_decimation", None)
            if sim_fps:
                sim_decimation = int(sim_decimation or 1)
                sim_hz = float(sim_fps) / max(1, sim_decimation)

        if sync_to_sim and sim_hz:
            update_hz = sim_hz
        elif update_hz <= 0 and sim_hz:
            update_hz = sim_hz
        if self._faithful_mode:
            update_hz = 0.0
            force_dt = False

        self._update_period = 0.0 if update_hz <= 0 else 1.0 / update_hz
        self._force_dt = force_dt and self._update_period > 0
        self._next_tick = time.perf_counter()
        self._recenter = bool(getattr(cfg, "viser_recenter", True))
        self._scandots_enabled = bool(getattr(cfg, "viser_show_scandots", False))
        self._scandots_point_size = float(getattr(cfg, "viser_scandots_point_size", 0.02))

        if self._load_urdf_visuals:
            viser_mod, viser_urdf_cls, err = _import_viser()
            if err is not None or viser_mod is None or viser_urdf_cls is None:
                logger.warning("Viser live viewer disabled: {}", err or "missing dependency")
                self._enabled = False
                return
            self._viser_urdf_cls = viser_urdf_cls
        else:
            ensure_viser_on_path()
            try:
                import viser as viser_mod  # type: ignore[import-not-found]
            except Exception as exc:
                logger.warning("Viser live viewer disabled: {}", exc)
                self._enabled = False
                return
            logger.info("Viser running in Isaac-sim-state mode (URDF visuals disabled).")

        port_cfg = int(getattr(cfg, "viser_port", 0) or 0)
        port = resolve_viser_port(port_cfg)
        self._server = viser_mod.ViserServer(port=port)
        _ensure_viser_api_compat(self._server)

        self._global_frame_wxyz = _parse_quat_wxyz(getattr(cfg, "viser_global_frame_quat_wxyz", None))
        if self._global_frame_wxyz is not None:
            self._scene_prefix = "/viser_root"
            self._global_root = self._server.scene.add_frame(self._scene_prefix, show_axes=False)
            self._global_root.wxyz = self._global_frame_wxyz

        self._robot_root = self._server.scene.add_frame(self._scene_path("/robot"), show_axes=False)
        self._object_root = self._server.scene.add_frame(self._scene_path("/object"), show_axes=False)
        self._grid_handle = self._server.scene.add_grid(
            self._scene_path("/grid"), width=8.0, height=8.0, position=(0.0, 0.0, 0.0)
        )
        self._grid_handle.visible = False

        if self._load_urdf_visuals:
            robot_urdf = _resolve_robot_urdf_path(env.robot_config)
            self._vr = _create_viser_urdf_handle(
                self._viser_urdf_cls,
                self._server,
                robot_urdf,
                root_node_name=self._scene_path("/robot"),
            )
            self._setup_joint_order()
            self._setup_secondary_env_handles(self._viser_urdf_cls, robot_urdf)
        else:
            self._setup_secondary_env_frames_only()
        if self._load_urdf_visuals or self._sim_meshes_enabled:
            self._refresh_primary_object_handle()
            for env_id in self._secondary_env_ids:
                self._ensure_secondary_object_handle(env_id)
        self._load_terrain()
        self._setup_controls()

        logger.info("Viser live viewer listening on port {}", port)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _set_handle_visible(self, handle, visible: bool) -> None:
        if handle is None:
            return
        try:
            handle.visible = bool(visible)
        except Exception:
            return

    def _scene_path(self, path: str) -> str:
        if not self._scene_prefix:
            return path
        if not path.startswith("/"):
            path = "/" + path
        return f"{self._scene_prefix}{path}"

    def _viewer_env_shift(self, env_id: int) -> np.ndarray:
        slot = self._secondary_env_slot.get(int(env_id), 0)
        if slot <= 0:
            return np.zeros(3, dtype=np.float32)
        cols = max(1, int(self._viser_multi_env_cols))
        if cols <= 1:
            return np.array([0.0, float(slot) * self._viser_multi_env_spacing, 0.0], dtype=np.float32)
        row = slot // cols
        col = slot % cols
        return np.array(
            [
                float(row) * self._viser_multi_env_spacing,
                float(col) * self._viser_multi_env_spacing,
                0.0,
            ],
            dtype=np.float32,
        )

    def _env_sequence_label_position(self, env_id: int) -> np.ndarray:
        display_shift = np.zeros(3, dtype=np.float32)
        if int(env_id) != self._env_id:
            display_shift = self._viewer_env_shift(env_id)
        return display_shift + np.array([0.0, 0.0, self._env_sequence_label_height], dtype=np.float32)

    def _ensure_env_sequence_label_handle(self, env_id: int):
        if not self._show_env_sequence_labels or self._server is None:
            return None
        label_handle = self._env_sequence_label_handles.get(int(env_id))
        if label_handle is not None:
            return label_handle
        try:
            label_handle = self._server.scene.add_label(
                self._scene_path(f"/env_sequence_label_{int(env_id)}"),
                text="",
                position=self._env_sequence_label_position(env_id),
                visible=False,
            )
        except Exception:
            return None
        self._env_sequence_label_handles[int(env_id)] = label_handle
        return label_handle

    def _format_env_sequence_label_text(self, env_id: int, clip_name: str) -> str:
        clip_text = str(clip_name).strip()
        if not clip_text:
            return f"env{int(env_id)}"

        parts = [part for part in clip_text.split("_") if part]
        short_text = clip_text
        if len(parts) >= 3 and parts[0].startswith("sub") and parts[2].isdigit():
            short_text = f"{parts[0]}/{parts[2]}"
        elif len(parts) >= 2 and parts[-1].isdigit():
            short_text = f"{parts[0]}/{parts[-1]}"
        elif len(clip_text) > 24:
            short_text = f"{clip_text[:10]}...{clip_text[-8:]}"
        return f"env{int(env_id)}: {short_text}"

    def _log_env_sequence_assignments_once(self, motion_cmd) -> None:
        if self._env_sequence_label_debug_logged or motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return
        visible_env_ids = [self._env_id, *self._secondary_env_ids]
        assignments: list[str] = []
        for env_id in visible_env_ids:
            clip_idx = None
            try:
                clip_idx = int(motion_cmd.clip_ids[int(env_id)].item())
            except Exception:
                clip_idx = None
            clip_name = self._current_clip_name(motion_cmd, clip_idx)
            assignments.append(f"env{int(env_id)}={clip_name or 'unknown'}")
        logger.info("Viser env sequence labels: {}", assignments)
        self._env_sequence_label_debug_logged = True

    def _update_env_sequence_label(self, env_id: int, clip_name: str | None) -> None:
        if not self._show_env_sequence_labels:
            return
        label_handle = self._ensure_env_sequence_label_handle(env_id)
        if label_handle is None:
            return
        clip_text = "" if clip_name is None else str(clip_name).strip()
        if not clip_text:
            try:
                label_handle.visible = False
            except Exception:
                pass
            return
        label_handle.position = self._env_sequence_label_position(env_id)
        label_text = self._format_env_sequence_label_text(env_id, clip_text)
        try:
            if getattr(label_handle, "text", "") != label_text:
                label_handle.text = label_text
            label_handle.visible = True
        except Exception:
            pass

    def _current_mesh_mode(self) -> str:
        if self._mesh_mode_dropdown is None:
            return self._mesh_mode_default
        value = str(getattr(self._mesh_mode_dropdown, "value", self._mesh_mode_default)).strip().lower()
        if value not in SIM_MESH_MODE_OPTIONS:
            return self._mesh_mode_default
        return value

    def _mesh_mode_shows_visual(self) -> bool:
        return self._current_mesh_mode() in {"visual", "both"}

    def _mesh_mode_shows_collision(self) -> bool:
        return self._current_mesh_mode() in {"collision", "both"}

    def _robot_mesh_frame_node(self, env_id: int, body_name: str) -> str:
        if int(env_id) == self._env_id:
            return self._scene_path(f"/robot_mesh/{body_name}")
        return self._scene_path(f"/env_{env_id}/robot_mesh/{body_name}")

    def _resolve_live_robot_body_prim_path(self, env_id: int, body_name: str) -> str | None:
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]
        except Exception:
            return None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None

        direct_path = f"/World/envs/env_{env_id}/Robot/{body_name}"
        prim = stage.GetPrimAtPath(direct_path)
        if prim.IsValid():
            return direct_path

        robot_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
        if not robot_root.IsValid():
            return None

        for prim in Usd.PrimRange(robot_root):
            if not prim.IsValid():
                continue
            if prim.GetName() != body_name:
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        return None

    def _get_robot_body_states_wxyz_for(self, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        sim = getattr(self._env, "simulator", None)
        rigid_body_pos = getattr(sim, "_rigid_body_pos", None)
        rigid_body_quat_xyzw = getattr(sim, "_rigid_body_rot", None)
        env_idx = int(env_id)
        if rigid_body_pos is None or rigid_body_quat_xyzw is None:
            return None
        if env_idx < 0 or env_idx >= int(rigid_body_pos.shape[0]) or env_idx >= int(rigid_body_quat_xyzw.shape[0]):
            return None
        try:
            pos = rigid_body_pos[env_idx].detach().cpu().numpy()
            quat_xyzw = rigid_body_quat_xyzw[env_idx].detach().cpu().numpy()
        except Exception:
            return None
        if pos.ndim != 2 or quat_xyzw.ndim != 2 or pos.shape[0] != quat_xyzw.shape[0]:
            return None
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        return pos, quat_wxyz

    def _ensure_robot_mesh_handles_for_env(self, env_id: int) -> None:
        if not self._sim_meshes_enabled or self._server is None:
            return

        body_frames = self._robot_mesh_frames.setdefault(int(env_id), {})
        visual_handles = self._robot_visual_mesh_handles.setdefault(int(env_id), {})
        collision_handles = self._robot_collision_mesh_handles.setdefault(int(env_id), {})
        body_names = list(self._body_name_to_index.keys())
        unresolved_body_names: list[str] = []
        for body_name in body_names:
            if body_name in body_frames:
                continue

            prim_path = self._resolve_live_robot_body_prim_path(env_id, body_name)
            if not prim_path:
                unresolved_body_names.append(body_name)
                continue

            visual_mesh = _load_combined_live_usd_visual_mesh(prim_path)
            collision_mesh = _load_combined_live_usd_collision_mesh(prim_path)
            if visual_mesh is None and collision_mesh is None:
                continue

            frame_node = self._robot_mesh_frame_node(env_id, body_name)
            frame = self._server.scene.add_frame(frame_node, show_axes=False)
            body_frames[body_name] = frame

            if visual_mesh is not None:
                visual_handles[body_name] = self._server.scene.add_mesh_simple(
                    f"{frame_node}/visual",
                    visual_mesh.vertices,
                    visual_mesh.faces,
                    color=SIM_VISUAL_MESH_COLOR,
                    side="double",
                )
            if collision_mesh is not None:
                collision_handles[body_name] = self._server.scene.add_mesh_simple(
                    f"{frame_node}/collision",
                    collision_mesh.vertices,
                    collision_mesh.faces,
                    color=SIM_COLLISION_MESH_COLOR,
                    side="double",
                )

        if body_frames and int(env_id) not in self._robot_mesh_envs_logged:
            logger.info(
                "Viser robot env {} registered {} body frames (visual={} collision={})",
                env_id,
                len(body_frames),
                len(visual_handles),
                len(collision_handles),
            )
            self._robot_mesh_envs_logged.add(int(env_id))
        elif not body_frames and int(env_id) not in self._robot_mesh_envs_missing_logged:
            logger.warning(
                "Viser robot env {} found no live USD body meshes under /World/envs/env_{}/Robot. "
                "Unresolved bodies sample={}",
                env_id,
                env_id,
                unresolved_body_names[:8],
            )
            self._robot_mesh_envs_missing_logged.add(int(env_id))

        self._set_robot_mesh_visibility_for_env(env_id, visible=True)

    def _set_robot_mesh_visibility_for_env(self, env_id: int, visible: bool) -> None:
        show_visual = bool(visible and self._mesh_mode_shows_visual())
        show_collision = bool(visible and self._mesh_mode_shows_collision())
        for handle in self._robot_visual_mesh_handles.get(int(env_id), {}).values():
            _set_visual_handle_visible(handle, show_visual)
        for handle in self._robot_collision_mesh_handles.get(int(env_id), {}).values():
            _set_visual_handle_visible(handle, show_collision)

    def _update_robot_mesh_frames_for_env(
        self,
        env_id: int,
        *,
        offset: np.ndarray,
        display_shift: np.ndarray | None = None,
        visible: bool,
    ) -> bool:
        if not self._sim_meshes_enabled:
            return False

        self._ensure_robot_mesh_handles_for_env(env_id)
        body_frames = self._robot_mesh_frames.get(int(env_id), {})
        if not body_frames:
            return False

        body_states = self._get_robot_body_states_wxyz_for(env_id)
        if body_states is None:
            self._set_robot_mesh_visibility_for_env(env_id, visible=False)
            return False

        body_pos, body_quat_wxyz = body_states
        for body_name, frame in body_frames.items():
            body_idx = self._body_name_to_index.get(body_name)
            if body_idx is None or body_idx >= body_pos.shape[0]:
                continue
            position = body_pos[body_idx] - offset
            if display_shift is not None:
                position = position + display_shift
            frame.position = position.astype(np.float32, copy=False)
            frame.wxyz = body_quat_wxyz[body_idx].astype(np.float32, copy=False)

        self._set_robot_mesh_visibility_for_env(env_id, visible=visible)
        return True

    def _primary_object_variant_node(self, visual_key: str, mesh_kind: str = "visual") -> str:
        normalized_kind = _normalize_mesh_kind(mesh_kind)
        digest = hashlib.sha1(f"{normalized_kind}:{visual_key}".encode("utf-8")).hexdigest()[:10]
        return self._scene_path(f"/object/{normalized_kind}_variant_{digest}")

    def _summarize_object_mesh(self, mesh: Any) -> str:
        try:
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            extents = bounds[1] - bounds[0]
            extents_str = f"[{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]"
        except Exception:
            extents_str = "unknown"
        visual_prim_count = getattr(mesh, '_holosoma_visual_prim_count', None)
        if visual_prim_count is None:
            return f"extents={extents_str}"
        return f"extents={extents_str}, visual_prims={int(visual_prim_count)}"

    def _object_mesh_vertex_cache(self, mesh_kind: str = "visual") -> dict[str, np.ndarray]:
        normalized_kind = _normalize_mesh_kind(mesh_kind)
        if normalized_kind == "collision":
            return self._object_collision_mesh_vertices_local
        return self._object_visual_mesh_vertices_local

    def _resolve_object_extents_for_env(self, env_id: int) -> np.ndarray | None:
        object_urdf = self._resolve_object_urdf_for_env(env_id)
        if object_urdf:
            extents = load_urdf_geometry_extents(object_urdf)
            if extents is not None:
                return np.asarray(extents, dtype=np.float32)

        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not hasattr(motion_cmd, "object_size"):
            return None
        try:
            return motion_cmd.object_size[int(env_id)].detach().float().cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def _compute_box_bottom_z_from_state(
        self,
        object_state_wxyz: tuple[np.ndarray, np.ndarray] | None,
        extents_xyz: np.ndarray | None,
    ) -> float | None:
        if object_state_wxyz is None or extents_xyz is None:
            return None
        extents = np.asarray(extents_xyz, dtype=np.float32).reshape(-1)
        if extents.shape[0] != 3 or not np.all(np.isfinite(extents)):
            return None
        half_extents = 0.5 * np.maximum(extents, 1.0e-6)
        pos_w, quat_wxyz = object_state_wxyz
        try:
            quat_tensor = torch.as_tensor(quat_wxyz, dtype=torch.float32).reshape(1, 4)
            rot = quaternion_to_matrix(quat_tensor, w_last=False)[0].detach().cpu().numpy()
        except Exception:
            return None
        z_axis = np.asarray(rot[2], dtype=np.float32)
        support = float(np.abs(z_axis) @ half_extents)
        return float(pos_w[2]) - support

    def _invalidate_object_mesh_caches(self, *, remove_handles: bool = True) -> None:
        self._object_visual_mesh_vertices_local.clear()
        self._object_collision_mesh_vertices_local.clear()
        try:
            _load_combined_live_usd_visual_mesh.cache_clear()
            _load_combined_live_usd_collision_mesh.cache_clear()
        except Exception:
            pass
        self._active_primary_object_key = None
        self._active_primary_object_collision_key = None
        self._secondary_object_visual_key.clear()
        self._secondary_object_collision_key.clear()
        if not remove_handles:
            return
        for handle in list(self._primary_object_variants.values()) + list(self._primary_object_collision_variants.values()):
            try:
                handle.remove()
            except Exception:
                pass
        self._primary_object_variants.clear()
        self._primary_object_collision_variants.clear()
        self._vo = None
        self._vo_collision = None
        for handle_map in (self._secondary_vo, self._secondary_vo_collision):
            for handle in list(handle_map.values()):
                try:
                    handle.remove()
                except Exception:
                    pass
            handle_map.clear()

    def _get_object_mesh_vertices_local_for_env(
        self,
        env_id: int,
        mesh_kind: str = "visual",
    ) -> tuple[np.ndarray | None, str | None]:
        normalized_kind = _normalize_mesh_kind(mesh_kind)
        cache = self._object_mesh_vertex_cache(normalized_kind)

        if int(env_id) == self._env_id:
            active_key = (
                self._active_primary_object_collision_key
                if normalized_kind == "collision"
                else self._active_primary_object_key
            )
        else:
            active_key = (
                self._secondary_object_collision_key.get(int(env_id))
                if normalized_kind == "collision"
                else self._secondary_object_visual_key.get(int(env_id))
            )

        if active_key is not None and active_key in cache:
            return cache[active_key], active_key

        resolved_key, mesh, _ = self._resolve_object_mesh_spec_for_env(int(env_id), normalized_kind)
        if resolved_key is None or mesh is None:
            return None, resolved_key

        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
        except Exception:
            return None, resolved_key
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
            return None, resolved_key

        cache[resolved_key] = vertices
        return vertices, resolved_key

    def _compute_object_bottom_z_from_state(
        self,
        env_id: int,
        object_state_wxyz: tuple[np.ndarray, np.ndarray] | None,
        mesh_kind: str = "visual",
    ) -> float | None:
        if object_state_wxyz is None:
            return None
        normalized_kind = _normalize_mesh_kind(mesh_kind)
        if normalized_kind == "collision":
            analytic_bottom = self._compute_box_bottom_z_from_state(
                object_state_wxyz,
                self._resolve_object_extents_for_env(env_id),
            )
            if analytic_bottom is not None:
                return analytic_bottom
        vertices_local, _ = self._get_object_mesh_vertices_local_for_env(env_id, mesh_kind)
        if vertices_local is None:
            return None
        pos_w, quat_wxyz = object_state_wxyz
        try:
            quat_tensor = torch.as_tensor(quat_wxyz, dtype=torch.float32).reshape(1, 4)
            rot = quaternion_to_matrix(quat_tensor, w_last=False)[0].detach().cpu().numpy()
        except Exception:
            return None
        z_axis = rot[2].astype(np.float32, copy=False)
        bottom_z = np.min(vertices_local @ z_axis) + float(pos_w[2])
        return float(bottom_z)

    def _resolve_live_object_root_prim_path(self, env_id: int) -> str | None:
        sim_object_name = self._resolve_sim_object_name_for_env(int(env_id))
        if not sim_object_name:
            return None

        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import Usd  # type: ignore[import-not-found]
        except Exception:
            return None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None

        simulator = getattr(self._env, 'simulator', None)
        scene = getattr(simulator, 'scene', None)
        rigid_objects = getattr(scene, 'rigid_objects', None)
        rigid_object = rigid_objects.get(sim_object_name) if hasattr(rigid_objects, 'get') else None

        candidate_paths: list[str] = []
        prim_expr = str(getattr(getattr(rigid_object, 'cfg', None), 'prim_path', '')).strip()
        if prim_expr:
            candidate_paths.append(prim_expr.replace('env_.*', f'env_{env_id}'))
            candidate_paths.append(prim_expr.replace('env_.*/', f'env_{env_id}/'))

        for candidate_path in dict.fromkeys(candidate_paths):
            prim = stage.GetPrimAtPath(candidate_path)
            if prim.IsValid():
                return str(prim.GetPath())

        env_root = stage.GetPrimAtPath(f'/World/envs/env_{env_id}')
        if not env_root.IsValid():
            return None

        target_name = sim_object_name.casefold()
        for prim in Usd.PrimRange(env_root):
            if not prim.IsValid():
                continue
            prim_name = prim.GetName()
            prim_name_key = prim_name.casefold()
            if prim_name.startswith('Object') and (prim_name_key == target_name or prim_name_key.endswith(f'_{target_name}')):
                return str(prim.GetPath())
        return None

    def _live_object_mesh_signature_for_env(self, env_id: int) -> str:
        parts: list[str] = []
        motion_cmd = self._get_motion_command()
        clip_idx: int | None = None
        if motion_cmd is not None and hasattr(motion_cmd, "clip_ids"):
            try:
                clip_idx = int(motion_cmd.clip_ids[int(env_id)].item())
            except Exception:
                clip_idx = None
        clip_name = self._current_clip_name(motion_cmd, clip_idx)
        if clip_name:
            parts.append(f"clip={clip_name}")
        object_urdf = self._resolve_object_urdf_for_env(env_id)
        if object_urdf:
            parts.append(f"urdf={object_urdf}")
        sim_object_name = self._resolve_sim_object_name_for_env(env_id)
        if sim_object_name:
            parts.append(f"obj={sim_object_name}")
        if not parts:
            parts.append(f"env={int(env_id)}")
        return "|".join(parts)

    def _resolve_object_mesh_spec_for_env(self, env_id: int, mesh_kind: str = "visual") -> tuple[str | None, Any | None, str]:
        normalized_kind = _normalize_mesh_kind(mesh_kind)
        live_root_prim_path = self._resolve_live_object_root_prim_path(env_id)
        if live_root_prim_path:
            live_signature = self._live_object_mesh_signature_for_env(env_id)
            loader = (
                _load_combined_live_usd_collision_mesh
                if normalized_kind == "collision"
                else _load_combined_live_usd_visual_mesh
            )
            mesh = loader(live_root_prim_path, live_signature)
            if mesh is not None:
                return f"{live_root_prim_path}::{live_signature}", mesh, f'live-usd:{live_root_prim_path}'

        if self._mesh_source == "sim":
            return None, None, ''

        object_urdf = self._resolve_object_urdf_for_env(env_id)
        if object_urdf:
            loader = _load_combined_urdf_collision_mesh if normalized_kind == "collision" else _load_combined_urdf_visual_mesh
            mesh = loader(object_urdf)
            if mesh is not None:
                return object_urdf, mesh, f'urdf-asset:{object_urdf}'

        return None, None, ''

    def _ensure_primary_object_variant(self, env_id: int, mesh_kind: str = "visual") -> Any | None:
        if self._server is None:
            return None

        normalized_kind = _normalize_mesh_kind(mesh_kind)
        variant_cache = (
            self._primary_object_collision_variants if normalized_kind == "collision" else self._primary_object_variants
        )
        visual_key, mesh, source_label = self._resolve_object_mesh_spec_for_env(env_id, normalized_kind)
        if not visual_key or mesh is None:
            return None

        cached = variant_cache.get(visual_key)
        if cached is not None:
            return cached

        try:
            handle = self._server.scene.add_mesh_simple(
                self._primary_object_variant_node(visual_key, normalized_kind),
                mesh.vertices,
                mesh.faces,
                color=OBJECT_MESH_COLOR,
                side='double',
            )
        except Exception as exc:
            logger.warning('Viser object {} mesh disabled (failed to create {}): {}', normalized_kind, source_label, exc)
            return None

        _set_visual_handle_visible(handle, False)
        variant_cache[visual_key] = handle
        logger.info(
            'Viser primary object env {} using {} {} ({})',
            env_id,
            normalized_kind,
            source_label,
            self._summarize_object_mesh(mesh),
        )
        return handle

    def _apply_primary_object_mesh_visibility(self) -> None:
        show_object = self._show_object_cb is None or bool(self._show_object_cb.value)
        show_visual = bool(show_object and self._mesh_mode_shows_visual())
        show_collision = bool(show_object and self._mesh_mode_shows_collision())
        # Primitive boxes in live Isaac Sim sometimes expose only collision geometry.
        # In visual-only mode, fall back to collision so the active box remains visible.
        if show_visual and self._vo is None and self._vo_collision is not None:
            show_collision = True
        for variant in self._primary_object_variants.values():
            _set_visual_handle_visible(variant, False)
        for variant in self._primary_object_collision_variants.values():
            _set_visual_handle_visible(variant, False)
        if self._vo is not None:
            _set_visual_handle_visible(self._vo, show_visual)
        if self._vo_collision is not None:
            _set_visual_handle_visible(self._vo_collision, show_collision)

    def _set_primary_object_meshes(self, env_id: int) -> None:
        visual_key, _, _ = self._resolve_object_mesh_spec_for_env(env_id, "visual")
        collision_key, _, _ = self._resolve_object_mesh_spec_for_env(env_id, "collision")

        self._active_primary_object_key = visual_key
        self._active_primary_object_collision_key = collision_key
        self._vo = self._ensure_primary_object_variant(env_id, "visual") if visual_key else None
        self._vo_collision = self._ensure_primary_object_variant(env_id, "collision") if collision_key else None
        self._apply_primary_object_mesh_visibility()

    def _refresh_primary_object_handle(self) -> None:
        self._set_primary_object_meshes(self._env_id)

    def _resolve_sim_object_name_for_env(self, env_id: int) -> str | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None

        # Multi-object mapping (clip -> object_id -> simulator object name)
        sim_object_names = list(getattr(motion_cmd, "_sim_object_names", []))
        clip_object_ids = getattr(motion_cmd, "_clip_object_ids", None)
        clip_ids = getattr(motion_cmd, "clip_ids", None)
        representative_clip_ids = getattr(motion_cmd, "_debug_representative_clip_ids", None)
        if sim_object_names and clip_object_ids is not None:
            if clip_ids is not None:
                try:
                    clip_idx = int(clip_ids[int(env_id)].item())
                    object_id = int(clip_object_ids[int(clip_idx)].item())
                    if 0 <= object_id < len(sim_object_names):
                        return str(sim_object_names[object_id])
                except Exception:
                    pass
            if representative_clip_ids is not None:
                try:
                    rep_count = int(representative_clip_ids.numel())
                except Exception:
                    rep_count = 0
                if rep_count > 0:
                    try:
                        clip_idx = int(representative_clip_ids[int(env_id) % rep_count].item())
                        object_id = int(clip_object_ids[int(clip_idx)].item())
                        if 0 <= object_id < len(sim_object_names):
                            return str(sim_object_names[object_id])
                    except Exception:
                        pass
            if bool(getattr(motion_cmd, "_multi_object_enabled", False)):
                return None

        # Single-object mapping fallback
        object_name = str(getattr(motion_cmd, "object_name", "")).strip()
        if object_name:
            return object_name
        return None

    def _get_simulator_object_state_xyzw_for(self, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        sim = getattr(self._env, "simulator", None)
        if sim is None:
            return None

        env_ids = torch.tensor([int(env_id)], device=self._env.device, dtype=torch.long)
        sim_object_name = self._resolve_sim_object_name_for_env(int(env_id))
        candidate_names: list[str] = []
        if sim_object_name:
            candidate_names.append(sim_object_name)
        if "object" not in candidate_names:
            candidate_names.append("object")

        for object_name in candidate_names:
            try:
                states = sim.get_actor_states([object_name], env_ids)
            except Exception:
                states = None
            if states is None or states.numel() == 0:
                continue
            state = states[0]
            pos = state[0:3].detach().cpu().numpy()
            quat_xyzw = state[3:7].detach().cpu().numpy()
            return pos, quat_xyzw
        return None

    def _resolve_object_urdf_for_env(self, env_id: int) -> str | None:
        motion_cmd = self._get_motion_command()
        multi_object_enabled = bool(getattr(motion_cmd, "_multi_object_enabled", False)) if motion_cmd else False
        fallback = _resolve_object_urdf_path(self._env.robot_config)

        sim = getattr(self._env, "simulator", None)
        env_object_urdf_paths = getattr(sim, "_env_object_urdf_paths", None)
        if isinstance(env_object_urdf_paths, list) and 0 <= int(env_id) < len(env_object_urdf_paths):
            env_urdf = str(env_object_urdf_paths[int(env_id)]).strip()
            candidate = Path(env_urdf)
            if env_urdf and candidate.exists() and candidate.suffix.lower() == ".urdf":
                return str(candidate)

        object_urdf_by_name = getattr(sim, "_object_urdf_by_name", {})
        if isinstance(object_urdf_by_name, dict) and object_urdf_by_name:
            sim_object_name = self._resolve_sim_object_name_for_env(int(env_id))
            if sim_object_name:
                urdf_path = str(object_urdf_by_name.get(sim_object_name, "")).strip()
                if urdf_path:
                    candidate = Path(urdf_path)
                    if candidate.exists() and candidate.suffix.lower() == ".urdf":
                        return str(candidate)
            if len(object_urdf_by_name) == 1:
                only_urdf = str(next(iter(object_urdf_by_name.values()))).strip()
                candidate = Path(only_urdf)
                if candidate.exists() and candidate.suffix.lower() == ".urdf":
                    return str(candidate)

        if multi_object_enabled:
            return None
        return fallback

    def _setup_secondary_env_handles(self, viser_urdf_cls: Any, robot_urdf: str) -> None:
        if not self._secondary_env_ids or self._server is None:
            return
        for env_id in self._secondary_env_ids:
            robot_node = self._scene_path(f"/env_{env_id}/robot")
            object_node = self._scene_path(f"/env_{env_id}/object")
            try:
                self._secondary_robot_roots[env_id] = self._server.scene.add_frame(robot_node, show_axes=False)
                self._secondary_vr[env_id] = _create_viser_urdf_handle(
                    viser_urdf_cls,
                    self._server,
                    robot_urdf,
                    root_node_name=robot_node,
                )
            except Exception as exc:
                logger.warning("Failed to initialize secondary robot view for env {}: {}", env_id, exc)
                continue

            if env_id not in self._secondary_object_roots:
                try:
                    self._secondary_object_roots[env_id] = self._server.scene.add_frame(object_node, show_axes=False)
                except Exception:
                    pass
            self._ensure_secondary_object_handle(env_id)

    def _setup_secondary_env_frames_only(self) -> None:
        if not self._secondary_env_ids or self._server is None:
            return
        for env_id in self._secondary_env_ids:
            robot_node = self._scene_path(f"/env_{env_id}/robot")
            object_node = self._scene_path(f"/env_{env_id}/object")
            if env_id not in self._secondary_robot_roots:
                self._secondary_robot_roots[env_id] = self._server.scene.add_frame(robot_node, show_axes=False)
            if env_id not in self._secondary_object_roots:
                self._secondary_object_roots[env_id] = self._server.scene.add_frame(object_node, show_axes=False)

    def _apply_secondary_object_mesh_visibility(self, env_id: int, visible: bool) -> None:
        show_visual = bool(visible and self._mesh_mode_shows_visual())
        show_collision = bool(visible and self._mesh_mode_shows_collision())
        if show_visual and self._secondary_vo.get(env_id) is None and self._secondary_vo_collision.get(env_id) is not None:
            show_collision = True
        _set_visual_handle_visible(self._secondary_vo.get(env_id), show_visual)
        _set_visual_handle_visible(self._secondary_vo_collision.get(env_id), show_collision)

    def _ensure_secondary_object_handle(self, env_id: int) -> None:
        if self._server is None:
            return

        root_handle = self._secondary_object_roots.get(env_id)
        if root_handle is None:
            try:
                self._secondary_object_roots[env_id] = self._server.scene.add_frame(
                    self._scene_path(f"/env_{env_id}/object"), show_axes=False
                )
            except Exception:
                return

        mesh_configs = (
            ("visual", self._secondary_vo, self._secondary_object_visual_key, OBJECT_MESH_COLOR),
            ("collision", self._secondary_vo_collision, self._secondary_object_collision_key, OBJECT_MESH_COLOR),
        )
        for mesh_kind, handle_map, key_map, color in mesh_configs:
            visual_key, mesh, source_label = self._resolve_object_mesh_spec_for_env(env_id, mesh_kind)
            if not visual_key or mesh is None:
                key_map.pop(env_id, None)
                stale_handle = handle_map.pop(env_id, None)
                if stale_handle is not None:
                    try:
                        stale_handle.remove()
                    except Exception:
                        pass
                continue

            current_key = key_map.get(env_id)
            if env_id in handle_map and current_key == visual_key:
                continue

            stale_handle = handle_map.pop(env_id, None)
            if stale_handle is not None:
                try:
                    stale_handle.remove()
                except Exception:
                    pass

            object_node = self._scene_path(f"/env_{env_id}/object/{mesh_kind}")
            try:
                handle = self._server.scene.add_mesh_simple(
                    object_node,
                    mesh.vertices,
                    mesh.faces,
                    color=color,
                    side='double',
                )
            except Exception as exc:
                key_map.pop(env_id, None)
                logger.warning(
                    'Failed to create secondary object {} mesh env {} from {}: {}',
                    mesh_kind,
                    env_id,
                    source_label,
                    exc,
                )
                continue

            handle_map[env_id] = handle
            key_map[env_id] = visual_key
            logger.info(
                'Viser secondary object env {} using {} {} ({})',
                env_id,
                mesh_kind,
                source_label,
                self._summarize_object_mesh(mesh),
            )

        self._apply_secondary_object_mesh_visibility(env_id, visible=True)

    def _resolve_root_body_index(self) -> None:
        body_names = getattr(self._env, "body_names", None)
        if body_names is None:
            return
        try:
            names = [str(name) for name in body_names]
        except Exception:
            return
        if not names:
            return
        for candidate in ("pelvis", "base_link", "torso_link"):
            if candidate in names:
                self._root_body_index = int(names.index(candidate))
                self._root_body_name = candidate
                return
        self._root_body_index = 0
        self._root_body_name = names[0]

    def _set_sim_cfg(self, field: str, value) -> None:
        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if sim_cfg is None:
            return
        try:
            object.__setattr__(sim_cfg, field, value)
        except Exception:
            return

    def _clear_terrain_handles(self) -> None:
        for handle in (self._terrain_handle, self._ground_handle):
            if handle is None:
                continue
            try:
                handle.remove()
            except Exception:
                pass
        self._terrain_handle = None
        self._ground_handle = None

    def _update_terrain_transform(self, viewer_offset: np.ndarray | None = None) -> None:
        handles = [handle for handle in (self._terrain_handle, self._ground_handle) if handle is not None]
        if not handles:
            return
        terrain_offset = np.zeros(3, dtype=np.float32)
        if self._terrain_is_local:
            resolved = self._resolve_env_origin()
            if resolved is not None:
                terrain_offset = resolved
        if viewer_offset is None:
            if self._recenter and self._offset is not None:
                viewer_offset = self._offset
            else:
                viewer_offset = np.zeros(3, dtype=np.float32)
        for handle in handles:
            try:
                handle.position = terrain_offset - viewer_offset
            except Exception:
                pass

    def _reload_terrain_for_clip(self, clip_name: str | None) -> None:
        if not self._enabled or self._server is None:
            return
        if clip_name == self._terrain_clip_name:
            return
        self._clear_terrain_handles()
        self._load_terrain(clip_name=clip_name)
        self._update_terrain_transform()

    def _invalidate_isaac_scandots_payload(self) -> None:
        # Clip/env switches can update robot pose immediately while Isaac payload still carries prior-step rays.
        if hasattr(self._env, "_isaac_scandots_payload"):
            self._env._isaac_scandots_payload = None
        if hasattr(self._env, "_isaac_scandots_last_update"):
            try:
                self._env._isaac_scandots_last_update = 0.0
            except Exception:
                pass

    def wait_if_paused(self) -> None:
        if not self._enabled or self._play_control is None:
            return
        while not bool(self._play_control.value):
            if self._step_requested:
                self._step_requested = False
                return
            if (
                self._pending_control_sync
                or self._reset_visible_requested
                or self._reset_requested
                or self._pending_clip_idx is not None
                or self._pending_clip_start is not None
                or self._pending_visible_group_index is not None
            ):
                self.apply_pending_controls()
            time.sleep(self._control_sleep_s)
        if (
            self._pending_control_sync
            or self._reset_visible_requested
            or self._reset_requested
            or self._pending_clip_idx is not None
            or self._pending_clip_start is not None
            or self._pending_visible_group_index is not None
        ):
            self.apply_pending_controls()

    def queue_pending_controls(self) -> None:
        self._pending_control_sync = True

    def apply_pending_controls(self) -> None:
        if not self._enabled:
            return
        self._pending_control_sync = False
        self._update_manual_root_command()
        self._update_auto_forward_after_lift()
        self._update_manual_drop_button()
        self._update_manual_object_reset_override()
        if self._reset_visible_requested:
            self._reset_visible_requested = False
            self._reset_visible_envs()
        if self._reset_requested:
            self._reset_requested = False
            self._reset_env()
        if self._pending_visible_group_index is not None:
            self._apply_visible_group_selection()
        if self._pending_clip_idx is not None or self._pending_clip_start is not None:
            self._apply_clip_selection()

    def _bridge_hw_joystick_enabled(self) -> bool:
        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        bridge_cfg = getattr(sim_cfg, "bridge", None)
        return bool(getattr(bridge_cfg, "use_joystick", False))

    def _resolve_manual_hw_axes(self) -> dict[str, int]:
        if self._manual_hw_type == "switch":
            defaults = {"LX": 0, "LY": 1, "RX": 2}
        elif sys.platform.startswith("linux"):
            defaults = {"LX": 0, "LY": 1, "RX": 3}
        else:
            defaults = {"LX": 0, "LY": 1, "RX": 2}
        return {
            "LX": _axis_override("VISER_MANUAL_HW_AXIS_LX", defaults["LX"]),
            "LY": _axis_override("VISER_MANUAL_HW_AXIS_LY", defaults["LY"]),
            "RX": _axis_override("VISER_MANUAL_HW_AXIS_RX", defaults["RX"]),
        }

    def _init_manual_hw_joystick(self) -> bool:
        if self._manual_hw_init_attempted:
            return self._manual_hw_joystick is not None and self._manual_hw_pygame is not None
        self._manual_hw_init_attempted = True
        try:
            import pygame
        except Exception as exc:
            if not self._manual_hw_warned:
                self._manual_hw_warned = True
                logger.warning("Viser manual joystick disabled: pygame import failed ({})", exc)
            return False
        try:
            pygame.init()
            pygame.joystick.init()
            joystick_count = int(pygame.joystick.get_count())
            if joystick_count <= self._manual_hw_device:
                raise RuntimeError(f"device index {self._manual_hw_device} is unavailable; found {joystick_count} device(s)")
            joystick = pygame.joystick.Joystick(self._manual_hw_device)
            joystick.init()
            self._manual_hw_pygame = pygame
            self._manual_hw_joystick = joystick
            self._manual_hw_axes = self._resolve_manual_hw_axes()
            logger.info(
                "Viser manual joystick initialized (backend=pygame, device={}, name='{}', type={})",
                self._manual_hw_device,
                joystick.get_name(),
                self._manual_hw_type,
            )
            return True
        except Exception as exc:
            if not self._manual_hw_warned:
                self._manual_hw_warned = True
                logger.warning("Viser manual joystick unavailable via pygame: {}", exc)
            return False

    def _hw_joystick_mode_enabled(self) -> bool:
        if not self._manual_use_hw_joystick:
            return False
        if self._manual_hw_backend in ("auto", "bridge") and self._bridge_hw_joystick_enabled():
            return True
        if self._manual_hw_backend in ("auto", "pygame"):
            return self._init_manual_hw_joystick()
        return False

    def _hw_joystick_axes(self) -> tuple[float, float, float]:
        if self._manual_hw_backend in ("auto", "bridge") and self._bridge_hw_joystick_enabled():
            bridge = getattr(self._env.simulator, "bridge", None)
            robot_bridge = getattr(bridge, "robot_bridge", None)
            wc = getattr(robot_bridge, "wireless_controller", None)
            if wc is not None:
                try:
                    lx = float(getattr(wc, "lx", 0.0))
                    ly = float(getattr(wc, "ly", 0.0))
                    rx = float(getattr(wc, "rx", 0.0))
                except Exception:
                    lx, ly, rx = 0.0, 0.0, 0.0
                if np.isfinite(lx) and np.isfinite(ly) and np.isfinite(rx):
                    return lx, ly, rx

        if self._manual_hw_backend not in ("auto", "pygame"):
            return 0.0, 0.0, 0.0
        if not self._init_manual_hw_joystick():
            return 0.0, 0.0, 0.0
        if self._manual_hw_joystick is None or self._manual_hw_pygame is None:
            return 0.0, 0.0, 0.0

        lx_axis = self._manual_hw_axes.get("LX", 0)
        ly_axis = self._manual_hw_axes.get("LY", 1)
        rx_axis = self._manual_hw_axes.get("RX", 2)
        try:
            self._manual_hw_pygame.event.pump()
            lx = float(self._manual_hw_joystick.get_axis(lx_axis))
            ly = -float(self._manual_hw_joystick.get_axis(ly_axis))
            rx = float(self._manual_hw_joystick.get_axis(rx_axis))
        except Exception:
            return 0.0, 0.0, 0.0

        if not np.isfinite(lx):
            lx = 0.0
        if not np.isfinite(ly):
            ly = 0.0
        if not np.isfinite(rx):
            rx = 0.0
        return lx, ly, rx

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self._manual_hw_deadzone else value

    def _clear_manual_commands(self, *, clear_gui_toggles: bool = False) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is not None:
            manual_xy = getattr(motion_cmd, "manual_xy_rel", None)
            if isinstance(manual_xy, torch.Tensor) and manual_xy.numel() > 0:
                manual_xy.zero_()
            manual_yaw = getattr(motion_cmd, "manual_yaw_rel", None)
            if isinstance(manual_yaw, torch.Tensor) and manual_yaw.numel() > 0:
                manual_yaw.zero_()
        self._clear_manual_drop_button(clear_gui_toggle=clear_gui_toggles)
        self._hide_manual_command_arrow()
        self._update_manual_root_status()

        if not clear_gui_toggles:
            return
        for cb in (
            self._manual_forward_cb,
            self._manual_back_cb,
            self._manual_left_cb,
            self._manual_right_cb,
            self._manual_yaw_left_cb,
            self._manual_yaw_right_cb,
        ):
            if cb is not None:
                try:
                    cb.value = False
                except Exception:
                    pass

    def _set_manual_drop_button(self, value: bool) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return
        device = self._env.device
        drop_value = 1.0 if value else 0.0
        motion_cmd.manual_drop_button_override_enabled = bool(self._drop_button_gui_enabled)
        motion_cmd.manual_drop_button = torch.full(
            (self._env.num_envs, 1),
            drop_value,
            device=device,
            dtype=torch.float32,
        )

    def _clear_manual_drop_button(self, *, clear_gui_toggle: bool) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is not None:
            manual_drop_button = getattr(motion_cmd, "manual_drop_button", None)
            if isinstance(manual_drop_button, torch.Tensor) and manual_drop_button.numel() > 0:
                manual_drop_button.zero_()
            motion_cmd.manual_drop_button_override_enabled = bool(self._drop_button_gui_enabled)
        if clear_gui_toggle and self._drop_button_cb is not None:
            try:
                self._drop_button_cb.value = False
            except Exception:
                pass
        self._update_drop_button_status()

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return float(default)
        try:
            value = float(str(raw).strip())
        except Exception:
            return float(default)
        return value if np.isfinite(value) else float(default)

    @staticmethod
    def _parse_float_list_env(name: str, default: list[float], *, expected_len: int) -> list[float]:
        raw = os.environ.get(name)
        if raw is None or not str(raw).strip():
            return list(default)
        compact = str(raw).strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parts = [part.strip() for part in compact.split(",") if part.strip()]
        values: list[float] = []
        for part in parts:
            try:
                values.append(float(part))
            except Exception:
                return list(default)
        if len(values) != expected_len or any(not np.isfinite(value) for value in values):
            return list(default)
        return values

    def _set_manual_root_command(
        self,
        command_xy: np.ndarray | list[float] | tuple[float, float],
        command_yaw: float,
        *,
        sync_gui: bool,
    ) -> None:
        cmd_xy = np.asarray(command_xy, dtype=np.float32).reshape(-1)
        if cmd_xy.size < 2:
            return
        self._manual_root_cmd_xy = np.array([float(cmd_xy[0]), float(cmd_xy[1])], dtype=np.float32)
        self._manual_root_cmd_yaw = self._wrap_to_pi(float(command_yaw))
        if not sync_gui:
            return
        try:
            if self._manual_root_pos_x_slider is not None:
                self._manual_root_pos_x_slider.value = float(self._manual_root_cmd_xy[0])
            if self._manual_root_pos_y_slider is not None:
                self._manual_root_pos_y_slider.value = float(self._manual_root_cmd_xy[1])
            if self._manual_root_yaw_slider is not None:
                self._manual_root_yaw_slider.value = float(self._manual_root_cmd_yaw)
        except Exception:
            pass

    def _sync_manual_root_command_from_gui(self) -> bool:
        if (
            self._manual_root_pos_x_slider is None
            or self._manual_root_pos_y_slider is None
            or self._manual_root_yaw_slider is None
        ):
            return False
        self._set_manual_root_command(
            (
                float(self._manual_root_pos_x_slider.value),
                float(self._manual_root_pos_y_slider.value),
            ),
            float(self._manual_root_yaw_slider.value),
            sync_gui=False,
        )
        return True

    def _sync_manual_root_command_from_robot(self) -> None:
        self._set_manual_root_command((0.0, 0.0), 0.0, sync_gui=True)

    def _manual_root_command_payload(
        self,
        *,
        current_pos: np.ndarray | None = None,
        current_quat_wxyz: np.ndarray | None = None,
        command_xy: np.ndarray | list[float] | tuple[float, float] | None = None,
        command_yaw: float | None = None,
    ) -> dict[str, np.ndarray | float] | None:
        if current_pos is None or current_quat_wxyz is None:
            current_pos, current_quat_wxyz = self._get_root_state_wxyz()
        if current_pos is None or current_quat_wxyz is None:
            return None

        if command_xy is None or command_yaw is None:
            if self._manual_root_cmd_xy is None or self._manual_root_cmd_yaw is None:
                if not self._sync_manual_root_command_from_gui():
                    self._set_manual_root_command((0.0, 0.0), 0.0, sync_gui=False)
            command_xy = np.asarray(self._manual_root_cmd_xy, dtype=np.float32)
            command_yaw = float(self._manual_root_cmd_yaw)
        else:
            command_xy = np.asarray(command_xy, dtype=np.float32).reshape(-1)
            if command_xy.size < 2:
                return None
            command_xy = np.array([float(command_xy[0]), float(command_xy[1])], dtype=np.float32)
            command_yaw = float(command_yaw)

        current_pos = np.asarray(current_pos, dtype=np.float32)
        current_quat_wxyz = np.asarray(current_quat_wxyz, dtype=np.float32)
        current_yaw = self._yaw_from_quat_wxyz(current_quat_wxyz)
        target_yaw = self._wrap_to_pi(current_yaw + float(command_yaw))
        cy = float(np.cos(current_yaw))
        sy = float(np.sin(current_yaw))
        delta_world = np.array(
            [
                cy * float(command_xy[0]) - sy * float(command_xy[1]),
                sy * float(command_xy[0]) + cy * float(command_xy[1]),
                0.0,
            ],
            dtype=np.float32,
        )
        target_pos = current_pos.copy()
        target_pos[:2] = current_pos[:2] + delta_world[:2]

        return {
            "cmd_xy": command_xy,
            "cmd_yaw": self._wrap_to_pi(float(command_yaw)),
            "current_pos": current_pos,
            "target_pos": target_pos,
            "current_yaw": float(current_yaw),
            "target_yaw": float(target_yaw),
            "delta_world": delta_world,
            "dist_xy": float(np.linalg.norm(delta_world[:2])),
        }

    @staticmethod
    def _format_policy_root_command(cmd_x: float, cmd_y: float, cmd_yaw: float) -> str:
        return f"Policy cmd(root): `dx={cmd_x:+.2f}` `dy={cmd_y:+.2f}` `dyaw={cmd_yaw:+.2f}`"

    def _manual_policy_root_command(self) -> tuple[float, float, float] | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None

        manual_xy = getattr(motion_cmd, "manual_xy_rel", None)
        manual_yaw = getattr(motion_cmd, "manual_yaw_rel", None)
        if not isinstance(manual_xy, torch.Tensor) or manual_xy.ndim < 2:
            return None
        if self._env_id < 0 or self._env_id >= int(manual_xy.shape[0]):
            return None

        cmd_x = float(manual_xy[self._env_id, 0].item())
        cmd_y = float(manual_xy[self._env_id, 1].item())
        cmd_yaw = 0.0
        if isinstance(manual_yaw, torch.Tensor) and manual_yaw.ndim >= 2 and self._env_id < int(manual_yaw.shape[0]):
            cmd_yaw = float(manual_yaw[self._env_id, 0].item())
        return cmd_x, cmd_y, cmd_yaw

    def _current_effective_goal_xy(self) -> np.ndarray | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None

        obs_mgr = getattr(self._env, "observation_manager", None)
        if obs_mgr is None:
            return None

        if hasattr(obs_mgr, "compute_group"):
            try:
                obs = obs_mgr.compute_group("actor_obs_drop")
            except Exception:
                return None
            if isinstance(obs, torch.Tensor) and obs.ndim >= 2 and int(obs.shape[1]) >= 2:
                width = min(int(obs.shape[1]), 2)
                return obs[self._env_id, :width].detach().float().cpu().numpy()
        return None

    def _zero_object_reset_overrides(self) -> None:
        controls = (
            self._object_reset_pos_x_slider,
            self._object_reset_pos_y_slider,
            self._object_reset_pos_z_slider,
            self._object_reset_roll_slider,
            self._object_reset_pitch_slider,
            self._object_reset_yaw_slider,
        )
        for control in controls:
            if control is None:
                continue
            try:
                control.value = 0.0
            except Exception:
                pass

    def _clip_lock_enabled(self) -> bool:
        if self._clip_lock_cb is not None:
            return bool(self._clip_lock_cb.value)
        return bool(self._clip_lock_default)

    def _set_manual_object_reset_override(
        self,
        *,
        enabled: bool,
        pos_offset_w: np.ndarray | None = None,
        rpy_offset: np.ndarray | None = None,
    ) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._update_object_reset_status()
            return

        if pos_offset_w is None:
            pos_offset_w = np.zeros(3, dtype=np.float32)
        else:
            pos_offset_w = np.asarray(pos_offset_w, dtype=np.float32).reshape(3)
        if rpy_offset is None:
            rpy_offset = np.zeros(3, dtype=np.float32)
        else:
            rpy_offset = np.asarray(rpy_offset, dtype=np.float32).reshape(3)

        if self._object_reset_override_cb is not None:
            self._object_reset_override_cb.value = bool(enabled)
        slider_controls = (
            self._object_reset_pos_x_slider,
            self._object_reset_pos_y_slider,
            self._object_reset_pos_z_slider,
            self._object_reset_roll_slider,
            self._object_reset_pitch_slider,
            self._object_reset_yaw_slider,
        )
        for control, value in zip(slider_controls, np.concatenate((pos_offset_w, rpy_offset)), strict=False):
            if control is None:
                continue
            control.value = float(value)

        device = self._env.device
        motion_cmd.manual_object_reset_enabled = bool(enabled)
        motion_cmd.manual_object_reset_pos_offset_w = torch.tensor(
            [pos_offset_w],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)
        motion_cmd.manual_object_reset_rpy_offset = torch.tensor(
            [rpy_offset],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)
        self._update_object_reset_status()

    def _clear_manual_object_reset_override(self) -> None:
        self._set_manual_object_reset_override(enabled=False)

    def _randomize_object_reset_override(self) -> None:
        xy_range = max(0.0, float(os.environ.get("VISER_RESET_BOX_RANDOM_XY_RANGE", "0.5")))
        pos_offset_w = np.array(
            [
                np.random.uniform(-xy_range, xy_range),
                np.random.uniform(-xy_range, xy_range),
                0.0,
            ],
            dtype=np.float32,
        )
        self._set_manual_object_reset_override(enabled=True, pos_offset_w=pos_offset_w)
        self._reset_requested = True
        self.queue_pending_controls()

    def _update_object_reset_status(self) -> None:
        if self._object_reset_status is None:
            return
        motion_cmd = self._get_motion_command()
        enabled = bool(self._object_reset_override_cb.value) if self._object_reset_override_cb is not None else False
        if self._object_reset_override_cb is None and motion_cmd is not None:
            enabled = bool(getattr(motion_cmd, "manual_object_reset_enabled", False))
        if not enabled:
            self._object_reset_status.content = (
                "Mode: `off`\n\n"
                "Applies on reset while enabled.\n\n"
                "Runtime size scaling is not supported yet; size still comes from the spawned URDF scale."
            )
            return

        dx = float(self._object_reset_pos_x_slider.value) if self._object_reset_pos_x_slider is not None else 0.0
        dy = float(self._object_reset_pos_y_slider.value) if self._object_reset_pos_y_slider is not None else 0.0
        dz = float(self._object_reset_pos_z_slider.value) if self._object_reset_pos_z_slider is not None else 0.0
        dr = float(self._object_reset_roll_slider.value) if self._object_reset_roll_slider is not None else 0.0
        dp = float(self._object_reset_pitch_slider.value) if self._object_reset_pitch_slider is not None else 0.0
        dyaw = float(self._object_reset_yaw_slider.value) if self._object_reset_yaw_slider is not None else 0.0
        if motion_cmd is not None and self._object_reset_pos_x_slider is None:
            pos_offset_w = getattr(motion_cmd, "manual_object_reset_pos_offset_w", None)
            rpy_offset = getattr(motion_cmd, "manual_object_reset_rpy_offset", None)
            if isinstance(pos_offset_w, torch.Tensor) and pos_offset_w.ndim >= 2 and pos_offset_w.shape[0] > self._env_id:
                dx = float(pos_offset_w[self._env_id, 0].item())
                dy = float(pos_offset_w[self._env_id, 1].item())
                dz = float(pos_offset_w[self._env_id, 2].item())
            if isinstance(rpy_offset, torch.Tensor) and rpy_offset.ndim >= 2 and rpy_offset.shape[0] > self._env_id:
                dr = float(rpy_offset[self._env_id, 0].item())
                dp = float(rpy_offset[self._env_id, 1].item())
                dyaw = float(rpy_offset[self._env_id, 2].item())
        self._object_reset_status.content = (
            "Mode: `on`\n\n"
            "Frame: `world offset on reset`\n\n"
            f"Position: `dx={dx:+.2f}` `dy={dy:+.2f}` `dz={dz:+.2f}`\n\n"
            f"Rotation: `droll={dr:+.2f}` `dpitch={dp:+.2f}` `dyaw={dyaw:+.2f}`\n\n"
            "Applies on reset while enabled.\n\n"
            "Runtime size scaling is not supported yet; size still comes from the spawned URDF scale."
        )

    def _update_object_ground_status(self) -> None:
        if self._object_ground_status is None:
            return

        ground_top_z = 0.0
        env_id = int(self._env_id)
        mesh_source = str(self._mesh_source)
        mesh_mode = self._current_mesh_mode()
        sim_state = self._get_object_state_wxyz_for(env_id)
        ref_state = self._get_reference_object_state_wxyz_for(env_id)

        def _format_bottom(kind: str, state: tuple[np.ndarray, np.ndarray] | None) -> str:
            bottom_z = self._compute_object_bottom_z_from_state(env_id, state, kind)
            if bottom_z is None:
                return "n/a"
            gap = bottom_z - ground_top_z
            return f"`{bottom_z:+.4f}` (gap `{gap:+.4f}`)"

        self._object_ground_status.content = (
            f"Env: `{env_id}`\n\n"
            f"Mesh source: `{mesh_source}`  Mode: `{mesh_mode}`\n\n"
            f"Ground top z: `{ground_top_z:+.4f}`\n\n"
            f"Sim visual bottom: {_format_bottom('visual', sim_state)}\n\n"
            f"Ref visual bottom: {_format_bottom('visual', ref_state)}\n\n"
            f"Sim collision bottom: {_format_bottom('collision', sim_state)}\n\n"
            f"Ref collision bottom: {_format_bottom('collision', ref_state)}"
        )

    def _log_object_ground_status_once(self, motion_cmd) -> None:
        if self._object_ground_status_debug_logged or motion_cmd is None:
            return
        rows: list[str] = []
        for env_id in self._visible_env_ids():
            sim_state = self._get_object_state_wxyz_for(env_id)
            ref_state = self._get_reference_object_state_wxyz_for(env_id)
            sim_bottom = self._compute_object_bottom_z_from_state(env_id, sim_state, "collision")
            ref_bottom = self._compute_object_bottom_z_from_state(env_id, ref_state, "collision")
            try:
                clip_idx = int(motion_cmd.clip_ids[int(env_id)].item())
            except Exception:
                clip_idx = None
            clip_name = self._current_clip_name(motion_cmd, clip_idx)
            rows.append(
                "env{} clip={} sim_collision_bottom={} ref_collision_bottom={}".format(
                    int(env_id),
                    clip_name or "unknown",
                    "n/a" if sim_bottom is None else f"{float(sim_bottom):+.6f}",
                    "n/a" if ref_bottom is None else f"{float(ref_bottom):+.6f}",
                )
            )
        if rows:
            logger.info("Viser object ground status: {}", rows)
            self._object_ground_status_debug_logged = True

    def _update_manual_object_reset_override(self) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not hasattr(motion_cmd, "motion") or not bool(getattr(motion_cmd.motion, "has_object", False)):
            self._update_object_reset_status()
            return
        if (
            self._object_reset_override_cb is None
            and self._object_reset_pos_x_slider is None
            and self._object_reset_pos_y_slider is None
            and self._object_reset_pos_z_slider is None
            and self._object_reset_roll_slider is None
            and self._object_reset_pitch_slider is None
            and self._object_reset_yaw_slider is None
        ):
            self._update_object_reset_status()
            return

        enabled = bool(self._object_reset_override_cb.value) if self._object_reset_override_cb is not None else False
        motion_cmd.manual_object_reset_enabled = enabled

        device = self._env.device
        pos_offset = torch.tensor(
            [[
                float(self._object_reset_pos_x_slider.value) if self._object_reset_pos_x_slider is not None else 0.0,
                float(self._object_reset_pos_y_slider.value) if self._object_reset_pos_y_slider is not None else 0.0,
                float(self._object_reset_pos_z_slider.value) if self._object_reset_pos_z_slider is not None else 0.0,
            ]],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)
        rpy_offset = torch.tensor(
            [[
                float(self._object_reset_roll_slider.value) if self._object_reset_roll_slider is not None else 0.0,
                float(self._object_reset_pitch_slider.value) if self._object_reset_pitch_slider is not None else 0.0,
                float(self._object_reset_yaw_slider.value) if self._object_reset_yaw_slider is not None else 0.0,
            ]],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)

        motion_cmd.manual_object_reset_pos_offset_w = pos_offset
        motion_cmd.manual_object_reset_rpy_offset = rpy_offset
        self._update_object_reset_status()

    def _update_manual_root_status(
        self,
        *,
        current_root_pos: np.ndarray | None = None,
        current_root_quat_wxyz: np.ndarray | None = None,
    ) -> None:
        if self._manual_root_status is None:
            return

        motion_cmd = self._get_motion_command()
        manual_enabled = bool(getattr(motion_cmd, "manual_control_enabled", False)) if motion_cmd is not None else False
        if self._manual_control_cb is not None:
            manual_enabled = bool(manual_enabled or bool(self._manual_control_cb.value))
        hw_enabled = self._hw_joystick_mode_enabled()

        if manual_enabled and not hw_enabled:
            payload = self._manual_root_command_payload(
                current_pos=current_root_pos,
                current_quat_wxyz=current_root_quat_wxyz,
            )
            if payload is not None:
                manual_cmd = self._manual_policy_root_command()
                if manual_cmd is None:
                    cmd_xy = np.asarray(payload["cmd_xy"], dtype=np.float32)
                    manual_cmd = (float(cmd_xy[0]), float(cmd_xy[1]), float(payload["cmd_yaw"]))
                current_pos_np = np.asarray(payload["current_pos"], dtype=np.float32)
                target_pos_np = np.asarray(payload["target_pos"], dtype=np.float32)
                delta_world = np.asarray(payload["delta_world"], dtype=np.float32)
                self._manual_root_status.content = (
                    "Mode: `manual(gui)`\n\n"
                    "Input: `root-relative [dx, dy, dyaw]`\n\n"
                    f"Current root: `({current_pos_np[0]:+.2f}, {current_pos_np[1]:+.2f}, {current_pos_np[2]:+.2f})` yaw=`{float(payload['current_yaw']):+.2f}`\n\n"
                    f"Projected target: `({target_pos_np[0]:+.2f}, {target_pos_np[1]:+.2f}, {target_pos_np[2]:+.2f})` yaw=`{float(payload['target_yaw']):+.2f}`\n\n"
                    f"{self._format_policy_root_command(*manual_cmd)}\n\n"
                    f"Gap world: `dx={delta_world[0]:+.2f}` `dy={delta_world[1]:+.2f}` `dist={float(payload['dist_xy']):.2f}`"
                )
                return

        if manual_enabled and hw_enabled:
            manual_cmd = self._manual_policy_root_command()
            payload = self._manual_root_command_payload(
                current_pos=current_root_pos,
                current_quat_wxyz=current_root_quat_wxyz,
            )
            if payload is not None:
                command_line = (
                    self._format_policy_root_command(*manual_cmd)
                    if manual_cmd is not None
                    else "Policy cmd(root): unavailable"
                )
                current_pos_np = np.asarray(payload["current_pos"], dtype=np.float32)
                target_pos_np = np.asarray(payload["target_pos"], dtype=np.float32)
                delta_world = np.asarray(payload["delta_world"], dtype=np.float32)
                self._manual_root_status.content = (
                    "Mode: `manual(hw)`\n\n"
                    "Input: `root-relative [dx, dy, dyaw]`\n\n"
                    f"Current root: `({current_pos_np[0]:+.2f}, {current_pos_np[1]:+.2f}, {current_pos_np[2]:+.2f})` yaw=`{float(payload['current_yaw']):+.2f}`\n\n"
                    f"Projected target: `({target_pos_np[0]:+.2f}, {target_pos_np[1]:+.2f}, {target_pos_np[2]:+.2f})` yaw=`{float(payload['target_yaw']):+.2f}`\n\n"
                    f"{command_line}\n\n"
                    f"Gap world: `dx={delta_world[0]:+.2f}` `dy={delta_world[1]:+.2f}` `dist={float(payload['dist_xy']):.2f}`\n\n"
                    "Joystick writes a root-relative command in the robot heading frame."
                )
            else:
                command_line = (
                    self._format_policy_root_command(*manual_cmd)
                    if manual_cmd is not None
                    else "Policy cmd(root): unavailable"
                )
                self._manual_root_status.content = (
                    "Mode: `manual(hw)`\n\n"
                    "Input: `root-relative [dx, dy, dyaw]`\n\n"
                    f"{command_line}\n\n"
                    "Joystick writes a root-relative command in the robot heading frame."
                )
            return

        if motion_cmd is None:
            self._manual_root_status.content = "Mode: `idle`\n\nPolicy cmd(root): n/a"
            return

        try:
            current_pos = motion_cmd.robot_root_pos_w[self._env_id].detach().float().cpu().numpy()
            target_pos = motion_cmd.root_pos_w[self._env_id].detach().float().cpu().numpy()
            current_quat_xyzw = motion_cmd.robot_root_quat_w[self._env_id].detach().float().cpu().numpy()
            target_quat_xyzw = motion_cmd.root_quat_w[self._env_id].detach().float().cpu().numpy()
        except Exception:
            self._manual_root_status.content = "Mode: `motion`\n\nPolicy cmd(root): unavailable"
            return

        current_yaw = self._yaw_from_quat_wxyz(current_quat_xyzw[[3, 0, 1, 2]])
        target_yaw = self._yaw_from_quat_wxyz(target_quat_xyzw[[3, 0, 1, 2]])
        delta_world = np.asarray(target_pos - current_pos, dtype=np.float32)
        delta_world[2] = 0.0
        cy = float(np.cos(current_yaw))
        sy = float(np.sin(current_yaw))
        delta_body = np.array(
            [
                cy * float(delta_world[0]) + sy * float(delta_world[1]),
                -sy * float(delta_world[0]) + cy * float(delta_world[1]),
            ],
            dtype=np.float32,
        )
        yaw_gap = self._wrap_to_pi(target_yaw - current_yaw)
        dist_xy = float(np.linalg.norm(delta_world[:2]))
        self._manual_root_status.content = (
            "Mode: `motion`\n\n"
            f"Current: `({current_pos[0]:+.2f}, {current_pos[1]:+.2f})` yaw=`{current_yaw:+.2f}`\n\n"
            f"Target: `({target_pos[0]:+.2f}, {target_pos[1]:+.2f})` yaw=`{target_yaw:+.2f}`\n\n"
            f"{self._format_policy_root_command(float(delta_body[0]), float(delta_body[1]), yaw_gap)}\n\n"
            f"Gap world: `dx={delta_world[0]:+.2f}` `dy={delta_world[1]:+.2f}` `dist={dist_xy:.2f}`"
        )

    def _update_drop_button_status(self) -> None:
        if self._drop_button_status is None:
            return
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._drop_button_status.content = "drop_button: `n/a`"
            return
        manual_drop_button = getattr(motion_cmd, "manual_drop_button", None)
        if isinstance(manual_drop_button, torch.Tensor) and manual_drop_button.numel() > 0:
            try:
                value = float(manual_drop_button[self._env_id].reshape(-1)[0].item())
            except Exception:
                value = 0.0
        else:
            value = 1.0 if self._drop_button_cb is not None and bool(self._drop_button_cb.value) else 0.0
        override = bool(getattr(motion_cmd, "manual_drop_button_override_enabled", False))
        self._drop_button_status.content = f"drop_button: `{value:.0f}`\n\noverride: `{override}`"

    def _update_manual_drop_button(self) -> None:
        if not self._drop_button_gui_enabled:
            motion_cmd = self._get_motion_command()
            if motion_cmd is not None:
                motion_cmd.manual_drop_button_override_enabled = False
            return
        value = bool(self._drop_button_cb.value) if self._drop_button_cb is not None else bool(self._drop_button_default)
        self._set_manual_drop_button(value)
        self._update_drop_button_status()

    def _auto_forward_after_lift_snapshot(self) -> dict[str, Any] | None:
        root_pos, root_quat_wxyz = self._get_root_state_wxyz()
        object_state = self._get_object_state_wxyz()
        if root_pos is None or root_quat_wxyz is None or object_state is None:
            return None

        object_pos, object_quat_wxyz = object_state
        root_pos = np.asarray(root_pos, dtype=np.float32).reshape(3)
        root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4)
        object_pos = np.asarray(object_pos, dtype=np.float32).reshape(3)
        object_quat_wxyz = np.asarray(object_quat_wxyz, dtype=np.float32).reshape(4)
        rel_z = float(object_pos[2] - root_pos[2])

        robot_root_state = None
        root_lin_vel_w = None
        root_ang_vel_w = None
        try:
            root_states = getattr(self._env.simulator, "robot_root_states", None)
            if isinstance(root_states, torch.Tensor) and self._env_id < int(root_states.shape[0]):
                root_state = root_states[self._env_id].detach().float().cpu().numpy()
                robot_root_state = root_state.astype(float).tolist()
                if root_state.shape[0] >= 13:
                    root_lin_vel_w = root_state[7:10].astype(float).tolist()
                    root_ang_vel_w = root_state[10:13].astype(float).tolist()
        except Exception:
            pass

        manual_command = self._manual_policy_root_command()
        motion_cmd = self._get_motion_command()
        motion_timestep = None
        if motion_cmd is not None:
            raw_step = getattr(motion_cmd, "time_steps", None)
            if isinstance(raw_step, torch.Tensor) and raw_step.numel() > self._env_id:
                try:
                    motion_timestep = int(raw_step.reshape(-1)[self._env_id].item())
                except Exception:
                    motion_timestep = None

        wall_time = time.time()
        root_yaw = self._yaw_from_quat_wxyz(root_quat_wxyz)
        root_yaw_rate_est = None
        if (
            self._auto_forward_after_lift_last_snapshot_wall_time is not None
            and self._auto_forward_after_lift_last_snapshot_yaw is not None
            and wall_time > self._auto_forward_after_lift_last_snapshot_wall_time
        ):
            yaw_delta = self._wrap_to_pi(root_yaw - self._auto_forward_after_lift_last_snapshot_yaw)
            time_delta = wall_time - self._auto_forward_after_lift_last_snapshot_wall_time
            root_yaw_rate_est = float(yaw_delta / time_delta)
        self._auto_forward_after_lift_last_snapshot_wall_time = wall_time
        self._auto_forward_after_lift_last_snapshot_yaw = root_yaw

        return {
            "wall_time": wall_time,
            "env_id": int(self._env_id),
            "state": self._auto_forward_after_lift_state,
            "root_pos_w": root_pos.astype(float).tolist(),
            "root_yaw": root_yaw,
            "root_yaw_rate_est": root_yaw_rate_est,
            "object_pos_w": object_pos.astype(float).tolist(),
            "object_rel_z": rel_z,
            "object_rel_z_delta": (
                None
                if self._auto_forward_after_lift_baseline_rel_z is None
                else float(rel_z - self._auto_forward_after_lift_baseline_rel_z)
            ),
            "root_lin_vel_w": root_lin_vel_w,
            "root_ang_vel_w": root_ang_vel_w,
            "robot_root_state": robot_root_state,
            "manual_command": None if manual_command is None else [float(v) for v in manual_command],
            "configured_command": [float(v) for v in self._auto_forward_after_lift_command],
            "applied_command": (
                None
                if self._auto_forward_after_lift_applied_command is None
                else [float(v) for v in self._auto_forward_after_lift_applied_command]
            ),
            "effective_command": (
                None
                if self._auto_forward_after_lift_effective_command is None
                else [float(v) for v in self._auto_forward_after_lift_effective_command]
            ),
            "heading_hold_enabled": bool(self._auto_forward_after_lift_heading_hold_enabled),
            "heading_hold_target_yaw": (
                None
                if self._auto_forward_after_lift_heading_target_yaw is None
                else float(self._auto_forward_after_lift_heading_target_yaw)
            ),
            "heading_hold_error": (
                None
                if self._auto_forward_after_lift_heading_error is None
                else float(self._auto_forward_after_lift_heading_error)
            ),
            "heading_hold_xy_compensation": bool(
                self._auto_forward_after_lift_heading_xy_compensation_enabled
            ),
            "motion_timestep": motion_timestep,
        }

    def _write_auto_forward_after_lift_log(self, snapshot: dict[str, Any], *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._auto_forward_after_lift_last_log_time < self._auto_forward_after_lift_log_period_s:
            return
        self._auto_forward_after_lift_last_log_time = now
        try:
            if self._auto_forward_after_lift_log_handle is None:
                self._auto_forward_after_lift_log_path.parent.mkdir(parents=True, exist_ok=True)
                self._auto_forward_after_lift_log_handle = self._auto_forward_after_lift_log_path.open(
                    "a",
                    encoding="utf-8",
                    buffering=1,
                )
            self._auto_forward_after_lift_log_handle.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
        except Exception as exc:
            if not self._auto_forward_after_lift_logged_start:
                logger.warning("Failed to write auto-forward-after-lift log: {}", exc)

    def _apply_auto_forward_after_lift_command(self, command: list[float]) -> bool:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return False
        if self._manual_control_cb is not None:
            try:
                self._manual_control_cb.value = True
            except Exception:
                pass
        motion_cmd.manual_control_enabled = True
        self._set_manual_root_command((command[0], command[1]), command[2], sync_gui=True)

        device = self._env.device
        cmd_xy = torch.tensor([[float(command[0]), float(command[1])]], device=device, dtype=torch.float32).repeat(
            self._env.num_envs,
            1,
        )
        cmd_yaw = torch.tensor([[float(command[2])]], device=device, dtype=torch.float32).repeat(
            self._env.num_envs,
            1,
        )
        motion_cmd.manual_xy_rel = cmd_xy
        motion_cmd.manual_yaw_rel = cmd_yaw
        self._auto_forward_after_lift_applied_command = [float(v) for v in command]
        self._update_manual_root_status()
        return True

    def _auto_forward_after_lift_command_for_snapshot(self, snapshot: dict[str, Any]) -> list[float]:
        command = [float(v) for v in self._auto_forward_after_lift_command]
        if not self._auto_forward_after_lift_heading_hold_enabled:
            self._auto_forward_after_lift_effective_command = list(command)
            self._auto_forward_after_lift_heading_error = None
            return command

        root_yaw_raw = snapshot.get("root_yaw")
        if root_yaw_raw is None:
            self._auto_forward_after_lift_effective_command = list(command)
            self._auto_forward_after_lift_heading_error = None
            return command

        root_yaw = float(root_yaw_raw)
        if self._auto_forward_after_lift_heading_target_yaw is None:
            self._auto_forward_after_lift_heading_target_yaw = root_yaw

        heading_error = self._wrap_to_pi(float(self._auto_forward_after_lift_heading_target_yaw) - root_yaw)
        yaw_control_error = heading_error
        if abs(yaw_control_error) < float(self._auto_forward_after_lift_heading_deadband):
            yaw_control_error = 0.0

        yaw_cmd = float(self._auto_forward_after_lift_heading_kp) * yaw_control_error
        yaw_rate_raw = snapshot.get("root_yaw_rate_est")
        if yaw_rate_raw is not None:
            try:
                yaw_cmd -= float(self._auto_forward_after_lift_heading_kd) * float(yaw_rate_raw)
            except Exception:
                pass
        yaw_limit = float(self._auto_forward_after_lift_heading_yaw_limit)
        if yaw_limit > 0.0:
            yaw_cmd = float(np.clip(yaw_cmd, -yaw_limit, yaw_limit))

        if self._auto_forward_after_lift_heading_xy_compensation_enabled:
            c = float(np.cos(heading_error))
            s = float(np.sin(heading_error))
            cmd_x = c * command[0] - s * command[1]
            cmd_y = s * command[0] + c * command[1]
        else:
            cmd_x = command[0]
            cmd_y = command[1]
        effective_command = [float(cmd_x), float(cmd_y), float(yaw_cmd)]
        self._auto_forward_after_lift_heading_error = float(heading_error)
        self._auto_forward_after_lift_effective_command = list(effective_command)
        return effective_command

    def _annotate_auto_forward_after_lift_command(
        self,
        snapshot: dict[str, Any],
        command: list[float],
    ) -> None:
        snapshot["effective_command"] = [float(v) for v in command]
        snapshot["heading_hold_enabled"] = bool(self._auto_forward_after_lift_heading_hold_enabled)
        snapshot["heading_hold_target_yaw"] = (
            None
            if self._auto_forward_after_lift_heading_target_yaw is None
            else float(self._auto_forward_after_lift_heading_target_yaw)
        )
        snapshot["heading_hold_error"] = (
            None
            if self._auto_forward_after_lift_heading_error is None
            else float(self._auto_forward_after_lift_heading_error)
        )
        snapshot["heading_hold_xy_compensation"] = bool(
            self._auto_forward_after_lift_heading_xy_compensation_enabled
        )

    def _update_auto_forward_after_lift(self) -> None:
        if not self._auto_forward_after_lift_enabled:
            return

        snapshot = self._auto_forward_after_lift_snapshot()
        if snapshot is None:
            return

        if not self._auto_forward_after_lift_logged_start:
            logger.info(
                "Auto-forward-after-lift enabled: command={} rel_z_delta_threshold={:.3f} consecutive_steps={} "
                "duration_s={:.2f} heading_hold={} kp={:.3f} kd={:.3f} yaw_limit={:.3f} log_path={}",
                self._auto_forward_after_lift_command,
                self._auto_forward_after_lift_rel_z_delta,
                self._auto_forward_after_lift_consecutive_steps,
                self._auto_forward_after_lift_duration_s,
                self._auto_forward_after_lift_heading_hold_enabled,
                self._auto_forward_after_lift_heading_kp,
                self._auto_forward_after_lift_heading_kd,
                self._auto_forward_after_lift_heading_yaw_limit,
                self._auto_forward_after_lift_log_path,
            )
            self._auto_forward_after_lift_logged_start = True

        rel_z = float(snapshot["object_rel_z"])
        if self._auto_forward_after_lift_baseline_rel_z is None:
            self._auto_forward_after_lift_baseline_rel_z = rel_z
            snapshot["event"] = "baseline"
            snapshot["object_rel_z_delta"] = 0.0
            self._write_auto_forward_after_lift_log(snapshot, force=True)
            return

        rel_z_delta = float(rel_z - self._auto_forward_after_lift_baseline_rel_z)
        snapshot["object_rel_z_delta"] = rel_z_delta

        if self._auto_forward_after_lift_state == "waiting":
            if rel_z_delta >= self._auto_forward_after_lift_rel_z_delta:
                self._auto_forward_after_lift_ready_count += 1
            else:
                self._auto_forward_after_lift_ready_count = 0

            if self._auto_forward_after_lift_ready_count >= self._auto_forward_after_lift_consecutive_steps:
                self._auto_forward_after_lift_state = "active"
                self._auto_forward_after_lift_trigger_time = time.time()
                command = self._auto_forward_after_lift_command_for_snapshot(snapshot)
                command_applied = self._apply_auto_forward_after_lift_command(command)
                snapshot["state"] = self._auto_forward_after_lift_state
                snapshot["event"] = "trigger"
                snapshot["command_applied"] = command_applied
                snapshot["applied_command"] = [float(v) for v in command]
                self._annotate_auto_forward_after_lift_command(snapshot, command)
                logger.info(
                    "Auto-forward-after-lift triggered: rel_z_delta={:.3f}, command={} configured={}",
                    rel_z_delta,
                    command,
                    self._auto_forward_after_lift_command,
                )
                self._write_auto_forward_after_lift_log(snapshot, force=True)
                return

        elif self._auto_forward_after_lift_state == "active":
            command = self._auto_forward_after_lift_command_for_snapshot(snapshot)
            command_applied = self._apply_auto_forward_after_lift_command(command)
            snapshot["command_applied"] = command_applied
            snapshot["applied_command"] = [float(v) for v in command]
            self._annotate_auto_forward_after_lift_command(snapshot, command)
            if (
                self._auto_forward_after_lift_duration_s > 0.0
                and self._auto_forward_after_lift_trigger_time is not None
                and time.time() - self._auto_forward_after_lift_trigger_time >= self._auto_forward_after_lift_duration_s
            ):
                self._auto_forward_after_lift_state = "done"
                command_applied = self._apply_auto_forward_after_lift_command([0.0, 0.0, 0.0])
                snapshot["state"] = self._auto_forward_after_lift_state
                snapshot["event"] = "duration_complete"
                snapshot["command_applied"] = command_applied
                snapshot["applied_command"] = [0.0, 0.0, 0.0]
                self._auto_forward_after_lift_effective_command = [0.0, 0.0, 0.0]
                snapshot["effective_command"] = [0.0, 0.0, 0.0]
                logger.info("Auto-forward-after-lift duration complete; zeroed manual root command.")
                self._write_auto_forward_after_lift_log(snapshot, force=True)
                return

        snapshot["state"] = self._auto_forward_after_lift_state
        self._write_auto_forward_after_lift_log(snapshot)

    def _update_manual_root_command(self) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return
        gui_enabled = bool(self._manual_control_cb.value) if self._manual_control_cb is not None else False
        hw_enabled = self._hw_joystick_mode_enabled()
        enabled = bool(self._manual_force_enabled or gui_enabled or hw_enabled)
        motion_cmd.manual_control_enabled = enabled
        if not enabled:
            self._clear_manual_commands(clear_gui_toggles=False)
            # If clip-lock is off, release any manual forced-clip override.
            lock_enabled = self._clip_lock_enabled()
            if not lock_enabled:
                try:
                    motion_cmd.set_forced_clip(None)
                except Exception:
                    motion_cmd._forced_clip_idx = None
                try:
                    motion_cmd.set_forced_clip_start(None)
                except Exception:
                    motion_cmd._forced_start_step = None
            return

        device = self._env.device
        lin_scale = float(self._manual_lin_scale_slider.value) if self._manual_lin_scale_slider is not None else 0.5
        yaw_scale = float(self._manual_yaw_scale_slider.value) if self._manual_yaw_scale_slider is not None else 0.3
        if hw_enabled:
            lx, ly, rx = self._hw_joystick_axes()
            cmd_x = self._apply_deadzone(ly) * lin_scale
            cmd_y = self._apply_deadzone(lx) * lin_scale
            cmd_yaw_val = self._apply_deadzone(rx) * yaw_scale
        else:
            if self._sync_manual_root_command_from_gui():
                cmd_x = float(self._manual_root_cmd_xy[0])
                cmd_y = float(self._manual_root_cmd_xy[1])
                cmd_yaw_val = float(self._manual_root_cmd_yaw)
            else:
                forward = 1.0 if self._manual_forward_cb is not None and bool(self._manual_forward_cb.value) else 0.0
                back = 1.0 if self._manual_back_cb is not None and bool(self._manual_back_cb.value) else 0.0
                left = 1.0 if self._manual_left_cb is not None and bool(self._manual_left_cb.value) else 0.0
                right = 1.0 if self._manual_right_cb is not None and bool(self._manual_right_cb.value) else 0.0
                yaw_left = 1.0 if self._manual_yaw_left_cb is not None and bool(self._manual_yaw_left_cb.value) else 0.0
                yaw_right = 1.0 if self._manual_yaw_right_cb is not None and bool(self._manual_yaw_right_cb.value) else 0.0
                cmd_x = (forward - back) * lin_scale
                cmd_y = (left - right) * lin_scale
                cmd_yaw_val = (yaw_left - yaw_right) * yaw_scale
        cmd_xy = torch.tensor(
            [[cmd_x, cmd_y]],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)
        cmd_yaw = torch.tensor(
            [[cmd_yaw_val]],
            device=device,
            dtype=torch.float32,
        ).repeat(self._env.num_envs, 1)
        motion_cmd.manual_xy_rel = cmd_xy
        motion_cmd.manual_yaw_rel = cmd_yaw
        self._update_manual_root_status()

    def _hide_manual_command_arrow(self) -> None:
        if self._manual_command_arrow_handle is None:
            return
        try:
            self._manual_command_arrow_handle.visible = False
        except Exception:
            pass

    @staticmethod
    def _yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
        qw, qx, qy, qz = [float(v) for v in quat_wxyz]
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def _update_manual_command_arrow(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        if self._server is None:
            return
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._hide_manual_command_arrow()
            return
        if not bool(getattr(motion_cmd, "manual_control_enabled", False)):
            self._hide_manual_command_arrow()
            return

        manual_xy = getattr(motion_cmd, "manual_xy_rel", None)
        manual_yaw = getattr(motion_cmd, "manual_yaw_rel", None)
        if not isinstance(manual_xy, torch.Tensor) or manual_xy.ndim < 2:
            self._hide_manual_command_arrow()
            return
        if self._env_id < 0 or self._env_id >= int(manual_xy.shape[0]):
            self._hide_manual_command_arrow()
            return

        cmd_xy = manual_xy[self._env_id].detach().float().cpu().numpy()
        cmd_yaw = 0.0
        if isinstance(manual_yaw, torch.Tensor) and manual_yaw.ndim >= 2 and self._env_id < int(manual_yaw.shape[0]):
            cmd_yaw = float(manual_yaw[self._env_id, 0].item())

        cmd_xy_norm = float(np.linalg.norm(cmd_xy))
        cmd_yaw_abs = abs(cmd_yaw)
        if cmd_xy_norm < 1.0e-4 and cmd_yaw_abs < 1.0e-4:
            self._hide_manual_command_arrow()
            return

        segments: list[np.ndarray] = []

        if cmd_xy_norm >= 1.0e-4:
            body_cmd = np.array([float(cmd_xy[0]), float(cmd_xy[1]), 0.0], dtype=np.float32)
            cmd_len = float(np.linalg.norm(body_cmd[:2]))
            if cmd_len > float(self._manual_command_arrow_max_len):
                body_cmd *= float(self._manual_command_arrow_max_len / max(cmd_len, 1.0e-6))
                cmd_len = float(np.linalg.norm(body_cmd[:2]))
            if cmd_len >= 1.0e-6:
                yaw = self._yaw_from_quat_wxyz(root_quat_wxyz)
                cy = float(np.cos(yaw))
                sy = float(np.sin(yaw))
                world_cmd = np.array(
                    [
                        cy * float(body_cmd[0]) - sy * float(body_cmd[1]),
                        sy * float(body_cmd[0]) + cy * float(body_cmd[1]),
                        0.0,
                    ],
                    dtype=np.float32,
                )

                origin = root_pos.astype(np.float32, copy=True)
                origin[2] += float(self._manual_command_arrow_height)
                tip = origin + world_cmd
                direction = tip - origin
                length = float(np.linalg.norm(direction))
                if length >= 1.0e-6:
                    direction /= length
                    perp = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
                    head_len = min(
                        float(length * self._manual_command_arrow_head_ratio),
                        0.5 * length,
                    )
                    head_width = float(head_len * self._manual_command_arrow_head_width_ratio)
                    left = tip - direction * head_len + perp * head_width
                    right = tip - direction * head_len - perp * head_width
                    segments.append(np.stack([origin, tip], axis=0))
                    segments.append(np.stack([tip, left], axis=0))
                    segments.append(np.stack([tip, right], axis=0))
        if cmd_yaw_abs >= 1.0e-4:
            yaw = self._yaw_from_quat_wxyz(root_quat_wxyz)
            sign = 1.0 if cmd_yaw >= 0.0 else -1.0
            radius = float(self._manual_command_yaw_radius)
            arc_angle = float(self._manual_command_yaw_arc_angle)
            seg_count = max(8, int(self._manual_command_yaw_segments))
            start = yaw - sign * (0.5 * arc_angle)
            end = yaw + sign * (0.5 * arc_angle)
            angles = np.linspace(start, end, seg_count + 1, dtype=np.float32)

            center = root_pos.astype(np.float32, copy=True)
            center[2] += float(self._manual_command_arrow_height)
            arc_pts = np.stack(
                [
                    center[0] + radius * np.cos(angles),
                    center[1] + radius * np.sin(angles),
                    np.full_like(angles, center[2]),
                ],
                axis=1,
            )
            for idx in range(seg_count):
                segments.append(np.stack([arc_pts[idx], arc_pts[idx + 1]], axis=0))

            tip = arc_pts[-1]
            tangent = sign * np.array([-np.sin(angles[-1]), np.cos(angles[-1]), 0.0], dtype=np.float32)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1.0e-6:
                tangent /= tangent_norm
                radial = np.array([np.cos(angles[-1]), np.sin(angles[-1]), 0.0], dtype=np.float32)
                h_len = float(self._manual_command_yaw_head_len)
                h_width = float(self._manual_command_yaw_head_width)
                head_base = tip - tangent * h_len
                left = head_base + radial * h_width
                right = head_base - radial * h_width
                segments.append(np.stack([tip, left], axis=0))
                segments.append(np.stack([tip, right], axis=0))

        if len(segments) == 0:
            self._hide_manual_command_arrow()
            return

        points = np.stack(segments, axis=0)
        if self._recenter:
            points = points - offset[None, None, :]
        points = points.astype(np.float32, copy=False)
        colors = np.full((points.shape[0], 2, 3), COMMAND_ARROW_COLOR, dtype=np.uint8)

        if self._manual_command_arrow_handle is None:
            self._manual_command_arrow_handle = self._server.scene.add_line_segments(
                self._scene_path("/manual_command_arrow"),
                points=points,
                colors=colors,
                line_width=float(self._manual_command_arrow_line_width),
            )
        else:
            self._manual_command_arrow_handle.visible = True
            self._manual_command_arrow_handle.points = points
            self._manual_command_arrow_handle.colors = colors

    def on_reset(self, env_ids) -> None:
        if not self._enabled:
            return
        if getattr(self, "_env_id", 0) not in _normalize_env_ids(env_ids):
            return
        self._clear_rollout_root_trajectory()
        self._clear_rollout_object_trajectory()
        self._invalidate_object_mesh_caches(remove_handles=True)
        self._object_ground_status_debug_logged = False
        if self._server is None:
            return
        if not getattr(self, "_recenter", True):
            return
        self._offset = self._resolve_env_origin()
        self._reload_terrain_for_clip(self._current_clip_name(self._get_motion_command()))

    def record_step(self) -> None:
        if self._enabled and self._server is None:
            self._initialize_server()
        if not self._enabled or self._server is None or self._robot_root is None:
            return
        if not self._record_step_debug_logged:
            logger.info(
                "Viser record_step active: sim_meshes_enabled={} load_urdf_visuals={} body_count={}",
                self._sim_meshes_enabled,
                self._load_urdf_visuals,
                len(self._body_name_to_index),
            )
            self._record_step_debug_logged = True

        if self._force_dt:
            now = time.perf_counter()
            if now < self._next_tick:
                time.sleep(self._next_tick - now)
            now = time.perf_counter()
            self._next_tick = now + self._update_period
        else:
            now = time.perf_counter()
        if self._update_period > 0 and (now - self._last_update) < self._update_period:
            return
        self._last_update = now

        root_pos, root_quat_wxyz = self._get_root_state_wxyz()
        if root_pos is None or root_quat_wxyz is None:
            return

        dof_pos = self._get_dof_pos() if self._vr is not None else None
        if self._vr is not None and dof_pos is None:
            return

        if self._offset is None:
            self._offset = self._resolve_env_origin() if self._recenter else np.zeros(3, dtype=np.float32)
            if self._offset is None:
                self._offset = root_pos.copy()
        offset = self._offset if self._recenter else np.zeros(3, dtype=np.float32)
        self._update_terrain_transform(offset)
        show_robot = self._show_robot_cb is None or bool(self._show_robot_cb.value)
        show_object = self._show_object_cb is None or bool(self._show_object_cb.value)
        motion_cmd = self._get_motion_command()

        with self._server.atomic():
            self._robot_root.position = root_pos - offset
            self._robot_root.wxyz = root_quat_wxyz
            self._update_env_sequence_label(
                self._env_id,
                self._current_clip_name(motion_cmd, self._active_clip_index(motion_cmd)),
            )

            if self._vr is not None and dof_pos is not None:
                joints = dof_pos
                if self._joint_order is not None:
                    joints = joints[self._joint_order]
                if joints.shape[0] != self._joint_count:
                    return
                self._vr.update_cfg(joints.astype(np.float32, copy=False))
                try:
                    self._vr.show_visual = bool(show_robot)
                except Exception:
                    pass
            else:
                robot_mesh_updated = self._update_robot_mesh_frames_for_env(
                    self._env_id,
                    offset=offset,
                    visible=bool(show_robot),
                )
                if not robot_mesh_updated:
                    self._update_robot_points(env_id=self._env_id, offset=offset, visible=bool(show_robot))
                elif self._robot_points_handle is not None:
                    try:
                        self._robot_points_handle.visible = False
                    except Exception:
                        pass

            if self._scandots_enabled:
                self._update_scandots(offset)
            self._update_target_keypoints(offset)
            self._update_target_object_trajectory(offset)
            self._update_rollout_root_trajectory(root_pos, offset)
            self._update_rollout_object_trajectory(offset)
            self._update_target_object_center(offset)
            self._update_target_box(offset)
            self._update_future_goal_box(offset)
            self._update_manual_command_arrow(root_pos, root_quat_wxyz, offset)
            self._update_manual_root_status(
                current_root_pos=root_pos,
                current_root_quat_wxyz=root_quat_wxyz,
            )
            if self._perception_enabled:
                self._update_perception_visuals(offset)
            if not self._disable_contact_force_viz:
                self._update_contact_forces(offset)

            if self._load_urdf_visuals or self._sim_meshes_enabled:
                self._refresh_primary_object_handle()
            obj_state = self._get_object_state_wxyz()
            if obj_state is not None and self._object_root is not None:
                obj_pos, obj_quat_wxyz = obj_state
                self._object_root.position = obj_pos - offset
                self._object_root.wxyz = obj_quat_wxyz
            else:
                obj_pos = None
            if self._sim_meshes_enabled:
                if not show_object or obj_state is None:
                    _set_visual_handle_visible(self._vo, False)
                    _set_visual_handle_visible(self._vo_collision, False)
                else:
                    self._apply_primary_object_mesh_visibility()
                if self._object_points_handle is not None:
                    try:
                        self._object_points_handle.visible = False
                    except Exception:
                        pass
            elif self._vo is not None and self._object_root is not None:
                if not show_object or obj_state is None:
                    _set_visual_handle_visible(self._vo, False)
                else:
                    _set_visual_handle_visible(self._vo, True)
            if self._vo is None and self._vo_collision is None:
                self._update_object_point(
                    env_id=self._env_id,
                    object_pos=obj_pos,
                    offset=offset,
                    visible=bool(show_object and obj_pos is not None),
                )
            self._update_object_ground_status()
            self._log_object_ground_status_once(motion_cmd)

            for env_id in self._secondary_env_ids:
                robot_root = self._secondary_robot_roots.get(env_id)
                vr = self._secondary_vr.get(env_id)
                if robot_root is None:
                    continue
                secondary_root, secondary_quat = self._get_root_state_wxyz_for(env_id)
                if secondary_root is None or secondary_quat is None:
                    continue
                secondary_dof = self._get_dof_pos_for(env_id) if vr is not None else None
                if vr is not None and secondary_dof is None:
                    continue
                secondary_offset = (
                    self._resolve_env_origin_for(env_id) if self._recenter else np.zeros(3, dtype=np.float32)
                )
                if secondary_offset is None:
                    secondary_offset = np.zeros(3, dtype=np.float32)
                display_shift = self._viewer_env_shift(env_id)
                robot_root.position = secondary_root - secondary_offset + display_shift
                robot_root.wxyz = secondary_quat
                clip_idx = None
                if motion_cmd is not None and hasattr(motion_cmd, "clip_ids"):
                    try:
                        clip_idx = int(motion_cmd.clip_ids[env_id].item())
                    except Exception:
                        clip_idx = None
                self._update_env_sequence_label(env_id, self._current_clip_name(motion_cmd, clip_idx))

                if vr is not None and secondary_dof is not None:
                    secondary_joints = secondary_dof
                    if self._joint_order is not None:
                        secondary_joints = secondary_joints[self._joint_order]
                    if secondary_joints.shape[0] == self._joint_count:
                        vr.update_cfg(secondary_joints.astype(np.float32, copy=False))
                    try:
                        vr.show_visual = bool(show_robot)
                    except Exception:
                        pass
                elif vr is None:
                    robot_mesh_updated = self._update_robot_mesh_frames_for_env(
                        env_id,
                        offset=secondary_offset,
                        display_shift=display_shift,
                        visible=bool(show_robot),
                    )
                    if not robot_mesh_updated:
                        self._update_robot_points(
                            env_id=env_id,
                            offset=secondary_offset,
                            display_shift=display_shift,
                            visible=bool(show_robot),
                        )
                    elif env_id in self._secondary_robot_points_handles:
                        try:
                            self._secondary_robot_points_handles[env_id].visible = False
                        except Exception:
                            pass

                if self._load_urdf_visuals or self._sim_meshes_enabled:
                    self._ensure_secondary_object_handle(env_id)
                secondary_vo = self._secondary_vo.get(env_id)
                secondary_vo_collision = self._secondary_vo_collision.get(env_id)
                secondary_object_root = self._secondary_object_roots.get(env_id)
                if (secondary_vo is None and secondary_vo_collision is None) or secondary_object_root is None:
                    secondary_obj_state = self._get_object_state_wxyz_for(env_id)
                    if secondary_obj_state is not None and secondary_object_root is not None:
                        secondary_obj_pos, secondary_obj_quat = secondary_obj_state
                        secondary_object_root.position = secondary_obj_pos - secondary_offset + display_shift
                        secondary_object_root.wxyz = secondary_obj_quat
                        self._update_object_point(
                            env_id=env_id,
                            object_pos=secondary_obj_pos,
                            offset=secondary_offset,
                            display_shift=display_shift,
                            visible=bool(show_object),
                        )
                    continue
                if not show_object:
                    if self._sim_meshes_enabled:
                        self._apply_secondary_object_mesh_visibility(env_id, visible=False)
                    else:
                        _set_visual_handle_visible(secondary_vo, False)
                    self._update_object_point(
                        env_id=env_id,
                        object_pos=None,
                        offset=secondary_offset,
                        display_shift=display_shift,
                        visible=False,
                    )
                    continue
                secondary_obj_state = self._get_object_state_wxyz_for(env_id)
                if secondary_obj_state is None:
                    if self._sim_meshes_enabled:
                        self._apply_secondary_object_mesh_visibility(env_id, visible=False)
                    else:
                        _set_visual_handle_visible(secondary_vo, False)
                    self._update_object_point(
                        env_id=env_id,
                        object_pos=None,
                        offset=secondary_offset,
                        display_shift=display_shift,
                        visible=False,
                    )
                    continue
                secondary_obj_pos, secondary_obj_quat = secondary_obj_state
                secondary_object_root.position = secondary_obj_pos - secondary_offset + display_shift
                secondary_object_root.wxyz = secondary_obj_quat
                if self._sim_meshes_enabled:
                    self._apply_secondary_object_mesh_visibility(env_id, visible=True)
                    if env_id in self._secondary_object_points_handles:
                        try:
                            self._secondary_object_points_handles[env_id].visible = False
                        except Exception:
                            pass
                else:
                    _set_visual_handle_visible(secondary_vo, True)

        if self._clip_label is not None:
            if motion_cmd is not None and hasattr(motion_cmd, "clip_ids"):
                try:
                    clip_idx = int(motion_cmd.clip_ids[self._env_id].item())
                    clip_names = getattr(motion_cmd.motion, "clip_ids", [])
                    clip_name = clip_names[clip_idx] if 0 <= clip_idx < len(clip_names) else str(clip_idx)
                    self._clip_label.content = f"Current clip: `{clip_name}`"
                    self._reload_terrain_for_clip(str(clip_name))
                except Exception:
                    pass
        self._update_clip_group_ui(motion_cmd)
        self._log_env_sequence_assignments_once(motion_cmd)

    def _setup_joint_order(self) -> None:
        if self._vr is None:
            return
        viser_joint_names = list(self._vr.get_actuated_joint_names())
        self._joint_count = len(viser_joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(self._env.robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_idx]
        if missing:
            if len(viser_joint_names) == len(self._env.robot_config.dof_names):
                logger.warning("Viser joint names mismatch; using robot DOF order.")
                self._joint_order = None
            else:
                logger.warning("Viser joints missing in robot config: {}", missing)
                self._enabled = False
        else:
            self._joint_order = np.array([name_to_idx[name] for name in viser_joint_names], dtype=np.int64)

    def _load_terrain(self, clip_name: str | None = None) -> None:
        if self._server is None:
            return
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is None:
            return

        motion_cmd = self._get_motion_command()
        if clip_name is None:
            clip_name = self._current_clip_name(motion_cmd)

        terrain_state = terrain_mgr.get_state("locomotion_terrain")
        terrain_cfg = getattr(terrain_mgr, "cfg", None)
        terrain_term = getattr(terrain_cfg, "terrain_term", None) if terrain_cfg is not None else None
        show_ground_plane = bool(
            terrain_term is not None
            and getattr(terrain_state, "mesh_type", None) == "load_obj"
            and bool(getattr(terrain_term, "add_ground_plane_collision", False))
        )

        mesh = None
        mesh_is_local = False
        if terrain_term is not None:
            obj_path = getattr(terrain_term, "obj_file_path", None) or ""
            obj_meta = getattr(terrain_term, "obj_metadata_path", None)
            rows = getattr(terrain_term, "num_rows", None)
            cols = getattr(terrain_term, "num_cols", None)
            if obj_path:
                terrain_mesh = _load_terrain_mesh(
                    obj_path,
                    obj_metadata_path=obj_meta,
                    num_rows=rows,
                    num_cols=cols,
                    clip_name=clip_name,
                )
                if terrain_mesh is not None:
                    mesh, mesh_is_local = terrain_mesh

        if mesh is None:
            mesh = getattr(terrain_state, "mesh", None)
            if mesh is None:
                terrain = getattr(terrain_state, "terrain", None)
                mesh = getattr(terrain, "mesh", None) if terrain is not None else None
            if mesh is not None:
                try:
                    import trimesh  # type: ignore[import-not-found]
                except Exception:
                    mesh = None
                else:
                    if isinstance(mesh, trimesh.Scene):
                        meshes = mesh.dump(concatenate=False)
                        if meshes:
                            mesh = max(
                                meshes,
                                key=lambda m: len(getattr(m, "faces", [])) or len(m.vertices),
                            )
                    if not isinstance(mesh, trimesh.Trimesh):
                        try:
                            mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
                        except Exception:
                            mesh = None
                    if mesh is not None:
                        mesh_is_local = False

        if mesh is None:
            try:
                import trimesh  # type: ignore[import-not-found]
            except Exception:
                return
            ground_mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.01))
            ground_mesh.apply_translation([0.0, 0.0, -0.005])
            self._ground_handle = self._server.scene.add_mesh_simple(
                self._scene_path("/ground"),
                ground_mesh.vertices,
                ground_mesh.faces,
                color=GROUND_DARK_GRAY,
                side="double",
            )
            if self._show_terrain_cb is not None:
                self._ground_handle.visible = bool(self._show_terrain_cb.value)
            self._terrain_is_local = False
            self._terrain_clip_name = clip_name
            self._update_terrain_transform()
            return

        self._terrain_handle = self._server.scene.add_mesh_simple(
            self._scene_path("/terrain"),
            mesh.vertices,
            mesh.faces,
            color=TERRAIN_GRAY,
            side="double",
        )
        if self._show_terrain_cb is not None:
            self._terrain_handle.visible = bool(self._show_terrain_cb.value)
        if show_ground_plane:
            try:
                import trimesh  # type: ignore[import-not-found]
            except Exception:
                pass
            else:
                ground_mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.01))
                ground_mesh.apply_translation([0.0, 0.0, -0.005])
                self._ground_handle = self._server.scene.add_mesh_simple(
                    self._scene_path("/ground"),
                    ground_mesh.vertices,
                    ground_mesh.faces,
                    color=GROUND_DARK_GRAY,
                    side="double",
                )
                if self._show_terrain_cb is not None:
                    self._ground_handle.visible = bool(self._show_terrain_cb.value)
        self._terrain_is_local = bool(mesh_is_local)
        self._terrain_clip_name = clip_name
        self._update_terrain_transform()

    def _get_perception_manager(self):
        mgr = getattr(self._env, "perception_manager", None)
        if mgr is None or not getattr(mgr, "enabled", False):
            return None
        return mgr

    def _resolve_perception_shape(self, perception_mgr) -> tuple[int, int] | None:
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return None
        output_mode = getattr(cfg, "output_mode", None)
        if output_mode == "camera_depth":
            depth_source = self._get_perception_depth_source()
            if depth_source == "obs":
                width = int(getattr(perception_mgr, "_camera_obs_width", 0) or 0)
                height = int(getattr(perception_mgr, "_camera_obs_height", 0) or 0)
                if width > 0 and height > 0:
                    return height, width
                resize_cfg = getattr(cfg, "camera_warp_resize", None)
                if resize_cfg is not None:
                    try:
                        resize_h, resize_w = resize_cfg
                        resize_h = int(resize_h)
                        resize_w = int(resize_w)
                        if resize_h > 0 and resize_w > 0:
                            return resize_h, resize_w
                    except Exception:
                        pass
            width = int(getattr(perception_mgr, "_camera_width", 0) or 0)
            height = int(getattr(perception_mgr, "_camera_height", 0) or 0)
            if width > 0 and height > 0:
                return height, width
            width = int(getattr(cfg, "camera_width", 0) or 0)
            height = int(getattr(cfg, "camera_height", 0) or 0)
            if width > 0 and height > 0:
                return height, width
            grid = int(getattr(cfg, "grid_size", 0) or 0)
            if grid > 0:
                return grid, grid
            return None
        if output_mode == "heightmap":
            heightmap = getattr(perception_mgr, "_heightmap", None)
            if isinstance(heightmap, torch.Tensor) and heightmap.ndim >= 3:
                return int(heightmap.shape[-2]), int(heightmap.shape[-1])
            grid_x = int(getattr(perception_mgr, "_heightmap_grid_x", 0) or 0)
            grid_y = int(getattr(perception_mgr, "_heightmap_grid_y", 0) or 0)
            if grid_x > 0 and grid_y > 0:
                return grid_x, grid_y
        return None

    def _get_perception_depth_source(self) -> str:
        if self._perception_depth_source_dropdown is not None:
            try:
                value = str(self._perception_depth_source_dropdown.value).strip().lower()
            except Exception:
                value = self._perception_depth_source
            if value in ("obs", "raw"):
                return value
        return self._perception_depth_source

    def _resolve_heightmap_fov_aspect(self, perception_mgr) -> tuple[float, float]:
        grid_x = int(getattr(perception_mgr, "_heightmap_grid_x", 0) or 0)
        grid_y = int(getattr(perception_mgr, "_heightmap_grid_y", 0) or 0)
        interval_x = float(getattr(perception_mgr, "_heightmap_interval_x", 0.0) or 0.0)
        interval_y = float(getattr(perception_mgr, "_heightmap_interval_y", 0.0) or 0.0)
        cfg = getattr(perception_mgr, "cfg", None)
        ray_height = float(getattr(cfg, "ray_start_height", 0.6)) if cfg is not None else 0.6
        if grid_x <= 1 or grid_y <= 1 or interval_x <= 0 or interval_y <= 0 or ray_height <= 0:
            return float(np.deg2rad(90.0)), 1.0
        half_x = 0.5 * (grid_x - 1) * interval_x
        half_y = 0.5 * (grid_y - 1) * interval_y
        if half_y <= 0:
            return float(np.deg2rad(90.0)), 1.0
        fov = float(np.degrees(2.0 * np.arctan(half_y / ray_height)))
        aspect = float(half_x / half_y) if half_y > 0 else 1.0
        fov = float(np.clip(fov, 5.0, 175.0))
        aspect = float(max(aspect, 0.1))
        return float(np.deg2rad(fov)), aspect

    def _get_heightmap_visual_start_shift(self, perception_mgr, env_ids: torch.Tensor) -> torch.Tensor | None:
        try:
            start_pose = perception_mgr.get_heightmap_pose(
                env_ids,
                apply_offsets=True,
                apply_heading_only=True,
            )
            anchor_pose = perception_mgr.get_heightmap_pose(
                env_ids,
                apply_offsets=False,
                apply_heading_only=True,
            )
        except Exception:
            return None
        if start_pose is None or anchor_pose is None:
            return None
        start_pos_t, _ = start_pose
        anchor_pos_t, _ = anchor_pose
        if start_pos_t.numel() == 0 or anchor_pos_t.numel() == 0:
            return None
        return start_pos_t[0] - anchor_pos_t[0]

    def _setup_perception_controls(self) -> None:
        if self._server is None:
            return
        perception_mgr = self._get_perception_manager()
        if perception_mgr is None:
            return
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return
        self._perception_enabled = True
        output_mode = getattr(cfg, "output_mode", "")
        shape = self._resolve_perception_shape(perception_mgr) or (64, 64)
        height, width = shape
        self._perception_last_shape = (height, width)
        self._perception_last_mode = str(output_mode)

        with self._server.gui.add_folder("Perception"):
            show_image_hint = "Display perception image in GUI"
            if not self._disable_perception_frustum:
                show_image_hint = "Display perception image (depth or rendered RGB) in GUI and frustum"
            self._perception_show_depth_cb = self._server.gui.add_checkbox(
                "Show Image",
                initial_value=True,
                hint=show_image_hint,
            )
            if output_mode == "camera_depth":
                self._perception_depth_source_dropdown = self._server.gui.add_dropdown(
                    "Depth Source",
                    options=("obs", "raw"),
                    initial_value=self._perception_depth_source,
                    hint="obs = policy input after crop/resize/normalize; raw = sensor/render output",
                )
            else:
                self._perception_depth_source_dropdown = None
            if not self._disable_perception_frustum:
                self._perception_show_frustum_cb = self._server.gui.add_checkbox(
                    "Show Frustum",
                    initial_value=self._show_perception_frustum_default,
                    hint="Display the perception camera frustum in 3D",
                )
            else:
                self._perception_show_frustum_cb = None
            self._perception_show_points_cb = self._server.gui.add_checkbox(
                "Show Perception Points",
                initial_value=self._scandots_enabled,
                hint="Toggle perception hit points (heightmap or camera scandots)",
            )
            if getattr(cfg, "heightmap_body_name", None):
                self._perception_show_heightmap_joint_cb = self._server.gui.add_checkbox(
                    "Show Heightmap Joint",
                    initial_value=True,
                    hint="Show the joint/body anchor used for heightmap sampling",
                )
            if getattr(cfg, "camera_body_name", None):
                self._perception_show_camera_joint_cb = self._server.gui.add_checkbox(
                    "Show Camera Joint",
                    initial_value=True,
                    hint="Show the body used for camera sampling",
                )
            self._perception_depth_handle = self._server.gui.add_image(
                np.zeros((height, width, 3), dtype=np.uint8),
                label="Perception Image",
                format=self._perception_transport_format,
                jpeg_quality=self._perception_jpeg_quality,
            )
            self._perception_stats = self._server.gui.add_markdown("Depth range (valid): n/a")

        @self._perception_show_depth_cb.on_update
        def _(_evt) -> None:
            if self._perception_depth_handle is None:
                return
            if not bool(self._perception_show_depth_cb.value):
                self._perception_depth_handle.image = np.zeros((height, width, 3), dtype=np.uint8)

        if self._perception_show_frustum_cb is not None:
            @self._perception_show_frustum_cb.on_update
            def _(_evt) -> None:
                if self._perception_frustum is not None:
                    self._perception_frustum.visible = bool(self._perception_show_frustum_cb.value)

        @self._perception_show_points_cb.on_update
        def _(_evt) -> None:
            self._scandots_enabled = bool(self._perception_show_points_cb.value)
            if self._show_scandots_cb is not None and bool(self._show_scandots_cb.value) != self._scandots_enabled:
                self._show_scandots_cb.value = self._scandots_enabled
            if not self._scandots_enabled and self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if not self._scandots_enabled and self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False

        if self._perception_show_heightmap_joint_cb is not None:
            @self._perception_show_heightmap_joint_cb.on_update
            def _(_evt) -> None:
                if self._heightmap_marker_handle is not None:
                    self._heightmap_marker_handle.visible = bool(self._perception_show_heightmap_joint_cb.value)

        if self._perception_show_camera_joint_cb is not None:
            @self._perception_show_camera_joint_cb.on_update
            def _(_evt) -> None:
                if self._camera_marker_handle is not None:
                    self._camera_marker_handle.visible = bool(self._perception_show_camera_joint_cb.value)

        if output_mode == "heightmap":
            fov, aspect = self._resolve_heightmap_fov_aspect(perception_mgr)
        else:
            # viser frustum expects vertical FOV in radians.
            fov = float(np.deg2rad(float(getattr(cfg, "camera_vfov_deg", 90.0))))
            aspect = float(width / max(1, height))
        self._perception_last_fov = fov
        self._perception_last_aspect = aspect

        self._perception_frame = self._server.scene.add_frame(self._scene_path("/perception_camera"), show_axes=False)
        if not self._disable_perception_frustum:
            self._perception_frustum = self._server.scene.add_camera_frustum(
                self._scene_path("/perception_frustum"),
                fov=fov,
                aspect=aspect,
                scale=0.3,
                line_width=2.0,
                color=(0, 0, 0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
                image=np.zeros((height, width, 3), dtype=np.uint8),
                format=self._perception_transport_format,
                jpeg_quality=self._perception_jpeg_quality,
            )
            if self._perception_show_frustum_cb is not None:
                self._perception_frustum.visible = bool(self._perception_show_frustum_cb.value)
        else:
            self._perception_frustum = None
    def _setup_controls(self) -> None:
        if self._server is None:
            return

        def _gui_section_enabled(env_name: str, default: bool) -> bool:
            raw = os.environ.get(env_name)
            if raw is None:
                return default
            return raw.lower() not in ("0", "false", "no", "off")

        visualization_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_VISUALIZATION_GUI",
            not self._distill_minimal_ui,
        )
        perception_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_PERCEPTION_GUI",
            not self._distill_minimal_ui,
        )
        manual_gui_enabled = _gui_section_enabled("VISER_ENABLE_MANUAL_GUI", True)
        manual_root_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_MANUAL_ROOT_GUI",
            manual_gui_enabled and not self._distill_minimal_ui,
        )
        drop_button_gui_enabled = _gui_section_enabled("VISER_ENABLE_DROP_BUTTON_GUI", False)
        self._drop_button_gui_enabled = bool(drop_button_gui_enabled)
        contact_force_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_CONTACT_FORCE_GUI",
            not self._distill_minimal_ui,
        )
        simulation_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_SIMULATION_CONTROL_GUI",
            True,
        )
        reset_object_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_RESET_OBJECT_GUI",
            True,
        )
        world_viz_gui_enabled = _gui_section_enabled(
            "VISER_ENABLE_WORLD_VIZ_GUI",
            not self._distill_minimal_ui,
        )
        clip_gui_enabled = _gui_section_enabled("VISER_ENABLE_CLIP_GUI", True)

        if visualization_gui_enabled:
            with self._server.gui.add_folder("Visualization"):
                self._show_scandots_cb = self._server.gui.add_checkbox(
                    "Show Scandots",
                    initial_value=self._scandots_enabled,
                    hint="Toggle mesh-sampled point cloud",
                )
                self._scandots_size_slider = self._server.gui.add_slider(
                    "Scandots Size",
                    min=0.001,
                    max=0.05,
                    step=0.001,
                    initial_value=float(self._scandots_point_size),
                    hint="Point size for scandots visualization",
                )

            @self._show_scandots_cb.on_update
            def _(_evt) -> None:
                self._scandots_enabled = bool(self._show_scandots_cb.value)
                if (
                    self._perception_show_points_cb is not None
                    and bool(self._perception_show_points_cb.value) != self._scandots_enabled
                ):
                    self._perception_show_points_cb.value = self._scandots_enabled
                if not self._scandots_enabled and self._scandots_handle is not None:
                    self._scandots_handle.visible = False
                if not self._scandots_enabled and self._scandots_rays_handle is not None:
                    self._scandots_rays_handle.visible = False

            @self._scandots_size_slider.on_update
            def _(_evt) -> None:
                self._scandots_point_size = float(self._scandots_size_slider.value)
                if self._scandots_handle is not None:
                    self._scandots_handle.point_size = float(self._scandots_point_size)

        if perception_gui_enabled:
            self._setup_perception_controls()

        if manual_root_gui_enabled:
            with self._server.gui.add_folder("Manual Control", expand_by_default=False):
                self._manual_control_cb = self._server.gui.add_checkbox(
                    "Enable Manual Root Command",
                    initial_value=bool(self._manual_control_default),
                    hint="Set a root-frame relative command [dx, dy, dyaw].",
                )
                self._manual_root_sync_button = self._server.gui.add_button(
                    "Zero Root Command",
                    hint="Reset the relative root command sliders to zero.",
                )
                self._manual_root_pos_x_slider = self._server.gui.add_slider(
                    "Root dX (forward, m)",
                    min=0.0,
                    max=0.7,
                    step=0.02,
                    initial_value=0.0,
                )
                self._manual_root_pos_y_slider = self._server.gui.add_slider(
                    "Root dY (left, m)",
                    min=0.0,
                    max=0.7,
                    step=0.02,
                    initial_value=0.0,
                )
                self._manual_root_yaw_slider = self._server.gui.add_slider(
                    "Root dYaw (rad)",
                    min=-np.pi,
                    max=np.pi,
                    step=0.02,
                    initial_value=0.0,
                )
                self._manual_root_status = self._server.gui.add_markdown("Mode: `idle`\n\nPolicy cmd(root): n/a")

                @self._manual_root_sync_button.on_click
                def _(_evt) -> None:
                    self._sync_manual_root_command_from_robot()
                    self.queue_pending_controls()

                @self._manual_control_cb.on_update
                def _(_evt) -> None:
                    if self._manual_control_cb is not None and bool(self._manual_control_cb.value):
                        self._sync_manual_root_command_from_robot()
                    self.queue_pending_controls()

                for control in (
                    self._manual_root_pos_x_slider,
                    self._manual_root_pos_y_slider,
                    self._manual_root_yaw_slider,
                ):

                    @control.on_update
                    def _(_evt) -> None:
                        self.queue_pending_controls()
                self._sync_manual_root_command_from_robot()

        if drop_button_gui_enabled:
            with self._server.gui.add_folder("Drop Control", expand_by_default=True):
                self._drop_button_cb = self._server.gui.add_checkbox(
                    "Drop Button",
                    initial_value=bool(self._drop_button_default),
                    hint="Explicit scalar for actor_obs_drop_button.",
                )
                self._drop_button_reset_button = self._server.gui.add_button(
                    "Reset Drop Button",
                    hint="Set actor_obs_drop_button back to 0.",
                )
                self._drop_button_status = self._server.gui.add_markdown("drop_button: `0`\n\noverride: `False`")

                @self._drop_button_cb.on_update
                def _(_evt) -> None:
                    self.queue_pending_controls()

                @self._drop_button_reset_button.on_click
                def _(_evt) -> None:
                    if self._drop_button_cb is not None:
                        self._drop_button_cb.value = False
                    self.queue_pending_controls()

                self._update_manual_drop_button()

        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if (
            contact_force_gui_enabled
            and (not self._disable_contact_force_viz)
            and sim_cfg is not None
            and hasattr(sim_cfg, "contact_force_viz")
        ):
            with self._server.gui.add_folder("Contact Forces"):
                self._contact_force_cb = self._server.gui.add_checkbox(
                    "Show Contact Forces",
                    initial_value=bool(getattr(sim_cfg, "contact_force_viz", False)),
                    hint="Toggle contact force debug lines",
                )
                self._contact_force_scale_slider = self._server.gui.add_slider(
                    "Force Scale",
                    min=0.0,
                    max=0.01,
                    step=0.0001,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_scale", 0.001)),
                    hint="Scale factor for contact force lines",
                )
                self._contact_force_threshold_slider = self._server.gui.add_slider(
                    "Force Threshold",
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_threshold", 1.0)),
                    hint="Minimum force magnitude to display",
                )

            def _sync_contact_cfg() -> None:
                if self._contact_force_cb is None:
                    return
                self._set_sim_cfg("contact_force_viz", bool(self._contact_force_cb.value))
                if self._contact_force_scale_slider is not None:
                    self._set_sim_cfg("contact_force_viz_scale", float(self._contact_force_scale_slider.value))
                if self._contact_force_threshold_slider is not None:
                    self._set_sim_cfg(
                        "contact_force_viz_threshold",
                        float(self._contact_force_threshold_slider.value),
                    )

            @self._contact_force_cb.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_scale_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_threshold_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

        advanced_gui_ctx = (
            self._server.gui.add_folder("Advanced", expand_by_default=False)
            if not self._distill_minimal_ui
            else nullcontext()
        )
        with advanced_gui_ctx:
            motion_cmd = self._get_motion_command()
            if simulation_gui_enabled:
                with self._server.gui.add_folder("Simulation Control"):
                    self._play_control = self._server.gui.add_checkbox(
                        "Play",
                        initial_value=not self._start_paused,
                        hint="Toggle simulation play/pause",
                    )
                    self._reset_button = self._server.gui.add_button(
                        "Reset",
                        hint="Reset the selected environment",
                    )
                    if not self._distill_minimal_ui:
                        self._step_button = self._server.gui.add_button(
                            "Step",
                            hint="Step the simulation forward by one frame",
                        )
                        self._default_pose_init_cb = self._server.gui.add_checkbox(
                            "Default Pose Init",
                            initial_value=bool(self._reset_to_default_pose),
                            hint="When enabled, reset/clip apply initializes from the robot default pose instead of the motion pose.",
                        )

                if self._step_button is not None:
                    @self._step_button.on_click
                    def _(_evt) -> None:
                        self._step_requested = True

                @self._reset_button.on_click
                def _(_evt) -> None:
                    if self._reset_restarts_visible_replay:
                        self._reset_visible_requested = True
                    else:
                        self._reset_requested = True

                @self._play_control.on_update
                def _(_evt) -> None:
                    current_value = bool(self._play_control.value)
                    if current_value and not self._play_last_value and self._play_restarts_visible_replay:
                        self._reset_visible_requested = True
                    self._play_last_value = current_value

                if self._default_pose_init_cb is not None:
                    @self._default_pose_init_cb.on_update
                    def _(_evt) -> None:
                        self._set_default_pose_init_enabled(bool(self._default_pose_init_cb.value))

            has_resettable_object = bool(
                motion_cmd is not None
                and hasattr(motion_cmd, "motion")
                and bool(getattr(motion_cmd.motion, "has_object", False))
            )
            if has_resettable_object and self._enable_object_reset_override and reset_object_gui_enabled:
                with self._server.gui.add_folder("Random Box" if self._distill_minimal_ui else "Reset Object"):
                    if self._distill_minimal_ui:
                        self._object_reset_random_button = self._server.gui.add_button(
                            "Randomize Box + Reset",
                            hint="Sample a random XY box reset offset and immediately reset the selected environment.",
                        )
                        self._object_reset_zero_button = self._server.gui.add_button(
                            "Clear Box Override",
                            hint="Disable the box reset override and return to the clip-defined placement.",
                        )
                        self._object_reset_status = self._server.gui.add_markdown(
                            "Mode: `off`\n\n"
                            "Applies on reset while enabled.\n\n"
                            "Runtime size scaling is not supported yet; size still comes from the spawned URDF scale."
                        )
                    else:
                        self._object_reset_override_cb = self._server.gui.add_checkbox(
                            "Enable Reset Box Override",
                            initial_value=False,
                            hint="Apply world-frame box pose offsets on reset while enabled.",
                        )
                        self._object_reset_zero_button = self._server.gui.add_button(
                            "Zero Reset Box Override",
                            hint="Reset box position/rotation offsets to zero.",
                        )
                        self._object_reset_pos_x_slider = self._server.gui.add_slider(
                            "Reset Box dX (world m)",
                            min=-1.0,
                            max=1.0,
                            step=0.01,
                            initial_value=0.0,
                        )
                        self._object_reset_pos_y_slider = self._server.gui.add_slider(
                            "Reset Box dY (world m)",
                            min=-1.0,
                            max=1.0,
                            step=0.01,
                            initial_value=0.0,
                        )
                        self._object_reset_pos_z_slider = self._server.gui.add_slider(
                            "Reset Box dZ (world m)",
                            min=-0.5,
                            max=0.5,
                            step=0.01,
                            initial_value=0.0,
                        )
                        self._object_reset_roll_slider = self._server.gui.add_slider(
                            "Reset Box dRoll (rad)",
                            min=-np.pi,
                            max=np.pi,
                            step=0.02,
                            initial_value=0.0,
                        )
                        self._object_reset_pitch_slider = self._server.gui.add_slider(
                            "Reset Box dPitch (rad)",
                            min=-np.pi,
                            max=np.pi,
                            step=0.02,
                            initial_value=0.0,
                        )
                        self._object_reset_yaw_slider = self._server.gui.add_slider(
                            "Reset Box dYaw (rad)",
                            min=-np.pi,
                            max=np.pi,
                            step=0.02,
                            initial_value=0.0,
                        )
                        self._object_reset_status = self._server.gui.add_markdown(
                            "Mode: `off`\n\n"
                            "Applies on reset while enabled.\n\n"
                            "Runtime size scaling is not supported yet; size still comes from the spawned URDF scale."
                        )

                if self._distill_minimal_ui:
                    @self._object_reset_random_button.on_click
                    def _(_evt) -> None:
                        self._randomize_object_reset_override()

                    @self._object_reset_zero_button.on_click
                    def _(_evt) -> None:
                        self._clear_manual_object_reset_override()

                    self._update_object_reset_status()
                else:
                    @self._object_reset_zero_button.on_click
                    def _(_evt) -> None:
                        self._zero_object_reset_overrides()
                        self._update_manual_object_reset_override()

                    @self._object_reset_override_cb.on_update
                    def _(_evt) -> None:
                        self._update_manual_object_reset_override()

                    for control in (
                        self._object_reset_pos_x_slider,
                        self._object_reset_pos_y_slider,
                        self._object_reset_pos_z_slider,
                        self._object_reset_roll_slider,
                        self._object_reset_pitch_slider,
                        self._object_reset_yaw_slider,
                    ):

                        @control.on_update
                        def _(_evt) -> None:
                            self._update_manual_object_reset_override()

                    self._update_manual_object_reset_override()

            if world_viz_gui_enabled:
                with self._server.gui.add_folder("World Viz"):
                    self._show_robot_cb = self._server.gui.add_checkbox(
                        "Show Robot",
                        initial_value=bool(getattr(self._vr, "show_visual", True)),
                        hint="Toggle robot mesh visibility",
                    )
                    object_cfg = getattr(self._env.robot_config, "object", None)
                    has_object_enabled = bool(object_cfg is not None and getattr(object_cfg, "enabled", False))
                    if self._vo is not None or self._secondary_vo or has_object_enabled:
                        self._show_object_cb = self._server.gui.add_checkbox(
                            "Show Object",
                            initial_value=_get_visual_handle_visible(self._vo, True)
                            if self._vo is not None
                            else True,
                            hint="Toggle object mesh visibility",
                        )
                    if self._sim_meshes_enabled:
                        self._mesh_mode_dropdown = self._server.gui.add_dropdown(
                            "Mesh Mode",
                            options=SIM_MESH_MODE_OPTIONS,
                            initial_value=self._mesh_mode_default,
                            hint="Switch between simulator visual meshes and collision meshes",
                        )
                    if self._terrain_handle is not None or self._ground_handle is not None:
                        self._show_terrain_cb = self._server.gui.add_checkbox(
                            "Show Terrain",
                            initial_value=bool(
                                getattr(self._terrain_handle or self._ground_handle, "visible", True)
                            ),
                            hint="Toggle terrain mesh visibility",
                        )
                    if self._grid_handle is not None:
                        self._show_grid_cb = self._server.gui.add_checkbox(
                            "Show Grid",
                            initial_value=bool(getattr(self._grid_handle, "visible", False)),
                            hint="Toggle the ground grid",
                        )
                    self._recenter_cb = self._server.gui.add_checkbox(
                        "Recenter to Env Origin",
                        initial_value=bool(self._recenter),
                        hint="Keep the selected env centered in view",
                    )
                    if has_object_enabled:
                        self._object_ground_status = self._server.gui.add_markdown(
                            "Env: `0`\n\n"
                            "Mesh source: n/a  Mode: n/a\n\n"
                            "Ground top z: `+0.0000`\n\n"
                            "Sim visual bottom: n/a\n\n"
                            "Ref visual bottom: n/a\n\n"
                            "Sim collision bottom: n/a\n\n"
                            "Ref collision bottom: n/a"
                        )

                def _apply_world_vis() -> None:
                    if self._show_robot_cb is not None and self._vr is not None:
                        try:
                            self._vr.show_visual = bool(self._show_robot_cb.value)
                        except Exception:
                            pass
                    if self._show_robot_cb is not None:
                        show_robot = bool(self._show_robot_cb.value)
                        for vr in self._secondary_vr.values():
                            try:
                                vr.show_visual = show_robot
                            except Exception:
                                pass
                        if self._sim_meshes_enabled:
                            self._set_robot_mesh_visibility_for_env(self._env_id, visible=show_robot)
                            for env_id in self._secondary_env_ids:
                                self._set_robot_mesh_visibility_for_env(env_id, visible=show_robot)
                    if self._show_object_cb is not None:
                        show_object = bool(self._show_object_cb.value)
                        if self._sim_meshes_enabled:
                            self._apply_primary_object_mesh_visibility()
                            for env_id in self._secondary_env_ids:
                                self._apply_secondary_object_mesh_visibility(env_id, visible=show_object)
                        else:
                            if self._vo is not None:
                                try:
                                    _set_visual_handle_visible(self._vo, show_object)
                                except Exception:
                                    pass
                            for vo in self._secondary_vo.values():
                                try:
                                    _set_visual_handle_visible(vo, show_object)
                                except Exception:
                                    pass
                    if self._show_terrain_cb is not None:
                        show_terrain = bool(self._show_terrain_cb.value)
                        self._set_handle_visible(self._terrain_handle, show_terrain)
                        self._set_handle_visible(self._ground_handle, show_terrain)
                    if self._show_grid_cb is not None:
                        self._set_handle_visible(self._grid_handle, bool(self._show_grid_cb.value))

                if self._show_robot_cb is not None:
                    @self._show_robot_cb.on_update
                    def _(_evt) -> None:
                        _apply_world_vis()

                if self._show_object_cb is not None:
                    @self._show_object_cb.on_update
                    def _(_evt) -> None:
                        _apply_world_vis()

                if self._mesh_mode_dropdown is not None:
                    @self._mesh_mode_dropdown.on_update
                    def _(_evt) -> None:
                        _apply_world_vis()

                if self._show_terrain_cb is not None:
                    @self._show_terrain_cb.on_update
                    def _(_evt) -> None:
                        _apply_world_vis()

                if self._show_grid_cb is not None:
                    @self._show_grid_cb.on_update
                    def _(_evt) -> None:
                        _apply_world_vis()

                if self._recenter_cb is not None:
                    @self._recenter_cb.on_update
                    def _(_evt) -> None:
                        self._recenter = bool(self._recenter_cb.value)
                        if self._recenter:
                            offset = self._resolve_env_origin()
                            if offset is None:
                                offset = np.zeros(3, dtype=np.float32)
                            self._offset = offset
                        else:
                            self._offset = np.zeros(3, dtype=np.float32)
                        self._update_terrain_transform(self._offset)

            if clip_gui_enabled:
                motion_cmd = self._get_motion_command()
                if motion_cmd is not None and hasattr(motion_cmd, "motion"):
                    clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
                    if clip_names:
                        self._clip_names = clip_names
                        with self._server.gui.add_folder("Clip" if self._distill_minimal_ui else "Clip Playback"):
                            if len(clip_names) > 1:
                                self._clip_dropdown = self._server.gui.add_dropdown(
                                    "Clip",
                                    options=clip_names,
                                    initial_value=clip_names[0],
                                    hint="Select which motion clip to visualize",
                                )
                            else:
                                self._clip_dropdown = None
                                self._clip_label = self._server.gui.add_markdown(
                                    f"Clip: `{clip_names[0]}`"
                                )
                            if not self._distill_minimal_ui:
                                self._clip_start_slider = self._server.gui.add_slider(
                                    "Clip Start Frame",
                                    min=0,
                                    max=10000,
                                    step=1,
                                    initial_value=0,
                                    hint="Select starting frame in the clip",
                                )
                                self._clip_lock_cb = self._server.gui.add_checkbox(
                                    "Lock Clip",
                                    initial_value=bool(self._clip_lock_default),
                                    hint="Keep the selected clip fixed across resets",
                                )
                            self._clip_apply = self._server.gui.add_button("Apply Clip")
                            if self._clip_label is None:
                                self._clip_label = self._server.gui.add_markdown("")

                        def _update_clip_slider(idx: int | None) -> None:
                            if self._clip_start_slider is None or idx is None:
                                return
                            length = self._get_clip_length(motion_cmd, idx)
                            if length is None:
                                return
                            max_frame = max(0, int(length) - 2)
                            self._clip_start_slider.max = max_frame
                            if int(self._clip_start_slider.value) > max_frame:
                                self._clip_start_slider.value = max_frame

                        def _queue_clip_change() -> None:
                            idx = self._current_clip_index(motion_cmd)
                            if idx is None:
                                return
                            self._pending_clip_idx = idx
                            self._pending_clip_start = (
                                int(self._clip_start_slider.value) if self._clip_start_slider is not None else 0
                            )
                            _update_clip_slider(idx)

                        if self._clip_dropdown is not None:
                            @self._clip_dropdown.on_update
                            def _(_evt) -> None:
                                _queue_clip_change()

                        if self._clip_start_slider is not None:
                            @self._clip_start_slider.on_update
                            def _(_evt) -> None:
                                _queue_clip_change()

                        if self._clip_lock_cb is not None:
                            @self._clip_lock_cb.on_update
                            def _(_evt) -> None:
                                if bool(self._clip_lock_cb.value):
                                    _queue_clip_change()
                                else:
                                    try:
                                        motion_cmd.set_forced_clip(None)
                                    except Exception:
                                        motion_cmd._forced_clip_idx = None
                                    try:
                                        motion_cmd.set_forced_clip_start(None)
                                    except Exception:
                                        motion_cmd._forced_start_step = None

                        @self._clip_apply.on_click
                        def _(_evt) -> None:
                            _queue_clip_change()
                            self.queue_pending_controls()

                        # Force the initial clip so we don't randomize across the bank.
                        self._pending_clip_idx = 0
                        _update_clip_slider(0)

            if self._clip_group_gui_enabled:
                motion_cmd = self._get_motion_command()
                if motion_cmd is not None and hasattr(motion_cmd, "motion"):
                    clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
                    if len(clip_names) > max(1, self._clip_group_size()):
                        with self._server.gui.add_folder("Group Playback"):
                            self._clip_group_index_in = self._server.gui.add_number(
                                "Group Index",
                                initial_value=float(self._initial_clip_group_index),
                                min=0,
                                max=max(0, self._clip_group_total(motion_cmd) - 1),
                                step=1,
                            )
                            self._clip_group_prev_btn = self._server.gui.add_button("Prev Group")
                            self._clip_group_next_btn = self._server.gui.add_button("Next Group")
                            self._clip_group_apply_btn = self._server.gui.add_button("Apply Group")
                            self._clip_group_info = self._server.gui.add_markdown("")

                        @self._clip_group_prev_btn.on_click
                        def _(_evt) -> None:
                            active_group = self._active_visible_group_index(motion_cmd) or 0
                            self._queue_visible_group_index(max(0, active_group - 1))

                        @self._clip_group_next_btn.on_click
                        def _(_evt) -> None:
                            active_group = self._active_visible_group_index(motion_cmd) or 0
                            total_groups = self._clip_group_total(motion_cmd)
                            self._queue_visible_group_index(min(max(0, total_groups - 1), active_group + 1))

                        @self._clip_group_apply_btn.on_click
                        def _(_evt) -> None:
                            if self._clip_group_index_in is None:
                                return
                            self._queue_visible_group_index(int(self._clip_group_index_in.value))

                        @self._clip_group_index_in.on_update
                        def _(_evt) -> None:
                            if self._clip_group_ui_syncing:
                                return
                            self._update_clip_group_ui(motion_cmd)

                        self._update_clip_group_ui(motion_cmd)
                        self._pending_visible_group_index = int(self._initial_clip_group_index)

    def _get_motion_command(self):
        cmd_mgr = getattr(self._env, "command_manager", None)
        if cmd_mgr is None:
            return None
        return cmd_mgr.get_state("motion_command")

    def _current_clip_index(self, motion_cmd) -> int | None:
        if self._clip_dropdown is not None:
            try:
                return self._clip_names.index(str(self._clip_dropdown.value))
            except Exception:
                return None
        if motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return None
        try:
            return int(motion_cmd.clip_ids[self._env_id].item())
        except Exception:
            return None

    def _active_clip_index(self, motion_cmd) -> int | None:
        if motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return None
        try:
            return int(motion_cmd.clip_ids[self._env_id].item())
        except Exception:
            return None

    def _current_clip_name(self, motion_cmd, clip_idx: int | None = None) -> str | None:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return None
        if clip_idx is None:
            clip_idx = self._current_clip_index(motion_cmd)
        if clip_idx is None:
            return None
        clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
        if 0 <= int(clip_idx) < len(clip_names):
            return str(clip_names[int(clip_idx)])
        return None

    def _get_clip_length(self, motion_cmd, clip_idx: int) -> int | None:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return None
        lengths = getattr(motion_cmd.motion, "clip_lengths", None)
        if lengths is None:
            return None
        try:
            if isinstance(lengths, torch.Tensor):
                return int(lengths[clip_idx].item())
            return int(lengths[clip_idx])
        except Exception:
            return None

    def _visible_env_ids(self) -> list[int]:
        return [self._env_id, *self._secondary_env_ids]

    def _clip_group_size(self) -> int:
        if self._clip_group_size_override > 0:
            return int(self._clip_group_size_override)
        return max(1, len(self._visible_env_ids()))

    def _clip_group_total(self, motion_cmd) -> int:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return 1
        clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
        if not clip_names:
            return 1
        group_size = max(1, self._clip_group_size())
        return max(1, int((len(clip_names) + group_size - 1) // group_size))

    def _active_visible_group_index(self, motion_cmd) -> int | None:
        if motion_cmd is None:
            return None
        clip_idx = None
        fixed_clip_ids = getattr(motion_cmd, "_fixed_clip_ids", None)
        if isinstance(fixed_clip_ids, torch.Tensor) and int(fixed_clip_ids.numel()) > int(self._env_id):
            try:
                clip_idx = int(fixed_clip_ids[self._env_id].item())
            except Exception:
                clip_idx = None
        if clip_idx is None:
            clip_idx = self._active_clip_index(motion_cmd)
        if clip_idx is None:
            return None
        return max(0, int(clip_idx) // max(1, self._clip_group_size()))

    def _group_clip_ids(self, motion_cmd, group_index: int) -> list[int]:
        clip_names = list(getattr(motion_cmd.motion, "clip_ids", [])) if motion_cmd is not None and hasattr(motion_cmd, "motion") else []
        if not clip_names:
            return []
        group_size = max(1, self._clip_group_size())
        start = max(0, int(group_index)) * group_size
        last_idx = len(clip_names) - 1
        return [min(start + slot, last_idx) for slot in range(group_size)]

    def _queue_visible_group_index(self, group_index: int) -> None:
        self._pending_visible_group_index = max(0, int(group_index))
        self.queue_pending_controls()

    def _update_clip_group_ui(self, motion_cmd) -> None:
        if self._clip_group_info is None and self._clip_group_index_in is None:
            return
        total_groups = self._clip_group_total(motion_cmd)
        active_group = self._active_visible_group_index(motion_cmd)
        if active_group is None:
            active_group = 0
        active_group = min(max(0, int(active_group)), max(0, total_groups - 1))
        if self._clip_group_index_in is not None:
            self._clip_group_ui_syncing = True
            try:
                self._clip_group_index_in.min = 0
                self._clip_group_index_in.max = max(0, total_groups - 1)
                if int(self._clip_group_index_in.value) != active_group:
                    self._clip_group_index_in.value = active_group
            finally:
                self._clip_group_ui_syncing = False
        if self._clip_group_info is not None and motion_cmd is not None and hasattr(motion_cmd, "motion"):
            env_ids = self._visible_env_ids()
            assignments: list[str] = []
            for env_id in env_ids:
                clip_idx = None
                try:
                    clip_idx = int(motion_cmd.clip_ids[int(env_id)].item())
                except Exception:
                    clip_idx = None
                clip_name = self._current_clip_name(motion_cmd, clip_idx) or "unknown"
                assignments.append(f"`env{int(env_id)}`: `{clip_name}`")
            self._clip_group_info.content = (
                f"Group `{active_group}` / `{max(0, total_groups - 1)}`"
                + "\n\n"
                + " | ".join(assignments)
            )

    def _apply_visible_group_selection(self) -> None:
        motion_cmd = self._get_motion_command()
        group_index = self._pending_visible_group_index
        self._pending_visible_group_index = None
        if motion_cmd is None or group_index is None:
            return
        total_groups = self._clip_group_total(motion_cmd)
        group_index = min(max(0, int(group_index)), max(0, total_groups - 1))
        env_ids_list = self._visible_env_ids()
        clip_ids_list = self._group_clip_ids(motion_cmd, group_index)[: len(env_ids_list)]
        if not env_ids_list or not clip_ids_list:
            return
        env_ids = torch.tensor(env_ids_list, device=self._env.device, dtype=torch.long)
        clip_ids = torch.tensor(clip_ids_list, device=self._env.device, dtype=torch.long)
        try:
            motion_cmd.set_fixed_clip_ids_for_envs(env_ids, clip_ids)
        except Exception:
            if getattr(motion_cmd, "_fixed_clip_ids", None) is None:
                motion_cmd._fixed_clip_ids = motion_cmd.clip_ids.clone()
            motion_cmd._fixed_clip_ids[env_ids] = clip_ids
        logger.info("Viser switched visible clip group {} -> {}", group_index, clip_ids_list)
        self._object_ground_status_debug_logged = False
        self._reset_visible_envs()
        self._update_clip_group_ui(motion_cmd)

    def _set_default_pose_init_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._reset_to_default_pose = enabled
        os.environ["HOLOSOMA_DEFAULT_POSE_INIT"] = "1" if enabled else "0"
        os.environ["HOLOSOMA_RESET_TO_DEFAULT_POSE"] = "1" if enabled else "0"
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return
        try:
            motion_cmd._reset_to_default_pose = enabled
        except Exception:
            pass

    def _reset_env(self) -> None:
        if not hasattr(self._env, "reset_envs_idx"):
            return

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        motion_cmd = self._get_motion_command()
        manual_enabled = bool(getattr(motion_cmd, "manual_control_enabled", False))
        if self._manual_control_cb is not None:
            manual_enabled = bool(manual_enabled or bool(self._manual_control_cb.value))

        if motion_cmd is not None:
            if self._drop_button_gui_enabled:
                self._clear_manual_drop_button(clear_gui_toggle=True)
            if manual_enabled:
                self._clear_manual_commands(clear_gui_toggles=True)
            clip_idx = self._current_clip_index(motion_cmd)
            if clip_idx is None and hasattr(motion_cmd, "clip_ids"):
                try:
                    clip_idx = int(motion_cmd.clip_ids[self._env_id].item())
                except Exception:
                    clip_idx = None
            lock_enabled = self._clip_lock_enabled()
            if clip_idx is not None and lock_enabled:
                try:
                    motion_cmd.set_forced_clip(int(clip_idx))
                except Exception:
                    motion_cmd._forced_clip_idx = int(clip_idx)
                if self._reset_to_default_pose:
                    self._env.reset_envs_idx(env_ids)
                    if hasattr(self._env, "reset_buf"):
                        self._env.reset_buf[env_ids] = 0
                    if hasattr(self._env, "time_out_buf"):
                        self._env.time_out_buf[env_ids] = 0
                    active_idx = self._active_clip_index(motion_cmd)
                    if active_idx is None:
                        active_idx = int(clip_idx)
                    self._reload_terrain_for_clip(self._current_clip_name(motion_cmd, int(active_idx)))
                    self._sync_after_reset(env_ids)
                    return
                clip_start = int(self._clip_start_slider.value) if self._clip_start_slider is not None else 0
                try:
                    motion_cmd.set_forced_clip_start(int(clip_start))
                except Exception:
                    motion_cmd._forced_start_step = int(clip_start)
                self._force_clip_state(motion_cmd, int(clip_idx), clip_start)
                active_idx = self._active_clip_index(motion_cmd)
                if active_idx is None:
                    active_idx = int(clip_idx)
                self._reload_terrain_for_clip(self._current_clip_name(motion_cmd, int(active_idx)))
                self._sync_after_reset(env_ids)
                return

        self._env.reset_envs_idx(env_ids)
        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0
        self._sync_after_reset(env_ids)

    def _reset_visible_envs(self) -> None:
        if not hasattr(self._env, "reset_envs_idx"):
            return

        visible_env_ids = [self._env_id, *self._secondary_env_ids]
        logger.info("Viser reset visible replay envs: {}", visible_env_ids)
        env_ids = torch.tensor(visible_env_ids, device=self._env.device, dtype=torch.long)
        motion_cmd = self._get_motion_command()
        manual_enabled = bool(getattr(motion_cmd, "manual_control_enabled", False))
        if self._manual_control_cb is not None:
            manual_enabled = bool(manual_enabled or bool(self._manual_control_cb.value))
        if self._drop_button_gui_enabled:
            self._clear_manual_drop_button(clear_gui_toggle=True)
        if manual_enabled:
            self._clear_manual_commands(clear_gui_toggles=True)

        self._env.reset_envs_idx(env_ids)
        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0
        self._sync_after_reset(env_ids)

    def _sync_after_reset(self, env_ids: torch.Tensor) -> None:
        refresh_hook = getattr(self._env, "_refresh_envs_after_reset", None)
        if callable(refresh_hook):
            try:
                refresh_hook(env_ids)
            except Exception:
                pass
        if hasattr(self._env, "_compute_observations"):
            self._env._compute_observations()
        if hasattr(self._env, "_post_compute_observations_callback"):
            self._env._post_compute_observations_callback()
        if hasattr(self._env, "_clip_observations"):
            self._env._clip_observations()
        self._invalidate_isaac_scandots_payload()
        if hasattr(self._env, "_draw_scandots_in_viewer"):
            self._env._draw_scandots_in_viewer()
        self._invalidate_object_mesh_caches(remove_handles=True)
        sim = getattr(self._env, "simulator", None)
        scene = getattr(sim, "scene", None)
        if scene is not None and hasattr(scene, "update"):
            try:
                fps = float(getattr(getattr(sim, "simulator_config", None), "sim", None).fps)
                dt = 1.0 / fps if fps > 0.0 else 0.0
            except Exception:
                dt = 0.0
            try:
                scene.update(dt=dt)
            except Exception:
                pass
            if hasattr(sim, "refresh_sim_tensors"):
                try:
                    sim.refresh_sim_tensors()
                except Exception:
                    pass
        self._sync_manual_root_command_from_robot()
        self.record_step()

    def _apply_clip_selection(self) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._pending_clip_idx = None
            self._pending_clip_start = None
            return
        clip_idx = self._pending_clip_idx
        clip_start = self._pending_clip_start
        self._pending_clip_idx = None
        self._pending_clip_start = None
        if clip_idx is None:
            return
        self._invalidate_object_mesh_caches(remove_handles=True)
        lock_enabled = self._clip_lock_enabled()
        if lock_enabled:
            try:
                motion_cmd.set_forced_clip(int(clip_idx))
            except Exception:
                motion_cmd._forced_clip_idx = int(clip_idx)
            start_frame = int(clip_start or 0)
            try:
                motion_cmd.set_forced_clip_start(start_frame)
            except Exception:
                motion_cmd._forced_start_step = start_frame
        else:
            try:
                motion_cmd.set_forced_clip(None)
            except Exception:
                motion_cmd._forced_clip_idx = None
            try:
                motion_cmd.set_forced_clip_start(None)
            except Exception:
                motion_cmd._forced_start_step = None
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._force_clip_state(motion_cmd, int(clip_idx), clip_start, exact_motion_state=True)
        active_idx = self._active_clip_index(motion_cmd)
        if active_idx is None:
            active_idx = int(clip_idx)
        self._reload_terrain_for_clip(self._current_clip_name(motion_cmd, int(active_idx)))
        self._sync_after_reset(env_ids)

    def _force_clip_state(
        self,
        motion_cmd,
        clip_idx: int,
        clip_start: int | None,
        *,
        exact_motion_state: bool = False,
    ) -> None:
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._env.reset_envs_idx(env_ids)

        clip_length = self._get_clip_length(motion_cmd, clip_idx)
        max_valid = max(0, int(clip_length or 1) - 2)
        start_frame = max(0, min(int(clip_start or 0), max_valid))

        motion_cmd.clip_ids[env_ids] = int(clip_idx)
        motion_cmd.time_steps[env_ids] = start_frame
        if motion_cmd.motion_cfg.align_motion_to_init_yaw:
            motion_cmd._update_motion_alignment(env_ids)
        if exact_motion_state and hasattr(motion_cmd, "_clear_runtime_default_pose_prepend"):
            motion_cmd._clear_runtime_default_pose_prepend(env_ids)

        root_pos = motion_cmd.root_pos_w[env_ids].clone()
        root_rot = motion_cmd.root_quat_w[env_ids].clone()
        if exact_motion_state:
            root_lin_vel = motion_cmd.body_lin_vel_w[env_ids, 0].clone()
            root_ang_vel = motion_cmd.body_ang_vel_w[env_ids, 0].clone()
        else:
            motion_idx = motion_cmd._get_motion_indices(motion_cmd.time_steps[env_ids], env_ids)
            root_lin_vel = motion_cmd.motion.body_lin_vel_w[motion_idx, 0].clone()
            root_ang_vel = motion_cmd.motion.body_ang_vel_w[motion_idx, 0].clone()
            if motion_cmd.motion_cfg.align_motion_to_init_yaw:
                root_lin_vel = motion_cmd._apply_motion_alignment_vec(root_lin_vel)
                root_ang_vel = motion_cmd._apply_motion_alignment_vec(root_ang_vel)

        dof_pos = motion_cmd.joint_pos[env_ids].clone()
        dof_vel = motion_cmd.joint_vel[env_ids].clone()

        sim = self._env.simulator
        sim.dof_pos[env_ids] = dof_pos
        sim.dof_vel[env_ids] = dof_vel
        sim.robot_root_states[env_ids, :3] = root_pos
        sim.robot_root_states[env_ids, 3:7] = root_rot
        sim.robot_root_states[env_ids, 7:10] = root_lin_vel
        sim.robot_root_states[env_ids, 10:13] = root_ang_vel

        if hasattr(sim, "set_actor_root_state_tensor_robots"):
            sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        else:
            sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)

        if hasattr(sim, "set_dof_state_tensor_robots"):
            sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
        else:
            sim.set_dof_state_tensor(env_ids, sim.dof_state)

        if motion_cmd.motion.has_object:
            obj_pos = motion_cmd.object_pos_w[env_ids].clone()
            obj_ori = motion_cmd.object_quat_w[env_ids].clone()
            obj_lin_vel = motion_cmd.object_lin_vel_w[env_ids].clone()
            if not exact_motion_state and hasattr(motion_cmd, "_apply_manual_object_reset_overrides"):
                obj_pos, obj_ori = motion_cmd._apply_manual_object_reset_overrides(obj_pos, obj_ori, env_ids)
            obj_states = torch.cat([obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
            if hasattr(motion_cmd, "_set_simulator_object_states"):
                motion_cmd._set_simulator_object_states(env_ids, obj_states)
            else:
                sim.set_actor_states([motion_cmd.object_name], env_ids, obj_states)

            if hasattr(motion_cmd, "_reset_pickup_anchor_state"):
                motion_cmd._reset_pickup_anchor_state(
                    env_ids,
                    root_pos_w=root_pos,
                    root_quat_w=root_rot,
                    object_pos_w=obj_pos,
                )

        if hasattr(sim, "scene") and hasattr(sim.scene, "write_data_to_sim"):
            sim.scene.write_data_to_sim()
        sim.refresh_sim_tensors()

        motion_cmd._update_future_target_poses()
        if hasattr(self._env, "_refresh_envs_after_reset"):
            self._env._refresh_envs_after_reset(env_ids)

        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0
        if hasattr(self._env, "episode_length_buf"):
            self._env.episode_length_buf[env_ids] = 0

        self._env._compute_observations()
        self._env._post_compute_observations_callback()
        self._env._clip_observations()
        self._invalidate_isaac_scandots_payload()

    def _resolve_env_origin_for(self, env_id: int) -> np.ndarray | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is not None and hasattr(motion_cmd, "_get_env_offsets"):
            try:
                device = getattr(motion_cmd, "device", None) or getattr(self._env, "device", "cpu")
                env_ids = torch.tensor([int(env_id)], device=device, dtype=torch.long)
                offsets = motion_cmd._get_env_offsets(env_ids)
                if offsets is not None:
                    return offsets[0].detach().cpu().numpy()
            except Exception:
                pass

        origins = None
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is not None and hasattr(terrain_mgr, "get_state"):
            terrain_state = terrain_mgr.get_state("locomotion_terrain")
            if terrain_state is not None and hasattr(terrain_state, "env_origins"):
                origins = terrain_state.env_origins
        if origins is None:
            scene = getattr(self._env.simulator, "scene", None)
            if scene is not None and hasattr(scene, "env_origins"):
                origins = scene.env_origins
        if origins is None:
            return None
        try:
            origin = origins[int(env_id)]
        except Exception:
            return None
        if isinstance(origin, torch.Tensor):
            return origin.detach().cpu().numpy()
        return np.asarray(origin, dtype=np.float32)

    def _resolve_env_origin(self) -> np.ndarray | None:
        return self._resolve_env_origin_for(self._env_id)

    def _get_root_state_wxyz_for(self, env_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        sim = self._env.simulator
        root_states = getattr(sim, "robot_root_states", None)

        rb_pos_all = getattr(sim, "_rigid_body_pos", None)
        rb_quat_all = getattr(sim, "_rigid_body_rot", None)
        rb_idx = self._root_body_index
        if (
            rb_idx is not None
            and isinstance(rb_pos_all, torch.Tensor)
            and isinstance(rb_quat_all, torch.Tensor)
            and rb_pos_all.ndim >= 3
            and rb_quat_all.ndim >= 3
            and int(env_id) < rb_pos_all.shape[0]
            and int(env_id) < rb_quat_all.shape[0]
            and rb_idx < rb_pos_all.shape[1]
            and rb_idx < rb_quat_all.shape[1]
        ):
            pos = rb_pos_all[int(env_id), rb_idx]
            quat_xyzw = rb_quat_all[int(env_id), rb_idx]
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

            if int(env_id) == self._env_id and (not self._root_pose_debug_logged) and root_states is not None:
                try:
                    root = root_states[int(env_id)]
                    root_pos = root[0:3]
                    root_quat_xyzw = root[3:7]
                    pos_err = float(torch.linalg.norm(pos - root_pos).item())
                    q0 = quat_xyzw / torch.norm(quat_xyzw).clamp(min=1.0e-6)
                    q1 = root_quat_xyzw / torch.norm(root_quat_xyzw).clamp(min=1.0e-6)
                    dot = torch.clamp(torch.abs(torch.sum(q0 * q1)), min=0.0, max=1.0)
                    ang = float(2.0 * torch.rad2deg(torch.arccos(dot)).item())
                    logger.info(
                        "Viser root pose uses rigid body '{}' (idx={}): delta_vs_root_state pos={:.4f}m ang={:.2f}deg",
                        self._root_body_name or "unknown",
                        int(rb_idx),
                        pos_err,
                        ang,
                    )
                except Exception:
                    pass
                self._root_pose_debug_logged = True

            return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

        if root_states is None:
            return None, None

        root = root_states[int(env_id)]
        pos = root[0:3]
        quat_xyzw = root[3:7]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

    def _get_root_state_wxyz(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._get_root_state_wxyz_for(self._env_id)

    def _get_dof_pos_for(self, env_id: int) -> np.ndarray | None:
        dof_pos = getattr(self._env.simulator, "dof_pos", None)
        if dof_pos is None:
            return None
        if int(env_id) < 0 or int(env_id) >= int(dof_pos.shape[0]):
            return None
        return dof_pos[int(env_id)].detach().cpu().numpy()

    def _get_dof_pos(self) -> np.ndarray | None:
        return self._get_dof_pos_for(self._env_id)

    def _get_object_state_wxyz_for(self, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not getattr(self._env.robot_config, "object", None):
            return None
        if not getattr(self._env.robot_config.object, "enabled", False):
            return None
        if not getattr(self._env.robot_config.object, "object_urdf_path", None):
            return None

        sim_state = self._get_simulator_object_state_xyzw_for(int(env_id))
        if sim_state is not None:
            pos, quat_xyzw = sim_state
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            return pos, quat_wxyz

        if self._mesh_source == "sim":
            return None

        motion_cmd = self._get_motion_command()
        if motion_cmd is not None and hasattr(motion_cmd, "simulator_object_pos_w"):
            try:
                pos = motion_cmd.simulator_object_pos_w[int(env_id)]
                quat_xyzw = motion_cmd.simulator_object_quat_w[int(env_id)]
                quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
                return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()
            except Exception:
                pass

        return None

    def _get_object_state_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._get_object_state_wxyz_for(self._env_id)

    def _get_reference_object_state_wxyz_for(self, env_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None
        if not hasattr(motion_cmd, "object_pos_w") or not hasattr(motion_cmd, "object_quat_w"):
            return None
        try:
            pos = motion_cmd.object_pos_w[int(env_id)]
            quat_xyzw = motion_cmd.object_quat_w[int(env_id)]
        except Exception:
            return None
        try:
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()
        except Exception:
            return None

    def _get_reference_object_state_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        return self._get_reference_object_state_wxyz_for(self._env_id)

    def _get_robot_body_points_for(self, env_id: int) -> np.ndarray | None:
        sim = self._env.simulator
        rigid_body_pos = getattr(sim, "_rigid_body_pos", None)
        if rigid_body_pos is None:
            return None
        env_idx = int(env_id)
        if env_idx < 0 or env_idx >= int(rigid_body_pos.shape[0]):
            return None
        try:
            pts = rigid_body_pos[env_idx].detach().cpu().numpy()
        except Exception:
            return None
        if pts.ndim != 2 or pts.shape[-1] != 3:
            return None
        return pts

    def _update_robot_points(
        self,
        *,
        env_id: int,
        offset: np.ndarray,
        display_shift: np.ndarray | None = None,
        visible: bool,
    ) -> None:
        if self._server is None:
            return
        if env_id == self._env_id:
            handle = self._robot_points_handle
            path = self._scene_path("/robot/body_points")
        else:
            handle = self._secondary_robot_points_handles.get(int(env_id))
            path = self._scene_path(f"/env_{env_id}/robot/body_points")
        points = self._get_robot_body_points_for(env_id)
        if points is None or points.size == 0:
            if handle is not None:
                try:
                    handle.visible = False
                except Exception:
                    pass
            return
        pts = points - offset
        if display_shift is not None:
            pts = pts + display_shift
        if handle is None:
            handle = self._server.scene.add_point_cloud(
                path,
                points=pts.astype(np.float32, copy=False),
                colors=np.tile(SIM_ROBOT_POINTS_COLOR, (pts.shape[0], 1)),
                point_size=0.02,
                point_shape="circle",
                precision="float32",
            )
            if env_id == self._env_id:
                self._robot_points_handle = handle
            else:
                self._secondary_robot_points_handles[int(env_id)] = handle
        else:
            handle.points = pts.astype(np.float32, copy=False)
        try:
            handle.visible = bool(visible)
        except Exception:
            pass

    def _update_object_point(
        self,
        *,
        env_id: int,
        object_pos: np.ndarray | None,
        offset: np.ndarray,
        display_shift: np.ndarray | None = None,
        visible: bool,
    ) -> None:
        if self._server is None:
            return
        if env_id == self._env_id:
            handle = self._object_points_handle
            path = self._scene_path("/object/position")
        else:
            handle = self._secondary_object_points_handles.get(int(env_id))
            path = self._scene_path(f"/env_{env_id}/object/position")
        if object_pos is None:
            if handle is not None:
                try:
                    handle.visible = False
                except Exception:
                    pass
            return
        pts = object_pos.reshape(1, 3) - offset.reshape(1, 3)
        if display_shift is not None:
            pts = pts + display_shift.reshape(1, 3)
        if handle is None:
            handle = self._server.scene.add_point_cloud(
                path,
                points=pts.astype(np.float32, copy=False),
                colors=np.tile(SIM_OBJECT_POINTS_COLOR, (1, 1)),
                point_size=0.045,
                point_shape="circle",
                precision="float32",
            )
            if env_id == self._env_id:
                self._object_points_handle = handle
            else:
                self._secondary_object_points_handles[int(env_id)] = handle
        else:
            handle.points = pts.astype(np.float32, copy=False)
        try:
            handle.visible = bool(visible)
        except Exception:
            pass

    def _update_scandots(self, offset: np.ndarray) -> None:
        if not self._server or not self._scandots_enabled:
            return

        self._ray_direction_stats_suffix = ""
        source_env = os.environ.get("VISER_SCANDOTS_SOURCE", "auto").strip().lower()
        isaac_only = source_env in ("isaac", "isaacsim", "isaac_payload")
        if isaac_only and self._update_scandots_from_isaac_payload(offset):
            return
        if isaac_only and not self._scandots_warned:
            logger.warning(
                "Viser scandots source=isaac payload unavailable/stale; falling back to live perception rays."
            )
            self._scandots_warned = True
        if source_env == "auto" and self._update_scandots_from_isaac_payload(offset):
            return

        perception_mgr = getattr(self._env, "perception_manager", None)
        if perception_mgr is None:
            if not self._scandots_warned:
                logger.warning("Viser scandots requested but perception_manager is unavailable.")
                self._scandots_warned = True
            return

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        output_mode = getattr(getattr(perception_mgr, "cfg", None), "output_mode", None)
        use_heightmap = output_mode == "heightmap"
        include_misses_env = os.environ.get("VISER_SCANDOTS_INCLUDE_MISSES")
        if include_misses_env is None:
            include_misses = False
        else:
            include_misses = include_misses_env.lower() not in (
                "0",
                "false",
                "no",
            )
        use_depth_mask_env = os.environ.get("VISER_SCANDOTS_USE_DEPTH_MASK")
        if use_depth_mask_env is None:
            use_depth_mask = False
        else:
            use_depth_mask = use_depth_mask_env.lower() not in (
                "0",
                "false",
                "no",
            )
        points_follow_lines_env = os.environ.get("VISER_SCANDOTS_POINTS_FOLLOW_LINES")
        if points_follow_lines_env is None:
            points_follow_lines = True
        else:
            points_follow_lines = points_follow_lines_env.lower() not in (
                "0",
                "false",
                "no",
            )
        try:
            with torch.no_grad():
                if use_heightmap and hasattr(perception_mgr, "get_heightmap_points"):
                    result = perception_mgr.get_heightmap_points(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
                else:
                    if self._strict_camera_rays and output_mode == "camera_depth":
                        if not hasattr(perception_mgr, "get_camera_depth_ray_samples"):
                            raise RuntimeError("PerceptionManager does not expose get_camera_depth_ray_samples().")
                        result = perception_mgr.get_camera_depth_ray_samples(
                            env_ids,
                            include_misses=include_misses,
                            return_rays=True,
                        )
                    else:
                        result = perception_mgr.get_camera_scandots_points(
                            env_ids,
                            include_misses=include_misses,
                            return_rays=True,
                        )
        except Exception as exc:
            if not self._scandots_warned:
                logger.warning("Viser scandots disabled: {}", exc)
                self._scandots_warned = True
            return

        if result is None:
            if not self._scandots_warned:
                if use_heightmap:
                    logger.warning("Viser scandots disabled: heightmap points are unavailable.")
                elif self._strict_camera_rays and output_mode == "camera_depth":
                    logger.warning(
                        "Viser rays disabled: strict camera ray sync requested but active perception source has no ray samples."
                    )
                else:
                    logger.warning("Viser scandots disabled: perception is not using mesh_raycast_scandots.")
                self._scandots_warned = True
            self._scandots_enabled = False
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        if not isinstance(result, tuple) or len(result) < 2:
            return
        points = result[0]
        mask = result[1]
        ray_starts = result[2] if len(result) > 2 else None
        ray_dirs = result[3] if len(result) > 3 else None
        ray_hits_world = result[4] if len(result) > 4 else None
        if points.numel() == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return
        points_all_env = points[0]
        mask_env = mask[0] if mask is not None else None
        mask_env_bool = None
        if mask_env is not None and mask_env.numel() > 0:
            mask_env_bool = mask_env.to(torch.bool)

        points_env = points_all_env
        hits_env = None
        mesh_hit_mask_env = None
        if ray_hits_world is not None and ray_hits_world.numel() > 0:
            hits_candidate = ray_hits_world[0]
            if hits_candidate.shape == points_all_env.shape:
                hits_env = hits_candidate
                mesh_hit_mask_env = torch.isfinite(hits_env).all(dim=-1)

        if hits_env is not None and mesh_hit_mask_env is not None:
            # Red dots should represent true mesh intersections.
            # By default respect the depth-valid hit mask so dots match the final depth map.
            # For debugging, VISER_SCANDOTS_USE_DEPTH_MASK=0 keeps all finite mesh intersections.
            point_mask_env = mesh_hit_mask_env
            if (
                use_depth_mask
                and (mask_env_bool is not None)
                and (mask_env_bool.shape == point_mask_env.shape)
            ):
                point_mask_env = point_mask_env & mask_env_bool
            points_env = hits_env[point_mask_env]
        elif (mask_env_bool is not None) and use_depth_mask:
            points_env = points_all_env[mask_env_bool]
        if points_env.numel() == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        pts = points_env.detach().cpu().numpy()
        if self._recenter:
            pts = pts - offset

        if self._scandots_handle is None:
            self._scandots_handle = self._server.scene.add_point_cloud(
                self._scene_path("/scandots"),
                points=pts.astype(np.float32, copy=False),
                colors=self._scandots_color,
                point_size=float(self._scandots_point_size),
                point_shape="circle",
            )
        else:
            self._scandots_handle.visible = True
            self._scandots_handle.points = pts.astype(np.float32, copy=False)

        if ray_starts is None or ray_dirs is None:
            return
        starts_all_env = ray_starts[0]
        dirs_env = ray_dirs[0]
        if starts_all_env.numel() == 0 or dirs_env.numel() == 0:
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return
        if use_heightmap:
            shift = self._get_heightmap_visual_start_shift(perception_mgr, env_ids)
            if shift is not None:
                starts_all_env = starts_all_env - shift.unsqueeze(0)

        if output_mode == "camera_depth":
            try:
                dirs_norm = dirs_env / torch.norm(dirs_env, dim=-1, keepdim=True).clamp(min=1.0e-6)

                use_frame_quat = bool(getattr(perception_mgr, "_use_camera_frame_quat", False))
                strict_warp = bool(getattr(perception_mgr, "_camera_strict_warp", False))
                cam_pose = perception_mgr.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=True,
                )
                cam_quat = cam_pose[1][0:1]
                if strict_warp:
                    forward_local = torch.tensor([[0.0, 0.0, 1.0]], device=cam_quat.device, dtype=cam_quat.dtype)
                elif use_frame_quat:
                    forward_local = torch.tensor([[0.0, 0.0, -1.0]], device=cam_quat.device, dtype=cam_quat.dtype)
                else:
                    forward_local = torch.tensor([[1.0, 0.0, 0.0]], device=cam_quat.device, dtype=cam_quat.dtype)
                cam_forward = quat_apply(cam_quat, forward_local, w_last=True)[0]
                cam_forward = cam_forward / torch.norm(cam_forward).clamp(min=1.0e-6)
                dots_cam = torch.sum(dirs_norm * cam_forward.unsqueeze(0), dim=-1)
                back_cam = int((dots_cam <= 0.0).sum().item())
                total = int(dots_cam.numel())
                min_dot_cam = float(dots_cam.min().item())

                root_suffix = ""
                root_quat = getattr(self._env, "base_quat", None)
                if isinstance(root_quat, torch.Tensor) and root_quat.ndim >= 2 and self._env_id < root_quat.shape[0]:
                    root_quat_env = root_quat[self._env_id : self._env_id + 1]
                    root_forward_local = torch.tensor(
                        [[1.0, 0.0, 0.0]],
                        device=root_quat_env.device,
                        dtype=root_quat_env.dtype,
                    )
                    root_forward = quat_apply(root_quat_env, root_forward_local, w_last=True)[0]
                    root_forward = root_forward / torch.norm(root_forward).clamp(min=1.0e-6)
                    dots_root = torch.sum(dirs_norm * root_forward.unsqueeze(0), dim=-1)
                    back_root = int((dots_root <= 0.0).sum().item())
                    min_dot_root = float(dots_root.min().item())
                    root_suffix = f" | rays_back_root={back_root}/{total} min_dot_root={min_dot_root:.3f}"

                self._ray_direction_stats_suffix = (
                    f" | rays_back_cam={back_cam}/{total} min_dot_cam={min_dot_cam:.3f}" + root_suffix
                )
            except Exception:
                self._ray_direction_stats_suffix = ""
        if include_misses:
            ends_all_env = points_all_env
        elif hits_env is not None:
            ends_all_env = hits_env if hits_env.shape == starts_all_env.shape else points_all_env
        else:
            ends_all_env = points_all_env

        starts_env = starts_all_env
        ends_env = ends_all_env
        if not include_misses:
            draw_mask = None
            if use_depth_mask and mask_env_bool is not None and mask_env_bool.numel() > 0:
                # Render only depth-valid hits.
                draw_mask = mask_env_bool
            elif mesh_hit_mask_env is not None and mesh_hit_mask_env.numel() > 0:
                # Debug mode: render finite mesh hits even if they are depth-invalid.
                draw_mask = mesh_hit_mask_env
            if draw_mask is not None:
                starts_env = starts_all_env[draw_mask]
                ends_env = ends_all_env[draw_mask]
        if starts_env.numel() == 0 or ends_env.numel() == 0:
            if points_follow_lines and self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        if points_follow_lines:
            # Keep dots and line endpoints exactly consistent for debugging.
            points_env = ends_env
            pts = points_env.detach().cpu().numpy()
            if self._recenter:
                pts = pts - offset
            if self._scandots_handle is None:
                self._scandots_handle = self._server.scene.add_point_cloud(
                    self._scene_path("/scandots"),
                    points=pts.astype(np.float32, copy=False),
                    colors=self._scandots_color,
                    point_size=float(self._scandots_point_size),
                    point_shape="circle",
                )
            else:
                self._scandots_handle.visible = True
                self._scandots_handle.points = pts.astype(np.float32, copy=False)

        lines = torch.stack([starts_env, ends_env], dim=1).detach().cpu().numpy()
        if self._recenter:
            lines = lines - offset[None, None, :]
        colors = np.full((lines.shape[0], 2, 3), [0, 0, 0], dtype=np.uint8)
        if self._scandots_rays_handle is None:
            self._scandots_rays_handle = self._server.scene.add_line_segments(
                self._scene_path("/scandots_rays"),
                points=lines.astype(np.float32, copy=False),
                colors=colors.astype(np.uint8, copy=False),
                line_width=1.0,
            )
        else:
            self._scandots_rays_handle.visible = True
            self._scandots_rays_handle.points = lines.astype(np.float32, copy=False)
            try:
                self._scandots_rays_handle.colors = colors.astype(np.uint8, copy=False)
            except Exception:
                pass

    def _update_scandots_from_isaac_payload(self, offset: np.ndarray) -> bool:
        payload = getattr(self._env, "_isaac_scandots_payload", None)
        if not isinstance(payload, dict):
            return False
        try:
            payload_env_id = int(payload.get("env_id", -1))
        except Exception:
            return False
        if payload_env_id != self._env_id:
            return False

        points = payload.get("points")
        if points is None:
            return False
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return False
        if pts.shape[0] == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return True

        pts_draw = pts
        if self._recenter:
            pts_draw = pts_draw - offset
        if self._scandots_handle is None:
            self._scandots_handle = self._server.scene.add_point_cloud(
                self._scene_path("/scandots"),
                points=pts_draw.astype(np.float32, copy=False),
                colors=self._scandots_color,
                point_size=float(self._scandots_point_size),
                point_shape="circle",
            )
        else:
            self._scandots_handle.visible = True
            self._scandots_handle.points = pts_draw.astype(np.float32, copy=False)

        line_starts = payload.get("line_starts")
        line_ends = payload.get("line_ends")
        if line_starts is None or line_ends is None:
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return True

        starts = np.asarray(line_starts, dtype=np.float32)
        ends = np.asarray(line_ends, dtype=np.float32)
        if starts.ndim != 2 or ends.ndim != 2 or starts.shape != ends.shape or starts.shape[1] != 3:
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return True
        if starts.shape[0] == 0:
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return True
        perception_mgr = getattr(self._env, "perception_manager", None)
        output_mode = getattr(getattr(perception_mgr, "cfg", None), "output_mode", None)
        if output_mode == "heightmap" and perception_mgr is not None:
            env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
            shift = self._get_heightmap_visual_start_shift(perception_mgr, env_ids)
            if shift is not None:
                starts = starts - shift.detach().cpu().numpy().reshape(1, 3)

        lines = np.stack([starts, ends], axis=1).astype(np.float32, copy=False)
        if self._recenter:
            lines = lines - offset[None, None, :]
        colors = np.full((lines.shape[0], 2, 3), [0, 0, 0], dtype=np.uint8)
        if self._scandots_rays_handle is None:
            self._scandots_rays_handle = self._server.scene.add_line_segments(
                self._scene_path("/scandots_rays"),
                points=lines,
                colors=colors.astype(np.uint8, copy=False),
                line_width=1.0,
            )
        else:
            self._scandots_rays_handle.visible = True
            self._scandots_rays_handle.points = lines
            try:
                self._scandots_rays_handle.colors = colors.astype(np.uint8, copy=False)
            except Exception:
                pass
        return True

    def _depth_map_from_strict_camera_rays(
        self,
        perception_mgr: Any,
        env_ids: torch.Tensor,
        *,
        near: float,
        far: float,
    ) -> tuple[np.ndarray | None, str]:
        """Rebuild a depth map from the exact strict ray samples shown in Viser."""
        if not self._strict_camera_rays:
            return None, ""
        if not hasattr(perception_mgr, "get_camera_depth_ray_samples"):
            return None, ""

        try:
            with torch.no_grad():
                result = perception_mgr.get_camera_depth_ray_samples(
                    env_ids,
                    include_misses=False,
                    return_rays=True,
                )
        except Exception:
            return None, ""

        if not isinstance(result, tuple) or len(result) < 5:
            return None, ""

        _points, hit_mask, ray_starts, ray_dirs_world, ray_hits_world = result
        if ray_starts is None or ray_dirs_world is None or ray_hits_world is None:
            return None, ""
        if ray_starts.numel() == 0 or ray_dirs_world.numel() == 0 or ray_hits_world.numel() == 0:
            return None, ""

        starts = ray_starts[0]
        dirs = ray_dirs_world[0]
        hits = ray_hits_world[0]
        if starts.shape != dirs.shape or hits.shape != dirs.shape:
            return None, ""

        if hit_mask is not None and hit_mask.numel() > 0:
            mask = hit_mask[0].to(torch.bool)
        else:
            mask = torch.isfinite(hits).all(dim=-1)

        far_val = float(far)
        if far_val <= 0.0:
            return None, ""

        dirs_norm = dirs / torch.norm(dirs, dim=-1, keepdim=True).clamp(min=1.0e-6)
        delta = hits - starts
        ranges = torch.sum(delta * dirs_norm, dim=-1)
        ranges = torch.where(torch.isfinite(ranges), ranges, torch.full_like(ranges, far_val))
        ranges = torch.clamp(ranges, min=0.0, max=far_val)

        width = int(getattr(perception_mgr, "_camera_width", 0) or 0)
        height = int(getattr(perception_mgr, "_camera_height", 0) or 0)
        center_dir = None
        if hasattr(perception_mgr, "_get_camera_forward_axis"):
            try:
                _cam_pos, body_quat = perception_mgr.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=False,
                    apply_pitch=False,
                )
                forward_world = perception_mgr._get_camera_forward_axis(body_quat)[0]  # noqa: SLF001
                center_dir = forward_world / torch.norm(forward_world).clamp(min=1.0e-6)
            except Exception:
                center_dir = None
        if center_dir is None:
            if width > 0 and height > 0 and (width * height) == int(dirs_norm.shape[0]):
                center_idx = (height // 2) * width + (width // 2)
            else:
                center_idx = int(dirs_norm.shape[0] // 2)
            center_dir = dirs_norm[center_idx]
            center_dir = center_dir / torch.norm(center_dir).clamp(min=1.0e-6)

        # Match warp_sensors depth projection multiplier: clamp dot(rd, rd_principal) to [eps, 1].
        dots = torch.sum(dirs_norm * center_dir.unsqueeze(0), dim=-1)
        dots = torch.clamp(dots, min=1.0e-6, max=1.0)

        depth = ranges * dots
        depth = torch.where(mask, depth, torch.full_like(depth, far_val))
        depth = torch.clamp(depth, min=0.0, max=far_val)

        if width <= 0 or height <= 0 or (width * height) != int(depth.numel()):
            return None, ""

        valid = torch.isfinite(depth)
        valid &= depth >= float(near)
        valid &= depth < (far_val - 1.0e-6)
        strict_suffix = (
            f" | strict_hits={int(mask.sum().item())}/{int(depth.numel())}"
            f" strict_valid={int(valid.sum().item())}/{int(depth.numel())}"
        )

        depth_map = depth.view(height, width).detach().cpu().numpy()
        return depth_map, strict_suffix

    def _update_perception_visuals(self, offset: np.ndarray) -> None:
        if not self._server or not self._perception_enabled:
            return
        perception_mgr = self._get_perception_manager()
        if perception_mgr is None:
            return
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return

        output_mode = str(getattr(cfg, "output_mode", ""))
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        depth_map = None
        near = float(getattr(cfg, "camera_near", 0.3) or 0.3)
        max_distance = float(getattr(cfg, "max_distance", 10.0))
        camera_far = float(getattr(cfg, "camera_far", max_distance))
        far = float(min(max_distance, camera_far))
        cam_pos = None
        cam_quat_xyzw = None
        cam_body_quat_xyzw = None
        strict_depth_suffix = ""

        if output_mode == "camera_depth":
            depth_source = self._get_perception_depth_source()
            if not self._disable_perception_image_pipeline:
                try:
                    if depth_source == "obs":
                        depth = perception_mgr.get_camera_depth_obs_map()
                    else:
                        depth = perception_mgr.get_camera_depth_map()
                except Exception:
                    depth = None
                if isinstance(depth, torch.Tensor) and depth.numel() > 0:
                    depth_map = depth[self._env_id].detach().cpu().numpy()
                    if depth_source == "obs" and bool(getattr(cfg, "camera_warp_normalize", False)):
                        denom = max(1.0e-6, far - near)
                        depth_map = np.clip(depth_map + 0.5, 0.0, 1.0) * denom + near
                    if self._perception_flip_vertical:
                        depth_map = np.flipud(depth_map)
                strict_depth_map, strict_depth_suffix = self._depth_map_from_strict_camera_rays(
                    perception_mgr,
                    env_ids,
                    near=near,
                    far=far,
                )
                if strict_depth_map is not None and depth_source == "raw":
                    depth_map = strict_depth_map
            try:
                cam_pos_t, cam_quat_t = perception_mgr.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=True,
                )
                cam_pos = cam_pos_t[0].detach().cpu().numpy()
                cam_quat_xyzw = cam_quat_t[0].detach().cpu()
            except Exception:
                cam_pos = None
                cam_quat_xyzw = None
            try:
                cam_body_pos_t, cam_body_quat_t = perception_mgr.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=False,
                )
                cam_body_quat_xyzw = cam_body_quat_t[0].detach().cpu()
            except Exception:
                cam_body_quat_xyzw = None
        elif output_mode == "heightmap":
            near = 0.0
            try:
                if hasattr(perception_mgr, "get_heightmap_map"):
                    hm = perception_mgr.get_heightmap_map(env_ids)
                else:
                    hm = getattr(perception_mgr, "_heightmap", None)
            except Exception:
                hm = None
            if isinstance(hm, torch.Tensor) and hm.numel() > 0:
                if hm.ndim == 3:
                    depth_map = hm[0].detach().cpu().numpy()
                elif hm.ndim == 2:
                    depth_map = hm.detach().cpu().numpy()
            try:
                pose = perception_mgr.get_heightmap_pose(
                    env_ids,
                    apply_offsets=False,
                    apply_heading_only=True,
                )
            except Exception:
                pose = None
            if pose is not None:
                cam_pos_t, body_quat = pose
                forward_world = quat_apply(
                    body_quat,
                    torch.tensor([0.0, 0.0, -1.0], device=self._env.device),
                    w_last=True,
                )
                up_hint = quat_apply(
                    body_quat,
                    torch.tensor([1.0, 0.0, 0.0], device=self._env.device),
                    w_last=True,
                )
                cam_quat_t = _quat_from_forward_up(forward_world, up_hint)
                cam_pos = cam_pos_t[0].detach().cpu().numpy()
                cam_quat_xyzw = cam_quat_t.detach().cpu()

        self._update_perception_markers(perception_mgr, env_ids, offset)

        if self._disable_perception_image_pipeline:
            h, w = self._perception_last_shape
            depth_map = np.full((h, w), np.nan, dtype=np.float32)

        if depth_map is None:
            return

        if self._depth_colormap == "fixed":
            depth_img = _depth_to_rgb_fixed_range(depth_map, near, far)
        else:
            depth_img = _depth_to_rgb(depth_map, near, far)
        display_img = depth_img
        image_mode = "depth" if self._disable_perception_image_pipeline else self._perception_image_mode
        source_mode = str(getattr(cfg, "camera_source", ""))
        can_show_rendered_rgb = output_mode == "camera_depth" and source_mode in {
            "rendered",
            "rendered_depth_sensor",
        }
        if image_mode == "rgb":
            if not can_show_rendered_rgb:
                raise RuntimeError(
                    "VISER_PERCEPTION_IMAGE_MODE=rgb requires camera_source=rendered or rendered_depth_sensor."
                )
            rgb = perception_mgr.capture_rendered_rgb()
            rgb_img = np.asarray(rgb)
            if rgb_img.ndim != 3 or rgb_img.shape[-1] < 3:
                raise RuntimeError(f"Rendered RGB image has invalid shape: {tuple(rgb_img.shape)}")
            rgb_img = rgb_img[:, :, :3]
            if self._perception_flip_vertical:
                rgb_img = np.flipud(rgb_img)
            if rgb_img.dtype != np.uint8:
                rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
            display_img = rgb_img
        elif image_mode == "auto" and can_show_rendered_rgb:
            try:
                rgb = perception_mgr.capture_rendered_rgb()
            except Exception:
                rgb = None
            if rgb is not None:
                rgb_img = np.asarray(rgb)
                if rgb_img.ndim == 3 and rgb_img.shape[-1] >= 3:
                    rgb_img = rgb_img[:, :, :3]
                    if self._perception_flip_vertical:
                        rgb_img = np.flipud(rgb_img)
                    if rgb_img.dtype != np.uint8:
                        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
                    display_img = rgb_img

        display_shape = (int(display_img.shape[0]), int(display_img.shape[1]))
        if display_shape != self._perception_last_shape or output_mode != self._perception_last_mode:
            self._perception_last_shape = display_shape
            self._perception_last_mode = output_mode
            if self._perception_depth_handle is not None:
                self._perception_depth_handle.image = np.zeros(display_img.shape, dtype=np.uint8)

        if self._perception_show_depth_cb is None or bool(self._perception_show_depth_cb.value):
            if self._perception_depth_handle is not None:
                self._perception_depth_handle.image = display_img
        if self._perception_stats is not None:
            direction_stats_suffix = ""
            if output_mode == "camera_depth":
                pitch_deg = float(getattr(cfg, "camera_pitch_deg", 0.0) or 0.0)
                direction_stats_suffix = f" | pitch={pitch_deg:.1f}{self._ray_direction_stats_suffix}{strict_depth_suffix}"
            if self._disable_perception_image_pipeline:
                self._perception_stats.content = (
                    "Perception image pipeline disabled (VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE=1)"
                    f"{direction_stats_suffix}"
                )
            else:
                min_d, max_d, count = _valid_depth_stats(depth_map, near, far)
                frame_crc = _depth_crc32(depth_map, far)
                depth_source = self._get_perception_depth_source() if output_mode == "camera_depth" else "raw"
                if count == 0:
                    self._perception_stats.content = (
                        "Depth range (valid): n/a (no hits)"
                        f" | crc32={frame_crc}"
                        f" | src={depth_source}"
                        f" | map={self._depth_colormap}"
                        f" | flip_v={int(self._perception_flip_vertical)}"
                        f" | tx={self._perception_transport_format}"
                        f"{direction_stats_suffix}"
                    )
                else:
                    total = depth_map.size
                    self._perception_stats.content = (
                        f"Depth range (valid): {min_d:.3f} - {max_d:.3f} m | valid: {count}/{total}"
                        f" | crc32={frame_crc}"
                        f" | src={depth_source}"
                        f" | map={self._depth_colormap}"
                        f" | flip_v={int(self._perception_flip_vertical)}"
                        f" | tx={self._perception_transport_format}"
                        f"{direction_stats_suffix}"
                    )

        if cam_pos is None or cam_quat_xyzw is None:
            return

        cam_pos = cam_pos - offset
        cam_quat_wxyz = cam_quat_xyzw.detach().cpu().numpy()[[3, 0, 1, 2]]
        frustum_quat_wxyz = None
        if output_mode == "camera_depth":
            # Prefer orientation reconstructed from this frame's actual ray directions.
            if hasattr(perception_mgr, "get_camera_depth_ray_samples"):
                try:
                    with torch.no_grad():
                        frustum_samples = perception_mgr.get_camera_depth_ray_samples(
                            env_ids,
                            include_misses=False,
                            return_rays=True,
                        )
                except Exception:
                    frustum_samples = None
                if isinstance(frustum_samples, tuple) and len(frustum_samples) >= 4:
                    ray_dirs_world = frustum_samples[3]
                    if isinstance(ray_dirs_world, torch.Tensor) and ray_dirs_world.numel() > 0:
                        dirs_env = ray_dirs_world[0]
                        width = int(getattr(perception_mgr, "_camera_width", 0) or 0)
                        height = int(getattr(perception_mgr, "_camera_height", 0) or 0)
                        use_grid = (width > 0 and height > 0 and (width * height) == int(dirs_env.shape[0]))
                        frustum_quat = _frustum_quat_from_world_rays(
                            dirs_env,
                            width=width if use_grid else None,
                            height=height if use_grid else None,
                        )
                        if frustum_quat is not None:
                            frustum_quat_wxyz = frustum_quat.detach().cpu().numpy()

        if output_mode == "camera_depth" and cam_body_quat_xyzw is not None and frustum_quat_wxyz is None:
            use_frame_quat = bool(getattr(perception_mgr, "_use_camera_frame_quat", False))
            strict_warp = bool(getattr(perception_mgr, "_camera_strict_warp", False))
            ray_dirs_base = getattr(perception_mgr, "_camera_scandots_ray_dirs_base", None)
            width = getattr(perception_mgr, "_camera_scandots_width", None)
            height = getattr(perception_mgr, "_camera_scandots_height", None)
            if ray_dirs_base is None:
                ray_dirs_base = getattr(perception_mgr, "_camera_ray_dirs_base", None)
                width = int(getattr(perception_mgr, "_camera_width", 0) or 0)
                height = int(getattr(perception_mgr, "_camera_height", 0) or 0)
            if strict_warp or (not use_frame_quat):
                frustum_quat = _frustum_quat_from_rays(
                    ray_dirs_base,
                    cam_body_quat_xyzw,
                    width=width if width and height else None,
                    height=height if width and height else None,
                )
                if frustum_quat is not None:
                    frustum_quat_wxyz = frustum_quat.detach().cpu().numpy()
        if frustum_quat_wxyz is None:
            frustum_quat_wxyz = _frustum_quat_from_camera(cam_quat_xyzw).detach().cpu().numpy()

        if self._perception_frame is not None:
            self._perception_frame.position = cam_pos
            self._perception_frame.wxyz = cam_quat_wxyz

        if self._perception_frustum is None:
            return

        if output_mode == "heightmap":
            fov, aspect = self._resolve_heightmap_fov_aspect(perception_mgr)
        else:
            # viser frustum expects vertical FOV in radians.
            fov = float(np.deg2rad(float(getattr(cfg, "camera_vfov_deg", 90.0))))
            aspect = float(display_shape[1] / max(1, display_shape[0]))

        if fov != self._perception_last_fov:
            try:
                self._perception_frustum.fov = fov
            except Exception:
                pass
            self._perception_last_fov = fov
        if aspect != self._perception_last_aspect:
            try:
                self._perception_frustum.aspect = aspect
            except Exception:
                pass
            self._perception_last_aspect = aspect

        self._perception_frustum.position = cam_pos
        self._perception_frustum.wxyz = frustum_quat_wxyz
        if self._perception_show_frustum_cb is None or bool(self._perception_show_frustum_cb.value):
            self._perception_frustum.visible = True
        if self._perception_show_depth_cb is None or bool(self._perception_show_depth_cb.value):
            self._perception_frustum.image = display_img

    def _update_target_keypoints(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        motion_cmd = self._get_motion_command()
        manual_active = False
        if motion_cmd is not None:
            manual_active = bool(getattr(motion_cmd, "manual_control_enabled", False))
        if (not self._show_target_keypoints) or manual_active:
            if self._target_keypoints_handle is not None:
                try:
                    self._target_keypoints_handle.visible = False
                except Exception:
                    pass
            if self._target_keypoint_skeleton_handle is not None:
                try:
                    self._target_keypoint_skeleton_handle.visible = False
                except Exception:
                    pass
            return
        if motion_cmd is None or not hasattr(motion_cmd, "body_pos_w"):
            return
        try:
            points = motion_cmd.body_pos_w
            pts_env = points[self._env_id]
        except Exception:
            return

        pts = pts_env.detach().cpu().numpy()
        if self._recenter:
            pts = pts - offset

        if pts.size == 0:
            return

        if self._target_keypoints_handle is None:
            colors = np.tile(self._target_keypoints_color, (pts.shape[0], 1))
            self._target_keypoints_handle = self._server.scene.add_point_cloud(
                self._scene_path("/target_keypoints"),
                points=pts.astype(np.float32, copy=False),
                colors=colors.astype(np.uint8, copy=False),
                point_size=float(self._target_keypoints_point_size),
                point_shape="circle",
                precision="float32",
            )
        else:
            try:
                self._target_keypoints_handle.visible = True
            except Exception:
                pass
            self._target_keypoints_handle.points = pts.astype(np.float32, copy=False)
            if getattr(self._target_keypoints_handle, "colors", None) is not None:
                colors = np.tile(self._target_keypoints_color, (pts.shape[0], 1))
                try:
                    self._target_keypoints_handle.colors = colors.astype(np.uint8, copy=False)
                except Exception:
                    pass

        if self._target_skeleton_edges is None:
            tracked_body_names = tuple(getattr(getattr(motion_cmd, "motion_cfg", None), "body_names_to_track", []) or [])
            if tracked_body_names and len(tracked_body_names) == int(pts.shape[0]):
                self._target_skeleton_edges = _load_target_skeleton_edges(
                    _resolve_robot_urdf_path(self._env.robot_config),
                    tracked_body_names,
                )
            else:
                self._target_skeleton_edges = ()

        if not self._target_skeleton_edges:
            if self._target_keypoint_skeleton_handle is not None:
                try:
                    self._target_keypoint_skeleton_handle.visible = False
                except Exception:
                    pass
            return

        segments = np.asarray(
            [[pts[parent_idx], pts[child_idx]] for parent_idx, child_idx in self._target_skeleton_edges],
            dtype=np.float32,
        )
        segment_colors = np.full((segments.shape[0], 2, 3), self._target_keypoints_color, dtype=np.uint8)
        if self._target_keypoint_skeleton_handle is None:
            self._target_keypoint_skeleton_handle = self._server.scene.add_line_segments(
                self._scene_path("/target_keypoint_skeleton"),
                points=segments,
                colors=segment_colors,
                line_width=float(self._target_keypoints_line_width),
            )
        else:
            try:
                self._target_keypoint_skeleton_handle.visible = True
            except Exception:
                pass
            self._target_keypoint_skeleton_handle.points = segments
            try:
                self._target_keypoint_skeleton_handle.colors = segment_colors
            except Exception:
                pass

    def _update_target_object_center(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_target_object_center:
            if self._target_object_center_handle is not None:
                try:
                    self._target_object_center_handle.visible = False
                except Exception:
                    pass
            return

        ref_positions = self._reference_object_future_trajectory_w()
        if ref_positions is None or ref_positions.shape[0] == 0:
            if self._target_object_center_handle is not None:
                try:
                    self._target_object_center_handle.visible = False
                except Exception:
                    pass
            return

        center_idx = 1 if ref_positions.shape[0] > 1 else 0
        center = np.asarray(ref_positions[center_idx], dtype=np.float32).reshape(1, 3)
        if self._recenter:
            center = center - offset.reshape(1, 3)
        colors = self._target_object_center_color.reshape(1, 3).astype(np.uint8, copy=False)

        if self._target_object_center_handle is None:
            self._target_object_center_handle = self._server.scene.add_point_cloud(
                self._scene_path("/target_object_center"),
                points=center,
                colors=colors,
                point_size=float(self._target_object_center_point_size),
                point_shape="circle",
                precision="float32",
            )
        else:
            try:
                self._target_object_center_handle.visible = True
            except Exception:
                pass
            self._target_object_center_handle.points = center
            if getattr(self._target_object_center_handle, "colors", None) is not None:
                try:
                    self._target_object_center_handle.colors = colors
                except Exception:
                    pass

    def _reference_object_future_trajectory_w(self) -> np.ndarray | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not bool(getattr(getattr(motion_cmd, "motion", None), "has_object", False)):
            return None

        try:
            env_ids = torch.tensor([self._env_id], device=motion_cmd.device, dtype=torch.long)
            current_pos = motion_cmd.object_pos_w[env_ids]
            num_future_steps = max(1, int(getattr(motion_cmd, "num_future_steps", 1) or 1))
            time_offsets = torch.arange(1, num_future_steps + 1, device=motion_cmd.device, dtype=torch.long)
            future_steps = motion_cmd.time_steps[env_ids].unsqueeze(1) + time_offsets.unsqueeze(0)
            clip_lengths = motion_cmd._current_clip_lengths()[env_ids].unsqueeze(1)
            future_steps = torch.minimum(future_steps, torch.clamp(clip_lengths - 1, min=0))
            future_steps_global = motion_cmd._get_motion_indices(future_steps, env_ids=env_ids)
            future_pos = motion_cmd.motion.object_pos_w[future_steps_global]
            if bool(getattr(motion_cmd.motion_cfg, "align_motion_to_init_yaw", False)):
                future_pos = motion_cmd._apply_motion_alignment_pos_subset(future_pos, env_ids)
            else:
                future_pos = future_pos + motion_cmd._get_env_offsets(env_ids)[:, None, :]
            trajectory = torch.cat([current_pos[:, None, :], future_pos], dim=1)
        except Exception:
            return None

        return trajectory[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def _reference_object_clip_trajectory_w(self) -> np.ndarray | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not bool(getattr(getattr(motion_cmd, "motion", None), "has_object", False)):
            return None

        try:
            env_ids = torch.tensor([self._env_id], device=motion_cmd.device, dtype=torch.long)
            clip_lengths = motion_cmd._current_clip_lengths(env_ids)
            clip_length = int(clip_lengths[0].item()) if clip_lengths.numel() > 0 else 0
            if clip_length <= 0:
                return None
            local_steps = torch.arange(clip_length, device=motion_cmd.device, dtype=torch.long).unsqueeze(0)
            clip_steps_global = motion_cmd._get_motion_indices(local_steps, env_ids=env_ids)
            clip_pos = motion_cmd.motion.object_pos_w[clip_steps_global]
            if bool(getattr(motion_cmd.motion_cfg, "align_motion_to_init_yaw", False)):
                clip_pos = motion_cmd._apply_motion_alignment_pos_subset(clip_pos, env_ids)
            else:
                clip_pos = clip_pos + motion_cmd._get_env_offsets(env_ids)[:, None, :]
        except Exception:
            return None

        return clip_pos[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def _clear_rollout_object_trajectory(self) -> None:
        self._rollout_object_trajectory_points_w.clear()
        self._rollout_object_trajectory_last_clip_idx = None
        self._rollout_object_trajectory_last_time_step = None
        if self._rollout_object_trajectory_handle is not None:
            try:
                self._rollout_object_trajectory_handle.visible = False
            except Exception:
                pass

    def _clear_rollout_root_trajectory(self) -> None:
        self._rollout_root_trajectory_points_w.clear()
        self._rollout_root_trajectory_last_clip_idx = None
        self._rollout_root_trajectory_last_time_step = None
        if self._rollout_root_trajectory_handle is not None:
            try:
                self._rollout_root_trajectory_handle.visible = False
            except Exception:
                pass

    def _current_rollout_object_pos_w(self) -> np.ndarray | None:
        state = self._get_object_state_wxyz()
        if state is None:
            return None
        pos_w, _quat_wxyz = state
        return np.asarray(pos_w, dtype=np.float32).reshape(3)

    def _trajectory_ribbon_mesh(
        self,
        positions_w: np.ndarray,
        offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        positions = np.asarray(positions_w, dtype=np.float32).reshape(-1, 3)
        if positions.shape[0] < 2:
            return None
        if self._recenter:
            positions = positions - offset.reshape(1, 3)
        positions = positions.copy()
        positions[:, 2] += float(self._rollout_trajectory_z_offset)

        half_width = 0.5 * max(float(self._rollout_trajectory_mesh_width), 1.0e-3)
        vertices: list[np.ndarray] = []
        faces: list[tuple[int, int, int]] = []
        for idx in range(positions.shape[0] - 1):
            p0 = positions[idx]
            p1 = positions[idx + 1]
            direction = p1 - p0
            direction[2] = 0.0
            length = float(np.linalg.norm(direction[:2]))
            if length < 1.0e-5:
                continue
            normal = np.array([-direction[1] / length, direction[0] / length, 0.0], dtype=np.float32)
            base = len(vertices)
            vertices.extend(
                [
                    p0 + normal * half_width,
                    p0 - normal * half_width,
                    p1 + normal * half_width,
                    p1 - normal * half_width,
                ]
            )
            faces.append((base, base + 1, base + 2))
            faces.append((base + 1, base + 3, base + 2))
        if not faces:
            return None
        return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    def _update_trajectory_mesh(
        self,
        *,
        handle_attr: str,
        node_path: str,
        positions_w: np.ndarray,
        offset: np.ndarray,
        color: np.ndarray,
    ) -> None:
        mesh = self._trajectory_ribbon_mesh(positions_w, offset)
        handle = getattr(self, handle_attr)
        if mesh is None:
            if handle is not None:
                try:
                    handle.visible = False
                except Exception:
                    pass
            return

        vertices, faces = mesh
        if handle is not None:
            try:
                handle.remove()
            except Exception:
                pass
        handle = self._server.scene.add_mesh_simple(
            self._scene_path(node_path),
            vertices,
            faces,
            color=tuple(int(v) for v in np.asarray(color, dtype=np.uint8).reshape(3)),
            side="double",
            visible=True,
        )
        setattr(self, handle_attr, handle)
        if node_path not in self._rollout_trajectory_mesh_logged:
            logger.info(
                "Viser rollout trajectory mesh created: path={} points={} vertices={} faces={} width={:.3f} z_offset={:.3f}",
                self._scene_path(node_path),
                int(np.asarray(positions_w).reshape(-1, 3).shape[0]),
                int(vertices.shape[0]),
                int(faces.shape[0]),
                float(self._rollout_trajectory_mesh_width),
                float(self._rollout_trajectory_z_offset),
            )
            self._rollout_trajectory_mesh_logged.add(node_path)

    def _update_rollout_root_trajectory(self, root_pos_w: np.ndarray, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_rollout_root_trajectory:
            if self._rollout_root_trajectory_handle is not None:
                try:
                    self._rollout_root_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        motion_cmd = self._get_motion_command()
        clip_idx = self._active_clip_index(motion_cmd)
        time_step: int | None = None
        if motion_cmd is not None and hasattr(motion_cmd, "time_steps"):
            try:
                time_step = int(motion_cmd.time_steps[self._env_id].item())
            except Exception:
                time_step = None

        if (
            self._rollout_root_trajectory_last_clip_idx is not None
            and clip_idx is not None
            and clip_idx != self._rollout_root_trajectory_last_clip_idx
        ) or (
            self._rollout_root_trajectory_last_time_step is not None
            and time_step is not None
            and time_step < self._rollout_root_trajectory_last_time_step
        ):
            self._clear_rollout_root_trajectory()

        pos_w = np.asarray(root_pos_w, dtype=np.float32).reshape(3)
        should_append = False
        if not self._rollout_root_trajectory_points_w:
            should_append = True
        elif time_step is not None and time_step != self._rollout_root_trajectory_last_time_step:
            should_append = True
        elif not np.allclose(self._rollout_root_trajectory_points_w[-1], pos_w, atol=1.0e-5):
            should_append = True
        if should_append:
            self._rollout_root_trajectory_points_w.append(pos_w.copy())

        self._rollout_root_trajectory_last_clip_idx = clip_idx
        self._rollout_root_trajectory_last_time_step = time_step

        if len(self._rollout_root_trajectory_points_w) < 2:
            if self._rollout_root_trajectory_handle is not None:
                try:
                    self._rollout_root_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        rollout_positions = np.asarray(self._rollout_root_trajectory_points_w, dtype=np.float32)
        self._update_trajectory_mesh(
            handle_attr="_rollout_root_trajectory_handle",
            node_path="/rollout_root_trajectory",
            positions_w=rollout_positions,
            offset=offset,
            color=self._rollout_root_trajectory_color,
        )

    def _update_rollout_object_trajectory(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_rollout_object_trajectory:
            if self._rollout_object_trajectory_handle is not None:
                try:
                    self._rollout_object_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        motion_cmd = self._get_motion_command()
        clip_idx = self._active_clip_index(motion_cmd)
        time_step: int | None = None
        if motion_cmd is not None and hasattr(motion_cmd, "time_steps"):
            try:
                time_step = int(motion_cmd.time_steps[self._env_id].item())
            except Exception:
                time_step = None

        if (
            self._rollout_object_trajectory_last_clip_idx is not None
            and clip_idx is not None
            and clip_idx != self._rollout_object_trajectory_last_clip_idx
        ) or (
            self._rollout_object_trajectory_last_time_step is not None
            and time_step is not None
            and time_step < self._rollout_object_trajectory_last_time_step
        ):
            self._clear_rollout_object_trajectory()

        pos_w = self._current_rollout_object_pos_w()
        if pos_w is None:
            if self._rollout_object_trajectory_handle is not None:
                try:
                    self._rollout_object_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        should_append = False
        if not self._rollout_object_trajectory_points_w:
            should_append = True
        elif time_step is not None and time_step != self._rollout_object_trajectory_last_time_step:
            should_append = True
        elif not np.allclose(self._rollout_object_trajectory_points_w[-1], pos_w, atol=1.0e-5):
            should_append = True
        if should_append:
            self._rollout_object_trajectory_points_w.append(pos_w.copy())

        self._rollout_object_trajectory_last_clip_idx = clip_idx
        self._rollout_object_trajectory_last_time_step = time_step

        if len(self._rollout_object_trajectory_points_w) < 2:
            if self._rollout_object_trajectory_handle is not None:
                try:
                    self._rollout_object_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        rollout_positions = np.asarray(self._rollout_object_trajectory_points_w, dtype=np.float32)
        self._update_trajectory_mesh(
            handle_attr="_rollout_object_trajectory_handle",
            node_path="/rollout_object_trajectory",
            positions_w=rollout_positions,
            offset=offset,
            color=self._rollout_object_trajectory_color,
        )

    def _update_target_object_trajectory(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_target_object_trajectory:
            if self._target_object_trajectory_handle is not None:
                try:
                    self._target_object_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        ref_positions = self._reference_object_clip_trajectory_w()
        if ref_positions is None or ref_positions.shape[0] < 2:
            if self._target_object_trajectory_handle is not None:
                try:
                    self._target_object_trajectory_handle.visible = False
                except Exception:
                    pass
            return

        segments = np.stack([ref_positions[:-1], ref_positions[1:]], axis=1).astype(np.float32, copy=False)
        if self._recenter:
            segments = segments - offset.reshape(1, 1, 3)
        colors = np.full((segments.shape[0], 2, 3), self._target_object_center_color, dtype=np.uint8)

        if self._target_object_trajectory_handle is None:
            self._target_object_trajectory_handle = self._server.scene.add_line_segments(
                self._scene_path("/target_object_trajectory"),
                points=segments,
                colors=colors,
                line_width=float(self._target_object_trajectory_line_width),
            )
        else:
            try:
                self._target_object_trajectory_handle.visible = True
            except Exception:
                pass
            self._target_object_trajectory_handle.points = segments
            try:
                self._target_object_trajectory_handle.colors = colors
            except Exception:
                pass

    @staticmethod
    def _rot6d_to_quat_wxyz(rot6d: torch.Tensor) -> np.ndarray:
        rot6d = rot6d.reshape(-1, 6)
        first_col = torch.nn.functional.normalize(rot6d[:, 0:3], dim=-1)
        second_col_raw = rot6d[:, 3:6]
        second_col = torch.nn.functional.normalize(
            second_col_raw - torch.sum(first_col * second_col_raw, dim=-1, keepdim=True) * first_col,
            dim=-1,
        )
        third_col = torch.cross(first_col, second_col, dim=-1)
        rot_mat = torch.stack((first_col, second_col, third_col), dim=-1)
        quat_wxyz = matrix_to_quaternion(rot_mat)
        return quat_wxyz[0].detach().cpu().numpy()

    @staticmethod
    def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
        half = 0.5 * float(yaw)
        return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    def _resolve_target_box_dimensions(self) -> np.ndarray | None:
        object_urdf = self._resolve_object_urdf_for_env(self._env_id)
        if object_urdf:
            extents = load_urdf_geometry_extents(object_urdf)
            if extents is not None:
                return np.asarray(extents, dtype=np.float32)

        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None
        try:
            return motion_cmd.object_size[self._env_id].detach().float().cpu().numpy().astype(np.float32)
        except Exception:
            return None

    def _target_box_resting_center_z(self, box_size: np.ndarray) -> float:
        try:
            env_origin_z = float(self._env.simulator.scene.env_origins[self._env_id, 2].item())
        except Exception:
            env_origin_z = 0.0
        return env_origin_z + 0.5 * float(box_size[2])

    def _target_box_pose_from_command(self, goal_xy: np.ndarray, box_size: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return None

        try:
            env_ids = torch.tensor([self._env_id], device=motion_cmd.device, dtype=torch.long)
            anchor_pos_t = motion_cmd.robot_root_pos_w[env_ids]
            anchor_quat_t = motion_cmd.robot_root_quat_w[env_ids]
            anchor_pos_w = anchor_pos_t[0].detach().float().cpu().numpy()
            anchor_quat_xyzw = anchor_quat_t[0].detach().float().cpu().numpy()
        except Exception:
            return None

        anchor_quat_wxyz = anchor_quat_xyzw[[3, 0, 1, 2]]
        anchor_yaw = self._yaw_from_quat_wxyz(anchor_quat_wxyz)
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy.shape[0] < 2:
            return None
        dx = float(goal_xy[0])
        dy = float(goal_xy[1])
        cos_yaw = float(np.cos(anchor_yaw))
        sin_yaw = float(np.sin(anchor_yaw))
        world_xy = np.asarray(
            [
                anchor_pos_w[0] + cos_yaw * dx - sin_yaw * dy,
                anchor_pos_w[1] + sin_yaw * dx + cos_yaw * dy,
            ],
            dtype=np.float32,
        )
        goal_pos_w = np.asarray(
            [world_xy[0], world_xy[1], self._target_box_resting_center_z(box_size)],
            dtype=np.float32,
        )
        goal_quat_wxyz = self._quat_wxyz_from_yaw(anchor_yaw)
        return goal_pos_w, goal_quat_wxyz

    def _get_effective_target_box_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not hasattr(motion_cmd, "motion") or not bool(getattr(motion_cmd.motion, "has_object", False)):
            return None

        object_size = self._resolve_target_box_dimensions()
        if object_size is None:
            return None

        goal_xy = self._current_effective_goal_xy()
        if goal_xy is None:
            return None
        goal_pose = self._target_box_pose_from_command(goal_xy, np.asarray(object_size, dtype=np.float32))
        if goal_pose is None:
            return None
        goal_pos_w, goal_quat_wxyz = goal_pose
        return goal_pos_w, goal_quat_wxyz, np.asarray(object_size, dtype=np.float32)

    def _get_future_goal_box_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not hasattr(motion_cmd, "motion") or not bool(getattr(motion_cmd.motion, "has_object", False)):
            return None

        object_size = self._resolve_target_box_dimensions()
        if object_size is None:
            return None
        object_size = np.asarray(object_size, dtype=np.float32)

        try:
            sample_fn = getattr(motion_cmd, "_sample_clip_based_object_goal_pose_w", None)
            if callable(sample_fn):
                env_ids = torch.tensor([self._env_id], device=motion_cmd.device, dtype=torch.long)
                clip_id = int(motion_cmd.clip_ids[self._env_id].item())
                clip_lengths = motion_cmd.motion.clip_lengths[clip_id : clip_id + 1].to(device=motion_cmd.device)
                goal_pos_t, goal_quat_xyzw_t = sample_fn(env_ids, clip_lengths)
                goal_pos_w = goal_pos_t[0].detach().float().cpu().numpy().astype(np.float32, copy=True)
                goal_quat_xyzw = goal_quat_xyzw_t[0].detach().float().cpu().numpy()
                goal_quat_wxyz = goal_quat_xyzw[[3, 0, 1, 2]]
                return goal_pos_w, goal_quat_wxyz, object_size
        except Exception:
            pass

        return None

    def _resolve_target_box_mesh(
        self,
    ) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
        visual_key, mesh, _source_label = self._resolve_object_mesh_spec_for_env(self._env_id, "visual")
        if not visual_key or mesh is None:
            return None, None, None
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)
        except Exception:
            return None, None, None
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
            return None, None, None
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            return None, None, None
        return str(visual_key), vertices, faces

    def _update_target_box(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_target_box:
            if self._target_box_handle is not None:
                try:
                    self._target_box_handle.visible = False
                except Exception:
                    pass
            return

        target_pose = self._get_effective_target_box_pose()
        if target_pose is None:
            if self._target_box_handle is not None:
                try:
                    self._target_box_handle.visible = False
                except Exception:
                    pass
            return

        goal_pos_w, goal_quat_wxyz, goal_size = target_pose
        dimensions = tuple(float(max(1.0e-3, v)) for v in np.asarray(goal_size, dtype=np.float32).reshape(3))
        mesh_key, mesh_vertices, mesh_faces = self._resolve_target_box_mesh()
        geometry_changed = self._target_box_last_dimensions != dimensions or self._target_box_last_mesh_key != mesh_key
        if self._target_box_handle is not None and geometry_changed:
            try:
                self._target_box_handle.remove()
            except Exception:
                pass
            self._target_box_handle = None
            self._target_box_last_dimensions = None
            self._target_box_last_mesh_key = None
        if self._target_box_handle is None:
            if mesh_key is not None and mesh_vertices is not None and mesh_faces is not None:
                self._target_box_handle = self._server.scene.add_mesh_simple(
                    self._scene_path("/target_box"),
                    vertices=mesh_vertices,
                    faces=mesh_faces,
                    color=TARGET_BOX_COLOR,
                    opacity=0.25,
                    flat_shading=True,
                    side="double",
                    wxyz=goal_quat_wxyz,
                    position=goal_pos_w - offset,
                    visible=True,
                )
            else:
                try:
                    self._target_box_handle = self._server.scene.add_box(
                        self._scene_path("/target_box"),
                        color=TARGET_BOX_COLOR,
                        dimensions=dimensions,
                        wireframe=False,
                        opacity=0.25,
                        flat_shading=True,
                        cast_shadow=False,
                        receive_shadow=False,
                        wxyz=goal_quat_wxyz,
                        position=goal_pos_w - offset,
                        visible=True,
                    )
                except TypeError:
                    self._target_box_handle = self._server.scene.add_box(
                        self._scene_path("/target_box"),
                        color=TARGET_BOX_COLOR,
                        dimensions=dimensions,
                        wxyz=goal_quat_wxyz,
                        position=goal_pos_w - offset,
                        visible=True,
                    )
            self._target_box_last_dimensions = dimensions
            self._target_box_last_mesh_key = mesh_key
        else:
            try:
                self._target_box_handle.visible = True
            except Exception:
                pass
            self._target_box_handle.position = goal_pos_w - offset
            self._target_box_handle.wxyz = goal_quat_wxyz

    def _update_future_goal_box(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if not self._show_future_goal_box:
            if self._future_goal_box_handle is not None:
                try:
                    self._future_goal_box_handle.visible = False
                except Exception:
                    pass
            return

        future_goal_pose = self._get_future_goal_box_pose()
        if future_goal_pose is None:
            if self._future_goal_box_handle is not None:
                try:
                    self._future_goal_box_handle.visible = False
                except Exception:
                    pass
            return

        goal_pos_w, goal_quat_wxyz, goal_size = future_goal_pose
        dimensions = tuple(float(max(1.0e-3, v)) for v in np.asarray(goal_size, dtype=np.float32).reshape(3))
        mesh_key, mesh_vertices, mesh_faces = self._resolve_target_box_mesh()
        geometry_changed = self._future_goal_box_last_dimensions != dimensions or self._future_goal_box_last_mesh_key != mesh_key
        if self._future_goal_box_handle is not None and geometry_changed:
            try:
                self._future_goal_box_handle.remove()
            except Exception:
                pass
            self._future_goal_box_handle = None
            self._future_goal_box_last_dimensions = None
            self._future_goal_box_last_mesh_key = None
        if self._future_goal_box_handle is None:
            if mesh_key is not None and mesh_vertices is not None and mesh_faces is not None:
                self._future_goal_box_handle = self._server.scene.add_mesh_simple(
                    self._scene_path("/future_goal_box"),
                    vertices=mesh_vertices,
                    faces=mesh_faces,
                    color=FUTURE_GOAL_BOX_COLOR,
                    opacity=0.28,
                    flat_shading=True,
                    side="double",
                    wxyz=goal_quat_wxyz,
                    position=goal_pos_w - offset,
                    visible=True,
                )
            else:
                try:
                    self._future_goal_box_handle = self._server.scene.add_box(
                        self._scene_path("/future_goal_box"),
                        color=FUTURE_GOAL_BOX_COLOR,
                        dimensions=dimensions,
                        wireframe=False,
                        opacity=0.28,
                        flat_shading=True,
                        cast_shadow=False,
                        receive_shadow=False,
                        wxyz=goal_quat_wxyz,
                        position=goal_pos_w - offset,
                        visible=True,
                    )
                except TypeError:
                    self._future_goal_box_handle = self._server.scene.add_box(
                        self._scene_path("/future_goal_box"),
                        color=FUTURE_GOAL_BOX_COLOR,
                        dimensions=dimensions,
                        wxyz=goal_quat_wxyz,
                        position=goal_pos_w - offset,
                        visible=True,
                    )
            self._future_goal_box_last_dimensions = dimensions
            self._future_goal_box_last_mesh_key = mesh_key
        else:
            try:
                self._future_goal_box_handle.visible = True
            except Exception:
                pass
            self._future_goal_box_handle.position = goal_pos_w - offset
            self._future_goal_box_handle.wxyz = goal_quat_wxyz

    def _ensure_marker_handle(
        self,
        name: str,
        color: tuple[int, int, int],
        radius: float,
    ) -> tuple[object | None, bool]:
        if self._server is None:
            return None, False
        name = self._scene_path(name)
        mesh = _make_marker_mesh(color, radius)
        if mesh is None:
            handle = self._server.scene.add_point_cloud(
                name,
                points=np.zeros((1, 3), dtype=np.float32),
                colors=np.array([color], dtype=np.uint8),
                point_size=max(0.005, radius * 2.0),
                point_shape="circle",
                precision="float32",
            )
            return handle, True
        try:
            handle = self._server.scene.add_mesh_trimesh(
                name,
                mesh,
                cast_shadow=False,
                receive_shadow=False,
            )
        except TypeError:
            # Older viser builds do not support shadow-control kwargs here.
            handle = self._server.scene.add_mesh_trimesh(
                name,
                mesh,
            )
        return handle, False

    def _update_perception_markers(
        self,
        perception_mgr: Any,
        env_ids: torch.Tensor,
        offset: np.ndarray,
    ) -> None:
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return

        if getattr(cfg, "heightmap_body_name", None):
            show = True if self._perception_show_heightmap_joint_cb is None else bool(
                self._perception_show_heightmap_joint_cb.value
            )
            if show:
                try:
                    pose = perception_mgr.get_heightmap_pose(
                        env_ids,
                        apply_offsets=False,
                        apply_heading_only=False,
                    )
                except Exception:
                    pose = None
                if pose is not None:
                    pos_t, _ = pose
                    pos = pos_t[0].detach().cpu().numpy() - offset
                    if self._heightmap_marker_handle is None:
                        handle, is_pc = self._ensure_marker_handle(
                            "/heightmap_joint_marker",
                            HEIGHTMAP_MARKER_COLOR,
                            SENSOR_MARKER_RADIUS,
                        )
                        self._heightmap_marker_handle = handle
                        self._heightmap_marker_is_pc = is_pc
                    if self._heightmap_marker_handle is not None:
                        self._heightmap_marker_handle.visible = True
                        if self._heightmap_marker_is_pc:
                            self._heightmap_marker_handle.points = pos.reshape(1, 3).astype(np.float32)
                        else:
                            self._heightmap_marker_handle.position = pos
            elif self._heightmap_marker_handle is not None:
                self._heightmap_marker_handle.visible = False

        if getattr(cfg, "camera_body_name", None) and getattr(cfg, "output_mode", "") == "camera_depth":
            show = True if self._perception_show_camera_joint_cb is None else bool(
                self._perception_show_camera_joint_cb.value
            )
            if show:
                try:
                    cam_pos_t, _ = perception_mgr.get_camera_pose(
                        env_ids,
                        apply_sensor_offset=False,
                        apply_pitch=False,
                    )
                except Exception:
                    cam_pos_t = None
                if cam_pos_t is not None:
                    pos = cam_pos_t[0].detach().cpu().numpy() - offset
                    if self._camera_marker_handle is None:
                        handle, is_pc = self._ensure_marker_handle(
                            "/camera_joint_marker",
                            CAMERA_MARKER_COLOR,
                            SENSOR_MARKER_RADIUS,
                        )
                        self._camera_marker_handle = handle
                        self._camera_marker_is_pc = is_pc
                    if self._camera_marker_handle is not None:
                        self._camera_marker_handle.visible = True
                        if self._camera_marker_is_pc:
                            self._camera_marker_handle.points = pos.reshape(1, 3).astype(np.float32)
                        else:
                            self._camera_marker_handle.position = pos
            elif self._camera_marker_handle is not None:
                self._camera_marker_handle.visible = False

    def _update_contact_forces(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if self._contact_force_cb is None or not bool(self._contact_force_cb.value):
            if self._contact_force_handle is not None:
                self._contact_force_handle.visible = False
            return
        sim = getattr(self._env, "simulator", None)
        if sim is None:
            return
        forces = getattr(sim, "contact_forces", None)
        positions = getattr(sim, "_rigid_body_pos", None)
        if forces is None or positions is None:
            return
        if not isinstance(forces, torch.Tensor) or not isinstance(positions, torch.Tensor):
            return
        if forces.numel() == 0 or positions.numel() == 0:
            return
        if self._env_id < 0 or self._env_id >= forces.shape[0]:
            return

        forces_env = forces[self._env_id]
        positions_env = positions[self._env_id]
        body_count = min(forces_env.shape[0], positions_env.shape[0])
        if body_count == 0:
            return

        forces_env = forces_env[:body_count]
        positions_env = positions_env[:body_count]

        magnitudes = torch.linalg.norm(forces_env, dim=-1)
        threshold = float(getattr(self._env.simulator.simulator_config, "contact_force_viz_threshold", 1.0))
        mask = magnitudes > threshold
        if torch.count_nonzero(mask) == 0:
            if self._contact_force_handle is not None:
                self._contact_force_handle.visible = False
            return

        scale = float(getattr(self._env.simulator.simulator_config, "contact_force_viz_scale", 0.001))
        forces_np = forces_env.detach().cpu().numpy()
        positions_np = positions_env.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        positions_np = positions_np[mask_np]
        forces_np = forces_np[mask_np]
        magnitudes_np = np.linalg.norm(forces_np, axis=1, keepdims=True)
        directions = np.zeros_like(forces_np)
        nonzero = magnitudes_np > 1.0e-6
        directions[nonzero[:, 0]] = forces_np[nonzero[:, 0]] / magnitudes_np[nonzero[:, 0]]
        arrow_ends = positions_np + directions * magnitudes_np * scale

        points = np.stack([positions_np, arrow_ends], axis=1)
        if self._recenter:
            points = points - offset[None, None, :]
        colors = np.full((points.shape[0], 2, 3), [255, 0, 0], dtype=np.uint8)

        if self._contact_force_handle is None:
            self._contact_force_handle = self._server.scene.add_line_segments(
                self._scene_path("/contact_force_arrows"),
                points=points,
                colors=colors,
                line_width=2.0,
            )
        else:
            self._contact_force_handle.visible = True
            self._contact_force_handle.points = points
            self._contact_force_handle.colors = colors


def _normalize_env_ids(env_ids) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        return [int(idx.item()) for idx in env_ids.flatten()]
    if isinstance(env_ids, np.ndarray):
        return [int(idx) for idx in env_ids.flatten().tolist()]
    return [int(idx) for idx in env_ids]
