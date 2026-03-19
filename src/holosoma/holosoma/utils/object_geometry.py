from __future__ import annotations

import functools
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from holosoma.utils.path import resolve_data_file_path


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


def _normalize_urdf_path(urdf_path: str | Path) -> Path:
    raw = str(urdf_path)
    if raw.startswith("@holosoma/") or raw.startswith("holosoma/"):
        return Path(resolve_data_file_path(raw)).resolve()
    return Path(raw).expanduser().resolve()


@functools.lru_cache(maxsize=128)
def load_urdf_geometry_extents(urdf_path: str | Path) -> tuple[float, float, float] | None:
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        resolved_urdf = _normalize_urdf_path(urdf_path)
        root = ET.parse(resolved_urdf).getroot()
    except Exception:
        return None

    meshes: list["trimesh.Trimesh"] = []
    for geom_tag in ("collision", "visual"):
        for geom_parent in root.findall(f".//link/{geom_tag}"):
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

    min_corner = np.min(np.stack([mesh.bounds[0] for mesh in meshes], axis=0), axis=0)
    max_corner = np.max(np.stack([mesh.bounds[1] for mesh in meshes], axis=0), axis=0)
    extents = np.maximum(max_corner - min_corner, 1.0e-4)
    return float(extents[0]), float(extents[1]), float(extents[2])
