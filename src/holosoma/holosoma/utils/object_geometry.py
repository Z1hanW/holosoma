from __future__ import annotations

import functools
import math
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from holosoma.utils.path import resolve_data_file_path

_LARGEBOX_BEST_IOU_EXTENTS = (
    0.3249185391136043,
    0.31860981675930306,
    0.326778873323969,
)
_LARGEBOX_BEST_IOU_CENTER_OFFSET = (
    0.000861508081686495,
    -0.0004369894302321542,
    -0.0025913614928369986,
)


def get_largebox_best_iou_primitive_extents() -> tuple[float, float, float]:
    return _LARGEBOX_BEST_IOU_EXTENTS


def get_largebox_best_iou_primitive_center_offset() -> tuple[float, float, float]:
    return _LARGEBOX_BEST_IOU_CENTER_OFFSET


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
    return Path(resolve_data_file_path(raw)).expanduser().resolve()


def _parse_float(raw: str | None, default: float) -> float:
    try:
        return float(raw) if raw is not None else float(default)
    except Exception:
        return float(default)


def _is_identity_origin(origin_el: ET.Element | None, *, atol: float = 1.0e-6) -> bool:
    if origin_el is None:
        return True
    xyz = _parse_vec3(origin_el.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec3(origin_el.get("rpy"), (0.0, 0.0, 0.0))
    return bool(np.all(np.abs(xyz) <= atol) and np.all(np.abs(rpy) <= atol))


def _build_geometry_meshes(
    resolved_urdf: Path,
    root: ET.Element,
    *,
    geom_tags: tuple[str, ...],
):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None

    meshes: list["trimesh.Trimesh"] = []
    for geom_tag in geom_tags:
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

    return meshes


def _load_urdf_geometry_extents_from_root(
    resolved_urdf: Path,
    root: ET.Element,
    *,
    geom_tags: tuple[str, ...],
) -> tuple[float, float, float] | None:
    meshes = _build_geometry_meshes(resolved_urdf, root, geom_tags=geom_tags)
    if not meshes:
        return None

    min_corner = np.min(np.stack([mesh.bounds[0] for mesh in meshes], axis=0), axis=0)
    max_corner = np.max(np.stack([mesh.bounds[1] for mesh in meshes], axis=0), axis=0)
    extents = np.maximum(max_corner - min_corner, 1.0e-4)
    return float(extents[0]), float(extents[1]), float(extents[2])


def _load_urdf_geometry_bbox_fill_ratio_from_root(
    resolved_urdf: Path,
    root: ET.Element,
    *,
    geom_tags: tuple[str, ...],
) -> float | None:
    """Return convex-hull volume divided by AABB volume for the selected geometry."""
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None

    meshes = _build_geometry_meshes(resolved_urdf, root, geom_tags=geom_tags)
    if not meshes:
        return None

    try:
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
        bbox_volume = float(np.prod(np.maximum(extents, 0.0)))
        if not math.isfinite(bbox_volume) or bbox_volume <= 1.0e-12:
            return None
        hull_volume = float(mesh.convex_hull.volume)
    except Exception:
        return None

    if not math.isfinite(hull_volume) or hull_volume <= 0.0:
        return None
    return hull_volume / bbox_volume


def _has_explicit_box_geometry(root: ET.Element, *, geom_tags: tuple[str, ...]) -> bool:
    for geom_tag in geom_tags:
        for geom_parent in root.findall(f".//link/{geom_tag}"):
            geometry_el = geom_parent.find("geometry")
            if geometry_el is not None and geometry_el.find("box") is not None:
                return True
    return False


def _geometry_extents_close(
    first: tuple[float, float, float] | None,
    second: tuple[float, float, float] | None,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-4,
) -> bool:
    if first is None or second is None:
        return True
    return bool(np.allclose(np.asarray(first), np.asarray(second), atol=atol, rtol=rtol))


@dataclass(frozen=True)
class UrdfBoxPrimitiveMetadata:
    """Subset of URDF properties needed to preserve simple box behavior with a cuboid primitive."""

    extents: tuple[float, float, float]
    center_offset: tuple[float, float, float]
    mass: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    compliant_contact_stiffness: float
    compliant_contact_damping: float
    visual_color: tuple[float, float, float] | None


@functools.lru_cache(maxsize=128)
def load_urdf_geometry_extents(urdf_path: str | Path) -> tuple[float, float, float] | None:
    try:
        resolved_urdf = _normalize_urdf_path(urdf_path)
        root = ET.parse(resolved_urdf).getroot()
    except Exception:
        return None
    return _load_urdf_geometry_extents_from_root(resolved_urdf, root, geom_tags=("collision", "visual"))


@functools.lru_cache(maxsize=128)
def load_urdf_box_primitive_metadata(urdf_path: str | Path) -> UrdfBoxPrimitiveMetadata | None:
    """Resolve cuboid extents for simple box-like URDFs.

    This is intentionally conservative. It only accepts single-link, joint-free URDFs whose geometry is
    simple enough that replacing the imported mesh with a cuboid primitive should preserve the intended
    training object semantics.
    """

    try:
        resolved_urdf = _normalize_urdf_path(urdf_path)
        root = ET.parse(resolved_urdf).getroot()
    except Exception:
        return None

    links = root.findall("link")
    joints = root.findall("joint")
    if len(links) != 1 or joints:
        return None

    link = links[0]
    visuals = link.findall("visual")
    collisions = link.findall("collision")
    if len(visuals) > 1 or len(collisions) > 1:
        return None
    if not visuals and not collisions:
        return None

    has_explicit_box_geom = False
    has_only_mesh_or_box_geom = True
    for geom_tag in ("visual", "collision"):
        for geom_parent in link.findall(geom_tag):
            if not _is_identity_origin(geom_parent.find("origin")):
                return None
            geometry_el = geom_parent.find("geometry")
            if geometry_el is None:
                return None
            child_tags = [child.tag for child in geometry_el if isinstance(child.tag, str)]
            if any(tag not in {"mesh", "box"} for tag in child_tags):
                has_only_mesh_or_box_geom = False
                break
            if "box" in child_tags:
                has_explicit_box_geom = True
        if not has_only_mesh_or_box_geom:
            break
    if not has_only_mesh_or_box_geom:
        return None

    name_hints = " ".join(
        [
            resolved_urdf.stem.lower(),
            str(root.get("name", "")).strip().lower(),
            str(link.get("name", "")).strip().lower(),
        ]
    )
    if not has_explicit_box_geom and "box" not in name_hints:
        return None

    inertial_el = link.find("inertial")
    if inertial_el is None or not _is_identity_origin(inertial_el.find("origin")):
        return None
    mass_el = inertial_el.find("mass")
    inertia_el = inertial_el.find("inertia")
    if mass_el is None or inertia_el is None:
        return None
    if any(abs(_parse_float(inertia_el.get(attr), 0.0)) > 1.0e-6 for attr in ("ixy", "ixz", "iyz")):
        return None

    extents = _load_urdf_geometry_extents_from_root(
        resolved_urdf,
        root,
        geom_tags=("collision",) if collisions else ("visual",),
    )
    if extents is None:
        return None

    if not has_explicit_box_geom:
        visual_extents = _load_urdf_geometry_extents_from_root(resolved_urdf, root, geom_tags=("visual",))
        collision_extents = _load_urdf_geometry_extents_from_root(resolved_urdf, root, geom_tags=("collision",))
        if not _geometry_extents_close(visual_extents, collision_extents):
            return None

        for geom_tag in ("visual", "collision"):
            if not root.findall(f".//link/{geom_tag}"):
                continue
            if _has_explicit_box_geometry(root, geom_tags=(geom_tag,)):
                continue
            fill_ratio = _load_urdf_geometry_bbox_fill_ratio_from_root(
                resolved_urdf,
                root,
                geom_tags=(geom_tag,),
            )
            if fill_ratio is None or fill_ratio < 0.95:
                return None

    mass = _parse_float(mass_el.get("value"), -1.0)
    if not math.isfinite(mass) or mass <= 0.0:
        return None

    visual_color: tuple[float, float, float] | None = None
    if visuals:
        material_el = visuals[0].find("material")
        color_el = material_el.find("color") if material_el is not None else None
        if color_el is not None:
            rgba_raw = [part for part in str(color_el.get("rgba", "")).replace(",", " ").split() if part]
            if len(rgba_raw) >= 3:
                try:
                    visual_color = (
                        float(rgba_raw[0]),
                        float(rgba_raw[1]),
                        float(rgba_raw[2]),
                    )
                except Exception:
                    visual_color = None

    contact_el = link.find("contact")
    root_dynamics_el = root.find("dynamics")

    lateral_friction_el = contact_el.find("lateral_friction") if contact_el is not None else None
    restitution_el = contact_el.find("restitution") if contact_el is not None else None
    stiffness_el = contact_el.find("stiffness") if contact_el is not None else None
    damping_el = contact_el.find("damping") if contact_el is not None else None

    if lateral_friction_el is not None:
        lateral_friction = _parse_float(lateral_friction_el.get("value"), 0.5)
    elif root_dynamics_el is not None:
        lateral_friction = _parse_float(root_dynamics_el.get("friction"), 0.5)
    else:
        lateral_friction = 0.5

    restitution = _parse_float(restitution_el.get("value") if restitution_el is not None else None, 0.0)
    compliant_contact_stiffness = _parse_float(stiffness_el.get("value") if stiffness_el is not None else None, 0.0)
    compliant_contact_damping = _parse_float(damping_el.get("value") if damping_el is not None else None, 0.0)

    return UrdfBoxPrimitiveMetadata(
        extents=extents,
        center_offset=(0.0, 0.0, 0.0),
        mass=mass,
        static_friction=lateral_friction,
        dynamic_friction=lateral_friction,
        restitution=restitution,
        compliant_contact_stiffness=compliant_contact_stiffness,
        compliant_contact_damping=compliant_contact_damping,
        visual_color=visual_color,
    )


@functools.lru_cache(maxsize=128)
def load_urdf_box_primitive_extents(urdf_path: str | Path) -> tuple[float, float, float] | None:
    metadata = load_urdf_box_primitive_metadata(urdf_path)
    return None if metadata is None else metadata.extents
