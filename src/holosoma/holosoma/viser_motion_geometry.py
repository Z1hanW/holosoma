from __future__ import annotations

import os
import sys
import threading
import time
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.utils.viser_utils import ensure_viser_on_path  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402
from holosoma.utils.viser_utils import resolve_viser_port  # noqa: E402


FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


@dataclass(frozen=True)
class MotionGeometryViewerConfig:
    motion_dir: str
    overlay_motion_dir: str = ""
    geometry_dir: str = ""
    object_urdf: str = ""
    object_urdf_dir: str = ""
    object_urdf_mode: str = "stem"
    robot: str = "g1_29dof"
    port: int = 0
    fps: int | None = None
    autoplay: bool = True
    loop: bool = True
    preload: bool = True
    show_meshes: bool = True
    overlay_show_meshes: bool = True
    overlay_tint_rgba: str = "0.15,0.75,1.00,0.55"
    overlay_sync_mode: str = "normalized"  # normalized | time
    show_geometry: bool = True
    show_object: bool = True
    add_grid: bool = True
    grid_size: float = 10.0
    start_clip: str | None = None
    object_filter_csv: str = ""


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _resolve_robot_config(name: str) -> RobotConfig:
    defaults = robot_values.DEFAULTS
    if name not in defaults:
        raise ValueError(f"Unknown robot '{name}'. Available: {sorted(defaults.keys())}")
    return defaults[name]


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> Path:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    return _resolve_data_path(os.path.join(str(asset_root), robot_config.asset.urdf_file))


def _list_pairs(
    motion_dir: Path,
    geometry_dir: Path | None,
    object_urdf_dir: Path | None,
    object_urdf_mode: str = "stem",
) -> tuple[list[str], dict[str, Path], dict[str, Path], bool, dict[str, Path], bool]:
    motion_paths = sorted(list(motion_dir.glob("*.npz")) + list(motion_dir.glob("*.NPZ")))
    motion_map = {path.stem: path for path in motion_paths}
    pair_names = sorted(motion_map)
    geom_map: dict[str, Path] = {}
    object_map: dict[str, Path] = {}
    geometry_available = False
    object_available = False

    if geometry_dir is not None:
        geom_paths = sorted(list(geometry_dir.glob("*.obj")) + list(geometry_dir.glob("*.OBJ")))
        geom_map = {path.stem: path for path in geom_paths}
        shared = sorted(set(pair_names) & set(geom_map))
        if not shared:
            logger.warning(
                "No matching motion/geometry pairs found. Falling back to ground-only. motions=%d geometry=%d",
                len(motion_paths),
                len(geom_paths),
            )
        else:
            geometry_available = True
            pair_names = shared
            missing_geom = sorted(set(motion_map) - set(geom_map))
            missing_motion = sorted(set(geom_map) - set(motion_map))
            if missing_geom:
                logger.warning("No geometry for motions: {}", missing_geom[:10])
            if missing_motion:
                logger.warning("No motion for geometry: {}", missing_motion[:10])

    if object_urdf_dir is not None:
        mode = (object_urdf_mode or "stem").strip().lower()
        recursive = mode in {"recursive", "behave"}
        if recursive:
            urdf_paths = sorted(list(object_urdf_dir.rglob("*.urdf")) + list(object_urdf_dir.rglob("*.URDF")))
        else:
            urdf_paths = sorted(list(object_urdf_dir.glob("*.urdf")) + list(object_urdf_dir.glob("*.URDF")))

        if mode == "behave":
            object_by_name = {path.stem.lower(): path for path in urdf_paths}
            for clip_name in pair_names:
                parts = clip_name.split("_")
                if len(parts) > 2:
                    obj_key = parts[2].lower()
                else:
                    obj_key = clip_name.lower()
                urdf_path = object_by_name.get(obj_key)
                if urdf_path is not None:
                    object_map[clip_name] = urdf_path
            shared_obj = sorted(object_map.keys())
        else:
            object_map = {path.stem: path for path in urdf_paths}
            shared_obj = sorted(set(pair_names) & set(object_map))

        if not shared_obj:
            logger.warning(
                "No matching motion/object URDF pairs found. Disabling object URDF. motions=%d urdf=%d",
                len(motion_paths),
                len(urdf_paths),
            )
        else:
            object_available = True
            pair_names = shared_obj
            missing_obj = sorted(set(motion_map) - set(object_map))
            missing_motion_obj = sorted(set(object_map) - set(motion_map))
            if missing_obj:
                logger.warning("No object URDF for motions: {}", missing_obj[:10])
            if missing_motion_obj:
                logger.warning("No motion for object URDF: {}", missing_motion_obj[:10])

    return pair_names, motion_map, geom_map, geometry_available, object_map, object_available


def _load_object_spec_map(object_spec_path: Path) -> dict[str, Path]:
    payload = json.loads(object_spec_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        payload = payload["clips"]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid object spec json: {object_spec_path}")

    object_map: dict[str, Path] = {}
    for clip_name, entry in payload.items():
        if not isinstance(clip_name, str):
            continue

        if isinstance(entry, str):
            raw_urdf = entry.strip()
        elif isinstance(entry, dict):
            raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        else:
            continue

        if not raw_urdf:
            continue

        urdf_path = Path(raw_urdf)
        if not urdf_path.is_absolute():
            urdf_path = (object_spec_path.parent / urdf_path).resolve()
        if urdf_path.exists():
            object_map[clip_name] = urdf_path

    return object_map


def _restrict_pairs_to_object_map(
    pair_names: list[str],
    motion_map: dict[str, Path],
    object_map: dict[str, Path],
) -> tuple[list[str], bool]:
    shared_obj = sorted(set(pair_names) & set(object_map))
    if not shared_obj:
        logger.warning(
            "No matching motion/object URDF pairs found from JSON map. Disabling object URDF. motions=%d urdf=%d",
            len(motion_map),
            len(object_map),
        )
        return pair_names, False

    missing_obj = sorted(set(motion_map) - set(object_map))
    missing_motion_obj = sorted(set(object_map) - set(motion_map))
    if missing_obj:
        logger.warning("No object URDF for motions: {}", missing_obj[:10])
    if missing_motion_obj:
        logger.warning("No motion for object URDF map entries: {}", missing_motion_obj[:10])
    return shared_obj, True


def _try_load_qpos_npz(path: Path) -> tuple[np.ndarray, int] | None:
    if not path.exists() or path.suffix.lower() != ".npz":
        return None
    with np.load(path, allow_pickle=True) as data:
        if "qpos" not in data:
            return None
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        fps_val = data.get("fps", 30)
        fps = int(np.array(fps_val).reshape(-1)[0]) if fps_val is not None else 30
        return qpos, fps


def _filter_pair_names(pair_names: list[str], object_filter_csv: str) -> list[str]:
    terms = [s.strip().lower() for s in object_filter_csv.split(",") if s.strip()]
    if not terms:
        return pair_names
    filtered = [name for name in pair_names if any(term in name.lower() for term in terms)]
    if not filtered:
        raise ValueError(f"No clips match object filter: {terms}")
    return filtered


def _quat_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion array that is already stored as wxyz."""
    q = np.asarray(quat, dtype=np.float32)
    if q.ndim < 2 or q.shape[-1] != 4:
        raise ValueError(f"Unsupported quaternion shape: {q.shape}")

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    return (q / norm).astype(np.float32, copy=False)


def _load_motion_qpos(
    motion_path: Path,
    robot_config: RobotConfig,
    viser_joint_names: list[str],
) -> tuple[np.ndarray, int]:
    qpos_payload = _try_load_qpos_npz(motion_path)
    if qpos_payload is not None:
        return qpos_payload

    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

    with np.load(motion_path, allow_pickle=True) as data:
        required_keys = ("joint_names", "joint_pos", "body_names", "body_pos_w", "body_quat_w")
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Motion file missing keys required for viewer fallback: {missing_keys}")

        motion_joint_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in data["joint_names"]
        ]
        motion_body_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in data["body_names"]
        ]
        joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float32)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w_xyzw = np.asarray(data["body_quat_w"], dtype=np.float32)

        if joint_pos_raw.ndim != 2:
            raise ValueError(f"Unsupported joint_pos shape in {motion_path}: {joint_pos_raw.shape}")
        if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
            raise ValueError(f"Unsupported body_pos_w shape in {motion_path}: {body_pos_w.shape}")
        if body_quat_w_xyzw.ndim != 3 or body_quat_w_xyzw.shape[2] != 4:
            raise ValueError(f"Unsupported body_quat_w shape in {motion_path}: {body_quat_w_xyzw.shape}")

        motion_joint_count = len(motion_joint_names)
        # Training-ready format often stores joint_pos as [root_xyz, root_quat, dof...].
        has_root_in_joint_pos = joint_pos_raw.shape[1] >= (7 + motion_joint_count)
        if has_root_in_joint_pos:
            root_pos = joint_pos_raw[:, :3]
            root_quat_wxyz = _quat_to_wxyz(joint_pos_raw[:, 3:7])
            joint_pos_source = joint_pos_raw[:, 7 : 7 + motion_joint_count]
        else:
            joint_pos_source = joint_pos_raw

        motion_joint_map = {name: idx for idx, name in enumerate(motion_joint_names)}
        joint_pos_robot = np.zeros((joint_pos_source.shape[0], len(robot_config.dof_names)), dtype=np.float32)
        missing_robot_joints = []
        for robot_idx, joint_name in enumerate(robot_config.dof_names):
            motion_idx = motion_joint_map.get(joint_name)
            if motion_idx is None:
                missing_robot_joints.append(joint_name)
                continue
            joint_pos_robot[:, robot_idx] = joint_pos_source[:, motion_idx]
        if missing_robot_joints:
            logger.warning("Missing joints in motion file {}; zero-filling: {}", motion_path, missing_robot_joints)
        joint_pos = joint_pos_robot[:, joint_order]

        if not has_root_in_joint_pos:
            motion_body_map = {name: idx for idx, name in enumerate(motion_body_names)}
            root_body_name = FAKE_BODY_NAME_ALIASES.get(robot_config.body_names[0], robot_config.body_names[0])
            if root_body_name not in motion_body_map:
                raise ValueError(f"Root body '{root_body_name}' not found in motion file: {motion_path}")
            root_idx = motion_body_map[root_body_name]
            root_pos = body_pos_w[:, root_idx]
            root_quat_wxyz = _quat_to_wxyz(body_quat_w_xyzw[:, root_idx])

        qpos_parts: list[np.ndarray] = [root_pos, root_quat_wxyz, joint_pos]
        if "object_pos_w" in data and "object_quat_w" in data:
            object_pos = np.asarray(data["object_pos_w"], dtype=np.float32)
            object_quat_wxyz = _quat_to_wxyz(np.asarray(data["object_quat_w"], dtype=np.float32))
            qpos_parts.extend([object_pos, object_quat_wxyz])

        qpos = np.concatenate(qpos_parts, axis=1).astype(np.float32, copy=False)
        fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 30
    return qpos, fps


def _load_obj_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded geometry is not a trimesh: {type(mesh)}")
    return mesh


def _parse_rgba(rgba_csv: str) -> tuple[float, float, float, float]:
    values = [item.strip() for item in rgba_csv.split(",") if item.strip()]
    if len(values) != 4:
        raise ValueError(f"overlay_tint_rgba must have 4 comma-separated values, got: {rgba_csv!r}")
    rgba = tuple(float(v) for v in values)
    for channel in rgba:
        if channel < 0.0 or channel > 1.0:
            raise ValueError(f"overlay_tint_rgba channel out of range [0,1]: {rgba_csv!r}")
    return rgba  # type: ignore[return-value]


def _make_tinted_urdf(source_urdf: Path, rgba: tuple[float, float, float, float]) -> Path:
    rgba_text = " ".join(f"{v:.6f}" for v in rgba)
    tree = ET.parse(source_urdf)
    root = tree.getroot()
    source_dir = source_urdf.parent
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        # Keep package:// URIs unchanged; convert relative file paths to absolute.
        if filename.startswith("package://") or os.path.isabs(filename):
            continue
        mesh.set("filename", str((source_dir / filename).resolve()))
    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is None:
            material = ET.SubElement(visual, "material", {"name": "overlay_tint"})
        texture = material.find("texture")
        if texture is not None:
            material.remove(texture)
        color = material.find("color")
        if color is None:
            color = ET.SubElement(material, "color")
        color.set("rgba", rgba_text)

    fd, tmp_path = tempfile.mkstemp(prefix="viser_overlay_", suffix=".urdf")
    os.close(fd)
    tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    return Path(tmp_path)


def run_viewer(cfg: MotionGeometryViewerConfig) -> None:
    motion_dir = _resolve_data_path(cfg.motion_dir)
    overlay_motion_dir = None
    overlay_motion_dir_raw = (cfg.overlay_motion_dir or "").strip()
    if overlay_motion_dir_raw:
        overlay_motion_dir = _resolve_data_path(overlay_motion_dir_raw)
    geometry_dir = None
    geometry_dir_raw = (cfg.geometry_dir or "").strip()
    if geometry_dir_raw:
        geometry_dir = _resolve_data_path(geometry_dir_raw)
    object_urdf_path = None
    object_urdf_raw = (cfg.object_urdf or "").strip()
    if object_urdf_raw:
        object_urdf_path = _resolve_data_path(object_urdf_raw)
    object_dir = None
    object_dir_raw = (cfg.object_urdf_dir or "").strip()
    if object_dir_raw:
        object_dir = _resolve_data_path(object_dir_raw)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"Motion dir not found: {motion_dir}")
    if overlay_motion_dir is not None and not overlay_motion_dir.is_dir():
        raise FileNotFoundError(f"Overlay motion dir not found: {overlay_motion_dir}")
    if geometry_dir is not None and not geometry_dir.is_dir():
        logger.warning("Geometry dir not found ({}); falling back to ground-only.", geometry_dir)
        geometry_dir = None
    if object_urdf_path is not None and not object_urdf_path.exists():
        logger.warning("Object URDF not found ({}); disabling object URDF.", object_urdf_path)
        object_urdf_path = None
    if object_dir is not None and not object_dir.is_dir():
        logger.warning("Object URDF dir not found ({}); disabling object URDF.", object_dir)
        object_dir = None

    object_map_from_spec: dict[str, Path] | None = None
    if object_urdf_path is not None and object_urdf_path.suffix.lower() == ".json":
        object_map_from_spec = _load_object_spec_map(object_urdf_path)
        if not object_map_from_spec:
            logger.warning("Object URDF map is empty or unresolved ({}); disabling object URDF.", object_urdf_path)
        else:
            logger.info("Loaded clip-object URDF map '{}' ({} entries).", object_urdf_path, len(object_map_from_spec))
        object_urdf_path = None

    if object_urdf_path is None and object_dir is not None:
        urdf_paths = sorted(list(object_dir.glob("*.urdf")) + list(object_dir.glob("*.URDF")))
        if len(urdf_paths) == 1:
            object_urdf_path = urdf_paths[0]
            object_dir = None
            logger.info("Using single object URDF for all clips: {}", object_urdf_path)

    if object_map_from_spec is not None:
        pair_names, motion_map, geom_map, geometry_available, object_map, object_available = _list_pairs(
            motion_dir, geometry_dir, None
        )
        object_map = object_map_from_spec
        pair_names, object_available = _restrict_pairs_to_object_map(pair_names, motion_map, object_map)
    elif object_urdf_path is not None:
        pair_names, motion_map, geom_map, geometry_available, object_map, object_available = _list_pairs(
            motion_dir, geometry_dir, None
        )
        object_available = True
    else:
        pair_names, motion_map, geom_map, geometry_available, object_map, object_available = _list_pairs(
            motion_dir, geometry_dir, object_dir, cfg.object_urdf_mode
        )

    pair_count_before_filter = len(pair_names)
    pair_names = _filter_pair_names(pair_names, cfg.object_filter_csv)
    if cfg.object_filter_csv.strip():
        logger.info(
            "Applied object filter '{}': {} -> {} clips",
            cfg.object_filter_csv,
            pair_count_before_filter,
            len(pair_names),
        )

    overlay_motion_map: dict[str, Path] = {}
    overlay_enabled = overlay_motion_dir is not None
    if overlay_enabled:
        overlay_paths = sorted(list(overlay_motion_dir.glob("*.npz")) + list(overlay_motion_dir.glob("*.NPZ")))
        overlay_motion_map = {path.stem: path for path in overlay_paths}
        if not overlay_motion_map:
            raise ValueError(f"No motion files found in overlay_motion_dir: {overlay_motion_dir}")
        shared = sorted(set(pair_names) & set(overlay_motion_map))
        if not shared:
            raise ValueError(
                "No shared clip names between motion_dir and overlay_motion_dir. "
                f"motion_dir={motion_dir} overlay_motion_dir={overlay_motion_dir}"
            )
        if len(shared) != len(pair_names):
            logger.warning("Overlay mode: {} clips shared out of {} base clips.", len(shared), len(pair_names))
        pair_names = shared

    if cfg.start_clip and cfg.start_clip not in pair_names:
        raise ValueError(f"start_clip '{cfg.start_clip}' not found in pairs.")

    robot_config = _resolve_robot_config(cfg.robot)
    urdf_path = _resolve_robot_urdf_path(robot_config)

    port = resolve_viser_port(cfg.port)
    server = viser.ViserServer(port=port)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    vr = ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/robot")
    vr.show_visual = cfg.show_meshes
    overlay_root: viser.FrameHandle | None = None
    overlay_vr: ViserUrdf | None = None
    overlay_sync_mode = (cfg.overlay_sync_mode or "normalized").strip().lower()
    if overlay_sync_mode not in {"normalized", "time"}:
        raise ValueError(f"overlay_sync_mode must be 'normalized' or 'time', got: {cfg.overlay_sync_mode!r}")
    if overlay_enabled:
        overlay_root = server.scene.add_frame("/robot_overlay", show_axes=False)
        overlay_tint = _parse_rgba(cfg.overlay_tint_rgba)
        tinted_urdf_path = _make_tinted_urdf(urdf_path, overlay_tint)
        overlay_vr = ViserUrdf(server, urdf_or_path=tinted_urdf_path, root_node_name="/robot_overlay")
        overlay_vr.show_visual = cfg.overlay_show_meshes
        logger.info(
            "Overlay enabled: overlay_motion_dir='{}' sync='{}' tint_rgba='{}'",
            overlay_motion_dir,
            overlay_sync_mode,
            cfg.overlay_tint_rgba,
        )
    object_state: dict[str, object | None] = {"name": None, "urdf": None, "frame": None, "single": None}

    viser_joint_names = list(vr.get_actuated_joint_names())
    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

    if cfg.add_grid:
        server.scene.add_grid(
            "/grid",
            width=cfg.grid_size,
            height=cfg.grid_size,
            position=(0.0, 0.0, 0.0),
        )
    if not geometry_available:
        logger.info("Geometry disabled: using ground-only view.")
    if not object_available:
        logger.info("Object URDF disabled.")

    motion_cache: dict[str, dict[str, object]] = {}
    overlay_motion_cache: dict[str, dict[str, object]] = {}
    geometry_cache: dict[str, trimesh.Trimesh] = {}
    object_cache: dict[str, ViserUrdf] = {}
    object_frame_cache: dict[str, viser.FrameHandle] = {}

    def _ensure_motion_loaded(name: str) -> dict[str, object]:
        if name in motion_cache:
            return motion_cache[name]
        qpos, fps = _load_motion_qpos(motion_map[name], robot_config, viser_joint_names)
        if qpos.shape[0] == 0:
            raise ValueError(f"Motion {name} has zero frames.")
        motion_cache[name] = {
            "qpos": qpos,
            "fps": int(fps),
            "n_frames": int(qpos.shape[0]),
        }
        return motion_cache[name]

    def _ensure_overlay_motion_loaded(name: str) -> dict[str, object]:
        if not overlay_enabled:
            raise RuntimeError("overlay not enabled")
        if name in overlay_motion_cache:
            return overlay_motion_cache[name]
        qpos, fps = _load_motion_qpos(overlay_motion_map[name], robot_config, viser_joint_names)
        if qpos.shape[0] == 0:
            raise ValueError(f"Overlay motion {name} has zero frames.")
        overlay_motion_cache[name] = {
            "qpos": qpos,
            "fps": int(fps),
            "n_frames": int(qpos.shape[0]),
        }
        return overlay_motion_cache[name]

    def _ensure_geometry_loaded(name: str) -> trimesh.Trimesh:
        if not geometry_available:
            raise FileNotFoundError("Geometry is not available for this viewer session.")
        if name in geometry_cache:
            return geometry_cache[name]
        mesh = _load_obj_mesh(geom_map[name])
        geometry_cache[name] = mesh
        return mesh

    def _ensure_object_loaded(name: str) -> ViserUrdf:
        if not object_available:
            raise FileNotFoundError("Object URDF is not available for this viewer session.")
        if name in object_cache:
            return object_cache[name]
        frame_path = f"/object/{name}"
        frame = server.scene.add_frame(frame_path, show_axes=False)
        urdf = ViserUrdf(server, urdf_or_path=object_map[name], root_node_name=frame_path)
        urdf.show_visual = bool(cfg.show_object)
        object_cache[name] = urdf
        object_frame_cache[name] = frame
        return urdf

    if cfg.preload:
        for name in pair_names:
            _ensure_motion_loaded(name)
            if overlay_enabled:
                _ensure_overlay_motion_loaded(name)
            if geometry_available:
                _ensure_geometry_loaded(name)
            if object_available and object_urdf_path is None:
                _ensure_object_loaded(name)

    geometry_state: dict[str, viser.GlbHandle | None] = {"handle": None}
    motion_state: dict[str, object] = {}
    overlay_motion_state: dict[str, object] = {}

    def _set_geometry(name: str) -> None:
        if not geometry_available:
            handle = geometry_state["handle"]
            if handle is not None:
                handle.remove()
                geometry_state["handle"] = None
            return
        handle = geometry_state["handle"]
        if handle is not None:
            handle.remove()
            geometry_state["handle"] = None
        mesh = _ensure_geometry_loaded(name)
        geometry_state["handle"] = server.scene.add_mesh_trimesh("/geometry", mesh)

    def _set_motion(name: str) -> None:
        state = _ensure_motion_loaded(name)
        motion_state.update({"name": name, **state})
        if overlay_enabled:
            overlay_state = _ensure_overlay_motion_loaded(name)
            overlay_motion_state.update({"name": name, **overlay_state})

    def _set_object(name: str) -> None:
        if not object_available:
            object_state["name"] = None
            object_state["urdf"] = None
            object_state["frame"] = None
            object_state["single"] = None
            return
        if object_urdf_path is not None:
            if object_state["single"] is None:
                obj_frame = server.scene.add_frame("/object", show_axes=False)
                obj_urdf = ViserUrdf(server, urdf_or_path=object_urdf_path, root_node_name="/object")
                obj_urdf.show_visual = bool(cfg.show_object)
                object_state["single"] = obj_urdf
                object_state["frame"] = obj_frame
            object_state["name"] = name
            object_state["urdf"] = object_state["single"]
            return
        if object_state["name"] is not None and object_state["name"] in object_cache:
            object_cache[object_state["name"]].show_visual = False
        urdf = _ensure_object_loaded(name)
        urdf.show_visual = bool(cfg.show_object)
        object_state["name"] = name
        object_state["urdf"] = urdf
        object_state["frame"] = object_frame_cache.get(name)

    active_clip = cfg.start_clip or pair_names[0]
    _set_motion(active_clip)
    _set_geometry(active_clip)
    _set_object(active_clip)

    with server.gui.add_folder("Motion"):
        clip_dropdown = server.gui.add_dropdown("Clip", options=tuple(pair_names), initial_value=active_clip)
        clip_info = server.gui.add_markdown("")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=cfg.show_meshes)
        show_overlay_meshes_cb = None
        if overlay_enabled:
            show_overlay_meshes_cb = server.gui.add_checkbox(
                "Show overlay meshes",
                initial_value=cfg.overlay_show_meshes,
            )
        show_geom_cb = server.gui.add_checkbox(
            "Show geometry",
            initial_value=cfg.show_geometry and geometry_available,
        )
        show_object_cb = server.gui.add_checkbox(
            "Show object URDF",
            initial_value=cfg.show_object and object_available,
        )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(motion_state["n_frames"]) - 1),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_initial = cfg.fps if cfg.fps is not None else int(motion_state["fps"])
        fps_in = server.gui.add_number("FPS", initial_value=int(fps_initial), min=1, max=240, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=cfg.loop)
        overlay_sync_dropdown = None
        if overlay_enabled:
            overlay_sync_dropdown = server.gui.add_dropdown(
                "Overlay Sync",
                options=("normalized", "time"),
                initial_value=overlay_sync_mode,
            )

    def _update_clip_info() -> None:
        content = f"Clip: `{motion_state['name']}` | A frames: {motion_state['n_frames']} | A fps: {motion_state['fps']}"
        if overlay_enabled:
            content += (
                f" | B frames: {overlay_motion_state['n_frames']} | B fps: {overlay_motion_state['fps']}"
                f" | sync: `{overlay_sync_mode}`"
            )
        clip_info.content = content

    _update_clip_info()

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)

    if show_overlay_meshes_cb is not None:

        @show_overlay_meshes_cb.on_update
        def _(_evt) -> None:
            if overlay_vr is not None:
                overlay_vr.show_visual = bool(show_overlay_meshes_cb.value)

    @show_geom_cb.on_update
    def _(_evt) -> None:
        handle = geometry_state["handle"]
        if handle is not None:
            handle.visible = bool(show_geom_cb.value)

    @show_object_cb.on_update
    def _(_evt) -> None:
        if object_state["name"] is None:
            return
        urdf = object_state["urdf"] if object_urdf_path is not None else object_cache.get(object_state["name"])
        if urdf is not None:
            urdf.show_visual = bool(show_object_cb.value)

    if overlay_sync_dropdown is not None:

        @overlay_sync_dropdown.on_update
        def _(_evt) -> None:
            nonlocal overlay_sync_mode
            overlay_sync_mode = str(overlay_sync_dropdown.value)
            _update_clip_info()
            _apply_frame(int(frame_slider.value))

    @clip_dropdown.on_update
    def _(_evt) -> None:
        name = str(clip_dropdown.value)
        _set_motion(name)
        _set_geometry(name)
        _set_object(name)
        _update_clip_info()
        frame_slider.max = max(0, int(motion_state["n_frames"]) - 1)
        frame_slider.value = 0
        if cfg.fps is None:
            fps_in.value = int(motion_state["fps"])
        _apply_frame(0)
        handle = geometry_state["handle"]
        if handle is not None:
            handle.visible = bool(show_geom_cb.value)
        if object_state["name"] is not None:
            urdf = object_state["urdf"] if object_urdf_path is not None else object_cache.get(object_state["name"])
            if urdf is not None:
                urdf.show_visual = bool(show_object_cb.value)

    playing = {"flag": bool(cfg.autoplay)}
    updating_slider = {"flag": False}

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if updating_slider["flag"]:
            return
        _apply_frame(int(frame_slider.value))

    joint_count = len(viser_joint_names)

    def _overlay_frame_index(frame_idx_a: int) -> int:
        if not overlay_enabled:
            return 0
        n_a = int(motion_state["n_frames"])
        n_b = int(overlay_motion_state["n_frames"])
        if n_b <= 1:
            return 0
        if overlay_sync_mode == "time":
            fps_a = max(1, int(motion_state["fps"]))
            fps_b = max(1, int(overlay_motion_state["fps"]))
            t = float(frame_idx_a) / float(fps_a)
            idx_b = int(round(t * float(fps_b)))
        else:
            if n_a <= 1:
                return 0
            ratio = float(frame_idx_a) / float(n_a - 1)
            idx_b = int(round(ratio * float(n_b - 1)))
        return max(0, min(n_b - 1, idx_b))

    def _apply_frame(frame_idx: int) -> None:
        qpos = motion_state["qpos"]
        qpos_arr = qpos[frame_idx]
        if qpos_arr.shape[0] < 7 + joint_count:
            raise ValueError(f"qpos frame is too small: {qpos_arr.shape[0]} < {7 + joint_count}")
        root_pos = qpos_arr[:3]
        root_quat = qpos_arr[3:7]
        joints = qpos_arr[7 : 7 + joint_count]
        robot_root.position = root_pos
        robot_root.wxyz = root_quat
        vr.update_cfg(joints)
        if overlay_enabled and overlay_vr is not None and overlay_root is not None:
            qpos_b = overlay_motion_state["qpos"]
            frame_idx_b = _overlay_frame_index(frame_idx)
            qpos_arr_b = qpos_b[frame_idx_b]
            if qpos_arr_b.shape[0] < 7 + joint_count:
                raise ValueError(f"overlay qpos frame is too small: {qpos_arr_b.shape[0]} < {7 + joint_count}")
            overlay_root.position = qpos_arr_b[:3]
            overlay_root.wxyz = qpos_arr_b[3:7]
            overlay_vr.update_cfg(qpos_arr_b[7 : 7 + joint_count])
        if object_state["frame"] is not None and qpos_arr.shape[0] >= (7 + joint_count + 7):
            obj_pos = qpos_arr[-7:-4]
            obj_quat = qpos_arr[-4:]
            object_state["frame"].position = obj_pos
            object_state["frame"].wxyz = obj_quat

    def _player_loop() -> None:
        next_tick = time.time()
        frame_idx = 0
        while True:
            if playing["flag"]:
                fps_val = int(fps_in.value)
                if fps_val <= 0:
                    fps_val = 1
                now = time.time()
                if now >= next_tick:
                    next_tick = now + 1.0 / float(fps_val)
                    frame_idx = int(frame_slider.value) + 1
                    last_frame = int(motion_state["n_frames"]) - 1
                    if frame_idx > last_frame:
                        if loop_cb.value:
                            frame_idx = 0
                        else:
                            frame_idx = last_frame
                            playing["flag"] = False
                    updating_slider["flag"] = True
                    frame_slider.value = frame_idx
                    updating_slider["flag"] = False
                    _apply_frame(frame_idx)
            time.sleep(0.001)

    _apply_frame(0)
    threading.Thread(target=_player_loop, daemon=True).start()
    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")

    while True:
        time.sleep(1.0)


def main() -> None:
    cfg = tyro.cli(MotionGeometryViewerConfig, config=TYRO_CONIFG)
    run_viewer(cfg)


if __name__ == "__main__":
    main()
