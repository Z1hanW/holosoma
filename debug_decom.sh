#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  echo "Usage: $0 <script-name>" >&2
  echo "Example: $0 train_as_general" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

RAW_SCRIPT_NAME=$1
SCRIPT_NAME=$(basename "${RAW_SCRIPT_NAME}")
SCRIPT_NAME=${SCRIPT_NAME%.sh}

case "${SCRIPT_NAME}" in
  train_as_general)
    DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
    OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${DATA_DIR}/_clip_object_urdf_map.json"}
    ;;
  *)
    echo "[ERROR] Unsupported script for debug decomposition: ${RAW_SCRIPT_NAME}" >&2
    echo "[ERROR] Supported: train_as_general" >&2
    exit 2
    ;;
esac

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
DATA_DIR=$(realpath -m "${DATA_DIR}")
OBJECT_MAP=$(realpath -m "${OBJECT_MAP}")

case "${DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] DATA_DIR must be local, not NFS: ${DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_real.sh first and debug from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
    exit 2
    ;;
esac
case "${DATA_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${DATA_DIR}" >&2
    exit 2
    ;;
esac
case "${OBJECT_MAP}" in
  /nfs|/nfs/*)
    echo "[ERROR] OBJECT_MAP must be local, not NFS: ${OBJECT_MAP}" >&2
    echo "[ERROR] Run ./cp_real.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    exit 2
    ;;
esac
case "${OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OBJECT_MAP}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[ERROR] DATA_DIR does not exist: ${DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_real.sh first." >&2
  exit 2
fi

if [[ ! -f "${OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing object map: ${OBJECT_MAP}" >&2
  echo "[ERROR] Run ./cp_real.sh first." >&2
  exit 2
fi

OUT_DIR=${DEBUG_DECOM_OUT_DIR:-"${SCRIPT_DIR}/debug_DECOM/${SCRIPT_NAME}"}
mkdir -p "${OUT_DIR}"

OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}
VHACD_MAX_HULLS=${DEBUG_DECOM_VHACD_MAX_HULLS:-32}
VHACD_RESOLUTION=${DEBUG_DECOM_VHACD_RESOLUTION:-100000}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python" ]]; then
    PYTHON_BIN="/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

"${PYTHON_BIN}" - "${SCRIPT_NAME}" "${DATA_DIR}" "${OBJECT_MAP}" "${OUT_DIR}" "${VHACD_MAX_HULLS}" "${VHACD_RESOLUTION}" "${OBJECT_COLLIDER_TYPE}" <<'PY'
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

script_name = sys.argv[1]
data_dir = Path(sys.argv[2]).expanduser().resolve()
map_path = Path(sys.argv[3]).expanduser().resolve()
out_dir = Path(sys.argv[4]).expanduser().resolve()
vhacd_max_hulls = int(sys.argv[5])
vhacd_resolution = int(sys.argv[6])
collider_type_raw = str(sys.argv[7]).strip().lower()
if collider_type_raw in {"convex_decomposition", "convex_decomp", "decomposition", "vhacd"}:
    collider_type = "convex_decomposition"
elif collider_type_raw in {"convex_hull", "hull"}:
    collider_type = "convex_hull"
else:
    raise SystemExit(
        "[ERROR] HOLOSOMA_OBJECT_COLLIDER_TYPE must be convex_decomposition or convex_hull; "
        f"got {collider_type_raw!r}"
    )


def load_clip_map(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if isinstance(clips, dict):
        return clips
    if isinstance(payload, dict):
        return payload
    raise SystemExit(f"[ERROR] Unsupported object map format: {path}")


def resolve_path(raw: str, base_dir: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("empty path")
    if value.startswith("package://"):
        value = value[len("package://") :]
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def tag_name(elem) -> str:
    return str(elem.tag).rsplit("}", 1)[-1]


def direct_child(parent, name: str):
    for child in list(parent):
        if tag_name(child) == name:
            return child
    return None


def iter_named(parent, name: str):
    for elem in parent.iter():
        if tag_name(elem) == name:
            yield elem


def parse_vec(raw, default, expected_len: int) -> np.ndarray:
    if raw is None or str(raw).strip() == "":
        return np.asarray(default, dtype=float)
    values = [float(part) for part in str(raw).replace(",", " ").split()]
    if len(values) == 1 and expected_len == 3:
        values = values * 3
    if len(values) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {raw!r}")
    return np.asarray(values, dtype=float)


def origin_matrix(origin_elem) -> np.ndarray:
    xyz = parse_vec(origin_elem.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0], 3)
    rpy = parse_vec(origin_elem.get("rpy") if origin_elem is not None else None, [0.0, 0.0, 0.0], 3)
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz
    return transform


def as_mesh(loaded) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
    raise TypeError(f"unsupported mesh payload: {type(loaded)!r}")


def colorize(mesh: trimesh.Trimesh, color) -> trimesh.Trimesh:
    rgba = np.asarray(color, dtype=np.uint8)
    if len(mesh.vertices):
        mesh.visual.vertex_colors = np.repeat(rgba.reshape(1, 4), len(mesh.vertices), axis=0)
    return mesh


def part_color(index: int, total: int) -> np.ndarray:
    import colorsys

    total = max(int(total), 1)
    hue = (float(index) / float(total) + 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return np.array([int(255 * value) for value in rgb] + [235], dtype=np.uint8)


def load_geometry_mesh(mesh_elem, urdf_dir: Path) -> trimesh.Trimesh:
    filename = mesh_elem.get("filename")
    if not filename:
        raise ValueError("mesh geometry is missing filename")
    mesh_path = resolve_path(filename, urdf_dir)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing mesh file: {mesh_path}")
    loaded = trimesh.load(mesh_path, process=False)
    mesh = as_mesh(loaded)
    scale = parse_vec(mesh_elem.get("scale"), [1.0, 1.0, 1.0], 3)
    scale_tf = np.eye(4)
    scale_tf[0, 0] = scale[0]
    scale_tf[1, 1] = scale[1]
    scale_tf[2, 2] = scale[2]
    mesh.apply_transform(scale_tf)
    return mesh


def load_urdf_kind(urdf_path: Path, kind: str, color) -> tuple[trimesh.Trimesh, list[str]]:
    root = ET.parse(urdf_path).getroot()
    meshes = []
    source_paths = []
    for container in iter_named(root, kind):
        origin_elem = direct_child(container, "origin")
        geometry_elem = direct_child(container, "geometry")
        if geometry_elem is None:
            continue
        mesh_elem = direct_child(geometry_elem, "mesh")
        if mesh_elem is None:
            continue
        mesh = load_geometry_mesh(mesh_elem, urdf_path.parent)
        source_paths.append(str(resolve_path(mesh_elem.get("filename"), urdf_path.parent)))
        mesh.apply_transform(origin_matrix(origin_elem))
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"{urdf_path} has no {kind} mesh geometry")
    combined = trimesh.util.concatenate(meshes)
    return colorize(combined, color), source_paths


def build_dynamic_collision_proxy(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, str]:
    """Approximate the dynamic-object physics collision proxy with VHACD convex hulls."""
    if collider_type == "convex_hull":
        hull = mesh.convex_hull
        return colorize(hull, [237, 125, 49, 235]), 1, "convex_hull"

    try:
        from vhacdx import compute_vhacd
    except Exception as exc:
        hull = mesh.convex_hull
        return colorize(hull, [237, 125, 49, 235]), 1, f"convex_hull_fallback_import_error:{exc}"

    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        hull = mesh.convex_hull
        return colorize(hull, [237, 125, 49, 235]), 1, "convex_hull_fallback_too_small"

    faces = np.column_stack((np.ones(len(mesh.faces), dtype=np.uint32) * 3, mesh.faces.astype(np.uint32))).ravel()
    try:
        parts_raw = compute_vhacd(
            np.asarray(mesh.vertices, dtype=np.float64),
            faces,
            maxConvexHulls=vhacd_max_hulls,
            resolution=vhacd_resolution,
            maxNumVerticesPerCH=64,
            shrinkWrap=True,
            fillMode="flood",
            asyncACD=True,
        )
    except Exception as exc:
        hull = mesh.convex_hull
        return colorize(hull, [237, 125, 49, 235]), 1, f"convex_hull_fallback_vhacd_error:{exc}"

    parts = []
    for idx, (vertices, faces_out) in enumerate(parts_raw):
        if len(vertices) == 0 or len(faces_out) == 0:
            continue
        part = trimesh.Trimesh(vertices=vertices, faces=faces_out, process=False)
        parts.append(colorize(part, part_color(idx, len(parts_raw))))
    if not parts:
        hull = mesh.convex_hull
        return colorize(hull, [237, 125, 49, 235]), 1, "convex_hull_fallback_empty_vhacd"
    return trimesh.util.concatenate(parts), len(parts), "vhacd"


def place_mesh(mesh: trimesh.Trimesh, center_x: float, center_y: float) -> trimesh.Trimesh:
    placed = mesh.copy()
    bounds = placed.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    tf = np.eye(4)
    tf[:3, 3] = np.array([center_x - center[0], center_y - center[1], -bounds[0, 2]], dtype=float)
    placed.apply_transform(tf)
    return placed


clips = load_clip_map(map_path)
objects = {}
for clip_name, entry in clips.items():
    if not isinstance(entry, dict):
        raise SystemExit(f"[ERROR] Invalid map entry for {clip_name}: expected dict")
    urdf_path = resolve_path(entry.get("object_urdf_path"), map_path.parent)
    if not urdf_path.is_file():
        raise SystemExit(f"[ERROR] Missing URDF for {clip_name}: {urdf_path}")
    object_name = str(entry.get("object_name") or urdf_path.stem.replace("objects_", ""))
    objects.setdefault(str(urdf_path), {"name": object_name, "urdf": urdf_path, "clips": []})
    objects[str(urdf_path)]["clips"].append(clip_name)

if not objects:
    raise SystemExit(f"[ERROR] Empty object map: {map_path}")

records = []
per_object_dir = out_dir / "per_object"
per_object_dir.mkdir(parents=True, exist_ok=True)
for item in sorted(objects.values(), key=lambda value: value["name"]):
    urdf_path = item["urdf"]
    visual, visual_sources = load_urdf_kind(urdf_path, "visual", [63, 142, 210, 230])
    raw_collision, collision_sources = load_urdf_kind(urdf_path, "collision", [237, 125, 49, 210])
    collision, collision_part_count, collision_mode = build_dynamic_collision_proxy(raw_collision)
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in item["name"])
    visual_path = per_object_dir / f"{safe_name}_visual.glb"
    collision_glb_path = per_object_dir / f"{safe_name}_dynamic_collision_vhacd.glb"
    collision_obj_path = per_object_dir / f"{safe_name}_dynamic_collision_vhacd.obj"
    visual.export(visual_path)
    collision.export(collision_glb_path)
    collision.export(collision_obj_path)
    visual_ext = visual.extents
    collision_ext = collision.extents
    records.append(
        {
            "name": item["name"],
            "urdf_path": str(urdf_path),
            "clip_count": len(item["clips"]),
            "visual": visual,
            "collision": collision,
            "collision_part_count": collision_part_count,
            "collision_mode": collision_mode,
            "visual_sources": visual_sources,
            "collision_sources": collision_sources,
            "visual_debug_path": str(visual_path),
            "collision_debug_glb_path": str(collision_glb_path),
            "collision_debug_obj_path": str(collision_obj_path),
            "visual_extent": visual_ext.tolist(),
            "collision_extent": collision_ext.tolist(),
        }
    )

max_width = max(max(record["visual"].extents[0], record["collision"].extents[0]) for record in records)
max_depth = max(max(record["visual"].extents[1], record["collision"].extents[1]) for record in records)
column_gap = max(max_width * 1.6, 1.0)
row_gap_base = max(max_depth * 0.6, 0.35)

placed_meshes = []
row_cursor = 0.0
for record in records:
    row_height = max(record["visual"].extents[1], record["collision"].extents[1])
    row_y = -row_cursor
    placed_meshes.append(place_mesh(record["visual"], 0.0, row_y))
    placed_meshes.append(place_mesh(record["collision"], column_gap, row_y))
    row_cursor += row_height + row_gap_base

combined = trimesh.util.concatenate(placed_meshes)
glb_path = out_dir / f"{script_name}_urdf_visual_collision_columns.glb"
obj_path = out_dir / f"{script_name}_urdf_visual_collision_columns.obj"
manifest_path = out_dir / f"{script_name}_urdf_visual_collision_manifest.json"
combined.export(glb_path)
combined.export(obj_path)

manifest = {
    "script_name": script_name,
    "data_dir": str(data_dir),
    "object_map": str(map_path),
    "layout": {
        "visual_column_x": 0.0,
        "collision_column_x": column_gap,
        "row_axis": "negative_y",
        "ground_z": 0.0,
    },
    "dynamic_collision": {
        "mode": f"{collider_type}_from_urdf_collision_mesh",
        "collider_type": collider_type,
        "max_hulls": vhacd_max_hulls,
        "resolution": vhacd_resolution,
        "note": "URDF collision mesh is the input; right column shows the dynamic-object physics collision proxy.",
    },
    "objects": [
        {
            "name": record["name"],
            "urdf_path": record["urdf_path"],
            "clip_count": record["clip_count"],
            "collision_mode": record["collision_mode"],
            "collision_part_count": record["collision_part_count"],
            "visual_sources": record["visual_sources"],
            "collision_sources": record["collision_sources"],
            "visual_debug_path": record["visual_debug_path"],
            "collision_debug_glb_path": record["collision_debug_glb_path"],
            "collision_debug_obj_path": record["collision_debug_obj_path"],
            "visual_extent": record["visual_extent"],
            "collision_extent": record["collision_extent"],
        }
        for record in records
    ],
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"[INFO] Object URDFs: {len(records)}")
print(f"[INFO] Clips in map: {len(clips)}")
print(
    f"[INFO] Dynamic collision proxy: collider_type={collider_type} "
    f"vhacd_max_hulls={vhacd_max_hulls} resolution={vhacd_resolution}"
)
for record in records:
    print(
        f"[INFO]   {record['name']}: collision_mode={record['collision_mode']} "
        f"parts={record['collision_part_count']}"
    )
print(f"[INFO] Visual column x=0.000, collision column x={column_gap:.3f}")
print(f"[INFO] Wrote GLB: {glb_path}")
print(f"[INFO] Wrote OBJ: {obj_path}")
print(f"[INFO] Wrote manifest: {manifest_path}")
PY
