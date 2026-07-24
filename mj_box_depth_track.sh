#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="$ROOT_DIR/outputs/motion_bank_success_box_0_92_0p3"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-https://wandb.ai/zihanw22/boxer/runs/shoo7sr1/model_29999.onnx}"
DEFAULT_OBJECT_MAP="$DEFAULT_MOTION_DIR/_clip_object_urdf_map.json"
DEFAULT_PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE_DEFAULT:-far_tracking_warp}"
MUJOCO_RENDER_848_PERCEPTION_PRESET="${MUJOCO_RENDER_848_PERCEPTION_PRESET:-camera_depth_d435i_mujoco_render_848x480}"

INFER_PYTHON_BIN="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
if [[ ! -x "$INFER_PYTHON_BIN" ]]; then
  INFER_PYTHON_BIN="${INFER_PYTHON_BIN:-python3}"
fi

usage() {
  cat <<EOF
Usage:
  MOTION_DIR=/path/to/rollout_motion_bank OBJECT_URDF=/path/to/object.urdf bash mj_box_depth_track.sh [depth|rendered|rendered848|warp] [clip_name|motion.npz] [model.onnx|wandb://...] [viser args...]

Defaults:
  motion_dir    = ${DEFAULT_MOTION_DIR}
  object_map    = ${DEFAULT_OBJECT_MAP}
  model         = ${DEFAULT_MODEL_INPUT}
  depth_source  = ${DEFAULT_PERCEPTION_CAMERA_SOURCE}
  object_mass   = MUJOCO_OBJECT_MASS_OVERRIDE:-<unset>
  lateral_friction = MUJOCO_OBJECT_LATERAL_FRICTION:-<unset>
  rolling_friction = MUJOCO_OBJECT_ROLLING_FRICTION:-<unset>
  contact_stiffness= MUJOCO_OBJECT_CONTACT_STIFFNESS:-<unset>
  contact_damping  = MUJOCO_OBJECT_CONTACT_DAMPING:-<unset>
  GT_MUJOCO_PHYSICS=1 forces GT-style object/G1/floor MuJoCo physics; default keeps training/URDF physics
  HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH=0 disables cuda:0 auto-selection for far_tracking_warp
EOF
}

is_truthy_env() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MOTION_DIR="${MOTION_DIR:-$DEFAULT_MOTION_DIR}"
OBJECT_MAP_INPUT="${OBJECT_URDF:-}"
MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-$DEFAULT_MODEL_INPUT}}"
MOTION_FILE="${MOTION_FILE:-}"
MOTION_CLIP_NAME="${MOTION_CLIP_NAME:-${MOTION_CLIP:-}}"
PERCEPTION_PRESET_EXPLICIT=0
[[ -n "${PERCEPTION_PRESET+x}" ]] && PERCEPTION_PRESET_EXPLICIT=1
EXTRA_ARGS=()
POSITIONAL_MODE=1

for arg in "$@"; do
  if [[ "$POSITIONAL_MODE" == "0" ]]; then
    EXTRA_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    depth)
      ;;
    rendered|render|mujoco)
      PERCEPTION_CAMERA_SOURCE="rendered"
      if [[ "$PERCEPTION_PRESET_EXPLICIT" == "0" ]]; then
        PERCEPTION_PRESET="$MUJOCO_RENDER_848_PERCEPTION_PRESET"
      fi
      PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-mujoco848}"
      ;;
    rendered848|render848|mujoco848|mujoco_render_848x480)
      PERCEPTION_CAMERA_SOURCE="rendered"
      if [[ "$PERCEPTION_PRESET_EXPLICIT" == "0" ]]; then
        PERCEPTION_PRESET="$MUJOCO_RENDER_848_PERCEPTION_PRESET"
      fi
      PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-mujoco848}"
      ;;
    warp|far_tracking_warp)
      PERCEPTION_CAMERA_SOURCE="far_tracking_warp"
      ;;
    *.npz)
      MOTION_FILE="$arg"
      ;;
    wandb://*|https://*|*.onnx|*.pt)
      MODEL_INPUT="$arg"
      ;;
    --*)
      POSITIONAL_MODE=0
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      if [[ -z "$MOTION_CLIP_NAME" ]]; then
        MOTION_CLIP_NAME="$arg"
      else
        EXTRA_ARGS+=("$arg")
      fi
      ;;
  esac
done

if [[ -z "$MOTION_FILE" ]]; then
  if [[ -n "$MOTION_CLIP_NAME" ]]; then
    MOTION_FILE="$MOTION_DIR/${MOTION_CLIP_NAME%.npz}.npz"
  else
    MOTION_FILE="$(find "$MOTION_DIR" -maxdepth 1 -name '*.npz' | sort | head -n 1)"
  fi
fi

if [[ -z "$MOTION_FILE" || ! -f "$MOTION_FILE" ]]; then
  echo "[ERROR] Motion clip not found: ${MOTION_FILE:-<empty>}" >&2
  exit 1
fi
MOTION_FILE="$(cd "$(dirname "$MOTION_FILE")" && pwd)/$(basename "$MOTION_FILE")"
MOTION_CLIP_STEM="$(basename "$MOTION_FILE")"
MOTION_CLIP_STEM="${MOTION_CLIP_STEM%.npz}"
if [[ -z "$OBJECT_MAP_INPUT" && -f "${MOTION_DIR%/}/_clip_object_urdf_map.json" ]]; then
  OBJECT_MAP_INPUT="${MOTION_DIR%/}/_clip_object_urdf_map.json"
fi

if [[ "$MODEL_INPUT" == wandb://*.pt ]]; then
  MODEL_INPUT="${MODEL_INPUT%.pt}.onnx"
fi

MODEL_LOCAL="$(
  "$INFER_PYTHON_BIN" - <<'PY' "$MODEL_INPUT" "$ROOT_DIR/logs/wandb_runs"
import sys
from urllib.parse import urlparse
from pathlib import Path

from holosoma_inference.utils.wandb import load_checkpoint

model = sys.argv[1]
root = Path(sys.argv[2])
download_dir = root / "box_depth"
if model.startswith("wandb://"):
    parts = model[len("wandb://") :].split("/", 3)
    if len(parts) >= 3:
        download_dir = root / parts[2]
elif model.startswith("https://"):
    parts = [part for part in urlparse(model).path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "runs":
        download_dir = root / parts[3]
path = load_checkpoint(None, model, str(download_dir))
print(Path(path).expanduser().resolve())
PY
)"
MODEL_LOCAL="$(printf '%s\n' "$MODEL_LOCAL" | tail -n 1)"

export MUJOCO_OBJECT_COLLISION_MODE="${MUJOCO_OBJECT_COLLISION_MODE:-mesh}"
export HOLOSOMA_MUJOCO_OBJECT_COLLISION_MODE="${HOLOSOMA_MUJOCO_OBJECT_COLLISION_MODE:-$MUJOCO_OBJECT_COLLISION_MODE}"
export MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE="${MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE:-1}"
export HOLOSOMA_MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE="${HOLOSOMA_MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE:-$MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE}"

OBJECT_URDF_RESOLVED="$(
  "$INFER_PYTHON_BIN" - <<'PY' "$OBJECT_MAP_INPUT" "$MOTION_FILE" "$ROOT_DIR"
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).expanduser().resolve()
stem = motion_path.stem
relative_urdf_roots: list[Path] = [motion_path.parent, repo_root]

def object_urdf_fallbacks(path: Path):
    """Yield current-repo fallbacks for older absolute box-object paths."""
    expanded = path.expanduser()
    name = expanded.stem

    parts = expanded.parts
    if "data" in parts:
        data_idx = parts.index("data")
        tail = parts[data_idx:]
        if len(tail) >= 2 and tail[1] == "ds_box_data":
            relative_tail = tail[2:]
            yield repo_root / "data/ds_box_data_legacy" / Path(*relative_tail)
        yield repo_root.joinpath(*tail)
        if len(tail) >= 2 and tail[1] == "ds_box_data":
            relative_tail = tail[2:]
            yield repo_root / "data/ds_box_data/scale_mix_all" / Path(*relative_tail)
            if name and "__" not in name:
                for candidate_name in [f"{name}__baseline", f"{name}__eff10", f"{name}__eff09"]:
                    yield (
                        repo_root
                        / "data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared/_generated_urdfs"
                        / f"{candidate_name}.urdf"
                    )

    if name:
        if "__" in name:
            yield repo_root / "data/ds_box_data_legacy/train_g1_w_obj_prepared/_generated_urdfs" / f"{name}.urdf"
            names = [name]
        else:
            yield repo_root / "data/ds_box_data_legacy/train_g1_w_obj_prepared/_generated_urdfs" / f"{name}.urdf"
            names = [f"{name}__baseline", f"{name}__eff10", f"{name}__eff09"]
        for candidate_name in names:
            yield repo_root / "data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared/_generated_urdfs" / f"{candidate_name}.urdf"


def _is_ds_box_data_path(path: Path) -> bool:
    parts = path.expanduser().parts
    return "data" in parts and "ds_box_data" in parts


def resolve_existing_urdf(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend(base / candidate for base in relative_urdf_roots)
    for current in candidates:
        if _is_ds_box_data_path(current):
            for fallback in object_urdf_fallbacks(current):
                if fallback.is_file():
                    return maybe_write_motion_sized_urdf(fallback.resolve())
        if current.is_file():
            return maybe_write_motion_sized_urdf(current.resolve())
    for current in candidates:
        for fallback in object_urdf_fallbacks(current):
            if fallback.is_file():
                return maybe_write_motion_sized_urdf(fallback.resolve())
    return maybe_write_motion_sized_urdf(candidates[-1].resolve())


def _parse_vec(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(part) for part in str(raw).replace(",", " ").split()], dtype=np.float64)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.size != 3:
        raise ValueError(f"Expected 3-vector, got {raw!r}")
    return values


def _obj_extents(path: Path) -> np.ndarray | None:
    vertices: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except OSError:
        return None
    if not vertices:
        return None
    arr = np.asarray(vertices, dtype=np.float64)
    return arr.max(axis=0) - arr.min(axis=0)


def _motion_object_size() -> np.ndarray | None:
    try:
        with np.load(motion_path, allow_pickle=True) as data:
            if "object_size" not in data:
                return None
            size = np.asarray(data["object_size"], dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return None
    return size


def _mesh_path(urdf_path: Path, filename: str) -> Path:
    mesh_path = Path(filename).expanduser()
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _object_collision_mode() -> str:
    raw = os.environ.get(
        "MUJOCO_OBJECT_COLLISION_MODE",
        os.environ.get("HOLOSOMA_MUJOCO_OBJECT_COLLISION_MODE", "mesh"),
    )
    return str(raw).strip().lower().replace("-", "_")


def _truthy_env(*names: str, default: bool = False) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def _set_collision_box(root_xml: ET.Element, desired_size: np.ndarray) -> bool:
    size_text = " ".join(str(float(value)) for value in desired_size.tolist())
    collision_elems = list(root_xml.findall(".//collision"))
    if not collision_elems:
        link = root_xml.find(".//link")
        if link is None:
            return False
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
        collision_elems = [collision]

    for collision in collision_elems:
        geometry = collision.find("geometry")
        if geometry is None:
            geometry = ET.SubElement(collision, "geometry")
        for child in list(geometry):
            geometry.remove(child)
        ET.SubElement(geometry, "box", {"size": size_text})
    name = str(root_xml.get("name") or "").strip()
    if name and not name.endswith("_box_collision"):
        root_xml.set("name", f"{name}_box_collision")
    return True


def _collision_box_matches(root_xml: ET.Element, desired_size: np.ndarray) -> bool:
    for box in root_xml.findall(".//collision/geometry/box"):
        try:
            size = _parse_vec(box.get("size"), (0.0, 0.0, 0.0))
        except Exception:
            continue
        if np.allclose(size, desired_size, rtol=0.02, atol=2.0e-3):
            return True
    return False


def maybe_write_motion_sized_urdf(urdf_path: Path) -> Path:
    if not urdf_path.is_file():
        return urdf_path
    collision_mode = _object_collision_mode()
    use_box_collision = collision_mode in {"box", "primitive", "primitive_box", "box_collision", "rollout"}
    resize_to_motion_size = _truthy_env(
        "MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE",
        "HOLOSOMA_MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE",
        default=False,
    )

    try:
        tree = ET.parse(urdf_path)
    except Exception:
        return urdf_path
    root_xml = tree.getroot()
    mesh_elems = list(root_xml.findall(".//mesh"))
    if not mesh_elems:
        return urdf_path

    first_mesh = mesh_elems[0]
    first_filename = str(first_mesh.get("filename") or "").strip()
    if not first_filename:
        return urdf_path
    first_scale = _parse_vec(first_mesh.get("scale"), (1.0, 1.0, 1.0))
    first_extents = _obj_extents(_mesh_path(urdf_path, first_filename))
    if first_extents is None:
        return urdf_path
    current_size = first_extents * first_scale
    if current_size.size != 3 or np.any(current_size <= 0.0):
        return urdf_path
    desired_size = _motion_object_size() if resize_to_motion_size else current_size
    if desired_size is None:
        return urdf_path
    mesh_already_sized = np.allclose(current_size, desired_size, rtol=0.02, atol=2.0e-3)
    box_already_sized = use_box_collision and _collision_box_matches(root_xml, desired_size)
    if mesh_already_sized and (not use_box_collision or box_already_sized):
        return urdf_path

    scale_ratio = np.ones(3, dtype=np.float64) if mesh_already_sized else desired_size / current_size
    if not np.all(np.isfinite(scale_ratio)) or np.any(scale_ratio <= 0.0):
        return urdf_path

    for mesh in mesh_elems:
        filename = str(mesh.get("filename") or "").strip()
        if filename:
            mesh.set("filename", str(_mesh_path(urdf_path, filename)))
        old_scale = _parse_vec(mesh.get("scale"), (1.0, 1.0, 1.0))
        new_scale = old_scale * scale_ratio
        mesh.set("scale", " ".join(f"{value:.9g}" for value in new_scale))

    mass_elem = root_xml.find(".//inertial/mass")
    inertia_elem = root_xml.find(".//inertial/inertia")
    if mass_elem is not None and inertia_elem is not None:
        try:
            mass = float(mass_elem.get("value", "0"))
        except ValueError:
            mass = 0.0
        if mass > 0.0:
            sx, sy, sz = desired_size.tolist()
            inertia_elem.set("ixx", f"{(mass / 12.0) * (sy * sy + sz * sz):.9g}")
            inertia_elem.set("iyy", f"{(mass / 12.0) * (sx * sx + sz * sz):.9g}")
            inertia_elem.set("izz", f"{(mass / 12.0) * (sx * sx + sy * sy):.9g}")
            inertia_elem.set("ixy", "0")
            inertia_elem.set("ixz", "0")
            inertia_elem.set("iyz", "0")

    if use_box_collision:
        _set_collision_box(root_xml, desired_size)

    out_dir = repo_root / "logs/sim2sim_exports/object_urdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    size_source = "motion_size" if resize_to_motion_size else "urdf_size"
    suffix = f"{size_source}_box_collision" if use_box_collision else size_source
    out_path = out_dir / f"{motion_path.stem}__{urdf_path.stem}__{suffix}.urdf"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(
        f"[INFO] generated {size_source} object URDF {out_path}: source={urdf_path} current_size={current_size.tolist()} desired_size={desired_size.tolist()} scale_ratio={scale_ratio.tolist()} collision_mode={collision_mode}",
        file=sys.stderr,
    )
    return out_path.resolve()


candidate = Path(raw).expanduser() if raw else None
if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".json":
    relative_urdf_roots.insert(0, candidate.parent.resolve())
    data = json.loads(candidate.read_text())
    clips = data.get("clips", data) if isinstance(data, dict) else {}
    entry = clips.get(stem) if isinstance(clips, dict) else None
    if not isinstance(entry, dict):
        raise SystemExit(f"Object map has no entry for clip '{stem}': {candidate}")
    path = entry.get("object_urdf_path") or entry.get("urdf_path")
    if not path:
        raise SystemExit(f"Object map entry for clip '{stem}' has no object_urdf_path")
    print(resolve_existing_urdf(str(path)))
elif candidate is not None and str(candidate):
    print(resolve_existing_urdf(str(candidate)))
else:
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" not in data:
            raise SystemExit(f"No OBJECT_URDF map provided and motion has no object_urdf_path: {motion_path}")
        print(resolve_existing_urdf(str(np.asarray(data["object_urdf_path"]).item())))
PY
)"

export OBJECT_URDF="$OBJECT_URDF_RESOLVED"
export ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-1}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-$DEFAULT_PERCEPTION_CAMERA_SOURCE}"
if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" ]]; then
  if [[ "$PERCEPTION_PRESET_EXPLICIT" == "0" ]]; then
    PERCEPTION_PRESET="$MUJOCO_RENDER_848_PERCEPTION_PRESET"
  fi
  export PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-mujoco848}"
fi
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-}"
export HOLOSOMA_ALLOW_FILE_BACKED_PERCEPTION="${HOLOSOMA_ALLOW_FILE_BACKED_PERCEPTION:-0}"
if ! is_truthy_env "$HOLOSOMA_ALLOW_FILE_BACKED_PERCEPTION"; then
  unset HOLOSOMA_POLICY_PERCEPTION_OBS_FILE
  unset HOLOSOMA_POLICY_PERCEPTION_OBS_FILE_KEY
  unset HOLOSOMA_POLICY_PERCEPTION_OBS_FILE_INDEX
  unset HOLOSOMA_POLICY_ACTION_FILE
  unset HOLOSOMA_POLICY_ACTION_FILE_KEY
  unset HOLOSOMA_POLICY_ACTION_FILE_INDEX
fi
export PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-}"
export PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-}"
export PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-}"
export PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH="${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-}"
export PERCEPTION_CAMERA_PITCH_DEG="${PERCEPTION_CAMERA_PITCH_DEG:-}"
export PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH="${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-}"
export PERCEPTION_UPDATE_HZ="${PERCEPTION_UPDATE_HZ:-}"
export PERCEPTION_CAMERA_FPS="${PERCEPTION_CAMERA_FPS:-}"
export PERCEPTION_CAMERA_WARP_BUFFER_LEN="${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-}"
export PERCEPTION_CAMERA_WARP_LATENCY_FRAME="${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-}"
export PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-}"
export PERCEPTION_CAMERA_WARP_EDGE_BORDER="${PERCEPTION_CAMERA_WARP_EDGE_BORDER:-}"
export PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB="${PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB:-}"
export PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB="${PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB:-}"
export PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY="${PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY:-}"
export PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY="${PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY:-}"
export PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH="${PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH:-}"
export PERCEPTION_CAMERA_WARP_ENABLE_HOLES="${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-}"
export PERCEPTION_CAMERA_WARP_HOLE_PROB="${PERCEPTION_CAMERA_WARP_HOLE_PROB:-}"
export PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-}"
export USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
export POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
export HOLOSOMA_W_OBJECT_URDF="${HOLOSOMA_W_OBJECT_URDF:-g1/g1_29dof.urdf}"
export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
export MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-}"
if [[ -n "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS" && "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS" != \[* ]]; then
  MUJOCO_OBJECT_CONTACT_BODY_MARKERS="$(
    "$INFER_PYTHON_BIN" - <<'PY' "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS"
import json
import sys

markers = [part.strip() for part in sys.argv[1].replace(",", " ").split() if part.strip()]
print(json.dumps(markers, separators=(",", ":")))
PY
  )"
  export MUJOCO_OBJECT_CONTACT_BODY_MARKERS
fi
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
export MUJOCO_OBJECT_LATERAL_FRICTION="${MUJOCO_OBJECT_LATERAL_FRICTION:-}"
export MUJOCO_OBJECT_ROLLING_FRICTION="${MUJOCO_OBJECT_ROLLING_FRICTION:-}"
export MUJOCO_OBJECT_CONTACT_STIFFNESS="${MUJOCO_OBJECT_CONTACT_STIFFNESS:-}"
export MUJOCO_OBJECT_CONTACT_DAMPING="${MUJOCO_OBJECT_CONTACT_DAMPING:-}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-0}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-0}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS:-0}"
if [[ -n "${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION+x}" ]]; then
  export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION
fi
if [[ -n "${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION+x}" ]]; then
  export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION
fi
if [[ -n "${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION+x}" ]]; then
  export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION
fi
export HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES="${HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES:-0}"
export HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES="${HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES:-0}"
if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET+x}" ]]; then
  if [[ "$MOTION_CLIP_STEM" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=5
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
else
  export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET
fi
export HOLOSOMA_PREFILL_OBS_HISTORY_ON_MOTION_START="${HOLOSOMA_PREFILL_OBS_HISTORY_ON_MOTION_START:-0}"
export HOLOSOMA_FORCE_MOTION_ALIGNMENT="${HOLOSOMA_FORCE_MOTION_ALIGNMENT:-1}"
export HOLOSOMA_ZMQ_LOWCMD_LATCH_CONTROL_BOUNDARY="${HOLOSOMA_ZMQ_LOWCMD_LATCH_CONTROL_BOUNDARY:-0}"
export HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY="${HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY:-1}"
export HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS="${HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS:-2}"
export HOLOSOMA_ZMQ_LOWCMD_KP_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KP_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_KD_SCALE="${HOLOSOMA_ZMQ_LOWCMD_KD_SCALE:-1.0}"
export HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE="${HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE:-1.0}"
export HOLOSOMA_CLIP_JOINT_TARGETS=0
export MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-0}"
export TERRAIN_STATIC_FRICTION="${TERRAIN_STATIC_FRICTION:-1.0}"
export TERRAIN_DYNAMIC_FRICTION="${TERRAIN_DYNAMIC_FRICTION:-1.0}"
export SIM_FPS="${SIM_FPS:-500}"
export SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-10}"
export SIM_SUBSTEPS="${SIM_SUBSTEPS:-1}"
export MUJOCO_BACKEND="${MUJOCO_BACKEND:-CLASSIC}"
export HOLOSOMA_MUJOCO_NOSLIP_ITERATIONS="${HOLOSOMA_MUJOCO_NOSLIP_ITERATIONS:-0}"
export HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST=0
export HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST=0
export HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST=0
export HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES=0
export HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES=0
export HOLOSOMA_MUJOCO_RESET_NOISE=0
if [[ -z "${GT_MUJOCO_PHYSICS+x}" && -z "${HOLOSOMA_GT_MUJOCO_PHYSICS+x}" ]]; then
  export GT_MUJOCO_PHYSICS=0
  export HOLOSOMA_GT_MUJOCO_PHYSICS=0
else
  export GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-${HOLOSOMA_GT_MUJOCO_PHYSICS:-0}}"
fi
if is_truthy_env "$GT_MUJOCO_PHYSICS"; then
  export GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS=0
  export SIM_USE_TRAINING_URDF_OBJECT_SCENE=1
  export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML=0
  export SIM_COPY_TENDONS_FROM_ROBOT_XML=0
  export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML=0
  export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML=0
  export MUJOCO_OBJECT_MASS_SCALE=""
  export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
  export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.6,0.02,0.005]}"
  export MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.6,0.02,0.005]}"
  export MUJOCO_OBJECT_LATERAL_FRICTION=""
  export MUJOCO_OBJECT_ROLLING_FRICTION=""
  export MUJOCO_OBJECT_CONTACT_STIFFNESS=""
  export MUJOCO_OBJECT_CONTACT_DAMPING=""
fi
export HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE="${HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE:-1}"
if [[ "$PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" && -z "${SIM_DEVICE:-}" ]]; then
  HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH="${HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH:-1}"
  if is_truthy_env "$HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH" && [[ "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
      export SIM_DEVICE="${HOLOSOMA_TRAINING_DEPTH_DEVICE:-cuda:0}"
    fi
  fi
fi
if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  case "$(printf '%s' "${TRAINING_HEADLESS:-${HEADLESS:-True}}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      export MUJOCO_GL=glfw
      ;;
    *)
      export MUJOCO_GL=egl
      ;;
  esac
fi
if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" ]]; then
  export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_ROBOT_VISUAL_MESHES:-0}"
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD="${HOLOSOMA_MUJOCO_RENDERED_DEPTH_FLIPUD:-0}"
fi
if [[ -z "${INFERENCE_CONFIG:-}" ]]; then
  INFERENCE_CONFIG="$(
    "$INFER_PYTHON_BIN" "$ROOT_DIR/scripts/mj_infer_inference_config.py" "$MODEL_LOCAL"
  )"
fi
export INFERENCE_CONFIG
export RUN_SECONDS="${RUN_SECONDS:-0}"

echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] object_urdf=$OBJECT_URDF"
echo "[INFO] model=$MODEL_LOCAL"
echo "[INFO] inference_config=${INFERENCE_CONFIG}"
echo "[INFO] perception=${ENABLE_SPLIT_PERCEPTION_OBS} preset=${PERCEPTION_PRESET} camera_source=${PERCEPTION_CAMERA_SOURCE} object_geometry_mode=${PERCEPTION_OBJECT_GEOMETRY_MODE:-<default>}"
echo "[INFO] mujoco_object_scene training_urdf=${SIM_USE_TRAINING_URDF_OBJECT_SCENE} copy_joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} copy_tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} copy_collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} copy_contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML}"
echo "[INFO] object_contact_body_markers=${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-<all robot collision bodies>}"
echo "[INFO] object_mass_override=${MUJOCO_OBJECT_MASS_OVERRIDE:-<none>} object_geom_friction=${MUJOCO_OBJECT_GEOM_FRICTION:-<none>} object_terrain_pair_friction=${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-<none>} lateral_friction=${MUJOCO_OBJECT_LATERAL_FRICTION:-<none>} rolling_friction=${MUJOCO_OBJECT_ROLLING_FRICTION:-<none>} contact_stiffness=${MUJOCO_OBJECT_CONTACT_STIFFNESS:-<none>} contact_damping=${MUJOCO_OBJECT_CONTACT_DAMPING:-<none>} resize_object_to_motion_size=${MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE} collision_mode=${MUJOCO_OBJECT_COLLISION_MODE} web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS} keep_reference_hand_collision=${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION} carry_arm_object_contacts=${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS}"
echo "[INFO] track_alignment sim=${SIM_FPS}Hz decimation=${SIM_CONTROL_DECIMATION} substeps=${SIM_SUBSTEPS} backend=${MUJOCO_BACKEND} terrain_friction=${TERRAIN_STATIC_FRICTION},${TERRAIN_DYNAMIC_FRICTION} motion_index_offset=${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET} prefill_history=${HOLOSOMA_PREFILL_OBS_HISTORY_ON_MOTION_START} lowcmd_lockstep=${HOLOSOMA_ZMQ_LOWCMD_LOCKSTEP_CONTROL_BOUNDARY} lowcmd_match_tolerance_ms=${HOLOSOMA_ZMQ_LOWCMD_MATCH_TOLERANCE_MS}"
echo "[INFO] contact_alignment training_object_contact_pairs=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS} contact_material=${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION:-<object-default>},${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION:-<scene-default>},${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION:-<scene-default>} assists=root:${HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST},dof:${HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST},object:${HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST} helper_spheres=wrist:${HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES},palm:${HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES} reset_noise=${HOLOSOMA_MUJOCO_RESET_NOISE} clip_joint_targets=${HOLOSOMA_CLIP_JOINT_TARGETS}"
echo "[INFO] gt_mujoco_physics=${GT_MUJOCO_PHYSICS} zero_passive_dynamics=${HOLOSOMA_GT_MUJOCO_ZERO_PASSIVE_DYNAMICS:-0}"

if is_truthy_env "${DRY_RUN:-0}"; then
  export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
fi

exec bash "$ROOT_DIR/mj_track.sh" "$MOTION_FILE" "$MODEL_LOCAL" "${EXTRA_ARGS[@]}"
