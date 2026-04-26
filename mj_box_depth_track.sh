#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="$ROOT_DIR/outputs/motion_bank_success_box_0_92_0p3"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-https://wandb.ai/zihanw22/boxer/runs/shoo7sr1/model_29999.onnx}"
DEFAULT_OBJECT_MAP="$DEFAULT_MOTION_DIR/_clip_object_urdf_map.json"
DEFAULT_PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE_DEFAULT:-far_tracking_warp}"

INFER_PYTHON_BIN="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
if [[ ! -x "$INFER_PYTHON_BIN" ]]; then
  INFER_PYTHON_BIN="${INFER_PYTHON_BIN:-python3}"
fi

usage() {
  cat <<EOF
Usage:
  MOTION_DIR=/path/to/rollout_motion_bank OBJECT_URDF=/path/to/object.urdf bash mj_box_depth_track.sh [depth|rendered|warp] [clip_name|motion.npz] [model.onnx|wandb://...] [viser args...]

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
  GT_MUJOCO_PHYSICS=1 forces GT-style object/G1/floor MuJoCo physics
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

OBJECT_URDF_RESOLVED="$(
  "$INFER_PYTHON_BIN" - <<'PY' "$OBJECT_MAP_INPUT" "$MOTION_FILE" "$ROOT_DIR"
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).expanduser().resolve()
stem = motion_path.stem

def object_urdf_fallbacks(path: Path):
    """Yield current-repo fallbacks for older absolute box-object paths."""
    expanded = path.expanduser()
    parts = expanded.parts
    if "data" in parts:
        data_idx = parts.index("data")
        yield repo_root.joinpath(*parts[data_idx:])

    name = expanded.stem
    if name:
        if "__" in name:
            names = [name]
        else:
            names = [f"{name}__eff10", f"{name}__eff09", f"{name}__baseline"]
        for candidate_name in names:
            yield repo_root / "data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared/_generated_urdfs" / f"{candidate_name}.urdf"


def resolve_existing_urdf(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_file():
        return maybe_write_motion_sized_urdf(candidate.resolve())
    for fallback in object_urdf_fallbacks(candidate):
        if fallback.is_file():
            return maybe_write_motion_sized_urdf(fallback.resolve())
    return maybe_write_motion_sized_urdf(candidate.resolve())


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


def maybe_write_motion_sized_urdf(urdf_path: Path) -> Path:
    desired_size = _motion_object_size()
    if desired_size is None or not urdf_path.is_file():
        return urdf_path

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
    if np.allclose(current_size, desired_size, rtol=0.02, atol=2.0e-3):
        return urdf_path

    scale_ratio = desired_size / current_size
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

    out_dir = repo_root / "logs/sim2sim_exports/object_urdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{motion_path.stem}__{urdf_path.stem}__motion_size.urdf"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(
        f"[INFO] generated motion-sized object URDF {out_path}: current_size={current_size.tolist()} desired_size={desired_size.tolist()} scale_ratio={scale_ratio.tolist()}",
        file=sys.stderr,
    )
    return out_path.resolve()


candidate = Path(raw).expanduser() if raw else None
if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".json":
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
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
export PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-$DEFAULT_PERCEPTION_CAMERA_SOURCE}"
export PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
export MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-}"
export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
export MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
export MUJOCO_OBJECT_LATERAL_FRICTION="${MUJOCO_OBJECT_LATERAL_FRICTION:-}"
export MUJOCO_OBJECT_ROLLING_FRICTION="${MUJOCO_OBJECT_ROLLING_FRICTION:-}"
export MUJOCO_OBJECT_CONTACT_STIFFNESS="${MUJOCO_OBJECT_CONTACT_STIFFNESS:-}"
export MUJOCO_OBJECT_CONTACT_DAMPING="${MUJOCO_OBJECT_CONTACT_DAMPING:-}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
export GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-${HOLOSOMA_GT_MUJOCO_PHYSICS:-1}}"
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
  export MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.4}"
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
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-29dof-wbt-object-distill}"
export RUN_SECONDS="${RUN_SECONDS:-0}"

echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] object_urdf=$OBJECT_URDF"
echo "[INFO] model=$MODEL_LOCAL"
echo "[INFO] inference_config=$INFERENCE_CONFIG"
echo "[INFO] perception=${ENABLE_SPLIT_PERCEPTION_OBS} preset=${PERCEPTION_PRESET} camera_source=${PERCEPTION_CAMERA_SOURCE}"
echo "[INFO] object_contact_body_markers=${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-<all robot collision bodies>}"
echo "[INFO] object_mass_override=${MUJOCO_OBJECT_MASS_OVERRIDE:-<none>} object_geom_friction=${MUJOCO_OBJECT_GEOM_FRICTION:-<none>} object_terrain_pair_friction=${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-<none>} lateral_friction=${MUJOCO_OBJECT_LATERAL_FRICTION:-<none>} rolling_friction=${MUJOCO_OBJECT_ROLLING_FRICTION:-<none>} contact_stiffness=${MUJOCO_OBJECT_CONTACT_STIFFNESS:-<none>} contact_damping=${MUJOCO_OBJECT_CONTACT_DAMPING:-<none>} web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS}"
echo "[INFO] gt_mujoco_physics=${GT_MUJOCO_PHYSICS} zero_passive_dynamics=${HOLOSOMA_GT_MUJOCO_ZERO_PASSIVE_DYNAMICS:-0}"

if is_truthy_env "${DRY_RUN:-0}"; then
  export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
fi

exec bash "$ROOT_DIR/mj_track.sh" "$MOTION_FILE" "$MODEL_LOCAL" "${EXTRA_ARGS[@]}"
