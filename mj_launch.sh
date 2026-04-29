#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-$ROOT_DIR/data_demo}"
DEFAULT_CLIP="${DEFAULT_CLIP:-box_75}"
DEFAULT_MUJOCO_PY="/home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python"

export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"

usage() {
  cat <<EOF
Usage:
  bash mj_launch.sh [rendered848|rendered] [clip_name|motion.npz]

Purpose:
  Launch only the native MuJoCo simulator/window for split rollout.
  This script intentionally uses only the hsmujoco Python environment.

Examples:
  bash mj_launch.sh box_75
  HEADLESS=False bash mj_launch.sh rendered848 box_75

Environment:
  MUJOCO_PY             default: ${DEFAULT_MUJOCO_PY}
  MOTION_DIR            default: ${DEFAULT_MOTION_DIR}
  OBJECT_URDF           optional URDF or _clip_object_urdf_map.json
  HEADLESS              default: False
  RUN_SECONDS           default: 0 (forever)
  SIM_STATE_PORT        default: 5657
  SIM_CONTROL_PORT      default: 5659
  PERCEPTION_OBS_PORT   default: 5658
EOF
}

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_bool_flag() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      printf 'True\n'
      ;;
    0|false|no|off)
      printf 'False\n'
      ;;
    *)
      echo "[ERROR] Expected boolean True/False/1/0, got: ${1:-}" >&2
      exit 2
      ;;
  esac
}

python_has_modules() {
  local python_bin="$1"
  shift
  "$python_bin" - "$@" <<'PY' >/dev/null 2>&1
import importlib
import sys

for module_name in sys.argv[1:]:
    importlib.import_module(module_name)
raise SystemExit(0)
PY
}

resolve_hsmujoco_python() {
  local configured="${MUJOCO_PY:-$DEFAULT_MUJOCO_PY}"
  if [[ ! -x "$configured" ]]; then
    echo "[ERROR] MUJOCO_PY is not executable: $configured" >&2
    exit 1
  fi
  if ! python_has_modules "$configured" mujoco holosoma torch tyro typeguard numpy; then
    echo "[ERROR] MUJOCO_PY must be the hsmujoco env with mujoco/holosoma/torch/tyro/typeguard/numpy: $configured" >&2
    exit 1
  fi
  printf '%s\n' "$configured"
}

resolve_motion_file() {
  local motion_dir="$1"
  local clip="$2"
  if [[ "$clip" == *.npz ]]; then
    if [[ -f "$clip" ]]; then
      realpath "$clip"
      return 0
    fi
    if [[ -f "$motion_dir/${clip##*/}" ]]; then
      realpath "$motion_dir/${clip##*/}"
      return 0
    fi
  elif [[ -f "$motion_dir/${clip}.npz" ]]; then
    realpath "$motion_dir/${clip}.npz"
    return 0
  fi
  echo "[ERROR] Motion clip not found: $clip (MOTION_DIR=$motion_dir)" >&2
  exit 1
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MODE="rendered848"
MOTION_CLIP="${MOTION_CLIP_NAME:-${MOTION_CLIP:-$DEFAULT_CLIP}}"
POSITIONAL_MODE=1
for arg in "$@"; do
  if [[ "$POSITIONAL_MODE" == "0" ]]; then
    continue
  fi
  case "$arg" in
    rendered848|render848|mujoco848|mujoco_render_848x480)
      MODE="rendered848"
      ;;
    rendered|render|mujoco)
      MODE="rendered"
      ;;
    --*)
      POSITIONAL_MODE=0
      ;;
    *.onnx|*.pt|wandb://*|https://*)
      ;;
    *.npz)
      MOTION_FILE="$arg"
      ;;
    *)
      MOTION_CLIP="$arg"
      ;;
  esac
done

MUJOCO_PY="$(resolve_hsmujoco_python)"
MOTION_DIR="$(realpath "${MOTION_DIR:-$DEFAULT_MOTION_DIR}")"
MOTION_FILE="${MOTION_FILE:-}"
if [[ -n "$MOTION_FILE" ]]; then
  MOTION_FILE="$(resolve_motion_file "$MOTION_DIR" "$MOTION_FILE")"
else
  MOTION_FILE="$(resolve_motion_file "$MOTION_DIR" "$MOTION_CLIP")"
fi
MOTION_STEM="$(basename "${MOTION_FILE%.npz}")"

OBJECT_MAP_INPUT="${OBJECT_URDF:-}"
if [[ -z "$OBJECT_MAP_INPUT" && -f "$MOTION_DIR/_clip_object_urdf_map.json" ]]; then
  OBJECT_MAP_INPUT="$MOTION_DIR/_clip_object_urdf_map.json"
fi

SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm_${SIM_STATE_PORT}}"
SIM_FPS="${SIM_FPS:-500}"
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-10}"
SIM_SUBSTEPS="${SIM_SUBSTEPS:-1}"
MUJOCO_BACKEND="${MUJOCO_BACKEND:-CLASSIC}"
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
TRAINING_HEADLESS="$(normalize_bool_flag "${TRAINING_HEADLESS:-${HEADLESS:-False}}")"
RUN_SECONDS="${RUN_SECONDS:-0}"
MJ_RUNTIME_DIR="${MJ_RUNTIME_DIR:-/tmp/holosoma_mj_assets}"

PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i_mujoco_render_848x480}"
PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-rendered}"
PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
PERCEPTION_CAMERA_WIDTH="${PERCEPTION_CAMERA_WIDTH:-848}"
PERCEPTION_CAMERA_HEIGHT="${PERCEPTION_CAMERA_HEIGHT:-480}"
PERCEPTION_CAMERA_WARP_CROP_TOP="${PERCEPTION_CAMERA_WARP_CROP_TOP:-16}"
PERCEPTION_CAMERA_WARP_CROP_BOTTOM="${PERCEPTION_CAMERA_WARP_CROP_BOTTOM:-0}"
PERCEPTION_CAMERA_WARP_CROP_LEFT="${PERCEPTION_CAMERA_WARP_CROP_LEFT:-32}"
PERCEPTION_CAMERA_WARP_CROP_RIGHT="${PERCEPTION_CAMERA_WARP_CROP_RIGHT:-32}"
PERCEPTION_CAMERA_PITCH_DEG="${PERCEPTION_CAMERA_PITCH_DEG:-10}"
PERCEPTION_CAMERA_VFOV_DEG="${PERCEPTION_CAMERA_VFOV_DEG:-58.6}"
PERCEPTION_CAMERA_HFOV_DEG="${PERCEPTION_CAMERA_HFOV_DEG:-89.5}"
PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH="${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-True}"
PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-0.1}"
PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-3}"
PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-3}"
PERCEPTION_UPDATE_HZ="${PERCEPTION_UPDATE_HZ:-30}"
PERCEPTION_CAMERA_FPS="${PERCEPTION_CAMERA_FPS:-30}"

if [[ "$MODE" == "rendered" ]]; then
  PERCEPTION_CAMERA_WIDTH="${PERCEPTION_CAMERA_WIDTH_RENDERED:-106}"
  PERCEPTION_CAMERA_HEIGHT="${PERCEPTION_CAMERA_HEIGHT_RENDERED:-60}"
  PERCEPTION_CAMERA_WARP_CROP_TOP="${PERCEPTION_CAMERA_WARP_CROP_TOP_RENDERED:-2}"
  PERCEPTION_CAMERA_WARP_CROP_LEFT="${PERCEPTION_CAMERA_WARP_CROP_LEFT_RENDERED:-4}"
  PERCEPTION_CAMERA_WARP_CROP_RIGHT="${PERCEPTION_CAMERA_WARP_CROP_RIGHT_RENDERED:-4}"
fi

if [[ "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  if [[ "$TRAINING_HEADLESS" == "False" ]]; then
    export MUJOCO_GL=glfw
  else
    export MUJOCO_GL=egl
  fi
fi

export PYTHONPATH="$ROOT_DIR/src/holosoma${PYTHONPATH:+:$PYTHONPATH}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-1}"
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION="${HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION:-1}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-1}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS:-1}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION:-1.2}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION:-0.005}"
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION="${HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION:-0.3}"
export HOLOSOMA_MUJOCO_RESET_NOISE=0
export HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES="${HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES:-0}"
export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
export HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}"
export HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES:-1}"

MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
if [[ -z "$MUJOCO_OBJECT_MASS_OVERRIDE" ]]; then
  if [[ "$MOTION_STEM" == "box_75" ]]; then
    MUJOCO_OBJECT_MASS_OVERRIDE=2.0
  else
    MUJOCO_OBJECT_MASS_OVERRIDE=1.0
  fi
fi

OBJECT_URDF_RESOLVED="$(
  "$MUJOCO_PY" - <<'PY' "$OBJECT_MAP_INPUT" "$MOTION_FILE" "$ROOT_DIR" "$MJ_RUNTIME_DIR"
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

raw = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).expanduser().resolve()
runtime_dir = Path(sys.argv[4]).expanduser().resolve()
stem = motion_path.stem
relative_roots = [motion_path.parent, repo_root]


def _parse_vec(value, default):
    if not value:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray([float(part) for part in str(value).replace(",", " ").split()], dtype=np.float64)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return arr


def _obj_extents(path):
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return None
    arr = np.asarray(vertices, dtype=np.float64)
    return arr.max(axis=0) - arr.min(axis=0)


def _resolve_path(path_str, *, base=None):
    candidate = Path(path_str).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        if base is not None:
            candidates.append(base / candidate)
        candidates.extend(root / candidate for root in relative_roots)
    name = candidate.stem
    if name:
        candidates.append(repo_root / "data_demo" / "objects" / f"{name}.urdf")
    for current in candidates:
        if current.is_file():
            return current.resolve()
    return candidates[-1].resolve()


def _object_urdf_from_spec():
    spec = Path(raw).expanduser() if raw else None
    if spec is not None and spec.is_file() and spec.suffix.lower() == ".json":
        relative_roots.insert(0, spec.parent.resolve())
        data = json.loads(spec.read_text())
        clips = data.get("clips", data) if isinstance(data, dict) else {}
        entry = clips.get(stem) if isinstance(clips, dict) else None
        if not isinstance(entry, dict):
            raise SystemExit(f"Object map has no entry for clip {stem!r}: {spec}")
        path = entry.get("object_urdf_path") or entry.get("urdf_path")
        if not path:
            raise SystemExit(f"Object map entry for clip {stem!r} has no object_urdf_path")
        return _resolve_path(str(path), base=spec.parent)
    if spec is not None and str(spec):
        return _resolve_path(str(spec))
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" in data:
            return _resolve_path(str(np.asarray(data["object_urdf_path"]).item()))
    return _resolve_path(f"data_demo/objects/{stem}.urdf")


def _motion_object_size():
    with np.load(motion_path, allow_pickle=True) as data:
        size = np.asarray(data["object_size"], dtype=np.float64).reshape(-1)
    if size.size != 3 or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise SystemExit(f"Invalid object_size in {motion_path}")
    return size


def _mesh_path(urdf_path, filename):
    mesh = Path(filename).expanduser()
    if not mesh.is_absolute():
        mesh = urdf_path.parent / mesh
    return mesh.resolve()


def _set_collision_box(root, size):
    link = root.find(".//link")
    if link is None:
        return
    for collision in list(link.findall("collision")):
        link.remove(collision)
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": " ".join(f"{value:.9g}" for value in size)})


urdf_path = _object_urdf_from_spec()
tree = ET.parse(urdf_path)
root = tree.getroot()
mesh_elems = root.findall(".//mesh")
if not mesh_elems:
    print(urdf_path.resolve())
    raise SystemExit(0)

first_mesh = mesh_elems[0]
first_mesh_path = _mesh_path(urdf_path, str(first_mesh.get("filename") or ""))
extents = _obj_extents(first_mesh_path)
if extents is None:
    print(urdf_path.resolve())
    raise SystemExit(0)

first_scale = _parse_vec(first_mesh.get("scale"), (1.0, 1.0, 1.0))
current_size = extents * first_scale
desired_size = _motion_object_size()
scale_ratio = desired_size / current_size
for mesh in mesh_elems:
    filename = str(mesh.get("filename") or "").strip()
    if filename:
        mesh.set("filename", str(_mesh_path(urdf_path, filename)))
    old_scale = _parse_vec(mesh.get("scale"), (1.0, 1.0, 1.0))
    mesh.set("scale", " ".join(f"{value:.9g}" for value in old_scale * scale_ratio))
_set_collision_box(root, desired_size)

runtime_dir.mkdir(parents=True, exist_ok=True)
out_path = runtime_dir / f"{stem}__{urdf_path.stem}__motion_size_box_collision.urdf"
tree.write(out_path, encoding="utf-8", xml_declaration=True)
print(out_path.resolve())
PY
)"

CMD=(
  "$MUJOCO_PY" -u "$ROOT_DIR/src/holosoma/holosoma/run_sim.py"
  simulator:mujoco
  robot:g1_29dof_w_object
  terrain:terrain_locomotion_plane
  "perception:${PERCEPTION_PRESET}"
  --training.headless "$TRAINING_HEADLESS"
  --simulator.config.debug-viz True
  --simulator.config.sim.fps "$SIM_FPS"
  --simulator.config.sim.control-decimation "$SIM_CONTROL_DECIMATION"
  --simulator.config.sim.substeps "$SIM_SUBSTEPS"
  --simulator.config.mujoco-backend "$MUJOCO_BACKEND"
  --robot.object.enabled=True
  --robot.object.object-urdf-path "$OBJECT_URDF_RESOLVED"
  --robot.object.mujoco-use-training-urdf-scene True
  --robot.object.mujoco-add-default-actuators True
  --robot.object.mujoco-copy-joint-defaults-from-robot-xml True
  --robot.object.mujoco-copy-tendons-from-robot-xml True
  --robot.object.mujoco-copy-collision-geoms-from-robot-xml True
  --robot.object.mujoco-copy-contact-pairs-from-robot-xml True
  --robot.object.mujoco-object-mass-override "$MUJOCO_OBJECT_MASS_OVERRIDE"
  --terrain.terrain-term.static-friction "${TERRAIN_STATIC_FRICTION:-1.0}"
  --terrain.terrain-term.dynamic-friction "${TERRAIN_DYNAMIC_FRICTION:-1.0}"
  --simulator.config.bridge.interface "$INTERFACE_NAME"
  --simulator.config.bridge.clock-port "$SIM_CLOCK_PORT"
  --simulator.config.bridge.publish-sim-state=True
  --simulator.config.bridge.listen-control=True
  --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT"
  --simulator.config.bridge.control-port "$SIM_CONTROL_PORT"
  --simulator.config.bridge.publish-perception-obs True
  --simulator.config.bridge.perception-obs-port "$PERCEPTION_OBS_PORT"
  --simulator.config.bridge.publish-perception-obs-shm True
  --simulator.config.bridge.perception-obs-shm-name "$PERCEPTION_OBS_SHM_NAME"
  --simulator.config.bridge.use-zmq-lowcmd True
  --simulator.config.bridge.ignore-default-idle-command True
  --simulator.config.bridge.freeze-until-first-command True
  --motion-init.enabled=True
  --motion-init.motion-file "$MOTION_FILE"
  --motion-init.mode "$SIM_MOTION_INIT_MODE"
  --motion-init.object-name object
  --perception.camera-source "$PERCEPTION_CAMERA_SOURCE"
  --perception.object-geometry-mode "$PERCEPTION_OBJECT_GEOMETRY_MODE"
  --perception.camera-width "$PERCEPTION_CAMERA_WIDTH"
  --perception.camera-height "$PERCEPTION_CAMERA_HEIGHT"
  --perception.camera-warp-crop-top "$PERCEPTION_CAMERA_WARP_CROP_TOP"
  --perception.camera-warp-crop-bottom "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM"
  --perception.camera-warp-crop-left "$PERCEPTION_CAMERA_WARP_CROP_LEFT"
  --perception.camera-warp-crop-right "$PERCEPTION_CAMERA_WARP_CROP_RIGHT"
  --perception.camera-pitch-deg "$PERCEPTION_CAMERA_PITCH_DEG"
  --perception.camera-vfov-deg "$PERCEPTION_CAMERA_VFOV_DEG"
  --perception.camera-hfov-deg "$PERCEPTION_CAMERA_HFOV_DEG"
  --perception.camera-include-robot-mesh "$PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH"
  --perception.camera-near "$PERCEPTION_CAMERA_NEAR"
  --perception.camera-far "$PERCEPTION_CAMERA_FAR"
  --perception.max-distance "$PERCEPTION_MAX_DISTANCE"
  --perception.update-hz "$PERCEPTION_UPDATE_HZ"
  --perception.camera-fps "$PERCEPTION_CAMERA_FPS"
)

echo "[INFO] launching native MuJoCo only"
echo "[INFO] python=$MUJOCO_PY"
echo "[INFO] motion_file=$MOTION_FILE"
echo "[INFO] object_urdf=$OBJECT_URDF_RESOLVED"
echo "[INFO] headless=$TRAINING_HEADLESS mujoco_gl=${MUJOCO_GL:-<unset>}"
echo "[INFO] ports clock=${SIM_CLOCK_PORT} state=${SIM_STATE_PORT} perception=${PERCEPTION_OBS_PORT} control=${SIM_CONTROL_PORT}"
echo "[INFO] start policy with: bash $ROOT_DIR/mj_rollout.sh ${MODE} ${MOTION_STEM}"

if is_truthy "${DRY_RUN:-0}"; then
  printf '[DRY_RUN] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ "$RUN_SECONDS" == "0" ]]; then
  exec "${CMD[@]}"
else
  exec timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${CMD[@]}"
fi
