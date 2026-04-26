#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_FILE:-$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# A repo-root MuJoCo source checkout (`mujoco/`) can shadow the installed
# Python binding when cwd is prepended to sys.path. We provide project modules
# through PYTHONPATH explicitly, so keep cwd out of Python import resolution.
export PYTHONSAFEPATH="${PYTHONSAFEPATH:-1}"

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

if [[ "${HOLOSOMA_MJ_TRACK_INTERNAL_CORE:-0}" != "1" ]]; then
  usage() {
    cat <<EOF
Usage:
  bash mj_track.sh [motion.npz] [checkpoint.pt|model.onnx] [viser args...]

Defaults:
  motion = ${DEFAULT_MOTION_FILE}
  model  = ${DEFAULT_MODEL_INPUT}
EOF
  }

  case "${1:-}" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac

  MOTION_FILE="${DEFAULT_MOTION_FILE}"
  MODEL_INPUT="${DEFAULT_MODEL_INPUT}"
  EXTRA_ARGS=()
  POSITIONAL_MODE=1

  for arg in "$@"; do
    if [[ "${POSITIONAL_MODE}" == "1" && "${arg}" != -* ]]; then
      if [[ "${MOTION_FILE}" == "${DEFAULT_MOTION_FILE}" ]]; then
        MOTION_FILE="${arg}"
        continue
      fi
      if [[ "${MODEL_INPUT}" == "${DEFAULT_MODEL_INPUT}" ]]; then
        MODEL_INPUT="${arg}"
        continue
      fi
    fi
    POSITIONAL_MODE=0
    EXTRA_ARGS+=("${arg}")
  done

  export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
  export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1

  exec "$PYTHON_BIN" "$ROOT_DIR/src/holosoma/holosoma/viser_mujoco_sim_state.py" \
    --launch-rollout \
    --run-script "$ROOT_DIR/mj_track.sh" \
    --motion-file "$MOTION_FILE" \
    --model-path "$MODEL_INPUT" \
    "${EXTRA_ARGS[@]}"
fi

usage() {
  cat <<EOF
Usage:
  bash mj_track.sh [motion.npz] [checkpoint.pt|model.onnx]

Defaults:
  motion = ${DEFAULT_MOTION_FILE}
  model  = ${DEFAULT_MODEL_INPUT}

Environment:
  GT_MUJOCO_PHYSICS=1  force GT-style object/G1/floor MuJoCo physics
EOF
}

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MOTION_FILE="${1:-$DEFAULT_MOTION_FILE}"
MODEL_INPUT="${2:-$DEFAULT_MODEL_INPUT}"

MUJOCO_PY="${MUJOCO_PY:-}"
INFER_PY="${INFER_PY:-}"
MUJOCO_CPUSET="${MUJOCO_CPUSET:-0}"
SIM_FPS_EXPLICIT=0
[[ -n "${SIM_FPS+x}" ]] && SIM_FPS_EXPLICIT=1
SIM_FPS="${SIM_FPS:-500}"
SIM_CONTROL_DECIMATION_EXPLICIT=0
[[ -n "${SIM_CONTROL_DECIMATION+x}" ]] && SIM_CONTROL_DECIMATION_EXPLICIT=1
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-4}"
SIM_SUBSTEPS_EXPLICIT=0
[[ -n "${SIM_SUBSTEPS+x}" ]] && SIM_SUBSTEPS_EXPLICIT=1
SIM_SUBSTEPS="${SIM_SUBSTEPS:-}"
SIM_DEVICE="${SIM_DEVICE:-}"
MUJOCO_BACKEND="${MUJOCO_BACKEND:-}"
TERRAIN_STATIC_FRICTION="${TERRAIN_STATIC_FRICTION:-}"
TERRAIN_DYNAMIC_FRICTION="${TERRAIN_DYNAMIC_FRICTION:-}"
TERRAIN_STATIC_FRICTION_EXPLICIT=0
TERRAIN_DYNAMIC_FRICTION_EXPLICIT=0
[[ -n "${TERRAIN_STATIC_FRICTION+x}" ]] && [[ -n "$TERRAIN_STATIC_FRICTION" ]] && TERRAIN_STATIC_FRICTION_EXPLICIT=1
[[ -n "${TERRAIN_DYNAMIC_FRICTION+x}" ]] && [[ -n "$TERRAIN_DYNAMIC_FRICTION" ]] && TERRAIN_DYNAMIC_FRICTION_EXPLICIT=1
SIM_VIRTUAL_GANTRY_ENABLED="${SIM_VIRTUAL_GANTRY_ENABLED:-False}"
SIM_MOTION_INIT_MODE_EXPLICIT=0
[[ -n "${SIM_MOTION_INIT_MODE+x}" ]] && SIM_MOTION_INIT_MODE_EXPLICIT=1
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS:-0}"
USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
SIM_IGNORE_DEFAULT_IDLE_COMMAND="${SIM_IGNORE_DEFAULT_IDLE_COMMAND:-1}"
SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND:-}"
SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND:-}"
SIM_FREEZE_UNTIL_FIRST_COMMAND="${SIM_FREEZE_UNTIL_FIRST_COMMAND:-}"
SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5655}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5657}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5659}"
PERCEPTION_OBS_PORT="${PERCEPTION_OBS_PORT:-5658}"
SPARSE_ROOT_COMMAND_PORT="${SPARSE_ROOT_COMMAND_PORT:-5661}"
POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-}"
ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-auto}"
ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-0}"
PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
PERCEPTION_CAMERA_WIDTH_EXPLICIT=0
PERCEPTION_CAMERA_HEIGHT_EXPLICIT=0
PERCEPTION_CAMERA_WARP_CROP_TOP_EXPLICIT=0
PERCEPTION_CAMERA_WARP_CROP_BOTTOM_EXPLICIT=0
PERCEPTION_CAMERA_WARP_CROP_LEFT_EXPLICIT=0
PERCEPTION_CAMERA_WARP_CROP_RIGHT_EXPLICIT=0
[[ -n "${PERCEPTION_CAMERA_WIDTH+x}" ]] && PERCEPTION_CAMERA_WIDTH_EXPLICIT=1
[[ -n "${PERCEPTION_CAMERA_HEIGHT+x}" ]] && PERCEPTION_CAMERA_HEIGHT_EXPLICIT=1
[[ -n "${PERCEPTION_CAMERA_WARP_CROP_TOP+x}" ]] && PERCEPTION_CAMERA_WARP_CROP_TOP_EXPLICIT=1
[[ -n "${PERCEPTION_CAMERA_WARP_CROP_BOTTOM+x}" ]] && PERCEPTION_CAMERA_WARP_CROP_BOTTOM_EXPLICIT=1
[[ -n "${PERCEPTION_CAMERA_WARP_CROP_LEFT+x}" ]] && PERCEPTION_CAMERA_WARP_CROP_LEFT_EXPLICIT=1
[[ -n "${PERCEPTION_CAMERA_WARP_CROP_RIGHT+x}" ]] && PERCEPTION_CAMERA_WARP_CROP_RIGHT_EXPLICIT=1
PERCEPTION_CAMERA_WIDTH="${PERCEPTION_CAMERA_WIDTH:-}"
PERCEPTION_CAMERA_HEIGHT="${PERCEPTION_CAMERA_HEIGHT:-}"
PERCEPTION_CAMERA_WARP_CROP_TOP="${PERCEPTION_CAMERA_WARP_CROP_TOP:-}"
PERCEPTION_CAMERA_WARP_CROP_BOTTOM="${PERCEPTION_CAMERA_WARP_CROP_BOTTOM:-}"
PERCEPTION_CAMERA_WARP_CROP_LEFT="${PERCEPTION_CAMERA_WARP_CROP_LEFT:-}"
PERCEPTION_CAMERA_WARP_CROP_RIGHT="${PERCEPTION_CAMERA_WARP_CROP_RIGHT:-}"
PERCEPTION_CAMERA_PITCH_DEG="${PERCEPTION_CAMERA_PITCH_DEG:-}"
PERCEPTION_CAMERA_VFOV_DEG="${PERCEPTION_CAMERA_VFOV_DEG:-}"
PERCEPTION_CAMERA_HFOV_DEG="${PERCEPTION_CAMERA_HFOV_DEG:-}"
PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-}"
PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-}"
PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-}"
PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH="${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-}"
PERCEPTION_UPDATE_HZ="${PERCEPTION_UPDATE_HZ:-}"
PERCEPTION_CAMERA_FPS="${PERCEPTION_CAMERA_FPS:-}"
PERCEPTION_CAMERA_WARP_BUFFER_LEN="${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-}"
PERCEPTION_CAMERA_WARP_LATENCY_FRAME="${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-}"
PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-False}"
PERCEPTION_CAMERA_WARP_ENABLE_HOLES="${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-False}"
PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-False}"
PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH="${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-}"
PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-training}"
PERCEPTION_OBS_TRANSPORT="${PERCEPTION_OBS_TRANSPORT:-shm}"
PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm}"
PERCEPTION_OBS_EXTERNAL="${PERCEPTION_OBS_EXTERNAL:-0}"
SIM_USE_ZMQ_LOWCMD="${SIM_USE_ZMQ_LOWCMD:-1}"
SKIP_POLICY="${SKIP_POLICY:-0}"
MJ_TRACK_MODE="${MJ_TRACK_MODE:-both}"
POLICY_STDIO="${POLICY_STDIO:-}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
RUN_SECONDS="${RUN_SECONDS:-20}"
if is_truthy_env "${HOLOSOMA_MJ_TRACK_RUN_FOREVER:-0}"; then
  RUN_SECONDS=0
fi
TRAINING_HEADLESS="${TRAINING_HEADLESS:-True}"
SIM_DEBUG_VIZ="${SIM_DEBUG_VIZ:-True}"
MUJOCO_SHOW_OBJECT_COLLISION="${MUJOCO_SHOW_OBJECT_COLLISION:-0}"
MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION="${MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION:-0}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-180}"
SIM_READY_PATTERN="${SIM_READY_PATTERN:-Starting direct simulation loop...}"
SIM_STARTUP_WAIT="${SIM_STARTUP_WAIT:-0}"
DEFAULT_OBJECT_URDF="$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
OBJECT_URDF="${OBJECT_URDF:-}"
PATCH_DIR="${PATCH_DIR:-$ROOT_DIR/logs/sim2sim_exports}"
POLICY_ACTION_SCALE="${POLICY_ACTION_SCALE:-}"
POLICY_RL_RATE="${POLICY_RL_RATE:-50}"
POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-0}"
POLICY_AUTO_START_MOTION_CLIP="${POLICY_AUTO_START_MOTION_CLIP:-}"
POLICY_MOTION_INDEX_OFFSET="${POLICY_MOTION_INDEX_OFFSET:-}"
SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY:-0}"
HOLOSOMA_ONNX_ALIGN_MAX_STEPS="${HOLOSOMA_ONNX_ALIGN_MAX_STEPS:-0}"
HOLOSOMA_ONNX_ALIGN_POSE_TOL="${HOLOSOMA_ONNX_ALIGN_POSE_TOL:-5e-3}"
HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX="${HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX:-1}"
HOLOSOMA_CLIP_JOINT_TARGETS="${HOLOSOMA_CLIP_JOINT_TARGETS:-0}"
HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE="${HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE:-}"
AUTO_START_STIFF_HOLD_SEC_RAW="${AUTO_START_STIFF_HOLD_SEC-__unset__}"
AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-}"
AUTO_START_STIFF_MAX_WAIT_SEC_RAW="${AUTO_START_STIFF_MAX_WAIT_SEC-__unset__}"
AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-}"
AUTO_START_STIFF_POSE_TOL="${AUTO_START_STIFF_POSE_TOL:-0.12}"
USE_ROOT_REFERENCE_AT_CLIP_START_RAW="${USE_ROOT_REFERENCE_AT_CLIP_START-__unset__}"
USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-}"
SIM_ADD_DEFAULT_OBJECT_ACTUATORS_RAW="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS-__unset__}"
SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-}"
SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
MUJOCO_OBJECT_MASS_SCALE="${MUJOCO_OBJECT_MASS_SCALE:-}"
MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-}"
MUJOCO_OBJECT_LATERAL_FRICTION="${MUJOCO_OBJECT_LATERAL_FRICTION:-}"
MUJOCO_OBJECT_ROLLING_FRICTION="${MUJOCO_OBJECT_ROLLING_FRICTION:-}"
MUJOCO_OBJECT_CONTACT_STIFFNESS="${MUJOCO_OBJECT_CONTACT_STIFFNESS:-}"
MUJOCO_OBJECT_CONTACT_DAMPING="${MUJOCO_OBJECT_CONTACT_DAMPING:-}"
MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-0}"
MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-}"
USE_TRAINING_OBJECT_CONTACT_MARKERS="${USE_TRAINING_OBJECT_CONTACT_MARKERS:-0}"
GT_MUJOCO_PHYSICS="${GT_MUJOCO_PHYSICS:-${HOLOSOMA_GT_MUJOCO_PHYSICS:-0}}"
PREFER_SIM_REF_FROM_SIM_STATE="${PREFER_SIM_REF_FROM_SIM_STATE:-1}"
USE_SIM_TIME="${USE_SIM_TIME:-}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-}"
ROBOT_INIT_STATE_POS="${ROBOT_INIT_STATE_POS:-}"
ROBOT_INIT_STATE_ROT="${ROBOT_INIT_STATE_ROT:-}"
ROBOT_ENABLE_SELF_COLLISIONS="${ROBOT_ENABLE_SELF_COLLISIONS:-}"
MOTION_METADATA_TOOL="$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/read_motion_clip_metadata.py"

case "$(printf '%s' "$MJ_TRACK_MODE" | tr '[:upper:]' '[:lower:]')" in
  both|env|policy)
    MJ_TRACK_MODE="$(printf '%s' "$MJ_TRACK_MODE" | tr '[:upper:]' '[:lower:]')"
    ;;
  *)
    echo "Unsupported MJ_TRACK_MODE=${MJ_TRACK_MODE}; expected both, env, or policy" >&2
    exit 2
    ;;
esac
if [[ -z "$POLICY_STDIO" ]]; then
  if [[ "$MJ_TRACK_MODE" == "policy" ]]; then
    POLICY_STDIO="inherit"
  else
    POLICY_STDIO="log"
  fi
fi
if [[ "$MJ_TRACK_MODE" == "env" ]]; then
  SKIP_POLICY=1
fi

mkdir -p "$PATCH_DIR"

MOTION_STEM="$(basename "${MOTION_FILE%.*}")"
MODEL_STEM="$(basename "${MODEL_INPUT%.*}")"
PATCHED_ONNX="$PATCH_DIR/${MODEL_STEM}__${MOTION_STEM}.onnx"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/logs/sim2sim_runs/${MOTION_STEM}__tracking}"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export HOLOSOMA_ONNX_ALIGN_MAX_STEPS
export HOLOSOMA_ONNX_ALIGN_POSE_TOL
export HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX
export HOLOSOMA_CLIP_JOINT_TARGETS
export HOLOSOMA_MUJOCO_APPLY_TRAINING_JOINT_DYNAMICS="${HOLOSOMA_MUJOCO_APPLY_TRAINING_JOINT_DYNAMICS:-1}"
if [[ -n "$POLICY_CONTROL_PORT" && -z "${HOLOSOMA_POLICY_CONTROL_PORT:-}" ]]; then
  export HOLOSOMA_POLICY_CONTROL_PORT="$POLICY_CONTROL_PORT"
fi

resolve_python() {
  local configured="$1"
  shift
  if [[ -n "$configured" ]]; then
    if [[ ! -x "$configured" ]]; then
      echo "Configured python is not executable: $configured" >&2
      exit 1
    fi
    printf '%s\n' "$configured"
    return
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No usable python interpreter found for split sim2sim launcher" >&2
  exit 1
}

python_has_modules() {
  local python_bin="$1"
  shift
  "$python_bin" - "$@" <<'PY' >/dev/null 2>&1
import importlib
import sys

for module_name in sys.argv[1:]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        raise SystemExit(1)
    if module_name == "mujoco":
        # A cloned source tree without the Python extension imports as an empty
        # namespace package. Require the actual binding APIs used by run_sim.py.
        required = ("MjModel", "MjData", "MjSpec", "mj_step")
        if any(not hasattr(module, attr) for attr in required):
            raise SystemExit(1)
    if module is None:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

resolve_python_with_modules() {
  local modules_csv="$1"
  local modules=()
  read -r -a modules <<< "$modules_csv"
  shift
  local candidate
  for candidate in "$@"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if python_has_modules "$candidate" "${modules[@]}"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "No usable python interpreter with modules '$modules_csv' found for split sim2sim launcher" >&2
  exit 1
}

if [[ -n "$MUJOCO_PY" ]]; then
  MUJOCO_PY="$(resolve_python_with_modules "mujoco holosoma torch tyro typeguard" "$MUJOCO_PY")"
else
  MUJOCO_PY="$(resolve_python_with_modules "mujoco holosoma torch tyro typeguard" \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)")"
fi
INFER_PY="$(resolve_python "$INFER_PY" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"

apply_motion_clip_object_defaults() {
  if [[ -f "$MOTION_METADATA_TOOL" ]]; then
    eval "$("$INFER_PY" "$MOTION_METADATA_TOOL" --motion-file "$MOTION_FILE" --format shell)"
    if [[ -z "$OBJECT_URDF" && -n "${SIM2SIM_CLIP_OBJECT_URDF_PATH:-}" ]]; then
      OBJECT_URDF="$SIM2SIM_CLIP_OBJECT_URDF_PATH"
    fi
  fi
  if [[ -z "$OBJECT_URDF" ]]; then
    OBJECT_URDF="$DEFAULT_OBJECT_URDF"
  fi
}

resolve_motion_sized_object_urdf() {
  local object_urdf="$1"
  "$INFER_PY" - <<'PY' "$object_urdf" "$MOTION_FILE" "$ROOT_DIR"
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

raw_path = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).expanduser().resolve()


def object_urdf_fallbacks(path):
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


def parse_vec(raw, default):
    if not raw:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(part) for part in str(raw).replace(",", " ").split()], dtype=np.float64)
    if values.size == 1:
        values = np.repeat(values, 3)
    if values.size != 3:
        raise ValueError(f"Expected 3-vector, got {raw!r}")
    return values


def obj_extents(path):
    vertices = []
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


def motion_object_size():
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


def mesh_path(urdf_path, filename):
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    return (urdf_path.parent / path).resolve()


def write_motion_sized_urdf(urdf_path):
    desired_size = motion_object_size()
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

    first_scale = parse_vec(first_mesh.get("scale"), (1.0, 1.0, 1.0))
    first_extents = obj_extents(mesh_path(urdf_path, first_filename))
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
            mesh.set("filename", str(mesh_path(urdf_path, filename)))
        old_scale = parse_vec(mesh.get("scale"), (1.0, 1.0, 1.0))
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


candidate = Path(raw_path).expanduser()
if candidate.is_file():
    print(write_motion_sized_urdf(candidate.resolve()))
    raise SystemExit(0)

for fallback in object_urdf_fallbacks(candidate):
    if fallback.is_file():
        print(write_motion_sized_urdf(fallback.resolve()))
        raise SystemExit(0)

print(write_motion_sized_urdf(candidate.resolve()))
PY
}

apply_training_sim_overrides() {
  if [[ "$USE_TRAINING_SIM_CONFIG" != "1" ]]; then
    return
  fi
  local model_path="$1"
  local override_lines
  override_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

exp_cfg = metadata.get("experiment_config")
if not isinstance(exp_cfg, dict):
    raise SystemExit(0)

sim_cfg = {}
sim_parent = exp_cfg.get("simulator")
if isinstance(sim_parent, dict):
    sim_cfg = sim_parent.get("config") if isinstance(sim_parent.get("config"), dict) else {}
sim_cfg = sim_cfg if isinstance(sim_cfg, dict) else {}
sim = sim_cfg.get("sim") if isinstance(sim_cfg.get("sim"), dict) else {}
terrain_term = {}
terrain_cfg = exp_cfg.get("terrain")
if isinstance(terrain_cfg, dict):
    terrain_term = terrain_cfg.get("terrain_term") if isinstance(terrain_cfg.get("terrain_term"), dict) else {}

def emit(key, value):
    if value is None:
        return
    if isinstance(value, bool):
        text = "True" if value else "False"
    elif isinstance(value, (int, float, str)):
        text = str(value)
    else:
        return
    print(f"{key}={text}")

emit("SIM_FPS", sim.get("fps"))
emit("SIM_CONTROL_DECIMATION", sim.get("control_decimation"))
emit("SIM_SUBSTEPS", sim.get("substeps"))
backend = sim_cfg.get("mujoco_backend")
if isinstance(backend, str):
    emit("MUJOCO_BACKEND", backend.upper())
emit("TERRAIN_STATIC_FRICTION", terrain_term.get("static_friction"))
emit("TERRAIN_DYNAMIC_FRICTION", terrain_term.get("dynamic_friction"))
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      SIM_FPS)
        if [[ "$SIM_FPS_EXPLICIT" != "1" ]]; then
          SIM_FPS="$value"
        fi
        ;;
      SIM_CONTROL_DECIMATION)
        if [[ "$SIM_CONTROL_DECIMATION_EXPLICIT" != "1" ]]; then
          SIM_CONTROL_DECIMATION="$value"
        fi
        ;;
      SIM_SUBSTEPS)
        if [[ "$SIM_SUBSTEPS_EXPLICIT" != "1" ]]; then
          SIM_SUBSTEPS="$value"
        fi
        ;;
      MUJOCO_BACKEND) MUJOCO_BACKEND="$value" ;;
      TERRAIN_STATIC_FRICTION)
        if [[ "$TERRAIN_STATIC_FRICTION_EXPLICIT" != "1" ]]; then
          TERRAIN_STATIC_FRICTION="$value"
        fi
        ;;
      TERRAIN_DYNAMIC_FRICTION)
        if [[ "$TERRAIN_DYNAMIC_FRICTION_EXPLICIT" != "1" ]]; then
          TERRAIN_DYNAMIC_FRICTION="$value"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

apply_training_robot_init_overrides() {
  local model_path="$1"
  local override_lines
  override_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

init_state = metadata.get("experiment_config", {}).get("robot", {}).get("init_state", {})
if not isinstance(init_state, dict):
    raise SystemExit(0)

pos = init_state.get("pos")
rot = init_state.get("rot")
if isinstance(pos, list) and len(pos) == 3:
    print("ROBOT_INIT_STATE_POS=" + json.dumps(pos, separators=(",", ":")))
if isinstance(rot, list) and len(rot) == 4:
    print("ROBOT_INIT_STATE_ROT=" + json.dumps(rot, separators=(",", ":")))
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      ROBOT_INIT_STATE_POS)
        if [[ -z "$ROBOT_INIT_STATE_POS" ]]; then
          ROBOT_INIT_STATE_POS="$value"
        fi
        ;;
      ROBOT_INIT_STATE_ROT)
        if [[ -z "$ROBOT_INIT_STATE_ROT" ]]; then
          ROBOT_INIT_STATE_ROT="$value"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

apply_training_robot_asset_overrides() {
  local model_path="$1"
  local override_lines
  override_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

asset_cfg = metadata.get("experiment_config", {}).get("robot", {}).get("asset", {})
if not isinstance(asset_cfg, dict):
    raise SystemExit(0)

value = asset_cfg.get("enable_self_collisions")
if isinstance(value, bool):
    print("ROBOT_ENABLE_SELF_COLLISIONS=" + ("True" if value else "False"))

urdf_file = asset_cfg.get("urdf_file")
if isinstance(urdf_file, str) and urdf_file.strip():
    print("HOLOSOMA_W_OBJECT_URDF=" + urdf_file.strip())
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      ROBOT_ENABLE_SELF_COLLISIONS)
        if [[ -z "$ROBOT_ENABLE_SELF_COLLISIONS" ]]; then
          ROBOT_ENABLE_SELF_COLLISIONS="$value"
        fi
        ;;
      HOLOSOMA_W_OBJECT_URDF)
        if [[ -z "${HOLOSOMA_W_OBJECT_URDF:-}" ]]; then
          export HOLOSOMA_W_OBJECT_URDF="$value"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

apply_training_object_overrides() {
  local model_path="$1"
  local override_lines
  override_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

object_cfg = metadata.get("experiment_config", {}).get("robot", {}).get("object", {})
if not isinstance(object_cfg, dict):
    raise SystemExit(0)

mapping = {
    "mujoco_add_default_actuators": "SIM_ADD_DEFAULT_OBJECT_ACTUATORS",
    "mujoco_copy_joint_defaults_from_robot_xml": "SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML",
    "mujoco_copy_tendons_from_robot_xml": "SIM_COPY_TENDONS_FROM_ROBOT_XML",
    "mujoco_copy_collision_geoms_from_robot_xml": "SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML",
    "mujoco_copy_contact_pairs_from_robot_xml": "SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML",
    "mujoco_use_training_urdf_scene": "SIM_USE_TRAINING_URDF_OBJECT_SCENE",
    "mujoco_limit_object_contacts_to_carry_bodies": "MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES",
}
for cfg_key, env_key in mapping.items():
    value = object_cfg.get(cfg_key)
    if isinstance(value, bool):
        print(f"{env_key}=" + ("1" if value else "0"))

markers = object_cfg.get("mujoco_object_contact_body_name_markers")
if isinstance(markers, list):
    print("MUJOCO_OBJECT_CONTACT_BODY_MARKERS=" + json.dumps(markers, separators=(",", ":")))
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      SIM_ADD_DEFAULT_OBJECT_ACTUATORS)
        if [[ -z "$SIM_ADD_DEFAULT_OBJECT_ACTUATORS" ]]; then
          SIM_ADD_DEFAULT_OBJECT_ACTUATORS="$value"
        fi
        ;;
      SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML)
        if [[ -z "$SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML" ]]; then
          SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="$value"
        fi
        ;;
      SIM_COPY_TENDONS_FROM_ROBOT_XML)
        if [[ -z "$SIM_COPY_TENDONS_FROM_ROBOT_XML" ]]; then
          SIM_COPY_TENDONS_FROM_ROBOT_XML="$value"
        fi
        ;;
      SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML)
        if [[ -z "$SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML" ]]; then
          SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="$value"
        fi
        ;;
      SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML)
        if [[ -z "$SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML" ]]; then
          SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="$value"
        fi
        ;;
      SIM_USE_TRAINING_URDF_OBJECT_SCENE)
        if [[ -z "$SIM_USE_TRAINING_URDF_OBJECT_SCENE" ]]; then
          SIM_USE_TRAINING_URDF_OBJECT_SCENE="$value"
        fi
        ;;
      MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES)
        if [[ -z "$MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES" ]]; then
          MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="$value"
        fi
        ;;
      MUJOCO_OBJECT_CONTACT_BODY_MARKERS)
        if [[ "$USE_TRAINING_OBJECT_CONTACT_MARKERS" == "1" && -z "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS" ]]; then
          MUJOCO_OBJECT_CONTACT_BODY_MARKERS="$value"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

apply_gt_mujoco_physics_overrides() {
  if ! is_truthy_env "$GT_MUJOCO_PHYSICS"; then
    return
  fi

  GT_MUJOCO_PHYSICS=1
  export GT_MUJOCO_PHYSICS
  export HOLOSOMA_GT_MUJOCO_PHYSICS=1
  export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS=0

  SIM_USE_TRAINING_URDF_OBJECT_SCENE=1
  SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML=0
  SIM_COPY_TENDONS_FROM_ROBOT_XML=0
  SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML=0
  SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML=0

  MUJOCO_OBJECT_MASS_SCALE=""
  MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-1.4}"
  MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.6,0.02,0.005]}"
  MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.6,0.02,0.005]}"
  MUJOCO_OBJECT_LATERAL_FRICTION=""
  MUJOCO_OBJECT_ROLLING_FRICTION=""
  MUJOCO_OBJECT_CONTACT_STIFFNESS=""
  MUJOCO_OBJECT_CONTACT_DAMPING=""
}

apply_training_perception_overrides() {
  local model_path="$1"
  local override_lines
  override_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

perception_cfg = metadata.get("experiment_config", {}).get("perception", {})
if not isinstance(perception_cfg, dict):
    raise SystemExit(0)

field_map = {
    "update_hz": "PERCEPTION_UPDATE_HZ",
    "camera_fps": "PERCEPTION_CAMERA_FPS",
    "camera_pitch_deg": "PERCEPTION_CAMERA_PITCH_DEG",
    "camera_near": "PERCEPTION_CAMERA_NEAR",
    "camera_far": "PERCEPTION_CAMERA_FAR",
    "max_distance": "PERCEPTION_MAX_DISTANCE",
    "camera_warp_min_valid_depth": "PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH",
    "camera_warp_buffer_len": "PERCEPTION_CAMERA_WARP_BUFFER_LEN",
    "camera_warp_latency_frame": "PERCEPTION_CAMERA_WARP_LATENCY_FRAME",
}
for src_key, env_key in field_map.items():
    value = perception_cfg.get(src_key)
    if value is None:
        continue
    if isinstance(value, bool):
        print(f"{env_key}={value}")
    elif isinstance(value, int):
        print(f"{env_key}={value}")
    elif isinstance(value, float):
        print(f"{env_key}={value:g}")
    else:
        print(f"{env_key}={value}")
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      PERCEPTION_UPDATE_HZ)
        if [[ -z "$PERCEPTION_UPDATE_HZ" ]]; then
          PERCEPTION_UPDATE_HZ="$value"
        fi
        ;;
      PERCEPTION_CAMERA_FPS)
        if [[ -z "$PERCEPTION_CAMERA_FPS" ]]; then
          PERCEPTION_CAMERA_FPS="$value"
        fi
        ;;
      PERCEPTION_CAMERA_PITCH_DEG)
        if [[ -z "$PERCEPTION_CAMERA_PITCH_DEG" ]]; then
          PERCEPTION_CAMERA_PITCH_DEG="$value"
        fi
        ;;
      PERCEPTION_CAMERA_NEAR)
        if [[ -z "$PERCEPTION_CAMERA_NEAR" ]]; then
          PERCEPTION_CAMERA_NEAR="$value"
        fi
        ;;
      PERCEPTION_CAMERA_FAR)
        if [[ -z "$PERCEPTION_CAMERA_FAR" ]]; then
          PERCEPTION_CAMERA_FAR="$value"
        fi
        ;;
      PERCEPTION_MAX_DISTANCE)
        if [[ -z "$PERCEPTION_MAX_DISTANCE" ]]; then
          PERCEPTION_MAX_DISTANCE="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH)
        if [[ -z "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH" ]]; then
          PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_BUFFER_LEN)
        if [[ -z "$PERCEPTION_CAMERA_WARP_BUFFER_LEN" ]]; then
          PERCEPTION_CAMERA_WARP_BUFFER_LEN="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_LATENCY_FRAME)
        if [[ -z "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME" ]]; then
          PERCEPTION_CAMERA_WARP_LATENCY_FRAME="$value"
        fi
        ;;
    esac
  done <<< "$override_lines"
}

apply_training_motion_launch_defaults() {
  local model_path="$1"
  local explicit_motion_init_mode
  explicit_motion_init_mode="$(echo "$SIM_MOTION_INIT_MODE" | tr '[:upper:]-' '[:lower:]_')"
  local default_lines
  default_lines="$(
    "$INFER_PY" - <<'PY' "$model_path"
import json
import sys
from pathlib import Path

import onnx


def resolve_model_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.suffix == ".pt":
        candidate = path.with_suffix(".onnx")
        if not candidate.is_file():
            raise FileNotFoundError(f"Expected sibling ONNX next to checkpoint: {candidate}")
        return candidate
    return path


model = onnx.load(resolve_model_path(sys.argv[1]))
metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

motion_cfg = (
    metadata.get("experiment_config", {})
    .get("command", {})
    .get("setup_terms", {})
    .get("motion_command", {})
    .get("params", {})
    .get("motion_config", {})
)
motion_cfg = motion_cfg if isinstance(motion_cfg, dict) else {}

needs_default_pose_transition = bool(
    (motion_cfg.get("enable_default_pose_prepend") and float(motion_cfg.get("default_pose_prepend_duration_s", 0.0) or 0.0) > 0.0)
    or (motion_cfg.get("enable_default_pose_append") and float(motion_cfg.get("default_pose_append_duration_s", 0.0) or 0.0) > 0.0)
)

if needs_default_pose_transition:
    print("APPLY_TRAINING_MOTION_TRANSITIONS=1")
    print("SIM_MOTION_INIT_MODE=training_default_pose")
    print("USE_ROOT_REFERENCE_AT_CLIP_START=1")
    print("AUTO_START_STIFF_HOLD_SEC=1.0")
    print("AUTO_START_STIFF_MAX_WAIT_SEC=1.0")
PY
  )"

  if [[ -z "$default_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      APPLY_TRAINING_MOTION_TRANSITIONS)
        if [[ "$SIM_MOTION_INIT_MODE_EXPLICIT" != "1" || "$explicit_motion_init_mode" == "training_default_pose" ]]; then
          APPLY_TRAINING_MOTION_TRANSITIONS="$value"
        fi
        ;;
      SIM_MOTION_INIT_MODE)
        if [[ "$SIM_MOTION_INIT_MODE_EXPLICIT" != "1" ]]; then
          SIM_MOTION_INIT_MODE="$value"
        fi
        ;;
      USE_ROOT_REFERENCE_AT_CLIP_START)
        if [[ "$USE_ROOT_REFERENCE_AT_CLIP_START_RAW" == "__unset__" && ( "$SIM_MOTION_INIT_MODE_EXPLICIT" != "1" || "$explicit_motion_init_mode" == "training_default_pose" ) ]]; then
          USE_ROOT_REFERENCE_AT_CLIP_START="$value"
        fi
        ;;
      AUTO_START_STIFF_HOLD_SEC)
        if [[ "$AUTO_START_STIFF_HOLD_SEC_RAW" == "__unset__" && ( "$SIM_MOTION_INIT_MODE_EXPLICIT" != "1" || "$explicit_motion_init_mode" == "training_default_pose" ) ]]; then
          AUTO_START_STIFF_HOLD_SEC="$value"
        fi
        ;;
      AUTO_START_STIFF_MAX_WAIT_SEC)
        if [[ "$AUTO_START_STIFF_MAX_WAIT_SEC_RAW" == "__unset__" && ( "$SIM_MOTION_INIT_MODE_EXPLICIT" != "1" || "$explicit_motion_init_mode" == "training_default_pose" ) ]]; then
          AUTO_START_STIFF_MAX_WAIT_SEC="$value"
        fi
        ;;
    esac
  done <<< "$default_lines"
}

infer_inference_config() {
  "$INFER_PY" - <<'PY' "$1"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
input_dims = {}
for value in model.graph.input:
    dims = [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]
    input_dims[value.name] = dims

obs_dim = None
obs_shape = input_dims.get("obs")
if obs_shape is not None and len(obs_shape) >= 2 and isinstance(obs_shape[1], int):
    obs_dim = obs_shape[1]

metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

groups = (
    metadata.get("experiment_config", {})
    .get("observation", {})
    .get("groups", {})
)
groups = groups if isinstance(groups, dict) else {}

actor_input_dim = (
    metadata.get("experiment_config", {})
    .get("algo", {})
    .get("config", {})
    .get("module_dict", {})
    .get("actor", {})
    .get("input_dim")
)
actor_input_dim = actor_input_dim if isinstance(actor_input_dim, list) else []

if "perception_obs" in input_dims:
    if obs_dim == 308 and actor_input_dim == ["actor_obs_root", "actor_obs_proprio_no_linvel"]:
        print("g1-29dof-wbt-object-distill")
        raise SystemExit(0)
    raise SystemExit(
        "Unsupported depth ONNX inputs: "
        f"obs_dim={obs_dim!r}, actor_input_dim={actor_input_dim!r}, inputs={sorted(input_dims)}"
    )

if any(name in groups for name in ("actor_obs_root", "actor_obs_torso", "actor_obs_proprio", "actor_obs_box")):
    print("g1-29dof-wbt-object-distill")
    raise SystemExit(0)

actor_obs = groups.get("actor_obs", {})
terms_cfg = actor_obs.get("terms", {}) if isinstance(actor_obs, dict) else {}
terms = list(terms_cfg.keys()) if isinstance(terms_cfg, dict) else []
terms_set = set(terms)

legacy_w_object_terms = {
    "motion_command",
    "motion_ref_ori_b",
    "base_ang_vel",
    "dof_pos",
    "dof_vel",
    "actions",
    "obj_target_pose_size_b",
    "obj_pos_b",
    "obj_ori_b",
}

if obs_dim == 123:
    print("g1-29dof-wbt-object-distill")
elif obs_dim == 875:
    print("g1-29dof-wbt-w-object")
elif obs_dim == 175:
    print("g1-29dof-wbt-w-object")
elif obs_dim == 181:
    print("g1-29dof-wbt-object-generalist")
elif {"obj_lin_vel_b", "obj_ang_vel_b"} & terms_set:
    print("g1-29dof-wbt-object-generalist")
elif legacy_w_object_terms.issubset(terms_set):
    print("g1-29dof-wbt-w-object")
elif terms_set:
    raise SystemExit(f"Unsupported actor_obs terms for non-depth split rollout: {terms}")
else:
    raise SystemExit(f"Unable to infer split rollout config from ONNX obs dim {obs_dim!r}")
PY
}

onnx_has_input() {
  "$INFER_PY" - <<'PY' "$1" "$2"
import sys

import onnx

model = onnx.load(sys.argv[1])
name = sys.argv[2]
print("1" if any(value.name == name for value in model.graph.input) else "0")
PY
}

apply_training_motion_launch_defaults "$MODEL_INPUT"
apply_motion_clip_object_defaults
OBJECT_URDF="$(resolve_motion_sized_object_urdf "$OBJECT_URDF")"

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_INPUT" \
  --motion-file "$MOTION_FILE" \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--apply-training-motion-transitions" ) \
  $( [[ -n "$HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE" ]] && printf '%s %s' "--action-scale-override" "$HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE" ) \
  --output-path "$PATCHED_ONNX"

apply_training_sim_overrides "$PATCHED_ONNX"
apply_training_robot_init_overrides "$PATCHED_ONNX"
apply_training_robot_asset_overrides "$PATCHED_ONNX"
apply_training_object_overrides "$PATCHED_ONNX"
apply_training_perception_overrides "$PATCHED_ONNX"
apply_gt_mujoco_physics_overrides

SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-0}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-0}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-0}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-0}"
SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-1}"
if [[ "$SIM_USE_TRAINING_URDF_OBJECT_SCENE" == "1" && "$SIM_ADD_DEFAULT_OBJECT_ACTUATORS_RAW" == "__unset__" ]]; then
  # Generated training-URDF object scenes do not contain MuJoCo actuators; the split
  # bridge still needs default torque actuators to apply lowcmd torques.
  SIM_ADD_DEFAULT_OBJECT_ACTUATORS="1"
fi
MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-0}"
MUJOCO_OBJECT_CONTACT_BODY_MARKERS="${MUJOCO_OBJECT_CONTACT_BODY_MARKERS:-}"

if [[ -z "$INFERENCE_CONFIG" ]]; then
  INFERENCE_CONFIG="$(infer_inference_config "$PATCHED_ONNX")"
fi

if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ -n "$POLICY_MOTION_INDEX_OFFSET" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET="$POLICY_MOTION_INDEX_OFFSET"
  elif [[ "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-distill" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  fi
fi

MODEL_EXPECTS_PERCEPTION_OBS="$(onnx_has_input "$PATCHED_ONNX" "perception_obs")"
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "auto" ]]; then
  ENABLE_SPLIT_PERCEPTION_OBS="$MODEL_EXPECTS_PERCEPTION_OBS"
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" && "$ENABLE_SPLIT_PERCEPTION_OBS" != "1" ]]; then
  echo "Model expects perception_obs but ENABLE_SPLIT_PERCEPTION_OBS=${ENABLE_SPLIT_PERCEPTION_OBS}" >&2
  exit 1
fi
export HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE="${HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE:-1}"
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" && -z "$SIM_DEVICE" ]]; then
  HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH="${HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH:-1}"
  if is_truthy_env "$HOLOSOMA_AUTO_CUDA_FOR_TRAINING_DEPTH" && [[ "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }; then
      SIM_DEVICE="${HOLOSOMA_TRAINING_DEPTH_DEVICE:-cuda:0}"
    fi
  fi
fi
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_CAMERA_SOURCE" == "rendered" && -z "${MUJOCO_GL:-}" ]]; then
  case "$(printf '%s' "$TRAINING_HEADLESS" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      export MUJOCO_GL=glfw
      ;;
    *)
      export MUJOCO_GL=egl
      ;;
  esac
fi
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_CAMERA_SOURCE" == "rendered" ]]; then
  export HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}"
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_ROBOT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_ROBOT_VISUAL_MESHES:-0}"
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_OBJECT_VISUAL_MESHES:-1}"
  case "$(printf '%s' "$PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN" | tr '[:upper:]' '[:lower:]')" in
    training|distill|shoo7sr1)
      [[ "$PERCEPTION_CAMERA_WIDTH_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WIDTH="106"
      [[ "$PERCEPTION_CAMERA_HEIGHT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_HEIGHT="60"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_TOP_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_TOP="2"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_BOTTOM="0"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_LEFT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_LEFT="4"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_RIGHT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_RIGHT="4"
      ;;
    1|true|yes|on|myholosoma)
      [[ "$PERCEPTION_CAMERA_WIDTH_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WIDTH="848"
      [[ "$PERCEPTION_CAMERA_HEIGHT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_HEIGHT="480"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_TOP_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_TOP="16"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_BOTTOM="0"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_LEFT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_LEFT="32"
      [[ "$PERCEPTION_CAMERA_WARP_CROP_RIGHT_EXPLICIT" == "0" ]] && PERCEPTION_CAMERA_WARP_CROP_RIGHT="32"
      ;;
  esac
fi

PERCEPTION_OBS_TRANSPORT_NORMALIZED="$(printf '%s' "$PERCEPTION_OBS_TRANSPORT" | tr '[:upper:]' '[:lower:]')"
PUBLISH_PERCEPTION_OBS_SHM=0
USE_POLICY_PERCEPTION_OBS_SHM=0
PERCEPTION_OBS_EXTERNAL_ENABLED=0
case "$PERCEPTION_OBS_TRANSPORT_NORMALIZED" in
  shm|shared_memory|shared-memory|myholosoma)
    PUBLISH_PERCEPTION_OBS_SHM=1
    USE_POLICY_PERCEPTION_OBS_SHM=1
    ;;
  zmq)
    ;;
  both)
    PUBLISH_PERCEPTION_OBS_SHM=1
    USE_POLICY_PERCEPTION_OBS_SHM=1
    ;;
  *)
    echo "[ERROR] PERCEPTION_OBS_TRANSPORT must be shm, zmq, or both. Got: ${PERCEPTION_OBS_TRANSPORT}" >&2
    exit 1
    ;;
esac
if is_truthy_env "$PERCEPTION_OBS_EXTERNAL"; then
  PERCEPTION_OBS_EXTERNAL_ENABLED=1
  PUBLISH_PERCEPTION_OBS_SHM=0
fi

if [[ "$INFERENCE_CONFIG" == "g1-29dof-wbt-w-object" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-generalist" ]]; then
  if [[ -z "$USE_SIM_TIME" ]]; then
    USE_SIM_TIME="1"
  fi
  if [[ -z "$PREFER_SIM_REF_FROM_SIM_STATE" ]]; then
    PREFER_SIM_REF_FROM_SIM_STATE="1"
  fi
  if [[ -z "$USE_ROOT_REFERENCE_AT_CLIP_START" ]]; then
    USE_ROOT_REFERENCE_AT_CLIP_START="1"
  fi
  if [[ -z "$AUTO_START_STIFF_HOLD_SEC" ]]; then
    AUTO_START_STIFF_HOLD_SEC="1.0"
  fi
  if [[ -z "$AUTO_START_STIFF_MAX_WAIT_SEC" ]]; then
    AUTO_START_STIFF_MAX_WAIT_SEC="1.0"
  fi
  if [[ -z "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="0"
  fi
  if [[ -z "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="0"
  fi
  if [[ -z "$SIM_FREEZE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_FREEZE_UNTIL_FIRST_COMMAND="1"
  fi
else
  if [[ -z "$USE_SIM_TIME" ]]; then
    USE_SIM_TIME="1"
  fi
  if [[ -z "$PREFER_SIM_REF_FROM_SIM_STATE" ]]; then
    PREFER_SIM_REF_FROM_SIM_STATE="1"
  fi
  if [[ -z "$USE_ROOT_REFERENCE_AT_CLIP_START" ]]; then
    USE_ROOT_REFERENCE_AT_CLIP_START="1"
  fi
  if [[ -z "$AUTO_START_STIFF_HOLD_SEC" ]]; then
    AUTO_START_STIFF_HOLD_SEC="0.0"
  fi
  if [[ -z "$AUTO_START_STIFF_MAX_WAIT_SEC" ]]; then
    AUTO_START_STIFF_MAX_WAIT_SEC="0.0"
  fi
  if [[ -z "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="0"
  fi
  if [[ -z "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="0"
  fi
  if [[ -z "$SIM_FREEZE_UNTIL_FIRST_COMMAND" ]]; then
    SIM_FREEZE_UNTIL_FIRST_COMMAND="1"
  fi
fi

if [[ -z "$POLICY_ACTION_SCALE" ]]; then
  POLICY_ACTION_SCALE="$(
    "$INFER_PY" - <<'PY' "$PATCHED_ONNX"
import json
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {prop.key: json.loads(prop.value) for prop in model.metadata_props}
scale = (
    metadata.get("experiment_config", {})
    .get("robot", {})
    .get("control", {})
    .get("action_scale")
)
print(scale if scale is not None else 1.0)
PY
  )"
fi
if [[ -z "$POLICY_AUTO_START_MOTION_CLIP" ]]; then
  if [[ -n "${HOLOSOMA_POLICY_CONTROL_PORT:-}" ]]; then
    POLICY_AUTO_START_MOTION_CLIP="0"
  else
    POLICY_AUTO_START_MOTION_CLIP="1"
  fi
fi

SIM_LOG="$RUN_DIR/mujoco.log"
POLICY_LOG="$RUN_DIR/policy.log"

: >"$SIM_LOG"
: >"$POLICY_LOG"

if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]]; then
  echo "[INFO] motion_file=${MOTION_FILE}"
  echo "[INFO] object_urdf=${OBJECT_URDF}"
  echo "[INFO] robot_urdf=${HOLOSOMA_W_OBJECT_URDF:-g1/g1_29dof.urdf}"
  echo "[INFO] model=${MODEL_INPUT}"
  echo "[INFO] inference_config=${INFERENCE_CONFIG}"
  echo "[INFO] sim_device=${SIM_DEVICE:-<default>}"
  echo "[INFO] mujoco_object_scene training_urdf=${SIM_USE_TRAINING_URDF_OBJECT_SCENE} default_actuators=${SIM_ADD_DEFAULT_OBJECT_ACTUATORS} copy_joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} copy_tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} copy_collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} copy_contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML}"
  echo "[INFO] perception camera: source=${PERCEPTION_CAMERA_SOURCE} raw=${PERCEPTION_CAMERA_WIDTH:-<default>}x${PERCEPTION_CAMERA_HEIGHT:-<default>} crop_top=${PERCEPTION_CAMERA_WARP_CROP_TOP:-<default>} crop_bottom=${PERCEPTION_CAMERA_WARP_CROP_BOTTOM:-<default>} crop_left=${PERCEPTION_CAMERA_WARP_CROP_LEFT:-<default>} crop_right=${PERCEPTION_CAMERA_WARP_CROP_RIGHT:-<default>} update_hz=${PERCEPTION_UPDATE_HZ:-<default>} camera_fps=${PERCEPTION_CAMERA_FPS:-<default>} pitch_deg=${PERCEPTION_CAMERA_PITCH_DEG:-<default>} vfov_deg=${PERCEPTION_CAMERA_VFOV_DEG:-<default>} hfov_deg=${PERCEPTION_CAMERA_HFOV_DEG:-<default>} include_robot_mesh=${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-<default>} near=${PERCEPTION_CAMERA_NEAR:-<default>} far=${PERCEPTION_CAMERA_FAR:-<default>} max_distance=${PERCEPTION_MAX_DISTANCE:-<default>} warp_min_valid_depth=${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-<default>} warp_buffer_len=${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-<default>} warp_latency_frame=${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-<default>} warp_edge_noise=${PERCEPTION_CAMERA_WARP_EDGE_NOISE} warp_holes=${PERCEPTION_CAMERA_WARP_ENABLE_HOLES} sensor_noise=${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE} transport=${PERCEPTION_OBS_TRANSPORT}"
  if is_truthy_env "$PERCEPTION_OBS_EXTERNAL"; then
    echo "[INFO] perception_obs_external=1; MuJoCo will not publish perception_obs. Start an external publisher/relay on port=${PERCEPTION_OBS_PORT} or shm=${PERCEPTION_OBS_SHM_NAME}."
  fi
fi
if is_truthy_env "$GT_MUJOCO_PHYSICS"; then
  echo "[INFO] GT MuJoCo physics: object_mass=${MUJOCO_OBJECT_MASS_OVERRIDE} object_friction=${MUJOCO_OBJECT_GEOM_FRICTION} object_terrain_pair_friction=${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-<none>} copy_joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} copy_tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} copy_collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} copy_contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML} zero_passive_dynamics=${HOLOSOMA_GT_MUJOCO_ZERO_PASSIVE_DYNAMICS:-0} web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
fi
if is_truthy_env "${DRY_RUN:-0}"; then
  echo "[INFO] DRY_RUN=1; not launching MuJoCo or policy."
  exit 0
fi

terminate_pid() {
  local pid="$1"
  [[ -n "${pid:-}" ]] || return
  kill -0 "$pid" 2>/dev/null || return
  kill "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.2
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  terminate_pid "${POLICY_PID:-}"
  terminate_pid "${SIM_PID:-}"
}
trap cleanup EXIT

MUJOCO_LAUNCH_PREFIX=()
if [[ -n "${MUJOCO_CPUSET}" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    MUJOCO_LAUNCH_PREFIX=(taskset -c "${MUJOCO_CPUSET}")
  else
    echo "Warning: taskset not found; ignoring MUJOCO_CPUSET=${MUJOCO_CPUSET}" >&2
  fi
fi

wait_for_sim_ready() {
  local deadline=$((SECONDS + SIM_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
      echo "MuJoCo simulator exited during startup. See $SIM_LOG" >&2
      tail -n 40 "$SIM_LOG" >&2 || true
      return 1
    fi
    if [[ -f "$SIM_LOG" ]] && grep -qF "$SIM_READY_PATTERN" "$SIM_LOG"; then
      return 0
    fi
    sleep 0.5
  done

  echo "Timed out waiting for MuJoCo readiness pattern '$SIM_READY_PATTERN'. See $SIM_LOG" >&2
  tail -n 40 "$SIM_LOG" >&2 || true
  return 1
}

if [[ "$MJ_TRACK_MODE" != "policy" ]]; then
  "${MUJOCO_LAUNCH_PREFIX[@]}" "$MUJOCO_PY" -u "$ROOT_DIR/src/holosoma/holosoma/run_sim.py" \
    simulator:mujoco \
    robot:g1_29dof_w_object \
    terrain:terrain_locomotion_plane \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" ]] && printf '%s' "perception:${PERCEPTION_PRESET}" ) \
    --training.headless "$TRAINING_HEADLESS" \
    --simulator.config.debug-viz "$SIM_DEBUG_VIZ" \
    $( [[ "$MUJOCO_SHOW_OBJECT_COLLISION" == "1" ]] && printf '%s %s' "--simulator.config.mujoco-show-object-collision" "True" ) \
    $( [[ "$MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION" == "1" ]] && printf '%s %s' "--simulator.config.mujoco-hide-object-visuals-when-showing-collision" "True" ) \
    --simulator.config.sim.fps "$SIM_FPS" \
    --simulator.config.sim.control-decimation "$SIM_CONTROL_DECIMATION" \
    $( [[ -n "$SIM_SUBSTEPS" ]] && printf '%s %s' "--simulator.config.sim.substeps" "$SIM_SUBSTEPS" ) \
    $( [[ -n "$MUJOCO_BACKEND" ]] && printf '%s %s' "--simulator.config.mujoco-backend" "$MUJOCO_BACKEND" ) \
    $( [[ -n "$SIM_DEVICE" ]] && printf '%s %s' "--device" "$SIM_DEVICE" ) \
    --simulator.config.virtual-gantry.enabled "$SIM_VIRTUAL_GANTRY_ENABLED" \
    $( [[ -n "$ROBOT_INIT_STATE_POS" ]] && printf '%s %s' "--robot.init-state.pos" "$ROBOT_INIT_STATE_POS" ) \
    $( [[ -n "$ROBOT_INIT_STATE_ROT" ]] && printf '%s %s' "--robot.init-state.rot" "$ROBOT_INIT_STATE_ROT" ) \
    $( [[ -n "$ROBOT_ENABLE_SELF_COLLISIONS" ]] && printf '%s %s' "--robot.asset.enable-self-collisions" "$ROBOT_ENABLE_SELF_COLLISIONS" ) \
    --robot.object.enabled=True \
    --robot.object.object-urdf-path "$OBJECT_URDF" \
    $( [[ "$SIM_USE_TRAINING_URDF_OBJECT_SCENE" == "1" ]] && printf '%s %s' "--robot.object.mujoco-use-training-urdf-scene" "True" ) \
    $( [[ "$SIM_ADD_DEFAULT_OBJECT_ACTUATORS" == "1" ]] && printf '%s %s' "--robot.object.mujoco-add-default-actuators" "True" ) \
    $( [[ "$SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-joint-defaults-from-robot-xml" "True" ) \
    $( [[ "$SIM_COPY_TENDONS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-tendons-from-robot-xml" "True" ) \
    $( [[ "$SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-collision-geoms-from-robot-xml" "True" ) \
    $( [[ "$SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-contact-pairs-from-robot-xml" "True" ) \
    $( [[ -n "$MUJOCO_OBJECT_MASS_SCALE" ]] && printf '%s %s' "--robot.object.mujoco-object-mass-scale" "$MUJOCO_OBJECT_MASS_SCALE" ) \
    $( [[ -n "$MUJOCO_OBJECT_MASS_OVERRIDE" ]] && printf '%s %s' "--robot.object.mujoco-object-mass-override" "$MUJOCO_OBJECT_MASS_OVERRIDE" ) \
    $( [[ -n "$MUJOCO_OBJECT_GEOM_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-geom-friction" "$MUJOCO_OBJECT_GEOM_FRICTION" ) \
    $( [[ -n "$MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-terrain-pair-friction" "$MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION" ) \
    $( [[ -n "$MUJOCO_OBJECT_LATERAL_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-lateral-friction" "$MUJOCO_OBJECT_LATERAL_FRICTION" ) \
    $( [[ -n "$MUJOCO_OBJECT_ROLLING_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-rolling-friction" "$MUJOCO_OBJECT_ROLLING_FRICTION" ) \
    $( [[ -n "$MUJOCO_OBJECT_CONTACT_STIFFNESS" ]] && printf '%s %s' "--robot.object.mujoco-object-contact-stiffness" "$MUJOCO_OBJECT_CONTACT_STIFFNESS" ) \
    $( [[ -n "$MUJOCO_OBJECT_CONTACT_DAMPING" ]] && printf '%s %s' "--robot.object.mujoco-object-contact-damping" "$MUJOCO_OBJECT_CONTACT_DAMPING" ) \
    $( [[ "$MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES" == "1" ]] && printf '%s %s' "--robot.object.mujoco-limit-object-contacts-to-carry-bodies" "True" ) \
    $( [[ -n "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS" ]] && printf '%s %s' "--robot.object.mujoco-object-contact-body-name-markers" "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS" ) \
    $( [[ -n "$TERRAIN_STATIC_FRICTION" ]] && printf '%s %s' "--terrain.terrain-term.static-friction" "$TERRAIN_STATIC_FRICTION" ) \
    $( [[ -n "$TERRAIN_DYNAMIC_FRICTION" ]] && printf '%s %s' "--terrain.terrain-term.dynamic-friction" "$TERRAIN_DYNAMIC_FRICTION" ) \
    --simulator.config.bridge.interface "$INTERFACE_NAME" \
    --simulator.config.bridge.clock-port "$SIM_CLOCK_PORT" \
    --simulator.config.bridge.publish-sim-state=True \
    --simulator.config.bridge.listen-control=True \
    --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT" \
    --simulator.config.bridge.control-port "$SIM_CONTROL_PORT" \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" ]] && printf '%s %s' "--simulator.config.bridge.publish-perception-obs" "True" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" ]] && printf '%s %s' "--simulator.config.bridge.perception-obs-port" "$PERCEPTION_OBS_PORT" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" && "$PUBLISH_PERCEPTION_OBS_SHM" == "1" ]] && printf '%s %s' "--simulator.config.bridge.publish-perception-obs-shm" "True" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" && "$PUBLISH_PERCEPTION_OBS_SHM" == "1" ]] && printf '%s %s' "--simulator.config.bridge.perception-obs-shm-name" "$PERCEPTION_OBS_SHM_NAME" ) \
    $( [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]] && printf '%s %s' "--simulator.config.bridge.use-zmq-lowcmd" "True" ) \
    $( [[ "$SIM_IGNORE_DEFAULT_IDLE_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.ignore-default-idle-command" "True" ) \
    $( [[ "$SIM_LOG_FIRST_COMMAND_SUMMARY" == "1" ]] && printf '%s %s' "--simulator.config.bridge.log-first-command-summary" "True" ) \
    $( [[ "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.hold-default-pose-until-first-command" "True" ) \
    $( [[ "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.hold-initial-pose-until-first-command" "True" ) \
    $( [[ "$SIM_FREEZE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.freeze-until-first-command" "True" ) \
    --motion-init.enabled=True \
    --motion-init.motion-file "$MOTION_FILE" \
    --motion-init.mode "$SIM_MOTION_INIT_MODE" \
    --motion-init.object-name object \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_SOURCE" ]] && printf '%s %s' "--perception.camera-source" "$PERCEPTION_CAMERA_SOURCE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_OBJECT_GEOMETRY_MODE" ]] && printf '%s %s' "--perception.object-geometry-mode" "$PERCEPTION_OBJECT_GEOMETRY_MODE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WIDTH" ]] && printf '%s %s' "--perception.camera-width" "$PERCEPTION_CAMERA_WIDTH" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_HEIGHT" ]] && printf '%s %s' "--perception.camera-height" "$PERCEPTION_CAMERA_HEIGHT" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_CROP_TOP" ]] && printf '%s %s' "--perception.camera-warp-crop-top" "$PERCEPTION_CAMERA_WARP_CROP_TOP" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM" ]] && printf '%s %s' "--perception.camera-warp-crop-bottom" "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_CROP_LEFT" ]] && printf '%s %s' "--perception.camera-warp-crop-left" "$PERCEPTION_CAMERA_WARP_CROP_LEFT" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_CROP_RIGHT" ]] && printf '%s %s' "--perception.camera-warp-crop-right" "$PERCEPTION_CAMERA_WARP_CROP_RIGHT" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_PITCH_DEG" ]] && printf '%s %s' "--perception.camera-pitch-deg" "$PERCEPTION_CAMERA_PITCH_DEG" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_VFOV_DEG" ]] && printf '%s %s' "--perception.camera-vfov-deg" "$PERCEPTION_CAMERA_VFOV_DEG" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_HFOV_DEG" ]] && printf '%s %s' "--perception.camera-hfov-deg" "$PERCEPTION_CAMERA_HFOV_DEG" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH" ]] && printf '%s %s' "--perception.camera-include-robot-mesh" "$PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_NEAR" ]] && printf '%s %s' "--perception.camera-near" "$PERCEPTION_CAMERA_NEAR" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_FAR" ]] && printf '%s %s' "--perception.camera-far" "$PERCEPTION_CAMERA_FAR" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_MAX_DISTANCE" ]] && printf '%s %s' "--perception.max-distance" "$PERCEPTION_MAX_DISTANCE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH" ]] && printf '%s %s' "--perception.camera-warp-min-valid-depth" "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_UPDATE_HZ" ]] && printf '%s %s' "--perception.update-hz" "$PERCEPTION_UPDATE_HZ" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_FPS" ]] && printf '%s %s' "--perception.camera-fps" "$PERCEPTION_CAMERA_FPS" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_BUFFER_LEN" ]] && printf '%s %s' "--perception.camera-warp-buffer-len" "$PERCEPTION_CAMERA_WARP_BUFFER_LEN" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME" ]] && printf '%s %s' "--perception.camera-warp-latency-frame" "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" ]] && printf '%s %s' "--perception.camera-warp-edge-noise" "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_ENABLE_HOLES" ]] && printf '%s %s' "--perception.camera-warp-enable-holes" "$PERCEPTION_CAMERA_WARP_ENABLE_HOLES" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_APPLY_SENSOR_NOISE" ]] && printf '%s %s' "--perception.camera-apply-sensor-noise" "$PERCEPTION_CAMERA_APPLY_SENSOR_NOISE" ) \
    >"$SIM_LOG" 2>&1 &
  SIM_PID=$!

  if ! wait_for_sim_ready; then
    exit 1
  fi

  if [[ "$SIM_STARTUP_WAIT" != "0" ]]; then
    sleep "$SIM_STARTUP_WAIT"
  fi
fi

if [[ "$MJ_TRACK_MODE" != "policy" && ( "$SKIP_POLICY" == "1" || "$SKIP_POLICY" == "true" || "$SKIP_POLICY" == "True" ) ]]; then
  echo "Policy launch skipped (SKIP_POLICY=${SKIP_POLICY}); simulator is running without external lowcmd."
  if [[ "$RUN_SECONDS" == "0" ]]; then
    wait "$SIM_PID"
  else
    sleep "$RUN_SECONDS"
  fi
  exit 0
fi

POLICY_CMD=(
  "$INFER_PY" -u "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py"
  "inference:${INFERENCE_CONFIG}"
  --task.model-path "$PATCHED_ONNX"
  --task.motion-file "$MOTION_FILE"
  --task.interface "$INTERFACE_NAME"
  --task.use-sim-state
  --task.sim-clock-port "$SIM_CLOCK_PORT"
  --task.sim-state-port "$SIM_STATE_PORT"
  --task.sim-control-port "$SIM_CONTROL_PORT"
  --task.no-auto-start-motion
  --task.auto-start-stiff-hold-sec "$AUTO_START_STIFF_HOLD_SEC"
  --task.auto-start-stiff-max-wait-sec "$AUTO_START_STIFF_MAX_WAIT_SEC"
  --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL"
  --task.policy-action-scale "$POLICY_ACTION_SCALE"
  --task.rl-rate "$POLICY_RL_RATE"
  --task.sim-object-name object
)
if is_truthy_env "$POLICY_AUTO_START_MOTION_CLIP"; then
  POLICY_CMD+=(--task.auto-start-motion-clip)
fi
if [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]]; then
  POLICY_CMD+=(--task.use-zmq-lowcmd)
fi
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]]; then
  POLICY_CMD+=(--task.use-split-perception-obs --task.perception-obs-port "$PERCEPTION_OBS_PORT")
  if [[ "$USE_POLICY_PERCEPTION_OBS_SHM" == "1" ]]; then
    POLICY_CMD+=(--task.use-split-perception-obs-shm --task.perception-obs-shm-name "$PERCEPTION_OBS_SHM_NAME")
  fi
fi
if [[ "$ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND" == "1" || "$ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND" == "true" || "$ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND" == "True" ]]; then
  POLICY_CMD+=(--task.use-external-sparse-root-command --task.sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT")
fi
if [[ "$USE_SIM_TIME" == "1" ]]; then
  POLICY_CMD+=(--task.use-sim-time)
fi
if [[ "$USE_ROOT_REFERENCE_AT_CLIP_START" == "1" ]]; then
  POLICY_CMD+=(--task.use-root-reference-at-clip-start)
fi
if [[ "$PREFER_SIM_REF_FROM_SIM_STATE" == "1" ]]; then
  POLICY_CMD+=(--task.prefer-sim-ref-from-sim-state)
fi
if [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]]; then
  POLICY_CMD+=(--task.apply-training-motion-transitions)
fi
if [[ "$POLICY_DEFER_UNTIL_VALID_STATE" == "1" ]]; then
  POLICY_CMD+=(--task.defer-policy-start-until-valid-state)
fi

set +e
if [[ "$POLICY_STDIO" == "inherit" ]]; then
  if [[ "$RUN_SECONDS" == "0" ]]; then
    "${POLICY_CMD[@]}"
    STATUS=$?
  else
    timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${POLICY_CMD[@]}"
    STATUS=$?
  fi
else
  if [[ "$RUN_SECONDS" == "0" ]]; then
    if is_truthy_env "${HOLOSOMA_POLICY_TTY_INPUT:-0}" && [[ -r /dev/tty ]]; then
      "${POLICY_CMD[@]}" </dev/tty >"$POLICY_LOG" 2>&1 &
    else
      "${POLICY_CMD[@]}" >"$POLICY_LOG" 2>&1 &
    fi
  else
    if is_truthy_env "${HOLOSOMA_POLICY_TTY_INPUT:-0}" && [[ -r /dev/tty ]]; then
      timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${POLICY_CMD[@]}" </dev/tty >"$POLICY_LOG" 2>&1 &
    else
      timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${POLICY_CMD[@]}" >"$POLICY_LOG" 2>&1 &
    fi
  fi
  POLICY_PID=$!
  wait "$POLICY_PID"
  STATUS=$?
fi
set -e

if [[ "$STATUS" -ne 0 && "$STATUS" -ne 124 && "$STATUS" -ne 130 ]]; then
  echo "Policy run failed. See $POLICY_LOG" >&2
  exit "$STATUS"
fi

echo "Patched ONNX: $PATCHED_ONNX"
echo "MuJoCo log:   $SIM_LOG"
echo "Policy log:   $POLICY_LOG"
