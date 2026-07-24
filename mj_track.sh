#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/scripts/mujoco_perception_env.sh"
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
MUJOCO_PYTHONPATH="${MUJOCO_PYTHONPATH:-}"
INFER_PY="${INFER_PY:-}"
MUJOCO_CPUSET="${MUJOCO_CPUSET:-0}"
SIM_FPS_EXPLICIT=0
[[ -n "${SIM_FPS+x}" ]] && SIM_FPS_EXPLICIT=1
SIM_FPS="${SIM_FPS:-500}"
SIM_CONTROL_DECIMATION_EXPLICIT=0
[[ -n "${SIM_CONTROL_DECIMATION+x}" ]] && SIM_CONTROL_DECIMATION_EXPLICIT=1
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-10}"
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
MUJOCO_RENDER_848_PERCEPTION_PRESET="${MUJOCO_RENDER_848_PERCEPTION_PRESET:-camera_depth_d435i_mujoco_render_848x480}"
PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-}"
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
holosoma_capture_explicit_env \
  PERCEPTION_CAMERA_WARP_NORMALIZE \
  PERCEPTION_CAMERA_WARP_EDGE_NOISE \
  PERCEPTION_CAMERA_WARP_ENABLE_HOLES \
  PERCEPTION_CAMERA_APPLY_SENSOR_NOISE \
  HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE
PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-}"
PERCEPTION_CAMERA_WARP_NORMALIZE="${PERCEPTION_CAMERA_WARP_NORMALIZE:-}"
PERCEPTION_CAMERA_WARP_EDGE_BORDER="${PERCEPTION_CAMERA_WARP_EDGE_BORDER:-}"
PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB="${PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB:-}"
PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB="${PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB:-}"
PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY="${PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY:-}"
PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY="${PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY:-}"
PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH="${PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH:-}"
PERCEPTION_CAMERA_WARP_ENABLE_HOLES="${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-}"
PERCEPTION_CAMERA_WARP_HOLE_PROB="${PERCEPTION_CAMERA_WARP_HOLE_PROB:-}"
PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE="${PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE:-}"
PERCEPTION_CAMERA_APPLY_SENSOR_NOISE="${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-}"
PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH="${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-}"
PERCEPTION_RANDOMIZATION_ENABLED="${PERCEPTION_RANDOMIZATION_ENABLED:-}"
PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE="${PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE:-}"
PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG="${PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG:-}"
PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE="${PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE:-}"
PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE="${PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE:-}"
PERCEPTION_RANDOMIZATION_CONTRACT_STATUS=""
PERCEPTION_CONTRACT_ENVELOPE_B64=""
PERCEPTION_PRODUCER_TICK_DT="${PERCEPTION_PRODUCER_TICK_DT:-}"
PERCEPTION_PRODUCER_SEED="${PERCEPTION_PRODUCER_SEED:-}"
PERCEPTION_ALLOW_MUJOCO_NOISE="${PERCEPTION_ALLOW_MUJOCO_NOISE:-}"
PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="${PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN:-}"
if [[ -z "$PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN" ]]; then
  if [[ "$PERCEPTION_PRESET" == "$MUJOCO_RENDER_848_PERCEPTION_PRESET" ]]; then
    PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="mujoco848"
  else
    PERCEPTION_RENDER_RAW_RESOLUTION_ALIGN="training"
  fi
fi
PERCEPTION_OBS_TRANSPORT="${PERCEPTION_OBS_TRANSPORT:-shm}"
PERCEPTION_OBS_SHM_NAME="${PERCEPTION_OBS_SHM_NAME:-depth_img_shm_${SIM_STATE_PORT}}"
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
SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-}"
SIM_USE_TRAINING_URDF_OBJECT_SCENE="${SIM_USE_TRAINING_URDF_OBJECT_SCENE:-}"
MUJOCO_OBJECT_MASS_SCALE="${MUJOCO_OBJECT_MASS_SCALE:-}"
MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-}"
MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-}"
MUJOCO_OBJECT_LATERAL_FRICTION="${MUJOCO_OBJECT_LATERAL_FRICTION:-}"
MUJOCO_OBJECT_ROLLING_FRICTION="${MUJOCO_OBJECT_ROLLING_FRICTION:-}"
MUJOCO_OBJECT_CONTACT_STIFFNESS="${MUJOCO_OBJECT_CONTACT_STIFFNESS:-}"
MUJOCO_OBJECT_CONTACT_DAMPING="${MUJOCO_OBJECT_CONTACT_DAMPING:-}"
MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES="${MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES:-}"
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

MOTION_STEM="$(basename "${MOTION_FILE%.*}")"
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
  PYTHONPATH="${MUJOCO_PYTHONPATH:-${PYTHONPATH:-}}" "$python_bin" - "$@" <<'PY' >/dev/null 2>&1
import importlib
import os
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

resolve_policy_model_path() {
  "$INFER_PY" - <<'PY' "$1"
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
if path.suffix == ".pt":
    path = path.with_suffix(".onnx")
if not path.is_file():
    raise SystemExit(f"Policy model not found: {path}")
print(path.resolve())
PY
}

if [[ -n "$MUJOCO_PY" ]]; then
  MUJOCO_PY="$(resolve_python_with_modules "mujoco holosoma torch tyro typeguard pydantic" "$MUJOCO_PY")"
else
  MUJOCO_PY="$(resolve_python_with_modules "mujoco holosoma torch tyro typeguard pydantic" \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
    /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)")"
fi
INFER_PY="$(resolve_python "$INFER_PY" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
POLICY_MODEL="$(resolve_policy_model_path "$MODEL_INPUT")"

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
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from holosoma.utils.path import resolve_data_file_path
except Exception:
    resolve_data_file_path = None

raw_path = sys.argv[1]
motion_path = Path(sys.argv[2]).expanduser().resolve()
repo_root = Path(sys.argv[3]).expanduser().resolve()


def object_urdf_fallbacks(path):
    expanded = path.expanduser()
    if resolve_data_file_path is not None:
        try:
            yield Path(resolve_data_file_path(str(expanded))).expanduser().resolve()
        except Exception:
            pass

    parts = expanded.parts
    if "data" in parts:
        data_idx = parts.index("data")
        yield repo_root.joinpath(*parts[data_idx:])
        rel_parts = parts[data_idx:]
        if len(rel_parts) >= 2 and rel_parts[0] == "data" and rel_parts[1] == "ds_box_data":
            yield repo_root.joinpath("data", "ds_box_data_legacy", *rel_parts[2:])

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


def object_collision_mode():
    raw = os.environ.get(
        "MUJOCO_OBJECT_COLLISION_MODE",
        os.environ.get("HOLOSOMA_MUJOCO_OBJECT_COLLISION_MODE", "mesh"),
    )
    return str(raw).strip().lower().replace("-", "_")


def truthy_env(*names, default=False):
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    return bool(default)


def set_collision_box(root_xml, desired_size):
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


def collision_box_matches(root_xml, desired_size):
    for box in root_xml.findall(".//collision/geometry/box"):
        try:
            size = parse_vec(box.get("size"), (0.0, 0.0, 0.0))
        except Exception:
            continue
        if np.allclose(size, desired_size, rtol=0.02, atol=2.0e-3):
            return True
    return False


def write_motion_sized_urdf(urdf_path):
    if not urdf_path.is_file():
        return urdf_path
    collision_mode = object_collision_mode()
    use_box_collision = collision_mode in {"box", "primitive", "primitive_box", "box_collision", "rollout"}
    resize_to_motion_size = truthy_env(
        "MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE",
        "HOLOSOMA_MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE",
        default=True,
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

    first_scale = parse_vec(first_mesh.get("scale"), (1.0, 1.0, 1.0))
    first_extents = obj_extents(mesh_path(urdf_path, first_filename))
    if first_extents is None:
        return urdf_path

    current_size = first_extents * first_scale
    if current_size.size != 3 or np.any(current_size <= 0.0):
        return urdf_path
    desired_size = motion_object_size() if resize_to_motion_size else current_size
    if desired_size is None:
        return urdf_path
    mesh_already_sized = np.allclose(current_size, desired_size, rtol=0.02, atol=2.0e-3)
    box_already_sized = use_box_collision and collision_box_matches(root_xml, desired_size)
    if mesh_already_sized and (not use_box_collision or box_already_sized):
        return urdf_path

    scale_ratio = np.ones(3, dtype=np.float64) if mesh_already_sized else desired_size / current_size
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

    if use_box_collision:
        set_collision_box(root_xml, desired_size)

    out_dir = repo_root / "logs/sim2sim_exports/object_urdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    size_source = "motion_size" if resize_to_motion_size else "urdf_size"
    suffix = f"{size_source}_box_collision" if use_box_collision else size_source
    out_path = out_dir / f"{motion_path.stem}__{urdf_path.stem}__{suffix}.urdf"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(
        f"[INFO] generated {size_source} object URDF {out_path}: current_size={current_size.tolist()} desired_size={desired_size.tolist()} scale_ratio={scale_ratio.tolist()} collision_mode={collision_mode}",
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
physx = sim.get("physx") if isinstance(sim.get("physx"), dict) else {}
emit("SIM_PHYSX_POSITION_ITERATIONS", physx.get("num_position_iterations"))
emit("SIM_PHYSX_VELOCITY_ITERATIONS", physx.get("num_velocity_iterations"))
emit("SIM_PHYSX_BOUNCE_THRESHOLD_VELOCITY", physx.get("bounce_threshold_velocity"))
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
      SIM_PHYSX_POSITION_ITERATIONS)
        if [[ -z "${SIM_PHYSX_POSITION_ITERATIONS:-}" ]]; then
          SIM_PHYSX_POSITION_ITERATIONS="$value"
        fi
        ;;
      SIM_PHYSX_VELOCITY_ITERATIONS)
        if [[ -z "${SIM_PHYSX_VELOCITY_ITERATIONS:-}" ]]; then
          SIM_PHYSX_VELOCITY_ITERATIONS="$value"
        fi
        ;;
      SIM_PHYSX_BOUNCE_THRESHOLD_VELOCITY)
        if [[ -z "${SIM_PHYSX_BOUNCE_THRESHOLD_VELOCITY:-}" ]]; then
          SIM_PHYSX_BOUNCE_THRESHOLD_VELOCITY="$value"
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
  MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-}"
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
import base64
import hashlib
import json
import math
import os
import re
import sys

import onnx

model = onnx.load(sys.argv[1])
metadata = {}
for prop in model.metadata_props:
    if prop.key in metadata:
        raise SystemExit(f"[ERROR] Duplicate ONNX metadata key {prop.key!r}.")
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

requires_perception = any(value.name == "perception_obs" for value in model.graph.input)
experiment_cfg = metadata.get("experiment_config", {})
if requires_perception and not isinstance(experiment_cfg, dict):
    raise SystemExit(
        "[ERROR] A perception ONNX must contain object-valued experiment_config metadata."
    )
experiment_cfg = experiment_cfg if isinstance(experiment_cfg, dict) else {}
perception_cfg = experiment_cfg.get("perception", {})
if not isinstance(perception_cfg, dict):
    if requires_perception:
        raise SystemExit(
            "[ERROR] A perception ONNX must contain object-valued experiment_config.perception metadata."
        )
    raise SystemExit(0)

encoder_type = perception_cfg.get("encoder_type")
if "camera_warp_normalize" in perception_cfg and not isinstance(
    perception_cfg["camera_warp_normalize"], bool
):
    raise SystemExit(
        "[ERROR] Policy metadata perception.camera_warp_normalize must be boolean."
    )
if isinstance(encoder_type, str) and encoder_type.strip().lower().startswith("defm_"):
    if perception_cfg.get("camera_warp_normalize") is not False:
        raise SystemExit(
            "[ERROR] DeFM policy metadata must explicitly set "
            "perception.camera_warp_normalize=false (metric depth in meters). "
            "Legacy normalized-depth DeFM artifacts require retraining."
        )

sensor_noise = perception_cfg.get("camera_apply_sensor_noise", False)
if not isinstance(sensor_noise, bool):
    raise SystemExit(
        "[ERROR] Policy metadata perception.camera_apply_sensor_noise must be boolean."
    )
for noise_key in ("camera_warp_edge_noise", "camera_warp_enable_holes"):
    if noise_key in perception_cfg and not isinstance(perception_cfg[noise_key], bool):
        raise SystemExit(
            f"[ERROR] Policy metadata perception.{noise_key} must be boolean."
        )
holes_enabled = perception_cfg.get("camera_warp_enable_holes", False)
reset_refresh_semantics = perception_cfg.get(
    "reset_refresh_semantics",
    "legacy_full_v1",
)
if not isinstance(reset_refresh_semantics, str):
    raise SystemExit(
        "[ERROR] Policy metadata perception.reset_refresh_semantics must be a string."
    )
noise_requested = any(
    perception_cfg.get(key, False)
    for key in (
        "camera_warp_edge_noise",
        "camera_warp_enable_holes",
        "camera_apply_sensor_noise",
    )
)


def parse_bool(value, *, path):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise SystemExit(f"[ERROR] {path} must be boolean, got {value!r}.")


if os.environ.get("PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT") == "1":
    explicit_sensor_noise = parse_bool(
        os.environ.get("PERCEPTION_CAMERA_APPLY_SENSOR_NOISE", ""),
        path="Explicit PERCEPTION_CAMERA_APPLY_SENSOR_NOISE",
    )
    if explicit_sensor_noise != sensor_noise:
        raise SystemExit(
            "[ERROR] Explicit PERCEPTION_CAMERA_APPLY_SENSOR_NOISE conflicts with "
            f"checkpoint metadata ({explicit_sensor_noise} != {sensor_noise}); refusing "
            "perception noise-distribution drift."
        )


def checked_number(value, *, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"[ERROR] {path} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"[ERROR] {path} must be finite, got {value!r}.")
    return result


for noise_key in (
    "camera_warp_additive_noise_std",
    "camera_warp_depth_offset_std",
):
    noise_value = checked_number(
        perception_cfg.get(noise_key, 0.0) or 0.0,
        path=f"experiment_config.perception.{noise_key}",
    )
    if noise_value < 0.0:
        raise SystemExit(
            f"[ERROR] experiment_config.perception.{noise_key} must be non-negative."
        )
    noise_requested = noise_requested or noise_value > 0.0


def checked_pair(value, *, path, minimum=None, maximum=None):
    if (
        isinstance(value, (str, bytes, dict))
        or not isinstance(value, (list, tuple))
        or len(value) != 2
    ):
        raise SystemExit(f"[ERROR] {path} must be a two-value [low, high] range.")
    low = checked_number(value[0], path=f"{path}[0]")
    high = checked_number(value[1], path=f"{path}[1]")
    if low > high:
        raise SystemExit(f"[ERROR] {path} must satisfy low <= high, got {value!r}.")
    if minimum is not None and low < minimum:
        raise SystemExit(f"[ERROR] {path} must be >= {minimum}, got {value!r}.")
    if maximum is not None and high > maximum:
        raise SystemExit(f"[ERROR] {path} must be <= {maximum}, got {value!r}.")
    return [low, high]


def checked_axes(value, *, path, axes):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SystemExit(f"[ERROR] {path} must be an axis-keyed object.")
    actual = set(value)
    expected = set(axes)
    if actual != expected:
        raise SystemExit(
            f"[ERROR] {path} axes must be exactly {list(axes)!r}; got {sorted(actual)!r}."
        )
    return {axis: checked_pair(value[axis], path=f"{path}.{axis}") for axis in axes}


canonical_camera_func = (
    "holosoma.managers.randomization.terms.locomotion:randomize_camera_raycast"
)
randomization_cfg = experiment_cfg.get("randomization", {})
if not isinstance(randomization_cfg, dict):
    if requires_perception:
        raise SystemExit(
            "[ERROR] experiment_config.randomization must be an object for a perception ONNX."
        )
    randomization_cfg = {}
reset_terms = randomization_cfg.get("reset_terms", {})
if not isinstance(reset_terms, dict):
    if requires_perception:
        raise SystemExit(
            "[ERROR] experiment_config.randomization.reset_terms must be an object."
        )
    reset_terms = {}

enabled_camera_terms = []
for term_name, term in reset_terms.items():
    if not isinstance(term, dict):
        if term_name == "randomize_camera_raycast":
            raise SystemExit(
                "[ERROR] randomize_camera_raycast reset term must be an object."
            )
        continue
    func = term.get("func")
    func_basename = str(func or "").replace(":", ".").rsplit(".", 1)[-1]
    if func != canonical_camera_func:
        if term_name == "randomize_camera_raycast" or func_basename == "randomize_camera_raycast":
            raise SystemExit(
                "[ERROR] Camera reset randomization must use the canonical function "
                f"{canonical_camera_func!r}; term {term_name!r} declares {func!r}."
            )
        continue
    params = term.get("params", {})
    if not isinstance(params, dict):
        raise SystemExit(
            f"[ERROR] Camera reset term {term_name!r}.params must be an object."
        )
    enabled = params.get("enabled", True)
    if not isinstance(enabled, bool):
        raise SystemExit(
            f"[ERROR] Camera reset term {term_name!r}.params.enabled must be boolean."
        )
    if enabled:
        allowed_params = {
            "enabled",
            "translation_range",
            "rotation_range_deg",
            "noise_std_mult_range",
            "noise_drop_prob_range",
        }
        unexpected_params = sorted(set(params) - allowed_params)
        if unexpected_params:
            raise SystemExit(
                "[ERROR] Enabled camera reset randomization contains unsupported/"
                f"unauthenticated parameters: {unexpected_params!r}."
            )
        enabled_camera_terms.append((term_name, params))

if len(enabled_camera_terms) > 1:
    raise SystemExit(
        "[ERROR] Expected at most one enabled canonical randomize_camera_raycast reset term; "
        f"found {[name for name, _ in enabled_camera_terms]!r}."
    )

camera_params = enabled_camera_terms[0][1] if enabled_camera_terms else None
translation_range = None
rotation_range_deg = None
noise_std_mult_range = None
noise_drop_prob_range = None
if camera_params is not None:
    translation_range = checked_axes(
        camera_params.get("translation_range"),
        path="randomize_camera_raycast.params.translation_range",
        axes=("x", "y", "z"),
    )
    rotation_range_deg = checked_axes(
        camera_params.get("rotation_range_deg"),
        path="randomize_camera_raycast.params.rotation_range_deg",
        axes=("roll", "pitch", "yaw"),
    )
    if camera_params.get("noise_std_mult_range") is not None:
        noise_std_mult_range = checked_pair(
            camera_params["noise_std_mult_range"],
            path="randomize_camera_raycast.params.noise_std_mult_range",
            minimum=0.0,
        )
    if camera_params.get("noise_drop_prob_range") is not None:
        noise_drop_prob_range = checked_pair(
            camera_params["noise_drop_prob_range"],
            path="randomize_camera_raycast.params.noise_drop_prob_range",
            minimum=0.0,
            maximum=1.0,
        )

if requires_perception and sensor_noise and (
    camera_params is None
    or (noise_std_mult_range is None and noise_drop_prob_range is None)
):
    raise SystemExit(
        "[ERROR] perception.camera_apply_sensor_noise=True requires one enabled canonical "
        "randomize_camera_raycast reset term with at least one explicit "
        "noise_std_mult_range or noise_drop_prob_range."
    )

expected_camera_summary = None
if camera_params is not None:
    expected_camera_summary = {
        "enabled": True,
        "translation_xyz": (
            None
            if translation_range is None
            else [translation_range[axis] for axis in ("x", "y", "z")]
        ),
        "rotation_rpy_deg": (
            None
            if rotation_range_deg is None
            else [rotation_range_deg[axis] for axis in ("roll", "pitch", "yaw")]
        ),
        "noise_std_mult": noise_std_mult_range,
        "noise_drop_prob": noise_drop_prob_range,
    }


def checked_contract_camera_summary(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SystemExit(
            "[ERROR] perception_observation_contract.camera_reset_randomization must be an object or null."
        )
    expected_keys = {
        "enabled",
        "translation_xyz",
        "rotation_rpy_deg",
        "noise_std_mult",
        "noise_drop_prob",
    }
    if set(value) != expected_keys:
        raise SystemExit(
            "[ERROR] perception_observation_contract.camera_reset_randomization has an "
            f"unexpected schema: {sorted(value)!r}."
        )
    if value["enabled"] is not True:
        raise SystemExit(
            "[ERROR] Attached camera_reset_randomization.enabled must be true when the summary is present."
        )

    def vector(value, *, path):
        if value is None:
            return None
        if (
            isinstance(value, (str, bytes, dict))
            or not isinstance(value, (list, tuple))
            or len(value) != 3
        ):
            raise SystemExit(f"[ERROR] {path} must contain three axis ranges.")
        return [checked_pair(pair, path=f"{path}[{index}]") for index, pair in enumerate(value)]

    std = value["noise_std_mult"]
    drop = value["noise_drop_prob"]
    return {
        "enabled": True,
        "translation_xyz": vector(
            value["translation_xyz"],
            path="perception_observation_contract.camera_reset_randomization.translation_xyz",
        ),
        "rotation_rpy_deg": vector(
            value["rotation_rpy_deg"],
            path="perception_observation_contract.camera_reset_randomization.rotation_rpy_deg",
        ),
        "noise_std_mult": (
            None
            if std is None
            else checked_pair(
                std,
                path="perception_observation_contract.camera_reset_randomization.noise_std_mult",
                minimum=0.0,
            )
        ),
        "noise_drop_prob": (
            None
            if drop is None
            else checked_pair(
                drop,
                path="perception_observation_contract.camera_reset_randomization.noise_drop_prob",
                minimum=0.0,
                maximum=1.0,
            )
        ),
    }


contract_present = "perception_observation_contract" in metadata
digest_present = "perception_observation_contract_sha256" in metadata
if contract_present != digest_present:
    raise SystemExit(
        "[ERROR] Attached perception observation contract and its SHA-256 digest must be present together."
    )
contract_status = "not-applicable"
perception_contract_envelope_b64 = None
hole_reference_batch_size = None
attached_producer_tick_dt = None
if requires_perception and contract_present:
    contract = metadata["perception_observation_contract"]
    declared_digest = metadata["perception_observation_contract_sha256"]
    if not isinstance(contract, dict):
        raise SystemExit("[ERROR] perception_observation_contract must be an object.")
    contract_version = contract.get("version")
    if (
        isinstance(contract_version, bool)
        or not isinstance(contract_version, int)
        or contract_version != 2
    ):
        raise SystemExit(
            "[ERROR] Direct perception deployment requires perception observation contract "
            "version=2. Re-export the policy after the targeted producer-lifecycle fix."
        )
    if not isinstance(declared_digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", declared_digest) is None:
        raise SystemExit(
            "[ERROR] perception_observation_contract_sha256 must be 64 hexadecimal characters."
        )
    try:
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] Invalid attached perception observation contract: {exc}")
    computed_digest = hashlib.sha256(payload).hexdigest()
    if computed_digest != declared_digest.lower():
        raise SystemExit(
            "[ERROR] Attached perception observation contract SHA-256 does not match its payload."
        )
    training_geometry_support = contract.get("training_geometry_support")
    if not isinstance(training_geometry_support, dict):
        raise SystemExit(
            "[ERROR] Version-2 perception contract lacks object-valued "
            "training_geometry_support. Re-export the policy with authenticated training geometry."
        )
    envelope_payload = json.dumps(
        {
            "contract": contract,
            "sha256": computed_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    perception_contract_envelope_b64 = base64.b64encode(envelope_payload).decode("ascii")
    missing = object()
    producer_lifecycle = contract.get("producer_lifecycle", missing)
    additive_noise_std_raw = perception_cfg.get("camera_warp_additive_noise_std", 0.0)
    additive_noise_std = checked_number(
        additive_noise_std_raw,
        path="experiment_config.perception.camera_warp_additive_noise_std",
    )
    if additive_noise_std < 0.0:
        raise SystemExit(
            "[ERROR] experiment_config.perception.camera_warp_additive_noise_std must be non-negative."
        )
    reset_randomization_consumes_rng = bool(
        camera_params is not None
        and any(
            value is not None
            for value in (
                translation_range,
                rotation_range_deg,
                noise_std_mult_range,
                noise_drop_prob_range,
            )
        )
    )
    sensor_noise_on_reset = bool(
        sensor_noise
        and camera_params is not None
        and (noise_std_mult_range is not None or noise_drop_prob_range is not None)
    )
    reset_refresh_pixel_rng = bool(
        perception_cfg.get("output_mode") == "camera_depth"
        and (
            perception_cfg.get("camera_warp_edge_noise", False)
            or additive_noise_std > 0.0
            or perception_cfg.get("camera_warp_latency_frame_range") is not None
            or sensor_noise_on_reset
        )
    )
    expected_reset_refresh_consumes_global_rng = bool(
        reset_randomization_consumes_rng or reset_refresh_pixel_rng
    )
    expected_producer_lifecycle = {
        "reset_refresh_semantics": "targeted_v2",
        "ordinary_manager_update_calls_per_control_tick": 1,
        "initialization_control_ticks_before_first_reset_output": 1,
        "initialization_ordinary_manager_update_calls_before_first_reset_output": 1,
        "reset_output_republished_until_physics_advances": True,
        "reset_output_scope": "reset_env_subset",
        "hole_clock_advances_on_reset_refresh": False,
        "camera_frequency_phase_advances_on_reset_refresh": False,
        "camera_producer_reset_refresh_consumes_process_global_rng": expected_reset_refresh_consumes_global_rng,
        "future_noise_sample_path_peer_reset_coupled": expected_reset_refresh_consumes_global_rng,
        "batch_size_invariant_sample_path": False,
        "stochastic_equivalence": "distribution_only",
        "seed_replay_scope": "same_execution_trace_only",
    }
    if producer_lifecycle != expected_producer_lifecycle:
        raise SystemExit(
            "[ERROR] Direct perception deployment requires the authenticated version-2 "
            "targeted_v2 producer_lifecycle distribution contract. It must honestly declare "
            "process-global RNG consumption, peer-reset sample-path coupling, and "
            "stochastic_equivalence='distribution_only' with seed replay limited to the same "
            "execution trace. Fix and re-export/retrain."
        )
    if reset_refresh_semantics != producer_lifecycle["reset_refresh_semantics"]:
        raise SystemExit(
            "[ERROR] Attached reset_refresh_semantics conflicts with "
            "experiment_config.perception.reset_refresh_semantics."
        )
    camera_setup_randomization = contract.get("camera_setup_randomization", missing)
    if camera_setup_randomization is missing:
        raise SystemExit(
            "[ERROR] Version-2 perception contract lacks camera_setup_randomization."
        )
    if camera_setup_randomization is not None and (
        not isinstance(camera_setup_randomization, dict)
        or camera_setup_randomization.get("enabled") is not False
    ):
        raise SystemExit(
            "[ERROR] Direct reset reconstruction requires attached camera_setup_randomization "
            "to be null or explicitly disabled; legacy one-shot jitter cannot be stacked."
        )
    attached_producer_tick_dt = checked_number(
        contract.get("producer_tick_dt"),
        path="perception_observation_contract.producer_tick_dt",
    )
    if attached_producer_tick_dt <= 0.0:
        raise SystemExit(
            "[ERROR] perception_observation_contract.producer_tick_dt must be positive."
        )
    hole_schema = contract.get("hole_generator_schema", missing)
    if hole_schema is missing:
        raise SystemExit(
            "[ERROR] Attached perception observation contract lacks hole_generator_schema."
        )
    if holes_enabled:
        if not isinstance(hole_schema, dict):
            raise SystemExit(
                "[ERROR] camera_warp_enable_holes=True requires an attached hole_generator_schema object."
            )
        if hole_schema.get("normalization_scope") != "reference_batch":
            raise SystemExit(
                "[ERROR] Attached hole_generator_schema.normalization_scope must be 'reference_batch'."
            )
        raw_reference_batch_size = hole_schema.get("reference_batch_size")
        if (
            isinstance(raw_reference_batch_size, bool)
            or not isinstance(raw_reference_batch_size, int)
            or raw_reference_batch_size <= 0
        ):
            raise SystemExit(
                "[ERROR] Attached hole_generator_schema.reference_batch_size must be a positive integer."
            )
        hole_reference_batch_size = raw_reference_batch_size
    elif hole_schema is not None:
        raise SystemExit(
            "[ERROR] Attached hole_generator_schema is present while camera_warp_enable_holes=False."
        )
    attached_summary = contract.get("camera_reset_randomization", missing)
    if attached_summary is missing:
        raise SystemExit(
            "[ERROR] Attached perception observation contract lacks camera_reset_randomization."
        )
    normalized_attached_summary = checked_contract_camera_summary(attached_summary)
    if normalized_attached_summary != expected_camera_summary:
        raise SystemExit(
            "[ERROR] Attached perception camera_reset_randomization conflicts with "
            "experiment_config.randomization.reset_terms."
        )
    contract_status = "attached-v2-targeted-v2-distribution-verified"
elif requires_perception:
    raise SystemExit(
        "[ERROR] Direct perception deployment requires an attached version-2 observation "
        "contract with a targeted_v2 producer_lifecycle. Legacy/missing-contract artifacts "
        "cannot establish an authenticated distribution-compatible direct producer; fix the "
        "reset path and re-export/retrain."
    )


def checked_json_env(name, expected, validator):
    if name not in os.environ:
        return
    raw = os.environ[name]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] Explicit {name} must be valid JSON: {exc}.")
    normalized = validator(parsed)
    if normalized != expected:
        raise SystemExit(
            f"[ERROR] Explicit {name} conflicts with checkpoint camera reset metadata."
        )


if requires_perception:
    if "PERCEPTION_RANDOMIZATION_ENABLED" in os.environ:
        explicit_enabled = parse_bool(
            os.environ["PERCEPTION_RANDOMIZATION_ENABLED"],
            path="Explicit PERCEPTION_RANDOMIZATION_ENABLED",
        )
        if explicit_enabled != (camera_params is not None):
            raise SystemExit(
                "[ERROR] Explicit PERCEPTION_RANDOMIZATION_ENABLED conflicts with checkpoint camera reset metadata."
            )
    checked_json_env(
        "PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE",
        translation_range,
        lambda value: checked_axes(
            value,
            path="Explicit PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE",
            axes=("x", "y", "z"),
        ),
    )
    checked_json_env(
        "PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG",
        rotation_range_deg,
        lambda value: checked_axes(
            value,
            path="Explicit PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG",
            axes=("roll", "pitch", "yaw"),
        ),
    )
    checked_json_env(
        "PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE",
        noise_std_mult_range,
        lambda value: (
            None
            if value is None
            else checked_pair(
                value,
                path="Explicit PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE",
                minimum=0.0,
            )
        ),
    )
    checked_json_env(
        "PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE",
        noise_drop_prob_range,
        lambda value: (
            None
            if value is None
            else checked_pair(
                value,
                path="Explicit PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE",
                minimum=0.0,
                maximum=1.0,
            )
        ),
    )
    if "HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT" in os.environ and parse_bool(
        os.environ["HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT"],
        path="Explicit HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT",
    ):
        raise SystemExit(
            "[ERROR] HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT enables legacy one-shot jitter and "
            "conflicts with authenticated direct reset randomization. Remove it or set it false."
        )
    if "PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE" in os.environ:
        explicit_hole_batch_raw = os.environ[
            "PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE"
        ]
        try:
            explicit_hole_batch = int(explicit_hole_batch_raw)
        except ValueError:
            raise SystemExit(
                "[ERROR] Explicit PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE must be an integer."
            )
        if (
            str(explicit_hole_batch) != explicit_hole_batch_raw.strip()
            or explicit_hole_batch != hole_reference_batch_size
        ):
            raise SystemExit(
                "[ERROR] Explicit PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE conflicts "
                "with the attached perception contract."
            )
    for allow_name in (
        "PERCEPTION_ALLOW_MUJOCO_NOISE",
        "HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE",
    ):
        marker_name = f"{allow_name}_EXPLICIT"
        is_explicit = allow_name in os.environ
        if allow_name == "HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE":
            is_explicit = os.environ.get(marker_name) == "1"
        if not is_explicit:
            continue
        explicit_allow = parse_bool(
            os.environ.get(allow_name, ""),
            path=f"Explicit {allow_name}",
        )
        if explicit_allow != noise_requested:
            raise SystemExit(
                f"[ERROR] Explicit {allow_name} conflicts with checkpoint perception noise settings."
            )

    simulator_cfg = experiment_cfg.get("simulator")
    if not isinstance(simulator_cfg, dict):
        raise SystemExit(
            "[ERROR] Perception ONNX metadata must declare experiment_config.simulator."
        )
    simulator_config = simulator_cfg.get("config")
    if not isinstance(simulator_config, dict):
        raise SystemExit(
            "[ERROR] Perception ONNX metadata must declare experiment_config.simulator.config."
        )
    training_sim = simulator_config.get("sim")
    if not isinstance(training_sim, dict):
        raise SystemExit(
            "[ERROR] Perception ONNX metadata must declare experiment_config.simulator.config.sim."
        )
    training_fps = checked_number(
        training_sim.get("fps"),
        path="experiment_config.simulator.config.sim.fps",
    )
    training_decimation_raw = training_sim.get("control_decimation")
    training_decimation = checked_number(
        training_decimation_raw,
        path="experiment_config.simulator.config.sim.control_decimation",
    )
    if training_fps <= 0.0 or training_decimation <= 0.0:
        raise SystemExit(
            "[ERROR] Training simulator fps and control_decimation must both be positive."
        )
    if not training_decimation.is_integer():
        raise SystemExit(
            "[ERROR] Training simulator control_decimation must be an integer."
        )
    producer_tick_dt = training_decimation / training_fps
    if not math.isfinite(producer_tick_dt) or producer_tick_dt <= 0.0:
        raise SystemExit("[ERROR] Computed perception producer tick dt must be finite and positive.")
    if attached_producer_tick_dt is None or not math.isclose(
        attached_producer_tick_dt,
        producer_tick_dt,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise SystemExit(
            "[ERROR] Attached perception producer_tick_dt conflicts with checkpoint simulator "
            f"cadence ({attached_producer_tick_dt!r} != {producer_tick_dt:.17g})."
        )
    if "PERCEPTION_PRODUCER_TICK_DT" in os.environ:
        explicit_tick = os.environ["PERCEPTION_PRODUCER_TICK_DT"]
        try:
            explicit_tick_value = float(explicit_tick)
        except ValueError:
            raise SystemExit(
                f"[ERROR] Explicit PERCEPTION_PRODUCER_TICK_DT must be numeric, got {explicit_tick!r}."
            )
        if (
            not math.isfinite(explicit_tick_value)
            or explicit_tick_value <= 0.0
            or not math.isclose(
                explicit_tick_value,
                producer_tick_dt,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise SystemExit(
                "[ERROR] Explicit PERCEPTION_PRODUCER_TICK_DT conflicts with checkpoint "
                f"training cadence ({explicit_tick!r} != {producer_tick_dt:.17g})."
            )
    training_cfg = experiment_cfg.get("training")
    if not isinstance(training_cfg, dict):
        raise SystemExit(
            "[ERROR] Perception ONNX metadata must declare experiment_config.training."
        )
    producer_seed = training_cfg.get("seed")
    if (
        isinstance(producer_seed, bool)
        or not isinstance(producer_seed, int)
        or producer_seed < 0
    ):
        raise SystemExit(
            "[ERROR] Checkpoint training seed must be a non-negative integer, "
            f"got {producer_seed!r}."
        )
    if "PERCEPTION_PRODUCER_SEED" in os.environ:
        explicit_seed_raw = os.environ["PERCEPTION_PRODUCER_SEED"]
        try:
            explicit_seed = int(explicit_seed_raw)
        except ValueError:
            raise SystemExit(
                f"[ERROR] Explicit PERCEPTION_PRODUCER_SEED must be an integer, got {explicit_seed_raw!r}."
            )
        if str(explicit_seed) != explicit_seed_raw.strip() or explicit_seed != producer_seed:
            raise SystemExit(
                "[ERROR] Explicit PERCEPTION_PRODUCER_SEED conflicts with checkpoint training seed."
            )
    print(f"PERCEPTION_RANDOMIZATION_ENABLED={camera_params is not None}")
    if translation_range is not None:
        print(
            "PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE="
            + json.dumps(translation_range, separators=(",", ":"), allow_nan=False)
        )
    if rotation_range_deg is not None:
        print(
            "PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG="
            + json.dumps(rotation_range_deg, separators=(",", ":"), allow_nan=False)
        )
    if noise_std_mult_range is not None:
        print(
            "PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE="
            + json.dumps(noise_std_mult_range, separators=(",", ":"), allow_nan=False)
        )
    if noise_drop_prob_range is not None:
        print(
            "PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE="
            + json.dumps(noise_drop_prob_range, separators=(",", ":"), allow_nan=False)
        )
    if hole_reference_batch_size is not None:
        print(
            "PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE="
            f"{hole_reference_batch_size}"
        )
    print(f"PERCEPTION_RANDOMIZATION_CONTRACT_STATUS={contract_status}")
    if not perception_contract_envelope_b64:
        raise SystemExit("[ERROR] Failed to construct the authenticated perception contract envelope.")
    print(f"PERCEPTION_CONTRACT_ENVELOPE_B64={perception_contract_envelope_b64}")
    print(f"PERCEPTION_PRODUCER_TICK_DT={producer_tick_dt:.17g}")
    print(f"PERCEPTION_PRODUCER_SEED={producer_seed}")
    print(f"PERCEPTION_ALLOW_MUJOCO_NOISE={noise_requested}")

field_map = {
    "object_geometry_mode": "PERCEPTION_OBJECT_GEOMETRY_MODE",
    "update_hz": "PERCEPTION_UPDATE_HZ",
    "camera_fps": "PERCEPTION_CAMERA_FPS",
    "camera_width": "PERCEPTION_CAMERA_WIDTH",
    "camera_height": "PERCEPTION_CAMERA_HEIGHT",
    "camera_pitch_deg": "PERCEPTION_CAMERA_PITCH_DEG",
    "camera_vfov_deg": "PERCEPTION_CAMERA_VFOV_DEG",
    "camera_hfov_deg": "PERCEPTION_CAMERA_HFOV_DEG",
    "camera_include_robot_mesh": "PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH",
    "camera_near": "PERCEPTION_CAMERA_NEAR",
    "camera_far": "PERCEPTION_CAMERA_FAR",
    "max_distance": "PERCEPTION_MAX_DISTANCE",
    "camera_warp_crop_top": "PERCEPTION_CAMERA_WARP_CROP_TOP",
    "camera_warp_crop_bottom": "PERCEPTION_CAMERA_WARP_CROP_BOTTOM",
    "camera_warp_crop_left": "PERCEPTION_CAMERA_WARP_CROP_LEFT",
    "camera_warp_crop_right": "PERCEPTION_CAMERA_WARP_CROP_RIGHT",
    "camera_warp_min_valid_depth": "PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH",
    "camera_warp_normalize": "PERCEPTION_CAMERA_WARP_NORMALIZE",
    "camera_warp_buffer_len": "PERCEPTION_CAMERA_WARP_BUFFER_LEN",
    "camera_warp_latency_frame": "PERCEPTION_CAMERA_WARP_LATENCY_FRAME",
    "camera_warp_edge_noise": "PERCEPTION_CAMERA_WARP_EDGE_NOISE",
    "camera_warp_edge_border": "PERCEPTION_CAMERA_WARP_EDGE_BORDER",
    "camera_warp_edge_shuffle_prob": "PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB",
    "camera_warp_edge_empty_prob": "PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB",
    "camera_warp_edge_thresh_primary": "PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY",
    "camera_warp_edge_thresh_secondary": "PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY",
    "camera_warp_edge_far_depth_thresh": "PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH",
    "camera_warp_enable_holes": "PERCEPTION_CAMERA_WARP_ENABLE_HOLES",
    "camera_warp_hole_prob": "PERCEPTION_CAMERA_WARP_HOLE_PROB",
    "camera_apply_sensor_noise": "PERCEPTION_CAMERA_APPLY_SENSOR_NOISE",
}
for src_key, env_key in field_map.items():
    value = perception_cfg.get(src_key)
    if value is None:
        continue
    if isinstance(value, bool):
        print(f"{env_key}={value}")
    elif isinstance(value, (list, tuple)):
        # Tyro consumes variable-length sequences as separate CLI tokens.
        # Emitting Python's ``[3, 4]`` representation makes the shell pass
        # invalid tokens (``[3,`` and ``4]``).
        print(f"{env_key}=" + " ".join(str(item) for item in value))
    elif isinstance(value, int):
        print(f"{env_key}={value}")
    elif isinstance(value, float):
        print(f"{env_key}={value:g}")
    else:
        print(f"{env_key}={value}")

print(f"HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE={noise_requested}")
PY
  )"

  if [[ -z "$override_lines" ]]; then
    return
  fi

  while IFS='=' read -r key value; do
    [[ -z "${key:-}" ]] && continue
    case "$key" in
      PERCEPTION_RANDOMIZATION_ENABLED)
        PERCEPTION_RANDOMIZATION_ENABLED="$value"
        ;;
      PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE)
        PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE="$value"
        ;;
      PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG)
        PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG="$value"
        ;;
      PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE)
        PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE="$value"
        ;;
      PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE)
        PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE="$value"
        ;;
      PERCEPTION_RANDOMIZATION_CONTRACT_STATUS)
        PERCEPTION_RANDOMIZATION_CONTRACT_STATUS="$value"
        ;;
      PERCEPTION_CONTRACT_ENVELOPE_B64)
        PERCEPTION_CONTRACT_ENVELOPE_B64="$value"
        ;;
      PERCEPTION_PRODUCER_TICK_DT)
        PERCEPTION_PRODUCER_TICK_DT="$value"
        ;;
      PERCEPTION_PRODUCER_SEED)
        PERCEPTION_PRODUCER_SEED="$value"
        ;;
      PERCEPTION_ALLOW_MUJOCO_NOISE)
        PERCEPTION_ALLOW_MUJOCO_NOISE="$value"
        ;;
      PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE)
        PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE="$value"
        ;;
      PERCEPTION_OBJECT_GEOMETRY_MODE)
        if [[ -z "$PERCEPTION_OBJECT_GEOMETRY_MODE" ]]; then
          PERCEPTION_OBJECT_GEOMETRY_MODE="$value"
        fi
        ;;
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
      PERCEPTION_CAMERA_WIDTH)
        if [[ "$PERCEPTION_CAMERA_WIDTH_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_WIDTH" ]]; then
          PERCEPTION_CAMERA_WIDTH="$value"
        fi
        ;;
      PERCEPTION_CAMERA_HEIGHT)
        if [[ "$PERCEPTION_CAMERA_HEIGHT_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_HEIGHT" ]]; then
          PERCEPTION_CAMERA_HEIGHT="$value"
        fi
        ;;
      PERCEPTION_CAMERA_PITCH_DEG)
        if [[ -z "$PERCEPTION_CAMERA_PITCH_DEG" ]]; then
          PERCEPTION_CAMERA_PITCH_DEG="$value"
        fi
        ;;
      PERCEPTION_CAMERA_VFOV_DEG)
        if [[ -z "$PERCEPTION_CAMERA_VFOV_DEG" ]]; then
          PERCEPTION_CAMERA_VFOV_DEG="$value"
        fi
        ;;
      PERCEPTION_CAMERA_HFOV_DEG)
        if [[ -z "$PERCEPTION_CAMERA_HFOV_DEG" ]]; then
          PERCEPTION_CAMERA_HFOV_DEG="$value"
        fi
        ;;
      PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH)
        if [[ -z "$PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH" ]]; then
          PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH="$value"
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
      PERCEPTION_CAMERA_WARP_NORMALIZE)
        checkpoint_normalize="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
        runtime_normalize="$(printf '%s' "${PERCEPTION_CAMERA_WARP_NORMALIZE:-}" | tr '[:upper:]' '[:lower:]')"
        if [[ "${PERCEPTION_CAMERA_WARP_NORMALIZE_EXPLICIT:-0}" == "1" && "$runtime_normalize" != "$checkpoint_normalize" ]]; then
          echo "[ERROR] Explicit PERCEPTION_CAMERA_WARP_NORMALIZE=${PERCEPTION_CAMERA_WARP_NORMALIZE} conflicts with checkpoint metadata ${value}; refusing depth-unit drift." >&2
          return 2
        fi
        PERCEPTION_CAMERA_WARP_NORMALIZE="$value"
        export PERCEPTION_CAMERA_WARP_NORMALIZE
        ;;
      PERCEPTION_CAMERA_WARP_CROP_TOP)
        if [[ "$PERCEPTION_CAMERA_WARP_CROP_TOP_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_WARP_CROP_TOP" ]]; then
          PERCEPTION_CAMERA_WARP_CROP_TOP="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_CROP_BOTTOM)
        if [[ "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM" ]]; then
          PERCEPTION_CAMERA_WARP_CROP_BOTTOM="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_CROP_LEFT)
        if [[ "$PERCEPTION_CAMERA_WARP_CROP_LEFT_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_WARP_CROP_LEFT" ]]; then
          PERCEPTION_CAMERA_WARP_CROP_LEFT="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_CROP_RIGHT)
        if [[ "$PERCEPTION_CAMERA_WARP_CROP_RIGHT_EXPLICIT" != "1" && -z "$PERCEPTION_CAMERA_WARP_CROP_RIGHT" ]]; then
          PERCEPTION_CAMERA_WARP_CROP_RIGHT="$value"
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
      PERCEPTION_CAMERA_WARP_EDGE_NOISE)
        holosoma_apply_checkpoint_value "$key" "$value"
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_BORDER)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_BORDER" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_BORDER="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH="$value"
        fi
        ;;
      PERCEPTION_CAMERA_WARP_ENABLE_HOLES)
        holosoma_apply_checkpoint_value "$key" "$value"
        ;;
      PERCEPTION_CAMERA_WARP_HOLE_PROB)
        if [[ -z "$PERCEPTION_CAMERA_WARP_HOLE_PROB" ]]; then
          PERCEPTION_CAMERA_WARP_HOLE_PROB="$value"
        fi
        ;;
      PERCEPTION_CAMERA_APPLY_SENSOR_NOISE)
        holosoma_apply_checkpoint_value "$key" "$value"
        ;;
      HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE)
        holosoma_apply_checkpoint_value "$key" "$value"
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
from holosoma_inference.utils.policy_contract import (
    effective_motion_transition_settings_from_metadata,
)


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

transition_settings = effective_motion_transition_settings_from_metadata(metadata)
effective_transition_applied = bool(
    transition_settings["prepend"]["applied"]
    or transition_settings["append"]["applied"]
)
effective_prepend_applied = bool(transition_settings["prepend"]["applied"])

if effective_transition_applied:
    print("APPLY_TRAINING_MOTION_TRANSITIONS=1")
if effective_prepend_applied:
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
  "$INFER_PY" "$ROOT_DIR/scripts/mj_infer_inference_config.py" "$1"
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

apply_training_motion_launch_defaults "$POLICY_MODEL"
apply_motion_clip_object_defaults
OBJECT_URDF="$(resolve_motion_sized_object_urdf "$OBJECT_URDF")"

apply_training_sim_overrides "$POLICY_MODEL"
apply_training_robot_init_overrides "$POLICY_MODEL"
apply_training_robot_asset_overrides "$POLICY_MODEL"
apply_training_object_overrides "$POLICY_MODEL"
apply_training_perception_overrides "$POLICY_MODEL"
apply_gt_mujoco_physics_overrides

PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-primitive}"

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
  INFERENCE_CONFIG="$(infer_inference_config "$POLICY_MODEL")"
fi

if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ -n "$POLICY_MOTION_INDEX_OFFSET" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET="$POLICY_MOTION_INDEX_OFFSET"
  elif [[ "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-contact-aware-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-contact-aware-drop-button-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-contact-aware-dual-button-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-contact-aware-dual-button-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-contact-aware-pickup-drop-button-depth-distill" || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-mocap-distill" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  fi
fi

MODEL_EXPECTS_PERCEPTION_OBS="$(onnx_has_input "$POLICY_MODEL" "perception_obs")"
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "auto" ]]; then
  ENABLE_SPLIT_PERCEPTION_OBS="$MODEL_EXPECTS_PERCEPTION_OBS"
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" && "$ENABLE_SPLIT_PERCEPTION_OBS" != "1" ]]; then
  echo "Model expects perception_obs but ENABLE_SPLIT_PERCEPTION_OBS=${ENABLE_SPLIT_PERCEPTION_OBS}" >&2
  exit 1
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" ]]; then
  case "$PERCEPTION_RANDOMIZATION_ENABLED" in
    True|False)
      ;;
    *)
      echo "[ERROR] Missing authenticated direct perception randomization state." >&2
      exit 1
      ;;
  esac
  case "$PERCEPTION_ALLOW_MUJOCO_NOISE" in
    True|False)
      ;;
    *)
      echo "[ERROR] Missing authenticated direct perception noise permission." >&2
      exit 1
      ;;
  esac
  if [[ -z "$PERCEPTION_PRODUCER_TICK_DT" ]]; then
    echo "[ERROR] Missing checkpoint-derived perception producer tick dt." >&2
    exit 1
  fi
  if [[ -z "$PERCEPTION_PRODUCER_SEED" ]]; then
    echo "[ERROR] Missing checkpoint-derived perception producer seed." >&2
    exit 1
  fi
  # Direct reset randomization is carried by RunSimConfig.  Always suppress the
  # historical one-shot mount jitter so the two distributions cannot stack.
  export HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT=False
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
    1|true|yes|on|myholosoma|mujoco848|rendered848|mujoco_render_848x480)
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

if [[ "$INFERENCE_CONFIG" == "g1-29dof-wbt-w-object" \
      || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-generalist" \
      || "$INFERENCE_CONFIG" == "g1-29dof-wbt-w-object-history1" \
      || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-generalist-history1" \
      || "$INFERENCE_CONFIG" == "g1-29dof-wbt-w-object-legacy" \
      || "$INFERENCE_CONFIG" == "g1-29dof-wbt-object-velocity-generalist" ]]; then
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
  if [[ -n "$HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE" ]]; then
    POLICY_ACTION_SCALE="$HOLOSOMA_ONNX_ACTION_SCALE_OVERRIDE"
  else
    POLICY_ACTION_SCALE="$(
      "$INFER_PY" - <<'PY' "$POLICY_MODEL"
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
  echo "[INFO] policy_model=${POLICY_MODEL}"
  echo "[INFO] inference_config=${INFERENCE_CONFIG}"
  echo "[INFO] sim_device=${SIM_DEVICE:-<default>}"
  echo "[INFO] mujoco_object_scene training_urdf=${SIM_USE_TRAINING_URDF_OBJECT_SCENE} default_actuators=${SIM_ADD_DEFAULT_OBJECT_ACTUATORS} copy_joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} copy_tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} copy_collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} copy_contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML}"
  echo "[INFO] perception camera: source=${PERCEPTION_CAMERA_SOURCE} object_geometry_mode=${PERCEPTION_OBJECT_GEOMETRY_MODE} raw=${PERCEPTION_CAMERA_WIDTH:-<default>}x${PERCEPTION_CAMERA_HEIGHT:-<default>} crop_top=${PERCEPTION_CAMERA_WARP_CROP_TOP:-<default>} crop_bottom=${PERCEPTION_CAMERA_WARP_CROP_BOTTOM:-<default>} crop_left=${PERCEPTION_CAMERA_WARP_CROP_LEFT:-<default>} crop_right=${PERCEPTION_CAMERA_WARP_CROP_RIGHT:-<default>} update_hz=${PERCEPTION_UPDATE_HZ:-<default>} camera_fps=${PERCEPTION_CAMERA_FPS:-<default>} pitch_deg=${PERCEPTION_CAMERA_PITCH_DEG:-<default>} vfov_deg=${PERCEPTION_CAMERA_VFOV_DEG:-<default>} hfov_deg=${PERCEPTION_CAMERA_HFOV_DEG:-<default>} include_robot_mesh=${PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH:-<default>} near=${PERCEPTION_CAMERA_NEAR:-<default>} far=${PERCEPTION_CAMERA_FAR:-<default>} max_distance=${PERCEPTION_MAX_DISTANCE:-<default>} warp_min_valid_depth=${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-<default>} warp_normalize=${PERCEPTION_CAMERA_WARP_NORMALIZE:-<default>} warp_buffer_len=${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-<default>} warp_latency_frame=${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-<default>} warp_edge_noise=${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-<default>} warp_edge_border=${PERCEPTION_CAMERA_WARP_EDGE_BORDER:-<default>} warp_edge_shuffle_prob=${PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB:-<default>} warp_edge_empty_prob=${PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB:-<default>} warp_edge_thresh=${PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY:-<default>}/${PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY:-<default>} warp_edge_far_depth_thresh=${PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH:-<default>} warp_holes=${PERCEPTION_CAMERA_WARP_ENABLE_HOLES:-<default>} warp_hole_prob=${PERCEPTION_CAMERA_WARP_HOLE_PROB:-<default>} sensor_noise=${PERCEPTION_CAMERA_APPLY_SENSOR_NOISE:-<default>} allow_mujoco_noise=${HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE:-<default>} transport=${PERCEPTION_OBS_TRANSPORT}"
  echo "[INFO] direct perception reset: contract=${PERCEPTION_RANDOMIZATION_CONTRACT_STATUS:-<missing>} enabled=${PERCEPTION_RANDOMIZATION_ENABLED:-<missing>} translation=${PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE:-<none>} rotation_deg=${PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG:-<none>} noise_std_mult=${PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE:-<none>} noise_drop_prob=${PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE:-<none>} producer_tick_dt=${PERCEPTION_PRODUCER_TICK_DT:-<missing>} producer_seed=${PERCEPTION_PRODUCER_SEED:-<missing>} seed_replay_scope=same_execution_trace_only allow_mujoco_noise=${PERCEPTION_ALLOW_MUJOCO_NOISE:-<missing>} hole_reference_batch_size=${PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE:-<none>} legacy_setup_jitter=${HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT:-<unset>}"
  if is_truthy_env "$PERCEPTION_OBS_EXTERNAL"; then
    echo "[INFO] perception_obs_external=1; MuJoCo will not publish perception_obs. Start an external publisher/relay on port=${PERCEPTION_OBS_PORT} or shm=${PERCEPTION_OBS_SHM_NAME}. Every frame must carry the target run_sim sim-state episode_generation (an unrelated renderer-local counter is rejected)."
  fi
fi
if is_truthy_env "$GT_MUJOCO_PHYSICS"; then
  echo "[INFO] GT MuJoCo physics: object_mass=${MUJOCO_OBJECT_MASS_OVERRIDE} object_friction=${MUJOCO_OBJECT_GEOM_FRICTION} object_terrain_pair_friction=${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-<none>} copy_joint_defaults=${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML} copy_tendons=${SIM_COPY_TENDONS_FROM_ROBOT_XML} copy_collision_geoms=${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML} copy_contact_pairs=${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML} zero_passive_dynamics=${HOLOSOMA_GT_MUJOCO_ZERO_PASSIVE_DYNAMICS:-0} web_demo_object_contacts=${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-0}"
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

DIRECT_PERCEPTION_PRODUCER=0
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_OBS_EXTERNAL_ENABLED" != "1" ]]; then
  DIRECT_PERCEPTION_PRODUCER=1
fi
if [[ "$DIRECT_PERCEPTION_PRODUCER" == "1" && -z "$PERCEPTION_CONTRACT_ENVELOPE_B64" ]]; then
  echo "[ERROR] Direct perception producer requires an authenticated checkpoint contract envelope." >&2
  exit 1
fi

RUN_SIM_CMD=()
if [[ -n "$MUJOCO_PYTHONPATH" ]]; then
  RUN_SIM_CMD+=(env "PYTHONPATH=${MUJOCO_PYTHONPATH}")
fi
RUN_SIM_CMD+=(
  "$MUJOCO_PY" -u "$ROOT_DIR/src/holosoma/holosoma/run_sim.py"
  simulator:mujoco
  robot:g1_29dof_w_object
  terrain:terrain_locomotion_plane
)
if [[ "$DIRECT_PERCEPTION_PRODUCER" == "1" ]]; then
  RUN_SIM_CMD+=("perception:${PERCEPTION_PRESET}")
fi
RUN_SIM_CMD+=(
  --training.headless "$TRAINING_HEADLESS"
  --simulator.config.debug-viz "$SIM_DEBUG_VIZ"
  --simulator.config.sim.fps "$SIM_FPS"
  --simulator.config.sim.control-decimation "$SIM_CONTROL_DECIMATION"
  --simulator.config.virtual-gantry.enabled "$SIM_VIRTUAL_GANTRY_ENABLED"
  --robot.object.enabled=True
  --robot.object.object-urdf-path "$OBJECT_URDF"
  --simulator.config.bridge.interface "$INTERFACE_NAME"
  --simulator.config.bridge.clock-port "$SIM_CLOCK_PORT"
  --simulator.config.bridge.publish-sim-state=True
  --simulator.config.bridge.listen-control=True
  --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT"
  --simulator.config.bridge.control-port "$SIM_CONTROL_PORT"
  --motion-init.enabled=True
  --motion-init.motion-file "$MOTION_FILE"
  --motion-init.mode "$SIM_MOTION_INIT_MODE"
  --motion-init.object-name object
)

append_run_sim_value() {
  local option="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    RUN_SIM_CMD+=("$option" "$value")
  fi
}

if [[ "$MUJOCO_SHOW_OBJECT_COLLISION" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.mujoco-show-object-collision True)
fi
if [[ "$MUJOCO_HIDE_OBJECT_VISUALS_WHEN_SHOWING_COLLISION" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.mujoco-hide-object-visuals-when-showing-collision True)
fi
append_run_sim_value --simulator.config.sim.substeps "$SIM_SUBSTEPS"
append_run_sim_value --simulator.config.sim.physx.num-position-iterations "${SIM_PHYSX_POSITION_ITERATIONS:-}"
append_run_sim_value --simulator.config.sim.physx.num-velocity-iterations "${SIM_PHYSX_VELOCITY_ITERATIONS:-}"
append_run_sim_value --simulator.config.sim.physx.bounce-threshold-velocity "${SIM_PHYSX_BOUNCE_THRESHOLD_VELOCITY:-}"
append_run_sim_value --simulator.config.mujoco-backend "$MUJOCO_BACKEND"
append_run_sim_value --device "$SIM_DEVICE"
append_run_sim_value --robot.init-state.pos "$ROBOT_INIT_STATE_POS"
append_run_sim_value --robot.init-state.rot "$ROBOT_INIT_STATE_ROT"
append_run_sim_value --robot.asset.enable-self-collisions "$ROBOT_ENABLE_SELF_COLLISIONS"
if [[ "$SIM_USE_TRAINING_URDF_OBJECT_SCENE" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-use-training-urdf-scene True)
fi
if [[ "$SIM_ADD_DEFAULT_OBJECT_ACTUATORS" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-add-default-actuators True)
fi
if [[ "$SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-copy-joint-defaults-from-robot-xml True)
fi
if [[ "$SIM_COPY_TENDONS_FROM_ROBOT_XML" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-copy-tendons-from-robot-xml True)
fi
if [[ "$SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-copy-collision-geoms-from-robot-xml True)
fi
if [[ "$SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-copy-contact-pairs-from-robot-xml True)
fi
append_run_sim_value --robot.object.mujoco-object-mass-scale "$MUJOCO_OBJECT_MASS_SCALE"
append_run_sim_value --robot.object.mujoco-object-mass-override "$MUJOCO_OBJECT_MASS_OVERRIDE"
append_run_sim_value --robot.object.mujoco-object-geom-friction "$MUJOCO_OBJECT_GEOM_FRICTION"
append_run_sim_value --robot.object.mujoco-object-terrain-pair-friction "$MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION"
append_run_sim_value --robot.object.mujoco-object-lateral-friction "$MUJOCO_OBJECT_LATERAL_FRICTION"
append_run_sim_value --robot.object.mujoco-object-rolling-friction "$MUJOCO_OBJECT_ROLLING_FRICTION"
append_run_sim_value --robot.object.mujoco-object-contact-stiffness "$MUJOCO_OBJECT_CONTACT_STIFFNESS"
append_run_sim_value --robot.object.mujoco-object-contact-damping "$MUJOCO_OBJECT_CONTACT_DAMPING"
if [[ "$MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES" == "1" ]]; then
  RUN_SIM_CMD+=(--robot.object.mujoco-limit-object-contacts-to-carry-bodies True)
fi
append_run_sim_value --robot.object.mujoco-object-contact-body-name-markers "$MUJOCO_OBJECT_CONTACT_BODY_MARKERS"
append_run_sim_value --terrain.terrain-term.static-friction "$TERRAIN_STATIC_FRICTION"
append_run_sim_value --terrain.terrain-term.dynamic-friction "$TERRAIN_DYNAMIC_FRICTION"
if [[ "$DIRECT_PERCEPTION_PRODUCER" == "1" ]]; then
  RUN_SIM_CMD+=(
    --simulator.config.bridge.publish-perception-obs True
    --simulator.config.bridge.perception-obs-port "$PERCEPTION_OBS_PORT"
    --perception-randomization.enabled "$PERCEPTION_RANDOMIZATION_ENABLED"
    --perception-producer-tick-dt "$PERCEPTION_PRODUCER_TICK_DT"
    --perception-allow-mujoco-noise "$PERCEPTION_ALLOW_MUJOCO_NOISE"
    --perception-contract-envelope-b64 "$PERCEPTION_CONTRACT_ENVELOPE_B64"
    --training.seed "$PERCEPTION_PRODUCER_SEED"
  )
  append_run_sim_value --perception-randomization.translation-range "$PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE"
  append_run_sim_value --perception-randomization.rotation-range-deg "$PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG"
  append_run_sim_value --perception-randomization.noise-std-mult-range "$PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE"
  append_run_sim_value --perception-randomization.noise-drop-prob-range "$PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE"
  if [[ "$PUBLISH_PERCEPTION_OBS_SHM" == "1" ]]; then
    RUN_SIM_CMD+=(
      --simulator.config.bridge.publish-perception-obs-shm True
      --simulator.config.bridge.perception-obs-shm-name "$PERCEPTION_OBS_SHM_NAME"
    )
  fi
  append_run_sim_value --perception.camera-source "$PERCEPTION_CAMERA_SOURCE"
  append_run_sim_value --perception.object-geometry-mode "$PERCEPTION_OBJECT_GEOMETRY_MODE"
  append_run_sim_value --perception.camera-width "$PERCEPTION_CAMERA_WIDTH"
  append_run_sim_value --perception.camera-height "$PERCEPTION_CAMERA_HEIGHT"
  append_run_sim_value --perception.camera-warp-crop-top "$PERCEPTION_CAMERA_WARP_CROP_TOP"
  append_run_sim_value --perception.camera-warp-crop-bottom "$PERCEPTION_CAMERA_WARP_CROP_BOTTOM"
  append_run_sim_value --perception.camera-warp-crop-left "$PERCEPTION_CAMERA_WARP_CROP_LEFT"
  append_run_sim_value --perception.camera-warp-crop-right "$PERCEPTION_CAMERA_WARP_CROP_RIGHT"
  append_run_sim_value --perception.camera-pitch-deg "$PERCEPTION_CAMERA_PITCH_DEG"
  append_run_sim_value --perception.camera-vfov-deg "$PERCEPTION_CAMERA_VFOV_DEG"
  append_run_sim_value --perception.camera-hfov-deg "$PERCEPTION_CAMERA_HFOV_DEG"
  append_run_sim_value --perception.camera-include-robot-mesh "$PERCEPTION_CAMERA_INCLUDE_ROBOT_MESH"
  append_run_sim_value --perception.camera-near "$PERCEPTION_CAMERA_NEAR"
  append_run_sim_value --perception.camera-far "$PERCEPTION_CAMERA_FAR"
  append_run_sim_value --perception.max-distance "$PERCEPTION_MAX_DISTANCE"
  append_run_sim_value --perception.camera-warp-min-valid-depth "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH"
  append_run_sim_value --perception.camera-warp-normalize "$PERCEPTION_CAMERA_WARP_NORMALIZE"
  append_run_sim_value --perception.update-hz "$PERCEPTION_UPDATE_HZ"
  append_run_sim_value --perception.camera-fps "$PERCEPTION_CAMERA_FPS"
  append_run_sim_value --perception.camera-warp-buffer-len "$PERCEPTION_CAMERA_WARP_BUFFER_LEN"
  append_run_sim_value --perception.camera-warp-latency-frame "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME"
  append_run_sim_value --perception.camera-warp-edge-noise "$PERCEPTION_CAMERA_WARP_EDGE_NOISE"
  append_run_sim_value --perception.camera-warp-edge-border "$PERCEPTION_CAMERA_WARP_EDGE_BORDER"
  append_run_sim_value --perception.camera-warp-edge-shuffle-prob "$PERCEPTION_CAMERA_WARP_EDGE_SHUFFLE_PROB"
  append_run_sim_value --perception.camera-warp-edge-empty-prob "$PERCEPTION_CAMERA_WARP_EDGE_EMPTY_PROB"
  append_run_sim_value --perception.camera-warp-edge-thresh-primary "$PERCEPTION_CAMERA_WARP_EDGE_THRESH_PRIMARY"
  append_run_sim_value --perception.camera-warp-edge-thresh-secondary "$PERCEPTION_CAMERA_WARP_EDGE_THRESH_SECONDARY"
  append_run_sim_value --perception.camera-warp-edge-far-depth-thresh "$PERCEPTION_CAMERA_WARP_EDGE_FAR_DEPTH_THRESH"
  append_run_sim_value --perception.camera-warp-enable-holes "$PERCEPTION_CAMERA_WARP_ENABLE_HOLES"
  append_run_sim_value --perception.camera-warp-hole-prob "$PERCEPTION_CAMERA_WARP_HOLE_PROB"
  append_run_sim_value --perception.camera-warp-hole-reference-batch-size "$PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE"
  append_run_sim_value --perception.camera-apply-sensor-noise "$PERCEPTION_CAMERA_APPLY_SENSOR_NOISE"
fi
if [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.use-zmq-lowcmd True)
fi
if [[ "$SIM_IGNORE_DEFAULT_IDLE_COMMAND" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.ignore-default-idle-command True)
fi
if [[ "$SIM_LOG_FIRST_COMMAND_SUMMARY" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.log-first-command-summary True)
fi
if [[ "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.hold-default-pose-until-first-command True)
fi
if [[ "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.hold-initial-pose-until-first-command True)
fi
if [[ "$SIM_FREEZE_UNTIL_FIRST_COMMAND" == "1" ]]; then
  RUN_SIM_CMD+=(--simulator.config.bridge.freeze-until-first-command True)
fi

if is_truthy_env "${DRY_RUN:-0}"; then
  if [[ "$MJ_TRACK_MODE" != "policy" ]]; then
    printf '[DRY_RUN] run_sim:'
    printf ' %q' "${MUJOCO_LAUNCH_PREFIX[@]}" "${RUN_SIM_CMD[@]}"
    printf '\n'
  fi
  echo "[INFO] DRY_RUN=1; not launching MuJoCo or policy."
  exit 0
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
  "${MUJOCO_LAUNCH_PREFIX[@]}" "${RUN_SIM_CMD[@]}" >"$SIM_LOG" 2>&1 &
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
  --task.model-path "$POLICY_MODEL"
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

echo "Policy model: $POLICY_MODEL"
echo "MuJoCo log:   $SIM_LOG"
echo "Policy log:   $POLICY_LOG"
