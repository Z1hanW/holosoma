#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_FILE:-$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-4}"
SIM_SUBSTEPS="${SIM_SUBSTEPS:-}"
SIM_DEVICE="${SIM_DEVICE:-}"
MUJOCO_BACKEND="${MUJOCO_BACKEND:-}"
TERRAIN_STATIC_FRICTION="${TERRAIN_STATIC_FRICTION:-}"
TERRAIN_DYNAMIC_FRICTION="${TERRAIN_DYNAMIC_FRICTION:-}"
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
ENABLE_SPLIT_PERCEPTION_OBS="${ENABLE_SPLIT_PERCEPTION_OBS:-auto}"
ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND="${ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND:-0}"
PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
PERCEPTION_CAMERA_SOURCE="${PERCEPTION_CAMERA_SOURCE:-far_tracking_warp}"
PERCEPTION_OBJECT_GEOMETRY_MODE="${PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
PERCEPTION_CAMERA_PITCH_DEG="${PERCEPTION_CAMERA_PITCH_DEG:-}"
PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-}"
PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-}"
PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-}"
PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH="${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-}"
PERCEPTION_UPDATE_HZ="${PERCEPTION_UPDATE_HZ:-}"
PERCEPTION_CAMERA_FPS="${PERCEPTION_CAMERA_FPS:-}"
PERCEPTION_CAMERA_WARP_EDGE_NOISE="${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-}"
PERCEPTION_CAMERA_WARP_BUFFER_LEN="${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-}"
PERCEPTION_CAMERA_WARP_LATENCY_FRAME="${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-}"
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
SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY:-0}"
HOLOSOMA_ONNX_ALIGN_MAX_STEPS="${HOLOSOMA_ONNX_ALIGN_MAX_STEPS:-0}"
HOLOSOMA_ONNX_ALIGN_POSE_TOL="${HOLOSOMA_ONNX_ALIGN_POSE_TOL:-5e-3}"
HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX="${HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX:-1}"
HOLOSOMA_CLIP_JOINT_TARGETS="${HOLOSOMA_CLIP_JOINT_TARGETS:-0}"
AUTO_START_STIFF_HOLD_SEC_RAW="${AUTO_START_STIFF_HOLD_SEC-__unset__}"
AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-}"
AUTO_START_STIFF_MAX_WAIT_SEC_RAW="${AUTO_START_STIFF_MAX_WAIT_SEC-__unset__}"
AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-}"
AUTO_START_STIFF_POSE_TOL="${AUTO_START_STIFF_POSE_TOL:-0.12}"
USE_ROOT_REFERENCE_AT_CLIP_START_RAW="${USE_ROOT_REFERENCE_AT_CLIP_START-__unset__}"
USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-}"
SIM_ADD_DEFAULT_OBJECT_ACTUATORS_RAW="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS-__unset__}"
SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
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
RUN_DIR="$ROOT_DIR/logs/sim2sim_runs/${MOTION_STEM}__tracking"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export HOLOSOMA_ONNX_ALIGN_MAX_STEPS
export HOLOSOMA_ONNX_ALIGN_POSE_TOL
export HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX
export HOLOSOMA_CLIP_JOINT_TARGETS

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
import importlib.util
import sys

for module_name in sys.argv[1:]:
    if importlib.util.find_spec(module_name) is None:
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

MUJOCO_PY="$(resolve_python_with_modules "mujoco holosoma torch tyro" \
  "$(resolve_python "$MUJOCO_PY")" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
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
      SIM_CONTROL_DECIMATION) SIM_CONTROL_DECIMATION="$value" ;;
      SIM_SUBSTEPS) SIM_SUBSTEPS="$value" ;;
      MUJOCO_BACKEND) MUJOCO_BACKEND="$value" ;;
      TERRAIN_STATIC_FRICTION) TERRAIN_STATIC_FRICTION="$value" ;;
      TERRAIN_DYNAMIC_FRICTION) TERRAIN_DYNAMIC_FRICTION="$value" ;;
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
    "camera_warp_edge_noise": "PERCEPTION_CAMERA_WARP_EDGE_NOISE",
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
      PERCEPTION_CAMERA_WARP_EDGE_NOISE)
        if [[ -z "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" ]]; then
          PERCEPTION_CAMERA_WARP_EDGE_NOISE="$value"
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

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_INPUT" \
  --motion-file "$MOTION_FILE" \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--apply-training-motion-transitions" ) \
  --output-path "$PATCHED_ONNX"

apply_training_sim_overrides "$PATCHED_ONNX"
apply_training_robot_init_overrides "$PATCHED_ONNX"
apply_training_robot_asset_overrides "$PATCHED_ONNX"
apply_training_object_overrides "$PATCHED_ONNX"
apply_training_perception_overrides "$PATCHED_ONNX"

SIM_ADD_DEFAULT_OBJECT_ACTUATORS="${SIM_ADD_DEFAULT_OBJECT_ACTUATORS:-1}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
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

MODEL_EXPECTS_PERCEPTION_OBS="$(onnx_has_input "$PATCHED_ONNX" "perception_obs")"
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "auto" ]]; then
  ENABLE_SPLIT_PERCEPTION_OBS="$MODEL_EXPECTS_PERCEPTION_OBS"
fi
if [[ "$MODEL_EXPECTS_PERCEPTION_OBS" == "1" && "$ENABLE_SPLIT_PERCEPTION_OBS" != "1" ]]; then
  echo "Model expects perception_obs but ENABLE_SPLIT_PERCEPTION_OBS=${ENABLE_SPLIT_PERCEPTION_OBS}" >&2
  exit 1
fi
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && "$PERCEPTION_CAMERA_SOURCE" == "far_tracking_warp" && -z "$SIM_DEVICE" && -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]]; then
  SIM_DEVICE="cuda:0"
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
  export HOLOSOMA_MUJOCO_DEPTH_PREFER_VISUAL_MESHES="${HOLOSOMA_MUJOCO_DEPTH_PREFER_VISUAL_MESHES:-1}"
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

SIM_LOG="$RUN_DIR/mujoco.log"
POLICY_LOG="$RUN_DIR/policy.log"

: >"$SIM_LOG"
: >"$POLICY_LOG"

if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]]; then
  echo "[INFO] perception camera: source=${PERCEPTION_CAMERA_SOURCE} update_hz=${PERCEPTION_UPDATE_HZ:-<default>} camera_fps=${PERCEPTION_CAMERA_FPS:-<default>} pitch_deg=${PERCEPTION_CAMERA_PITCH_DEG:-<default>} near=${PERCEPTION_CAMERA_NEAR:-<default>} far=${PERCEPTION_CAMERA_FAR:-<default>} max_distance=${PERCEPTION_MAX_DISTANCE:-<default>} warp_min_valid_depth=${PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH:-<default>} warp_buffer_len=${PERCEPTION_CAMERA_WARP_BUFFER_LEN:-<default>} warp_latency_frame=${PERCEPTION_CAMERA_WARP_LATENCY_FRAME:-<default>} warp_edge_noise=${PERCEPTION_CAMERA_WARP_EDGE_NOISE:-<default>}"
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
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]] && printf '%s' "perception:${PERCEPTION_PRESET}" ) \
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
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]] && printf '%s %s' "--simulator.config.bridge.publish-perception-obs" "True" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]] && printf '%s %s' "--simulator.config.bridge.perception-obs-port" "$PERCEPTION_OBS_PORT" ) \
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
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_PITCH_DEG" ]] && printf '%s %s' "--perception.camera-pitch-deg" "$PERCEPTION_CAMERA_PITCH_DEG" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_NEAR" ]] && printf '%s %s' "--perception.camera-near" "$PERCEPTION_CAMERA_NEAR" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_FAR" ]] && printf '%s %s' "--perception.camera-far" "$PERCEPTION_CAMERA_FAR" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_MAX_DISTANCE" ]] && printf '%s %s' "--perception.max-distance" "$PERCEPTION_MAX_DISTANCE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH" ]] && printf '%s %s' "--perception.camera-warp-min-valid-depth" "$PERCEPTION_CAMERA_WARP_MIN_VALID_DEPTH" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_UPDATE_HZ" ]] && printf '%s %s' "--perception.update-hz" "$PERCEPTION_UPDATE_HZ" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_FPS" ]] && printf '%s %s' "--perception.camera-fps" "$PERCEPTION_CAMERA_FPS" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" ]] && printf '%s %s' "--perception.camera-warp-edge-noise" "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_BUFFER_LEN" ]] && printf '%s %s' "--perception.camera-warp-buffer-len" "$PERCEPTION_CAMERA_WARP_BUFFER_LEN" ) \
    $( [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" && -n "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME" ]] && printf '%s %s' "--perception.camera-warp-latency-frame" "$PERCEPTION_CAMERA_WARP_LATENCY_FRAME" ) \
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
  --task.auto-start-motion-clip
  --task.auto-start-stiff-hold-sec "$AUTO_START_STIFF_HOLD_SEC"
  --task.auto-start-stiff-max-wait-sec "$AUTO_START_STIFF_MAX_WAIT_SEC"
  --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL"
  --task.policy-action-scale "$POLICY_ACTION_SCALE"
  --task.rl-rate "$POLICY_RL_RATE"
  --task.sim-object-name object
)
if [[ "$SIM_USE_ZMQ_LOWCMD" == "1" ]]; then
  POLICY_CMD+=(--task.use-zmq-lowcmd)
fi
if [[ "$ENABLE_SPLIT_PERCEPTION_OBS" == "1" ]]; then
  POLICY_CMD+=(--task.use-split-perception-obs --task.perception-obs-port "$PERCEPTION_OBS_PORT")
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
