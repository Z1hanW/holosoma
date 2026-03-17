#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_FILE="${1:?usage: sim2sim_box_split_depth.sh <motion.npz> <checkpoint.pt|model.onnx>}"
MODEL_INPUT="${2:?usage: sim2sim_box_split_depth.sh <motion.npz> <checkpoint.pt|model.onnx>}"

MUJOCO_PY="${MUJOCO_PY:-}"
INFER_PY="${INFER_PY:-}"
MUJOCO_CPUSET="${MUJOCO_CPUSET:-0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
SIM_FPS="${SIM_FPS:-200}"
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-4}"
SIM_SUBSTEPS="${SIM_SUBSTEPS:-}"
MUJOCO_BACKEND="${MUJOCO_BACKEND:-}"
TERRAIN_STATIC_FRICTION="${TERRAIN_STATIC_FRICTION:-}"
TERRAIN_DYNAMIC_FRICTION="${TERRAIN_DYNAMIC_FRICTION:-}"
SIM_VIRTUAL_GANTRY_ENABLED="${SIM_VIRTUAL_GANTRY_ENABLED:-False}"
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-raw_motion}"
APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS:-0}"
USE_TRAINING_SIM_CONFIG="${USE_TRAINING_SIM_CONFIG:-1}"
SIM_IGNORE_DEFAULT_IDLE_COMMAND="${SIM_IGNORE_DEFAULT_IDLE_COMMAND:-1}"
SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND:-0}"
SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND:-0}"
SIM_FREEZE_UNTIL_FIRST_COMMAND="${SIM_FREEZE_UNTIL_FIRST_COMMAND:-1}"
SIM_LOG_FIRST_COMMAND_SUMMARY="${SIM_LOG_FIRST_COMMAND_SUMMARY:-0}"
PERCEPTION_CAMERA_WIDTH="${PERCEPTION_CAMERA_WIDTH:-17}"
PERCEPTION_CAMERA_HEIGHT="${PERCEPTION_CAMERA_HEIGHT:-17}"
PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-0.001}"
PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-3.0}"
PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-3.0}"
SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5555}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
SIM_PERCEPTION_PORT="${SIM_PERCEPTION_PORT:-5558}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5559}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
RUN_SECONDS="${RUN_SECONDS:-20}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-45}"
SIM_READY_PATTERN="${SIM_READY_PATTERN:-Starting direct simulation loop...}"
SIM_STARTUP_WAIT="${SIM_STARTUP_WAIT:-0}"
DEFAULT_OBJECT_URDF="$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
OBJECT_URDF="${OBJECT_URDF:-}"
PATCH_DIR="${PATCH_DIR:-$ROOT_DIR/logs/sim2sim_exports}"
POLICY_ACTION_SCALE="${POLICY_ACTION_SCALE:-}"
POLICY_ACTION_SCALE_CAP="${POLICY_ACTION_SCALE_CAP:-0.05}"
AUTO_START_STIFF_HOLD_SEC="${AUTO_START_STIFF_HOLD_SEC:-0.0}"
AUTO_START_STIFF_MAX_WAIT_SEC="${AUTO_START_STIFF_MAX_WAIT_SEC:-0.0}"
AUTO_START_STIFF_POSE_TOL="${AUTO_START_STIFF_POSE_TOL:-0.12}"
POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-0}"
HOLOSOMA_ONNX_ALIGN_MAX_STEPS="${HOLOSOMA_ONNX_ALIGN_MAX_STEPS:-0}"
HOLOSOMA_ONNX_ALIGN_POSE_TOL="${HOLOSOMA_ONNX_ALIGN_POSE_TOL:-5e-3}"
HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX="${HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX:-1}"
HOLOSOMA_CLIP_JOINT_TARGETS="${HOLOSOMA_CLIP_JOINT_TARGETS:-0}"
DEFAULT_HOLOSOMA_KP_LEVEL="${DEFAULT_HOLOSOMA_KP_LEVEL:-0.2}"
DEFAULT_HOLOSOMA_KD_LEVEL="${DEFAULT_HOLOSOMA_KD_LEVEL:-0.2}"
USE_ROOT_REFERENCE_AT_CLIP_START="${USE_ROOT_REFERENCE_AT_CLIP_START:-1}"
SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
MUJOCO_OBJECT_MASS_SCALE="${MUJOCO_OBJECT_MASS_SCALE:-}"
MUJOCO_OBJECT_MASS_OVERRIDE="${MUJOCO_OBJECT_MASS_OVERRIDE:-2.0}"
MUJOCO_OBJECT_GEOM_FRICTION="${MUJOCO_OBJECT_GEOM_FRICTION:-[0.4,0.005,0.001]}"
MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION="${MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION:-[0.4,0.005,0.001]}"
PREFER_SIM_REF_FROM_SIM_STATE="${PREFER_SIM_REF_FROM_SIM_STATE:-1}"
MOTION_METADATA_TOOL="$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/read_motion_clip_metadata.py"

mkdir -p "$PATCH_DIR"

MOTION_STEM="$(basename "${MOTION_FILE%.*}")"
MODEL_STEM="$(basename "${MODEL_INPUT%.*}")"
PATCHED_ONNX="$PATCH_DIR/${MODEL_STEM}__${MOTION_STEM}.onnx"
RUN_DIR="$ROOT_DIR/logs/sim2sim_runs/${MOTION_STEM}__depth"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export HOLOSOMA_ONNX_ALIGN_MAX_STEPS
export HOLOSOMA_ONNX_ALIGN_POSE_TOL
export HOLOSOMA_ONNX_OFFSET_APPLIES_TO_MOTION_INDEX
export HOLOSOMA_CLIP_JOINT_TARGETS

if [[ -z "${HOLOSOMA_KP_LEVEL:-}" ]]; then
  export HOLOSOMA_KP_LEVEL="$DEFAULT_HOLOSOMA_KP_LEVEL"
fi
if [[ -z "${HOLOSOMA_KD_LEVEL:-}" ]]; then
  export HOLOSOMA_KD_LEVEL="$DEFAULT_HOLOSOMA_KD_LEVEL"
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
  "$python_bin" - <<'PY' "$@" >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if all(importlib.util.find_spec(mod) is not None for mod in sys.argv[1:]) else 1)
PY
}

resolve_python_with_modules() {
  local configured="$1"
  shift
  local -a required_modules=()
  while (( $# > 0 )); do
    if [[ "$1" == "--" ]]; then
      shift
      break
    fi
    required_modules+=("$1")
    shift
  done

  local candidate
  if [[ -n "$configured" ]]; then
    if [[ ! -x "$configured" ]]; then
      echo "Configured python is not executable: $configured" >&2
      exit 1
    fi
    if python_has_modules "$configured" "${required_modules[@]}"; then
      printf '%s\n' "$configured"
      return
    fi
    echo "Configured python is missing required modules (${required_modules[*]}): $configured" >&2
    exit 1
  fi

  for candidate in "$@"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if python_has_modules "$candidate" "${required_modules[@]}"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  local fallback
  for fallback in python python3; do
    if command -v "$fallback" >/dev/null 2>&1; then
      candidate="$(command -v "$fallback")"
      if python_has_modules "$candidate" "${required_modules[@]}"; then
        printf '%s\n' "$candidate"
        return
      fi
    fi
  done

  echo "No usable python interpreter with required modules (${required_modules[*]}) found for split sim2sim launcher" >&2
  exit 1
}

MUJOCO_PY="$(resolve_python_with_modules "$MUJOCO_PY" mujoco -- \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"
INFER_PY="$(resolve_python_with_modules "$INFER_PY" onnx onnxruntime -- \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python)"

resolve_sim_device() {
  local requested="$1"
  if [[ "$requested" != cuda* ]]; then
    printf '%s\n' "$requested"
    return
  fi

  local has_cuda
  has_cuda="$(
    "$MUJOCO_PY" - <<'PY' 2>/dev/null || true
try:
    import warp as wp
    wp.init()
    has_cuda = any(getattr(device, "is_cuda", False) for device in wp.get_devices())
    print("1" if has_cuda else "0")
except Exception:
    print("0")
PY
  )"

  if [[ "$has_cuda" == "1" ]]; then
    printf '%s\n' "$requested"
    return
  fi

  echo "Warning: warp CUDA backend is unavailable in $MUJOCO_PY; falling back to SIM_DEVICE=cpu" >&2
  printf '%s\n' "cpu"
}

SIM_DEVICE="$(resolve_sim_device "$SIM_DEVICE")"

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

apply_motion_clip_object_defaults

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
      SIM_FPS) SIM_FPS="$value" ;;
      SIM_CONTROL_DECIMATION) SIM_CONTROL_DECIMATION="$value" ;;
      SIM_SUBSTEPS) SIM_SUBSTEPS="$value" ;;
      MUJOCO_BACKEND) MUJOCO_BACKEND="$value" ;;
      TERRAIN_STATIC_FRICTION) TERRAIN_STATIC_FRICTION="$value" ;;
      TERRAIN_DYNAMIC_FRICTION) TERRAIN_DYNAMIC_FRICTION="$value" ;;
    esac
  done <<< "$override_lines"
}

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_INPUT" \
  --motion-file "$MOTION_FILE" \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--apply-training-motion-transitions" ) \
  --output-path "$PATCHED_ONNX"

apply_training_sim_overrides "$PATCHED_ONNX"

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

if [[ -n "$POLICY_ACTION_SCALE_CAP" ]]; then
  POLICY_ACTION_SCALE="$(
    "$INFER_PY" - <<'PY' "$POLICY_ACTION_SCALE" "$POLICY_ACTION_SCALE_CAP"
import sys

scale = float(sys.argv[1])
cap = float(sys.argv[2])
print(min(scale, cap))
PY
  )"
fi

SIM_LOG="$RUN_DIR/mujoco.log"
POLICY_LOG="$RUN_DIR/policy.log"

: >"$SIM_LOG"
: >"$POLICY_LOG"

cleanup() {
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
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

"${MUJOCO_LAUNCH_PREFIX[@]}" "$MUJOCO_PY" "$ROOT_DIR/src/holosoma/holosoma/run_sim.py" \
  simulator:mujoco \
  robot:g1_29dof_w_object \
  terrain:terrain_locomotion_plane \
  perception:camera_depth_d435i \
  --device "$SIM_DEVICE" \
  --training.headless=True \
  --simulator.config.sim.fps "$SIM_FPS" \
  --simulator.config.sim.control-decimation "$SIM_CONTROL_DECIMATION" \
  $( [[ -n "$SIM_SUBSTEPS" ]] && printf '%s %s' "--simulator.config.sim.substeps" "$SIM_SUBSTEPS" ) \
  $( [[ -n "$MUJOCO_BACKEND" ]] && printf '%s %s' "--simulator.config.mujoco-backend" "$MUJOCO_BACKEND" ) \
  --simulator.config.virtual-gantry.enabled "$SIM_VIRTUAL_GANTRY_ENABLED" \
  --perception.camera-width "$PERCEPTION_CAMERA_WIDTH" \
  --perception.camera-height "$PERCEPTION_CAMERA_HEIGHT" \
  --perception.camera-near "$PERCEPTION_CAMERA_NEAR" \
  --perception.camera-far "$PERCEPTION_CAMERA_FAR" \
  --perception.max-distance "$PERCEPTION_MAX_DISTANCE" \
  --robot.object.enabled=True \
  --robot.object.object-urdf-path "$OBJECT_URDF" \
  --robot.object.mujoco-add-default-actuators=True \
  $( [[ "$SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-joint-defaults-from-robot-xml" "True" ) \
  $( [[ "$SIM_COPY_TENDONS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-tendons-from-robot-xml" "True" ) \
  $( [[ "$SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-collision-geoms-from-robot-xml" "True" ) \
  $( [[ "$SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML" == "1" ]] && printf '%s %s' "--robot.object.mujoco-copy-contact-pairs-from-robot-xml" "True" ) \
  $( [[ -n "$MUJOCO_OBJECT_MASS_SCALE" ]] && printf '%s %s' "--robot.object.mujoco-object-mass-scale" "$MUJOCO_OBJECT_MASS_SCALE" ) \
  $( [[ -n "$MUJOCO_OBJECT_MASS_OVERRIDE" ]] && printf '%s %s' "--robot.object.mujoco-object-mass-override" "$MUJOCO_OBJECT_MASS_OVERRIDE" ) \
  $( [[ -n "$MUJOCO_OBJECT_GEOM_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-geom-friction" "$MUJOCO_OBJECT_GEOM_FRICTION" ) \
  $( [[ -n "$MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION" ]] && printf '%s %s' "--robot.object.mujoco-object-terrain-pair-friction" "$MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION" ) \
  $( [[ -n "$TERRAIN_STATIC_FRICTION" ]] && printf '%s %s' "--terrain.terrain-term.static-friction" "$TERRAIN_STATIC_FRICTION" ) \
  $( [[ -n "$TERRAIN_DYNAMIC_FRICTION" ]] && printf '%s %s' "--terrain.terrain-term.dynamic-friction" "$TERRAIN_DYNAMIC_FRICTION" ) \
  --simulator.config.bridge.interface "$INTERFACE_NAME" \
  --simulator.config.bridge.clock-port "$SIM_CLOCK_PORT" \
  --simulator.config.bridge.publish-sim-state=True \
  --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT" \
  --simulator.config.bridge.publish-perception-obs=True \
  --simulator.config.bridge.perception-obs-port "$SIM_PERCEPTION_PORT" \
  --simulator.config.bridge.control-port "$SIM_CONTROL_PORT" \
  $( [[ "$SIM_IGNORE_DEFAULT_IDLE_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.ignore-default-idle-command" "True" ) \
  $( [[ "$SIM_LOG_FIRST_COMMAND_SUMMARY" == "1" ]] && printf '%s %s' "--simulator.config.bridge.log-first-command-summary" "True" ) \
  $( [[ "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.hold-default-pose-until-first-command" "True" ) \
  $( [[ "$SIM_HOLD_INITIAL_POSE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.hold-initial-pose-until-first-command" "True" ) \
  $( [[ "$SIM_FREEZE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.freeze-until-first-command" "True" ) \
  --motion-init.enabled=True \
  --motion-init.motion-file "$MOTION_FILE" \
  --motion-init.mode "$SIM_MOTION_INIT_MODE" \
  --motion-init.object-name object \
  >"$SIM_LOG" 2>&1 &
SIM_PID=$!

if ! wait_for_sim_ready; then
  exit 1
fi

if [[ "$SIM_STARTUP_WAIT" != "0" ]]; then
  sleep "$SIM_STARTUP_WAIT"
fi

set +e
timeout --signal=INT "${RUN_SECONDS}s" \
  "$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py" \
  inference:g1-29dof-wbt-object-distill-depth \
  --task.model-path "$PATCHED_ONNX" \
  --task.motion-file "$MOTION_FILE" \
  --task.interface "$INTERFACE_NAME" \
  --task.use-sim-state \
  --task.sim-clock-port "$SIM_CLOCK_PORT" \
  --task.sim-state-port "$SIM_STATE_PORT" \
  --task.sim-control-port "$SIM_CONTROL_PORT" \
  --task.use-sim-perception \
  --task.sim-perception-port "$SIM_PERCEPTION_PORT" \
  --task.use-sim-time \
  --task.no-auto-start-motion \
  --task.auto-start-motion-clip \
  --task.auto-start-stiff-hold-sec "$AUTO_START_STIFF_HOLD_SEC" \
  --task.auto-start-stiff-max-wait-sec "$AUTO_START_STIFF_MAX_WAIT_SEC" \
  --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL" \
  $( [[ "$USE_ROOT_REFERENCE_AT_CLIP_START" == "1" ]] && printf '%s' "--task.use-root-reference-at-clip-start" ) \
  $( [[ "$PREFER_SIM_REF_FROM_SIM_STATE" == "1" ]] && printf '%s' "--task.prefer-sim-ref-from-sim-state" ) \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--task.apply-training-motion-transitions" ) \
  --task.policy-action-scale "$POLICY_ACTION_SCALE" \
  --task.sim-object-name object \
  $( [[ "$POLICY_DEFER_UNTIL_VALID_STATE" == "1" ]] && printf '%s' "--task.defer-policy-start-until-valid-state" ) \
  --viser.no-auto-reset-on-motion-end \
  >"$POLICY_LOG" 2>&1
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 && "$STATUS" -ne 124 && "$STATUS" -ne 130 ]]; then
  echo "Policy run failed. See $POLICY_LOG" >&2
  exit "$STATUS"
fi

echo "Patched ONNX: $PATCHED_ONNX"
echo "MuJoCo log:   $SIM_LOG"
echo "Policy log:   $POLICY_LOG"
