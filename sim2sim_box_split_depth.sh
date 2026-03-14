#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTION_FILE="${1:?usage: sim2sim_box_split_depth.sh <motion.npz> <checkpoint.pt|model.onnx>}"
MODEL_INPUT="${2:?usage: sim2sim_box_split_depth.sh <motion.npz> <checkpoint.pt|model.onnx>}"

MUJOCO_PY="${MUJOCO_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
INFER_PY="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
MUJOCO_CPUSET="${MUJOCO_CPUSET:-0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
SIM_FPS="${SIM_FPS:-200}"
SIM_CONTROL_DECIMATION="${SIM_CONTROL_DECIMATION:-4}"
SIM_VIRTUAL_GANTRY_ENABLED="${SIM_VIRTUAL_GANTRY_ENABLED:-False}"
SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-training_default_pose}"
APPLY_TRAINING_MOTION_TRANSITIONS="${APPLY_TRAINING_MOTION_TRANSITIONS:-1}"
SIM_IGNORE_DEFAULT_IDLE_COMMAND="${SIM_IGNORE_DEFAULT_IDLE_COMMAND:-1}"
SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND="${SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND:-1}"
PERCEPTION_CAMERA_WIDTH="${PERCEPTION_CAMERA_WIDTH:-17}"
PERCEPTION_CAMERA_HEIGHT="${PERCEPTION_CAMERA_HEIGHT:-17}"
PERCEPTION_CAMERA_NEAR="${PERCEPTION_CAMERA_NEAR:-0.001}"
PERCEPTION_CAMERA_FAR="${PERCEPTION_CAMERA_FAR:-3.0}"
PERCEPTION_MAX_DISTANCE="${PERCEPTION_MAX_DISTANCE:-3.0}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
SIM_PERCEPTION_PORT="${SIM_PERCEPTION_PORT:-5558}"
INTERFACE_NAME="${INTERFACE_NAME:-lo}"
RUN_SECONDS="${RUN_SECONDS:-20}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-45}"
SIM_READY_PATTERN="${SIM_READY_PATTERN:-Starting direct simulation loop...}"
SIM_STARTUP_WAIT="${SIM_STARTUP_WAIT:-0}"
OBJECT_URDF="${OBJECT_URDF:-$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf}"
PATCH_DIR="${PATCH_DIR:-$ROOT_DIR/logs/sim2sim_exports}"
POLICY_ACTION_SCALE="${POLICY_ACTION_SCALE:-}"

mkdir -p "$PATCH_DIR"

MOTION_STEM="$(basename "${MOTION_FILE%.*}")"
MODEL_STEM="$(basename "${MODEL_INPUT%.*}")"
PATCHED_ONNX="$PATCH_DIR/${MODEL_STEM}__${MOTION_STEM}.onnx"
RUN_DIR="$ROOT_DIR/logs/sim2sim_runs/${MOTION_STEM}__depth"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py" \
  --model-path "$MODEL_INPUT" \
  --motion-file "$MOTION_FILE" \
  $( [[ "$APPLY_TRAINING_MOTION_TRANSITIONS" == "1" ]] && printf '%s' "--apply-training-motion-transitions" ) \
  --output-path "$PATCHED_ONNX"

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
  --simulator.config.virtual-gantry.enabled "$SIM_VIRTUAL_GANTRY_ENABLED" \
  --perception.camera-width "$PERCEPTION_CAMERA_WIDTH" \
  --perception.camera-height "$PERCEPTION_CAMERA_HEIGHT" \
  --perception.camera-near "$PERCEPTION_CAMERA_NEAR" \
  --perception.camera-far "$PERCEPTION_CAMERA_FAR" \
  --perception.max-distance "$PERCEPTION_MAX_DISTANCE" \
  --robot.object.enabled=True \
  --robot.object.object-urdf-path "$OBJECT_URDF" \
  --robot.object.mujoco-add-default-actuators=True \
  --simulator.config.bridge.interface "$INTERFACE_NAME" \
  --simulator.config.bridge.publish-sim-state=True \
  --simulator.config.bridge.sim-state-port "$SIM_STATE_PORT" \
  --simulator.config.bridge.publish-perception-obs=True \
  --simulator.config.bridge.perception-obs-port "$SIM_PERCEPTION_PORT" \
  $( [[ "$SIM_IGNORE_DEFAULT_IDLE_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.ignore-default-idle-command" "True" ) \
  $( [[ "$SIM_HOLD_DEFAULT_POSE_UNTIL_FIRST_COMMAND" == "1" ]] && printf '%s %s' "--simulator.config.bridge.hold-default-pose-until-first-command" "True" ) \
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
  --task.sim-state-port "$SIM_STATE_PORT" \
  --task.use-sim-perception \
  --task.sim-perception-port "$SIM_PERCEPTION_PORT" \
  --task.use-sim-time \
  --task.auto-start-motion \
  --task.apply-training-motion-transitions \
  --task.policy-action-scale "$POLICY_ACTION_SCALE" \
  --task.sim-object-name object \
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
