#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_mujoco_forward_after_lift_rollout.sh \
  --motion-file MOTION.npz --object-urdf OBJECT.urdf \
  --model-onnx MODEL.onnx --output-dir DIR [--port-base 7255]
EOF
}

MOTION_FILE=""
OBJECT_URDF=""
MODEL_ONNX=""
OUTPUT_DIR=""
PORT_BASE=7255
FORWARD_COMMAND_M=0.15
LIFT_REL_Z_DELTA_M=0.30
LATEST_FORWARD_ACTOR_SIM_TIME_S=""
DEADLINE_PUBLISH_LEAD_MS=40
ACTOR_STEPS=501
STARTUP_TIMEOUT_S=180
ROLLOUT_TIMEOUT_S=90

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-file) MOTION_FILE="$2"; shift 2 ;;
    --object-urdf) OBJECT_URDF="$2"; shift 2 ;;
    --model-onnx) MODEL_ONNX="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --port-base) PORT_BASE="$2"; shift 2 ;;
    --forward-command-m) FORWARD_COMMAND_M="$2"; shift 2 ;;
    --lift-rel-z-delta-m) LIFT_REL_Z_DELTA_M="$2"; shift 2 ;;
    --latest-forward-actor-sim-time-s) LATEST_FORWARD_ACTOR_SIM_TIME_S="$2"; shift 2 ;;
    --deadline-publish-lead-ms) DEADLINE_PUBLISH_LEAD_MS="$2"; shift 2 ;;
    --actor-steps) ACTOR_STEPS="$2"; shift 2 ;;
    --startup-timeout-s) STARTUP_TIMEOUT_S="$2"; shift 2 ;;
    --rollout-timeout-s) ROLLOUT_TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in MOTION_FILE OBJECT_URDF MODEL_ONNX OUTPUT_DIR; do
  if [[ -z "${!required}" ]]; then
    echo "Missing required argument: $required" >&2
    usage >&2
    exit 2
  fi
done
for input_path in "$MOTION_FILE" "$OBJECT_URDF" "$MODEL_ONNX"; do
  if [[ ! -f "$input_path" ]]; then
    echo "Required input is not a file: $input_path" >&2
    exit 2
  fi
done
if ! [[ "$PORT_BASE" =~ ^[0-9]+$ ]] || (( PORT_BASE < 1024 || PORT_BASE > 65527 )); then
  echo "--port-base must be an integer from 1024 through 65527" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
MOTION_FILE="$(realpath "$MOTION_FILE")"
OBJECT_URDF="$(realpath "$OBJECT_URDF")"
MODEL_ONNX="$(realpath "$MODEL_ONNX")"

SIM_CLOCK_PORT=$PORT_BASE
SIM_STATE_PORT=$((PORT_BASE + 2))
PERCEPTION_OBS_PORT=$((PORT_BASE + 3))
SIM_CONTROL_PORT=$((PORT_BASE + 4))
SPARSE_ROOT_COMMAND_PORT=$((PORT_BASE + 6))
POLICY_CONTROL_PORT=$((PORT_BASE + 7))
POLICY_OVERLAY_PORT=$((PORT_BASE + 8))

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RUN_DIR="$OUTPUT_DIR/runtime"
export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
export MUJOCO_PY="${MUJOCO_PY:-/data/ubuntu/conda-envs/dexjoco/bin/python}"
export INFERENCE_CONFIG=g1-29dof-wbt-object-contact-aware-drop-button-depth-distill
export ENABLE_SPLIT_PERCEPTION_OBS=1
export ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND=1
export PERCEPTION_CAMERA_SOURCE=far_tracking_warp
export HOLOSOMA_EVAL_DISABLE_PERCEPTION_RESET_RANDOMIZATION=1
export SIM_CLOCK_PORT SIM_STATE_PORT PERCEPTION_OBS_PORT SIM_CONTROL_PORT
export SPARSE_ROOT_COMMAND_PORT POLICY_CONTROL_PORT POLICY_OVERLAY_PORT
export HOLOSOMA_POLICY_OVERLAY_PORT="$POLICY_OVERLAY_PORT"
export PERCEPTION_OBS_SHM_NAME="depth_img_shm_${SIM_STATE_PORT}"
export RUN_SECONDS=0 POLICY_STDIO=log POLICY_DEFER_UNTIL_VALID_STATE=1
export SIM_READY_TIMEOUT="$STARTUP_TIMEOUT_S"
export TRAINING_HEADLESS=True SIM_DEBUG_VIZ=False SIM_VIRTUAL_GANTRY_ENABLED=False
export SIM_MOTION_INIT_MODE=training_default_pose
export SIM_USE_TRAINING_URDF_OBJECT_SCENE=1
export SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML="${SIM_COPY_JOINT_DEFAULTS_FROM_ROBOT_XML:-1}"
export SIM_COPY_TENDONS_FROM_ROBOT_XML="${SIM_COPY_TENDONS_FROM_ROBOT_XML:-1}"
export SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML="${SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML:-1}"
export SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML="${SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML:-1}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS=0
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS=0
export HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_PAIRS=1
export HOLOSOMA_MUJOCO_KEEP_REFERENCE_HAND_COLLISION=0
export HOLOSOMA_MUJOCO_WRIST_ORIGIN_CONTACT_SPHERES=0
export HOLOSOMA_MUJOCO_PALM_CONTACT_SPHERES=0
export HOLOSOMA_POLICY_TARGET_OBJECT_STATE_ASSIST=0
export HOLOSOMA_POLICY_TARGET_ROBOT_ROOT_STATE_ASSIST=0
export HOLOSOMA_POLICY_TARGET_ROBOT_DOF_STATE_ASSIST=0
export HOLOSOMA_MUJOCO_REPLACE_CYLINDERS_WITH_CAPSULES=0
export MUJOCO_OBJECT_COLLISION_MODE=mesh MUJOCO_RESIZE_OBJECT_TO_MOTION_SIZE=0
export HOLOSOMA_DISABLE_AUTO_RESET=1
export HOLOSOMA_DISABLE_MOTION_END_RESET=1
export HOLOSOMA_DISABLE_CLIP_END_RESET=1
export HOLOSOMA_DISABLE_BAD_TRACKING_RESET=1
export HOLOSOMA_SIM_STATE_INCLUDE_OBJECT_CONTACT_DETAILS=1
export HOLOSOMA_POLICY_DEBUG_INPUT_PATH="$OUTPUT_DIR/policy_io.jsonl"
export HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="$ACTOR_STEPS"
export HOLOSOMA_POLICY_DEBUG_INCLUDE_VALUES=1
export HOLOSOMA_MUJOCO_EXPORT_XML_PATH="$OUTPUT_DIR/mujoco_scene.xml"
export OBJECT_URDF

setsid bash mj_track_generalist.sh \
  --motion-file "$MOTION_FILE" \
  --model-ref "$MODEL_ONNX" \
  >"$OUTPUT_DIR/launcher.log" 2>&1 &
launcher_pid=$!

cleanup_rollout() {
  kill -TERM -- "-$launcher_pid" 2>/dev/null || true
  wait "$launcher_pid" 2>/dev/null || true
}
trap cleanup_rollout EXIT INT TERM

controller_args=(
    --state-port "$SIM_STATE_PORT" \
    --sparse-root-command-port "$SPARSE_ROOT_COMMAND_PORT" \
    --policy-control-port "$POLICY_CONTROL_PORT" \
    --policy-overlay-port "$POLICY_OVERLAY_PORT" \
    --output-dir "$OUTPUT_DIR/audit" \
    --forward-command-m "$FORWARD_COMMAND_M" \
    --lift-rel-z-delta-m "$LIFT_REL_Z_DELTA_M" \
    --deadline-publish-lead-ms "$DEADLINE_PUBLISH_LEAD_MS" \
    --actor-steps "$ACTOR_STEPS" \
    --startup-timeout-s "$STARTUP_TIMEOUT_S" \
    --rollout-timeout-s "$ROLLOUT_TIMEOUT_S"
)
if [[ -n "$LATEST_FORWARD_ACTOR_SIM_TIME_S" ]]; then
  controller_args+=(--latest-forward-actor-sim-time-s "$LATEST_FORWARD_ACTOR_SIM_TIME_S")
fi

PYTHONPATH="src/holosoma_inference:src/holosoma${PYTHONPATH:+:$PYTHONPATH}" \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  scripts/mj_forward_after_lift_rollout.py \
    "${controller_args[@]}" \
    >"$OUTPUT_DIR/controller.log" 2>&1
