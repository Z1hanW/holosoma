#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# Usage:
#   ./run_multimotion_sim2sim.sh convert   # convert retargeted LAFAN npz -> training npz
#   ./run_multimotion_sim2sim.sh pack      # pack converted npz into an HDF5 motion bank
#   ./run_multimotion_sim2sim.sh train     # multi-motion WBT training
#   ./run_multimotion_sim2sim.sh sim2sim   # MuJoCo sim-to-sim with OBJ geometry
#   ./run_multimotion_sim2sim.sh all       # convert + pack + train
#
# Notes:
# - Source the proper env before running:
#     source scripts/source_isaacgym_setup.sh   (training)
#     source scripts/source_mujoco_setup.sh     (sim2sim)
#     source scripts/source_inference_setup.sh  (sim2sim policy)
# - Edit the paths below to your data.
# ==============================================================================

# ----- user config (edit these) -----
RETARGET_DIR="/ABS/PATH/demo_results_parallel/g1/robot_only/lafan"
CONVERTED_DIR="/ABS/PATH/converted_res/robot_only/lafan"
H5_FILE="${CONVERTED_DIR}/motion_bank.h5"

# Geometry for MuJoCo sim2sim
OBJ_PATH="/ABS/PATH/scene.obj"
OBJ_NUM_ROWS=1
OBJ_NUM_COLS=1

# ONNX policy for sim2sim
ONNX_PATH="/ABS/PATH/model.onnx"
INFERENCE_CONFIG="inference:g1-29dof-wbt"
RL_RATE=50
USE_SIM_TIME=1
INTERFACE="lo"
USE_JOYSTICK=0

# Training config (multi-motion WBT Stage-1)
TRAIN_SIMULATOR="isaacgym"  # isaacgym or isaacsim
TRAIN_EXP="g1-29dof-wbt-motion-tracking-transformer"
NUM_ENVS=8192
HEADLESS=True

# Python to use for conversion/training
PYTHON_BIN="python"

# ----- helpers -----
fail_if_placeholder() {
  local value="$1"
  local name="$2"
  if [[ "$value" == *"/ABS/PATH"* ]]; then
    echo "ERROR: Set $name in $(basename "$0")" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $name not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: $name not found: $path" >&2
    exit 1
  fi
}

convert_lafan() {
  fail_if_placeholder "$RETARGET_DIR" "RETARGET_DIR"
  require_dir "$RETARGET_DIR" "RETARGET_DIR"
  mkdir -p "$CONVERTED_DIR"

  local converter="$ROOT_DIR/holosoma/src/holosoma_retargeting/data_conversion/convert_data_format_mj.py"
  require_file "$converter" "convert_data_format_mj.py"

  shopt -s nullglob
  local files=("$RETARGET_DIR"/*.npz)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: No .npz files found in $RETARGET_DIR" >&2
    exit 1
  fi

  for f in "${files[@]}"; do
    local name
    name=$(basename "$f" .npz)
    "$PYTHON_BIN" "$converter" \
      --input_file "$f" \
      --output_fps 50 \
      --output_name "$CONVERTED_DIR/${name}_mj_fps50.npz" \
      --data_format lafan \
      --object_name ground \
      --once
  done
}

pack_h5() {
  require_dir "$CONVERTED_DIR" "CONVERTED_DIR"
  local packer="$ROOT_DIR/holosoma/src/holosoma_retargeting/data_conversion/convert_npz_to_h5.py"
  require_file "$packer" "convert_npz_to_h5.py"

  "$PYTHON_BIN" "$packer" \
    --input_dir "$CONVERTED_DIR" \
    --output_h5 "$H5_FILE" \
    --pattern "*.npz"
}

train_multimotion() {
  require_file "$H5_FILE" "HDF5 motion bank"
  "$PYTHON_BIN" "$ROOT_DIR/src/holosoma/holosoma/train_agent.py" \
    "exp:${TRAIN_EXP}" \
    "simulator:${TRAIN_SIMULATOR}" \
    --training.num_envs "$NUM_ENVS" \
    --training.headless "$HEADLESS" \
    --command.setup_terms.motion_command.params.motion_config.motion_file "$H5_FILE"
}

run_sim2sim() {
  fail_if_placeholder "$OBJ_PATH" "OBJ_PATH"
  fail_if_placeholder "$ONNX_PATH" "ONNX_PATH"
  require_file "$OBJ_PATH" "OBJ_PATH"
  require_file "$ONNX_PATH" "ONNX_PATH"

  local run_sim_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/src/holosoma/holosoma/run_sim.py"
    "simulator:mujoco"
    "robot:g1-29dof"
    "terrain:terrain-load-obj"
    --terrain.terrain-term.obj-file-path "$OBJ_PATH"
    --terrain.terrain-term.num-rows "$OBJ_NUM_ROWS"
    --terrain.terrain-term.num-cols "$OBJ_NUM_COLS"
    --simulator.config.bridge.interface "$INTERFACE"
  )

  if [[ "$USE_JOYSTICK" == "1" ]]; then
    run_sim_cmd+=(--simulator.config.bridge.use-joystick True)
  fi

  local run_policy_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py"
    "$INFERENCE_CONFIG"
    --task.model-path "$ONNX_PATH"
    --task.interface "$INTERFACE"
    --task.rl-rate "$RL_RATE"
  )

  if [[ "$USE_SIM_TIME" == "1" ]]; then
    run_policy_cmd+=(--task.use-sim-time)
  fi
  if [[ "$USE_JOYSTICK" == "1" ]]; then
    run_policy_cmd+=(--task.use-joystick)
  fi

  echo "[run_sim]   ${run_sim_cmd[*]}"
  echo "[run_policy]${run_policy_cmd[*]}"

  "${run_sim_cmd[@]}" &
  local sim_pid=$!
  trap 'kill ${sim_pid} >/dev/null 2>&1 || true' EXIT

  "${run_policy_cmd[@]}"
}

main() {
  local action="${1:-}"
  case "$action" in
    convert)
      convert_lafan
      ;;
    pack)
      pack_h5
      ;;
    train)
      train_multimotion
      ;;
    sim2sim)
      run_sim2sim
      ;;
    all)
      convert_lafan
      pack_h5
      train_multimotion
      ;;
    *)
      echo "Usage: $0 {convert|pack|train|sim2sim|all}" >&2
      exit 1
      ;;
  esac
}

main "$@"
