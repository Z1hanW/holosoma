#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

motion="${1:-box_75}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  shift
fi

MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-https://wandb.ai/zihanw22/boxer/runs/w5qostjn}}}"
if [[ "${1:-}" == wandb://* || "${1:-}" == https://* || "${1:-}" == *.onnx || "${1:-}" == *.pt ]]; then
  MODEL_INPUT="$1"
  shift
fi

INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-box-w5qostjn}"
INTERFACE="${INTERFACE:-lo}"
MUJOCO_PYTHON_BIN="${MUJOCO_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"

export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}"

clear_split_tracker_env() {
  unset SIM_CLOCK_PORT SIM_STATE_PORT PERCEPTION_OBS_PORT SIM_CONTROL_PORT
  unset SPARSE_ROOT_COMMAND_PORT POLICY_CONTROL_PORT POLICY_OVERLAY_PORT
  unset PERCEPTION_OBS_SHM_NAME HOLOSOMA_POLICY_PERCEPTION_OBS_SHM_NAME
  unset MJ_TRACK_MODE MJ_ENV_AUTO_LAUNCH_POLICY HOLOSOMA_MJ_TRACK_INTERNAL_CORE
  unset HOLOSOMA_MJ_TRACK_RUN_FOREVER ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND
  unset POLICY_USE_PERCEPTION_OBS_SHM PERCEPTION_OBS_TRANSPORT
  unset HOLOSOMA_MUJOCO_BACKSPACE_POLICY_CONTROL HOLOSOMA_MUJOCO_BACKSPACE_AUTORESTART_POLICY
}

if [[ "${HOLOSOMA_SIM2SIM_DIRECT_ROLLOUT:-1}" != "0" ]]; then
  clear_split_tracker_env
fi

RUN_DIR="${RUN_DIR:-${ROOT_DIR}/logs/sim2sim_runs/${motion%.*}__myholosoma}"
SIM_LOG="${SIM_LOG:-${RUN_DIR}/mujoco.log}"
mkdir -p "$RUN_DIR"
: >"$SIM_LOG"

MOTION_FILE="${MOTION_FILE:-${HOLOSOMA_MJ_MOTION:-}}"
if [[ -z "$MOTION_FILE" ]]; then
  if [[ "$motion" == *.npz || "$motion" == */* ]]; then
    MOTION_FILE="$motion"
  elif [[ -f "$ROOT_DIR/data_demo/${motion}.npz" ]]; then
    MOTION_FILE="$ROOT_DIR/data_demo/${motion}.npz"
  else
    MOTION_FILE="/home/user/FAR/holosoma/data_demo/${motion}.npz"
  fi
fi

PYTHON_BIN="$MUJOCO_PYTHON_BIN" bash "$ROOT_DIR/mj_env.sh" \
  --motion-init.enabled=True \
  --motion-init.motion-file "$MOTION_FILE" \
  --motion-init.mode "${SIM_MOTION_INIT_MODE:-raw_motion}" \
  --motion-init.object-name object \
  "$@" >"$SIM_LOG" 2>&1 &
sim_pid=$!

cleanup() {
  kill "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT

until grep -q "Starting direct simulation loop" "$SIM_LOG"; do
  if ! kill -0 "$sim_pid" 2>/dev/null; then
    tail -n 80 "$SIM_LOG" >&2 || true
    exit 1
  fi
  sleep 0.5
done

"$POLICY_PYTHON_BIN" -u src/holosoma_inference/holosoma_inference/run_policy.py "inference:${INFERENCE_CONFIG}" \
  --task.interface "$INTERFACE" \
  --task.model-path "$MODEL_INPUT"
