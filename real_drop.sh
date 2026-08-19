#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
default_model_path="wandb://zihanw22/carry-any/tuhu3ghf/model_38000.onnx"
model_path="${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}"

log_dir="${ROOT_DIR}/logs/real_drop_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1
command_status_path="${log_dir}/latest_command.json"

echo "[real_drop] log_dir=${log_dir}"
echo "[real_drop] model_path=${model_path}"
command_window_pid=""
viser_pid=""
sim_gt_pid=""
sim_gt_process_group=""

cleanup() {
  if [[ -n "$command_window_pid" ]]; then
    kill "$command_window_pid" 2>/dev/null || true
    wait "$command_window_pid" 2>/dev/null || true
  fi
  if [[ -n "$viser_pid" ]]; then
    kill "$viser_pid" 2>/dev/null || true
    wait "$viser_pid" 2>/dev/null || true
  fi
  if [[ -n "$sim_gt_process_group" ]]; then
    kill -- "-$sim_gt_process_group" 2>/dev/null || true
  elif [[ -n "$sim_gt_pid" ]]; then
    kill "$sim_gt_pid" 2>/dev/null || true
  fi
  if [[ -n "$sim_gt_pid" ]]; then
    wait "$sim_gt_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  python3 scripts/show_policy_command.py "$command_status_path" &
  command_window_pid=$!
else
  echo "[command_window] skipping: no DISPLAY/WAYLAND_DISPLAY"
fi

if [[ "${HOLOSOMA_REAL_DROP_SIM_GT:-1}" != "0" ]]; then
  sim_gt_log="${log_dir}/sim_gt_depth.log"
  sim_gt_shm_name="${HOLOSOMA_REAL_DROP_SIM_GT_SHM_NAME:-sim_gt_depth_raw_shm}"
  if command -v setsid >/dev/null 2>&1; then
    env HOLOSOMA_SIM_GT_STATE_PATH="$command_status_path" HOLOSOMA_SIM_GT_SHM_NAME="$sim_gt_shm_name" \
      setsid bash sim_gt_depth.sh >"$sim_gt_log" 2>&1 &
    sim_gt_pid=$!
    sim_gt_process_group=$sim_gt_pid
  else
    HOLOSOMA_SIM_GT_STATE_PATH="$command_status_path" HOLOSOMA_SIM_GT_SHM_NAME="$sim_gt_shm_name" \
      bash sim_gt_depth.sh >"$sim_gt_log" 2>&1 &
    sim_gt_pid=$!
  fi
  echo "[real_drop] MuJoCo sim GT started pid=${sim_gt_pid} log=${sim_gt_log}"
else
  echo "[real_drop] MuJoCo sim GT disabled by HOLOSOMA_REAL_DROP_SIM_GT=0"
fi
source scripts/source_inference_setup.sh

if [[ "${HOLOSOMA_REAL_VISER:-1}" != "0" ]]; then
  viser_browser_args=()
  if [[ "${HOLOSOMA_REAL_VISER_OPEN_BROWSER:-1}" != "0" ]] \
      && { [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; }; then
    viser_browser_args+=(--open-browser)
  fi
  PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
  python3 scripts/real_viser.py \
    --state-path "$command_status_path" \
    --host "${HOLOSOMA_REAL_VISER_HOST:-127.0.0.1}" \
    --port "${HOLOSOMA_REAL_VISER_PORT:-8080}" \
    --depth-profile "Real D435: 0mcqao8k processing" \
    --depth-source-height 60 \
    --depth-source-width 106 \
    --depth-crop-y-start 2 \
    --depth-crop-x-start 4 \
    --depth-crop-x-end -4 \
    --sim-gt-depth-shm-name "${HOLOSOMA_REAL_DROP_SIM_GT_SHM_NAME:-sim_gt_depth_raw_shm}" \
    "${viser_browser_args[@]}" &
  viser_pid=$!
  echo "[real_viser] started pid=${viser_pid}"
else
  echo "[real_viser] disabled by HOLOSOMA_REAL_VISER=0"
fi

HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="$command_status_path" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1 \
  --task.model-path "$model_path" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface"
