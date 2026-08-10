#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
default_model_path="${ROOT_DIR}/_ckps/swl41n4x_model_20000.onnx"
if [[ ! -f "$default_model_path" ]]; then
  default_model_path="wandb://zihanw22/carry-any/swl41n4x/model_20000.onnx"
fi
model_path="${HOLOSOMA_REAL_STUDENT_MODEL_PATH:-${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}}"

log_dir="${ROOT_DIR}/logs/real_student_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1
command_status_path="${log_dir}/latest_command.json"

echo "[real_student] log_dir=${log_dir}"
echo "[real_student] model_path=${model_path}"
command_window_pid=""
viser_pid=""

cleanup() {
  if [[ -n "$command_window_pid" ]]; then
    kill "$command_window_pid" 2>/dev/null || true
    wait "$command_window_pid" 2>/dev/null || true
  fi
  if [[ -n "$viser_pid" ]]; then
    kill "$viser_pid" 2>/dev/null || true
    wait "$viser_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  python3 scripts/show_policy_command.py "$command_status_path" &
  command_window_pid=$!
else
  echo "[command_window] skipping: no DISPLAY/WAYLAND_DISPLAY"
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
