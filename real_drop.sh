#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
default_model_path="${ROOT_DIR}/_ckps/34qv1qqp_model_40000.onnx"
if [[ ! -f "$default_model_path" ]]; then
  default_model_path="wandb://zihanw22/carry-any/34qv1qqp/latest"
fi
model_path="${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}"

log_dir="${ROOT_DIR}/logs/real_drop_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_drop] log_dir=${log_dir}"
echo "[real_drop] model_path=${model_path}"
command_window_pid=""
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  python3 scripts/show_policy_command.py "${log_dir}/latest_command.json" &
  command_window_pid=$!
  trap 'kill "$command_window_pid" 2>/dev/null || true' EXIT
else
  echo "[command_window] skipping: no DISPLAY/WAYLAND_DISPLAY"
fi
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${log_dir}/latest_command.json" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1 \
  --task.model-path "$model_path" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface"
