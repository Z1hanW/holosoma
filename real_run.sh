#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_run] log_dir=${log_dir}"
python3 scripts/show_policy_command.py "${log_dir}/latest_command.json" &
command_window_pid=$!
trap 'kill "$command_window_pid" 2>/dev/null || true' EXIT
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${log_dir}/latest_command.json" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-root_pos-contact-aware-actions-no-linvel \
  --task.model-path _ckps/lk9ocrn6_model_11500.onnx \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface eth0
