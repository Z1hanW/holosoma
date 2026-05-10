#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

log_dir="${ROOT_DIR}/logs/real_lk9ocrn6_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_lk9ocrn6] log_dir=${log_dir}"
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=1 \
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
