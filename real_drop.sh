#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
checkpoint="${HOLOSOMA_REAL_MODEL_PATH:-_ckps/swl41n4x_model_20000.onnx}"

log_dir="${ROOT_DIR}/logs/real_drop_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
if [[ "${HOLOSOMA_DEPLOYMENT_AUDIT:-0}" == "1" ]]; then
  export HOLOSOMA_DEPLOYMENT_AUDIT_DIR="${log_dir}/evidence"
else
  unset HOLOSOMA_DEPLOYMENT_AUDIT_DIR
fi
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_drop] log_dir=${log_dir}"
echo "[real_drop] checkpoint=${checkpoint}"
echo "[real_drop] behavior_reference=c416c75ac5bd09c3449336ba3de4b466874ff728"
python3 scripts/show_policy_command.py "${log_dir}/latest_command.json" &
command_window_pid=$!
trap 'kill "$command_window_pid" 2>/dev/null || true' EXIT
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${log_dir}/latest_command.json" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1 \
  --task.model-path "$checkpoint" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface eth0
