#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ $# -lt 1 || $# -gt 2 || ( $# == 2 && $2 != --preflight-only ) ]]; then
  echo "Usage: bash real_training.sh <exact-model.onnx-or-wandb-uri> [--preflight-only]" >&2
  exit 2
fi
model=$1
mode=${2:-}
if [[ -z ${HOLOSOMA_PYTHON:-} ]]; then
  # The existing setup script is not nounset-compatible.
  set +u
  source scripts/source_inference_setup.sh
  set -u
fi
python=${HOLOSOMA_PYTHON:-python3}
export PYTHONPATH="$PWD/src/holosoma:$PWD/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
log_dir="$PWD/logs/real_training_$(date -u +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"
"$python" scripts/real_training_preflight.py --model-path "$model" --output-dir "$log_dir"
[[ $mode != --preflight-only ]] || exit 0
model=$(<"$log_dir/resolved_model_path.txt")
export HOLOSOMA_TRAINING_DEPTH_PROFILE="$log_dir/camera_profile.json"
export HOLOSOMA_DEPTH_SHM_NAME="training_depth_${UID}_$$"
export HOLOSOMA_DEPTH_STATUS_PATH="$log_dir/depth_status.json"
export HOLOSOMA_POLICY_COMMAND_STATUS_PATH="$log_dir/latest_command.json"
export HOLOSOMA_POLICY_COMMAND_CONTROL_PATH="$log_dir/command_control.json"
export HOLOSOMA_POLICY_DEBUG_INPUT_PATH="$log_dir/policy_inputs.jsonl"
export HOLOSOMA_POLICY_DROP_BUTTON=0
export HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0
unset HOLOSOMA_FORCE_MANUAL_SPARSE_ROOT_COMMAND HOLOSOMA_USE_MOTION_DATA_AS_Q_TARGET
depth_pid=""
cleanup() {
  if [[ -n $depth_pid ]]; then
    kill "$depth_pid" 2>/dev/null || true
    wait "$depth_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Do not stop any other camera owner/service. A busy camera must fail explicitly.
"$python" src/holosoma/holosoma/sensors/image_server.py real_d435i \
  --training-depth-profile "$HOLOSOMA_TRAINING_DEPTH_PROFILE" \
  --shared-memory-name "$HOLOSOMA_DEPTH_SHM_NAME" \
  --depth-status-path "$HOLOSOMA_DEPTH_STATUS_PATH" --no-save-images \
  >"$log_dir/depth.log" 2>&1 &
depth_pid=$!
ready=0
for _ in $(seq 1 200); do
  if ! kill -0 "$depth_pid" 2>/dev/null; then
    echo "Depth server exited; see $log_dir/depth.log" >&2
    exit 1
  fi
  if [[ -s $HOLOSOMA_DEPTH_STATUS_PATH ]]; then ready=1; break; fi
  sleep 0.1
done
[[ $ready == 1 ]] || { echo "Depth warm-up timed out" >&2; exit 1; }
echo "Depth and policy bound to $model; audit: $log_dir"
echo "Manual heading-frame dx/dy/dyaw/drop; no automatic pickup detection on real hardware."
"$python" src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1 \
  --task.model-path "$model" --task.use-joystick --task.rl-rate 50 \
  --task.interface "${HOLOSOMA_REAL_INTERFACE:-eth0}"
