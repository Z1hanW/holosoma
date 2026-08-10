#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
default_model_path="${ROOT_DIR}/_ckps/sx9wctkd_model_40000.onnx"
model_path="${HOLOSOMA_REAL_DEBUG_MODEL_PATH:-${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}}"

log_dir="${ROOT_DIR}/logs/real_debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_debug] log_dir=${log_dir}"
echo "[real_debug] model_path=${model_path}"
echo "[real_debug] interface=${interface}"
echo "[real_debug] target: Unitree L2+A suspended diagnostic position"
echo "[real_debug] joint target: all 29 G1 joints at 0 rad (zero-joint posture)"
echo "[real_debug] transition: smooth 5-second move from the measured joint pose"
echo "[real_debug] policy and motion activation are locked out"
echo "[real_debug] keep the robot supported and press Enter only when the area is clear"

source scripts/source_inference_setup.sh
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-debug-diagnostic \
  --task.model-path "$model_path" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface"
