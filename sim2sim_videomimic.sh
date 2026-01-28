#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <videomimic_checkpoint.pt> <holosoma_motion.npz> [output_onnx]"
  exit 1
fi

CKPT_PATH="$1"
MOTION_NPZ="$2"
OUT_ONNX="${3:-/tmp/videomimic_stage2.onnx}"

python3 scripts/export_videomimic_onnx.py \
  --checkpoint "$CKPT_PATH" \
  --task g1_deepmimic_proj_heightfield \
  --motion-file "$MOTION_NPZ" \
  --output "$OUT_ONNX"

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-videomimic \
  --task.model-path "$OUT_ONNX" \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.interface lo
