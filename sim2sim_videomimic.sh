#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <videomimic_checkpoint_or_run> <holosoma_motion.npz> [output_onnx]"
  exit 1
fi

CKPT_OR_RUN="$1"
MOTION_NPZ="$2"
OUT_ONNX="${3:-/tmp/videomimic_stage2.onnx}"

VM_TASK=${VM_TASK:-g1_deepmimic_proj_heightfield}
VM_ADAPTER=${VM_ADAPTER:-legacy}
HOLOSOMA_OBS=${HOLOSOMA_OBS:-videomimic}
INF_CONFIG=${INF_CONFIG:-g1-29dof-videomimic}
VM_LOG_ROOT=${VM_LOG_ROOT:-}

CKPT_ARGS=()
if [[ -f "${CKPT_OR_RUN}" ]]; then
  CKPT_ARGS+=(--checkpoint "${CKPT_OR_RUN}")
else
  CKPT_ARGS+=(--load-run "${CKPT_OR_RUN}")
  if [[ -n "${VM_LOG_ROOT}" ]]; then
    CKPT_ARGS+=(--log-root "${VM_LOG_ROOT}")
  fi
fi

python3 scripts/export_videomimic_onnx.py \
  "${CKPT_ARGS[@]}" \
  --task "${VM_TASK}" \
  --adapter "${VM_ADAPTER}" \
  --holosoma-obs "${HOLOSOMA_OBS}" \
  --motion-file "$MOTION_NPZ" \
  --output "$OUT_ONNX"

python3 src/holosoma_inference/holosoma_inference/run_policy.py "inference:${INF_CONFIG}" \
  --task.model-path "$OUT_ONNX" \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.interface lo
