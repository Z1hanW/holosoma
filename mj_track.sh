#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_FILE:-$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}"
DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<EOF
Usage:
  bash mj_track.sh [motion.npz] [checkpoint.pt|model.onnx] [viser args...]

Defaults:
  motion = ${DEFAULT_MOTION_FILE}
  model  = ${DEFAULT_MODEL_INPUT}
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

MOTION_FILE="${DEFAULT_MOTION_FILE}"
MODEL_INPUT="${DEFAULT_MODEL_INPUT}"
EXTRA_ARGS=()
POSITIONAL_MODE=1

for arg in "$@"; do
  if [[ "${POSITIONAL_MODE}" == "1" && "${arg}" != -* ]]; then
    if [[ "${MOTION_FILE}" == "${DEFAULT_MOTION_FILE}" ]]; then
      MOTION_FILE="${arg}"
      continue
    fi
    if [[ "${MODEL_INPUT}" == "${DEFAULT_MODEL_INPUT}" ]]; then
      MODEL_INPUT="${arg}"
      continue
    fi
  fi
  POSITIONAL_MODE=0
  EXTRA_ARGS+=("${arg}")
done

export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$ROOT_DIR/src/holosoma/holosoma/viser_mujoco_sim_state.py" \
  --launch-rollout \
  --run-script "$ROOT_DIR/mj_track_core.sh" \
  --motion-file "$MOTION_FILE" \
  --model-path "$MODEL_INPUT" \
  "${EXTRA_ARGS[@]}"
