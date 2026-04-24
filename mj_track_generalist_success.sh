#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${ROOT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"
DEFAULT_CLIP_NAME="${DEFAULT_CLIP_NAME:-box_74}"

usage() {
  cat <<EOF
Usage:
  bash mj_track_generalist_success.sh [clip_name|motion.npz] [model.onnx|wandb://...] [mj_track args...]

Purpose:
  Run the object-generalist MuJoCo tracker on the success-rollout motion bank.

Defaults:
  model      = ${DEFAULT_MODEL_REF}
  motion dir = ${DEFAULT_MOTION_DIR}
  clip       = ${DEFAULT_CLIP_NAME}

Examples:
  bash mj_track_generalist_success.sh
  bash mj_track_generalist_success.sh box_74
  bash mj_track_generalist_success.sh --motion-dir /abs/path/to/success_bank
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

export DEFAULT_MODEL_REF
export DEFAULT_MOTION_DIR
export DEFAULT_CLIP_NAME

exec bash "${ROOT_DIR}/mj_track_generalist.sh" "$@"
