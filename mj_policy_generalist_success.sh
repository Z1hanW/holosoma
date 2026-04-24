#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_REF="${DEFAULT_MODEL_REF:-wandb://zihanw22/boxer/u5lguxvl/model_17000.onnx}"
DEFAULT_MOTION_DIR="${DEFAULT_MOTION_DIR:-${ROOT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"

usage() {
  cat <<EOF
Usage:
  bash mj_policy_generalist_success.sh [clip_name|motion.npz] [model.onnx|wandb://...]

Purpose:
  Launch the policy side for the generalist success-rollout MuJoCo demo.
  Pair this with mj_env_generalist_success.sh.

Defaults:
  model      = ${DEFAULT_MODEL_REF}
  motion dir = ${DEFAULT_MOTION_DIR}

Examples:
  bash mj_policy_generalist_success.sh
  bash mj_policy_generalist_success.sh box_74
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
export MODEL_REF="${MODEL_REF:-${MODEL_PATH:-${MODEL_INPUT:-${DEFAULT_MODEL_REF}}}}"
export MODEL_INPUT="${MODEL_INPUT:-${MODEL_REF}}"
export MJ_TRACK_LAUNCHER="${ROOT_DIR}/mj_track_generalist_success.sh"

exec bash "${ROOT_DIR}/mj_policy.sh" "$@"
