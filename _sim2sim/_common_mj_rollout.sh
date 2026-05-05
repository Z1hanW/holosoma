#!/usr/bin/env bash
set -euo pipefail

SIM2SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SIM2SIM_DIR}/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -eq 0 ]]; then
  set -- box_75
fi

: "${SIM2SIM_POLICY_NAME:?SIM2SIM_POLICY_NAME is required}"
: "${SIM2SIM_MODEL_REF:?SIM2SIM_MODEL_REF is required}"
: "${SIM2SIM_INFERENCE_CONFIG:?SIM2SIM_INFERENCE_CONFIG is required}"

export DEFAULT_MODEL_INPUT="${DEFAULT_MODEL_INPUT:-${SIM2SIM_MODEL_REF}}"
export MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-${SIM2SIM_MODEL_REF}}}}"
export INFERENCE_CONFIG="${INFERENCE_CONFIG:-${SIM2SIM_INFERENCE_CONFIG}}"

export SIM_MOTION_INIT_MODE="${SIM_MOTION_INIT_MODE:-training_default_pose}"
export POLICY_STDIO="${POLICY_STDIO:-inherit}"

exec bash "$ROOT_DIR/mj_rollout.sh" "$@"
