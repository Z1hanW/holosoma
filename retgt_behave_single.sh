#!/usr/bin/env bash
set -euo pipefail

# Single-sequence BEHAVE z-up retargeting helper.
#
# Usage:
#   bash retgt_behave_single.sh [SEQ_NAME] [extra robot_retarget.py args...]
#
# Examples:
#   bash retgt_behave_single.sh
#   bash retgt_behave_single.sh Date03_Sub04_boxtiny
#   CUDA_VISIBLE_DEVICES=7 bash retgt_behave_single.sh Date03_Sub04_boxtiny
#   VISUALIZE=0 SAVE_MODE=1 bash retgt_behave_single.sh Date03_Sub04_boxtiny
#
# Optional env vars:
#   SEQ_NAME      (default: Date03_Sub04_boxtiny)
#   DATA_ROOT     (default: /data/behave/annotation_30fps_zup)
#   OBJECT_ROOT   (default: /data/behave/objects)
#   SAVE_DIR      (default: src/holosoma_retargeting/demo_results/g1/object_interaction/behave_single_test)
#   ROBOT         (default: g1)
#   PYTHON_BIN    (default: python)
#   DEBUG         (default: 1)        # 1/0
#   VISUALIZE     (default: 1)        # 1/0
#   SAVE_MODE     (default: 0)        # 1/0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

SEQ_NAME="${SEQ_NAME:-Date03_Sub04_boxtiny}"
EXTRA_ARGS=()

if [[ $# -gt 0 ]]; then
  if [[ "$1" == -* ]]; then
    EXTRA_ARGS=("$@")
  else
    SEQ_NAME="$1"
    shift
    EXTRA_ARGS=("$@")
  fi
fi

DATA_ROOT="${DATA_ROOT:-/data/behave/annotation_30fps_zup}"
OBJECT_ROOT="${OBJECT_ROOT:-/data/behave/objects}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/src/holosoma_retargeting/demo_results/g1/object_interaction/behave_single_test}"
ROBOT="${ROBOT:-g1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEBUG="${DEBUG:-1}"
VISUALIZE="${VISUALIZE:-1}"
SAVE_MODE="${SAVE_MODE:-0}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${OBJECT_ROOT}" ]]; then
  echo "[ERROR] OBJECT_ROOT not found: ${OBJECT_ROOT}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" src/holosoma_retargeting/examples/robot_retarget.py
  --task-type object_interaction
  --data-format behave_zup
  --data-path "${DATA_ROOT}"
  --task-name "${SEQ_NAME}"
  --task-config.object-mesh-root "${OBJECT_ROOT}"
  --save-dir "${SAVE_DIR}"
  --robot "${ROBOT}"
)

if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  cmd+=(--retargeter.debug)
fi
if [[ "${VISUALIZE}" == "1" || "${VISUALIZE}" == "true" || "${VISUALIZE}" == "True" ]]; then
  cmd+=(--retargeter.visualize)
fi
if [[ "${SAVE_MODE}" == "1" || "${SAVE_MODE}" == "true" || "${SAVE_MODE}" == "True" ]]; then
  cmd+=(--save-mode)
fi

cmd+=("${EXTRA_ARGS[@]}")

echo "[INFO] Sequence   : ${SEQ_NAME}"
echo "[INFO] Data root  : ${DATA_ROOT}"
echo "[INFO] Object root: ${OBJECT_ROOT}"
echo "[INFO] Save dir   : ${SAVE_DIR}"
echo "[INFO] Robot      : ${ROBOT}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
