#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# BEHAVE z-up retarget (SMPL-H joints + object trajectory)
DATA_ROOT="/data/behave/annotation_30fps_zup"
SEQ_NAME="Date01_Sub01_boxlong_backpack"
OBJECT_ROOT="/data/behave/objects"
SAVE_DIR="demo_results/g1/object_interaction/behave_zup"

OBJ_NAME=$(echo "${SEQ_NAME}" | awk -F'_' '{print $3}')
OBJ_NAME_LOWER=$(echo "${OBJ_NAME}" | tr '[:upper:]' '[:lower:]')
if [[ "${OBJ_NAME_LOWER}" != *box* || "${OBJ_NAME_LOWER}" == "toolbox" ]]; then
  echo "[ERROR] SEQ_NAME must be a box sequence (exclude toolbox). Got: ${SEQ_NAME}"
  exit 1
fi

python src/holosoma_retargeting/examples/robot_retarget.py \
  --task-type object_interaction \
  --data-format behave_zup \
  --data-path "${DATA_ROOT}" \
  --task-name "${SEQ_NAME}" \
  --task-config.object-mesh-root "${OBJECT_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --robot g1
