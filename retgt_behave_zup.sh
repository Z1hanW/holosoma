#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

# BEHAVE z-up retarget (SMPL-H joints + object trajectory)
DATA_ROOT="/data/behave/annotation_30fps_zup"
SEQ_NAME="Date01_Sub01_boxlong_backpack"
OBJECT_ROOT="/data/behave/objects"
SAVE_DIR="demo_results/g1/object_interaction/behave_zup"

python src/holosoma_retargeting/examples/robot_retarget.py \
  --task-type object_interaction \
  --data-format behave_zup \
  --data-path "${DATA_ROOT}" \
  --task-name "${SEQ_NAME}" \
  --task-config.object-mesh-root "${OBJECT_ROOT}" \
  --save-dir "${SAVE_DIR}" \
  --robot g1
