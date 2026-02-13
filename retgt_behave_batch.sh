#!/usr/bin/env bash
set -euo pipefail

# Batch retargeting for BEHAVE z-up data (step 2 only).
#
# Required inputs:
#   DATA_ROOT:  path to annotation_30fps_zup (per-sequence folders)
#   OBJECT_ROOT: path to BEHAVE objects (contains <obj>/<obj>_f1000.ply)
#
# Optional:
#   ROBOT: g1 or t1 (default: g1)
#   SAVE_DIR: output directory (default: demo_results_parallel/g1/object_interaction/behave_zup)
#   MAX_WORKERS: override parallel workers
#   AUGMENT: True/False (default: False)
#
# Example:
#   DATA_ROOT=/data/behave/annotation_30fps_zup \
#   OBJECT_ROOT=/data/behave/objects \
#   SAVE_DIR=demo_results_parallel/g1/object_interaction/behave_zup \
#   bash retgt_behave_batch.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

DATA_ROOT=${DATA_ROOT:-"/data/behave/annotation_30fps_zup"}
OBJECT_ROOT=${OBJECT_ROOT:-"/data/behave/objects"}
ROBOT=${ROBOT:-"g1"}
SAVE_DIR=${SAVE_DIR:-"demo_results_parallel/${ROBOT}/object_interaction/behave_zup"}
MAX_WORKERS=${MAX_WORKERS:-""}
AUGMENT=${AUGMENT:-"False"}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi
if [[ ! -d "${OBJECT_ROOT}" ]]; then
  echo "[ERROR] OBJECT_ROOT not found: ${OBJECT_ROOT}"
  exit 1
fi

cmd=(
  python src/holosoma_retargeting/examples/parallel_robot_retarget.py
  --task-type object_interaction
  --data-format behave_zup
  --data-dir "${DATA_ROOT}"
  --task-config.object-mesh-root "${OBJECT_ROOT}"
  --save-dir "${SAVE_DIR}"
  --robot "${ROBOT}"
)

if [[ "${AUGMENT}" == "True" || "${AUGMENT}" == "true" || "${AUGMENT}" == "1" ]]; then
  cmd+=(--augmentation)
fi

if [[ -n "${MAX_WORKERS}" ]]; then
  cmd+=(--max-workers "${MAX_WORKERS}")
fi

cmd+=("$@")

"${cmd[@]}"
