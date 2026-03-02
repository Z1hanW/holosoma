#!/usr/bin/env bash
set -euo pipefail

# Batch retargeting for BEHAVE z-up data (step 2 only).
#
# Required inputs:
#   DATA_ROOT:  path to annotation_30fps_zup (per-sequence folders)
#   OBJECT_ROOT: path to BEHAVE objects (contains <obj>/<obj>_f1000.ply by default)
#
# Optional:
#   ROBOT: g1 or t1 (default: g1)
#   SAVE_DIR: raw retarget output directory
#             (default: src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup)
#   CONVERTED_DIR: converted training-motion output directory
#                  (default: src/holosoma_retargeting/converted_res/behave)
#   AUTO_CONVERT: 1/0 whether to run post-conversion automatically (default: 1)
#   OBJECT_MESH_SUFFIX: object mesh filename suffix (default: _f1000.ply)
#   OBJECT_FILTER: optional comma-separated BEHAVE object names to process
#                  (e.g., boxmedium,boxlarge)
#   MAX_WORKERS: override parallel workers
#   AUGMENT: True/False (default: False)
#   PYTHON_BIN: python executable for post-conversion (default: python)
#   SCENE_XML_FILE: optional explicit MuJoCo scene xml (e.g., .../g1_29dof_w_obj.xml)
#   OBJECT_CONTACT_NAME: optional object geom token for collision filtering (e.g., obj)
#
# Example:
#   DATA_ROOT=/data/behave/annotation_30fps_zup \
#   OBJECT_ROOT=/data/behave/objects \
#   SAVE_DIR=demo_results_parallel/g1/object_interaction/behave_zup \
#   CONVERTED_DIR=src/holosoma_retargeting/converted_res/behave \
#   bash retgt_behave_batch.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

DATA_ROOT=${DATA_ROOT:-"/data/behave/annotation_30fps_zup"}
OBJECT_ROOT=${OBJECT_ROOT:-"/data/behave/objects"}
OBJECT_MESH_SUFFIX=${OBJECT_MESH_SUFFIX:-"_f1000.ply"}
OBJECT_FILTER=${OBJECT_FILTER:-""}
ROBOT=${ROBOT:-"g1"}
SAVE_DIR=${SAVE_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/demo_results_parallel/${ROBOT}/object_interaction/behave_zup"}
CONVERTED_DIR=${CONVERTED_DIR:-"${REPO_ROOT}/src/holosoma_retargeting/converted_res/behave"}
AUTO_CONVERT=${AUTO_CONVERT:-"1"}
MAX_WORKERS=${MAX_WORKERS:-""}
AUGMENT=${AUGMENT:-"False"}
PYTHON_BIN=${PYTHON_BIN:-"python"}
SCENE_XML_FILE=${SCENE_XML_FILE:-""}
OBJECT_CONTACT_NAME=${OBJECT_CONTACT_NAME:-""}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}"
  exit 1
fi
if [[ ! -d "${OBJECT_ROOT}" ]]; then
  echo "[ERROR] OBJECT_ROOT not found: ${OBJECT_ROOT}"
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" src/holosoma_retargeting/examples/parallel_robot_retarget.py
  --task-type object_interaction
  --data-format behave_zup
  --data-dir "${DATA_ROOT}"
  --task-config.object-mesh-root "${OBJECT_ROOT}"
  --task-config.object-mesh-suffix "${OBJECT_MESH_SUFFIX}"
  --save-dir "${SAVE_DIR}"
  --robot "${ROBOT}"
)

if [[ -n "${SCENE_XML_FILE}" ]]; then
  cmd+=(--task-config.scene-xml-file "${SCENE_XML_FILE}")
fi
if [[ -n "${OBJECT_CONTACT_NAME}" ]]; then
  cmd+=(--task-config.object-contact-name "${OBJECT_CONTACT_NAME}")
fi
if [[ -n "${OBJECT_FILTER}" ]]; then
  cmd+=(--task-config.object-name "${OBJECT_FILTER}")
fi

if [[ "${AUGMENT}" == "True" || "${AUGMENT}" == "true" || "${AUGMENT}" == "1" ]]; then
  cmd+=(--augmentation)
fi

if [[ -n "${MAX_WORKERS}" ]]; then
  cmd+=(--max-workers "${MAX_WORKERS}")
fi

cmd+=("$@")

"${cmd[@]}"

if [[ "${AUTO_CONVERT}" == "1" || "${AUTO_CONVERT}" == "true" || "${AUTO_CONVERT}" == "True" ]]; then
  POST_SCRIPT="${REPO_ROOT}/retgt_post_behave.sh"
  if [[ ! -f "${POST_SCRIPT}" ]]; then
    echo "[ERROR] post-conversion script not found: ${POST_SCRIPT}"
    exit 1
  fi
  INPUT_DIR="${SAVE_DIR}" \
  OUTPUT_DIR="${CONVERTED_DIR}" \
  ROBOT="${ROBOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  "${POST_SCRIPT}"
fi
