#!/usr/bin/env bash
set -euo pipefail

# Batch-convert BEHAVE retarget outputs into RL training motion files.
#
# Input (retarget raw):
#   src/holosoma_retargeting/demo_results_parallel/<robot>/object_interaction/behave_zup/*.npz
#
# Output (training-ready):
#   src/holosoma_retargeting/converted_res/behave/*_mj_w_obj.npz
#
# Notes:
# - Object is parsed per sequence from filename:
#   Date03_Sub03_boxlarge_original.npz -> object_name=boxlarge
# - Each file is converted with its own object XML/URDF (no shared fallback).
#
# Optional overrides:
#   INPUT_DIR=/abs/path/to/raw_npz_dir
#   OUTPUT_DIR=/abs/path/to/converted_npz_dir
#   PYTHON_BIN=python
#   ROBOT=g1
#   OUTPUT_FPS=50
#   DATA_FORMAT=behave_zup
#   USE_OMNIRETARGET_DATA=0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RETARGET_ROOT="${SCRIPT_DIR}/src/holosoma_retargeting"

INPUT_DIR=${INPUT_DIR:-"${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/behave_zup"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RETARGET_ROOT}/converted_res/behave"}
PYTHON_BIN=${PYTHON_BIN:-python}
ROBOT=${ROBOT:-g1}
OUTPUT_FPS=${OUTPUT_FPS:-50}
DATA_FORMAT=${DATA_FORMAT:-behave_zup}
USE_OMNIRETARGET_DATA=${USE_OMNIRETARGET_DATA:-0}

CONVERTER="${RETARGET_ROOT}/data_conversion/convert_data_format_mj.py"

case "${ROBOT}" in
  g1) ROBOT_DOF=29 ;;
  t1) ROBOT_DOF=23 ;;
  *)
    echo "[ERROR] Unsupported ROBOT='${ROBOT}'. Expected one of: g1, t1" >&2
    exit 1
    ;;
esac

parse_object_name() {
  local seq_stem="$1"
  local parts
  IFS='_' read -r -a parts <<< "${seq_stem}"
  if [[ ${#parts[@]} -lt 3 ]]; then
    return 1
  fi
  local object_name="${parts[2]}"
  object_name="${object_name,,}"
  if [[ -z "${object_name}" ]]; then
    return 1
  fi
  printf '%s' "${object_name}"
}

if [[ ! -f "${CONVERTER}" ]]; then
  echo "[ERROR] Converter not found: ${CONVERTER}" >&2
  exit 1
fi

if [[ ! -d "${RETARGET_ROOT}" ]]; then
  echo "[ERROR] RETARGET_ROOT not found: ${RETARGET_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
files=("${INPUT_DIR}"/*.npz)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "[ERROR] No .npz files found in INPUT_DIR: ${INPUT_DIR}" >&2
  exit 1
fi

echo "[INFO] Input dir : ${INPUT_DIR}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Robot     : ${ROBOT} (${ROBOT_DOF} dof)"
echo "[INFO] Converting ${#files[@]} files..."

pushd "${RETARGET_ROOT}" >/dev/null
trap 'popd >/dev/null' EXIT

for input_path in "${files[@]}"; do
  base_name=$(basename "${input_path}" .npz)
  if [[ "${base_name}" == *_original ]]; then
    stem="${base_name%_original}"
  else
    stem="${base_name}"
  fi

  if ! object_name=$(parse_object_name "${stem}"); then
    echo "[ERROR] Cannot parse object name from sequence stem: ${stem}" >&2
    exit 1
  fi

  object_urdf="${RETARGET_ROOT}/models/behave_objects/${object_name}/${object_name}.urdf"
  scene_xml="${RETARGET_ROOT}/models/${ROBOT}/${ROBOT}_${ROBOT_DOF}dof_w_${object_name}.xml"
  if [[ ! -f "${object_urdf}" ]]; then
    echo "[ERROR] Missing per-object URDF for '${object_name}': ${object_urdf}" >&2
    exit 1
  fi
  if [[ ! -f "${scene_xml}" ]]; then
    echo "[ERROR] Missing per-object scene XML for '${object_name}': ${scene_xml}" >&2
    exit 1
  fi

  output_path="${OUTPUT_DIR}/${stem}_mj_w_obj.npz"
  echo "[INFO] Converting ${base_name}.npz (object=${object_name}) -> $(basename "${output_path}")"

  cmd=(
    "${PYTHON_BIN}" "data_conversion/convert_data_format_mj.py"
    --input_file "${input_path}"
    --robot "${ROBOT}"
    --output_fps "${OUTPUT_FPS}"
    --output_name "${output_path}"
    --data_format "${DATA_FORMAT}"
    --object_name "${object_name}"
    --has_dynamic_object
    --once
    --headless
  )

  if [[ "${USE_OMNIRETARGET_DATA}" == "1" ]]; then
    cmd+=(--use_omniretarget_data)
  fi

  "${cmd[@]}"
done

echo "[INFO] Done. Converted files are in: ${OUTPUT_DIR}"
