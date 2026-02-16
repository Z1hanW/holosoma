#!/usr/bin/env bash
set -euo pipefail

# Batch-convert OMOMO carry retarget outputs into RL training motion files.
#
# Input (retarget raw):
#   src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry/*.npz
#
# Output (training-ready):
#   src/holosoma_retargeting/converted_res/object_interaction/omomo_carry/*_mj_w_obj.npz
#
# Optional overrides:
#   INPUT_DIR=/abs/path/to/raw_npz_dir
#   OUTPUT_DIR=/abs/path/to/converted_npz_dir
#   PYTHON_BIN=python
#   OUTPUT_FPS=50
#   DATA_FORMAT=smplh
#   OBJECT_NAME=largebox
#   USE_OMNIRETARGET_DATA=0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RETARGET_ROOT="${SCRIPT_DIR}/src/holosoma_retargeting"

INPUT_DIR=${INPUT_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_FPS=${OUTPUT_FPS:-50}
DATA_FORMAT=${DATA_FORMAT:-smplh}
OBJECT_NAME=${OBJECT_NAME:-largebox}
USE_OMNIRETARGET_DATA=${USE_OMNIRETARGET_DATA:-0}

CONVERTER="${RETARGET_ROOT}/data_conversion/convert_data_format_mj.py"

if [[ ! -f "${CONVERTER}" ]]; then
  echo "[ERROR] Converter not found: ${CONVERTER}"
  exit 1
fi

if [[ ! -d "${RETARGET_ROOT}" ]]; then
  echo "[ERROR] RETARGET_ROOT not found: ${RETARGET_ROOT}"
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "[ERROR] INPUT_DIR not found: ${INPUT_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
files=("${INPUT_DIR}"/*.npz)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "[ERROR] No .npz files found in INPUT_DIR: ${INPUT_DIR}"
  exit 1
fi

echo "[INFO] Input dir : ${INPUT_DIR}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Converting ${#files[@]} files..."

pushd "${RETARGET_ROOT}" >/dev/null
trap 'popd >/dev/null' EXIT

for input_path in "${files[@]}"; do
  base_name=$(basename "${input_path}" .npz)
  stem="${base_name%_original}"
  output_path="${OUTPUT_DIR}/${stem}_mj_w_obj.npz"

  echo "[INFO] Converting ${base_name}.npz -> $(basename "${output_path}")"

  cmd=(
    "${PYTHON_BIN}" "data_conversion/convert_data_format_mj.py"
    --input_file "${input_path}"
    --output_fps "${OUTPUT_FPS}"
    --output_name "${output_path}"
    --data_format "${DATA_FORMAT}"
    --object_name "${OBJECT_NAME}"
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
