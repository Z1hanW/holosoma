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
# - Strict mode: object_name / object_urdf_path / scene_xml_file / object_mesh_scale
#   must come from retarget output metadata.
# - No filename-based parsing fallback is allowed.
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
  g1|t1) ;;
  *)
    echo "[ERROR] Unsupported ROBOT='${ROBOT}'. Expected one of: g1, t1" >&2
    exit 1
    ;;
esac

extract_metadata() {
  local npz_path="$1"
  "${PYTHON_BIN}" - "${npz_path}" <<'PY'
import sys
from pathlib import Path

import numpy as np

npz_path = Path(sys.argv[1])
if not npz_path.exists():
    raise SystemExit(f"[ERROR] missing input file: {npz_path}")

with np.load(str(npz_path), allow_pickle=True) as data:
    required = ("object_name", "object_urdf_path", "scene_xml_file", "object_mesh_scale")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise SystemExit(
            f"[ERROR] Missing strict metadata in {npz_path}: {missing}. "
            "Please rerun retargeting with updated code."
        )

    def scalar_str(key: str) -> str:
        arr = np.asarray(data[key])
        if arr.size == 0:
            return ""
        if arr.shape == ():
            val = arr.item()
        else:
            val = arr.reshape(-1)[0]
            if hasattr(val, "item"):
                val = val.item()
        return str(val).strip()

    object_name = scalar_str("object_name")
    object_urdf_path = scalar_str("object_urdf_path")
    scene_xml_file = scalar_str("scene_xml_file")

    if not object_name:
        raise SystemExit(f"[ERROR] Empty object_name in metadata: {npz_path}")
    if not object_urdf_path:
        raise SystemExit(f"[ERROR] Empty object_urdf_path in metadata: {npz_path}")
    if not scene_xml_file:
        raise SystemExit(f"[ERROR] Empty scene_xml_file in metadata: {npz_path}")

    scale = np.asarray(data["object_mesh_scale"], dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size != 3:
        raise SystemExit(
            f"[ERROR] Invalid object_mesh_scale in metadata: {npz_path}, shape={np.asarray(data['object_mesh_scale']).shape}"
        )

    print(object_name)
    print(object_urdf_path)
    print(scene_xml_file)
    print(f"{scale[0]:.8g} {scale[1]:.8g} {scale[2]:.8g}")
PY
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
echo "[INFO] Robot     : ${ROBOT}"
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

  metadata_tmp=$(mktemp)
  if ! extract_metadata "${input_path}" >"${metadata_tmp}"; then
    rm -f "${metadata_tmp}"
    echo "[ERROR] Failed to extract strict metadata from: ${input_path}" >&2
    exit 1
  fi
  metadata_line_count=$(wc -l <"${metadata_tmp}" | tr -d ' ')
  if [[ "${metadata_line_count}" -ne 4 ]]; then
    rm -f "${metadata_tmp}"
    echo "[ERROR] Unexpected metadata format in ${input_path}: ${metadata_line_count} lines" >&2
    exit 1
  fi

  object_name=$(sed -n '1p' "${metadata_tmp}")
  object_urdf=$(sed -n '2p' "${metadata_tmp}")
  scene_xml=$(sed -n '3p' "${metadata_tmp}")
  object_scale=$(sed -n '4p' "${metadata_tmp}")
  rm -f "${metadata_tmp}"

  if [[ ! -f "${object_urdf}" ]]; then
    echo "[ERROR] Missing object URDF from metadata for '${object_name}': ${object_urdf}" >&2
    exit 1
  fi
  if [[ ! -f "${scene_xml}" ]]; then
    echo "[ERROR] Missing scene XML from metadata for '${object_name}': ${scene_xml}" >&2
    exit 1
  fi

  output_path="${OUTPUT_DIR}/${stem}_mj_w_obj.npz"
  echo "[INFO] Converting ${base_name}.npz (object=${object_name}, scale=${object_scale}) -> $(basename "${output_path}")"

  cmd=(
    "${PYTHON_BIN}" "data_conversion/convert_data_format_mj.py"
    --input_file "${input_path}"
    --robot "${ROBOT}"
    --output_fps "${OUTPUT_FPS}"
    --output_name "${output_path}"
    --data_format "${DATA_FORMAT}"
    --object_name "${object_name}"
    --scene_xml_file "${scene_xml}"
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
