#!/usr/bin/env bash
set -euo pipefail

# Build train-ready holosoma motion npz from AMASS LAFAN-style npz.
#
# Step 1: reorder AMASS joint order -> canonical G1 order (proxy qpos/qvel).
# Step 2: run convert_data_format_mj.py to generate full training fields
#         (joint_names/body_names/body_pos_w/body_quat_w/...).
#
# Usage:
#   bash prepare_amass_lafan_train.sh
#   AMASS_SRC_DIR=/abs/amass/LAFAN1_npz bash prepare_amass_lafan_train.sh
#   MAX_CLIPS=10 FORCE=1 bash prepare_amass_lafan_train.sh
#
# Optional env vars:
#   AMASS_SRC_DIR   source npz folder (default: ./amass/LAFAN1_npz)
#   PROXY_DIR       reordered proxy folder
#   OUT_DIR         final train-ready npz folder
#   ORDER_MODE      amass_csv|auto_ref|identity (default: amass_csv)
#   WRIST_POLICY    mapped|zero (default: mapped)
#   MAX_CLIPS       0=all (default 0)
#   FORCE           1=rebuild outputs even if exist
#   PYTHON_BIN      python for vis_amass conversion (default: python)
#   CONVERTER_PYTHON python for convert_data_format_mj.py (default: PYTHON_BIN)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
SCENE_XML_FILE_DEFAULT="${SCRIPT_DIR}/src/holosoma_retargeting_my/models/g1/g1_29dof.xml"

AMASS_SRC_DIR=${AMASS_SRC_DIR:-"${SCRIPT_DIR}/amass/LAFAN1_npz"}
PROXY_DIR=${PROXY_DIR:-"${SCRIPT_DIR}/.cache/amass_lafan_proxy_canonical"}
OUT_DIR=${OUT_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting_my/converted_res/robot_only/amass_lafan_fixed"}
ORDER_MODE=${ORDER_MODE:-"amass_csv"}
WRIST_POLICY=${WRIST_POLICY:-"mapped"}
MAX_CLIPS=${MAX_CLIPS:-0}
FORCE=${FORCE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
CONVERTER_PYTHON=${CONVERTER_PYTHON:-"${PYTHON_BIN}"}
SCENE_XML_FILE=${SCENE_XML_FILE:-"${SCENE_XML_FILE_DEFAULT}"}

if [[ ! -d "${AMASS_SRC_DIR}" ]]; then
  echo "[ERROR] AMASS_SRC_DIR not found: ${AMASS_SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${PROXY_DIR}" "${OUT_DIR}"

echo "[INFO] Step 1/2: build reordered proxy motions"
AMASS_SRC_DIR="${AMASS_SRC_DIR}" \
CACHE_DIR="${PROXY_DIR}" \
ORDER_MODE="${ORDER_MODE}" \
WRIST_POLICY="${WRIST_POLICY}" \
MAX_CLIPS="${MAX_CLIPS}" \
CONVERT_ONLY=True \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/vis_amass.sh"

echo "[INFO] Step 2/2: convert proxy motions to train-ready holosoma npz"

mapfile -t PROXY_FILES < <(find "${PROXY_DIR}" -maxdepth 1 -type f -name '*.npz' | sort)
if [[ "${#PROXY_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No proxy npz found in ${PROXY_DIR}" >&2
  exit 1
fi

converted=0
skipped=0

for f in "${PROXY_FILES[@]}"; do
  stem="$(basename "${f}" .npz)"
  out="${OUT_DIR}/${stem}_mj_fps50.npz"

  if [[ "${FORCE}" != "1" && -f "${out}" ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  "${CONVERTER_PYTHON}" src/holosoma_retargeting_my/data_conversion/convert_data_format_mj.py \
    --input-file "${f}" \
    --robot g1 \
    --data-format lafan \
    --object-name ground \
    --scene-xml-file "${SCENE_XML_FILE}" \
    --output-name "${out}" \
    --once \
    --headless

  converted=$((converted + 1))
done

echo "[INFO] Done. converted=${converted}, skipped=${skipped}, total=${#PROXY_FILES[@]}"
echo "[INFO] Train motion dir: ${OUT_DIR}"
