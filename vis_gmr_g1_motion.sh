#!/usr/bin/env bash
set -euo pipefail

# Visualize GMR retargeted G1 motions with the same clip-dropdown GUI used by vis_motion_geometry.sh.
#
# Usage:
#   bash ./vis_gmr_g1_motion.sh
#   bash ./vis_gmr_g1_motion.sh kinematic
#   START_CLIP=stair_16 bash ./vis_gmr_g1_motion.sh
#
# Optional env vars:
#   GMR_RETARGET_ROOT  (default: /home/ubuntu/FAR/CRISP-Real2Sim/results/output/retargeting/gmr)
#   GMR_ROBOT          (default: unitree_g1)
#   STAGE_MOTION_DIR   (default: holosoma/data/ds_crisp_data/___crisp_clean_motion_gmr_g1)
#   GEOMETRY_DIR       (default: holosoma/data/ds_crisp_data/___crisp_clean_geometry)
#   REBUILD_LINKS      (default: 1) clear existing staged symlinks before rebuilding
#   DRY_RUN            (default: 0) only stage links and print viewer command

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

VIS_MODE=${1:-${VIS_MODE:-kinematic}}
GMR_RETARGET_ROOT="${GMR_RETARGET_ROOT:-/home/ubuntu/FAR/CRISP-Real2Sim/results/output/retargeting/gmr}"
GMR_ROBOT="${GMR_ROBOT:-unitree_g1}"
STAGE_MOTION_DIR="${STAGE_MOTION_DIR:-${SCRIPT_DIR}/data/ds_crisp_data/___crisp_clean_motion_gmr_g1}"
GEOMETRY_DIR="${GEOMETRY_DIR:-${SCRIPT_DIR}/data/ds_crisp_data/___crisp_clean_geometry}"
REBUILD_LINKS="${REBUILD_LINKS:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${GMR_RETARGET_ROOT}" ]]; then
  echo "[ERROR] GMR retarget root not found: ${GMR_RETARGET_ROOT}" >&2
  exit 1
fi

mkdir -p "${STAGE_MOTION_DIR}"
if [[ "${REBUILD_LINKS}" == "1" ]]; then
  find "${STAGE_MOTION_DIR}" -maxdepth 1 -type l -name "*.npz" -delete
fi

total_seq=0
linked_seq=0
missing_seq=0
missing_list=()

for seq_dir in "${GMR_RETARGET_ROOT}"/stair_*; do
  [[ -d "${seq_dir}" ]] || continue
  seq_name=$(basename "${seq_dir}")
  total_seq=$((total_seq + 1))
  src_npz="${seq_dir}/${GMR_ROBOT}/${seq_name}_${GMR_ROBOT}_qpos.npz"
  dst_npz="${STAGE_MOTION_DIR}/${seq_name}.npz"
  if [[ -f "${src_npz}" ]]; then
    ln -sfn "${src_npz}" "${dst_npz}"
    linked_seq=$((linked_seq + 1))
  else
    missing_seq=$((missing_seq + 1))
    missing_list+=("${seq_name}")
  fi
done

if [[ "${linked_seq}" -eq 0 ]]; then
  echo "[ERROR] No qpos files found for robot '${GMR_ROBOT}' under ${GMR_RETARGET_ROOT}." >&2
  exit 1
fi

echo "[INFO] gmr_retarget_root=${GMR_RETARGET_ROOT}"
echo "[INFO] gmr_robot=${GMR_ROBOT}"
echo "[INFO] staged_motion_dir=${STAGE_MOTION_DIR}"
echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
echo "[INFO] staged_links=${linked_seq}/${total_seq}"

if [[ "${missing_seq}" -gt 0 ]]; then
  echo "[WARN] missing_gmr_qpos_for=${missing_seq} sequences (showing up to 10): ${missing_list[*]:0:10}" >&2
fi

if [[ -n "${START_CLIP:-}" && ! -f "${STAGE_MOTION_DIR}/${START_CLIP}.npz" ]]; then
  echo "[WARN] START_CLIP not found in staged motions: ${START_CLIP}.npz (viewer will auto-pick)." >&2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1"
  echo "[INFO] launch_cmd=bash ${SCRIPT_DIR}/vis_motion_geometry.sh crisp ${VIS_MODE}"
  exit 0
fi

MOTION_DIR="${STAGE_MOTION_DIR}" \
GEOMETRY_DIR="${GEOMETRY_DIR}" \
PAIR_TERRAIN_WITH_MOTION=True \
OBJECT_URDF="" \
OBJECT_URDF_DIR="" \
OBJECT_FILTER="" \
DATASET_KNOB=crisp \
bash "${SCRIPT_DIR}/vis_motion_geometry.sh" crisp "${VIS_MODE}"
