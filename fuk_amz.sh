#!/usr/bin/env bash
set -euo pipefail

# Restore the latest terrain-aware training bundle from NFS into the current
# local layout used by training/inference scripts.
#
# Usage:
#   bash fuk_amz.sh
#   bash fuk_amz.sh /nfs/zzzihanw/terrain-aware/training_batch_gmr_rebuilt_s074_20260424

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_ROOT="${1:-/nfs/zzzihanw/terrain-aware/training_batch_gmr_rebuilt_s074_20260424}"
DST_GEN_ROOT="${SCRIPT_DIR}/data/ds_crisp_data/_generated"

MOTION_DIR_NAME="___crisp_clean_motion_gmr_g1_trainready_rebuilt_20260423"
GEOMETRY_DIR_NAME="___crisp_clean_geometry_s0p7415730337"
FUSED_OBJ_NAME="terrain_generalist_19x112.obj"
FUSED_META_NAME="terrain_generalist_19x112.json"

SRC_MOTION_DIR="${SRC_ROOT}/${MOTION_DIR_NAME}"
SRC_GEOMETRY_DIR="${SRC_ROOT}/${GEOMETRY_DIR_NAME}"
SRC_FUSED_OBJ="${SRC_ROOT}/${FUSED_OBJ_NAME}"
SRC_FUSED_META="${SRC_ROOT}/${FUSED_META_NAME}"

DST_MOTION_DIR="${DST_GEN_ROOT}/${MOTION_DIR_NAME}"
DST_GEOMETRY_DIR="${DST_GEN_ROOT}/${GEOMETRY_DIR_NAME}"
DST_FUSED_DIR="${DST_GEN_ROOT}/fused"
DST_FUSED_OBJ="${DST_FUSED_DIR}/${FUSED_OBJ_NAME}"
DST_FUSED_META="${DST_FUSED_DIR}/${FUSED_META_NAME}"

if [[ ! -d "${SRC_MOTION_DIR}" ]]; then
  echo "[ERROR] Missing source motion dir: ${SRC_MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -d "${SRC_GEOMETRY_DIR}" ]]; then
  echo "[ERROR] Missing source geometry dir: ${SRC_GEOMETRY_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SRC_FUSED_OBJ}" ]]; then
  echo "[ERROR] Missing source fused obj: ${SRC_FUSED_OBJ}" >&2
  exit 1
fi
if [[ ! -f "${SRC_FUSED_META}" ]]; then
  echo "[ERROR] Missing source fused meta: ${SRC_FUSED_META}" >&2
  exit 1
fi

mkdir -p "${DST_MOTION_DIR}" "${DST_GEOMETRY_DIR}" "${DST_FUSED_DIR}"

echo "[INFO] Copy motion dir..."
rsync -a --delete "${SRC_MOTION_DIR}/" "${DST_MOTION_DIR}/"

echo "[INFO] Copy geometry dir..."
rsync -a --delete "${SRC_GEOMETRY_DIR}/" "${DST_GEOMETRY_DIR}/"

echo "[INFO] Copy fused terrain files..."
rsync -a "${SRC_FUSED_OBJ}" "${DST_FUSED_OBJ}"
rsync -a "${SRC_FUSED_META}" "${DST_FUSED_META}"

src_motion_count="$(find "${SRC_MOTION_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
dst_motion_count="$(find "${DST_MOTION_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
src_geom_count="$(find "${SRC_GEOMETRY_DIR}" -maxdepth 1 -type f -name '*.obj' | wc -l | tr -d ' ')"
dst_geom_count="$(find "${DST_GEOMETRY_DIR}" -maxdepth 1 -type f -name '*.obj' | wc -l | tr -d ' ')"
src_obj_size="$(stat -c '%s' "${SRC_FUSED_OBJ}")"
dst_obj_size="$(stat -c '%s' "${DST_FUSED_OBJ}")"
src_meta_size="$(stat -c '%s' "${SRC_FUSED_META}")"
dst_meta_size="$(stat -c '%s' "${DST_FUSED_META}")"

echo "[INFO] Verify counts/sizes:"
echo "  motion npz: ${src_motion_count} -> ${dst_motion_count}"
echo "  geometry obj: ${src_geom_count} -> ${dst_geom_count}"
echo "  fused obj bytes: ${src_obj_size} -> ${dst_obj_size}"
echo "  fused json bytes: ${src_meta_size} -> ${dst_meta_size}"

if [[ "${src_motion_count}" != "${dst_motion_count}" ]]; then
  echo "[ERROR] Motion file count mismatch." >&2
  exit 2
fi
if [[ "${src_geom_count}" != "${dst_geom_count}" ]]; then
  echo "[ERROR] Geometry file count mismatch." >&2
  exit 2
fi
if [[ "${src_obj_size}" != "${dst_obj_size}" ]]; then
  echo "[ERROR] Fused OBJ size mismatch." >&2
  exit 2
fi
if [[ "${src_meta_size}" != "${dst_meta_size}" ]]; then
  echo "[ERROR] Fused metadata size mismatch." >&2
  exit 2
fi

echo "[OK] Restore complete."
echo "[OK] MOTION_DIR=${DST_MOTION_DIR}"
echo "[OK] OBJ_SOURCE=${DST_GEOMETRY_DIR}"
echo "[OK] FUSED_OBJ=${DST_FUSED_OBJ}"
echo "[OK] FUSED_META=${DST_FUSED_META}"
