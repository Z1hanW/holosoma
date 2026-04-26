#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEGACY_ROOT="${LEGACY_ROOT:-/nfs/zzzihanw/ds_box_data_v3_apr_21}"
LEGACY_SUBSET_NAME="${LEGACY_SUBSET_NAME:-train_g1_w_obj_prepared_old_tracker_box_le_92_portable}"
LEGACY_SUBSET_DIR="${LEGACY_SUBSET_DIR:-${LEGACY_ROOT}/_motion_subsets/${LEGACY_SUBSET_NAME}}"
LEGACY_OUT_DIR="${LEGACY_OUT_DIR:-${SCRIPT_DIR}/data/ds_box_data_legacy}"
LEGACY_TARGET_NAME="${LEGACY_TARGET_NAME:-train_g1_w_obj_prepared}"
LEGACY_TARGET_DIR="${LEGACY_OUT_DIR}/${LEGACY_TARGET_NAME}"

if [[ ! -d "${LEGACY_SUBSET_DIR}" ]]; then
  echo "[ERROR] legacy subset dir not found: ${LEGACY_SUBSET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LEGACY_SUBSET_DIR}/_generated_urdfs/box_10.urdf" ]]; then
  echo "[ERROR] expected legacy URDF missing under: ${LEGACY_SUBSET_DIR}/_generated_urdfs" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${LEGACY_OUT_DIR}"

if [[ -L "${LEGACY_TARGET_DIR}" ]]; then
  rm -f "${LEGACY_TARGET_DIR}"
fi
mkdir -p "${LEGACY_TARGET_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -avh --delete "${LEGACY_SUBSET_DIR}/" "${LEGACY_TARGET_DIR}/"
else
  rm -rf "${LEGACY_TARGET_DIR}"
  mkdir -p "${LEGACY_TARGET_DIR}"
  cp -a "${LEGACY_SUBSET_DIR}/." "${LEGACY_TARGET_DIR}/"
fi

echo "[INFO] legacy_root=${LEGACY_ROOT}"
echo "[INFO] legacy_subset_dir=${LEGACY_SUBSET_DIR}"
echo "[INFO] copied ${LEGACY_SUBSET_DIR} -> ${LEGACY_TARGET_DIR}"
