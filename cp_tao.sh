#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

NFS_TAO_ROOT=${NFS_TAO_ROOT:-/nfs/zzzihanw/tao}
ROLLOUT_ASSET_ROOT=${ROLLOUT_ASSET_ROOT:-${NFS_TAO_ROOT}/teacher_rollout_assets_20260415}
ROLLOUT_ARCHIVE=${ROLLOUT_ARCHIVE:-${NFS_TAO_ROOT}/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc.tar.gz}
RAW_EXPORT_DEST=${RAW_EXPORT_DEST:-${SCRIPT_DIR}/outputs/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc}

mkdir -p outputs

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

if [[ -f "${ROLLOUT_ARCHIVE}" ]]; then
  echo "[INFO] Restoring raw rollout export -> ${RAW_EXPORT_DEST}"
  rm -rf "${RAW_EXPORT_DEST}"
  mkdir -p "${RAW_EXPORT_DEST}"
  tar -xzf "${ROLLOUT_ARCHIVE}" --strip-components=1 -C "${RAW_EXPORT_DEST}"
else
  echo "[WARN] Rollout archive not found at ${ROLLOUT_ARCHIVE}; skipping raw export restore"
fi

SOURCE_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank"
SOURCE_DROP_FINAL_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank_drop_final_1aaf51f7c2"
SOURCE_CLIPS="${ROLLOUT_ASSET_ROOT}/outputs/clips"
SOURCE_OUTPUTS_VIS="${ROLLOUT_ASSET_ROOT}/outputs_vis"
SOURCE_OUTPUTS_STS="${ROLLOUT_ASSET_ROOT}/outputs_sts"

require_dir "${SOURCE_MOTION_BANK}"
require_dir "${SOURCE_CLIPS}"

validate_rollout_assets() {
  local motion_bank_dir="$1"
  local clips_dir="$2"
  local motion_count clip_count
  motion_count=$(find "${motion_bank_dir}" -maxdepth 1 -type f -name '*.npz' ! -name '_clip_object_urdf_map.json' | wc -l | tr -d ' ')
  clip_count=$(find "${clips_dir}" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')
  if [[ "${motion_count}" -eq 0 || "${clip_count}" -eq 0 ]]; then
    echo "[ERROR] rollout assets are empty: motion_bank=${motion_count}, clips=${clip_count}" >&2
    exit 1
  fi
  if [[ "${motion_count}" -ne "${clip_count}" ]]; then
    echo "[ERROR] rollout assets are incomplete or mismatched: motion_bank=${motion_count}, clips=${clip_count}" >&2
    exit 1
  fi
}

validate_rollout_assets "${SOURCE_MOTION_BANK}" "${SOURCE_CLIPS}"

mkdir -p outputs/motion_bank
mkdir -p outputs/clips
mkdir -p outputs_vis
mkdir -p outputs_sts

count_npz_in_dir() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    find "${path}" -maxdepth 1 -type f -name '*.npz' ! -name '_clip_object_urdf_map.json' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

echo "[INFO] Syncing teacher rollout motion bank -> outputs/motion_bank"
rsync -avh --delete "${SOURCE_MOTION_BANK}/" outputs/motion_bank/

if [[ -n "${SOURCE_DROP_FINAL_MOTION_BANK}" && -d "${SOURCE_DROP_FINAL_MOTION_BANK}" ]]; then
  mkdir -p outputs/motion_bank_drop_final_1aaf51f7c2
  echo "[INFO] Syncing drop-final motion bank -> outputs/motion_bank_drop_final_1aaf51f7c2"
  rsync -avh --delete "${SOURCE_DROP_FINAL_MOTION_BANK}/" outputs/motion_bank_drop_final_1aaf51f7c2/
else
  echo "[WARN] Drop-final motion bank not found in NFS/archive sources; leaving outputs/motion_bank_drop_final_1aaf51f7c2 unchanged"
fi

echo "[INFO] Syncing rollout clip references -> outputs/clips"
rsync -avh --delete "${SOURCE_CLIPS}/" outputs/clips/

if [[ -d "${SOURCE_OUTPUTS_VIS}" ]]; then
  echo "[INFO] Syncing rollout visualizations -> outputs_vis"
  rsync -avh --delete "${SOURCE_OUTPUTS_VIS}/" outputs_vis/
else
  echo "[WARN] Visualization directory not found at ${SOURCE_OUTPUTS_VIS}; skipping outputs_vis sync"
fi

if [[ -d "${SOURCE_OUTPUTS_STS}" ]]; then
  echo "[INFO] Syncing rollout statistics -> outputs_sts"
  rsync -avh --delete "${SOURCE_OUTPUTS_STS}/" outputs_sts/
else
  echo "[WARN] Statistics directory not found at ${SOURCE_OUTPUTS_STS}; skipping outputs_sts sync"
fi

echo "[INFO] Restored rollout assets:"
echo "  - $(count_npz_in_dir outputs/motion_bank) clips in outputs/motion_bank"
echo "  - $(count_npz_in_dir outputs/motion_bank_drop_final_1aaf51f7c2) clips in outputs/motion_bank_drop_final_1aaf51f7c2"
echo "  - $(find outputs/clips -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ') clip dirs in outputs/clips"
echo "  - $(find outputs_vis/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_vis/clips"
echo "  - $(find outputs_sts/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_sts/clips"
