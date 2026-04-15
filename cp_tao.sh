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

require_dir "${ROLLOUT_ASSET_ROOT}"
require_dir "${ROLLOUT_ASSET_ROOT}/motion_bank"
require_dir "${ROLLOUT_ASSET_ROOT}/motion_bank_drop_final_1aaf51f7c2"
require_dir "${ROLLOUT_ASSET_ROOT}/clips_rollout_ref_44"

mkdir -p outputs/motion_bank
mkdir -p outputs/motion_bank_drop_final_1aaf51f7c2
mkdir -p outputs/clips

echo "[INFO] Syncing teacher rollout motion bank -> outputs/motion_bank"
rsync -avh --delete "${ROLLOUT_ASSET_ROOT}/motion_bank/" outputs/motion_bank/

echo "[INFO] Syncing drop-final motion bank -> outputs/motion_bank_drop_final_1aaf51f7c2"
rsync -avh --delete "${ROLLOUT_ASSET_ROOT}/motion_bank_drop_final_1aaf51f7c2/" outputs/motion_bank_drop_final_1aaf51f7c2/

echo "[INFO] Syncing rollout clip references -> outputs/clips"
rsync -avh "${ROLLOUT_ASSET_ROOT}/clips_rollout_ref_44/" outputs/clips/

if [[ -f "${ROLLOUT_ARCHIVE}" ]]; then
  echo "[INFO] Restoring raw rollout export -> ${RAW_EXPORT_DEST}"
  rm -rf "${RAW_EXPORT_DEST}"
  mkdir -p "${RAW_EXPORT_DEST}"
  tar -xzf "${ROLLOUT_ARCHIVE}" --strip-components=1 -C "${RAW_EXPORT_DEST}"
else
  echo "[WARN] Rollout archive not found at ${ROLLOUT_ARCHIVE}; skipping raw export restore"
fi

echo "[INFO] Restored rollout assets:"
echo "  - $(find outputs/motion_bank -maxdepth 1 -name 'box_*.npz' | wc -l | tr -d ' ') clips in outputs/motion_bank"
echo "  - $(find outputs/motion_bank_drop_final_1aaf51f7c2 -maxdepth 1 -name 'box_*.npz' | wc -l | tr -d ' ') clips in outputs/motion_bank_drop_final_1aaf51f7c2"
echo "  - $(find outputs/clips -maxdepth 1 -type d -name '*box_*' | wc -l | tr -d ' ') clip dirs in outputs/clips"
