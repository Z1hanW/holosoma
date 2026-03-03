#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/data_paths.sh"

NFS_ROOT=${NFS_ROOT:-/nfs/zzzihanw}
BOX_BUCKET=${BOX_BUCKET:-box3r}
CRISP_BUCKET=${CRISP_BUCKET:-crisp}
DRY_RUN=${DRY_RUN:-0}
STRICT=${STRICT:-0}
# Sync mode:
#   missing (default): only restore files that don't already exist locally
#   full: restore and update changed files (default rsync behavior)
SYNC_MODE=${SYNC_MODE:-missing}

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found in PATH." >&2
  exit 1
fi

rsync_opts=(-a --human-readable --info=stats2,progress2)
if [[ "${DRY_RUN}" == "1" ]]; then
  rsync_opts+=(--dry-run)
fi
if [[ "${SYNC_MODE}" == "missing" ]]; then
  rsync_opts+=(--ignore-existing)
elif [[ "${SYNC_MODE}" != "full" ]]; then
  echo "[ERROR] Invalid SYNC_MODE='${SYNC_MODE}'. Use: missing|full" >&2
  exit 1
fi

restored=0
missing=0

restore_one() {
  local bucket="$1"
  local dst="$2"
  local rel src

  rel="${dst#/}"
  src="${NFS_ROOT}/${bucket}/${rel}"

  if [[ ! -e "${src}" ]]; then
    echo "[WARN] Missing source on NFS, skip: ${src}"
    missing=$((missing + 1))
    if [[ "${STRICT}" == "1" ]]; then
      return 1
    fi
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  echo "[INFO] Restore <- ${NFS_ROOT}/${bucket}: ${dst}"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    rsync "${rsync_opts[@]}" "${src}/" "${dst}/"
  else
    rsync "${rsync_opts[@]}" "${src}" "${dst}"
  fi
  restored=$((restored + 1))
}

for p in "${BOX3R_PATHS[@]}"; do
  restore_one "${BOX_BUCKET}" "${p}"
done

for p in "${CRISP_PATHS[@]}"; do
  restore_one "${CRISP_BUCKET}" "${p}"
done

echo "[INFO] Done."
echo "[INFO] Restored entries: ${restored}"
echo "[INFO] Missing entries : ${missing}"
echo "[INFO] NFS root        : ${NFS_ROOT}"
echo "[INFO] Sync mode       : ${SYNC_MODE}"
