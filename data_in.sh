#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/data_paths.sh"

NFS_ROOT=${NFS_ROOT:-/nfs/zzzihanw}
BOX_BUCKET=${BOX_BUCKET:-box3r}
CRISP_BUCKET=${CRISP_BUCKET:-crisp}
DRY_RUN=${DRY_RUN:-0}
# Sync mode:
#   missing (default): only copy files that don't already exist at destination
#   full: copy and update changed files (default rsync behavior)
SYNC_MODE=${SYNC_MODE:-missing}

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found in PATH." >&2
  exit 1
fi

rsync_opts=(-aL --human-readable --info=stats2,progress2)
if [[ "${DRY_RUN}" == "1" ]]; then
  rsync_opts+=(--dry-run)
fi
if [[ "${SYNC_MODE}" == "missing" ]]; then
  rsync_opts+=(--ignore-existing)
elif [[ "${SYNC_MODE}" != "full" ]]; then
  echo "[ERROR] Invalid SYNC_MODE='${SYNC_MODE}'. Use: missing|full" >&2
  exit 1
fi

copied=0
missing=0

copy_one() {
  local bucket="$1"
  local src="$2"
  local rel dst

  if [[ ! -e "${src}" ]]; then
    echo "[WARN] Missing source, skip: ${src}"
    missing=$((missing + 1))
    return 0
  fi

  rel="${src#/}"
  dst="${NFS_ROOT}/${bucket}/${rel}"
  mkdir -p "$(dirname "${dst}")"

  echo "[INFO] Copy -> ${NFS_ROOT}/${bucket}: ${src}"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    rsync "${rsync_opts[@]}" "${src}/" "${dst}/"
  else
    rsync "${rsync_opts[@]}" "${src}" "${dst}"
  fi
  copied=$((copied + 1))
}

mkdir -p "${NFS_ROOT}/${BOX_BUCKET}" "${NFS_ROOT}/${CRISP_BUCKET}"

for p in "${BOX3R_PATHS[@]}"; do
  copy_one "${BOX_BUCKET}" "${p}"
done

for p in "${CRISP_PATHS[@]}"; do
  copy_one "${CRISP_BUCKET}" "${p}"
done

echo "[INFO] Done."
echo "[INFO] Copied entries : ${copied}"
echo "[INFO] Missing entries: ${missing}"
echo "[INFO] NFS root       : ${NFS_ROOT}"
echo "[INFO] Sync mode      : ${SYNC_MODE}"
