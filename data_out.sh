#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/data_paths.sh"

NFS_ROOT=${NFS_ROOT:-/nfs/zzzihanw}
BOX_BUCKET=${BOX_BUCKET:-box3r}
CRISP_BUCKET=${CRISP_BUCKET:-crisp}
RETARGETING_CONVERTED_RES_LOCAL=${RETARGETING_CONVERTED_RES_LOCAL:-/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res}
RETARGETING_CONVERTED_RES_NFS=${RETARGETING_CONVERTED_RES_NFS:-/nfs/zzzihanw/amass/converted_res}
DRY_RUN=${DRY_RUN:-0}
STRICT=${STRICT:-0}
# Restore current data layout before payload restore.
# 1 = recreate inferred symlinks (recommended), 0 = skip.
RESTORE_LAYOUT=${RESTORE_LAYOUT:-1}
# If inferred link path already exists as a real file/dir:
# 0 = keep and warn, 1 = move aside to *.bak.<timestamp> and create symlink.
FORCE_LAYOUT_LINKS=${FORCE_LAYOUT_LINKS:-0}
# Layout mapping base roots:
#   /home/ubuntu/FAR/holosoma/...  -> /data/holosoma_moved/...
LAYOUT_SRC_ROOT=${LAYOUT_SRC_ROOT:-/home/ubuntu/FAR/holosoma}
LAYOUT_DST_ROOT=${LAYOUT_DST_ROOT:-/data/holosoma_moved}
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
links_created=0
links_skipped=0

ensure_layout_symlink() {
  local link_path="$1"
  local link_target="$2"

  mkdir -p "$(dirname "${link_path}")"
  mkdir -p "$(dirname "${link_target}")"

  if [[ -L "${link_path}" ]]; then
    local cur
    cur="$(readlink "${link_path}")"
    if [[ "${cur}" == "${link_target}" ]]; then
      return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[DRY-RUN] Relink: ${link_path} -> ${link_target}"
      return 0
    fi
    rm -f "${link_path}"
    ln -s "${link_target}" "${link_path}"
    links_created=$((links_created + 1))
    return 0
  fi

  if [[ -e "${link_path}" ]]; then
    if [[ "${FORCE_LAYOUT_LINKS}" == "1" ]]; then
      local backup
      backup="${link_path}.bak.$(date +%Y%m%d_%H%M%S)"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DRY-RUN] Move aside then link:"
        echo "          mv ${link_path} ${backup}"
        echo "          ln -s ${link_target} ${link_path}"
        return 0
      fi
      mv "${link_path}" "${backup}"
      ln -s "${link_target}" "${link_path}"
      links_created=$((links_created + 1))
    else
      echo "[WARN] Existing non-symlink, keep as-is: ${link_path}"
      links_skipped=$((links_skipped + 1))
    fi
    return 0
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] Symlink: ${link_path} -> ${link_target}"
    return 0
  fi
  ln -s "${link_target}" "${link_path}"
  links_created=$((links_created + 1))
}

restore_inferred_layout() {
  if [[ "${RESTORE_LAYOUT}" != "1" ]]; then
    echo "[INFO] Skip layout restore (RESTORE_LAYOUT=${RESTORE_LAYOUT})"
    return 0
  fi

  local tmp_pairs
  tmp_pairs="$(mktemp)"
  : > "${tmp_pairs}"

  infer_one() {
    local p="$1"
    local prefix rest pkg
    # Match: <LAYOUT_SRC_ROOT>/src/<retarget_pkg>/converted_res/<...>
    prefix="${LAYOUT_SRC_ROOT}/src/"
    if [[ "${p}" == "${prefix}"*"/converted_res/"* ]]; then
      rest="${p#${prefix}}"
      pkg="${rest%%/*}"
      local link_path="${LAYOUT_SRC_ROOT}/src/${pkg}/converted_res"
      local link_target="${LAYOUT_DST_ROOT}/src/${pkg}/converted_res"
      printf '%s\t%s\n' "${link_path}" "${link_target}" >> "${tmp_pairs}"
    fi
  }

  local p
  for p in "${BOX3R_PATHS[@]}"; do
    infer_one "${p}"
  done
  for p in "${CRISP_PATHS[@]}"; do
    infer_one "${p}"
  done

  sort -u -o "${tmp_pairs}" "${tmp_pairs}"
  if [[ ! -s "${tmp_pairs}" ]]; then
    rm -f "${tmp_pairs}"
    return 0
  fi

  echo "[INFO] Restoring inferred symlink layout..."
  while IFS=$'\t' read -r link_path link_target; do
    [[ -z "${link_path}" || -z "${link_target}" ]] && continue
    ensure_layout_symlink "${link_path}" "${link_target}"
  done < "${tmp_pairs}"
  rm -f "${tmp_pairs}"
}

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

restore_inferred_layout

restore_converted_res_from_amass() {
  local src="${RETARGETING_CONVERTED_RES_NFS}"
  local dst="${RETARGETING_CONVERTED_RES_LOCAL}"

  if [[ ! -e "${src}" ]]; then
    echo "[WARN] Missing converted_res on NFS, skip: ${src}"
    missing=$((missing + 1))
    if [[ "${STRICT}" == "1" ]]; then
      return 1
    fi
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  echo "[INFO] Restore <- amass: ${src} => ${dst}"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    rsync "${rsync_opts[@]}" "${src}/" "${dst}/"
  else
    rsync "${rsync_opts[@]}" "${src}" "${dst}"
  fi
  restored=$((restored + 1))
}

restore_converted_res_from_amass

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
echo "[INFO] Links created   : ${links_created}"
echo "[INFO] Links skipped   : ${links_skipped}"
echo "[INFO] amass path      : ${RETARGETING_CONVERTED_RES_NFS}"
