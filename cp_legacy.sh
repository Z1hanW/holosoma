#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DS_PATH_HELPERS="${SCRIPT_DIR}/scripts/object_generalist_ds_paths.sh"
if [[ -f "${DS_PATH_HELPERS}" ]]; then
  # shellcheck disable=SC1090
  source "${DS_PATH_HELPERS}"
fi

LEGACY_OUT_DIR="${LEGACY_OUT_DIR:-${SCRIPT_DIR}/data/ds_box_data_legacy}"
LEGACY_TARGET_NAME="${LEGACY_TARGET_NAME:-train_g1_w_obj_prepared}"
LEGACY_TARGET_DIR="${LEGACY_OUT_DIR}/${LEGACY_TARGET_NAME}"
DEFAULT_LEGACY_SOURCE_ROOT="${DEFAULT_LEGACY_SOURCE_ROOT:-/nfs/zzzihanw/ds_box_data_v3_apr_21}"
LEGACY_SOURCE_ROOT="${LEGACY_SOURCE_ROOT:-${DEFAULT_LEGACY_SOURCE_ROOT}}"
LEGACY_SOURCE_DIR_EXPLICIT=0
[[ -n "${LEGACY_SOURCE_DIR+x}" ]] && LEGACY_SOURCE_DIR_EXPLICIT=1
LEGACY_SOURCE_DIR="${LEGACY_SOURCE_DIR:-${LEGACY_TARGET_DIR}}"
ALLOW_NFS_LEGACY_SOURCE="${ALLOW_NFS_LEGACY_SOURCE:-auto}"

resolve_legacy_source_dir() {
  local candidate="${1:-}"
  local resolved_root=""
  local resolved_dir=""

  if [[ -z "${candidate}" ]]; then
    return 1
  fi

  if [[ -f "${candidate}/_generated_urdfs/box_10.urdf" ]]; then
    printf '%s\n' "${candidate%/}"
    return 0
  fi

  if [[ -d "${candidate}" ]] && declare -F ogds_resolve_data_root >/dev/null 2>&1; then
    resolved_root="$(ogds_resolve_data_root "${candidate}")"
    resolved_dir="${resolved_root%/}/${LEGACY_TARGET_NAME}"
    if [[ -f "${resolved_dir}/_generated_urdfs/box_10.urdf" ]]; then
      printf '%s\n' "${resolved_dir}"
      return 0
    fi
  fi

  return 1
}

if resolved_legacy_source_dir="$(resolve_legacy_source_dir "${LEGACY_SOURCE_DIR}")"; then
  LEGACY_SOURCE_DIR="${resolved_legacy_source_dir}"
elif [[ -n "${LEGACY_SOURCE_ROOT}" ]] && resolved_legacy_source_dir="$(resolve_legacy_source_dir "${LEGACY_SOURCE_ROOT}")"; then
  LEGACY_SOURCE_DIR="${resolved_legacy_source_dir}"
fi

case "$(echo "${ALLOW_NFS_LEGACY_SOURCE}" | tr '[:upper:]' '[:lower:]')" in
  auto)
    ;;
  1|true|yes|on)
    ;;
  0|false|no|off)
    if [[ "${LEGACY_SOURCE_DIR}" == /nfs/* ]]; then
      echo "[ERROR] legacy source must be local, not NFS: ${LEGACY_SOURCE_DIR}" >&2
      echo "[ERROR] Set ALLOW_NFS_LEGACY_SOURCE=1, or copy the legacy subset into ${LEGACY_TARGET_DIR} first." >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] ALLOW_NFS_LEGACY_SOURCE must be one of: auto|0|1|true|false|yes|no|on|off" >&2
    exit 2
    ;;
esac

if [[ ! -d "${LEGACY_SOURCE_DIR}" ]]; then
  echo "[ERROR] legacy source dir not found: ${LEGACY_SOURCE_DIR}" >&2
  if [[ "${LEGACY_SOURCE_DIR_EXPLICIT}" -eq 0 ]]; then
    echo "[ERROR] also tried legacy source root: ${LEGACY_SOURCE_ROOT}" >&2
  fi
  exit 1
fi

if [[ ! -f "${LEGACY_SOURCE_DIR}/_generated_urdfs/box_10.urdf" ]]; then
  echo "[ERROR] expected legacy URDF missing under: ${LEGACY_SOURCE_DIR}/_generated_urdfs" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${LEGACY_OUT_DIR}"

if [[ "${LEGACY_SOURCE_DIR}" != "${LEGACY_TARGET_DIR}" ]]; then
  if [[ -L "${LEGACY_TARGET_DIR}" ]]; then
    rm -f "${LEGACY_TARGET_DIR}"
  fi
  mkdir -p "${LEGACY_TARGET_DIR}"

  if command -v rsync >/dev/null 2>&1; then
    rsync -avh --delete "${LEGACY_SOURCE_DIR}/" "${LEGACY_TARGET_DIR}/"
  else
    rm -rf "${LEGACY_TARGET_DIR}"
    mkdir -p "${LEGACY_TARGET_DIR}"
    cp -a "${LEGACY_SOURCE_DIR}/." "${LEGACY_TARGET_DIR}/"
  fi
  echo "[INFO] copied ${LEGACY_SOURCE_DIR} -> ${LEGACY_TARGET_DIR}"
else
  mkdir -p "${LEGACY_TARGET_DIR}"
  echo "[INFO] using local legacy dir: ${LEGACY_TARGET_DIR}"
fi

echo "[INFO] legacy_source_dir=${LEGACY_SOURCE_DIR}"
echo "[INFO] legacy_source_root=${LEGACY_SOURCE_ROOT}"
echo "[INFO] legacy_target_dir=${LEGACY_TARGET_DIR}"
