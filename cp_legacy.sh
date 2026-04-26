#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LEGACY_OUT_DIR="${LEGACY_OUT_DIR:-${SCRIPT_DIR}/data/ds_box_data_legacy}"
LEGACY_TARGET_NAME="${LEGACY_TARGET_NAME:-train_g1_w_obj_prepared}"
LEGACY_TARGET_DIR="${LEGACY_OUT_DIR}/${LEGACY_TARGET_NAME}"
LEGACY_SOURCE_DIR="${LEGACY_SOURCE_DIR:-${LEGACY_TARGET_DIR}}"
ALLOW_NFS_LEGACY_SOURCE="${ALLOW_NFS_LEGACY_SOURCE:-0}"

case "$(echo "${ALLOW_NFS_LEGACY_SOURCE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    ;;
  0|false|no|off)
    if [[ "${LEGACY_SOURCE_DIR}" == /nfs/* ]]; then
      echo "[ERROR] legacy source must be local, not NFS: ${LEGACY_SOURCE_DIR}" >&2
      echo "[ERROR] Copy the legacy subset into ${LEGACY_TARGET_DIR} first, or set LEGACY_SOURCE_DIR to a local path." >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] ALLOW_NFS_LEGACY_SOURCE must be one of: 0|1|true|false|yes|no|on|off" >&2
    exit 2
    ;;
esac

if [[ ! -d "${LEGACY_SOURCE_DIR}" ]]; then
  echo "[ERROR] legacy source dir not found: ${LEGACY_SOURCE_DIR}" >&2
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
echo "[INFO] legacy_target_dir=${LEGACY_TARGET_DIR}"
