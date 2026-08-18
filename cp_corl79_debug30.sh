#!/usr/bin/env bash
set -euo pipefail

# Install the exact portable CORL79 + debug30 training bank from NFS.
#
# This script never replaces an existing target.  Both the compressed archive
# and every extracted payload file are hashed before the final atomic rename.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh" ]]; then
  source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
fi
PYTHON_BIN=${PYTHON_BIN:-python3}

BANK_NAME=corl79_plus_debug30_realmesh_categorymass_v1
PAYLOAD_DIGEST=aa4dcb12bc14df37446417d98d7179236960d2c715975d0753438d164ceafa5c
MANIFEST_SHA256=4ce2d9bae329be7f4a89ada211d2a3cf70ce92d4b59afdc094c70d8e3063c878
ARCHIVE_SHA256=651c3459e26752c4318d201730fbd1b5de98e841b79f56f34922a1872d0e3387
ARCHIVE_SIZE_BYTES=1080025978
NFS_ARCHIVE=${NFS_ARCHIVE:-"/nfs/zzzihanw/ds_as_data/_distill/${BANK_NAME}/archives/${PAYLOAD_DIGEST}.tar.gz"}
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-/data/holosoma_inputs}
LOCAL_ARCHIVE_CACHE_ROOT=${LOCAL_ARCHIVE_CACHE_ROOT:-/data/holosoma_cache}

REPO_LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data/ds_as_data")
EXTERNAL_LOCAL_DATA_ROOT=$(realpath -m /data/holosoma_inputs)
LOCAL_DATA_ROOT=$(realpath -m "${LOCAL_DATA_ROOT}")
if [[ "${LOCAL_DATA_ROOT}" != "${REPO_LOCAL_DATA_ROOT}" \
      && "${LOCAL_DATA_ROOT}" != "${EXTERNAL_LOCAL_DATA_ROOT}" ]]; then
  echo "[ERROR] Refusing unexpected LOCAL_DATA_ROOT: ${LOCAL_DATA_ROOT}" >&2
  echo "[ERROR] Allowed roots: ${EXTERNAL_LOCAL_DATA_ROOT}, ${REPO_LOCAL_DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${NFS_ARCHIVE}" || -L "${NFS_ARCHIVE}" ]]; then
  echo "[ERROR] NFS archive is missing or symlinked: ${NFS_ARCHIVE}" >&2
  exit 2
fi
ACTUAL_ARCHIVE_SIZE=$(stat -c '%s' "${NFS_ARCHIVE}")
if [[ "${ACTUAL_ARCHIVE_SIZE}" != "${ARCHIVE_SIZE_BYTES}" ]]; then
  echo "[ERROR] NFS archive size mismatch: expected=${ARCHIVE_SIZE_BYTES} actual=${ACTUAL_ARCHIVE_SIZE}" >&2
  exit 2
fi

LOCAL_PARENT="${LOCAL_DATA_ROOT}/${BANK_NAME}/by-source"
LOCAL_BANK="${LOCAL_PARENT}/${PAYLOAD_DIGEST}"
VERIFY_TOOL="${SCRIPT_DIR}/scripts/build_merged_training_bank.py"
if [[ ! -f "${VERIFY_TOOL}" || -L "${VERIFY_TOOL}" ]]; then
  echo "[ERROR] Missing merged-bank verifier: ${VERIFY_TOOL}" >&2
  exit 2
fi

verify_bank() {
  "${PYTHON_BIN}" "${VERIFY_TOOL}" verify \
    --bank "$1" \
    --expected-digest "${PAYLOAD_DIGEST}" \
    --expected-manifest-sha256 "${MANIFEST_SHA256}"
}

if [[ -e "${LOCAL_BANK}" || -L "${LOCAL_BANK}" ]]; then
  if verify_bank "${LOCAL_BANK}" >/dev/null; then
    echo "[INFO] Exact merged bank is already installed: ${LOCAL_BANK}"
    exit 0
  fi
  echo "[ERROR] Existing target is not the expected immutable bank; refusing to replace it: ${LOCAL_BANK}" >&2
  exit 2
fi

mkdir -p "${LOCAL_PARENT}" "${LOCAL_ARCHIVE_CACHE_ROOT}"
EXTRACT_TMP=$(mktemp -d "${LOCAL_PARENT}/.${PAYLOAD_DIGEST}.incoming.XXXXXX")
CACHE_TMP=$(mktemp -d "${LOCAL_ARCHIVE_CACHE_ROOT}/${BANK_NAME}.${PAYLOAD_DIGEST}.XXXXXX")
CACHED_ARCHIVE="${CACHE_TMP}/${PAYLOAD_DIGEST}.tar.gz"

cleanup() {
  if [[ -d "${EXTRACT_TMP}" ]]; then
    chmod -R u+w "${EXTRACT_TMP}" 2>/dev/null || true
    rm -rf -- "${EXTRACT_TMP}"
  fi
  if [[ -d "${CACHE_TMP}" ]]; then
    chmod -R u+w "${CACHE_TMP}" 2>/dev/null || true
    rm -rf -- "${CACHE_TMP}"
  fi
}
trap cleanup EXIT

echo "[INFO] Copying one content-addressed archive from NFS to local cache."
echo "[INFO] source=${NFS_ARCHIVE}"
echo "[INFO] target=${LOCAL_BANK}"
cp "${NFS_ARCHIVE}" "${CACHED_ARCHIVE}"

"${PYTHON_BIN}" "${VERIFY_TOOL}" verify-archive \
  --archive "${CACHED_ARCHIVE}" \
  --expected-digest "${PAYLOAD_DIGEST}" \
  --expected-manifest-sha256 "${MANIFEST_SHA256}" \
  --expected-archive-sha256 "${ARCHIVE_SHA256}" >/dev/null

tar -xzf "${CACHED_ARCHIVE}" -C "${EXTRACT_TMP}"
EXTRACTED_BANK="${EXTRACT_TMP}/${PAYLOAD_DIGEST}"
if [[ ! -d "${EXTRACTED_BANK}" || -L "${EXTRACTED_BANK}" ]]; then
  echo "[ERROR] Archive did not produce the exact digest root: ${EXTRACTED_BANK}" >&2
  exit 2
fi
verify_bank "${EXTRACTED_BANK}" >/dev/null

if [[ -e "${LOCAL_BANK}" || -L "${LOCAL_BANK}" ]]; then
  echo "[ERROR] Target appeared during installation; refusing to replace it: ${LOCAL_BANK}" >&2
  exit 2
fi
# Linux requires write permission on a moved directory when its parent changes
# because the directory's ``..`` entry is updated.  Only the generation root
# is made writable for the rename; all children remain frozen, and the root is
# immediately restored before final verification.
chmod u+w "${EXTRACTED_BANK}"
mv "${EXTRACTED_BANK}" "${LOCAL_BANK}"
chmod u-w "${LOCAL_BANK}"
rmdir "${EXTRACT_TMP}"
EXTRACT_TMP=""
verify_bank "${LOCAL_BANK}" >/dev/null

echo "[INFO] Installed merged training bank."
echo "[INFO] local_bank=${LOCAL_BANK}"
echo "[INFO] clips=109 categories=box:25,ball:9,barrel:36,bin:39"
echo "[INFO] payload_digest=${PAYLOAD_DIGEST}"
echo "[INFO] manifest_sha256=${MANIFEST_SHA256}"
