#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

# Legacy-named entrypoint for the complete scientific Python runtime overlay.
# Package NumPy, OmegaConf, OmegaConf's optional attrs semantics, and every
# active non-extra dependency into one immutable, content-addressed
# site-packages tree. Heterogeneous nodes then execute the same AS runtime
# without mutating their Conda environments. DeFM's forbidden network-download
# branch is deliberately outside this profile.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/gpu_launch_defaults.sh"

OUTPUT_ROOT=${1:-${TMPDIR:-/tmp}/holosoma-python-runtime-overlays-${USER:-unknown}}
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(cd "${OUTPUT_ROOT}" && pwd -P)

for command_name in sha256sum tar gzip stat flock mv id; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[ERROR] Required Python runtime overlay command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

BUILD_ROOT=$(mktemp -d "${OUTPUT_ROOT}/.build.XXXXXXXX")
cleanup() {
  # The staged overlay is deliberately sealed read-only below.  Re-open only
  # this private temporary tree so the EXIT trap can remove it.
  chmod -R u+w "${BUILD_ROOT}" 2>/dev/null || true
  rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT
SITE_PACKAGES="${BUILD_ROOT}/site-packages"
mkdir -p "${SITE_PACKAGES}"

"${PYTHON_BIN}" -I "${SCRIPT_DIR}/stage_python_runtime_overlay.py" \
  --site-packages "${SITE_PACKAGES}"
printf '%s\n' \
  'holosoma-python-runtime-exact-tree-v2' \
  'distribution-closure=required' \
  'bytecode-cache=forbidden' \
  'empty-directories=forbidden' \
  'permissions=read-only' \
  >"${SITE_PACKAGES}/.holosoma-runtime-contract"

# Bytecode caches are deliberately absent from the scientific runtime.  They
# are interpreter-version-specific, are not needed for correctness, and would
# otherwise permit a second importable representation of a module.  Empty
# directories are removed as well: a directory on sys.path can participate in
# namespace-package resolution even when it contains no regular file.
if find "${SITE_PACKAGES}" \
  \( -type d -name '__pycache__' -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print -quit | grep -q .; then
  echo "[ERROR] Python runtime overlay unexpectedly contains Python bytecode cache files." >&2
  exit 2
fi
find "${SITE_PACKAGES}" -depth -mindepth 1 -type d -empty -delete

if find "${SITE_PACKAGES}" -mindepth 1 ! -type f ! -type d -print -quit | grep -q .; then
  echo "[ERROR] Python runtime overlay contains a symlink or special file." >&2
  exit 2
fi
(
  cd "${SITE_PACKAGES}"
  find . -type f ! -name '.holosoma-runtime-manifest.sha256' -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    > .holosoma-runtime-manifest.sha256
)
MANIFEST_SHA256=$(sha256sum "${SITE_PACKAGES}/.holosoma-runtime-manifest.sha256" | awk '{print $1}')
RUNTIME_ID="python-runtime-v2-${MANIFEST_SHA256}"
ARCHIVE_PATH="${OUTPUT_ROOT}/${RUNTIME_ID}.tar.gz"

# Seal and canonicalize the complete runtime tree.  PYTHONDONTWRITEBYTECODE is
# also exported by the launcher, but directory immutability is the fail-closed
# guarantee that no post-build __pycache__ or extra importable module can
# appear in the overlay.  Explicit 0555/0444 modes also keep both the installed
# semantics and deterministic tar bytes independent of the controller umask
# and of incidental modes in the mutable source environment.
find "${SITE_PACKAGES}" -type d -exec chmod 555 {} +
find "${SITE_PACKAGES}" -type f -exec chmod 444 {} +
"${PYTHON_BIN}" -I -S "${SCRIPT_DIR}/verify_python_runtime_overlay.py" \
  --site-packages "${SITE_PACKAGES}" \
  --manifest-sha256 "${MANIFEST_SHA256}" \
  --require-distribution-closure >&2

# Build a complete candidate on every invocation, even on a cache hit.  A
# content-addressed filename is not authentication: the cached path can be
# stale, tampered, writable, or replaced with an alias.  Keeping the candidate
# below BUILD_ROOT also guarantees that publication is a same-filesystem
# atomic rename and that no partial archive is ever visible at ARCHIVE_PATH.
CANDIDATE_ARCHIVE="${BUILD_ROOT}/${RUNTIME_ID}.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "${BUILD_ROOT}" -cf - site-packages \
  | gzip -n >"${CANDIDATE_ARCHIVE}"
chmod 444 "${CANDIDATE_ARCHIVE}"
CANDIDATE_ARCHIVE_SHA256=$(sha256sum "${CANDIDATE_ARCHIVE}" | awk '{print $1}')

# Serialize cooperating publishers for this exact runtime identity.  Every
# publisher has already built and verified its private candidate before taking
# the lock, so the critical section contains only bounded cache inspection and
# one atomic rename (or an authenticated reuse).  OUTPUT_ROOT is an
# owner-controlled operational cache, not a boundary against malicious
# same-UID replacement; nevertheless, never follow or truncate a pre-existing
# lock alias, and bind the opened descriptor back to the exact inspected inode.
PUBLISH_LOCK="${OUTPUT_ROOT}/.${RUNTIME_ID}.publish.lock"
if [[ ! -e "${PUBLISH_LOCK}" && ! -L "${PUBLISH_LOCK}" ]]; then
  # noclobber creates the owner-only lock inode without following an entry
  # concurrently installed at this name.  A racing cooperating creator is
  # harmless and is validated below.
  (umask 077; set -o noclobber; : >"${PUBLISH_LOCK}") 2>/dev/null || true
fi
if [[ ! -f "${PUBLISH_LOCK}" || -L "${PUBLISH_LOCK}" \
      || "$(stat -c '%h' -- "${PUBLISH_LOCK}" 2>/dev/null || true)" != 1 \
      || "$(stat -c '%u' -- "${PUBLISH_LOCK}" 2>/dev/null || true)" != "$(id -u)" \
      || "$(stat -c '%a' -- "${PUBLISH_LOCK}" 2>/dev/null || true)" != 600 \
      || "$(stat -c '%s' -- "${PUBLISH_LOCK}" 2>/dev/null || true)" != 0 ]]; then
  echo "[ERROR] Refusing aliased or malformed Python runtime overlay publish lock: ${PUBLISH_LOCK}" >&2
  exit 2
fi
LOCK_PATH_FINGERPRINT=$(stat -c '%d:%i:%h:%u:%a:%s' -- "${PUBLISH_LOCK}")
exec 9<>"${PUBLISH_LOCK}"
if [[ ! -f "${PUBLISH_LOCK}" || -L "${PUBLISH_LOCK}" ]]; then
  echo "[ERROR] Python runtime overlay publish lock changed while opening: ${PUBLISH_LOCK}" >&2
  exit 2
fi
LOCK_FD_FINGERPRINT=$(stat -Lc '%d:%i:%h:%u:%a:%s' -- "/proc/$$/fd/9")
LOCK_PATH_RECHECK=$(stat -c '%d:%i:%h:%u:%a:%s' -- "${PUBLISH_LOCK}")
if [[ "${LOCK_FD_FINGERPRINT}" != "${LOCK_PATH_FINGERPRINT}" \
      || "${LOCK_PATH_RECHECK}" != "${LOCK_PATH_FINGERPRINT}" ]]; then
  echo "[ERROR] Python runtime overlay publish lock identity changed while opening: ${PUBLISH_LOCK}" >&2
  exit 2
fi
if ! flock -w 60 -x 9; then
  echo "[ERROR] Timed out after 60 seconds acquiring Python runtime overlay publish lock: ${PUBLISH_LOCK}" >&2
  exit 2
fi

if [[ -e "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" ]]; then
  if [[ -L "${ARCHIVE_PATH}" ]]; then
    echo "[ERROR] Refusing symlink Python runtime overlay cache entry: ${ARCHIVE_PATH}" >&2
    exit 2
  fi
  if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "[ERROR] Refusing non-regular Python runtime overlay cache entry: ${ARCHIVE_PATH}" >&2
    exit 2
  fi
  if [[ "$(stat -c '%h' -- "${ARCHIVE_PATH}")" != 1 ]]; then
    echo "[ERROR] Refusing multiply-linked Python runtime overlay cache entry: ${ARCHIVE_PATH}" >&2
    exit 2
  fi

  CACHED_ARCHIVE_SHA256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
  if [[ "${CACHED_ARCHIVE_SHA256}" == "${CANDIDATE_ARCHIVE_SHA256}" ]]; then
    # A byte-exact single-link cache hit is safe to reuse after repairing only
    # its accidental write bits.  The owning account is not a security
    # boundary, so consumers must still verify the returned digest/manifest.
    chmod 444 "${ARCHIVE_PATH}"
    rm -f "${CANDIDATE_ARCHIVE}"
  else
    # The existing single-link regular entry is stale or tampered.  Replace
    # its directory entry atomically; never modify its bytes in place.
    mv -fT "${CANDIDATE_ARCHIVE}" "${ARCHIVE_PATH}"
  fi
else
  # The per-runtime lock excludes cooperating publishers.  --no-clobber makes
  # an unexpected non-cooperating directory-entry race fail closed rather than
  # overwrite a path that was not inspected above.
  if ! mv -T --no-clobber "${CANDIDATE_ARCHIVE}" "${ARCHIVE_PATH}"; then
    echo "[ERROR] Python runtime overlay cache path appeared during atomic publication: ${ARCHIVE_PATH}" >&2
    exit 2
  fi
fi

if [[ ! -f "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" ]]; then
  echo "[ERROR] Python runtime overlay publication did not produce a regular non-symlink archive: ${ARCHIVE_PATH}" >&2
  exit 2
fi
if [[ "$(stat -c '%h' -- "${ARCHIVE_PATH}")" != 1 ]]; then
  echo "[ERROR] Published Python runtime overlay archive is not single-link: ${ARCHIVE_PATH}" >&2
  exit 2
fi
if [[ "$(stat -c '%a' -- "${ARCHIVE_PATH}")" != 444 ]]; then
  echo "[ERROR] Published Python runtime overlay archive is not sealed 0444: ${ARCHIVE_PATH}" >&2
  exit 2
fi
ARCHIVE_SHA256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
if [[ "${ARCHIVE_SHA256}" != "${CANDIDATE_ARCHIVE_SHA256}" ]]; then
  echo "[ERROR] Python runtime overlay archive changed while finalizing the cache: ${ARCHIVE_PATH}" >&2
  exit 2
fi
flock -u 9
exec 9>&-
unset CACHED_ARCHIVE_SHA256 CANDIDATE_ARCHIVE CANDIDATE_ARCHIVE_SHA256 \
  LOCK_FD_FINGERPRINT LOCK_PATH_FINGERPRINT LOCK_PATH_RECHECK PUBLISH_LOCK

printf '%s\t%s\t%s\t%s\n' \
  "${RUNTIME_ID}" "${ARCHIVE_PATH}" "${ARCHIVE_SHA256}" "${MANIFEST_SHA256}"
