#!/usr/bin/env bash
set -euo pipefail

# Source-manifest and archive identity must not depend on the controller's
# collation locale.  In particular, GNU sort orders ASCII case variants
# differently under locales such as en_US.utf8.  Pin every find/sort pipeline
# (and the deterministic tar walk) before discovering any source paths.
export LC_ALL=C

# Build a content-addressed, self-verifying source archive for distributed runs.
# Large datasets and generated robot assets are deliberately excluded: batch_ne.sh
# links those paths to the node-local asset repository after extraction.

usage() {
  cat <<'EOF'
Usage:
  build_run_snapshot.sh [--repo-root PATH] [--cache-root PATH]

Output (single TSV line on stdout):
  <snapshot_id>  <archive_path>  <archive_sha256>  <source_manifest_sha256>

The archive contains the current filesystem contents (including uncommitted and
untracked source files), not merely Git HEAD or the Git index.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)
CACHE_ROOT=${HOLOSOMA_SNAPSHOT_CACHE_ROOT:-${TMPDIR:-/tmp}/holosoma-run-snapshots-${USER:-unknown}}

while (( $# > 0 )); do
  case "$1" in
    --repo-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --repo-root requires a value." >&2; exit 2; }
      REPO_ROOT=$2
      shift 2
      ;;
    --cache-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --cache-root requires a value." >&2; exit 2; }
      CACHE_ROOT=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT=$(cd "${REPO_ROOT}" && pwd -P)
mkdir -p "${CACHE_ROOT}"
CACHE_ROOT=$(cd "${CACHE_ROOT}" && pwd -P)

for required in src scripts pyproject.toml; do
  if [[ ! -e "${REPO_ROOT}/${required}" ]]; then
    echo "[ERROR] Snapshot source is missing required path: ${REPO_ROOT}/${required}" >&2
    exit 2
  fi
done
for command_name in rsync sha256sum tar gzip realpath stat; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[ERROR] Required snapshot command is unavailable: ${command_name}" >&2
    exit 2
  fi
done

BUILD_ROOT=$(mktemp -d "${CACHE_ROOT}/.build.XXXXXXXX")
STAGING_ROOT="${BUILD_ROOT}/root"
mkdir -p "${STAGING_ROOT}/.holosoma_snapshot"
# Snapshot metadata is generated rather than copied from the source tree.  Its
# modes must therefore not inherit the controller's umask: otherwise identical
# source bytes/modes can produce the same snapshot ID but different archive
# SHA256 values on two controllers.  Keep the staging root traversable while
# metadata is being written; generated file and directory modes are
# canonicalized below.
chmod 755 "${STAGING_ROOT}" "${STAGING_ROOT}/.holosoma_snapshot"
cleanup() {
  # The archive contract deliberately seals source directories.  Re-open the
  # private build tree only for best-effort cleanup; this does not affect the
  # already published archive modes.
  chmod -R u+w "${BUILD_ROOT}" 2>/dev/null || true
  rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT

# Root launchers are small and are frequently edited during experiment review.
# Snapshot all of them so wrapper chains cannot fall back to a node-local copy.
while IFS= read -r -d '' source_path; do
  cp -a "${source_path}" "${STAGING_ROOT}/"
done < <(find "${REPO_ROOT}" -maxdepth 1 \
  \( -type f -o -type l \) -name '*.sh' -print0 | sort -z)
for root_file in pyproject.toml conftest.py .gitmodules; do
  if [[ -f "${REPO_ROOT}/${root_file}" ]]; then
    cp -a "${REPO_ROOT}/${root_file}" "${STAGING_ROOT}/${root_file}"
  fi
done

COMMON_EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.py[co]'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
)
rsync -a "${COMMON_EXCLUDES[@]}" "${REPO_ROOT}/scripts/" "${STAGING_ROOT}/scripts/"
rsync -a "${COMMON_EXCLUDES[@]}" \
  --exclude='/holosoma/holosoma/data/' \
  --exclude='/holosoma_retargeting/models/' \
  --exclude='/holosoma_retargeting/demo_data/' \
  --exclude='/holosoma_retargeting/converted_res/' \
  --exclude='/holosoma_retargeting_my/models/' \
  --exclude='/holosoma_retargeting_my/models_gt/' \
  --exclude='/holosoma_retargeting_my/demo_data/' \
  --exclude='/holosoma_retargeting_my/converted_res/' \
  "${REPO_ROOT}/src/" "${STAGING_ROOT}/src/"
if [[ -d "${REPO_ROOT}/tests" ]]; then
  rsync -a "${COMMON_EXCLUDES[@]}" "${REPO_ROOT}/tests/" "${STAGING_ROOT}/tests/"
fi
if [[ -f "${REPO_ROOT}/submodules/defm/defm/model_factory.py" ]]; then
  # The superproject pins this external implementation by gitlink. Include
  # the exact checked-out source/config bytes while excluding its nested Git
  # administrative pointer and separately managed pretrained weights.
  mkdir -p "${STAGING_ROOT}/submodules/defm"
  # `submodules` is a structural parent that rsync does not populate from a
  # source path of its own.  Canonicalize both generated directories to their
  # source modes before rsync: otherwise the parent keeps a controller-umask
  # dependent 0755/0700 mode, which changes source_modes.nul and the snapshot
  # identity for identical source trees.
  chmod --reference="${REPO_ROOT}/submodules" "${STAGING_ROOT}/submodules"
  chmod --reference="${REPO_ROOT}/submodules/defm" \
    "${STAGING_ROOT}/submodules/defm"
  rsync -a "${COMMON_EXCLUDES[@]}" \
    --exclude='/.git' \
    --exclude='/weights/*.pth' \
    "${REPO_ROOT}/submodules/defm/" "${STAGING_ROOT}/submodules/defm/"
fi

# The signed snapshot identity closes regular files, directories, and the
# explicit symlink manifest below.  Device nodes, FIFOs, and sockets would be
# archived without participating in that identity, allowing the same
# SOURCE_SNAPSHOT_ID to name different tar bytes.  Fail before manifesting.
unsupported_entry=$(find "${STAGING_ROOT}" -xdev \
  \( -type b -o -type c -o -type p -o -type s \) -print -quit)
if [[ -n "${unsupported_entry}" ]]; then
  echo "[ERROR] Snapshot source contains an unsupported special filesystem entry: ${unsupported_entry#${STAGING_ROOT}/}" >&2
  exit 2
fi
# rsync intentionally flattens source hardlinks because inode topology is not
# part of the portable source contract.  Assert that invariant in staging so
# a future copy-option change cannot make tar hardlink encoding affect archive
# bytes without changing the snapshot ID.
staged_hardlink=$(find "${STAGING_ROOT}" -xdev -type f -links +1 -print -quit)
if [[ -n "${staged_hardlink}" ]]; then
  echo "[ERROR] Snapshot staging unexpectedly preserved a regular-file hardlink: ${staged_hardlink#${STAGING_ROOT}/}" >&2
  exit 2
fi
unset unsupported_entry staged_hardlink

# These paths are installed as absolute symlinks to the node-local asset repo.
# Keeping the mapping in the signed source manifest makes the runtime layout an
# explicit part of the snapshot contract.
ASSET_LINK_PATHS=(
  data
  src/holosoma/holosoma/data
  src/holosoma_retargeting/models
  src/holosoma_retargeting/demo_data
  src/holosoma_retargeting/converted_res
  src/holosoma_retargeting_my/models
  src/holosoma_retargeting_my/models_gt
  src/holosoma_retargeting_my/demo_data
  src/holosoma_retargeting_my/converted_res
)
: >"${STAGING_ROOT}/.holosoma_snapshot/asset_links.tsv"
for asset_path in "${ASSET_LINK_PATHS[@]}"; do
  if [[ -e "${REPO_ROOT}/${asset_path}" ]]; then
    printf '%s\t%s\n' "${asset_path}" "${asset_path}" \
      >>"${STAGING_ROOT}/.holosoma_snapshot/asset_links.tsv"
  fi
done

# Preserve and authenticate source-tree symlinks without dereferencing them.
while IFS= read -r -d '' link_path; do
  relative_path=${link_path#"${STAGING_ROOT}/"}
  link_target=$(readlink "${link_path}")
  if [[ "${relative_path}" == *$'\n'* || "${relative_path}" == *$'\t'* || "${link_target}" == *$'\n'* || "${link_target}" == *$'\t'* ]]; then
    echo "[ERROR] Snapshot symlink paths/targets cannot contain tabs or newlines: ${relative_path}" >&2
    exit 2
  fi
  if [[ "${link_target}" == /* ]]; then
    echo "[ERROR] Snapshot source symlink must use a repository-internal relative target: ${relative_path} -> ${link_target}" >&2
    exit 2
  fi
  link_parent=$(dirname -- "${relative_path}")
  if ! resolved_link_target=$(realpath -e -- "${STAGING_ROOT}/${link_parent}/${link_target}" 2>/dev/null); then
    echo "[ERROR] Snapshot source symlink target must resolve to an existing staged entry: ${relative_path} -> ${link_target}" >&2
    exit 2
  fi
  case "${resolved_link_target}" in
    "${STAGING_ROOT}"|"${STAGING_ROOT}"/*)
      ;;
    *)
      echo "[ERROR] Snapshot source symlink escapes the authenticated source tree: ${relative_path} -> ${link_target}" >&2
      exit 2
      ;;
  esac
  printf '%s\t%s\n' "${relative_path}" "${link_target}"
done < <(find "${STAGING_ROOT}" -type l -print0 | sort -z) \
  >"${STAGING_ROOT}/.holosoma_snapshot/source_symlinks.tsv"
unset relative_path link_target link_parent resolved_link_target

# Installed source is sealed against ordinary accidental mutation.  This is
# not a security boundary against the owning account (which can chmod its own
# directories), so every reuse/launch still re-verifies the signed closure.
# Canonicalize away write bits from both files and their parent directories
# before archiving and recording modes.  The generated metadata directory
# remains open only until its final files are written below.
find "${STAGING_ROOT}" -type f ! -path '*/.holosoma_snapshot/*' \
  -exec chmod a-w {} +
find "${STAGING_ROOT}" -type d \
  ! -path "${STAGING_ROOT}/.holosoma_snapshot" \
  ! -path "${STAGING_ROOT}/.holosoma_snapshot/*" \
  -exec chmod a-w {} +

# File permission bits are part of the executable source contract too.  A
# content-only cache key could otherwise reuse an older archive after an
# execute/read mode change.  Keep this manifest NUL-delimited so unusual but
# valid source paths cannot make the record ambiguous.  Generated snapshot
# metadata is excluded because it is separately made read-only after install
# and is already authenticated by the content manifest below.
(
  cd "${STAGING_ROOT}"
  {
    find . -type f ! -path './.holosoma_snapshot/*' \
      -printf 'f\t%m\t%p\0'
    find . -type d \
      ! -path './.holosoma_snapshot' \
      ! -path './.holosoma_snapshot/*' \
      -printf 'd\t%m\t%p\0'
    # The directory is still 0755 so the remaining generated files can be
    # written, but its final canonical 0555 mode is part of the signed mode
    # closure and is applied before tar reads the tree.
    printf 'd\t555\t./.holosoma_snapshot\0'
  } | sort -z \
    > .holosoma_snapshot/source_modes.nul
)

(
  cd "${STAGING_ROOT}"
  find . -type f ! -path './.holosoma_snapshot/source_manifest.sha256' -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    > .holosoma_snapshot/source_manifest.sha256
)
SOURCE_MANIFEST_SHA256=$(sha256sum "${STAGING_ROOT}/.holosoma_snapshot/source_manifest.sha256" | awk '{print $1}')
SNAPSHOT_ID="src-${SOURCE_MANIFEST_SHA256}"
printf '%s\n' "${SNAPSHOT_ID}" >"${STAGING_ROOT}/.holosoma_snapshot/id"

# Canonicalize the generated metadata modes after the final file is written.
# These modes are archive semantics (not source-tree semantics), but the
# archive digest must still be deterministic across controller umasks.
find "${STAGING_ROOT}/.holosoma_snapshot" -type f -exec chmod 444 {} +
chmod 555 "${STAGING_ROOT}/.holosoma_snapshot"

ARCHIVE_PATH="${CACHE_ROOT}/${SNAPSHOT_ID}.tar.gz"
CANDIDATE_ARCHIVE="${BUILD_ROOT}/archive.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "${STAGING_ROOT}" -cf - . \
  | gzip -n >"${CANDIDATE_ARCHIVE}"
chmod 444 "${CANDIDATE_ARCHIVE}"
CANDIDATE_ARCHIVE_SHA256=$(sha256sum "${CANDIDATE_ARCHIVE}" | awk '{print $1}')

# A cache filename alone is not authentication: it can be stale, truncated,
# replaced by a symlink, or share an inode with a mutable alias.  Reuse only a
# single-link regular file whose bytes match the deterministic candidate built
# from the current staging tree.  Otherwise atomically repair the cache.
REUSE_CACHED_ARCHIVE=0
if [[ -f "${ARCHIVE_PATH}" && ! -L "${ARCHIVE_PATH}" \
      && "$(stat -c '%h' -- "${ARCHIVE_PATH}")" == 1 ]]; then
  if chmod 444 "${ARCHIVE_PATH}" \
      && CACHED_ARCHIVE_SHA256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}') \
      && [[ "${CACHED_ARCHIVE_SHA256}" == "${CANDIDATE_ARCHIVE_SHA256}" ]]; then
    REUSE_CACHED_ARCHIVE=1
  fi
fi
if (( REUSE_CACHED_ARCHIVE == 1 )); then
  rm -f "${CANDIDATE_ARCHIVE}"
else
  mv -fT "${CANDIDATE_ARCHIVE}" "${ARCHIVE_PATH}"
fi
if [[ ! -f "${ARCHIVE_PATH}" || -L "${ARCHIVE_PATH}" \
      || "$(stat -c '%h' -- "${ARCHIVE_PATH}")" != 1 ]]; then
  echo "[ERROR] Snapshot cache did not produce a single-link regular archive: ${ARCHIVE_PATH}" >&2
  exit 2
fi
ARCHIVE_SHA256=$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')
if [[ "${ARCHIVE_SHA256}" != "${CANDIDATE_ARCHIVE_SHA256}" ]]; then
  echo "[ERROR] Snapshot archive changed while finalizing the cache: ${ARCHIVE_PATH}" >&2
  exit 2
fi
unset CANDIDATE_ARCHIVE CANDIDATE_ARCHIVE_SHA256 CACHED_ARCHIVE_SHA256 REUSE_CACHED_ARCHIVE

echo "[INFO] source_snapshot_id=${SNAPSHOT_ID} archive=${ARCHIVE_PATH}" >&2
printf '%s\t%s\t%s\t%s\n' \
  "${SNAPSHOT_ID}" "${ARCHIVE_PATH}" "${ARCHIVE_SHA256}" "${SOURCE_MANIFEST_SHA256}"
