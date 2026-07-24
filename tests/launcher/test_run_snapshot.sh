#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
TMP_DIR=$(mktemp -d)
cleanup() {
  chmod -R u+w "${TMP_DIR}" 2>/dev/null || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

expect_snapshot_failure() {
  local output_file="$1"
  local expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "snapshot command unexpectedly succeeded: $*"
  fi
  grep -F "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,30p' "${output_file}" >&2
    fail "missing expected snapshot failure: ${expected}"
  }
}

FIXTURE="${TMP_DIR}/repo"
CACHE="${TMP_DIR}/cache"
mkdir -p \
  "${FIXTURE}/src/example" \
  "${FIXTURE}/scripts" \
  "${FIXTURE}/tests" \
  "${FIXTURE}/data" \
  "${FIXTURE}/submodules/defm/defm"
printf '%s\n' 'value = 1' >"${FIXTURE}/src/example/model.py"
printf '%s\n' 'case_variant = "upper"' >"${FIXTURE}/src/example/Alpha.py"
printf '%s\n' 'case_variant = "lower"' >"${FIXTURE}/src/example/alpha.py"
printf '%s\n' '#!/usr/bin/env bash' 'echo tool' >"${FIXTURE}/scripts/tool.sh"
printf '%s\n' '#!/usr/bin/env bash' 'echo launch' >"${FIXTURE}/launch.sh"
printf '%s\n' '[build-system]' >"${FIXTURE}/pyproject.toml"
printf '%s\n' 'large node-local payload' >"${FIXTURE}/data/not_source.bin"
printf '%s\n' 'def build_model(): return None' \
  >"${FIXTURE}/submodules/defm/defm/model_factory.py"
chmod 640 "${FIXTURE}/src/example/model.py"
chmod 750 "${FIXTURE}/scripts/tool.sh"
chmod 751 "${FIXTURE}/scripts"
chmod 751 "${FIXTURE}/submodules"
chmod 750 "${FIXTURE}/submodules/defm"

record_one=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
archive_inode_one=$(stat -c '%i' "${CACHE}"/src-*.tar.gz)
record_two=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_one archive_one archive_sha_one manifest_sha_one <<<"${record_one}"
IFS=$'\t' read -r id_two archive_two archive_sha_two manifest_sha_two <<<"${record_two}"

[[ "${id_one}" =~ ^src-[0-9a-f]{64}$ ]] || fail "invalid snapshot id: ${id_one}"
[[ "${id_one}" == "${id_two}" ]] || fail 'unchanged source produced different snapshot ids'
[[ "${archive_sha_one}" == "${archive_sha_two}" ]] || fail 'unchanged source produced different archives'
[[ "${id_one}" == "src-${manifest_sha_one}" ]] || fail 'snapshot id is not the source-manifest digest'
[[ "${manifest_sha_one}" == "${manifest_sha_two}" ]] || fail 'manifest digest changed without a source change'
[[ -f "${archive_one}" && "${archive_one}" == "${archive_two}" ]] || fail 'content-addressed archive was not reused'
[[ "$(stat -c '%i' "${archive_two}")" == "${archive_inode_one}" ]] ||
  fail 'byte-identical single-link cache entry was replaced instead of reused'
[[ "$(stat -c '%a' "${archive_two}")" == 444 ]] ||
  fail 'cached snapshot archive mode is not canonical read-only 0444'
[[ "$(sha256sum "${archive_two}" | awk '{print $1}')" == "${archive_sha_two}" ]] ||
  fail 'reported archive SHA256 does not authenticate the returned archive path'
[[ "$(tar -xOzf "${archive_two}" ./.holosoma_snapshot/id)" == "${id_two}" ]] ||
  fail 'archive snapshot ID does not match the returned SOURCE_SNAPSHOT_ID'
[[ "$(tar -xOzf "${archive_two}" ./.holosoma_snapshot/source_manifest.sha256 | sha256sum | awk '{print $1}')" == "${manifest_sha_two}" ]] ||
  fail 'archive manifest does not match the returned manifest SHA256'
if tar -tzf "${archive_one}" | grep -E '^\./data/' >/dev/null; then
  fail 'node-local data payload leaked into the source archive'
fi
tar -xOzf "${archive_one}" ./.holosoma_snapshot/asset_links.tsv \
  | grep -F $'data\tdata' >/dev/null \
  || fail 'data asset-link contract is missing from the snapshot'

# A cache filename is not trusted as identity.  Corrupt regular files,
# hardlink aliases, and symlinks must all be atomically replaced by the exact
# deterministic candidate for the current signed source.
corrupt_cache_inode=$(stat -c '%i' "${archive_one}")
chmod u+w "${archive_one}"
printf '%s\n' 'cache corruption' >>"${archive_one}"
record_repaired=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_repaired archive_repaired archive_sha_repaired _ <<<"${record_repaired}"
[[ "${id_repaired}" == "${id_one}" && "${archive_sha_repaired}" == "${archive_sha_one}" ]] ||
  fail 'corrupt cache repair changed the source identity or deterministic archive digest'
[[ "$(sha256sum "${archive_repaired}" | awk '{print $1}')" == "${archive_sha_one}" ]] ||
  fail 'corrupt cache entry was not repaired to the authenticated archive bytes'
[[ "$(stat -c '%i' "${archive_repaired}")" != "${corrupt_cache_inode}" ]] ||
  fail 'corrupt cache inode was reused instead of atomically replaced'

cache_alias="${TMP_DIR}/snapshot-cache-hardlink-alias.tar.gz"
ln "${archive_repaired}" "${cache_alias}"
aliased_inode=$(stat -c '%i' "${archive_repaired}")
record_unlinked=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_unlinked archive_unlinked archive_sha_unlinked _ <<<"${record_unlinked}"
[[ "${id_unlinked}" == "${id_one}" && "${archive_sha_unlinked}" == "${archive_sha_one}" ]] ||
  fail 'hardlinked cache repair changed deterministic snapshot identity'
[[ "$(stat -c '%h' "${archive_unlinked}")" == 1 ]] ||
  fail 'returned cache archive still has a mutable hardlink alias'
[[ "$(stat -c '%i' "${archive_unlinked}")" != "${aliased_inode}" ]] ||
  fail 'hardlinked cache inode was trusted instead of replaced'
chmod u+w "${cache_alias}"
printf '%s\n' 'alias-only corruption' >>"${cache_alias}"
[[ "$(sha256sum "${archive_unlinked}" | awk '{print $1}')" == "${archive_sha_one}" ]] ||
  fail 'mutating a rejected hardlink alias changed the returned cache archive'

rm -f "${archive_unlinked}"
ln -s "${cache_alias}" "${archive_unlinked}"
record_unsymlinked=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_unsymlinked archive_unsymlinked archive_sha_unsymlinked _ <<<"${record_unsymlinked}"
[[ "${id_unsymlinked}" == "${id_one}" && "${archive_sha_unsymlinked}" == "${archive_sha_one}" ]] ||
  fail 'symlinked cache repair changed deterministic snapshot identity'
[[ -f "${archive_unsymlinked}" && ! -L "${archive_unsymlinked}" ]] ||
  fail 'symlinked cache pathname was followed or retained'
[[ "$(stat -c '%h' "${archive_unsymlinked}")" == 1 ]] ||
  fail 'repaired symlink cache pathname is not a single-link regular file'

concurrent_cache="${TMP_DIR}/cache-concurrent"
bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${concurrent_cache}" \
  >"${TMP_DIR}/concurrent-one.tsv" &
concurrent_pid_one=$!
bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${concurrent_cache}" \
  >"${TMP_DIR}/concurrent-two.tsv" &
concurrent_pid_two=$!
wait "${concurrent_pid_one}"
wait "${concurrent_pid_two}"
IFS=$'\t' read -r concurrent_id_one concurrent_archive_one concurrent_sha_one _ \
  <"${TMP_DIR}/concurrent-one.tsv"
IFS=$'\t' read -r concurrent_id_two concurrent_archive_two concurrent_sha_two _ \
  <"${TMP_DIR}/concurrent-two.tsv"
[[ "${concurrent_id_one}" == "${id_one}" && "${concurrent_id_two}" == "${id_one}" \
      && "${concurrent_archive_one}" == "${concurrent_archive_two}" \
      && "${concurrent_sha_one}" == "${archive_sha_one}" \
      && "${concurrent_sha_two}" == "${archive_sha_one}" ]] ||
  fail 'concurrent publishers did not converge on one deterministic cache entry'
[[ "$(stat -c '%h' "${concurrent_archive_one}")" == 1 \
      && "$(sha256sum "${concurrent_archive_one}" | awk '{print $1}')" == "${archive_sha_one}" ]] ||
  fail 'concurrent cache publication left an unauthenticated or aliased archive'

# Generated snapshot metadata must not inherit the controller umask or locale.
# Case-variant fixture paths collate differently in common non-C locales.  Use
# separate caches so this compares independently built archive bytes rather
# than exercising the normal content-addressed cache hit.  The explicit source
# assertion retains coverage on minimal systems that install only C/POSIX.
grep -Fx 'export LC_ALL=C' "${REPO_ROOT}/scripts/build_run_snapshot.sh" >/dev/null ||
  fail 'snapshot builder does not pin locale-independent collation'
alternate_locale=$(locale -a | awk '
  {
    normalized = tolower($0)
    if (normalized !~ /^(c|posix|c\.utf-?8)$/) {
      print
      exit
    }
  }
')
alternate_locale=${alternate_locale:-C}
record_umask_022=$( (umask 022; LC_ALL=C bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-umask-022") )
record_umask_077=$( (umask 077; LC_ALL="${alternate_locale}" bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-umask-077") )
IFS=$'\t' read -r id_umask_022 _ archive_sha_umask_022 _ <<<"${record_umask_022}"
IFS=$'\t' read -r id_umask_077 _ archive_sha_umask_077 _ <<<"${record_umask_077}"
[[ "${id_umask_022}" == "${id_umask_077}" ]] ||
  fail 'controller umask/locale changed the snapshot id'
[[ "${archive_sha_umask_022}" == "${archive_sha_umask_077}" ]] ||
  fail 'controller umask/locale changed deterministic archive bytes'
unset alternate_locale

EXTRACTED="${TMP_DIR}/extracted"
mkdir -p "${EXTRACTED}"
(umask 077; tar -xzf "${archive_one}" -C "${EXTRACTED}" --same-permissions)
(cd "${EXTRACTED}" && sha256sum --quiet -c .holosoma_snapshot/source_manifest.sha256)
[[ "$(stat -c '%a' "${EXTRACTED}/scripts/tool.sh")" == 550 ]] ||
  fail 'signed source mode changed under a restrictive extraction umask'
[[ "$(stat -c '%a' "${EXTRACTED}/src/example/model.py")" == 440 ]] ||
  fail 'regular-file read mode changed under a restrictive extraction umask'
[[ "$(stat -c '%a' "${EXTRACTED}")" == 555 ]] ||
  fail 'snapshot root is not sealed 0555 after extraction'
[[ "$(stat -c '%a' "${EXTRACTED}/scripts")" == 551 ]] ||
  fail 'signed source directory mode changed under a restrictive extraction umask'
[[ "$(stat -c '%a' "${EXTRACTED}/submodules")" == 551 ]] ||
  fail 'generated defm parent mode changed under a restrictive extraction umask'
[[ "$(stat -c '%a' "${EXTRACTED}/.holosoma_snapshot")" == 555 ]] ||
  fail 'snapshot metadata directory is not sealed 0555'
[[ "$(stat -c '%a' "${EXTRACTED}/.holosoma_snapshot/source_modes.nul")" == 444 ]] ||
  fail 'generated authenticated metadata mode changed under a restrictive extraction umask'
tr '\0' '\n' <"${EXTRACTED}/.holosoma_snapshot/source_modes.nul" \
  | grep -Fx $'d\t555\t.' >/dev/null ||
  fail 'sealed snapshot root mode is absent from the signed mode closure'
tr '\0' '\n' <"${EXTRACTED}/.holosoma_snapshot/source_modes.nul" \
  | grep -Fx $'d\t555\t./.holosoma_snapshot' >/dev/null ||
  fail 'sealed metadata directory mode is absent from the signed mode closure'
if rm -f "${EXTRACTED}/src/example/model.py" 2>/dev/null; then
  fail 'sealed source parent allowed ordinary unlink of a signed file'
fi
if printf '%s\n' injected 2>/dev/null >"${EXTRACTED}/scripts/injected.sh"; then
  fail 'sealed source parent allowed ordinary creation of an unsigned file'
fi
if rm -f "${EXTRACTED}/.holosoma_snapshot/id" 2>/dev/null; then
  fail 'sealed metadata parent allowed ordinary unlink of snapshot identity'
fi

# rsync deliberately flattens source hardlink topology.  The portable
# snapshot identity and tar bytes therefore depend only on path/content/mode,
# never on controller-local inode relationships.
ln "${FIXTURE}/src/example/model.py" "${FIXTURE}/src/example/model_alias.py"
record_source_hardlink=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-source-hardlink")
IFS=$'\t' read -r id_source_hardlink archive_source_hardlink sha_source_hardlink _ \
  <<<"${record_source_hardlink}"
hardlink_extract="${TMP_DIR}/hardlink-extract"
mkdir "${hardlink_extract}"
tar -xzf "${archive_source_hardlink}" -C "${hardlink_extract}" --same-permissions
[[ "$(stat -c '%h' "${hardlink_extract}/src/example/model.py")" == 1 \
      && "$(stat -c '%h' "${hardlink_extract}/src/example/model_alias.py")" == 1 ]] ||
  fail 'controller-local source hardlinks leaked into portable tar topology'
cp --preserve=mode "${FIXTURE}/src/example/model_alias.py" "${TMP_DIR}/model_alias.copy"
rm "${FIXTURE}/src/example/model_alias.py"
cp --preserve=mode "${TMP_DIR}/model_alias.copy" "${FIXTURE}/src/example/model_alias.py"
record_source_independent=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-source-independent")
IFS=$'\t' read -r id_source_independent _ sha_source_independent _ \
  <<<"${record_source_independent}"
[[ "${id_source_hardlink}" == "${id_source_independent}" \
      && "${sha_source_hardlink}" == "${sha_source_independent}" ]] ||
  fail 'source hardlink topology changed the portable snapshot identity or tar bytes'
rm "${FIXTURE}/src/example/model_alias.py"

# Source symlinks participate explicitly in identity, must resolve inside the
# staged tree, and must retain their target text in the archive.
cp --preserve=mode "${FIXTURE}/src/example/model.py" "${FIXTURE}/src/example/other.py"
ln -s model.py "${FIXTURE}/src/example/model_link.py"
record_link_model=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-source-links")
IFS=$'\t' read -r id_link_model archive_link_model sha_link_model _ <<<"${record_link_model}"
tar -xOzf "${archive_link_model}" ./.holosoma_snapshot/source_symlinks.tsv \
  | grep -Fx $'src/example/model_link.py\tmodel.py' >/dev/null ||
  fail 'source symlink target is absent from the authenticated symlink manifest'
ln -sfn other.py "${FIXTURE}/src/example/model_link.py"
record_link_other=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-source-links")
IFS=$'\t' read -r id_link_other _ _ _ <<<"${record_link_other}"
[[ "${id_link_other}" != "${id_link_model}" ]] ||
  fail 'changing only source symlink target text did not change snapshot identity'
ln -sfn model.py "${FIXTURE}/src/example/model_link.py"
record_link_restored=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-source-links")
IFS=$'\t' read -r id_link_restored _ sha_link_restored _ <<<"${record_link_restored}"
[[ "${id_link_restored}" == "${id_link_model}" \
      && "${sha_link_restored}" == "${sha_link_model}" ]] ||
  fail 'restoring source symlink target did not restore deterministic snapshot bytes'
rm "${FIXTURE}/src/example/model_link.py" "${FIXTURE}/src/example/other.py"

ln -s /etc/passwd "${FIXTURE}/src/example/absolute_link.py"
expect_snapshot_failure \
  "${TMP_DIR}/absolute-source-link.out" \
  'repository-internal relative target' \
  bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
    --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-bad-absolute-link"
rm "${FIXTURE}/src/example/absolute_link.py"
ln -s missing.py "${FIXTURE}/src/example/dangling_link.py"
expect_snapshot_failure \
  "${TMP_DIR}/dangling-source-link.out" \
  'must resolve to an existing staged entry' \
  bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
    --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-bad-dangling-link"
rm "${FIXTURE}/src/example/dangling_link.py"

# Root wrapper symlinks are launch-chain inputs too; they must not be silently
# omitted by the top-level regular-file scan.
ln -s launch.sh "${FIXTURE}/alias_launch.sh"
record_root_link=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-root-link")
IFS=$'\t' read -r _ archive_root_link _ _ <<<"${record_root_link}"
tar -xOzf "${archive_root_link}" ./.holosoma_snapshot/source_symlinks.tsv \
  | grep -Fx $'alias_launch.sh\tlaunch.sh' >/dev/null ||
  fail 'root launcher symlink was omitted from the signed snapshot closure'
rm "${FIXTURE}/alias_launch.sh"

mkfifo "${FIXTURE}/scripts/unsigned_pipe"
expect_snapshot_failure \
  "${TMP_DIR}/special-source-entry.out" \
  'unsupported special filesystem entry' \
  bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
    --repo-root "${FIXTURE}" --cache-root "${TMP_DIR}/cache-special-entry"
rm "${FIXTURE}/scripts/unsigned_pipe"

chmod 650 "${FIXTURE}/scripts/tool.sh"
record_mode=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_mode archive_mode _ <<<"${record_mode}"
[[ "${id_mode}" != "${id_one}" ]] || fail 'changed source mode reused the old snapshot id'
[[ "${archive_mode}" != "${archive_one}" ]] || fail 'changed source mode reused the old archive'

chmod 750 "${FIXTURE}/scripts/tool.sh"
record_restored_mode=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_restored_mode archive_restored_mode _ <<<"${record_restored_mode}"
[[ "${id_restored_mode}" == "${id_one}" ]] || fail 'restored source mode did not restore snapshot id'
[[ "${archive_restored_mode}" == "${archive_one}" ]] || fail 'restored source mode did not reuse archive'

scripts_dir_mode=$(stat -c '%a' "${FIXTURE}/scripts")
chmod g-x "${FIXTURE}/scripts"
record_dir_mode=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_dir_mode _ <<<"${record_dir_mode}"
[[ "${id_dir_mode}" != "${id_one}" ]] || fail 'changed source directory mode reused the old snapshot id'
chmod "${scripts_dir_mode}" "${FIXTURE}/scripts"

printf '%s\n' 'value = 2' >"${FIXTURE}/src/example/model.py"
record_three=$(bash "${REPO_ROOT}/scripts/build_run_snapshot.sh" \
  --repo-root "${FIXTURE}" --cache-root "${CACHE}")
IFS=$'\t' read -r id_three _ <<<"${record_three}"
[[ "${id_three}" != "${id_one}" ]] || fail 'changed source reused the old snapshot id'

echo '[PASS] content-addressed run snapshot'
