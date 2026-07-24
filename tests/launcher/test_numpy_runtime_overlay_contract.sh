#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
BUILDER="${REPO_ROOT}/scripts/build_numpy_runtime_overlay.sh"
VERIFIER="${REPO_ROOT}/scripts/verify_python_runtime_overlay.py"
INSTALLER="${REPO_ROOT}/scripts/install_python_runtime_overlay.py"
RUNTIME_SCHEMA="${REPO_ROOT}/scripts/python_runtime_schema.py"

TMP_DIR=$(mktemp -d)
cleanup() {
  chmod -R u+w "${TMP_DIR}" 2>/dev/null || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BUILDER_DIAGNOSTICS="${TMP_DIR}/builder-private.log"
: >"${BUILDER_DIAGNOSTICS}"
chmod 600 "${BUILDER_DIAGNOSTICS}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

source "${REPO_ROOT}/scripts/gpu_launch_defaults.sh"
TEST_PYTHON_BIN=${PYTHON_BIN}

assert_archive_contract() {
  local archive=$1
  local expected_sha=$2
  [[ -f "${archive}" && ! -L "${archive}" ]] ||
    fail "overlay archive is not a regular non-symlink file: ${archive}"
  [[ "$(stat -c '%h' -- "${archive}")" == 1 ]] ||
    fail "overlay archive is not single-link: ${archive}"
  [[ "$(stat -c '%a' -- "${archive}")" == 444 ]] ||
    fail "overlay archive is not sealed 0444: ${archive}"
  [[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${expected_sha}" ]] ||
    fail "overlay archive digest does not match its build record"
  gzip -t -- "${archive}" || fail "overlay archive is not valid gzip"
}

assert_record() {
  local runtime_id=$1
  local archive=$2
  local archive_sha=$3
  local manifest_sha=$4
  local output_root=$5
  [[ "${runtime_id}" =~ ^python-runtime-v2-[0-9a-f]{64}$ ]] ||
    fail "invalid runtime ID: ${runtime_id}"
  [[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid archive SHA256 in build record"
  [[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] ||
    fail "invalid manifest SHA256 in build record"
  [[ "${runtime_id}" == "python-runtime-v2-${manifest_sha}" ]] ||
    fail "runtime ID does not close the manifest digest"
  [[ "${archive}" == "${output_root}/${runtime_id}.tar.gz" ]] ||
    fail "archive path does not match its content-addressed runtime ID"
  assert_archive_contract "${archive}" "${archive_sha}"
}

build_overlay() {
  local build_umask=$1
  local output_root=$2
  local build_locale=${3:-C}
  mkdir -p "${output_root}"
  (
    umask "${build_umask}"
    LC_ALL="${build_locale}" PYTHON_BIN="${TEST_PYTHON_BIN}" \
      bash "${BUILDER}" "${output_root}"
  )
}

# Independently stage and archive the same scientific runtime under permissive and
# restrictive controller umasks.  Identity, manifest, archive digest, and the
# complete compressed bytes must all be identical.
CACHE_022="${TMP_DIR}/cache-022"
CACHE_077="${TMP_DIR}/cache-077"
record_022=$(build_overlay 022 "${CACHE_022}" C 2>>"${BUILDER_DIAGNOSTICS}")
record_077=$(build_overlay 077 "${CACHE_077}" en_US.utf8 2>>"${BUILDER_DIAGNOSTICS}")
IFS=$'\t' read -r runtime_022 archive_022 archive_sha_022 manifest_022 extra_022 <<<"${record_022}"
IFS=$'\t' read -r runtime_077 archive_077 archive_sha_077 manifest_077 extra_077 <<<"${record_077}"
[[ -z "${extra_022}" && -z "${extra_077}" ]] || fail "builder emitted a malformed TSV record"
assert_record "${runtime_022}" "${archive_022}" "${archive_sha_022}" "${manifest_022}" "${CACHE_022}"
assert_record "${runtime_077}" "${archive_077}" "${archive_sha_077}" "${manifest_077}" "${CACHE_077}"
[[ "${runtime_022}" == "${runtime_077}" ]] || fail "controller umask changed runtime ID"
[[ "${manifest_022}" == "${manifest_077}" ]] || fail "controller umask changed runtime manifest"
[[ "${archive_sha_022}" == "${archive_sha_077}" ]] || fail "controller umask changed archive digest"
cmp -s -- "${archive_022}" "${archive_077}" || fail "controller umask changed archive bytes"

EXTRACTED="${TMP_DIR}/extracted"
mkdir -p "${EXTRACTED}"
(umask 077; tar -xzf "${archive_077}" -C "${EXTRACTED}" --same-permissions)
SITE_PACKAGES="${EXTRACTED}/site-packages"
[[ "$(stat -c '%a' -- "${SITE_PACKAGES}")" == 555 ]] ||
  fail "archived site-packages root is not canonical 0555"
if find "${SITE_PACKAGES}" -type d ! -perm 0555 -print -quit | grep -q .; then
  fail "overlay contains a directory whose archived mode is not 0555"
fi
if find "${SITE_PACKAGES}" -type f ! -perm 0444 -print -quit | grep -q .; then
  fail "overlay contains a file whose archived mode is not 0444"
fi
if find "${SITE_PACKAGES}" -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
  fail "overlay archive contains a symlink or special entry"
fi
"${TEST_PYTHON_BIN}" -I -S "${VERIFIER}" \
  --site-packages "${SITE_PACKAGES}" \
  --manifest-sha256 "${manifest_077}" \
  --require-distribution-closure >/dev/null
"${TEST_PYTHON_BIN}" - "${SITE_PACKAGES}/.holosoma-runtime-distributions.json" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
assert contract["root_distributions"] == [
    "attrs",
    "numpy",
    "omegaconf",
]
names = {record["canonical_name"] for record in contract["distributions"]}
assert names == {
    "antlr4-python3-runtime",
    "attrs",
    "numpy",
    "omegaconf",
    "pyyaml",
}
PY

# A byte-correct but writable cache entry is repaired in place to 0444 only
# after a fresh private candidate has been generated and authenticated.
chmod 644 "${archive_022}"
[[ "$(stat -c '%a' -- "${archive_022}")" == 644 ]] || fail "could not create writable cache fixture"
writable_record=$(build_overlay 077 "${CACHE_022}" 2>>"${BUILDER_DIAGNOSTICS}")
IFS=$'\t' read -r writable_runtime writable_archive writable_sha writable_manifest writable_extra <<<"${writable_record}"
[[ -z "${writable_extra}" ]] || fail "writable-cache rebuild emitted a malformed record"
[[ "${writable_runtime}" == "${runtime_022}" && "${writable_manifest}" == "${manifest_022}" ]] ||
  fail "writable-cache repair changed runtime identity"
[[ "${writable_sha}" == "${archive_sha_022}" ]] || fail "writable-cache repair changed archive digest"
assert_archive_contract "${writable_archive}" "${archive_sha_022}"

# A tampered single-link regular archive is never trusted by filename.  It is
# replaced atomically with this invocation's complete deterministic candidate.
chmod 644 "${archive_022}"
printf 'tampered-cache-bytes\n' >>"${archive_022}"
chmod 444 "${archive_022}"
[[ "$(sha256sum "${archive_022}" | awk '{print $1}')" != "${archive_sha_022}" ]] ||
  fail "tamper fixture did not change archive bytes"
tamper_record=$(build_overlay 022 "${CACHE_022}" 2>>"${BUILDER_DIAGNOSTICS}")
IFS=$'\t' read -r tamper_runtime tamper_archive tamper_sha tamper_manifest tamper_extra <<<"${tamper_record}"
[[ -z "${tamper_extra}" ]] || fail "tampered-cache rebuild emitted a malformed record"
[[ "${tamper_runtime}" == "${runtime_022}" && "${tamper_manifest}" == "${manifest_022}" ]] ||
  fail "tampered-cache repair changed runtime identity"
[[ "${tamper_sha}" == "${archive_sha_022}" ]] || fail "tampered-cache repair returned the wrong digest"
assert_archive_contract "${tamper_archive}" "${archive_sha_022}"
cmp -s -- "${tamper_archive}" "${archive_077}" || fail "tampered cache was not restored byte-for-byte"

# Aliased cache entries are ambiguous ownership and are rejected without
# following or mutating their target/inode.
SYMLINK_CACHE="${TMP_DIR}/cache-symlink"
mkdir -p "${SYMLINK_CACHE}"
printf 'symlink-target-sentinel\n' >"${TMP_DIR}/symlink-target"
cp "${TMP_DIR}/symlink-target" "${TMP_DIR}/symlink-target.expected"
ln -s "${TMP_DIR}/symlink-target" "${SYMLINK_CACHE}/${runtime_022}.tar.gz"
if build_overlay 022 "${SYMLINK_CACHE}" >"${TMP_DIR}/symlink.out" 2>"${TMP_DIR}/symlink.err"; then
  fail "builder accepted a symlink cache entry"
fi
grep -F 'Refusing symlink Python runtime overlay cache entry' "${TMP_DIR}/symlink.err" >/dev/null ||
  fail "symlink cache rejection was not explicit"
cmp -s "${TMP_DIR}/symlink-target" "${TMP_DIR}/symlink-target.expected" ||
  fail "symlink cache rejection modified its target"
[[ -L "${SYMLINK_CACHE}/${runtime_022}.tar.gz" ]] || fail "symlink cache fixture was replaced"

HARDLINK_CACHE="${TMP_DIR}/cache-hardlink"
mkdir -p "${HARDLINK_CACHE}"
cp "${archive_077}" "${TMP_DIR}/hardlink-alias"
ln "${TMP_DIR}/hardlink-alias" "${HARDLINK_CACHE}/${runtime_022}.tar.gz"
hardlink_sha_before=$(sha256sum "${TMP_DIR}/hardlink-alias" | awk '{print $1}')
if build_overlay 077 "${HARDLINK_CACHE}" >"${TMP_DIR}/hardlink.out" 2>"${TMP_DIR}/hardlink.err"; then
  fail "builder accepted a multiply-linked cache entry"
fi
grep -F 'Refusing multiply-linked Python runtime overlay cache entry' "${TMP_DIR}/hardlink.err" >/dev/null ||
  fail "hardlink cache rejection was not explicit"
[[ "$(stat -c '%h' -- "${TMP_DIR}/hardlink-alias")" == 2 ]] ||
  fail "hardlink cache rejection changed link topology"
[[ "$(sha256sum "${TMP_DIR}/hardlink-alias" | awk '{print $1}')" == "${hardlink_sha_before}" ]] ||
  fail "hardlink cache rejection modified aliased bytes"

# The publication lock itself is also an ownership object.  Reject symlink or
# multiply-linked lock paths without following, truncating, or chmodding their
# aliased targets.
LOCK_SYMLINK_CACHE="${TMP_DIR}/cache-lock-symlink"
mkdir -p "${LOCK_SYMLINK_CACHE}"
printf 'lock-symlink-target-sentinel\n' >"${TMP_DIR}/lock-symlink-target"
cp "${TMP_DIR}/lock-symlink-target" "${TMP_DIR}/lock-symlink-target.expected"
chmod 600 "${TMP_DIR}/lock-symlink-target" "${TMP_DIR}/lock-symlink-target.expected"
ln -s "${TMP_DIR}/lock-symlink-target" \
  "${LOCK_SYMLINK_CACHE}/.${runtime_022}.publish.lock"
if build_overlay 022 "${LOCK_SYMLINK_CACHE}" \
    >"${TMP_DIR}/lock-symlink.out" 2>"${TMP_DIR}/lock-symlink.err"; then
  fail "builder accepted a symlink publication lock"
fi
grep -F 'Refusing aliased or malformed Python runtime overlay publish lock' \
  "${TMP_DIR}/lock-symlink.err" >/dev/null ||
  fail "symlink publication-lock rejection was not explicit"
cmp -s "${TMP_DIR}/lock-symlink-target" "${TMP_DIR}/lock-symlink-target.expected" ||
  fail "symlink publication-lock rejection followed or truncated its target"

LOCK_HARDLINK_CACHE="${TMP_DIR}/cache-lock-hardlink"
mkdir -p "${LOCK_HARDLINK_CACHE}"
printf 'lock-hardlink-target-sentinel\n' >"${TMP_DIR}/lock-hardlink-target"
chmod 600 "${TMP_DIR}/lock-hardlink-target"
cp "${TMP_DIR}/lock-hardlink-target" "${TMP_DIR}/lock-hardlink-target.expected"
ln "${TMP_DIR}/lock-hardlink-target" \
  "${LOCK_HARDLINK_CACHE}/.${runtime_022}.publish.lock"
if build_overlay 077 "${LOCK_HARDLINK_CACHE}" \
    >"${TMP_DIR}/lock-hardlink.out" 2>"${TMP_DIR}/lock-hardlink.err"; then
  fail "builder accepted a multiply-linked publication lock"
fi
grep -F 'Refusing aliased or malformed Python runtime overlay publish lock' \
  "${TMP_DIR}/lock-hardlink.err" >/dev/null ||
  fail "hardlink publication-lock rejection was not explicit"
[[ "$(stat -c '%h' -- "${TMP_DIR}/lock-hardlink-target")" == 2 ]] ||
  fail "hardlink publication-lock rejection changed link topology"
cmp -s "${TMP_DIR}/lock-hardlink-target" "${TMP_DIR}/lock-hardlink-target.expected" ||
  fail "hardlink publication-lock rejection truncated or changed aliased bytes"

# Concurrent cooperating publishers build private candidates and serialize a
# single atomic publication.  A polling reader must observe either absence or
# the complete sealed expected archive, never a partial gzip/member stream.
CONCURRENT_CACHE="${TMP_DIR}/cache-concurrent"
mkdir -p "${CONCURRENT_CACHE}"
CONCURRENT_ARCHIVE="${CONCURRENT_CACHE}/${runtime_022}.tar.gz"
MONITOR_DONE="${TMP_DIR}/monitor.done"
MONITOR_BAD="${TMP_DIR}/monitor.bad"
(
  while [[ ! -e "${MONITOR_DONE}" ]]; do
    if [[ -e "${CONCURRENT_ARCHIVE}" || -L "${CONCURRENT_ARCHIVE}" ]]; then
      if [[ ! -f "${CONCURRENT_ARCHIVE}" || -L "${CONCURRENT_ARCHIVE}" \
            || "$(stat -c '%h' -- "${CONCURRENT_ARCHIVE}" 2>/dev/null || true)" != 1 \
            || "$(stat -c '%a' -- "${CONCURRENT_ARCHIVE}" 2>/dev/null || true)" != 444 \
            || "$(sha256sum "${CONCURRENT_ARCHIVE}" 2>/dev/null | awk '{print $1}')" != "${archive_sha_022}" ]] \
          || ! gzip -t -- "${CONCURRENT_ARCHIVE}" 2>/dev/null; then
        printf 'reader observed a partial or unauthenticated publication\n' >"${MONITOR_BAD}"
        exit 1
      fi
    fi
    sleep 0.01
  done
) &
monitor_pid=$!

publisher_pids=()
for publisher in 1 2 3 4; do
  publisher_umask=022
  (( publisher % 2 == 0 )) && publisher_umask=077
  (
    umask "${publisher_umask}"
    PYTHON_BIN="${TEST_PYTHON_BIN}" bash "${BUILDER}" "${CONCURRENT_CACHE}" \
      >"${TMP_DIR}/publisher-${publisher}.out" \
      2>"${TMP_DIR}/publisher-${publisher}.err"
  ) &
  publisher_pids+=("$!")
done

publisher_failed=0
for publisher_pid in "${publisher_pids[@]}"; do
  if ! wait "${publisher_pid}"; then
    publisher_failed=1
  fi
done
: >"${MONITOR_DONE}"
if ! wait "${monitor_pid}"; then
  publisher_failed=1
fi
(( publisher_failed == 0 )) || fail "one or more concurrent publishers/readers failed"
[[ ! -e "${MONITOR_BAD}" ]] || fail "$(<"${MONITOR_BAD}")"

for publisher in 1 2 3 4; do
  cmp -s "${TMP_DIR}/publisher-1.out" "${TMP_DIR}/publisher-${publisher}.out" ||
    fail "concurrent publishers returned different build records"
done
concurrent_record=$(<"${TMP_DIR}/publisher-1.out")
IFS=$'\t' read -r concurrent_runtime concurrent_archive concurrent_sha concurrent_manifest concurrent_extra <<<"${concurrent_record}"
[[ -z "${concurrent_extra}" ]] || fail "concurrent publisher emitted a malformed record"
[[ "${concurrent_runtime}" == "${runtime_022}" && "${concurrent_manifest}" == "${manifest_022}" ]] ||
  fail "concurrent publication changed runtime identity"
[[ "${concurrent_sha}" == "${archive_sha_022}" ]] || fail "concurrent publication changed archive digest"
assert_record "${concurrent_runtime}" "${concurrent_archive}" "${concurrent_sha}" "${concurrent_manifest}" "${CONCURRENT_CACHE}"
[[ "$(find "${CONCURRENT_CACHE}" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)" == 1 ]] ||
  fail "concurrent publication left multiple visible archives"
if find "${CONCURRENT_CACHE}" -maxdepth 1 -name '.build.*' -print -quit | grep -q .; then
  fail "concurrent publication leaked a private build directory"
fi

# Exercise the node-local installer against the real scientific archive built
# above.  Keep every installer diagnostic private: successful messages contain
# content IDs, while assertions below intentionally report only semantic roles.
INSTALLER_TOOLS="${TMP_DIR}/installer-tools"
mkdir -m 700 "${INSTALLER_TOOLS}"
cp "${VERIFIER}" "${RUNTIME_SCHEMA}" "${INSTALLER_TOOLS}/"
chmod 444 \
  "${INSTALLER_TOOLS}/$(basename "${VERIFIER}")" \
  "${INSTALLER_TOOLS}/$(basename "${RUNTIME_SCHEMA}")"
SEALED_VERIFIER="${INSTALLER_TOOLS}/$(basename "${VERIFIER}")"

make_installer_root() {
  local runtime_root=$1
  mkdir -m 700 "${runtime_root}"
  mkdir -m 700 "${runtime_root}/.locks" "${runtime_root}/.incoming"
}

stage_installer_archive() {
  local runtime_root=$1
  local token=$2
  local source_archive=$3
  local expected_archive_sha=$4
  local transfer_root="${runtime_root}/.incoming/${token}"
  local destination="${transfer_root}/${runtime_022}.${expected_archive_sha}.tar.gz"
  [[ "${token}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid installer fixture token"
  mkdir -m 700 "${transfer_root}"
  cp "${source_archive}" "${destination}"
  chmod 400 "${destination}"
  printf '%s\n' "${destination}"
}

capture_installer_publish() {
  local output=$1
  local runtime_root=$2
  local staged_archive=$3
  local expected_archive_sha=$4
  local lock_timeout=${5:-60}
  : >"${output}"
  chmod 600 "${output}"
  "${TEST_PYTHON_BIN}" -I -S "${INSTALLER}" \
    --runtime-root "${runtime_root}" \
    --manifest-sha256 "${manifest_022}" \
    --verifier "${SEALED_VERIFIER}" \
    --archive "${staged_archive}" \
    --archive-sha256 "${expected_archive_sha}" \
    --lock-timeout-seconds "${lock_timeout}" \
    >"${output}" 2>&1
}

capture_installer_probe() {
  local output=$1
  local runtime_root=$2
  local lock_timeout=${3:-60}
  : >"${output}"
  chmod 600 "${output}"
  "${TEST_PYTHON_BIN}" -I -S "${INSTALLER}" \
    --runtime-root "${runtime_root}" \
    --manifest-sha256 "${manifest_022}" \
    --verifier "${SEALED_VERIFIER}" \
    --lock-timeout-seconds "${lock_timeout}" \
    --probe-only \
    >"${output}" 2>&1
}

assert_no_installer_scratch() {
  local runtime_root=$1
  if find "${runtime_root}" -maxdepth 1 -name '.*.candidate.*' -print -quit | grep -q .; then
    fail "installer leaked a private candidate"
  fi
  if find "${runtime_root}/.incoming" -mindepth 1 -print -quit | grep -q .; then
    fail "installer leaked an authenticated transfer"
  fi
}

# Fresh publication, locked probe, and an idempotent re-publication all verify
# the exact tree.  Both publication paths consume their authenticated transfer.
FRESH_RUNTIME_ROOT="${TMP_DIR}/installer-fresh"
make_installer_root "${FRESH_RUNTIME_ROOT}"
fresh_archive=$(stage_installer_archive \
  "${FRESH_RUNTIME_ROOT}" "$(printf '1%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
capture_installer_publish \
  "${TMP_DIR}/installer-fresh.out" "${FRESH_RUNTIME_ROOT}" \
  "${fresh_archive}" "${archive_sha_077}" || fail "fresh installer publication failed"
grep -F 'installed_verified_python_runtime=' "${TMP_DIR}/installer-fresh.out" >/dev/null ||
  fail "fresh installer publication did not report its verified outcome"
FRESH_FINAL="${FRESH_RUNTIME_ROOT}/${runtime_022}"
[[ -d "${FRESH_FINAL}" && ! -L "${FRESH_FINAL}" \
      && "$(stat -c '%a' -- "${FRESH_FINAL}")" == 555 ]] ||
  fail "fresh installer publication did not create a sealed runtime root"
[[ "$(stat -c '%a' -- "${FRESH_FINAL}/site-packages")" == 555 ]] ||
  fail "fresh installer publication did not seal site-packages"
assert_no_installer_scratch "${FRESH_RUNTIME_ROOT}"

capture_installer_probe \
  "${TMP_DIR}/installer-probe.out" "${FRESH_RUNTIME_ROOT}" ||
  fail "installer probe rejected a verified publication"
grep -F 'reused_verified_python_runtime=' "${TMP_DIR}/installer-probe.out" >/dev/null ||
  fail "installer probe did not report reuse"

reuse_archive=$(stage_installer_archive \
  "${FRESH_RUNTIME_ROOT}" "$(printf '2%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
capture_installer_publish \
  "${TMP_DIR}/installer-reuse.out" "${FRESH_RUNTIME_ROOT}" \
  "${reuse_archive}" "${archive_sha_077}" || fail "installer reuse failed"
grep -F 'reused_verified_python_runtime=' "${TMP_DIR}/installer-reuse.out" >/dev/null ||
  fail "installer re-publication did not reuse the verified runtime"
assert_no_installer_scratch "${FRESH_RUNTIME_ROOT}"

# Cooperating installers may extract concurrently, but the content lock must
# serialize one installed result and one verified reuse without scratch leaks.
INSTALLER_CONCURRENT_ROOT="${TMP_DIR}/installer-concurrent"
make_installer_root "${INSTALLER_CONCURRENT_ROOT}"
concurrent_archive_one=$(stage_installer_archive \
  "${INSTALLER_CONCURRENT_ROOT}" "$(printf '3%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
concurrent_archive_two=$(stage_installer_archive \
  "${INSTALLER_CONCURRENT_ROOT}" "$(printf '4%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
capture_installer_publish \
  "${TMP_DIR}/installer-concurrent-one.out" "${INSTALLER_CONCURRENT_ROOT}" \
  "${concurrent_archive_one}" "${archive_sha_077}" &
installer_pid_one=$!
capture_installer_publish \
  "${TMP_DIR}/installer-concurrent-two.out" "${INSTALLER_CONCURRENT_ROOT}" \
  "${concurrent_archive_two}" "${archive_sha_077}" &
installer_pid_two=$!
installer_concurrent_failed=0
if ! wait "${installer_pid_one}"; then
  installer_concurrent_failed=1
fi
if ! wait "${installer_pid_two}"; then
  installer_concurrent_failed=1
fi
(( installer_concurrent_failed == 0 )) || fail "concurrent installers failed"
installer_installed_count=$(awk \
  '/installed_verified_python_runtime=/{count += 1} END {print count + 0}' \
  "${TMP_DIR}/installer-concurrent-one.out" \
  "${TMP_DIR}/installer-concurrent-two.out")
installer_reused_count=$(awk \
  '/reused_verified_python_runtime=/{count += 1} END {print count + 0}' \
  "${TMP_DIR}/installer-concurrent-one.out" \
  "${TMP_DIR}/installer-concurrent-two.out")
[[ "${installer_installed_count}" == 1 && "${installer_reused_count}" == 1 ]] ||
  fail "concurrent installers did not produce one install and one reuse"
[[ -d "${INSTALLER_CONCURRENT_ROOT}/${runtime_022}" ]] ||
  fail "concurrent installers did not publish the runtime"
assert_no_installer_scratch "${INSTALLER_CONCURRENT_ROOT}"

# A corrupt content-addressed final is terminal: probe and re-publication both
# fail closed, the good candidate never replaces it, and all bound scratch is
# removed.
corrupt_manifest="${FRESH_FINAL}/site-packages/.holosoma-runtime-manifest.sha256"
chmod 644 "${corrupt_manifest}"
printf '\ncorrupt-final-fixture\n' >>"${corrupt_manifest}"
chmod 444 "${corrupt_manifest}"
corrupt_digest_before=$(sha256sum "${corrupt_manifest}" | awk '{print $1}')
if capture_installer_probe \
    "${TMP_DIR}/installer-corrupt-probe.out" "${FRESH_RUNTIME_ROOT}"; then
  fail "installer probe accepted a corrupt final"
else
  corrupt_probe_status=$?
fi
[[ "${corrupt_probe_status}" == 2 ]] ||
  fail "corrupt-final probe returned the wrong failure status"
grep -F 'failed strict verification' "${TMP_DIR}/installer-corrupt-probe.out" >/dev/null ||
  fail "corrupt-final probe did not fail through strict verification"
corrupt_republish_archive=$(stage_installer_archive \
  "${FRESH_RUNTIME_ROOT}" "$(printf '5%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
if capture_installer_publish \
    "${TMP_DIR}/installer-corrupt-republish.out" "${FRESH_RUNTIME_ROOT}" \
    "${corrupt_republish_archive}" "${archive_sha_077}"; then
  fail "installer replaced a corrupt content-addressed final"
else
  corrupt_republish_status=$?
fi
[[ "${corrupt_republish_status}" == 2 ]] ||
  fail "corrupt-final re-publication returned the wrong failure status"
grep -F 'failed strict verification' \
  "${TMP_DIR}/installer-corrupt-republish.out" >/dev/null ||
  fail "corrupt-final re-publication did not fail through strict verification"
corrupt_digest_after=$(sha256sum "${corrupt_manifest}" | awk '{print $1}')
[[ "${corrupt_digest_after}" == "${corrupt_digest_before}" ]] ||
  fail "corrupt-final rejection overwrote the existing final"
assert_no_installer_scratch "${FRESH_RUNTIME_ROOT}"

# Validate tar members before writing any payload.  Traversal, soft links, hard
# links, and writable files must all be rejected as structure, not merely by the
# later exact-tree verifier.
MALICIOUS_ARCHIVES="${TMP_DIR}/malicious-archives"
mkdir -m 700 "${MALICIOUS_ARCHIVES}"
"${TEST_PYTHON_BIN}" -I -S - "${MALICIOUS_ARCHIVES}" <<'PY'
import io
from pathlib import Path
import sys
import tarfile

output_root = Path(sys.argv[1])
for case in ("traversal", "symlink", "hardlink", "mode"):
    with tarfile.open(output_root / f"{case}.tar.gz", "w:gz") as archive:
        root = tarfile.TarInfo("site-packages")
        root.type = tarfile.DIRTYPE
        root.mode = 0o555
        root.uid = root.gid = 0
        root.mtime = 0
        archive.addfile(root)
        if case == "traversal":
            member = tarfile.TarInfo("site-packages/../escape")
            member.mode = 0o444
            member.uid = member.gid = 0
            member.mtime = 0
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
        elif case == "symlink":
            member = tarfile.TarInfo("site-packages/link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/escape"
            member.mode = 0o777
            member.uid = member.gid = 0
            member.mtime = 0
            archive.addfile(member)
        elif case == "hardlink":
            member = tarfile.TarInfo("site-packages/hard")
            member.type = tarfile.LNKTYPE
            member.linkname = "site-packages/target"
            member.mode = 0o444
            member.uid = member.gid = 0
            member.mtime = 0
            archive.addfile(member)
        else:
            member = tarfile.TarInfo("site-packages/writable")
            member.mode = 0o644
            member.uid = member.gid = 0
            member.mtime = 0
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
PY

MALICIOUS_RUNTIME_ROOT="${TMP_DIR}/installer-malicious"
make_installer_root "${MALICIOUS_RUNTIME_ROOT}"
for malicious_case in traversal symlink hardlink mode; do
  case "${malicious_case}" in
    traversal) malicious_token=$(printf 'a%.0s' {1..64}) ;;
    symlink) malicious_token=$(printf 'b%.0s' {1..64}) ;;
    hardlink) malicious_token=$(printf 'c%.0s' {1..64}) ;;
    mode) malicious_token=$(printf 'd%.0s' {1..64}) ;;
  esac
  malicious_source="${MALICIOUS_ARCHIVES}/${malicious_case}.tar.gz"
  malicious_sha=$(sha256sum "${malicious_source}" | awk '{print $1}')
  malicious_staged=$(stage_installer_archive \
    "${MALICIOUS_RUNTIME_ROOT}" "${malicious_token}" \
    "${malicious_source}" "${malicious_sha}")
  if capture_installer_publish \
      "${TMP_DIR}/installer-malicious-${malicious_case}.out" \
      "${MALICIOUS_RUNTIME_ROOT}" "${malicious_staged}" "${malicious_sha}"; then
    fail "installer accepted a malicious archive member"
  else
    malicious_status=$?
  fi
  [[ "${malicious_status}" == 2 ]] ||
    fail "malicious archive rejection returned the wrong failure status"
  case "${malicious_case}" in
    traversal)
      grep -F 'member is non-canonical' \
        "${TMP_DIR}/installer-malicious-${malicious_case}.out" >/dev/null ||
        fail "path traversal was not rejected as non-canonical"
      ;;
    symlink|hardlink)
      grep -F 'link or special member' \
        "${TMP_DIR}/installer-malicious-${malicious_case}.out" >/dev/null ||
        fail "archive link was not rejected structurally"
      ;;
    mode)
      grep -F 'file has non-canonical metadata' \
        "${TMP_DIR}/installer-malicious-${malicious_case}.out" >/dev/null ||
        fail "writable archive member was not rejected"
      ;;
  esac
  assert_no_installer_scratch "${MALICIOUS_RUNTIME_ROOT}"
done

# Digest failure occurs before archive authentication.  The installer must not
# infer deletion authority from a caller-provided path; the token-bound caller
# retains and then explicitly removes that untrusted transfer.
WRONG_DIGEST_ROOT="${TMP_DIR}/installer-wrong-digest"
make_installer_root "${WRONG_DIGEST_ROOT}"
wrong_archive_sha=$(printf '0%.0s' {1..64})
[[ "${wrong_archive_sha}" != "${archive_sha_077}" ]] ||
  fail "wrong-digest fixture unexpectedly matched the archive"
wrong_digest_archive=$(stage_installer_archive \
  "${WRONG_DIGEST_ROOT}" "$(printf '6%.0s' {1..64})" \
  "${archive_077}" "${wrong_archive_sha}")
wrong_archive_fingerprint=$(stat -c '%d:%i:%h:%u:%a:%s' -- "${wrong_digest_archive}")
if capture_installer_publish \
    "${TMP_DIR}/installer-wrong-digest.out" "${WRONG_DIGEST_ROOT}" \
    "${wrong_digest_archive}" "${wrong_archive_sha}"; then
  fail "installer accepted an archive with the wrong digest"
else
  wrong_digest_status=$?
fi
[[ "${wrong_digest_status}" == 2 ]] ||
  fail "wrong archive digest returned the wrong failure status"
grep -F 'runtime archive SHA256 mismatch' \
  "${TMP_DIR}/installer-wrong-digest.out" >/dev/null ||
  fail "wrong archive digest did not produce the expected rejection"
[[ -f "${wrong_digest_archive}" && ! -L "${wrong_digest_archive}" \
      && "$(stat -c '%d:%i:%h:%u:%a:%s' -- "${wrong_digest_archive}")" \
        == "${wrong_archive_fingerprint}" ]] ||
  fail "installer removed or changed an unauthenticated archive"
if find "${WRONG_DIGEST_ROOT}" -maxdepth 1 -name '.*.candidate.*' -print -quit | grep -q .; then
  fail "wrong-digest rejection leaked a candidate"
fi
chmod 600 "${wrong_digest_archive}"
rm -f "${wrong_digest_archive}"
rmdir "$(dirname "${wrong_digest_archive}")"
assert_no_installer_scratch "${WRONG_DIGEST_ROOT}"

# Invalid arguments and out-of-namespace paths never authorize cleanup of an
# unrelated current-UID file or its parent directory.
OUT_OF_NAMESPACE_ROOT="${TMP_DIR}/installer-out-of-namespace"
make_installer_root "${OUT_OF_NAMESPACE_ROOT}"
UNRELATED_ROOT="${TMP_DIR}/unrelated-archive-parent"
mkdir -m 700 "${UNRELATED_ROOT}"
unrelated_archive="${UNRELATED_ROOT}/victim.tar.gz"
printf 'unrelated-archive-sentinel\n' >"${unrelated_archive}"
cp "${unrelated_archive}" "${TMP_DIR}/unrelated-archive.expected"
chmod 400 "${unrelated_archive}"
unrelated_sha=$(sha256sum "${unrelated_archive}" | awk '{print $1}')
unrelated_fingerprint=$(stat -c '%d:%i:%h:%u:%a:%s' -- "${unrelated_archive}")
: >"${TMP_DIR}/installer-invalid-argument.out"
chmod 600 "${TMP_DIR}/installer-invalid-argument.out"
if "${TEST_PYTHON_BIN}" -I -S "${INSTALLER}" \
    --runtime-root "${OUT_OF_NAMESPACE_ROOT}" \
    --manifest-sha256 invalid \
    --verifier "${SEALED_VERIFIER}" \
    --archive "${unrelated_archive}" \
    --archive-sha256 "${unrelated_sha}" \
    >"${TMP_DIR}/installer-invalid-argument.out" 2>&1; then
  fail "installer accepted an invalid manifest digest"
else
  invalid_argument_status=$?
fi
[[ "${invalid_argument_status}" == 2 ]] ||
  fail "invalid installer argument returned the wrong failure status"
[[ -d "${UNRELATED_ROOT}" && -f "${unrelated_archive}" \
      && "$(stat -c '%d:%i:%h:%u:%a:%s' -- "${unrelated_archive}")" \
        == "${unrelated_fingerprint}" ]] ||
  fail "invalid installer arguments deleted an unrelated archive or parent"
: >"${TMP_DIR}/installer-out-of-namespace.out"
chmod 600 "${TMP_DIR}/installer-out-of-namespace.out"
if "${TEST_PYTHON_BIN}" -I -S "${INSTALLER}" \
    --runtime-root "${OUT_OF_NAMESPACE_ROOT}" \
    --manifest-sha256 "${manifest_022}" \
    --verifier "${SEALED_VERIFIER}" \
    --archive "${unrelated_archive}" \
    --archive-sha256 "${unrelated_sha}" \
    >"${TMP_DIR}/installer-out-of-namespace.out" 2>&1; then
  fail "installer accepted an out-of-namespace archive"
else
  out_of_namespace_status=$?
fi
[[ "${out_of_namespace_status}" == 2 ]] ||
  fail "out-of-namespace archive returned the wrong failure status"
grep -F 'not inside one token-bound incoming directory' \
  "${TMP_DIR}/installer-out-of-namespace.out" >/dev/null ||
  fail "out-of-namespace archive rejection was not explicit"
[[ -d "${UNRELATED_ROOT}" && -f "${unrelated_archive}" \
      && "$(stat -c '%d:%i:%h:%u:%a:%s' -- "${unrelated_archive}")" \
        == "${unrelated_fingerprint}" ]] ||
  fail "out-of-namespace rejection deleted an unrelated archive or parent"
cmp -s "${unrelated_archive}" "${TMP_DIR}/unrelated-archive.expected" ||
  fail "out-of-namespace rejection changed unrelated archive bytes"
assert_no_installer_scratch "${OUT_OF_NAMESPACE_ROOT}"

# Probe distinguishes a genuinely absent runtime (rc=3) from a malformed final
# (rc=2), allowing batch prepare to publish only for the former.
MISSING_RUNTIME_ROOT="${TMP_DIR}/installer-missing"
make_installer_root "${MISSING_RUNTIME_ROOT}"
if capture_installer_probe \
    "${TMP_DIR}/installer-missing.out" "${MISSING_RUNTIME_ROOT}"; then
  fail "installer probe accepted a missing runtime"
else
  missing_probe_status=$?
fi
[[ "${missing_probe_status}" == 3 ]] ||
  fail "missing installer probe did not return its dedicated status"
grep -F '[MISSING]' "${TMP_DIR}/installer-missing.out" >/dev/null ||
  fail "missing installer probe did not report a missing runtime"
assert_no_installer_scratch "${MISSING_RUNTIME_ROOT}"

# Interrupted private candidates are transaction scratch, not publications.
# Prove that probe cannot reap one before acquiring the per-runtime lock, then
# releases it only after removing a strictly named real directory and reporting
# the still-missing final with the dedicated rc=3 status.
STALE_RUNTIME_ROOT="${TMP_DIR}/installer-stale-candidate"
make_installer_root "${STALE_RUNTIME_ROOT}"
stale_candidate="${STALE_RUNTIME_ROOT}/.${runtime_022}.candidate.stale-transaction"
mkdir -m 700 "${stale_candidate}"
mkdir -m 700 "${stale_candidate}/site-packages"
printf 'interrupted-candidate-sentinel\n' \
  >"${stale_candidate}/site-packages/payload"
chmod 444 "${stale_candidate}/site-packages/payload"
chmod 555 "${stale_candidate}/site-packages" "${stale_candidate}"
stale_lock_path="${STALE_RUNTIME_ROOT}/.locks/${runtime_022}.lock"
: >"${stale_lock_path}"
chmod 600 "${stale_lock_path}"
exec {stale_lock_fd}<>"${stale_lock_path}"
flock -x "${stale_lock_fd}"
if capture_installer_probe \
    "${TMP_DIR}/installer-stale-blocked.out" "${STALE_RUNTIME_ROOT}" 1; then
  flock -u "${stale_lock_fd}"
  exec {stale_lock_fd}>&-
  fail "probe bypassed the held runtime lock"
else
  stale_blocked_status=$?
fi
[[ "${stale_blocked_status}" == 2 ]] || {
  flock -u "${stale_lock_fd}"
  exec {stale_lock_fd}>&-
  fail "blocked stale-candidate probe returned the wrong status"
}
grep -F 'timed out acquiring runtime install lock' \
  "${TMP_DIR}/installer-stale-blocked.out" >/dev/null || {
  flock -u "${stale_lock_fd}"
  exec {stale_lock_fd}>&-
  fail "blocked stale-candidate probe did not time out on its lock"
}
[[ -d "${stale_candidate}" && ! -L "${stale_candidate}" ]] || {
  flock -u "${stale_lock_fd}"
  exec {stale_lock_fd}>&-
  fail "probe reaped stale candidate without owning the runtime lock"
}
flock -u "${stale_lock_fd}"
exec {stale_lock_fd}>&-
if capture_installer_probe \
    "${TMP_DIR}/installer-stale-reap.out" "${STALE_RUNTIME_ROOT}"; then
  fail "stale-candidate probe accepted a missing final"
else
  stale_reap_status=$?
fi
[[ "${stale_reap_status}" == 3 ]] ||
  fail "stale-candidate probe did not return missing status"
grep -F '[MISSING]' "${TMP_DIR}/installer-stale-reap.out" >/dev/null ||
  fail "stale-candidate probe did not report the missing final"
[[ ! -e "${stale_candidate}" && ! -L "${stale_candidate}" ]] ||
  fail "locked probe did not reap its strict stale candidate"
assert_no_installer_scratch "${STALE_RUNTIME_ROOT}"

# A lookalike symlink or non-directory has no stale-transaction deletion
# authority.  Probe must reject each as malformed and leave both the alias
# target and the lookalike object byte-for-byte untouched.
STALE_SYMLINK_ROOT="${TMP_DIR}/installer-stale-symlink"
make_installer_root "${STALE_SYMLINK_ROOT}"
STALE_SYMLINK_TARGET="${TMP_DIR}/stale-symlink-target"
mkdir -m 700 "${STALE_SYMLINK_TARGET}"
printf 'stale-symlink-target-sentinel\n' >"${STALE_SYMLINK_TARGET}/payload"
cp "${STALE_SYMLINK_TARGET}/payload" "${TMP_DIR}/stale-symlink-target.expected"
stale_symlink="${STALE_SYMLINK_ROOT}/.${runtime_022}.candidate.alias"
ln -s "${STALE_SYMLINK_TARGET}" "${stale_symlink}"
if capture_installer_probe \
    "${TMP_DIR}/installer-stale-symlink.out" "${STALE_SYMLINK_ROOT}"; then
  fail "probe accepted a symlink stale-candidate lookalike"
else
  stale_symlink_status=$?
fi
[[ "${stale_symlink_status}" == 2 ]] ||
  fail "symlink stale-candidate rejection returned the wrong status"
grep -F 'stale Python runtime candidate is aliased or malformed' \
  "${TMP_DIR}/installer-stale-symlink.out" >/dev/null ||
  fail "symlink stale-candidate rejection was not explicit"
[[ -L "${stale_symlink}" && -d "${STALE_SYMLINK_TARGET}" ]] ||
  fail "symlink stale-candidate rejection removed an alias or target"
cmp -s "${STALE_SYMLINK_TARGET}/payload" \
  "${TMP_DIR}/stale-symlink-target.expected" ||
  fail "symlink stale-candidate rejection changed its target"
rm -f "${stale_symlink}"
assert_no_installer_scratch "${STALE_SYMLINK_ROOT}"

STALE_FILE_ROOT="${TMP_DIR}/installer-stale-file"
make_installer_root "${STALE_FILE_ROOT}"
stale_file="${STALE_FILE_ROOT}/.${runtime_022}.candidate.regular-file"
printf 'stale-file-sentinel\n' >"${stale_file}"
cp "${stale_file}" "${TMP_DIR}/stale-file.expected"
chmod 400 "${stale_file}"
stale_file_fingerprint=$(stat -c '%d:%i:%h:%u:%a:%s' -- "${stale_file}")
if capture_installer_probe \
    "${TMP_DIR}/installer-stale-file.out" "${STALE_FILE_ROOT}"; then
  fail "probe accepted a regular-file stale-candidate lookalike"
else
  stale_file_status=$?
fi
[[ "${stale_file_status}" == 2 ]] ||
  fail "file stale-candidate rejection returned the wrong status"
grep -F 'stale Python runtime candidate is aliased or malformed' \
  "${TMP_DIR}/installer-stale-file.out" >/dev/null ||
  fail "file stale-candidate rejection was not explicit"
[[ -f "${stale_file}" && ! -L "${stale_file}" \
      && "$(stat -c '%d:%i:%h:%u:%a:%s' -- "${stale_file}")" \
        == "${stale_file_fingerprint}" ]] ||
  fail "file stale-candidate rejection removed or changed its lookalike"
cmp -s "${stale_file}" "${TMP_DIR}/stale-file.expected" ||
  fail "file stale-candidate rejection changed lookalike bytes"
chmod 600 "${stale_file}"
rm -f "${stale_file}"
assert_no_installer_scratch "${STALE_FILE_ROOT}"

# Authentication before the lock does not grant a detached file descriptor
# permission to publish.  Revoke the exact transfer pathname while the installer
# waits on the per-runtime lock; its lock-internal pathname recheck must cancel
# the transaction before extraction, verification, or a late rename.
REVOKED_RUNTIME_ROOT="${TMP_DIR}/installer-revoked-transfer"
make_installer_root "${REVOKED_RUNTIME_ROOT}"
revoked_archive=$(stage_installer_archive \
  "${REVOKED_RUNTIME_ROOT}" "$(printf '7%.0s' {1..64})" \
  "${archive_077}" "${archive_sha_077}")
revoked_transfer_root=$(dirname "${revoked_archive}")
revoked_lock_path="${REVOKED_RUNTIME_ROOT}/.locks/${runtime_022}.lock"
: >"${revoked_lock_path}"
chmod 600 "${revoked_lock_path}"
exec {revoked_lock_fd}<>"${revoked_lock_path}"
flock -x "${revoked_lock_fd}"
revoked_output="${TMP_DIR}/installer-revoked-transfer.out"
: >"${revoked_output}"
chmod 600 "${revoked_output}"
(
  # The lock-holder FD belongs only to the controller side of this fixture.
  # Do not let the background installer inherit another reference that would
  # keep the flock alive after the controller releases its copy.
  exec {revoked_lock_fd}>&-
  exec "${TEST_PYTHON_BIN}" -I -S "${INSTALLER}" \
    --runtime-root "${REVOKED_RUNTIME_ROOT}" \
    --manifest-sha256 "${manifest_022}" \
    --verifier "${SEALED_VERIFIER}" \
    --archive "${revoked_archive}" \
    --archive-sha256 "${archive_sha_077}" \
    --lock-timeout-seconds 30 \
    >"${revoked_output}" 2>&1
) &
revoked_installer_pid=$!
archive_fd_seen=0
lock_fd_seen=0
for _attempt in {1..500}; do
  [[ -d "/proc/${revoked_installer_pid}/fd" ]] || {
    sleep 0.01
    continue
  }
  for descriptor_path in "/proc/${revoked_installer_pid}/fd/"*; do
    descriptor_target=$(readlink "${descriptor_path}" 2>/dev/null || true)
    if [[ "${descriptor_target}" == "${revoked_archive}" ]]; then
      archive_fd_seen=1
    elif [[ "${descriptor_target}" == "${revoked_lock_path}" ]]; then
      # _open_lock() is reached only after _bind_archive() has completed its
      # digest and final pathname-identity check.  The controller still owns
      # the exclusive lock, so seeing both descriptors proves the installer
      # is blocked at the intended pre-transaction boundary rather than still
      # authenticating the archive.
      lock_fd_seen=1
    fi
  done
  if (( archive_fd_seen == 1 && lock_fd_seen == 1 )); then
    break
  fi
  sleep 0.01
done
if (( archive_fd_seen != 1 || lock_fd_seen != 1 )); then
  flock -u "${revoked_lock_fd}"
  exec {revoked_lock_fd}>&-
  wait "${revoked_installer_pid}" 2>/dev/null || true
  fail "installer did not bind the archive and open the lock before the race deadline"
fi
rm -f "${revoked_archive}"
[[ -d "${revoked_transfer_root}" \
      && -z "$(find "${revoked_transfer_root}" -mindepth 1 -print -quit)" ]] || {
  flock -u "${revoked_lock_fd}"
  exec {revoked_lock_fd}>&-
  wait "${revoked_installer_pid}" 2>/dev/null || true
  fail "archive revocation did not leave one empty caller-owned token"
}
flock -u "${revoked_lock_fd}"
exec {revoked_lock_fd}>&-
if wait "${revoked_installer_pid}"; then
  fail "installer published through a revoked archive pathname"
else
  revoked_installer_status=$?
fi
[[ "${revoked_installer_status}" == 2 ]] ||
  fail "revoked archive transaction returned the wrong status"
grep -F 'runtime archive transfer was revoked before the install transaction' \
  "${revoked_output}" >/dev/null ||
  fail "revoked archive transaction did not fail its lock-internal recheck"
[[ ! -e "${REVOKED_RUNTIME_ROOT}/${runtime_022}" \
      && ! -L "${REVOKED_RUNTIME_ROOT}/${runtime_022}" ]] ||
  fail "revoked archive transaction created a late final"
if find "${REVOKED_RUNTIME_ROOT}" -maxdepth 1 \
    -name '.*.candidate.*' -print -quit | grep -q .; then
  fail "revoked archive transaction extracted a candidate"
fi
rmdir "${revoked_transfer_root}"
assert_no_installer_scratch "${REVOKED_RUNTIME_ROOT}"

echo '[PASS] scientific Python runtime overlay build/cache/publication contracts'
