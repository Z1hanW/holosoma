#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

helper_definitions=$(awk '
  /^private_lifecycle_file_validation_helpers\(\) \{/ { emit = 1 }
  emit && /^tmux_session_query_helpers\(\) \{/ { exit }
  emit { print }
' batch_ne.sh)
[[ -n "${helper_definitions}" ]] || fail 'could not extract lifecycle state helpers'
eval "${helper_definitions}"
eval "$(active_state_validation_helpers)"

harden_definitions=$(awk '
  /^harden_lifecycle_namespace_node\(\) \{/ { emit = 1 }
  emit && /^ensure_local_corl_package_metadata\(\) \{/ { exit }
  emit { print }
' batch_ne.sh)
[[ -n "${harden_definitions}" ]] || fail 'could not extract lifecycle namespace hardening'

tmp_dir=$(mktemp -d)
cleanup() {
  rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT

REMOTE_RUN_ROOT="${tmp_dir}/run-root"
NODE_LIST=(local-fixture)
LAUNCH_CONTROL_TIMEOUT_SECONDS=10
mkdir -m 0775 -- "${REMOTE_RUN_ROOT}"
mkdir -m 0775 -- \
  "${REMOTE_RUN_ROOT}/.active" \
  "${REMOTE_RUN_ROOT}/.rendezvous" \
  "${REMOTE_RUN_ROOT}/.status"

quote() { printf '%q' "$1"; }
remote_run_bounded() {
  local _node="$1" cmd="$2" _timeout="$3"
  # Reproduce the fleet's ambient SSH umask.  The production wrapper and the
  # hardening transaction must make their result independent of it.
  (umask 0002; bash -c "umask 077; ${cmd}")
}
eval "${harden_definitions}"

harden_lifecycle_namespaces_parallel
[[ "$(stat -c %a -- "${REMOTE_RUN_ROOT}")" == 755 ]] ||
  fail 'REMOTE_RUN_ROOT was not hardened to 0755 from umask-0002 mode'
for relative in .active .rendezvous .status .active/.locks; do
  path="${REMOTE_RUN_ROOT}/${relative}"
  [[ -d "${path}" && ! -L "${path}" \
      && "$(stat -c %a -- "${path}")" == 700 \
      && "$(stat -c %u -- "${path}")" == "$(id -u)" ]] ||
    fail "private lifecycle directory is unsafe: ${relative}"
done

# A symlinked namespace must fail without touching its target.
mv -T "${REMOTE_RUN_ROOT}/.status" "${REMOTE_RUN_ROOT}/.status.real"
mkdir -- "${tmp_dir}/status-target"
printf 'unchanged\n' > "${tmp_dir}/status-target/sentinel"
ln -s "${tmp_dir}/status-target" "${REMOTE_RUN_ROOT}/.status"
if harden_lifecycle_namespaces_parallel >/dev/null 2>&1; then
  fail 'symlinked lifecycle namespace was accepted'
fi
[[ "$(<"${tmp_dir}/status-target/sentinel")" == unchanged ]] ||
  fail 'symlink rejection modified its target'
rm -f -- "${REMOTE_RUN_ROOT}/.status"
mv -T "${REMOTE_RUN_ROOT}/.status.real" "${REMOTE_RUN_ROOT}/.status"

# A multiply-linked persistent lock is not one authoritative lock inode.
(umask 077; : > "${REMOTE_RUN_ROOT}/.rendezvous/.reservation.lock")
ln "${REMOTE_RUN_ROOT}/.rendezvous/.reservation.lock" "${tmp_dir}/lock-alias"
if harden_lifecycle_namespaces_parallel >/dev/null 2>&1; then
  fail 'multiply-linked rendezvous lock was accepted'
fi
rm -f -- "${tmp_dir}/lock-alias" "${REMOTE_RUN_ROOT}/.rendezvous/.reservation.lock"
harden_lifecycle_namespaces_parallel >/dev/null

hex=$(printf '%064d' 0)
state="${REMOTE_RUN_ROOT}/.active/security-fixture.state"
record=$(printf '2\tstopped\tsrc-%s\tlogs/batch_ne/security-fixture_1\t8\t%s\t%s\t1' \
  "${hex}" "${hex}" "${hex}")

# The wrapper's explicit umask produces a strict state even when its caller is
# running under 0002.
(umask 0002; (umask 0077; printf '%s\n' "${record}" > "${state}"))
[[ "$(stat -c %a -- "${state}")" == 600 ]] ||
  fail 'umask-0002 execution produced a non-0600 lifecycle state'
load_active_state_v2_exact "${state}"

chmod 0660 "${state}"
if load_active_state_v2_exact "${state}" >/dev/null 2>&1; then
  fail 'group-writable active metadata was accepted by the strict loader'
fi
chmod 0600 "${state}"

mv -T "${state}" "${state}.regular"
ln -s "$(basename "${state}.regular")" "${state}"
if load_active_state_v2_exact "${state}" >/dev/null 2>&1; then
  fail 'symlinked active metadata was accepted'
fi
rm -f -- "${state}"
mv -T "${state}.regular" "${state}"

ln "${state}" "${state}.hardlink"
if load_active_state_v2_exact "${state}" >/dev/null 2>&1; then
  fail 'multiply-linked active metadata was accepted'
fi
rm -f -- "${state}.hardlink"

# Only the explicit compatibility argument may load a legacy 0644/0664 inode;
# its one-time migration publishes a new strict 0600 inode.
chmod 0664 "${state}"
if load_active_state_v2_exact "${state}" >/dev/null 2>&1; then
  fail 'legacy mode was accepted without explicit authorization'
fi
load_active_state_v2_exact "${state}" 1
[[ "${active_state_legacy_mode}" == 1 ]] || fail 'legacy mode was not classified'
migrate_loaded_active_state_to_private "${state}"
[[ "$(stat -c %a -- "${state}")" == 600 \
    && "$(stat -c %h -- "${state}")" == 1 ]] ||
  fail 'legacy active metadata was not atomically migrated to a private inode'

lock="${REMOTE_RUN_ROOT}/.active/.locks/security-fixture.lock"
(umask 0002; open_private_lifecycle_lock "${lock}" 8; flock -x 8)
[[ "$(stat -c %a -- "${lock}")" == 600 \
    && "$(stat -c %h -- "${lock}")" == 1 ]] ||
  fail 'private lifecycle lock did not ignore ambient umask 0002'

mv -T "${lock}" "${lock}.regular"
ln -s "$(basename "${lock}.regular")" "${lock}"
if open_private_lifecycle_lock "${lock}" 8 >/dev/null 2>&1; then
  fail 'symlinked private lifecycle lock was accepted'
fi

wrapper_prefix_count=$(grep -Fxc '  cmd="umask 077; ${cmd}"' batch_ne.sh || true)
[[ "${wrapper_prefix_count}" == 2 ]] ||
  fail 'bounded and mutation-bounded SSH wrappers do not both force umask 077'

echo '[PASS] lifecycle namespaces, states, and locks are private under SSH umask 0002'
