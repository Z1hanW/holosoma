#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

helper_body=$(awk '
  /^legacy_stop_process_helpers\(\) \{/ { in_helper = 1; next }
  in_helper && /^  cat <<'\''EOF'\''$/ { emit = 1; next }
  emit && /^EOF$/ { exit }
  emit { print }
' batch_ne.sh)
[[ -n "${helper_body}" ]] || fail 'could not extract legacy stop helpers'
eval "${helper_body}"

tmp_dir=$(mktemp -d)

# A PID may exit after the initial /proc directory probe but before Bash opens
# its dynamic stat record.  Force that exact boundary with a DEBUG hook.  The
# helper must classify the vanished generation as absent without leaking the
# shell's otherwise-user-visible redirection warning.
proc_race_stderr="${tmp_dir}/proc-race.stderr"
sleep 60 &
proc_race_pid=$!
proc_race_hook_ran=0
proc_race_debug_hook() {
  if [[ "${BASH_COMMAND}" == mapfile\ -t\ -n\ 1\ stat_records* ]]; then
    trap - DEBUG
    kill -KILL "${proc_race_pid}" 2>/dev/null || true
    wait "${proc_race_pid}" 2>/dev/null || true
    proc_race_hook_ran=1
  fi
}
set -o functrace
trap proc_race_debug_hook DEBUG
set +e
legacy_read_proc_identity "${proc_race_pid}" 2>"${proc_race_stderr}"
proc_race_rc=$?
set -e
trap - DEBUG
set +o functrace
[[ "${proc_race_hook_ran}" == 1 ]] || fail 'procfs disappearance hook missed the first stat read'
[[ "${proc_race_rc}" == 1 ]] || fail "vanished procfs generation returned rc=${proc_race_rc}"
[[ ! -s "${proc_race_stderr}" ]] || {
  sed -n '1,20p' "${proc_race_stderr}" >&2
  fail 'normal procfs disappearance leaked a shell redirection warning'
}
unset -f proc_race_debug_hook

uuid=$(cat /proc/sys/kernel/random/uuid)
unit="tmux-spawn-${uuid}.scope"
scope_pid=''
cgroup_path=''
cgroup_dev=''
cgroup_ino=''
cleanup_fixture() {
  if [[ -n "${cgroup_path}" && -n "${cgroup_dev}" && -n "${cgroup_ino}" ]]; then
    legacy_set_cgroup_frozen_exact \
      "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" 0 >/dev/null 2>&1 || true
  fi
  systemctl --user stop "${unit}" >/dev/null 2>&1 || true
  [[ -z "${scope_pid}" ]] || wait "${scope_pid}" 2>/dev/null || true
  rm -rf -- "${tmp_dir}"
}
trap cleanup_fixture EXIT

command -v systemd-run >/dev/null 2>&1 || fail 'systemd-run is required'
systemctl --user show-environment >/dev/null 2>&1 || fail 'a user systemd manager is required'
systemd-run --user --scope --quiet --slice=- --unit="${unit}" \
  setsid bash -c 'while :; do sleep 1; done' &
scope_pid=$!
for ((poll = 0; poll < 200; poll++)); do
  if legacy_read_proc_identity "${scope_pid}"; then
    scope_start=${legacy_proc_start}
    if legacy_read_proc_cgroup_v2_exact "${scope_pid}" "${scope_start}"; then
      cgroup_path=${legacy_proc_cgroup_path}
      [[ "${cgroup_path##*/}" == "${unit}" ]] && break
    fi
  fi
  sleep 0.01
done
(( poll < 200 )) || fail 'isolated current-UID scope did not become visible'
expected_path="/user.slice/user-$(id -u).slice/user@$(id -u).service/${unit}"
[[ "${cgroup_path}" == "${expected_path}" ]] ||
  fail "scope escaped the exact current-user service boundary: ${cgroup_path}"
legacy_validate_leaf_cgroup_v2 "${cgroup_path}"
cgroup_dev=${legacy_cgroup_dev}
cgroup_ino=${legacy_cgroup_ino}
[[ "${legacy_cgroup_uid}" == "$(id -u)" ]] || fail 'scope owner is not current UID'
legacy_set_cgroup_frozen_exact "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" 1
legacy_read_cgroup_frozen_state "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}"
[[ "${legacy_cgroup_freeze_requested}:${legacy_cgroup_frozen}" == 1:1 ]] ||
  fail 'current-UID scope did not reach effective freeze'
legacy_set_cgroup_frozen_exact "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}" 0
legacy_read_cgroup_frozen_state "${cgroup_path}" "${cgroup_dev}" "${cgroup_ino}"
[[ "${legacy_cgroup_freeze_requested}:${legacy_cgroup_frozen}" == 0:0 ]] ||
  fail 'current-UID scope did not thaw exactly'

# A receipt cannot authorize a process owned by any UID other than both the
# receipt file owner and the current launch owner.
token=$(printf uid-token | sha256sum | awk '{print $1}')
command_sha=$(printf uid-command | sha256sum | awk '{print $1}')
snapshot="src-$(printf uid-snapshot | sha256sum | awk '{print $1}')"
epoch=123456
log_dir=logs/batch_ne/uid_contract
target=17
receipt="${tmp_dir}/receipt"
foreign_uid=$(( $(id -u) + 1 ))
printf '2\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t1\n' \
  "${token}" "${epoch}" "${command_sha}" "${snapshot}" "${log_dir}" \
  "${target}" "${scope_pid}" "${scope_start}" "${cgroup_path}" \
  "${cgroup_dev}" "${cgroup_ino}" > "${receipt}"
printf '%s\t%s\t%s\t0\t%s\t%s\n' \
  "${scope_pid}" "${scope_start}" "${foreign_uid}" "${scope_pid}" "${scope_pid}" \
  >> "${receipt}"
chmod 0400 "${receipt}"
set +e
receipt_output=$(legacy_load_capture_receipt \
  "${receipt}" "${token}" "${epoch}" "${command_sha}" \
  "${snapshot}" "${log_dir}" "${target}" 2>&1)
receipt_rc=$?
set -e
[[ "${receipt_rc}" == 2 ]] || fail "foreign-UID receipt returned rc=${receipt_rc}"
[[ "${receipt_output}" == *'receipt PID records are non-canonical'* ]] ||
  fail "foreign-UID receipt omitted fail-closed diagnostic: ${receipt_output}"

grep -F 'Frozen legacy cgroup member PID ${pid} is not owned by current UID' batch_ne.sh >/dev/null ||
  fail 'frozen membership capture lacks current-UID refusal'
grep -F 'Legacy pane cgroup identity/owner changed or is unsafe' batch_ne.sh >/dev/null ||
  fail 'cgroup boundary lacks current-UID owner refusal'
grep -F 'legacy_proc_uid}" != "${current_uid}' batch_ne.sh >/dev/null ||
  fail 'pre-kill receipt revalidation lacks current-UID comparison'

# Exercise the two delayed closure shapes deterministically against the exact
# production helpers.  A fake sleep advances Bash's monotonic SECONDS clock so
# these tests cover more than the old fixed 100 polls without adding wall time.
(
  delayed_retry_polls=0
  sleep() { SECONDS=$((SECONDS + 1)); }
  legacy_validate_frozen_receipt_cgroup() { return 0; }
  legacy_collect_receipt_executable_survivors() {
    legacy_receipt_executable_survivors=()
  }
  legacy_read_cgroup_frozen_state() {
    delayed_retry_polls=$((delayed_retry_polls + 1))
    legacy_cgroup_populated=1
    (( delayed_retry_polls <= 125 )) || legacy_cgroup_populated=0
  }
  legacy_receipt_cgroup_path=${cgroup_path}
  legacy_receipt_cgroup_dev=${cgroup_dev}
  legacy_receipt_cgroup_ino=${cgroup_ino}
  LEGACY_STOP_CLEANUP_DEADLINE_SECONDS=$((SECONDS + 150))
  legacy_terminate_receipt_bounded /unused/control
  (( delayed_retry_polls == 126 )) ||
    fail "stopping retry did not wait through delayed populated cleanup: polls=${delayed_retry_polls}"
)

(
  delayed_fd_dir="${tmp_dir}/delayed-fd"
  mkdir "${delayed_fd_dir}"
  : >"${delayed_fd_dir}/cgroup.kill"
  delayed_collect_calls=0
  delayed_fd_polls=0
  delayed_fd_closed=0
  sleep() { SECONDS=$((SECONDS + 1)); }
  legacy_validate_frozen_receipt_cgroup() { return 0; }
  legacy_collect_receipt_executable_survivors() {
    delayed_collect_calls=$((delayed_collect_calls + 1))
    if (( delayed_collect_calls == 1 )); then
      legacy_receipt_executable_survivors=(424242)
    else
      legacy_receipt_executable_survivors=()
    fi
  }
  legacy_open_exact_cgroup_fd() {
    legacy_open_cgroup_fd=91
    legacy_open_cgroup_fd_path=${delayed_fd_dir}
  }
  legacy_revalidate_open_receipt_members() { return 0; }
  legacy_read_cgroup_frozen_state_fd() {
    delayed_fd_polls=$((delayed_fd_polls + 1))
    legacy_cgroup_populated=1
    (( delayed_fd_polls <= 125 )) || legacy_cgroup_populated=0
  }
  legacy_close_exact_cgroup_fd() {
    delayed_fd_closed=1
    legacy_open_cgroup_fd=''
    legacy_open_cgroup_fd_path=''
  }
  legacy_receipt_cgroup_path=${cgroup_path}
  legacy_receipt_cgroup_dev=${cgroup_dev}
  legacy_receipt_cgroup_ino=${cgroup_ino}
  LEGACY_STOP_CLEANUP_DEADLINE_SECONDS=$((SECONDS + 150))
  legacy_terminate_receipt_bounded /unused/control
  [[ "$(<"${delayed_fd_dir}/cgroup.kill")" == 1 \
        && "${delayed_fd_polls}" == 126 && "${delayed_fd_closed}" == 1 ]] ||
    fail 'first cgroup.kill commit did not wait through delayed populated closure'
)

# A deleted exact pathname is terminal even when the authenticated held FD is
# still readable and exposes a stale populated=1 event.  Full PID/start reap is
# independently required by the caller after tmux cleanup.
(
  path_gone_fd_dir="${tmp_dir}/path-gone-fd"
  mkdir "${path_gone_fd_dir}"
  : >"${path_gone_fd_dir}/cgroup.kill"
  path_gone_collect_calls=0
  path_gone_fd_polls=0
  path_gone_fd_closed=0
  legacy_validate_frozen_receipt_cgroup() { return 0; }
  legacy_collect_receipt_executable_survivors() {
    path_gone_collect_calls=$((path_gone_collect_calls + 1))
    if (( path_gone_collect_calls == 1 )); then
      legacy_receipt_executable_survivors=(434343)
    else
      legacy_receipt_executable_survivors=()
    fi
  }
  legacy_open_exact_cgroup_fd() {
    legacy_open_cgroup_fd=92
    legacy_open_cgroup_fd_path=${path_gone_fd_dir}
  }
  legacy_revalidate_open_receipt_members() { return 0; }
  legacy_read_cgroup_frozen_state_fd() {
    path_gone_fd_polls=$((path_gone_fd_polls + 1))
    legacy_cgroup_populated=1
    return 0
  }
  legacy_close_exact_cgroup_fd() {
    path_gone_fd_closed=1
    legacy_open_cgroup_fd=''
    legacy_open_cgroup_fd_path=''
  }
  legacy_receipt_cgroup_path="/holosoma-held-fd-path-gone-${BASHPID}"
  legacy_receipt_cgroup_dev=1
  legacy_receipt_cgroup_ino=1
  LEGACY_STOP_CLEANUP_DEADLINE_SECONDS=$((SECONDS + 3))
  legacy_terminate_receipt_bounded /unused/control
  [[ "$(<"${path_gone_fd_dir}/cgroup.kill")" == 1 \
        && "${path_gone_fd_polls}" == 1 && "${path_gone_fd_closed}" == 1 ]] ||
    fail 'held-FD populated=1 plus exact pathname removal was not accepted as cgroup closure'
)

(
  bounded_retry_polls=0
  sleep() { SECONDS=$((SECONDS + 1)); }
  legacy_validate_frozen_receipt_cgroup() { return 0; }
  legacy_collect_receipt_executable_survivors() {
    legacy_receipt_executable_survivors=()
  }
  legacy_read_cgroup_frozen_state() {
    bounded_retry_polls=$((bounded_retry_polls + 1))
    legacy_cgroup_populated=1
  }
  legacy_receipt_cgroup_path=${cgroup_path}
  legacy_receipt_cgroup_dev=${cgroup_dev}
  legacy_receipt_cgroup_ino=${cgroup_ino}
  LEGACY_STOP_CLEANUP_DEADLINE_SECONDS=$((SECONDS + 3))
  set +e
  legacy_terminate_receipt_bounded /unused/control >/dev/null 2>&1
  bounded_retry_rc=$?
  set -e
  [[ "${bounded_retry_rc}" == 2 && "${bounded_retry_polls}" == 4 ]] ||
    fail "delayed stopping retry was not deadline-bounded: rc=${bounded_retry_rc} polls=${bounded_retry_polls}"
)

echo '[PASS] legacy cgroup stop requires exact current-UID scope and receipt ownership'
