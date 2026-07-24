#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

REAL_TMUX_BIN=$(command -v tmux || true)
REAL_MV_BIN=$(command -v mv || true)
REAL_RM_BIN=$(command -v rm || true)
[[ -n "${REAL_TMUX_BIN}" && -x "${REAL_TMUX_BIN}" ]] || fail 'real tmux is required'
[[ -n "${REAL_MV_BIN}" && -x "${REAL_MV_BIN}" ]] || fail 'real mv is required'
[[ -n "${REAL_RM_BIN}" && -x "${REAL_RM_BIN}" ]] || fail 'real rm is required'
command -v systemd-run >/dev/null 2>&1 || fail 'systemd-run is required'
systemctl --user show-environment >/dev/null 2>&1 || fail 'a user systemd manager is required'
command -v ss >/dev/null 2>&1 || fail 'ss is required'
command -v timeout >/dev/null 2>&1 || fail 'GNU timeout is required'

TMP_DIR=$(mktemp -d)
FAKE_BIN="${TMP_DIR}/bin"
REMOTE_RUN_ROOT="${TMP_DIR}/remote-root"
TMUX_TMPDIR="${TMP_DIR}/tmux"
TMUX_SOCKET="holosoma-legacy-v2-${BASHPID}"
PAUSE_MARKER="${TMP_DIR}/receipt-publish-pause.reached"
RETIRE_KILL_MARKER="${TMP_DIR}/intent-retire-kill.reached"
mkdir -p "${FAKE_BIN}" "${REMOTE_RUN_ROOT}" "${TMUX_TMPDIR}"
chmod 0700 "${TMUX_TMPDIR}"

node="real-tmux-node-${BASHPID}"
session="legacy-real-tmux-${BASHPID}"
run_stamp="${session}-stamp"
target=40000
epoch=1700000301
manifest_sha=$(printf 'real-tmux-snapshot:%s' "${BASHPID}" | sha256sum | awk '{print $1}')
snapshot="src-${manifest_sha}"
token=$(printf 'real-tmux-token:%s' "${BASHPID}" | sha256sum | awk '{print $1}')
snapshot_root="${REMOTE_RUN_ROOT}/${snapshot}"
log_dir="logs/batch_ne/${session}_${run_stamp}"
absolute_log_dir="${snapshot_root}/${log_dir}"
log_file="${absolute_log_dir}/node_0_${node}.log"
exit_marker="${TMP_DIR}/legacy-exit-trap-ran"
session_sha=$(printf '%s' "${session}" | sha256sum | awk '{print $1}')
active_state="${REMOTE_RUN_ROOT}/.active/${session_sha}_${node}.state"
tmux_lock="/tmp/holosoma-tmux-${session_sha}.lock"
scope_uuid=$(cat /proc/sys/kernel/random/uuid)
scope_unit="tmux-spawn-${scope_uuid}.scope"
expected_cgroup="/user.slice/user-$(id -u).slice/user@$(id -u).service/${scope_unit}"
cgroup_dir="/sys/fs/cgroup${expected_cgroup}"

main_port=$((40000 + BASHPID % 10000))
provenance_port=$((main_port + 1))
for ((attempt = 0; attempt < 100; attempt++)); do
  if [[ -z "$(ss -H -ltn "sport = :${main_port}")" \
        && -z "$(ss -H -ltn "sport = :${provenance_port}")" ]]; then
    break
  fi
  main_port=$((main_port + 2))
  provenance_port=$((main_port + 1))
done
(( attempt < 100 && provenance_port <= 65535 )) || fail 'could not find free ports'

read_proc_identity() {
  local pid="$1" record tail ignored
  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r record < "/proc/${pid}/stat" || return 1
  tail=${record##*) }
  proc_state='' proc_ppid='' proc_pgrp='' proc_session='' proc_start=''
  IFS=' ' read -r proc_state proc_ppid proc_pgrp proc_session \
    ignored ignored ignored ignored ignored ignored ignored ignored ignored \
    ignored ignored ignored ignored ignored ignored proc_start ignored <<<"${tail}" || true
  [[ "${proc_state}" =~ ^[A-Za-z]$ \
        && "${proc_ppid}" =~ ^[0-9]+$ \
        && "${proc_pgrp}" =~ ^[0-9]+$ \
        && "${proc_session}" =~ ^[0-9]+$ \
        && "${proc_start}" =~ ^[1-9][0-9]*$ ]]
}

proc_matches_start() {
  read_proc_identity "$1" && [[ "${proc_start}" == "$2" ]]
}

pane_pid=''
pane_start=''
unrelated_pid=''
unrelated_start=''
receipt=''
initial_pids=()
declare -A initial_starts=()
cleanup() {
  local pid start
  set +e
  "${REAL_TMUX_BIN}" -L "${TMUX_SOCKET}" kill-server >/dev/null 2>&1
  systemctl --user stop "${scope_unit}" >/dev/null 2>&1
  if [[ -n "${receipt}" && -f "${receipt}" ]]; then
    while IFS=$'\t' read -r pid start _; do
      [[ "${pid}" =~ ^[1-9][0-9]*$ && "${start}" =~ ^[1-9][0-9]*$ ]] || continue
      proc_matches_start "${pid}" "${start}" && command kill -KILL "${pid}" >/dev/null 2>&1
    done < <(tail -n +2 "${receipt}")
  fi
  if [[ -n "${unrelated_pid}" && -n "${unrelated_start}" ]] \
      && proc_matches_start "${unrelated_pid}" "${unrelated_start}"; then
    command kill -KILL "${unrelated_pid}" >/dev/null 2>&1
    wait "${unrelated_pid}" >/dev/null 2>&1
  fi
  rm -f -- "${tmux_lock}"
  rm -rf -- "${TMP_DIR}"
}
trap cleanup EXIT

cat > "${FAKE_BIN}/tmux" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${REAL_TMUX_BIN:?}" "${REAL_TMUX_SOCKET:?}"
if [[ "${1:-}" == kill-session && "${REAL_TMUX_FORCE_KILL_FAILURE:-0}" == 1 ]]; then
  echo '[FAKE] forced real-tmux kill-session failure' >&2
  exit 97
fi
exec "${REAL_TMUX_BIN}" -L "${REAL_TMUX_SOCKET}" "$@"
EOF

cat > "${FAKE_BIN}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cmd="${@: -1}"
exec bash -c "${cmd}"
EOF

cat > "${FAKE_BIN}/mv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${REAL_MV_BIN:?}" "${REAL_RECEIPT_PAUSE_MARKER:?}"
src=${@: -2:1}
dst=${@: -1}
if [[ "${REAL_PAUSE_RECEIPT_PUBLISH:-0}" == 1 \
      && "${src}" == *.legacy-processes.*.in \
      && "${dst}" == *.legacy-processes.* \
      && "${dst}" != *.freeze-intent ]]; then
  : > "${REAL_RECEIPT_PAUSE_MARKER}"
  sleep 60
fi
exec "${REAL_MV_BIN}" "$@"
EOF

cat > "${FAKE_BIN}/rm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
: "${REAL_RM_BIN:?}" "${REAL_RETIRE_KILL_MARKER:?}"
last=${@: -1}
if [[ "${REAL_KILL_ON_INTENT_RETIRE:-0}" == 1 \
      && "${last}" == *.legacy-processes.*.freeze-intent ]]; then
  : > "${REAL_RETIRE_KILL_MARKER}"
  kill -KILL "${PPID}"
  exit 99
fi
exec "${REAL_RM_BIN}" "$@"
EOF
chmod 0500 "${FAKE_BIN}/tmux" "${FAKE_BIN}/ssh" "${FAKE_BIN}/mv" "${FAKE_BIN}/rm"

export REAL_TMUX_BIN REAL_MV_BIN REAL_RM_BIN TMUX_TMPDIR
export REAL_TMUX_SOCKET="${TMUX_SOCKET}"
export REAL_RECEIPT_PAUSE_MARKER="${PAUSE_MARKER}"
export REAL_RETIRE_KILL_MARKER="${RETIRE_KILL_MARKER}"

mkdir -p "${snapshot_root}/.run_control" "${absolute_log_dir}" \
  "${REMOTE_RUN_ROOT}/.active" "${REMOTE_RUN_ROOT}/.rendezvous"

cat > "${snapshot_root}/distill_as_button_solid.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
bash -c 'sleep 600 & sleep 600 & wait'
EOF
chmod 0500 "${snapshot_root}/distill_as_button_solid.sh"

control_incoming="${snapshot_root}/.run_control/.legacy-real.in"
cat > "${control_incoming}" <<EOF
set -euo pipefail
cd ${snapshot_root}
mkdir -p ${absolute_log_dir}
export HOLOSOMA_SOURCE_SNAPSHOT_ID=${snapshot}
export HOLOSOMA_SOURCE_MANIFEST_SHA256=${manifest_sha}
export NPROC=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=${node}
export MASTER_PORT=${main_port}
export HOLOSOMA_PROVENANCE_MASTER_PORT=${provenance_port}
export TARGET_LEARNING_ITERATION=${target}
export RUN_NAME=${session}
echo "[INFO][${node}] master=${node}:${main_port} log=${log_file}"
trap 'printf legacy-exit-trap-ran >${exit_marker}' EXIT
TRAIN_EXTRA_ARGS=()
bash distill_as_button_solid.sh "\${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a ${log_file}
EOF
command_sha=$(sha256sum "${control_incoming}" | awk '{print $1}')
control="${snapshot_root}/.run_control/train-${command_sha}.sh"
mv -T "${control_incoming}" "${control}"
chmod 0500 "${control}"

tmux_command=$(printf \
  'exec systemd-run --user --scope --quiet --slice=- --unit=%q bash %q' \
  "${scope_unit}" "${control}")
env -u HOLOSOMA_LAUNCH_TOKEN -u HOLOSOMA_COMMAND_SHA256 -u HOLOSOMA_LAUNCH_EPOCH \
  "${FAKE_BIN}/tmux" new-session -d -s "${session}" "${tmux_command}"
"${FAKE_BIN}/tmux" set-option -w -t "${session}" remain-on-exit on
"${FAKE_BIN}/tmux" set-option -t "${session}" @holosoma_launch_token "${token}"
"${FAKE_BIN}/tmux" set-option -t "${session}" @holosoma_command_sha256 "${command_sha}"
"${FAKE_BIN}/tmux" set-option -t "${session}" @holosoma_launch_epoch "${epoch}"

for ((poll = 0; poll < 500; poll++)); do
  pane_pid=$("${FAKE_BIN}/tmux" list-panes -t "${session}" -F '#{pane_pid}')
  if read_proc_identity "${pane_pid}"; then
    pane_start=${proc_start}
    pane_cgroup=$(awk -F: '$1 == 0 { print $3 }' "/proc/${pane_pid}/cgroup" 2>/dev/null || true)
    mapfile -d '' -t pane_argv < "/proc/${pane_pid}/cmdline" || true
    if [[ "${pane_cgroup}" == "${expected_cgroup}" \
          && "${proc_pgrp}" == "${pane_pid}" \
          && "${proc_session}" == "${pane_pid}" ]] \
        && (( ${#pane_argv[@]} == 2 )) \
        && [[ "${pane_argv[0]##*/}" == bash && "${pane_argv[1]}" == "${control}" ]] \
        && [[ -r "${cgroup_dir}/cgroup.procs" ]]; then
      mapfile -t initial_pids < "${cgroup_dir}/cgroup.procs"
      (( ${#initial_pids[@]} >= 5 )) && break
    fi
  fi
  sleep 0.01
done
(( poll < 500 )) || fail 'real tmux did not expose its exact isolated cgroup topology'
for pid in "${initial_pids[@]}"; do
  read_proc_identity "${pid}" || fail "fixture PID ${pid} disappeared"
  initial_starts[${pid}]=${proc_start}
done
[[ $(stat -c %u "${cgroup_dir}") == $(id -u) \
      && -w "${cgroup_dir}/cgroup.freeze" \
      && -w "${cgroup_dir}/cgroup.kill" ]] || fail 'tmux scope lacks exact owner/freezer/kill capability'

sleep 600 &
unrelated_pid=$!
read_proc_identity "${unrelated_pid}" || fail 'unrelated fixture did not start'
unrelated_start=${proc_start}

printf '2\trunning\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${snapshot}" "${log_dir}" "${target}" "${token}" "${command_sha}" "${epoch}" \
  > "${active_state}"
receipt_identity=$(printf 'legacy-process-v2-cgroup\t%s\t%s\t%s' \
  "${token}" "${command_sha}" "${epoch}" | sha256sum | awk '{print $1}')
receipt="${active_state}.legacy-processes.${receipt_identity}"
intent="${receipt}.freeze-intent"
master_key=$(printf '%s' "${node}" | sha256sum | awk '{print $1}')
main_state="${REMOTE_RUN_ROOT}/.rendezvous/${master_key}_${main_port}.state"
provenance_state="${REMOTE_RUN_ROOT}/.rendezvous/${master_key}_${provenance_port}.state"
created_at=$((epoch + 1))
printf '2\t%s\t%s\t%s\t%s\n' "${token}" "${session}" "${main_port}" "${created_at}" > "${main_state}"
printf '2\t%s\t%s\t%s\t%s\n' "${token}" "${session}" "${provenance_port}" "${created_at}" > "${provenance_state}"

STOP_ENV=(
  env PATH="${FAKE_BIN}:${PATH}" NODES="${node}" MASTER_ADDR="${node}"
  MASTER_PORT="${main_port}" HOLOSOMA_PROVENANCE_MASTER_PORT="${provenance_port}"
  NPROC=1 NNODES=1 DRY_RUN=0 REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT}"
  SESSION="${session}" RUN_STAMP="${run_stamp}"
  LEGACY_STOP_EXPECTED_SNAPSHOT_ID="${snapshot}"
  LEGACY_STOP_EXPECTED_TOKEN="${token}" LEGACY_STOP_EXPECTED_EPOCH="${epoch}"
  LEGACY_STOP_EXPECTED_RUN_STAMP="${run_stamp}"
  LEGACY_STOP_EXPECTED_TARGET="${target}"
  LAUNCH_LOCK_TIMEOUT_SECONDS=5 LAUNCH_CLEANUP_TIMEOUT_SECONDS=20
)

# TERM while the canonical receipt .in file is ready but before its atomic
# publish. The precommit trap must thaw and delete intent/receipt residues.
set +e
"${STOP_ENV[@]}" REAL_PAUSE_RECEIPT_PUBLISH=1 \
  LAUNCH_LOCK_TIMEOUT_SECONDS=1 LAUNCH_CLEANUP_TIMEOUT_SECONDS=2 \
  bash batch_ne.sh stop \
  > "${TMP_DIR}/precommit-timeout.out" 2>&1
timeout_rc=$?
set -e
(( timeout_rc != 0 )) || fail 'receipt-publish timeout unexpectedly succeeded'
if [[ ! -f "${PAUSE_MARKER}" ]]; then
  sed -n '1,240p' "${TMP_DIR}/precommit-timeout.out" >&2 || true
  fail 'receipt-publish fault was not reached'
fi
[[ "$(cut -f2 "${active_state}")" == running ]] || fail 'precommit timeout changed active phase'
[[ ! -e "${receipt}" && ! -L "${receipt}" \
      && ! -e "${receipt}.in" && ! -L "${receipt}.in" \
      && ! -e "${intent}" && ! -L "${intent}" \
      && ! -e "${intent}.in" && ! -L "${intent}.in" ]] ||
  fail 'precommit timeout retained receipt/intent publication state'
[[ "$(< "${cgroup_dir}/cgroup.freeze")" == 0 ]] || fail 'precommit timeout left scope frozen'
proc_matches_start "${pane_pid}" "${pane_start}" || fail 'precommit timeout lost pane root'
proc_matches_start "${unrelated_pid}" "${unrelated_start}" || fail 'precommit timeout killed unrelated process'

# SIGKILL exactly when the remote transaction tries to retire the already
# published intent. This must leave running + frozen + receipt + intent, which
# a later stop recognizes without trusting any different boundary.
set +e
"${STOP_ENV[@]}" REAL_KILL_ON_INTENT_RETIRE=1 bash batch_ne.sh stop \
  > "${TMP_DIR}/intent-retire-kill.out" 2>&1
retire_rc=$?
set -e
(( retire_rc != 0 )) || fail 'intent-retire SIGKILL unexpectedly succeeded'
[[ -f "${RETIRE_KILL_MARKER}" ]] || fail 'intent-retire SIGKILL was not reached'
[[ "$(cut -f2 "${active_state}")" == running ]] || fail 'intent-retire SIGKILL changed active phase'
[[ -f "${receipt}" && -f "${intent}" \
      && "$(stat -c %a "${receipt}")" == 400 \
      && "$(stat -c %h "${receipt}")" == 1 ]] ||
  fail 'intent-retire SIGKILL did not preserve canonical receipt+intent'
[[ "$(< "${cgroup_dir}/cgroup.freeze")" == 1 \
      && "$(awk '$1 == "frozen" { print $2 }' "${cgroup_dir}/cgroup.events")" == 1 ]] ||
  fail 'intent-retire SIGKILL did not preserve effective freeze'

# Retry binds the same receipt/intent, retires the redundant marker, commits
# stopping, and uses cgroup.kill. A deliberate tmux-only failure leaves a
# durable stopping state for one final retry.
set +e
"${STOP_ENV[@]}" REAL_TMUX_FORCE_KILL_FAILURE=1 bash batch_ne.sh stop \
  > "${TMP_DIR}/tmux-kill-failure.out" 2>&1
tmux_failure_rc=$?
set -e
(( tmux_failure_rc != 0 )) || fail 'forced tmux kill failure unexpectedly succeeded'
grep -F 'Exact legacy processes are closed, but tmux kill-session failed (rc=97).' \
  "${TMP_DIR}/tmux-kill-failure.out" >/dev/null || fail 'tmux fault did not occur after cgroup.kill'
grep -F 'Legacy all-node freeze barrier accepted exact receipts from every node.' \
  "${TMP_DIR}/tmux-kill-failure.out" >/dev/null || fail 'commit began without the legacy arm barrier'
[[ "$(cut -f2 "${active_state}")" == stopping ]] || fail 'tmux fault did not preserve stopping'
[[ -f "${receipt}" && ! -e "${intent}" && ! -L "${intent}" ]] ||
  fail 'committed retry did not preserve receipt and retire intent'
"${FAKE_BIN}/tmux" has-session -t "${session}" 2>/dev/null || fail 'tmux fault lost retry session'
[[ "$("${FAKE_BIN}/tmux" list-panes -t "${session}" -F '#{pane_dead}')" == 1 ]] ||
  fail 'tmux fault retained a live pane after cgroup.kill'
for pid in "${initial_pids[@]}"; do
  if proc_matches_start "${pid}" "${initial_starts[${pid}]}"; then
    fail "cgroup.kill left captured PID ${pid} alive"
  fi
done
proc_matches_start "${unrelated_pid}" "${unrelated_start}" || fail 'cgroup.kill killed unrelated process'
[[ -f "${main_state}" && -f "${provenance_state}" ]] || fail 'failed closure released reservations'

header_fields=$(awk -F '\t' 'NR == 1 { print NF }' "${receipt}")
body_fields=$(awk -F '\t' 'NR == 2 { print NF }' "${receipt}")
IFS=$'\t' read -r version receipt_token receipt_epoch receipt_command \
  receipt_snapshot receipt_log receipt_target receipt_root receipt_root_start \
  receipt_cgroup receipt_dev receipt_ino receipt_count < "${receipt}"
[[ "${header_fields}:${body_fields}:${version}" == 13:6:2 \
      && "${receipt_token}" == "${token}" \
      && "${receipt_epoch}" == "${epoch}" \
      && "${receipt_command}" == "${command_sha}" \
      && "${receipt_snapshot}" == "${snapshot}" \
      && "${receipt_log}" == "${log_dir}" \
      && "${receipt_target}" == "${target}" \
      && "${receipt_root}" == "${pane_pid}" \
      && "${receipt_root_start}" == "${pane_start}" \
      && "${receipt_cgroup}" == "${expected_cgroup}" \
      && "${receipt_count}" == $(( $(wc -l < "${receipt}") - 1 )) ]] ||
  fail 'v2 receipt header/body contract is inconsistent'

"${STOP_ENV[@]}" bash batch_ne.sh stop > "${TMP_DIR}/retry-success.out" 2>&1
grep -F "stopped ${session} with exact legacy process closure" \
  "${TMP_DIR}/retry-success.out" >/dev/null || fail 'retry omitted stopped publication'
grep -F 'verified exact legacy stopped process/session closure' \
  "${TMP_DIR}/retry-success.out" >/dev/null || fail 'retry omitted terminal verification'
grep -F 'Legacy all-node freeze barrier accepted exact receipts from every node.' \
  "${TMP_DIR}/retry-success.out" >/dev/null || fail 'retry omitted the legacy arm barrier'
[[ "$(cut -f2 "${active_state}")" == stopped ]] || fail 'retry did not publish stopped'
if "${FAKE_BIN}/tmux" has-session -t "${session}" 2>/dev/null; then
  fail 'retry left tmux session alive'
else
  tmux_rc=$?
  [[ "${tmux_rc}" == 1 ]] || fail "tmux absence query failed rc=${tmux_rc}"
fi
[[ -f "${receipt}" && ! -L "${receipt}" ]] || fail 'retry removed immutable receipt'
[[ ! -e "${main_state}" && ! -L "${main_state}" \
      && ! -e "${provenance_state}" && ! -L "${provenance_state}" ]] ||
  fail 'retry did not release exact reservation pair'
proc_matches_start "${unrelated_pid}" "${unrelated_start}" || fail 'retry killed unrelated process'
[[ ! -e "${exit_marker}" ]] || fail 'cgroup.kill ran the legacy EXIT trap'

echo '[PASS] real tmux legacy cgroup-v2 stop survives publish, commit, and tmux retry faults'
