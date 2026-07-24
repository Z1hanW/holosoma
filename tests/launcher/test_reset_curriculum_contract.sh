#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "${ROOT_DIR}/scripts/reset_curriculum_contract.sh"

umask 077
OUTPUT=$(mktemp /tmp/holosoma-reset-curriculum-test.XXXXXX)
trap 'rm -f "${OUTPUT}"' EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

expect_failure() {
  local expected="$1"
  shift
  if "$@" >"${OUTPUT}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F "${expected}" "${OUTPUT}" >/dev/null || {
    sed -n '1,40p' "${OUTPUT}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

check_defaults() {
  local NUM_LEARNING_ITERATIONS="$1" expected_start="$2" expected_end="$3"
  local START_AT_TIMESTEP_ZERO_PROB=0.2
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=""
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=""
  local FREEZE_AT_TIMESTEP_ZERO_PROB=0.0
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
  local FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=""
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=""
  local RESUME_CKPT="" RESUME_CHECKPOINT="" POLICY_INIT_CKPT=""
  holosoma_configure_all_reset_curricula NUM_LEARNING_ITERATIONS
  [[ "${START_AT_TIMESTEP_ZERO_PROB_START_ITER}" == "${expected_start}" ]]
  [[ "${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" == "${expected_end}" ]]
  [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}" == "${expected_start}" ]]
  [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}" == "${expected_end}" ]]
}

check_defaults 1 0 0
check_defaults 8 0 7
check_defaults 2500 0 2499
check_defaults 2501 2500 2500
check_defaults 40000 2500 39999
check_defaults 00008 0 7

check_explicit_preservation() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=2
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=6
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
  local FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=3
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=5
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_all_reset_curricula NUM_LEARNING_ITERATIONS
  [[ "${START_AT_TIMESTEP_ZERO_PROB_START_ITER}/${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" == 2/6 ]]
  [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}/${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}" == 3/5 ]]
}
check_explicit_preservation

check_all_none() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=None
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=None
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=None
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END=0.0
  local FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER=""
  local FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER=""
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_all_reset_curricula NUM_LEARNING_ITERATIONS
  [[ "${START_AT_TIMESTEP_ZERO_PROB_END}/${START_AT_TIMESTEP_ZERO_PROB_START_ITER}/${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" == None/None/None ]]
  [[ "${FREEZE_AT_TIMESTEP_ZERO_PROB_START_ITER}/${FREEZE_AT_TIMESTEP_ZERO_PROB_END_ITER}" == 0/7 ]]
}
check_all_none

partial_none() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=None
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=0
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=None
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
}
expect_failure 'must either all be None or all use a numeric schedule' partial_none

reverse_schedule() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=6
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=5
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
}
expect_failure 'must be <=' reverse_schedule

beyond_target() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=0
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=9
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
}
expect_failure 'must be <= NUM_LEARNING_ITERATIONS' beyond_target

malicious_endpoint() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=0
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER="$1"
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
}
expect_failure 'must be an ASCII integer' malicious_endpoint '7;touch /tmp/nope'
expect_failure 'is overlong' malicious_endpoint "$(printf '9%.0s' {1..65})"

canonical_endpoint() {
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=0002
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=0007
  local RESUME_CKPT="" RESUME_CHECKPOINT=""
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
  [[ "${START_AT_TIMESTEP_ZERO_PROB_START_ITER}/${START_AT_TIMESTEP_ZERO_PROB_END_ITER}" == 2/7 ]]
}
canonical_endpoint

equal_target() {
  local resume_name="${1:-}" policy_init="${2:-0}"
  local NUM_LEARNING_ITERATIONS=8
  local START_AT_TIMESTEP_ZERO_PROB_END=1.0
  local START_AT_TIMESTEP_ZERO_PROB_START_ITER=0
  local START_AT_TIMESTEP_ZERO_PROB_END_ITER=8
  local RESUME_CKPT="" RESUME_CHECKPOINT="" POLICY_INIT_CKPT=""
  [[ "${resume_name}" == RESUME_CKPT ]] && RESUME_CKPT=/tmp/resume.pt
  [[ "${resume_name}" == RESUME_CHECKPOINT ]] && RESUME_CHECKPOINT=/tmp/resume.pt
  [[ "${policy_init}" == 1 ]] && POLICY_INIT_CKPT=/tmp/init.pt
  holosoma_configure_reset_curriculum START_AT_TIMESTEP_ZERO_PROB NUM_LEARNING_ITERATIONS
}
expect_failure 'Fresh/policy-init' equal_target
expect_failure 'Fresh/policy-init' equal_target '' 1
equal_target RESUME_CKPT
equal_target RESUME_CHECKPOINT

bad_target() {
  local NUM_LEARNING_ITERATIONS="$1"
  holosoma_canonicalize_positive_int32 NUM_LEARNING_ITERATIONS
}
expect_failure 'must be positive' bad_target 0
expect_failure 'must be an ASCII integer' bad_target '8;touch /tmp/nope'
expect_failure 'is overlong' bad_target "$(printf '9%.0s' {1..65})"
expect_failure 'must be <=' bad_target 2147483648

echo '[PASS] reset curriculum exclusive-target contract'
