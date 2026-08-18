#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

COMMON_ENV=(
  env
  NODES=test-node
  NPROC=1
  NNODES=1
  CUDA_VISIBLE_DEVICES=0
  PER_GPU_ENVS=1024
  DRY_RUN=1
  SKIP_GIT_PULL=1
  SKIP_NODE_HEALTH_CHECK=1
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0
  REMOTE_RUN_ROOT="${TMP_DIR}/remote-run-root"
  LOGGER_BASE_DIR="${TMP_DIR}/remote-run-root/training-logs"
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)

"${COMMON_ENV[@]}" bash batch_ne.sh launch >"${TMP_DIR}/default.out" 2>&1
if grep -F 'export CAMERA_PITCH_DEG=' "${TMP_DIR}/default.out" >/dev/null; then
  fail 'unset CAMERA_PITCH_DEG was unexpectedly exported to the wrapper'
fi

snapshot_id=$(sed -nE \
  's/^\[INFO\] source_snapshot_id=(src-[0-9a-f]{64}) .*/\1/p' \
  "${TMP_DIR}/default.out" | tail -1)
[[ "${snapshot_id}" =~ ^src-[0-9a-f]{64}$ ]] ||
  fail 'could not recover the dry-run source snapshot id'
snapshot_archive="${TMP_DIR}/snapshot-cache/${snapshot_id}.tar.gz"
[[ -f "${snapshot_archive}" ]] || fail 'dry-run source snapshot archive is missing'

"${COMMON_ENV[@]}" \
  SOURCE_SNAPSHOT_ID="${snapshot_id}" \
  SOURCE_SNAPSHOT_ARCHIVE="${snapshot_archive}" \
  CAMERA_PITCH_DEG=0 \
  bash batch_ne.sh launch >"${TMP_DIR}/explicit-zero.out" 2>&1
grep -F 'export CAMERA_PITCH_DEG=0' "${TMP_DIR}/explicit-zero.out" >/dev/null ||
  fail 'explicit zero residual camera pitch was not forwarded to the wrapper'

for invalid in nan inf -inf not-a-number; do
  output="${TMP_DIR}/invalid-${invalid//[^A-Za-z0-9]/_}.out"
  if "${COMMON_ENV[@]}" CAMERA_PITCH_DEG="${invalid}" \
      bash batch_ne.sh launch >"${output}" 2>&1; then
    fail "invalid CAMERA_PITCH_DEG unexpectedly succeeded: ${invalid}"
  fi
  grep -E 'CAMERA_PITCH_DEG must be (numeric|finite)' "${output}" >/dev/null ||
    fail "invalid CAMERA_PITCH_DEG did not produce an actionable error: ${invalid}"
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*(ssh|scp)' "${output}" >/dev/null; then
    fail "invalid CAMERA_PITCH_DEG reached snapshot or remote activity: ${invalid}"
  fi
done

echo '[PASS] batch camera-pitch propagation contract tests'
