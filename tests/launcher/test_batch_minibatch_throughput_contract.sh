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

expect_failure() {
  local output_file="$1"
  local expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,80p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

COMMON_ENV=(
  env
  NODES=test-node
  NPROC=2
  NNODES=1
  CUDA_VISIBLE_DEVICES=0,1
  PER_GPU_ENVS=1024
  DRY_RUN=1
  SKIP_GIT_PULL=1
  SKIP_NODE_HEALTH_CHECK=1
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)

# The production default remains 64.  The rollout horizon must be resolved
# from the source snapshot's PPOConfig rather than introduced as a launcher
# override or duplicated literal.
"${COMMON_ENV[@]}" bash batch_ne.sh launch >"${TMP_DIR}/default.out" 2>&1
grep -F 'export NUM_MINI_BATCHES=64' "${TMP_DIR}/default.out" >/dev/null ||
  fail 'default launch no longer exports NUM_MINI_BATCHES=64'
grep -F 'export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=0' \
  "${TMP_DIR}/default.out" >/dev/null ||
  fail 'default launch unexpectedly enables the throughput canary'
grep -F \
  'source=snapshot:PPOConfig.num_steps_per_env' \
  "${TMP_DIR}/default.out" >/dev/null ||
  fail 'throughput geometry is not attributed to the snapshot PPOConfig source of truth'
grep -F \
  'num_steps_per_env=24 rank_local_rollout_samples=24576 global_rollout_samples=49152 rank_local_samples_per_minibatch_update=384 global_samples_per_minibatch_update=768 minibatch_update_rounds_per_iteration=64 num_learning_epochs=1' \
  "${TMP_DIR}/default.out" >/dev/null ||
  fail 'default throughput summary has incorrect rank-local/global PPO geometry'

snapshot_id=$(sed -nE \
  's/^\[INFO\] source_snapshot_id=(src-[0-9a-f]{64}) .*/\1/p' \
  "${TMP_DIR}/default.out" | tail -1)
[[ "${snapshot_id}" =~ ^src-[0-9a-f]{64}$ ]] ||
  fail 'could not recover the dry-run source snapshot id'
snapshot_archive="${TMP_DIR}/snapshot-cache/${snapshot_id}.tar.gz"
[[ -f "${snapshot_archive}" ]] ||
  fail 'dry-run source snapshot archive is missing'
PINNED_SNAPSHOT_ENV=(
  SOURCE_SNAPSHOT_ID="${snapshot_id}"
  SOURCE_SNAPSHOT_ARCHIVE="${snapshot_archive}"
)

# NUM_MINI_BATCHES=16 is intentionally a two-key opt-in: neither the canary
# flag nor the value alone may silently select the algorithm-changing A/B.
expect_failure \
  "${TMP_DIR}/unflagged_16.out" \
  'NUM_MINI_BATCHES=16 is an algorithm-changing throughput canary' \
  "${COMMON_ENV[@]}" NUM_MINI_BATCHES=16 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/implicit_16.out" \
  'requires explicit NUM_MINI_BATCHES=16' \
  "${COMMON_ENV[@]}" HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 \
  bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/changed_epochs.out" \
  'requires NUM_LEARNING_EPOCHS=1' \
  "${COMMON_ENV[@]}" HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 \
  NUM_MINI_BATCHES=16 NUM_LEARNING_EPOCHS=2 bash batch_ne.sh launch

"${COMMON_ENV[@]}" "${PINNED_SNAPSHOT_ENV[@]}" \
  HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1 NUM_MINI_BATCHES=16 \
  bash batch_ne.sh launch >"${TMP_DIR}/canary_16.out" 2>&1
grep -F \
  '[WARN] minibatch_throughput_canary=enabled num_mini_batches=16 semantics=changes_Adam_and_PPO_update_trajectory math_equivalent=0' \
  "${TMP_DIR}/canary_16.out" >/dev/null ||
  fail '16-minibatch canary does not disclose its algorithmic semantics'
grep -F \
  'num_steps_per_env=24 rank_local_rollout_samples=24576 global_rollout_samples=49152 rank_local_samples_per_minibatch_update=1536 global_samples_per_minibatch_update=3072 minibatch_update_rounds_per_iteration=16 num_learning_epochs=1' \
  "${TMP_DIR}/canary_16.out" >/dev/null ||
  fail '16-minibatch canary summary has incorrect rank-local/global PPO geometry'
grep -F 'export NUM_MINI_BATCHES=16' "${TMP_DIR}/canary_16.out" >/dev/null ||
  fail '16-minibatch canary was not forwarded to the training wrapper'
grep -F 'export HOLOSOMA_MINIBATCH_THROUGHPUT_CANARY=1' \
  "${TMP_DIR}/canary_16.out" >/dev/null ||
  fail '16-minibatch canary label was not forwarded to the node log'

# Fail before remote lifecycle mutation when a rollout cannot be partitioned
# exactly.  1025 environments * the snapshot-owned 24 steps = 24600, which is
# not divisible by the unchanged default of 64.
expect_failure \
  "${TMP_DIR}/nondivisible_rollout.out" \
  'Rank-local rollout samples must be divisible by NUM_MINI_BATCHES' \
  "${COMMON_ENV[@]}" "${PINNED_SNAPSHOT_ENV[@]}" PER_GPU_ENVS=1025 \
  bash batch_ne.sh launch
if grep -F '[DRY_RUN] ssh' "${TMP_DIR}/nondivisible_rollout.out" >/dev/null; then
  fail 'non-divisible rollout reached a remote action before failing closed'
fi

expect_failure \
  "${TMP_DIR}/nondivisible_rollout_all.out" \
  'Rank-local rollout samples must be divisible by NUM_MINI_BATCHES' \
  "${COMMON_ENV[@]}" "${PINNED_SNAPSHOT_ENV[@]}" PER_GPU_ENVS=1025 \
  bash batch_ne.sh all
if grep -F '[DRY_RUN] ssh' "${TMP_DIR}/nondivisible_rollout_all.out" >/dev/null; then
  fail 'all performed remote preparation before rejecting non-divisible rollout geometry'
fi

echo '[PASS] batch minibatch throughput canary contract tests'
