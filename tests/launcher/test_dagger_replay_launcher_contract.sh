#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf -- "${TMP_DIR}"' EXIT

fail() { echo "[FAIL] $*" >&2; exit 1; }
expect_failure() {
  local output_file="$1" expected="$2"
  shift 2
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
  grep -F -- "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,80p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

bash -n batch_ne.sh distill_torso_box.sh

expect_failure "${TMP_DIR}/forwarded.out" \
  'Do not override launcher-owned distillation field via forwarded CLI: --algo.config.distill.dagger-replay-enabled=True' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" \
  bash distill_torso_box.sh --algo.config.distill.dagger-replay-enabled=True

BATCH_ENV=(
  env HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0 TORCH_DIST_BACKEND=gloo
  NODES=test-node NPROC=1 NNODES=1 PER_GPU_ENVS=1024 DRY_RUN=1 SKIP_GIT_PULL=1
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)
expect_failure "${TMP_DIR}/capacity.out" \
  'DAGGER_REPLAY_CAPACITY must be a canonical integer in [1, 2147483647]' \
  "${BATCH_ENV[@]}" DAGGER_REPLAY_CAPACITY=0512 bash batch_ne.sh launch
expect_failure "${TMP_DIR}/fraction.out" \
  'DAGGER_REPLAY_FRACTION must be finite and strictly between 0 and 1' \
  "${BATCH_ENV[@]}" DAGGER_REPLAY_FRACTION=1 bash batch_ne.sh launch
expect_failure "${TMP_DIR}/ppo.out" \
  'DAGGER_REPLAY_ENABLED=True requires operational float32 PPO to remain exactly zero' \
  "${BATCH_ENV[@]}" DAGGER_REPLAY_ENABLED=True PPO_TARGET_COEFF=0.1 bash batch_ne.sh launch
for output in "${TMP_DIR}/capacity.out" "${TMP_DIR}/fraction.out" "${TMP_DIR}/ppo.out"; do
  rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${output}" >/dev/null \
    && fail "invalid replay configuration reached snapshot/SSH work: ${output}"
done

python - batch_ne.sh <<'PY'
from pathlib import Path
import sys

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
digest = [i for i, line in enumerate(lines) if "train_cmd_sha256=$(printf" in line]
if len(digest) != 1:
    raise SystemExit(f"[FAIL] expected one command digest boundary, got {digest!r}")
for name in (
    "DAGGER_REPLAY_ENABLED",
    "DAGGER_REPLAY_CAPACITY",
    "DAGGER_REPLAY_BATCH_SIZE",
    "DAGGER_REPLAY_FRACTION",
    "DAGGER_REPLAY_SEED",
):
    exports = [i for i, line in enumerate(lines) if line == f'export {name}=$(quote "${{{name}}}")']
    unsets = [i for i, line in enumerate(lines) if line == f"unset {name}"]
    if len(exports) != 1 or len(unsets) != 1 or not unsets[0] < exports[0] < digest[0]:
        raise SystemExit(
            f"[FAIL] {name} must have one ambient scrub and one pre-hash canonical export: "
            f"unset={unsets!r} export={exports!r} digest={digest!r}"
        )
PY

source scripts/gpu_launch_defaults.sh
mkdir -p "${TMP_DIR}/motion"
touch "${TMP_DIR}/object.urdf"
"${PYTHON_BIN}" - "${TMP_DIR}/teacher.pt" <<'PY'
from pathlib import Path
import sys
import torch
torch.save({}, Path(sys.argv[1]))
PY

env DRY_RUN=1 NPROC=1 NNODES=1 PER_GPU_ENVS=2 CUDA_VISIBLE_DEVICES=0 \
  LOGGER=logger:disabled TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt" \
  TEACHER_CACHE_ROOT="${TMP_DIR}/teacher-cache" MOTION_DIR="${TMP_DIR}/motion" \
  OBJECT_URDF="${TMP_DIR}/object.urdf" DISTILL_MODE=dagger BC_LOSS_COEF=1.0 \
  PPO_START_EPOCH=0 DAGGER_END_EPOCH=700 DAGGER_LOSS_COEF=1.0 DAGGER_MATCH_STD=False \
  DAGGER_REPLAY_ENABLED=True DAGGER_REPLAY_CAPACITY=512 DAGGER_REPLAY_BATCH_SIZE=96 \
  DAGGER_REPLAY_FRACTION=0.5 DAGGER_REPLAY_SEED=17 NUM_LEARNING_ITERATIONS=1000 \
  bash distill_torso_box.sh \
    --algo.config.distill.ppo-start-coeff=0.0 \
    --algo.config.distill.ppo-target-coeff=0.0 \
    --algo.config.distill.ppo-schedule-step-epochs=100 \
    --algo.config.distill.fixed-bc-eval-log-interval=100 \
    --algo.config.distill.fixed-bc-guard-enabled=True \
    --algo.config.distill.fixed-bc-guard-reference-end-epoch=200 \
    --algo.config.distill.fixed-bc-guard-start-epoch=700 \
    --algo.config.distill.fixed-bc-guard-consecutive-evals=3 \
    >"${TMP_DIR}/direct.out"

"${PYTHON_BIN}" - "${TMP_DIR}/direct.out" <<'PY'
from pathlib import Path
import shlex
import sys

prefix = "[INFO] final_train_command:"
commands = [
    shlex.split(line[len(prefix):])
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.startswith(prefix)
]
if len(commands) != 1:
    raise SystemExit(f"[FAIL] expected one final train command, got {len(commands)}")
expected = {
    "--algo.config.distill.dagger-replay-enabled=True",
    "--algo.config.distill.dagger-replay-capacity=512",
    "--algo.config.distill.dagger-replay-batch-size=96",
    "--algo.config.distill.dagger-replay-fraction=0.5",
    "--algo.config.distill.dagger-replay-seed=17",
}
prefixes = {item.split("=", 1)[0] for item in expected}
actual = [arg for arg in commands[0] if arg.split("=", 1)[0].replace("_", "-") in prefixes]
if set(actual) != expected or len(actual) != len(expected):
    raise SystemExit(f"[FAIL] replay CLI was not emitted exactly once: {actual!r}")
PY

grep -F '[INFO] dagger_replay enabled=True capacity_per_rank=512 batch_per_update=96 fraction=0.5 seed=17' \
  "${TMP_DIR}/direct.out" >/dev/null || fail 'direct startup log omitted replay contract'

echo '[PASS] DAgger replay launcher contract'
