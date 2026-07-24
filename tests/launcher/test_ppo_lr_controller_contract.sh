#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf -- "${TMP_DIR}"
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
  grep -F -- "${expected}" "${output_file}" >/dev/null || {
    sed -n '1,80p' "${output_file}" >&2
    fail "missing expected failure text: ${expected}"
  }
}

bash -n batch_ne.sh distill_torso_box.sh

# String-valued and forwarded duplicate controls fail before Python/GPU helper
# execution, so an unusable interpreter proves their ordering.
expect_failure \
  "${TMP_DIR}/direct_bad_schedule.out" \
  'PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: Adaptive' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" PPO_LR_SCHEDULE=Adaptive \
  bash distill_torso_box.sh
expect_failure \
  "${TMP_DIR}/direct_forwarded_override.out" \
  'Do not override launcher-owned PPO LR controller field via forwarded CLI: --algo.config.desired-kl=0.02' \
  env PYTHON_BIN="${TMP_DIR}/must-not-run" \
  bash distill_torso_box.sh --algo.config.desired-kl=0.02

# Controller validation must reject malformed numerics and bound relations
# before source snapshot construction or an SSH command is attempted.
BATCH_ENV=(
  env
  HOLOSOMA_REQUIRE_PYTHON_RUNTIME_OVERLAY=0
  TORCH_DIST_BACKEND=gloo
  NODES=test-node
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=1024
  DRY_RUN=1
  SKIP_GIT_PULL=1
  SOURCE_SNAPSHOT_CACHE="${TMP_DIR}/snapshot-cache"
)
expect_failure \
  "${TMP_DIR}/batch_bad_schedule.out" \
  'PPO_LR_SCHEDULE must be exactly adaptive or fixed. Got: Adaptive' \
  "${BATCH_ENV[@]}" PPO_LR_SCHEDULE=Adaptive bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_bad_kl.out" \
  'PPO_DESIRED_KL must be finite. Got: nan' \
  "${BATCH_ENV[@]}" PPO_DESIRED_KL=nan bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_bad_actor_bounds.out" \
  'ACTOR_MIN_LR must be <= ACTOR_LR; got 0.002>0.001.' \
  "${BATCH_ENV[@]}" ACTOR_LR=0.001 ACTOR_MIN_LR=0.002 bash batch_ne.sh launch
expect_failure \
  "${TMP_DIR}/batch_bad_critic_bounds.out" \
  'CRITIC_LR must be <= CRITIC_MAX_LR; got 0.001>0.0005.' \
  "${BATCH_ENV[@]}" CRITIC_LR=0.001 CRITIC_MAX_LR=0.0005 bash batch_ne.sh launch
for output_file in \
  "${TMP_DIR}/batch_bad_schedule.out" \
  "${TMP_DIR}/batch_bad_kl.out" \
  "${TMP_DIR}/batch_bad_actor_bounds.out" \
  "${TMP_DIR}/batch_bad_critic_bounds.out"; do
  if rg -n 'source_snapshot_id=|\[DRY_RUN\].*ssh' "${output_file}" >/dev/null; then
    fail "invalid PPO LR controller reached snapshot/remote work: ${output_file}"
  fi
done
unset output_file

# The complete canonical controller contract is embedded in the generated
# launch script before that script is hashed, so any field change naturally
# changes the active command identity.
"${PYTHON:-python}" - batch_ne.sh <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
digest_lines = [i for i, line in enumerate(lines) if 'train_cmd_sha256=$(printf' in line]
if len(digest_lines) != 1:
    raise SystemExit(f"[FAIL] expected one generated-command digest boundary, got {digest_lines!r}")
digest_line = digest_lines[0]
for name in (
    "ACTOR_LR",
    "CRITIC_LR",
    "PPO_LR_SCHEDULE",
    "PPO_DESIRED_KL",
    "ACTOR_MIN_LR",
    "ACTOR_MAX_LR",
    "CRITIC_MIN_LR",
    "CRITIC_MAX_LR",
):
    export = f'export {name}=$(quote "${{{name}}}")'
    export_lines = [i for i, line in enumerate(lines) if line == export]
    if len(export_lines) != 1 or export_lines[0] >= digest_line:
        raise SystemExit(
            f"[FAIL] {name} must have one canonical export inside the pre-digest launch script: "
            f"lines={export_lines!r} digest={digest_line}"
        )
    unset_lines = [i for i, line in enumerate(lines) if line == f"unset {name}"]
    if len(unset_lines) != 1 or unset_lines[0] >= export_lines[0]:
        raise SystemExit(
            f"[FAIL] {name} ambient value must be erased before canonical export: "
            f"unset={unset_lines!r} export={export_lines!r}"
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

DIRECT_ENV=(
  env
  DRY_RUN=1
  NPROC=1
  NNODES=1
  PER_GPU_ENVS=2
  CUDA_VISIBLE_DEVICES=0
  LOGGER=logger:disabled
  TEACHER_CHECKPOINT="${TMP_DIR}/teacher.pt"
  TEACHER_CACHE_ROOT="${TMP_DIR}/teacher-cache"
  MOTION_DIR="${TMP_DIR}/motion"
  OBJECT_URDF="${TMP_DIR}/object.urdf"
)

"${DIRECT_ENV[@]}" \
  DISTILL_MODE=dagger DISTILL_LOSS_TYPE=huber \
  BC_LOSS_COEF=0.625 PPO_START_EPOCH=-1 DAGGER_END_EPOCH=-1 \
  PPO_LR_SCHEDULE=fixed PPO_DESIRED_KL=0.023 \
  ACTOR_LR=0.000321 ACTOR_MIN_LR=0.000123 ACTOR_MAX_LR=0.000456 \
  CRITIC_LR=0.000654 CRITIC_MIN_LR=0.000234 CRITIC_MAX_LR=0.000765 \
  bash distill_torso_box.sh >"${TMP_DIR}/direct_explicit.out"

# Exercise both branches of the historical implicit-bound resolver:
# initial below 1e-5 lowers min; initial above 1e-2 raises max.
"${DIRECT_ENV[@]}" \
  DISTILL_MODE=mse BC_LOSS_COEF=0.375 ACTOR_LR=1e-6 CRITIC_LR=0.1 \
  bash distill_torso_box.sh >"${TMP_DIR}/direct_defaults.out"

"${PYTHON_BIN}" - \
  "${TMP_DIR}/direct_explicit.out" \
  "${TMP_DIR}/direct_defaults.out" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

prefix = "[INFO] final_train_command:"
cases = {
    Path(sys.argv[1]): {
        "--algo.config.schedule=fixed",
        "--algo.config.desired-kl=0.023",
        "--algo.config.actor-learning-rate=0.000321",
        "--algo.config.critic-learning-rate=0.000654",
        "--algo.config.min-actor-learning-rate=0.000123",
        "--algo.config.max-actor-learning-rate=0.000456",
        "--algo.config.min-critic-learning-rate=0.000234",
        "--algo.config.max-critic-learning-rate=0.000765",
    },
    Path(sys.argv[2]): {
        "--algo.config.schedule=adaptive",
        "--algo.config.desired-kl=0.01",
        "--algo.config.actor-learning-rate=1e-6",
        "--algo.config.critic-learning-rate=0.1",
        "--algo.config.min-actor-learning-rate=1e-06",
        "--algo.config.max-actor-learning-rate=0.01",
        "--algo.config.min-critic-learning-rate=1e-05",
        "--algo.config.max-critic-learning-rate=0.1",
    },
}
all_prefixes = {
    expected.split("=", 1)[0].replace("_", "-")
    for expected_args in cases.values()
    for expected in expected_args
}
for path, expected in cases.items():
    commands = [
        shlex.split(line[len(prefix) :])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(commands) != 1:
        raise SystemExit(f"[FAIL] {path.name} expected one final command, got {len(commands)}")
    actual = [
        arg
        for arg in commands[0]
        if arg.split("=", 1)[0].replace("_", "-") in all_prefixes
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise SystemExit(
            f"[FAIL] {path.name} did not expose each controller field exactly once: "
            f"actual={actual!r} expected={sorted(expected)!r}"
        )
PY

grep -F \
  '[INFO] ppo_lr_controller schedule=fixed desired_kl=0.023 actor_lr=0.000321 actor_bounds=[0.000123,0.000456] critic_lr=0.000654 critic_bounds=[0.000234,0.000765]' \
  "${TMP_DIR}/direct_explicit.out" >/dev/null ||
  fail 'direct startup log omitted the explicit effective PPO LR controller'

echo '[PASS] PPO LR controller launcher contract'
