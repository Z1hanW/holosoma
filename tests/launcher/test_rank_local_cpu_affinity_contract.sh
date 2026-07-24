#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
cd "${REPO_ROOT}"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

bash -n batch_ne.sh distill_torso_box.sh

for contract in \
  'HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}' \
  'HOLOSOMA_RANK_LOCAL_CPU_AFFINITY="$(normalize_bool01 HOLOSOMA_RANK_LOCAL_CPU_AFFINITY "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}")"' \
  'export HOLOSOMA_RANK_LOCAL_CPU_AFFINITY=$(quote "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}")' \
  'rank_local_cpu_affinity=\${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY}' \
  'HOLOSOMA_CARB_TASKING_THREAD_COUNT=${HOLOSOMA_CARB_TASKING_THREAD_COUNT:-}' \
  'export HOLOSOMA_CARB_TASKING_THREAD_COUNT=$(quote "${HOLOSOMA_CARB_TASKING_THREAD_COUNT}")'; do
  grep -F -- "${contract}" batch_ne.sh >/dev/null ||
    fail "batch launcher is missing contract: ${contract}"
done
unset contract

for contract in \
  'holosoma_canonicalize_positive_int32 \' \
  'HOLOSOMA_CARB_TASKING_THREAD_COUNT || exit' \
  'HOLOSOMA_ISAACSIM_KIT_ARGS="--/plugins/carb.tasking.plugin/threadCount=${HOLOSOMA_CARB_TASKING_THREAD_COUNT}"' \
  'export HOLOSOMA_ISAACSIM_KIT_ARGS'; do
  grep -F -- "${contract}" distill_torso_box.sh >/dev/null ||
    fail "terminal launcher is missing Carb/AppLauncher contract: ${contract}"
done
unset contract

if grep -F -- 'TRAIN_EXTRA_ARGS+=("--/plugins/carb.tasking.plugin/threadCount=' \
    batch_ne.sh >/dev/null; then
  fail 'Carb Kit settings must never enter the strict Tyro training CLI'
fi

grep -F -- '|| flag_enabled "${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}"' \
  distill_torso_box.sh >/dev/null ||
  fail 'terminal launcher must select the pre-import wrapper when affinity alone is enabled'
grep -F -- 'HOLOSOMA_RANK_LOCAL_CPU_AFFINITY="${HOLOSOMA_RANK_LOCAL_CPU_AFFINITY:-0}"' \
  distill_torso_box.sh >/dev/null ||
  fail 'terminal launcher must forward the normalized affinity opt-in to torchrun children'

python3 - <<'PY'
from pathlib import Path

source = Path("src/holosoma/holosoma/train_agent_rank_visible.py").read_text(encoding="utf-8")
apply_index = source.index("    _apply_rank_local_cpu_affinity()")
remap_index = source.index("    _remap_rank_to_single_visible_gpu()")
run_index = source.index("    runpy.run_path(")
if not apply_index < remap_index < run_index:
    raise SystemExit("[FAIL] affinity must run before CUDA remapping and train_agent imports")
if '_env_flag(_RANK_LOCAL_CPU_AFFINITY_ENV, default=False)' not in source:
    raise SystemExit("[FAIL] rank-local affinity must remain default-off")
if "os.sched_setaffinity(0, selected_cpus)" not in source:
    raise SystemExit("[FAIL] rank-local affinity must use the process-local scheduler API")
if "except (Exception, SystemExit) as exc:" not in source or "fail-open" not in source:
    raise SystemExit("[FAIL] optional affinity must have an explicit fail-open path")
PY

echo "[PASS] rank-local CPU affinity launcher contract"
