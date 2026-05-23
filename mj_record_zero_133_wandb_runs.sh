#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_PY="${INFER_PY:-/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
MODEL_CACHE="${HOLOSOMA_ZERO_RECORD_MODEL_CACHE:-$ROOT_DIR/logs/wandb_models}"
SECONDS_TO_RECORD="${HOLOSOMA_ZERO_RECORD_SECONDS:-20}"
OBJECT_MASS="${HOLOSOMA_MJ_OBJECT_MASS:-2.0}"

RUNS=(
  "qihvpyqg|zihanw22/carry-any/qihvpyqg"
  "zxz3hd8h|zihanw22/carry-any/zxz3hd8h"
  "36k1vwdf|zihanw22/carry-any/36k1vwdf"
  "d9m3z369|zihanw22/boxer/d9m3z369"
)

want_run() {
  local label="$1"
  [[ "$#" -eq 1 ]] && return 0
  shift
  local arg
  for arg in "$@"; do
    [[ "$arg" == "$label" || "$arg" == *"/$label" ]] && return 0
  done
  return 1
}

resolve_latest_onnx() {
  "$INFER_PY" - "$MODEL_CACHE" "$1" <<'PY'
import re
import shutil
import sys
from pathlib import Path

import wandb

cache_root = Path(sys.argv[1]).expanduser()
run_path = sys.argv[2]
entity, project, run_id = run_path.split("/")
run = wandb.Api().run(run_path)
files = [f for f in run.files() if f.name.endswith(".onnx")]
if not files:
    raise SystemExit(f"no .onnx files found in {run_path}")

def key(file_obj):
    nums = [int(x) for x in re.findall(r"\d+", file_obj.name)]
    return (nums[-1] if nums else -1, str(getattr(file_obj, "updated_at", "")), file_obj.name)

latest = max(files, key=key)
out_path = cache_root / entity / project / run_id / latest.name
out_path.parent.mkdir(parents=True, exist_ok=True)
if not out_path.exists() or out_path.stat().st_size == 0:
    tmp_dir = out_path.parent / f".{out_path.name}.download"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    downloaded = latest.download(root=str(tmp_dir), replace=True)
    shutil.copy2(downloaded.name, out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
print(latest.name)
print(out_path)
PY
}

mkdir -p "$MODEL_CACHE"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r label run_path <<<"$spec"
  want_run "$label" "$@" || continue

  mapfile -t resolved < <(resolve_latest_onnx "$run_path")
  model_file="${resolved[0]}"
  model_path="${resolved[1]}"
  model_base="${model_file%.onnx}"
  mass_label="${OBJECT_MASS//./p}kg"
  out_dir="$ROOT_DIR/logs/zero_command_videos_133_${label}_${model_base}_${mass_label}_${SECONDS_TO_RECORD}s_mjdebug_g1_collision"

  echo "run:   $run_path"
  echo "model: $model_file"
  echo "out:   $out_dir"

  HOLOSOMA_ZERO_RECORD_MODEL="$model_path" \
  HOLOSOMA_ZERO_RECORD_SECONDS="$SECONDS_TO_RECORD" \
  HOLOSOMA_MJ_OBJECT_MASS="$OBJECT_MASS" \
  HOLOSOMA_ZERO_RECORD_OUT_DIR="$out_dir" \
  HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=1 \
  HOLOSOMA_POLICY_PICKUP_BUTTON=1 \
  HOLOSOMA_POLICY_DROP_BUTTON=0 \
  bash "$ROOT_DIR/mj_record_zero_133.sh"
done
