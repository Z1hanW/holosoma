#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 INPUT INITIAL_RESULT MODEL OUTPUT_ROOT [CANARY_ARGS...]" >&2
  exit 2
fi

input=$1
current_result=$2
model=$3
output_root=$4
shift 4

iterations=${ITERATIONS:-5}
cuda_device=${CUDA_DEVICE:-0}
python_bin=${PYTHON_BIN:-python}
mkdir -p "$output_root"

for ((iteration = 1; iteration <= iterations; iteration++)); do
  iteration_dir=$(printf "%s/iteration_%02d" "$output_root" "$iteration")
  mkdir -p "$iteration_dir"
  echo "iteration=$iteration input_result=$current_result"
  CUDA_VISIBLE_DEVICES=$cuda_device "$python_bin" \
    scripts/run_real_dynamics_stage_canary.py \
    --input "$input" \
    --geometric-result "$current_result" \
    --model "$model" \
    --output "$iteration_dir/result.npz" \
    --report "$iteration_dir/report.json" \
    "$@" 2>&1 | tee "$iteration_dir/run.log"

  accepted_step=$(
    "$python_bin" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["accepted_step_size"])' \
      "$iteration_dir/report.json"
  )
  current_result="$iteration_dir/result.npz"
  if [[ "$accepted_step" == "0.0" ]]; then
    echo "stopped_after_iteration=$iteration reason=line_search_rejected"
    break
  fi
done

echo "final_result=$current_result"
