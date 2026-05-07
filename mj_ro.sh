#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${1:-${HOLOSOMA_MJ_MOTION:-box_75}}"

motion_file="$clip"
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
fi

if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

export HOLOSOMA_MJ_MOTION="$motion_file"
export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference:${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"

python_code="$(cat <<'PY'
import os
import re
from dataclasses import replace

import wandb

from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.config.config_values.observation import wbt_object_perception_g1
from holosoma_inference.run_policy import run_policy

RUN_PATH = "zihanw22/boxer/tvtwx4to"
MOTION_FILE = os.environ.get("HOLOSOMA_MJ_MOTION", "data_demo/box_75.npz")

run = wandb.Api().run(RUN_PATH)
onnx_files = [file for file in run.files() if file.name.endswith(".onnx")]
if not onnx_files:
    raise RuntimeError(f"No .onnx files found in W&B run {RUN_PATH}")

def checkpoint_step(file_name):
    match = re.search(r"model_(\d+)\.onnx$", file_name)
    return int(match.group(1)) if match else -1


latest_file = max(onnx_files, key=lambda file: (checkpoint_step(file.name), file.updated_at or ""))
latest_model = latest_file.name
model_path = f"wandb://{RUN_PATH}/{latest_model}"

base_config = DEFAULTS["g1-wbt-distillation"]
config = replace(
    base_config,
    observation=wbt_object_perception_g1,
    task=replace(
        base_config.task,
        model_path=model_path,
        interface="lo",
        motion_file=MOTION_FILE,
    ),
)

print(f"[mj_ro] Using latest W&B ONNX: {model_path}")
run_policy(config)
PY
)"

if [[ -t 0 ]]; then
  python3 -c "$python_code"
elif { exec 3</dev/tty; } 2>/dev/null; then
  python3 -c "$python_code" <&3
  exec 3<&-
else
  python3 -c "$python_code"
fi
