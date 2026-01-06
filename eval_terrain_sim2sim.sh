#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./2100.onnx"
OBJ_PATH="stairs.obj"


python src/holosoma/holosoma/run_sim.py \
  simulator:mujoco \
  robot:g1-29dof \
  terrain:terrain-load-obj \
  --terrain.terrain-term.spawn.randomize_tiles=False \
  --terrain.terrain-term.obj-file-path "${OBJ_PATH}" \
  --terrain.terrain-term.num-rows 1 \
  --terrain.terrain-term.num-cols 1 \
  --robot.init-state.pos 1.2 -1.4 0.8 \



if [[ "${1:-}" == "policy" ]]; then
  if [[ "${MODEL_PATH}" == "/ABS/PATH/to/your_wbt_policy.onnx" ]]; then
    echo "Set MODEL_PATH to your WBT ONNX model path." >&2
    exit 1
  fi
  source scripts/source_inference_setup.sh
  python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path "${MODEL_PATH}" \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
  exit 0
fi

echo "Usage: $0 {sim|policy}" >&2
exit 1
