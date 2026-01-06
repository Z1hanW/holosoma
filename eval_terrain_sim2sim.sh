#!/usr/bin/env bash
set -euo pipefail

# Run in two terminals:
#   Terminal 1: ./eval_terrain_sim2sim.sh sim
#   Terminal 2: MODEL_PATH=/ABS/PATH/to/your_wbt_policy.onnx ./eval_terrain_sim2sim.sh policy
#
# Optional overrides for manual XY placement:
#   ROBOT_X=0.0 ROBOT_Y=0.0 ROBOT_Z=0.8 GANTRY_Z=3.0 ./eval_terrain_sim2sim.sh sim

MODEL_PATH=${MODEL_PATH:-"/ABS/PATH/to/your_wbt_policy.onnx"}
OBJ_PATH=${OBJ_PATH:-"stairs.obj"}

ROBOT_X=${ROBOT_X:-""}
ROBOT_Y=${ROBOT_Y:-""}
ROBOT_Z=${ROBOT_Z:-"0.8"}
GANTRY_Z=${GANTRY_Z:-"3.0"}

if [[ "${1:-}" == "sim" ]]; then
  source scripts/source_mujoco_setup.sh
  cmd=(
    python src/holosoma/holosoma/run_sim.py
    simulator:mujoco
    robot:g1-29dof
    terrain:terrain-load-obj
    --terrain.terrain-term.spawn.randomize_tiles=False
    --terrain.terrain-term.num-rows 1
    --terrain.terrain-term.num-cols 1
    --terrain.terrain-term.obj-file-path "${OBJ_PATH}"
  )
  if [[ -n "${ROBOT_X}" && -n "${ROBOT_Y}" ]]; then
    cmd+=(
      --robot.init-state.pos "${ROBOT_X}" "${ROBOT_Y}" "${ROBOT_Z}"
      --simulator.config.virtual-gantry.point "${ROBOT_X}" "${ROBOT_Y}" "${GANTRY_Z}"
    )
  fi
  "${cmd[@]}"
  exit 0
fi

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
