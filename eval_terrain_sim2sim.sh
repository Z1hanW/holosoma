#!/usr/bin/env bash
set -euo pipefail

# Run in two terminals:
#   Terminal 1: ./eval_terrain_sim2sim.sh sim
#   Terminal 2: MODEL_PATH=/ABS/PATH/to/your_wbt_policy.onnx ./eval_terrain_sim2sim.sh policy
#
# Optional overrides:
#   ROBOT_PRESET=g1-29dof-stairs ./eval_terrain_sim2sim.sh sim
#   ROBOT_XML=/ABS/PATH/to/scene.xml ./eval_terrain_sim2sim.sh sim
#   ROBOT_X=0.0 ROBOT_Y=0.0 ROBOT_Z=0.8 GANTRY_Z=3.0 ./eval_terrain_sim2sim.sh sim

MODEL_PATH=${MODEL_PATH:-"/ABS/PATH/to/your_wbt_policy.onnx"}
ROBOT_PRESET=${ROBOT_PRESET:-"g1-29dof"}
ROBOT_XML=${ROBOT_XML:-""}

ROBOT_X=${ROBOT_X:-""}
ROBOT_Y=${ROBOT_Y:-""}
ROBOT_Z=${ROBOT_Z:-"0.8"}
GANTRY_Z=${GANTRY_Z:-"3.0"}

if [[ "${1:-}" == "sim" ]]; then
  source scripts/source_mujoco_setup.sh
  cmd=(
    python src/holosoma/holosoma/run_sim.py
    simulator:mujoco
    robot:${ROBOT_PRESET}
    terrain:terrain-locomotion-plane
  )
  if [[ -n "${ROBOT_XML}" ]]; then
    cmd+=(--robot.asset.xml_file "${ROBOT_XML}")
  fi
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
