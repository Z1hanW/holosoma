#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${1:-${HOLOSOMA_MJ_MOTION:-box_75}}"

motion_file="$clip"
object_urdf=""
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
  if [[ -f "${ROOT_DIR}/data_demo/objects/${clip}.urdf" ]]; then
    object_urdf="${ROOT_DIR}/data_demo/objects/${clip}.urdf"
  fi
fi

if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

clip_name="$(basename "$motion_file" .npz)"
if [[ -z "$object_urdf" && -f "${ROOT_DIR}/data_demo/objects/${clip_name}.urdf" ]]; then
  object_urdf="${ROOT_DIR}/data_demo/objects/${clip_name}.urdf"
fi

export HOLOSOMA_MJ_MOTION="$motion_file"

robot_args=()
if [[ -n "$object_urdf" ]]; then
  robot_args+=(--robot.object.object-urdf-path="$object_urdf")
fi

PYTHONPATH="${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py" \
    robot:g1-29dof-w-object \
    camera:single_d435i_depth \
    image_server:mujoco_zed2i \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.ignore-default-idle-command=True \
    --simulator.config.bridge.hold-default-pose-until-first-command=True \
    --simulator.config.virtual-gantry.enabled=False \
    "${robot_args[@]}"
