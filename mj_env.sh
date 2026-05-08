#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
motion_init="${HOLOSOMA_MJ_MOTION_INIT:-0}"
explicit_motion_mode=0
clip_arg_seen=0
if [[ -n "${HOLOSOMA_MJ_MOTION_INIT:-}" ]]; then
  explicit_motion_mode=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-init)
      motion_init=1
      explicit_motion_mode=1
      ;;
    --manual)
      motion_init=0
      explicit_motion_mode=1
      ;;
    --clip)
      shift
      clip="$1"
      clip_arg_seen=1
      ;;
    *)
      clip="$1"
      clip_arg_seen=1
      ;;
  esac
  shift
done

if [[ "$explicit_motion_mode" == "0" && "$clip_arg_seen" == "1" ]]; then
  motion_init=1
fi

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
export HOLOSOMA_MJ_MOTION_INIT="$motion_init"
export HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND="${HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND:-$motion_init}"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5555}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"

robot_args=()
if [[ -n "$object_urdf" ]]; then
  robot_args+=(--robot.object.object-urdf-path="$object_urdf")
fi

bridge_args=(
  --simulator.config.bridge.enabled=True
  --simulator.config.bridge.clock-port="$SIM_CLOCK_PORT"
  --simulator.config.bridge.publish-sim-state=True
  --simulator.config.bridge.sim-state-port="$SIM_STATE_PORT"
)
if [[ -n "${HOLOSOMA_DDS_DOMAIN_ID:-}" ]]; then
  bridge_args+=(--simulator.config.bridge.domain-id="$HOLOSOMA_DDS_DOMAIN_ID")
fi
if [[ "$motion_init" == "1" ]]; then
  bridge_args+=(
    --simulator.config.bridge.ignore-default-idle-command=True
    --simulator.config.bridge.hold-default-pose-until-first-command=True
  )
fi

PYTHONPATH="${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py" \
    robot:g1-29dof-w-object \
    camera:single_d435i_depth \
    image_server:mujoco_d435i \
    --simulator.config.virtual-gantry.enabled=False \
    "${bridge_args[@]}" \
    "${robot_args[@]}"
