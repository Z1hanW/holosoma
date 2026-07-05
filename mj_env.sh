#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
motion_init="${HOLOSOMA_MJ_MOTION_INIT:-0}"
object_xy_offset="${HOLOSOMA_MJ_OBJECT_XY_OFFSET:-0,-0.0}"
object_mass="${HOLOSOMA_MJ_OBJECT_MASS:-2.0}"
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
    --box-offset|--box-xy|--object-offset|--object-xy)
      if [[ $# -lt 3 ]]; then
        echo "mj_env.sh: $1 needs two values: sideways_x away_y" >&2
        exit 2
      fi
      shift
      object_x="$1"
      shift
      object_y="$1"
      object_xy_offset="${object_x},${object_y}"
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
export HOLOSOMA_MJ_OBJECT_XY_OFFSET="$object_xy_offset"
export HOLOSOMA_MJ_OBJECT_MASS="$object_mass"
export HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND="${HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND:-$motion_init}"
export HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${HOLOSOMA_POLICY_COMMAND_STATUS_PATH:-/tmp/holosoma_policy_command_status.json}"
export HOLOSOMA_POLICY_COMMAND_CONTROL_PATH="${HOLOSOMA_POLICY_COMMAND_CONTROL_PATH:-/tmp/holosoma_policy_command_control.json}"
export HOLOSOMA_ENABLE_MUJOCO_POLICY_BUTTON_COMMANDS="${HOLOSOMA_ENABLE_MUJOCO_POLICY_BUTTON_COMMANDS:-1}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
rm -f "$HOLOSOMA_POLICY_COMMAND_STATUS_PATH" 2>/dev/null || true
rm -f "$HOLOSOMA_POLICY_COMMAND_CONTROL_PATH" 2>/dev/null || true

robot_args=()
if [[ -n "$object_urdf" ]]; then
  robot_args+=(--robot.object.object-urdf-path="$object_urdf")
fi

bridge_args=(
  --simulator.config.bridge.enabled=True
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

native_record_args=()
if [[ -n "${HOLOSOMA_MJ_NATIVE_RECORD_DIR:-}" ]]; then
  mkdir -p "$HOLOSOMA_MJ_NATIVE_RECORD_DIR"
  native_record_args+=(
    --logger.video.enabled=True
    --logger.video.interval=1
    --logger.video.save-dir="$HOLOSOMA_MJ_NATIVE_RECORD_DIR"
    --logger.video.upload-to-wandb=False
    --logger.video.output-format=mp4
    --logger.video.width="${HOLOSOMA_MJ_NATIVE_RECORD_WIDTH:-640}"
    --logger.video.height="${HOLOSOMA_MJ_NATIVE_RECORD_HEIGHT:-360}"
    --logger.video.playback-rate="${HOLOSOMA_MJ_NATIVE_RECORD_PLAYBACK_RATE:-0.0685714286}"
    --logger.video.show-command-overlay=False
  )
fi

PYTHONPATH="${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py" \
    robot:g1-29dof-w-object \
    camera:single_d435i_depth \
    image_server:mujoco_d435i \
    --simulator.config.virtual-gantry.enabled=False \
    "${bridge_args[@]}" \
    "${robot_args[@]}" \
    "${native_record_args[@]}"
