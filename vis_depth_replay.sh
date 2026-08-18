#!/usr/bin/env bash
set -euo pipefail

# Minimal, strict depth replay for far-tracking aligned setup.
# Usage:
#   bash vis_depth_replay.sh
# Optional env overrides:
#   HEADLESS=True|False
#   MOTION_FILE=/abs/path/to/motion.npz
#   TERRAIN_OBJ=/abs/path/to/terrain.obj
#   VISER_PORT=3953
#   NUM_ENVS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

EXP="g1-29dof-wbt"
COMMAND_PRESET="g1-29dof-wbt"
PERCEPTION_PRESET="camera-depth-d435i"

MOTION_FILE="${MOTION_FILE:-${ROOT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz}"
TERRAIN_OBJ="${TERRAIN_OBJ:-${ROOT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-42}"
VISER_PORT="${VISER_PORT:-3953}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"

HEADLESS_RAW="${HEADLESS:-False}"
case "$(printf '%s' "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TRAINING_HEADLESS="True"
    ISAAC_HEADLESS_ENV="1"
    ;;
  0|false|no|off)
    TRAINING_HEADLESS="False"
    ISAAC_HEADLESS_ENV="0"
    ;;
  *) echo "[ERROR] HEADLESS must be True/False/1/0, got: ${HEADLESS_RAW}" >&2; exit 1 ;;
esac

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi
if [[ ! -f "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi
if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ ]] || (( NUM_ENVS < 1 )); then
  echo "[ERROR] NUM_ENVS must be positive integer, got: ${NUM_ENVS}" >&2
  exit 1
fi

# Lock camera alignment to far-tracking depth training defaults.
CAMERA_BODY_NAME="torso_link"
CAMERA_MOUNT_QUAT="[0.0,0.40354529635239006,0.0,0.9149596678498247]"
CAMERA_FRAME_QUAT="[-0.5,0.5,-0.5,0.5]"
CAMERA_SENSOR_OFFSET="[0.0576235,0.01753,0.42987]"
CAMERA_PITCH_DEG="0.0"

# Deterministic camera behavior (no hidden auto-fix offsets/yaw hacks).
export HOLOSOMA_CAMERA_STRICT_WARP=1
export HOLOSOMA_CAMERA_AUTOFIX_BACKWARD=0
export HOLOSOMA_CAMERA_EXTRA_YAW_DEG=0
export HOLOSOMA_CAMERA_DISABLE_OFFSETS=0

# Clean replay defaults.
export VISER_STRICT_CAMERA_RAYS=1
export VISER_SCANDOTS_SOURCE="live"
export VISER_ENABLE_CLIP_GUI=0
export VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-1}"
export HEADLESS="${ISAAC_HEADLESS_ENV}"

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "command:${COMMAND_PRESET}"
  "perception:${PERCEPTION_PRESET}"
  --training.seed="${SEED}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.num-envs="${NUM_ENVS}"
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-show-scandots=False
  --training.isaac-show-scandots=False
  --simulator.config.debug-viz=True
  --simulator.config.contact-force-viz=False
  --simulator.config.scene.env-spacing=0.0
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_FILE}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0
  --command.setup-terms.motion-command.params.motion-config.align-motion-to-init-yaw False
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion False
  --perception.camera-width=106
  --perception.camera-height=60
  --perception.camera-vfov-deg=58.6
  --perception.camera-hfov-deg=89.5
  --perception.camera-near=0.3
  --perception.camera-far=3.0
  --perception.max-distance=3.0
  --perception.camera-pitch-deg="${CAMERA_PITCH_DEG}"
  --perception.camera-include-robot-mesh=True
  --perception.camera-scandots-stride=1
  --perception.camera-body-name "${CAMERA_BODY_NAME}"
  --perception.camera-mount-quat "${CAMERA_MOUNT_QUAT}"
  --perception.camera-frame-quat "${CAMERA_FRAME_QUAT}"
  --perception.sensor-offset "${CAMERA_SENSOR_OFFSET}"
  terrain:terrain-load-obj
  --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
  --terrain.terrain-term.num-rows=1
  --terrain.terrain-term.num-cols=1
  randomization:disabled
)

echo "[INFO] Running strict depth replay"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
echo "[INFO] HEADLESS=${TRAINING_HEADLESS}"

printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
