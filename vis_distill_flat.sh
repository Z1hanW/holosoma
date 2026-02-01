#!/usr/bin/env bash
set -euo pipefail

# Physics rollout with Viser for distill checkpoints on flat terrain.
#
# Required:
#   CKPT=/abs/path/to/model.pt
#
# Optional:
#   MOTION_DIR=/abs/path/to/holosoma/src/holosoma_retargeting/converted_res/robot_only/lafan
#   NUM_ENVS=4
#   HEADLESS=True
#   PAIR_TERRAIN=False
#   START_AT_TIMESTEP_ZERO_PROB=0.05
#   ENABLE_DEFAULT_POSE_APPEND=False
#   DEFAULT_POSE_APPEND_DURATION_S=0
#   ENABLE_DEFAULT_POSE_PREPEND=False
#   DEFAULT_POSE_PREPEND_DURATION_S=0
#   VISER_PORT=####
#   VISER_ENV_ID=0
#   VISER_UPDATE_HZ=30
#   VISER_RECENTER=False

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CKPT=${CKPT:-"/ABS/PATH/to/model.pt"}
MOTION_DIR=${MOTION_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/robot_only/lafan"}
NUM_ENVS=${NUM_ENVS:-4}
HEADLESS=${HEADLESS:-True}
PAIR_TERRAIN=${PAIR_TERRAIN:-False}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.05}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-False}

if [[ "${CKPT}" == "/ABS/PATH/to/model.pt" ]]; then
  echo "Set CKPT to your checkpoint path." >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

echo "[INFO] Viser port: ${VISER_PORT}"

python -m holosoma.visualize physics \
  --checkpoint "${CKPT}" \
  --motion-dir "${MOTION_DIR}" \
  --num-envs "${NUM_ENVS}" \
  --headless "${HEADLESS}" \
  --pair-terrain-with-motion "${PAIR_TERRAIN}" \
  --viser-port "${VISER_PORT}" \
  --viser-env-id "${VISER_ENV_ID}" \
  --viser-update-hz "${VISER_UPDATE_HZ}" \
  --viser-recenter "${VISER_RECENTER}" \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}" \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append "${ENABLE_DEFAULT_POSE_APPEND}" \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s "${DEFAULT_POSE_APPEND_DURATION_S}" \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${ENABLE_DEFAULT_POSE_PREPEND}" \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${DEFAULT_POSE_PREPEND_DURATION_S}"
