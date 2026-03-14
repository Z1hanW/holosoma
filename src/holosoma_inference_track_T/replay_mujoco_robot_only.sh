#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

REPLAY_SCRIPT="${REPO_ROOT}/src/holosoma/holosoma/replay_motion_mujoco.py"
DEFAULT_MOTION="${REPO_ROOT}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"

MOTION_FILE=${MOTION_FILE:-"${DEFAULT_MOTION}"}
HEADLESS=${HEADLESS:-"True"}
LOOP=${LOOP:-"False"}
END_FRAME=${END_FRAME:-""}
LOG_EVERY=${LOG_EVERY:-10}

TS=$(date -u +%Y%m%d_%H%M%S)
POSE_LOG=${POSE_LOG:-"logs/replay_motion_mujoco/robot_only_pose_${TS}.csv"}

if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
  echo "ERROR: Could not find replay script: ${REPLAY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "ERROR: Motion file not found: ${MOTION_FILE}" >&2
  exit 1
fi

cmd=(
  python3 "${REPLAY_SCRIPT}"
  --motion-file "${MOTION_FILE}"
  --object-mode disabled
  --headless "${HEADLESS}"
  --loop "${LOOP}"
  --log-pose-every-n-frames "${LOG_EVERY}"
  --pose-log-csv "${POSE_LOG}"
)

if [[ -n "${END_FRAME}" ]]; then
  cmd+=(--end-frame "${END_FRAME}")
fi

echo "[replay_mujoco_robot_only] motion=${MOTION_FILE}"
echo "[replay_mujoco_robot_only] pose_log=${POSE_LOG}"
"${cmd[@]}"
