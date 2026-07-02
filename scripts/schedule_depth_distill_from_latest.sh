#!/usr/bin/env bash
set -euo pipefail

# Wait for the current tracking run to mature, then start depth-student
# distillation from the latest local teacher checkpoint.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_DIR}"

DELAY_SECONDS="${DELAY_SECONDS:-25200}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_CHECKPOINT_WAIT_SECONDS="${MAX_CHECKPOINT_WAIT_SECONDS:-0}"
TRACKING_SESSION="${TRACKING_SESSION:-csp_multiterrain_heightmapwbt_20260630_053640}"
STOP_TRACKING="${STOP_TRACKING:-1}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-180}"
KILL_TRACKING_AFTER_GRACE="${KILL_TRACKING_AFTER_GRACE:-1}"
RUN_NAME_FILE="${RUN_NAME_FILE:-logs/run_commands/${TRACKING_SESSION}.run_name}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-logs/holosomatest}"

latest_checkpoint_for_run() {
  local run_name="$1"
  python - "$CHECKPOINT_ROOT" "$run_name" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
pattern = re.compile(r"model_(\d+)\.pt$")
matches: list[tuple[int, str]] = []
for path in root.glob(f"*-{run_name}-locomotion/model_*.pt"):
    match = pattern.search(path.name)
    if match:
        matches.append((int(match.group(1)), str(path)))
if matches:
    print(max(matches, key=lambda item: item[0])[1])
PY
}

stop_tracking_session() {
  if [[ "${STOP_TRACKING}" != "1" ]]; then
    echo "STOP_TRACKING=${STOP_TRACKING}; leaving tracking session running."
    return
  fi
  if ! tmux has-session -t "${TRACKING_SESSION}" 2>/dev/null; then
    echo "Tracking session ${TRACKING_SESSION} is not running."
    return
  fi

  echo "Stopping tracking session ${TRACKING_SESSION} before depth distillation."
  tmux send-keys -t "${TRACKING_SESSION}" C-c
  local waited=0
  while tmux has-session -t "${TRACKING_SESSION}" 2>/dev/null; do
    if (( waited >= STOP_GRACE_SECONDS )); then
      if [[ "${KILL_TRACKING_AFTER_GRACE}" == "1" ]]; then
        echo "Tracking session did not exit after ${STOP_GRACE_SECONDS}s; killing tmux session."
        tmux kill-session -t "${TRACKING_SESSION}" || true
      else
        echo "Tracking session still running after grace period; continuing without killing."
      fi
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

echo "Depth distill scheduler started at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
echo "Sleeping ${DELAY_SECONDS}s before selecting teacher checkpoint."
sleep "${DELAY_SECONDS}"

if [[ ! -f "${RUN_NAME_FILE}" ]]; then
  echo "Missing run_name file: ${RUN_NAME_FILE}" >&2
  exit 1
fi
TRACKING_RUN_NAME="$(tr -d '\n' < "${RUN_NAME_FILE}")"
echo "Tracking run_name: ${TRACKING_RUN_NAME}"

checkpoint_waited=0
TEACHER_CHECKPOINT=""
while [[ -z "${TEACHER_CHECKPOINT}" ]]; do
  TEACHER_CHECKPOINT="$(latest_checkpoint_for_run "${TRACKING_RUN_NAME}")"
  if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
    break
  fi
  if (( MAX_CHECKPOINT_WAIT_SECONDS > 0 && checkpoint_waited >= MAX_CHECKPOINT_WAIT_SECONDS )); then
    echo "No checkpoint found for ${TRACKING_RUN_NAME} under ${CHECKPOINT_ROOT}." >&2
    exit 1
  fi
  echo "No checkpoint found yet; polling again in ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
  checkpoint_waited=$((checkpoint_waited + POLL_SECONDS))
done

echo "Selected teacher checkpoint: ${TEACHER_CHECKPOINT}"
stop_tracking_session

export TEACHER_CHECKPOINT
export TRAINING_MODE="${TRAINING_MODE:-hybrid}"
export RUN_NAME="${RUN_NAME:-$(hostname)_g1_29dof_depth_student_${TRAINING_MODE}_distill_from_latest_tracking_$(date -u +%Y%m%d_%H%M%S)}"
echo "Starting depth distillation with run_name=${RUN_NAME}"
./csp_depth_distill.sh
