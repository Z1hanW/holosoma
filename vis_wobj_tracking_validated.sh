#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "[vis_wobj_tracking_validated] DISPLAY/WAYLAND_DISPLAY is not set. Run this from a graphical session." >&2
  exit 1
fi

export TRAINING_HEADLESS="${TRAINING_HEADLESS:-False}"
export SIM_DEBUG_VIZ="${SIM_DEBUG_VIZ:-True}"
export RUN_SECONDS="${RUN_SECONDS:-0}"

echo "[vis_wobj_tracking_validated] launching MuJoCo viewer"
echo "[vis_wobj_tracking_validated] press Ctrl+C to stop"
echo "[vis_wobj_tracking_validated] in the MuJoCo window, press 'c' to toggle object collision view and 'h' to dim object visuals"

exec "$ROOT_DIR/run_wobj_tracking_validated.sh" "$@"
