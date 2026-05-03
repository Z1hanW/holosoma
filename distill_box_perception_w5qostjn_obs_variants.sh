#!/usr/bin/env bash
set -euo pipefail

# Run observation ablations on top of the w5qostjn / shoo7sr1-near03-debug setup.
#
# Usage:
#   bash distill_box_perception_w5qostjn_obs_variants.sh linvel [extra distill_box_perception.sh args...]
#   bash distill_box_perception_w5qostjn_obs_variants.sh action-history [extra distill_box_perception.sh args...]
#   bash distill_box_perception_w5qostjn_obs_variants.sh both [extra distill_box_perception.sh args...]
#
# Optional env:
#   RUN_NAME=...                         Override the default W&B run name.
#   STUDENT_ACTION_HISTORY_LENGTH=5      Action history length for action-history/both.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <linvel|action-history|both> [extra distill_box_perception.sh args...]" >&2
  exit 2
fi
shift

mode_norm="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
case "${mode_norm}" in
  linvel|linear_velocity|base_lin_vel|base_linear_velocity)
    variant="linvel"
    default_run_name="w5qostjn_linvel"
    ;;
  action_history|actions_history|action_hist|actions_hist)
    variant="action_history"
    default_run_name="w5qostjn_action_history"
    export STUDENT_ACTION_HISTORY_LENGTH="${STUDENT_ACTION_HISTORY_LENGTH:-5}"
    ;;
  both|linvel_action_history|linear_velocity_action_history|base_lin_vel_action_history)
    variant="linvel_action_history"
    default_run_name="w5qostjn_linvel_action_history"
    export STUDENT_ACTION_HISTORY_LENGTH="${STUDENT_ACTION_HISTORY_LENGTH:-5}"
    ;;
  *)
    echo "[ERROR] Unsupported mode '${MODE}'. Use linvel, action-history, or both." >&2
    exit 2
    ;;
esac

export SHOO7SR1_OBS_VARIANT="${variant}"
RUN_NAME="${RUN_NAME:-${default_run_name}}"

exec bash "${SCRIPT_DIR}/distill_box_perception.sh" \
  pure-sd \
  shoo7sr1-near03-debug \
  "${RUN_NAME}" \
  "$@"
