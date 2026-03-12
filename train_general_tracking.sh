#!/usr/bin/env bash
set -euo pipefail

# Robot-only general tracking launcher.
# Loads all robot-only AMASS motions from:
#   src/holosoma_retargeting/converted_res/robot_only/amass_all_trainready
# and delegates to train_amass_base.sh, which already runs:
# - exp:g1-29dof-wbt
# - no object
# - no terrain pairing
# - no perception

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/robot_only/amass_all_trainready"

exec env \
  EXP="${EXP:-g1-29dof-wbt}" \
  MOTION_DIR="${MOTION_DIR:-${DEFAULT_MOTION_DIR}}" \
  TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_general_tracking_amass}" \
  LOGGER_NAME="${LOGGER_NAME:-general_tracking_29dof_wbt}" \
  bash "${SCRIPT_DIR}/train_amass_base.sh" "$@"
