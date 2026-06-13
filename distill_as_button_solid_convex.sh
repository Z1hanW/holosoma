#!/usr/bin/env bash
set -euo pipefail

# Launch AS drop-button distillation on the solid-object convex-hull bank.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_AS_SUCCESS155_BANK_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj"
export CORL_SOLID80_BANK_NAME="${CORL_SOLID80_BANK_NAME:-${DEFAULT_AS_SUCCESS155_BANK_NAME}_solid80_clean_box_bin_barrel_ball_convexhull}"

exec bash "${SCRIPT_DIR}/distill_as_button_solid.sh" "$@"
