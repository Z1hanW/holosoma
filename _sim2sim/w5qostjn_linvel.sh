#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIM2SIM_POLICY_NAME="w5qostjn_linvel"
export SIM2SIM_RUN_ID="c1gaknfu"
export SIM2SIM_MODEL_REF="https://wandb.ai/zihanw22/boxer/runs/${SIM2SIM_RUN_ID}"
export SIM2SIM_INFERENCE_CONFIG="g1-29dof-wbt-object-linvel-depth-distill"

exec bash "$SCRIPT_DIR/_common_mj_rollout.sh" "$@"
