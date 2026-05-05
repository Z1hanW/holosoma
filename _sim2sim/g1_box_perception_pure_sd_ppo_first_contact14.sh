#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIM2SIM_POLICY_NAME="g1_box_perception_pure_sd_ppo_first_contact14"
export SIM2SIM_RUN_ID="shoo7sr1"
export SIM2SIM_MODEL_REF="https://wandb.ai/zihanw22/boxer/runs/${SIM2SIM_RUN_ID}"
export SIM2SIM_INFERENCE_CONFIG="g1-29dof-wbt-object-distill"

exec bash "$SCRIPT_DIR/_common_mj_rollout.sh" "$@"
