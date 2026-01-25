#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable config
# =========================
CKPT="/home/ubuntu/FAR/model_52000.pt"
MOTION_DIR="/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/robot_only/lafan"
PORT=${PORT:-$((RANDOM % 8976 + 1024))}
HEADLESS=True

# =========================
# Run
# =========================
python vis_scripts/eval_agent_viser.py \
  --checkpoint "$CKPT" \
  --training.headless "$HEADLESS" \
  --command.setup_terms.motion_command.params.motion_config.motion_file "$MOTION_DIR" \
  --port "$PORT"
