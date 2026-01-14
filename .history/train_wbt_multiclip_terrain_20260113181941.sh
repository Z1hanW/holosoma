#!/usr/bin/env bash
set -euo pipefail

# Multi-motion tracking with per-clip terrain OBJ pairing.
# - Motion clips: .npz files in MOTION_DIR
# - Terrain tiles: .obj files in OBJ_DIR with the same stem as motion files
# - OBJ count must equal num_rows * num_cols

MOTION_DIR="/ABS/PATH/converted_res/robot_only/lafan"
OBJ_DIR="/ABS/PATH/terrain_objs/lafan"
NUM_ROWS=1
NUM_COLS=1

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  terrain:terrain-load-obj \
  --training.num_envs=30720 \
  --simulator.config.scene.env_spacing=0.0 \
  --terrain.terrain-term.obj-file-path "${OBJ_DIR}" \
  --terrain.terrain-term.num-rows "${NUM_ROWS}" \
  --terrain.terrain-term.num-cols "${NUM_COLS}" \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=1000 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}" \
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  logger:wandb \
  --logger.video.interval=1000 \
  --logger.name="g1_videomimic_multiclip_terrain"
