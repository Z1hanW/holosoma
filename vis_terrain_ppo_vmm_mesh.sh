#!/usr/bin/env bash
set -euo pipefail

HOLOSOMA_VISER_PORT=${HOLOSOMA_VISER_PORT:-6060} \
python src/holosoma/holosoma/viser_perception.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  perception:camera_depth_d435i \
  --training.num_envs=1 \
  --training.headless=True \
  --perception.camera_width=160 \
  --perception.camera_height=90 \
  --perception.camera_body_name=d435_joint \
  --perception.max_distance=10.0 \
  --perception.update_hz=30.0 \
  \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --terrain.terrain-term.num_rows=1 \
  --terrain.terrain-term.num_cols=1 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler=True \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale=0.77 \
  --simulator.config.scene.env_spacing=0.0
