#!/usr/bin/env bash
set -euo pipefail

# Randomization presets (choose one):
#   randomization:g1-29dof-wbt
#   randomization:g1-29dof-wbt-w-object
#   randomization:g1-29dof
#   randomization:t1-29dof
RANDOMIZATION_PRESET="randomization:g1-29dof-wbt"

# Per-term toggles (uncomment to disable specific randomizers):
RANDOMIZATION_OVERRIDES=(
  # --randomization.setup_terms.push_randomizer_state.params.enabled=False
  # --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain=False
  # --randomization.setup_terms.actuator_randomizer_state.params.enable_rfi_lim=False
  # --randomization.setup_terms.setup_action_delay_buffers.params.enabled=False
  # --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled=False
  # --randomization.setup_terms.randomize_base_com_startup.params.enabled=False
  # --randomization.setup_terms.setup_dof_pos_bias.params.enabled=False
  # --randomization.reset_terms.randomize_push_schedule.params.enabled=False
  # --randomization.reset_terms.randomize_action_delay.params.enabled=False
  # --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias=False
  # --randomization.reset_terms.randomize_dof_state.params.joint_pos_bias_range='[0.0, 0.0]'
  # --randomization.step_terms.apply_pushes.params.enabled=False
)

# Clip weighting across multi-clip motion bank.
CLIP_WEIGHTING_STRATEGY="uniform_clip" # uniform_clip | uniform_step | success_rate_adaptive
CLIP_WEIGHTING_MIN_FACTOR="0.33"
CLIP_WEIGHTING_MAX_FACTOR="3.0"
CLIP_WEIGHTING_ARGS=(
  --command.setup_terms.motion_command.params.motion_config.clip_weighting_strategy="${CLIP_WEIGHTING_STRATEGY}"
)
if [[ "${CLIP_WEIGHTING_STRATEGY}" == "success_rate_adaptive" ]]; then
  CLIP_WEIGHTING_ARGS+=(
    --command.setup_terms.motion_command.params.motion_config.min_weight_factor="${CLIP_WEIGHTING_MIN_FACTOR}"
    --command.setup_terms.motion_command.params.motion_config.max_weight_factor="${CLIP_WEIGHTING_MAX_FACTOR}"
  )
fi

CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  "${RANDOMIZATION_PRESET}" \
  --training.num_envs=30720 \
  \
  --algo.config.actor_learning_rate=7e-5 \
  --algo.config.critic_learning_rate=7e-5 \
  --algo.config.normalize_actor_obs=False \
  --algo.config.normalize_critic_obs=False \
  --algo.config.save_interval=100 \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/robot_only/lafan \
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob=0.05 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s=0 \
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend=False \
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s=0 \
  "${CLIP_WEIGHTING_ARGS[@]}" \
  "${RANDOMIZATION_OVERRIDES[@]}" \
  logger:wandb \
  --logger.video.interval=1000
