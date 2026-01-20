
RANDOMIZATION_PRESET="randomization:g1-29dof-wbt"
RANDOMIZATION_OVERRIDES=(
  --randomization.setup_terms.push_randomizer_state.params.enabled=False
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

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=$((29500 + RANDOM % 1000)) src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-videomimic-mlp \
  "${RANDOMIZATION_PRESET_ARGS[@]}" \
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
