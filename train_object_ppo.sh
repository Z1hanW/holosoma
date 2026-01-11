python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-motion-tracking-transformer \
  --training.num_envs=4 \
  --training.headless=False \
  --algo.config.actor_learning_rate=1e-4 \
  --algo.config.critic_learning_rate=1e-4 \
  --command.setup_terms.motion_command.params.motion_config.num_future_steps=10 \
  --command.setup_terms.motion_command.params.motion_config.target_pose_type=max-coords-future-rel-with-time \
  --algo.config.module_dict.actor.layer_config.encoder_num_steps=10 \
  --algo.config.module_dict.critic.layer_config.encoder_num_steps=10 \
  --robot.object.object_urdf_path=src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.urdf \
  --command.setup_terms.motion_command.params.motion_config.motion_file=src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --randomization.setup_terms.push_randomizer_state.params.enabled=False \
  logger:wandb \
  --logger.video.enabled=False \
  --simulator.config.scene.env_spacing=0.0



