#!/bin/bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-motion-tracking-transformer \
  --training.num_envs=8192 \
  --simulator.config.sim.max_episode_length_s=20.0 \
  --algo.config.module_dict.actor.layer_config.encoder_activation=GELU \
  --algo.config.module_dict.critic.layer_config.encoder_activation=GELU \
  --command.setup_terms.motion_command.params.motion_config.num_future_steps=10 \
  --command.setup_terms.motion_command.params.motion_config.target_pose_type=max-coords-future-rel-with-time \
  --algo.config.module_dict.actor.layer_config.encoder_num_steps=10 \
  --algo.config.module_dict.critic.layer_config.encoder_num_steps=10 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz \
  --reward.terms.limits_dof_pos.weight=-10.0 \
  --reward.terms.action_rate_l2.weight=-0.02 \
  --reward.terms.undesired_contacts.weight=-0.2 \
  --termination.terms.bad_tracking.params.bad_ref_pos_threshold=0.8 \
  --termination.terms.bad_tracking.params.bad_ref_ori_threshold=1.2 \
  --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold=0.4 \
  --randomization.setup_terms.push_randomizer_state.params.enabled=False \
  --randomization.reset_terms.randomize_push_schedule.params.enabled=False \
  --randomization.step_terms.apply_pushes.params.enabled=False \
  --randomization.setup_terms.randomize_base_com_startup.params.enabled=False \
  --randomization.setup_terms.setup_dof_pos_bias.params.enabled=False \
  --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias=False \
  --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain=False \
  --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled=False
