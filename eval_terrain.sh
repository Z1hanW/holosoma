python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=./model_02100.pt \
  --training.num_envs=1 \
  terrain:terrain-load-obj \
  --terrain.terrain-term.spawn.randomize_tiles=False \
  --terrain.terrain-term.obj-file-path src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj \
  --command.setup_terms.motion_command.params.motion_config.motion_file src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz
