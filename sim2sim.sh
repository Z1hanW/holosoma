 python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-videomimic \
  --task.model-path /home/ANT.AMAZON.COM/zzzihanw/Downloads/prefill_pose/prefill_pose/model_06600.onnx \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.interface lo  # use lo0 on macOS