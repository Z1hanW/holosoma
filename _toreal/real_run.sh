#!/usr/bin/env bash
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
python3 /src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-wbt-object-perception-no-linvel \
  --task.model-path /home/user/FAR/PHolosoma/_ckps/w5qostjn_model_20000.onnx \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface eth0
