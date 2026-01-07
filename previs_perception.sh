MOTION_FILE=${MOTION_FILE:-/ABS/PATH/TO/motion_mj.npz}
OUTPUT_DIR=${OUTPUT_DIR:-previs_depth}
HOLOSOMA_PREVIS_PERCEPTION_DIR="$OUTPUT_DIR" \
HOLOSOMA_PREVIS_PERCEPTION_STRIDE=1 \
python src/holosoma/holosoma/previs_perception.py \
  exp:g1-29dof-wbt \
  perception:camera_depth_d435i_rendered \
  --training.headless=True \
  --training.num_envs=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file "$MOTION_FILE"
