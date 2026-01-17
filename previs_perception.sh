MOTION_FILE=${MOTION_FILE:-/ABS/PATH/TO/motion_mj.npz}
OUTPUT_DIR=${OUTPUT_DIR:-../pre_depth}
STRIDE=${STRIDE:-1}
ENABLE_VIDEO=${ENABLE_VIDEO:-1}
VIDEO_FPS=${VIDEO_FPS:-}
DEPTH_IMPL=${DEPTH_IMPL:-rendered}

case "${DEPTH_IMPL}" in
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i_rendered"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i_depth_sensor"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  *)
    echo "Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use rendered|depth_sensor|raycast." >&2
    exit 1
    ;;
esac

export HOLOSOMA_PREVIS_PERCEPTION_DIR="$OUTPUT_DIR"
export HOLOSOMA_PREVIS_PERCEPTION_STRIDE="$STRIDE"
export HOLOSOMA_PREVIS_PERCEPTION_VIDEO="$ENABLE_VIDEO"
if [[ -n "$VIDEO_FPS" ]]; then
  export HOLOSOMA_PREVIS_PERCEPTION_VIDEO_FPS="$VIDEO_FPS"
fi

python src/holosoma/holosoma/previs_perception.py \
  exp:g1-29dof-wbt \
  "perception:${PERCEPTION_PRESET}" \
  --training.headless=True \
  --training.num_envs=1 \
  --command.setup_terms.motion_command.params.motion_config.motion_file "$MOTION_FILE"
