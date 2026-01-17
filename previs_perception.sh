MOTION_FILE=${MOTION_FILE:-src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz}
OUTPUT_DIR=${OUTPUT_DIR:-../pre_depth}
STRIDE=${STRIDE:-1}
ENABLE_VIDEO=${ENABLE_VIDEO:-1}
SAVE_NPY=${SAVE_NPY:-0}
SAVE_PNG=${SAVE_PNG:-1}
SAVE_RGB_PNG=${SAVE_RGB_PNG:-1}
RGB_VIDEO=${RGB_VIDEO:-1}
VIDEO_FPS=${VIDEO_FPS:-}
DEPTH_IMPL=${DEPTH_IMPL:-rendered}
TERRAIN_OBJ=${TERRAIN_OBJ:-src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj}
TERRAIN_ROWS=${TERRAIN_ROWS:-1}
TERRAIN_COLS=${TERRAIN_COLS:-1}

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
export HOLOSOMA_PREVIS_PERCEPTION_SAVE_NPY="$SAVE_NPY"
export HOLOSOMA_PREVIS_PERCEPTION_SAVE_PNG="$SAVE_PNG"
export HOLOSOMA_PREVIS_PERCEPTION_SAVE_RGB_PNG="$SAVE_RGB_PNG"
export HOLOSOMA_PREVIS_PERCEPTION_RGB_VIDEO="$RGB_VIDEO"
if [[ -n "$VIDEO_FPS" ]]; then
  export HOLOSOMA_PREVIS_PERCEPTION_VIDEO_FPS="$VIDEO_FPS"
fi

python src/holosoma/holosoma/previs_perception.py \
  exp:g1-29dof-wbt \
  "perception:${PERCEPTION_PRESET}" \
  --training.headless=True \
  --training.num_envs=1 \
  \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path "$TERRAIN_OBJ" \
  --terrain.terrain-term.num_rows "$TERRAIN_ROWS" \
  --terrain.terrain-term.num_cols "$TERRAIN_COLS" \
  \
  --command.setup_terms.motion_command.params.motion_config.motion_file "$MOTION_FILE"
