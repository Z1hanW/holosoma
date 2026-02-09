#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

DATA_PATH=${1:-"demo_data/far_robot"}
TASK_NAME=${2:-"far_robot"}
OBJECT_NAME=${3:-"stairs"}
OBJECT_DIR=${4:-"$SCRIPT_DIR/demo_data/far_robot/far_robot"}
ROBOT_URDF=${5:-"models/g1/g1_29dof.urdf"}
DATA_FORMAT=${6:-"smplx"}
SAVE_DIR=${7:-""}
SCENE_XML_FILE=${8:-""}
SAVE_MODE=${SAVE_MODE:-"False"}

SAVE_DIR_ARG=()
if [ -n "$SAVE_DIR" ]; then
    SAVE_DIR_ARG=(--save_dir "$SAVE_DIR")
fi

SCENE_XML_ARG=()
if [ -n "$SCENE_XML_FILE" ]; then
    SCENE_XML_ARG=(--task-config.scene-xml-file "$SCENE_XML_FILE")
fi

python examples/robot_retarget.py \
    --data_path "$DATA_PATH" \
    --task-type climbing \
    --task-name "$TASK_NAME" \
    --task-config.object_name "$OBJECT_NAME" \
    --task-config.object-dir "$OBJECT_DIR" \
    --task-config.human-z-offset 0.1 \
    "${SCENE_XML_ARG[@]}" \
    --robot-config.robot-urdf-file "$ROBOT_URDF" \
    --data_format "$DATA_FORMAT" \
    "${SAVE_DIR_ARG[@]}" \
    --save-mode "$SAVE_MODE" \
    --retargeter.penetration-tolerance 0.0 \
    --retargeter.step-size 0.05 \
    --retargeter.debug \
    --retargeter.visualize


    # interaction_mesh_foot

# python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
#    --object_urdf demo_data/far_robot/far_robot/stairs.urdf \
#    --qpos_npz demo_results/g1/climbing/mocap_climb/far_robot_original.npz
