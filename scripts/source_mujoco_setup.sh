#!/usr/bin/env bash
# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi
source "${SCRIPT_DIR}/source_common.sh"
source "${CONDA_ROOT}/bin/activate" hsmujoco

CLIP="${1:-${HOLOSOMA_MJ_CLIP:-box_75}}"
CLIP="${CLIP%.npz}"
export HOLOSOMA_MJ_CLIP="$CLIP"
export HOLOSOMA_MJ_MOTION="${HOLOSOMA_MJ_MOTION:-data_demo/${CLIP}.npz}"
export HOLOSOMA_MJ_OBJECT_URDF="${HOLOSOMA_MJ_OBJECT_URDF:-data_demo/objects/${CLIP}.urdf}"
if [ -z "${HOLOSOMA_MJ_OBJECT_MASS:-}" ]; then
    [ "$CLIP" = "box_75" ] && export HOLOSOMA_MJ_OBJECT_MASS=2.0 || export HOLOSOMA_MJ_OBJECT_MASS=1.0
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="src/holosoma${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_ROOT}/envs/hsmujoco/lib"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS:-1}"
export HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS="${HOLOSOMA_MUJOCO_CARRY_ARM_OBJECT_CONTACTS:-1}"
export HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES="${HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES:-1}"
