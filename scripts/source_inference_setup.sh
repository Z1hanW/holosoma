#!/usr/bin/env bash
# Detect script directory (works in both bash and zsh)
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
elif [ -n "${ZSH_VERSION:-}" ]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%x}" )" &> /dev/null && pwd )
fi
source "${SCRIPT_DIR}/source_common.sh"
source "${CONDA_ROOT}/bin/activate" hsinference

CLIP="${1:-${HOLOSOMA_MJ_CLIP:-box_75}}"
CLIP="${CLIP%.npz}"
export HOLOSOMA_MJ_CLIP="$CLIP"
export HOLOSOMA_MJ_MOTION="${HOLOSOMA_MJ_MOTION:-data_demo/${CLIP}.npz}"
export HOLOSOMA_MJ_MODEL="${HOLOSOMA_MJ_MODEL:-logs/wandb_runs/shoo7sr1/model_29999.onnx}"
if [ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]; then
    [ "$CLIP" = "box_75" ] && export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=5 || export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
fi

export PYTHONSAFEPATH=1
export PYTHONPATH="src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CONDA_ROOT}/envs/hsinference/lib/python3.10/site-packages/lib"
export HOLOSOMA_POLICY_TTY_INPUT="${HOLOSOMA_POLICY_TTY_INPUT:-1}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND="${HOLOSOMA_KEYBOARD_ROOT_COMMAND:-1}"
export HOLOSOMA_SKIP_STIFF_PROMPT="${HOLOSOMA_SKIP_STIFF_PROMPT:-1}"
