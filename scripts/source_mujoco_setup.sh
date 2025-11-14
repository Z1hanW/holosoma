#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate fcmujoco

# Set MuJoCo-specific environment variables
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/fcmujoco/lib

# MuJoCo-specific environment variables (if needed)
# export MUJOCO_GL=egl  # For headless rendering
# export MUJOCO_GL=osmesa  # Alternative headless option

# Validate environment is properly activated
if python -c "import mujoco" 2>/dev/null; then
    echo "MuJoCo environment activated successfully"
    echo "MuJoCo version: $(python -c 'import mujoco; print(mujoco.__version__)')"
else
    echo "Warning: MuJoCo environment activation may have issues"
fi
