#!/bin/bash
# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/hsmujoco
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_mujoco

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda (reuse existing logic)
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n hsmujoco python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate hsmujoco

  # Install system dependencies for MuJoCo
  # Note: These may require sudo access - document this requirement
  echo "Installing system dependencies for MuJoCo..."
  # sudo apt-get update
  # sudo apt-get install -y libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev

  # Install libstdcxx-ng to fix potential GLIBCXX issues
  conda install -c conda-forge -y libstdcxx-ng

  # Install ffmpeg for video encoding (consistent with other envs)
  conda install -c conda-forge -y ffmpeg

  # Install MuJoCo and related packages
  echo "Installing MuJoCo Python bindings..."
  pip install --upgrade pip
  
  # Core MuJoCo packages
  pip install 'mujoco>=3.0.0'
  pip install mujoco-python-viewer
  
  # Optional: Gymnasium MuJoCo environments (if needed for compatibility)
 # pip install "gymnasium[mujoco]"
  
  # Scientific computing stack (ensure compatibility)
  #pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  #pip install numpy scipy matplotlib

  # Install Unitree Python SDK for sim2sim
  if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
  fi
  pip install -e $WORKSPACE_DIR/unitree_sdk2_python/

  
  # Install Holosoma packages
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Validate MuJoCo installation
  echo "Validating MuJoCo installation..."
  python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
  python -c "import mujoco_viewer; print('MuJoCo viewer imported successfully')"
  
  # Create validation script for later testing
  cat > $WORKSPACE_DIR/validate_mujoco.py << 'EOF'
#!/usr/bin/env python3
"""Validation script for MuJoCo installation."""

import sys
import mujoco
import numpy as np

def validate_mujoco():
    """Validate MuJoCo installation with basic functionality test."""
    print(f"MuJoCo version: {mujoco.__version__}")
    
    # Test basic model creation
    xml_string = """
    <mujoco>
      <worldbody>
        <body name="box" pos="0 0 1">
          <geom type="box" size="0.1 0.1 0.1"/>
          <joint type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        
        # Test simulation step
        mujoco.mj_step(model, data)
        
        print("✓ Basic MuJoCo functionality validated")
        print(f"✓ Model has {model.nbody} bodies, {model.nq} DOFs")
        return True
        
    except Exception as e:
        print(f"✗ MuJoCo validation failed: {e}")
        return False

if __name__ == "__main__":
    success = validate_mujoco()
    sys.exit(0 if success else 1)
EOF

  # Run validation
  python $WORKSPACE_DIR/validate_mujoco.py
  
  touch $SENTINEL_FILE
  echo "MuJoCo environment setup completed successfully!"
fi

echo "MuJoCo environment is ready. Use 'source source_mujoco_setup.sh' to activate."
