

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
eval "$(conda shell.bash hook)"
conda activate sim
WORKSPACE_DIR="./"
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_isaacsim

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then

  

  # Install ffmpeg for video encoding
  conda install -c conda-forge -y ffmpeg
  conda install -c conda-forge -y libiconv
  conda install -c conda-forge -y libglu

  # Below follows https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
  # Install IsaacSim
  pip install --upgrade pip
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

  # Install dependencies from PyPI first
  pip install pyperclip
  # Then install isaacsim from NVIDIA index only
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  
  git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 $WORKSPACE_DIR/IsaacLab


  sudo apt install -y cmake build-essential
  cd $WORKSPACE_DIR/IsaacLab
  # work-around for egl_probe cmake max version issue
  export CMAKE_POLICY_VERSION_MINIMUM=3.5
  ./isaaclab.sh --install

 # Install Holosoma
  pip install -U pip
  pip install -e $ROOT_DIR/src/holosoma[unitree,booster]

  # Force upgrade wandb to override rl-games constraint
  pip install --upgrade 'wandb>=0.21.1'
  touch $SENTINEL_FILE
fi
