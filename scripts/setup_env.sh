set -ex

# Adjust these to your setup
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
WORKSPACE_DIR=${WORKSPACE_DIR:-$HOME/workspace}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# Make conda available + activate env
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -n sim python=3.11 -y
conda activate sim

which python && python -V
ldd --version

pip install -U pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install pyperclip
pip install tyro
pip install loguru

git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.0 "$WORKSPACE_DIR/IsaacLab"

sudo apt-get update
sudo apt-get install -y cmake build-essential

cd "$WORKSPACE_DIR/IsaacLab"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
./isaaclab.sh --install

pip install -U pip
pip install -e $ROOT_DIR/src/holosoma[unitree,booster]


