# Exit on error, and print commands
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Parse optional conda environment name argument
ENV_NAME=${1:-fcreal}

# Validate environment name (basic validation for conda environment naming)
if [[ ! "$ENV_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Error: Invalid environment name '$ENV_NAME'. Environment names must contain only letters, numbers, underscores, and hyphens."
  echo "Usage: $0 [ENV_NAME]"
  echo "  ENV_NAME: Optional name for the conda environment (default: fcreal)"
  exit 1
fi

echo "Setting up sim2real environment: $ENV_NAME"

OS=$(uname -s)
ARCH=$(uname -m)

case $ARCH in
  "aarch64"|"arm64") ARCH="aarch64" ;;
  "x86_64") ARCH="x86_64" ;;
  *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case $OS in
  "Linux")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh"
    PACKAGE_MANAGER="apt-get"
    INSTALL_CMD="sudo apt-get install -y"
    ;;
  "Darwin")
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    PACKAGE_MANAGER="brew"
    INSTALL_CMD="brew install"
    ;;
  *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Create overall workspace
source ${SCRIPT_DIR}/source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/$ENV_NAME

SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_finished_sim2real_v2_$ENV_NAME

mkdir -p $WORKSPACE_DIR

if [[ ! -f $SENTINEL_FILE ]]; then
  # Install swig based on OS
  if [[ $OS == "Linux" ]]; then
    $INSTALL_CMD swig
  elif [[ $OS == "Darwin" ]]; then
    # Install brew if needed
    if ! command -v brew &> /dev/null; then
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      echo >> $HOME/.zprofile
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> $HOME/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    $INSTALL_CMD swig
  fi

  # Install miniconda
  if [[ ! -d $CONDA_ROOT ]]; then
    mkdir -p $CONDA_ROOT
    curl $MINICONDA_URL -o $CONDA_ROOT/miniconda.sh
    bash $CONDA_ROOT/miniconda.sh -b -u -p $CONDA_ROOT
    rm $CONDA_ROOT/miniconda.sh
  fi

  # Create the conda environment
  if [[ ! -d $ENV_ROOT ]]; then
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    $CONDA_ROOT/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    $CONDA_ROOT/bin/conda install -y mamba -c conda-forge -n base
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n $ENV_NAME python=3.10 -c conda-forge --override-channels
  fi

  source $CONDA_ROOT/bin/activate $ENV_NAME

  # Install libstdcxx-ng to fix the error: `version `GLIBCXX_3.4.32' not found` on Ubuntu 24.04
  conda install -c conda-forge -y libstdcxx-ng

  # Install holosoma & holosoma_inference
  pip install -e $SCRIPT_DIR/holosoma
  pip install -e $SCRIPT_DIR/holosoma_ext
  pip install -e $SCRIPT_DIR/holosoma_inference[unitree]
  pip install -e $SCRIPT_DIR/holosoma_inference[booster]

  # Setup a few things for ARM64 Linux (G1 Jetson)
  # Otherwise we get this error:
  # /opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130: ...
  if [[ $OS == "Linux" && $ARCH == "aarch64" ]]; then
    sudo nvpmodel -m 0 2>/dev/null || true
    pip install pin>=3.8.0
  else
    if [[ ! -d $WORKSPACE_DIR/unitree_sdk2_python ]]; then
      git clone https://github.com/unitreerobotics/unitree_sdk2_python.git $WORKSPACE_DIR/unitree_sdk2_python
    fi
    pip install -e $WORKSPACE_DIR/unitree_sdk2_python/
    $CONDA_ROOT/bin/conda install pinocchio -y -c conda-forge --override-channels
  fi
  
  cd $SCRIPT_DIR
  touch $SENTINEL_FILE
fi
