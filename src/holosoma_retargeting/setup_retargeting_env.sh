# Exit on error, and print commands
set -ex
 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
 
# Create overall workspace
source ${SCRIPT_DIR}/../../source_common.sh
ENV_ROOT=$CONDA_ROOT/envs/rt_env
SENTINEL_FILE=${WORKSPACE_DIR}/.env_setup_retargeting
 
mkdir -p $WORKSPACE_DIR
 
if [[ ! -f $SENTINEL_FILE ]]; then
  # Install miniconda
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
    MAMBA_ROOT_PREFIX=$CONDA_ROOT $CONDA_ROOT/bin/mamba create -y -n rt_env python=3.11 -c conda-forge --override-channels
  fi
 
  source $CONDA_ROOT/bin/activate rt_env
 
  # Install FALCON/humanoidverse/retargeting/
  cd $SCRIPT_DIR
  pip install -U pip
  pip install -e .
  touch $SENTINEL_FILE
fi