# Exit on error, and print commands
set -ex

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

conda create -n retgt python=3.11 -y
CONDA_ROOT="/home/ubuntu/miniconda3"
WORKSPACE_DIR="$HOME/.holosoma_deps"
SENTINEL_FILE="$WORKSPACE_DIR/.env_setup_retargeting"

mkdir -p "$WORKSPACE_DIR"

if [[ ! -f "$SENTINEL_FILE" ]]; then
  # Validate conda root
  if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found under CONDA_ROOT=$CONDA_ROOT"
    echo "Try: conda info --base"
    exit 1
  fi

  # Enable conda in this script + activate env
  source "$CONDA_ROOT/etc/profile.d/conda.sh"


  conda activate retgt

  python -m pip install -U pip
  python -m pip install -e "$ROOT_DIR/src/holosoma_retargeting"

  touch "$SENTINEL_FILE"
fi