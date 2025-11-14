SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate fcreal
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT}/envs/fcreal/lib/python3.10/site-packages/lib

# Check UFW status if ufw command exists
if command -v ufw >/dev/null 2>&1; then
    if sudo ufw status | grep -q "Status: inactive"; then
        echo "✓ UFW disabled"
    else
        echo "Warning: UFW is currently enabled."
    fi
fi
