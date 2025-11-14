SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/source_common.sh
source ${CONDA_ROOT}/bin/activate fclab
export OMNI_KIT_ACCEPT_EULA=1
