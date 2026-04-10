if [[ -n "${CONDA_ROOT:-}" ]]; then
    export CONDA_ROOT
elif [[ -n "${WORKSPACE_DIR:-}" && -d "${WORKSPACE_DIR}/miniconda3" ]]; then
    export CONDA_ROOT="${WORKSPACE_DIR}/miniconda3"
elif [[ -d "${HOME}/.holosoma_deps/miniconda3" ]]; then
    export CONDA_ROOT="${HOME}/.holosoma_deps/miniconda3"
else
    export CONDA_ROOT="${HOME}/miniconda3"
fi
