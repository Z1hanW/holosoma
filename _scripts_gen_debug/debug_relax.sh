#!/usr/bin/env bash
set -euo pipefail

# Same launcher as train_object_generalist_ds.sh, with only global root-position
# tracking relaxed. CLI overrides are appended last so they win over the forced
# GT/u5 reward alignment inside the base launcher.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_SEQUENCE_NAME=${SEQUENCE_NAME:-debug-relax-rootpos-w025-sigma06}
ROOT_POS_RELAX_WEIGHT=${ROOT_POS_RELAX_WEIGHT:-${ROOT_POS_WEIGHT:-0.25}}
ROOT_POS_RELAX_SIGMA=${ROOT_POS_RELAX_SIGMA:-${ROOT_POS_SIGMA:-0.6}}
export REFERENCE_ROOT_POS_W="${ROOT_POS_RELAX_WEIGHT}"
export REFERENCE_ROOT_POS_SIGMA="${ROOT_POS_RELAX_SIGMA}"

TRAIN_ARGS=()

prepare_train_args() {
  local default_sequence_name="$1"
  shift
  local args=("$@")
  local first_arg_normalized

  while ((${#args[@]} > 0)); do
    case "${args[0]}" in
      --data-subset-mode|--sample-mode|--data-subset-seed|--data-subset-bank-root)
        if ((${#args[@]} < 2)); then
          echo "[ERROR] ${args[0]} requires a value." >&2
          exit 2
        fi
        TRAIN_ARGS+=("${args[0]}" "${args[1]}")
        args=("${args[@]:2}")
        ;;
      --data-subset-mode=*|--sample-mode=*|--data-subset-seed=*|--data-subset-bank-root=*)
        TRAIN_ARGS+=("${args[0]}")
        args=("${args[@]:1}")
        ;;
      *)
        break
        ;;
    esac
  done

  if ((${#args[@]} > 0)); then
    first_arg_normalized=$(printf '%s' "${args[0]}" | tr '[:upper:]' '[:lower:]')
    case "${first_arg_normalized}" in
      pure-sd|pure-ds|pure-real|pure-omomo|mix-naive|mix-curriculum|mix-clean-noisy|mix-curr|fix-omomo-quater|fix_omomo_quater|fix-real|fixed-real|fix_real|fixed_real|resume|resume-mix-naive|mix-naive-resume)
        TRAIN_ARGS+=("${args[0]}")
        args=("${args[@]:1}")
        ;;
    esac
  fi

  if ((${#args[@]} == 0)) || [[ "${args[0]}" == --* ]]; then
    TRAIN_ARGS+=("${default_sequence_name}")
  fi

  TRAIN_ARGS+=("${args[@]}")
}

prepare_train_args "${DEFAULT_SEQUENCE_NAME}" "$@"

exec bash "${REPO_DIR}/train_object_generalist_ds.sh" "${TRAIN_ARGS[@]}"
