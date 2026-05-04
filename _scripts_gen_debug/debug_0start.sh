#!/usr/bin/env bash
set -euo pipefail

# Same launcher as train_object_generalist_ds.sh, with reset sampling forced to
# timestep zero for every sequence. Freeze-at-zero is forced off so clips start
# at zero and then advance normally.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

DEFAULT_SEQUENCE_NAME=${SEQUENCE_NAME:-debug-0start}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
START_AT_TIMESTEP_ZERO_PROB_END=${START_AT_TIMESTEP_ZERO_PROB_END:-${START_AT_TIMESTEP_ZERO_PROB}}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB_END=${FREEZE_AT_TIMESTEP_ZERO_PROB_END:-${FREEZE_AT_TIMESTEP_ZERO_PROB}}
export FREEZE_AT_TIMESTEP_ZERO_PROB

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

exec bash "${REPO_DIR}/train_object_generalist_ds.sh" "${TRAIN_ARGS[@]}" \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob="${START_AT_TIMESTEP_ZERO_PROB}" \
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end="${START_AT_TIMESTEP_ZERO_PROB_END}" \
  --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end="${FREEZE_AT_TIMESTEP_ZERO_PROB_END}"
