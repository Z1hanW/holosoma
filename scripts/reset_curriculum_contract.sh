#!/usr/bin/env bash

# Shared, source-only validation for the reset-at-timestep-zero curricula used
# by the direct box launchers.  NUM_LEARNING_ITERATIONS is an exclusive upper
# bound: a fresh run with target N executes rollout iterations [0, N), so the
# last reachable curriculum endpoint is N-1.

_holosoma_reset_curriculum_error() {
  echo "[ERROR] $*" >&2
  return 2
}

_holosoma_canonicalize_uint32() {
  local variable_name="$1" allow_zero="$2" label="$3"
  local raw_value normalized leading_zero_prefix lower_bound=1
  local maximum=2147483647
  local LC_ALL=C

  [[ "${allow_zero}" == 1 ]] && lower_bound=0
  if ! [[ -v "${variable_name}" ]]; then
    _holosoma_reset_curriculum_error "${label} must be an ASCII integer in [${lower_bound}, 2147483647]. Got: <unset>"
    return 2
  fi
  raw_value=${!variable_name}
  if [[ ! "${raw_value}" =~ ^[0-9]+$ ]]; then
    _holosoma_reset_curriculum_error "${label} must be an ASCII integer. Got: ${raw_value}"
    return 2
  fi
  if (( ${#raw_value} > 64 )); then
    _holosoma_reset_curriculum_error "${label} is overlong (${#raw_value} digits)."
    return 2
  fi

  leading_zero_prefix=${raw_value%%[!0]*}
  normalized=${raw_value#"${leading_zero_prefix}"}
  [[ -n "${normalized}" ]] || normalized=0
  if [[ "${allow_zero}" != 1 && "${normalized}" == 0 ]]; then
    _holosoma_reset_curriculum_error "${label} must be positive. Got: ${raw_value}"
    return 2
  fi
  if (( ${#normalized} > ${#maximum} )) \
      || { (( ${#normalized} == ${#maximum} )) && [[ "${normalized}" > "${maximum}" ]]; }; then
    _holosoma_reset_curriculum_error "${label} must be <= ${maximum}. Got: ${raw_value}"
    return 2
  fi
  printf -v "${variable_name}" '%s' "${normalized}"
}

holosoma_canonicalize_positive_int32() {
  local variable_name="$1"
  _holosoma_canonicalize_uint32 \
    "${variable_name}" 0 "${variable_name}"
}

holosoma_configure_reset_curriculum() {
  local prefix="$1" target_name="$2"
  local end_value_name="${prefix}_END"
  local start_iter_name="${prefix}_START_ITER"
  local end_iter_name="${prefix}_END_ITER"
  local end_value="" start_iter="" end_iter=""
  local none_count=0 default_start default_end target full_resume=0

  holosoma_canonicalize_positive_int32 "${target_name}" || return
  target=${!target_name}
  [[ -v "${end_value_name}" ]] && end_value=${!end_value_name}
  [[ -v "${start_iter_name}" ]] && start_iter=${!start_iter_name}
  [[ -v "${end_iter_name}" ]] && end_iter=${!end_iter_name}

  [[ "${end_value}" == None ]] && ((none_count += 1))
  [[ "${start_iter}" == None ]] && ((none_count += 1))
  [[ "${end_iter}" == None ]] && ((none_count += 1))
  if (( none_count == 3 )); then
    return 0
  fi
  if (( none_count != 0 )); then
    _holosoma_reset_curriculum_error \
      "${end_value_name}, ${start_iter_name}, and ${end_iter_name} must either all be None or all use a numeric schedule."
    return 2
  fi

  default_end=$((target - 1))
  default_start=2500
  if (( default_end < default_start )); then
    default_start=0
  fi
  if [[ -z "${start_iter}" ]]; then
    printf -v "${start_iter_name}" '%s' "${default_start}"
  fi
  if [[ -z "${end_iter}" ]]; then
    printf -v "${end_iter_name}" '%s' "${default_end}"
  fi

  _holosoma_canonicalize_uint32 \
    "${start_iter_name}" 1 "${start_iter_name}" || return
  _holosoma_canonicalize_uint32 \
    "${end_iter_name}" 1 "${end_iter_name}" || return
  start_iter=${!start_iter_name}
  end_iter=${!end_iter_name}

  if (( start_iter > end_iter )); then
    _holosoma_reset_curriculum_error \
      "${start_iter_name} must be <= ${end_iter_name}; got ${start_iter}>${end_iter}."
    return 2
  fi
  if (( end_iter > target )); then
    _holosoma_reset_curriculum_error \
      "${end_iter_name} must be <= ${target_name}; got ${end_iter}>${target}."
    return 2
  fi

  if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME_CHECKPOINT:-}" ]]; then
    full_resume=1
  fi
  if (( end_iter == target && full_resume == 0 )); then
    _holosoma_reset_curriculum_error \
      "Fresh/policy-init ${end_iter_name} must be < ${target_name}; got ${end_iter}==${target}. Equality is accepted only for an explicit full-training resume."
    return 2
  fi
}

holosoma_configure_all_reset_curricula() {
  local target_name="$1"
  holosoma_configure_reset_curriculum \
    START_AT_TIMESTEP_ZERO_PROB "${target_name}" || return
  holosoma_configure_reset_curriculum \
    FREEZE_AT_TIMESTEP_ZERO_PROB "${target_name}"
}
