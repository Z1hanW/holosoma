#!/usr/bin/env bash
set -euo pipefail

# Distill an AS/OMOMO real-mesh generalist teacher into a depth-perception student.
#
# The teacher is expected to be a checkpoint produced by train_as_general.sh.
# This wrapper mirrors train_as_general.sh's local AS/OMOMO data validation and
# delegates the actual perception distillation launch to distill_box_perception.sh.
#
# Usage:
#   bash distill_as_perception.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [schedule/run_name/extra args...]
#   TEACHER_CHECKPOINT=<teacher_checkpoint> bash distill_as_perception.sh [schedule/run_name/extra args...]

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_perception.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  TEACHER_CHECKPOINT=<teacher_checkpoint> bash distill_as_perception.sh [extra args...]
  bash distill_as_perception.sh success133 [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]

Examples:
  bash distill_as_perception.sh /data/logs_new/carry-any/<run>/model_01000.pt
  bash distill_as_perception.sh wandb://<entity>/carry-any/<run_id>/model_01000.pt
  bash distill_as_perception.sh https://wandb.ai/<entity>/carry-any/runs/<run_id>
  bash distill_as_perception.sh /abs/model.pt ppo-first run:as_depth_student

This launcher defaults to PPO+DAgger ppo-first distillation and always uses
the repo-local AS/OMOMO real-mesh bank by default:
  OMOMO_DATA_DIR=./data/ds_as_data/omomo
  OMOMO_OBJECT_MAP=./data/ds_as_data/omomo/_clip_object_urdf_map.json

If no teacher checkpoint is passed, the default teacher is the latest model
from:
  https://wandb.ai/zihanw22/carry-any/runs/bcleb5oi

Strict policy initialization is only supported through an architecture-specific
wrapper such as distill_as_button.sh. The generic actor input contract is not
shape-compatible with the default box-button checkpoint.

For the teacher-rollout filtered 133-clip AS bank:
  bash cp_tao.sh success133
  bash distill_as_perception.sh success133
The success133 mode uses contact-aware student inputs by default.
The contact-aware-history selector is a fixed deployment contract with five
proprio/action frames; other history lengths are rejected.
EOF
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    return 1
  fi
  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 || "${parts[2]}" != "runs" ]]; then
    return 1
  fi
  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[3]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi
  if [[ "${#parts[@]}" -ge 6 && "${parts[4]}" == "files" ]]; then
    explicit_file="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  fi
  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_uri() {
  local ref="$1"
  if [[ "${ref}" != wandb://* ]]; then
    return 1
  fi
  local trimmed="${ref#wandb://}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 3 ]]; then
    return 1
  fi
  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[2]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi
  if [[ "${#parts[@]}" -gt 3 ]]; then
    explicit_file="${trimmed#${entity}/${project}/${run_id}/}"
  fi
  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_reference() {
  local ref="$1"
  parse_wandb_run_url "${ref}" || parse_wandb_uri "${ref}"
}

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  local requested_step="${4:-}"
  "${PYTHON_BIN:-python}" - "${entity}" "${project}" "${run_id}" "${requested_step}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sys.path = [
    entry
    for entry in sys.path
    if entry not in {"", "."} and Path(entry).resolve() != repo_root
]

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id, requested_step = sys.argv[1:5]
requested_step_int = int(requested_step) if requested_step else None
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
pattern = re.compile(r"^model_(\d+)\.pt$")
best: tuple[int, str] | None = None
for file_obj in run.files():
    name = str(getattr(file_obj, "name", "") or "")
    match = pattern.match(name)
    if match is None:
        continue
    step = int(match.group(1))
    if requested_step_int is not None:
        if step == requested_step_int:
            print(name)
            sys.exit(0)
        continue
    try:
        size = int(getattr(file_obj, "size", 0) or 0)
    except Exception:
        size = 0
    if size <= 0:
        continue
    candidate = (step, name)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is not None and requested_step_int is None:
    print(best[1])
PY
}

normalize_wandb_checkpoint_ref() {
  local ref="$1"
  local requested_model_file="${2:-}"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${requested_model_file}"

  parsed="$(parse_wandb_reference "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "${RESUME_STEP:-}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved W&B reference to checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL, set BOX_RESUME_MODEL_FILE/WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_EXPLICIT=0
[[ -n "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION+x}" ]] \
  && CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_EXPLICIT=1
SCHEDULE_NAME_USER_SET=0
[[ -n "${SCHEDULE_NAME+x}" ]] && SCHEDULE_NAME_USER_SET=1
SCHEDULE_NOTES_USER_SET=0
[[ -n "${SCHEDULE_NOTES+x}" ]] && SCHEDULE_NOTES_USER_SET=1

AS_SUCCESS133_FINAL0P5=${AS_SUCCESS133_FINAL0P5:-0}
AS_CONTACT_AWARE=${AS_CONTACT_AWARE:-${CONTACT_AWARE:-}}
AS_CONTACT_AWARE_HISTORY=${AS_CONTACT_AWARE_HISTORY:-0}
AS_POSITIONAL_SCHEDULE_SELECTOR=""
AS_POSITIONAL_CONTACT_SELECTOR=""

consume_as_mode_prefix() {
  AS_MODE_PREFIX_CONSUMED=0
  local normalized=""
  while [[ $# -gt 0 ]]; do
    normalized="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "${normalized}" in
      resume-from-box|resume_from_box)
        RESUME_FROM_BOX=1
        ;;
      success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
        AS_SUCCESS133_FINAL0P5=1
        ;;
      contact-aware|contact_aware|contactaware)
        if [[ -n "${AS_POSITIONAL_CONTACT_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS contact modes were provided: ${AS_POSITIONAL_CONTACT_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_CONTACT_SELECTOR="${normalized}"
        AS_CONTACT_AWARE=1
        ;;
      contact-aware-history|contact_aware_history|contactaware-history|contactaware_history)
        if [[ -n "${AS_POSITIONAL_CONTACT_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS contact modes were provided: ${AS_POSITIONAL_CONTACT_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_CONTACT_SELECTOR="${normalized}"
        AS_CONTACT_AWARE=1
        AS_CONTACT_AWARE_HISTORY=1
        ;;
      no-contact-aware|no_contact_aware|no-contactaware|no_contactaware)
        if [[ -n "${AS_POSITIONAL_CONTACT_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS contact modes were provided: ${AS_POSITIONAL_CONTACT_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_CONTACT_SELECTOR="${normalized}"
        AS_CONTACT_AWARE=0
        AS_CONTACT_AWARE_HISTORY=0
        ;;
      default)
        if [[ -n "${AS_POSITIONAL_SCHEDULE_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS schedule variants were provided: ${AS_POSITIONAL_SCHEDULE_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_SCHEDULE_SELECTOR="${normalized}"
        SCHEDULE_VARIANT=default
        ;;
      dagger_mix|dagger-mix|daggermix)
        if [[ -n "${AS_POSITIONAL_SCHEDULE_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS schedule variants were provided: ${AS_POSITIONAL_SCHEDULE_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_SCHEDULE_SELECTOR="${normalized}"
        SCHEDULE_VARIANT=dagger_mix
        ;;
      dag_first|dag-first|dagger-first)
        if [[ -n "${AS_POSITIONAL_SCHEDULE_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS schedule variants were provided: ${AS_POSITIONAL_SCHEDULE_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_SCHEDULE_SELECTOR="${normalized}"
        SCHEDULE_VARIANT=dag_first
        ;;
      ppo_first|ppo-first)
        if [[ -n "${AS_POSITIONAL_SCHEDULE_SELECTOR}" ]]; then
          echo "[ERROR] Multiple positional AS schedule variants were provided: ${AS_POSITIONAL_SCHEDULE_SELECTOR}, ${normalized}." >&2
          exit 2
        fi
        AS_POSITIONAL_SCHEDULE_SELECTOR="${normalized}"
        SCHEDULE_VARIANT=ppo_first
        ;;
      as|as-perception|as_perception|omomo|omomo-real|omomo_real|pure-real|pure_real|pure-omomo|pure_omomo|real)
        ;;
      mix-naive|pure-sd|pure-ds|shoo7sr1-near03-debug|shoo7sr1_near03_debug|shoo7sr1-debug|shoo7sr1_debug|shoo7sr1-linvel|shoo7sr1_linvel|shoo7sr1-action-history|shoo7sr1_action_history|shoo7sr1-linvel-action-history|shoo7sr1_linvel_action_history|shoo7sr1-both|shoo7sr1_both)
        echo "[ERROR] Positional mode ${1@Q} is incompatible with the AS real-mesh wrapper; launch distill_box_perception.sh directly for that mode." >&2
        exit 2
        ;;
      *)
        break
        ;;
    esac
    AS_MODE_PREFIX_CONSUMED=$((AS_MODE_PREFIX_CONSUMED + 1))
    shift
  done
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# Accept harmless AS/dataset aliases for muscle memory; this wrapper owns the
# actual data selection and always launches pure-real AS/OMOMO distillation.
# Parse both sides of an optional checkpoint so the advertised
# `<checkpoint> <schedule/contact mode>` ordering is resolved by this wrapper
# before it freezes actor inputs, schedule notes, and provenance.
consume_as_mode_prefix "$@"
shift "${AS_MODE_PREFIX_CONSUMED}"

DEFAULT_AS_TEACHER_CHECKPOINT=${DEFAULT_AS_TEACHER_CHECKPOINT:-"wandb://zihanw22/carry-any/bcleb5oi/model_67000.pt"}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_AS_TEACHER_CHECKPOINT}}}
if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  TEACHER_CHECKPOINT="$1"
  shift
fi
consume_as_mode_prefix "$@"
shift "${AS_MODE_PREFIX_CONSUMED}"

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing teacher checkpoint from train_as_general.sh." >&2
  usage >&2
  exit 1
fi

# Reject policy-profile errors before any checkpoint download.
case "$(echo "${RESUME_FROM_BOX:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) _early_resume_from_box=1 ;;
  0|false|no|off|"") _early_resume_from_box=0 ;;
  *)
    echo "[ERROR] RESUME_FROM_BOX must be a boolean. Got: ${RESUME_FROM_BOX}" >&2
    exit 2
    ;;
esac
case "${AS_POLICY_INIT_PROFILE:-}" in
  ""|drop_button_mlp_perception) ;;
  *)
    echo "[ERROR] Unknown AS_POLICY_INIT_PROFILE=${AS_POLICY_INIT_PROFILE}." >&2
    exit 2
    ;;
esac
if [[ "${_early_resume_from_box}" == "1" && "${AS_POLICY_INIT_PROFILE:-}" != "drop_button_mlp_perception" ]]; then
  echo "[ERROR] Generic distill_as_perception.sh cannot safely use RESUME_FROM_BOX=1: its default actor inputs differ from the box-button checkpoint." >&2
  echo "[ERROR] Use distill_as_button.sh, which validates and marks the compatible drop-button MLP architecture." >&2
  exit 2
fi

case "$(echo "${AS_CONTACT_AWARE_HISTORY}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_CONTACT_AWARE_HISTORY=1
    ;;
  0|false|no|off|"")
    AS_CONTACT_AWARE_HISTORY=0
    ;;
  *)
    echo "[ERROR] AS_CONTACT_AWARE_HISTORY must be a boolean. Got: ${AS_CONTACT_AWARE_HISTORY}" >&2
    exit 2
    ;;
esac
export CONTACT_AWARE_HISTORY="${AS_CONTACT_AWARE_HISTORY}"

PYTHON_BIN=${PYTHON_BIN:-python}

# These two launcher-facing aliases are converted into process-wide semantic
# environment controls by delegated wrappers later in the launch chain.  Bind
# them now, before phase-one provenance is generated, so provenance and every
# worker consume the same canonical values.  A pre-set HOLOSOMA value is audit
# input, not a second source of truth: accept equivalent spellings, but reject
# a conflicting value instead of allowing a downstream wrapper to overwrite
# it after provenance capture.
normalize_as_semantic_bool() {
  local name="$1"
  local value="$2"
  case "$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      printf '%s\n' True
      ;;
    0|false|no|off|"")
      printf '%s\n' False
      ;;
    *)
      echo "[ERROR] ${name} must be a boolean. Got: ${value}" >&2
      return 2
      ;;
  esac
}

PERCEPTION_INTO_POLICY_MODULES=$(normalize_as_semantic_bool \
  PERCEPTION_INTO_POLICY_MODULES "${PERCEPTION_INTO_POLICY_MODULES:-True}")
RESET_TO_DEFAULT_POSE=$(normalize_as_semantic_bool \
  RESET_TO_DEFAULT_POSE "${RESET_TO_DEFAULT_POSE:-False}")

if [[ -v HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES ]]; then
  _semantic_perception_modules=$(normalize_as_semantic_bool \
    HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES \
    "${HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES}")
  if [[ "${_semantic_perception_modules}" != "${PERCEPTION_INTO_POLICY_MODULES}" ]]; then
    echo "[ERROR] HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES conflicts with PERCEPTION_INTO_POLICY_MODULES: semantic=${_semantic_perception_modules} alias=${PERCEPTION_INTO_POLICY_MODULES}." >&2
    exit 2
  fi
fi
if [[ -v HOLOSOMA_RESET_TO_DEFAULT_POSE ]]; then
  _semantic_reset_to_default=$(normalize_as_semantic_bool \
    HOLOSOMA_RESET_TO_DEFAULT_POSE "${HOLOSOMA_RESET_TO_DEFAULT_POSE}")
  if [[ "${_semantic_reset_to_default}" != "${RESET_TO_DEFAULT_POSE}" ]]; then
    echo "[ERROR] HOLOSOMA_RESET_TO_DEFAULT_POSE conflicts with RESET_TO_DEFAULT_POSE: semantic=${_semantic_reset_to_default} alias=${RESET_TO_DEFAULT_POSE}." >&2
    exit 2
  fi
fi
export PERCEPTION_INTO_POLICY_MODULES RESET_TO_DEFAULT_POSE
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}"
export HOLOSOMA_RESET_TO_DEFAULT_POSE="${RESET_TO_DEFAULT_POSE}"
unset _semantic_perception_modules _semantic_reset_to_default

AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH=5
if [[ "${AS_CONTACT_AWARE_HISTORY}" == "1" ]]; then
  CONTACT_AWARE_HISTORY_LENGTH=${CONTACT_AWARE_HISTORY_LENGTH:-${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}}
  if [[ "${CONTACT_AWARE_HISTORY_LENGTH}" != "${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}" ]]; then
    echo "[ERROR] contact-aware-history is a fixed deployment contract with CONTACT_AWARE_HISTORY_LENGTH=${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}; got ${CONTACT_AWARE_HISTORY_LENGTH}." >&2
    exit 2
  fi
  if [[ -n "${AS_POLICY_INIT_PROFILE:-}" ]]; then
    echo "[ERROR] contact-aware-history is supported only by the generic AS student contract; AS_POLICY_INIT_PROFILE must be unset." >&2
    exit 2
  fi
  _as_history_actor_inputs="['actor_obs_root_contact_aware','actor_obs_proprio_with_actions_no_linvel']"
  "${PYTHON_BIN}" - "${STUDENT_ACTOR_INPUTS:-${_as_history_actor_inputs}}" <<'PY'
from __future__ import annotations

import ast
import sys

expected = [
    "actor_obs_root_contact_aware",
    "actor_obs_proprio_with_actions_no_linvel",
]
try:
    actual = list(ast.literal_eval(sys.argv[1]))
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid STUDENT_ACTOR_INPUTS={sys.argv[1]!r}: {exc}") from exc
if actual != expected:
    raise SystemExit(
        "[ERROR] contact-aware-history requires exact actor inputs "
        f"{expected!r}; got {actual!r}."
    )
PY
  STUDENT_ACTOR_INPUTS="${_as_history_actor_inputs}"
  for _history_var in STUDENT_PROPRIO_HISTORY_LENGTH CRITIC_PROPRIO_HISTORY_LENGTH; do
    _history_value="${!_history_var:-}"
    if [[ -n "${_history_value}" && "${_history_value}" != "${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}" ]]; then
      echo "[ERROR] contact-aware-history requires ${_history_var}=${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}; got ${_history_value}." >&2
      exit 2
    fi
    printf -v "${_history_var}" '%s' "${AS_CONTACT_AWARE_DEPLOYMENT_HISTORY_LENGTH}"
    export "${_history_var}"
  done
  export CONTACT_AWARE_HISTORY_LENGTH STUDENT_ACTOR_INPUTS
  unset _as_history_actor_inputs _history_var _history_value
fi

WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
AS_TEACHER_CACHE_ROOT=${AS_TEACHER_CACHE_ROOT:-"${HOME}/.cache/holosoma/teacher"}
TEACHER_CHECKPOINT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
  --ref "${TEACHER_CHECKPOINT}" \
  --cache-root "${AS_TEACHER_CACHE_ROOT}")
export TEACHER_CHECKPOINT

AS_TRAINING_RESUME_REF=${RESUME_CKPT:-${RESUME_CHECKPOINT:-}}
if [[ -n "${AS_TRAINING_RESUME_REF}" ]]; then
  AS_TRAINING_RESUME_CACHE_ROOT=${AS_TRAINING_RESUME_CACHE_ROOT:-"${HOME}/.cache/holosoma/training_resume"}
  AS_TRAINING_RESUME_REF=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
    --ref "${AS_TRAINING_RESUME_REF}" \
    --cache-root "${AS_TRAINING_RESUME_CACHE_ROOT}")
  export RESUME_CKPT="${AS_TRAINING_RESUME_REF}"
  unset RESUME_CHECKPOINT
  if [[ "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_EXPLICIT}" -eq 0 ]]; then
    CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=$("${PYTHON_BIN}" - "${AS_TRAINING_RESUME_REF}" <<'PY'
from __future__ import annotations

import os
import sys

from holosoma.utils.checkpoint_validation import load_verified_torch_checkpoint

checkpoint, _ = load_verified_torch_checkpoint(
    sys.argv[1],
    expected_sha256=os.environ.get("RESUME_SOURCE_EXPECTED_SHA256") or None,
    map_location="cpu",
)
config = checkpoint.get("experiment_config", {})
try:
    motion_config = config["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
except (KeyError, TypeError):
    motion_config = {}
value = motion_config.get("contact_interval_runtime_prepend_compensation", False)
if not isinstance(value, bool):
    raise SystemExit(
        "[ERROR] Invalid saved contact_interval_runtime_prepend_compensation value: " + repr(value)
    )
print("True" if value else "False")
PY
    )
    echo "[INFO] Inferred contact_interval_runtime_prepend_compensation=${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION} from training-resume checkpoint (missing legacy field means False)."
  fi
elif [[ "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION_EXPLICIT}" -eq 0 ]]; then
  CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True
fi

case "$(echo "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True
    ;;
  0|false|no|off)
    CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=False
    ;;
  *)
    echo "[ERROR] CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION must be a boolean. Got: ${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}" >&2
    exit 2
    ;;
esac
export CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION

STUDENT_MOTION_END_MODE=$(echo "${STUDENT_MOTION_END_MODE:-episodic}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
case "${STUDENT_MOTION_END_MODE}" in
  episodic)
    STUDENT_TERMINATION_PROFILE=g1_29dof_wbt_generalist
    ;;
  continuing)
    STUDENT_TERMINATION_PROFILE=g1_29dof_wbt_distill
    echo "[WARN] STUDENT_MOTION_END_MODE=continuing explicitly disables motion_ends; this is incompatible with the episodic teacher action-history contract and permits GAE/action state across clip rollover." >&2
    ;;
  *)
    echo "[ERROR] STUDENT_MOTION_END_MODE must be episodic or continuing. Got: ${STUDENT_MOTION_END_MODE}" >&2
    exit 2
    ;;
esac
export STUDENT_MOTION_END_MODE
# Training semantics must not depend on inference/debug variables inherited
# from the caller's shell.  The selected termination profile is the sole
# authority for episodic versus continuing motion execution.
export HOLOSOMA_DISABLE_AUTO_RESET=0
export HOLOSOMA_DISABLE_CLIP_END_RESET=0
export HOLOSOMA_DISABLE_MOTION_END_RESET=0
export HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE=1
export HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE=1
echo "[INFO] training_reset_contract disable_auto_reset=${HOLOSOMA_DISABLE_AUTO_RESET} disable_clip_end_reset=${HOLOSOMA_DISABLE_CLIP_END_RESET} disable_motion_end_reset=${HOLOSOMA_DISABLE_MOTION_END_RESET}"
echo "[INFO] contact_coverage_contract intervals=${HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE} targets=${HOLOSOMA_REQUIRE_CONTACT_TARGET_COVERAGE}"
case "$(echo "${AS_SUCCESS133_FINAL0P5}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_SUCCESS133_FINAL0P5=1
    ;;
  0|false|no|off|"")
    AS_SUCCESS133_FINAL0P5=0
    ;;
  *)
    echo "[ERROR] AS_SUCCESS133_FINAL0P5 must be a boolean. Got: ${AS_SUCCESS133_FINAL0P5}" >&2
    exit 2
    ;;
esac
RESUME_FROM_BOX=${RESUME_FROM_BOX:-0}
case "$(echo "${RESUME_FROM_BOX}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    RESUME_FROM_BOX=1
    ;;
  0|false|no|off|"")
    RESUME_FROM_BOX=0
    ;;
  *)
    echo "[ERROR] RESUME_FROM_BOX must be a boolean. Got: ${RESUME_FROM_BOX}" >&2
    exit 2
    ;;
esac
if [[ "${RESUME_FROM_BOX}" == "1" && "${AS_POLICY_INIT_PROFILE:-}" != "drop_button_mlp_perception" ]]; then
  echo "[ERROR] Generic distill_as_perception.sh cannot safely use RESUME_FROM_BOX=1: its default actor inputs differ from the box-button checkpoint." >&2
  echo "[ERROR] Use distill_as_button.sh, which validates and marks the compatible drop-button MLP architecture." >&2
  exit 2
fi
case "${AS_POLICY_INIT_PROFILE:-}" in
  ""|drop_button_mlp_perception)
    ;;
  *)
    echo "[ERROR] Unknown AS_POLICY_INIT_PROFILE=${AS_POLICY_INIT_PROFILE}." >&2
    exit 2
    ;;
esac
if [[ "${AS_POLICY_INIT_PROFILE:-}" == "drop_button_mlp_perception" ]]; then
  if [[ "$(echo "${STUDENT_POLICY_TYPE:-mlp}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')" != "mlp" ]]; then
    echo "[ERROR] AS_POLICY_INIT_PROFILE=drop_button_mlp_perception requires STUDENT_POLICY_TYPE=mlp." >&2
    exit 2
  fi
  if [[ -z "${STUDENT_ACTOR_INPUTS:-}" ]]; then
    echo "[ERROR] AS_POLICY_INIT_PROFILE=drop_button_mlp_perception requires explicit validated STUDENT_ACTOR_INPUTS from distill_as_button.sh." >&2
    exit 2
  fi
  "${PYTHON_BIN}" - "${STUDENT_ACTOR_INPUTS}" <<'PY'
from __future__ import annotations

import ast
import sys

expected = [
    "actor_obs_root_contact_aware",
    "actor_obs_drop_button",
    "actor_obs_proprio_with_actions_no_linvel",
]
try:
    actual = list(ast.literal_eval(sys.argv[1]))
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid STUDENT_ACTOR_INPUTS={sys.argv[1]!r}: {exc}") from exc
if actual != expected:
    raise SystemExit(
        "[ERROR] AS_POLICY_INIT_PROFILE=drop_button_mlp_perception requires exact actor inputs "
        f"{expected!r}; got {actual!r}."
    )
PY
fi
if [[ -z "${AS_CONTACT_AWARE}" ]]; then
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" || "${RESUME_FROM_BOX}" == "1" ]]; then
    AS_CONTACT_AWARE=1
  else
    AS_CONTACT_AWARE=0
  fi
fi
case "$(echo "${AS_CONTACT_AWARE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_CONTACT_AWARE=1
    ;;
  0|false|no|off|"")
    AS_CONTACT_AWARE=0
    ;;
  *)
    echo "[ERROR] AS_CONTACT_AWARE must be a boolean. Got: ${AS_CONTACT_AWARE}" >&2
    exit 2
    ;;
esac
if [[ "${AS_CONTACT_AWARE_HISTORY}" == "1" && "${AS_CONTACT_AWARE}" != "1" ]]; then
  echo "[ERROR] AS_CONTACT_AWARE_HISTORY=1 requires AS_CONTACT_AWARE=1." >&2
  exit 2
fi
export CONTACT_AWARE_HISTORY="${AS_CONTACT_AWARE_HISTORY}"

DEFAULT_RESUME_FROM_BOX_AS_BANK=${DEFAULT_RESUME_FROM_BOX_AS_BANK:-carryany_filter_scale_noscale_keep169_20260513}
DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR="${SCRIPT_DIR}/data/ds_as_data/${DEFAULT_RESUME_FROM_BOX_AS_BANK}"
DEFAULT_RESUME_FROM_BOX_LOCAL_CONTACT_ROOT="${DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR}/contact_export_from_retarget"
DEFAULT_RESUME_FROM_BOX_CONTACT_ROOT="${DEFAULT_RESUME_FROM_BOX_LOCAL_CONTACT_ROOT}"
DEFAULT_RESUME_FROM_BOX_DATA_DIR="${DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR}"
AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5}
AS_SUCCESS133_DATA_DIR="${SCRIPT_DIR}/data/ds_as_data/${AS_SUCCESS133_BANK_NAME}"
AS_SUCCESS133_CONTACT_EXPORT_ROOT="${AS_SUCCESS133_DATA_DIR}/contact_export_from_teacher_success133_final0p5"
DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/d9m3z369-recovered"}
DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${WANDB_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}}
DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-"${DEFAULT_BOX_RESUME_RUN}/files/${BOX_RESUME_MODEL_FILE}"}
BOX_RESUME_CKPT=${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}
if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  RESUME_FROM_BOX_AS_DATA_DIR=${RESUME_FROM_BOX_AS_DATA_DIR:-${AS_RESUME_DATA_DIR:-"${AS_SUCCESS133_DATA_DIR}"}}
  RESUME_FROM_BOX_AS_OBJECT_MAP=${RESUME_FROM_BOX_AS_OBJECT_MAP:-${AS_RESUME_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_DATA_DIR}/_clip_object_urdf_map.json"}}
  RESUME_FROM_BOX_CONTACT_EXPORT_ROOT=${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT:-${AS_CONTACT_EXPORT_ROOT:-"${AS_SUCCESS133_CONTACT_EXPORT_ROOT}"}}
  RESUME_FROM_BOX_EXPECTED_TOTAL=${RESUME_FROM_BOX_EXPECTED_TOTAL:-133}
else
  RESUME_FROM_BOX_AS_DATA_DIR=${RESUME_FROM_BOX_AS_DATA_DIR:-${AS_RESUME_DATA_DIR:-"${DEFAULT_RESUME_FROM_BOX_DATA_DIR}"}}
  RESUME_FROM_BOX_AS_OBJECT_MAP=${RESUME_FROM_BOX_AS_OBJECT_MAP:-${AS_RESUME_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_DATA_DIR}/_clip_object_urdf_map.json"}}
  RESUME_FROM_BOX_CONTACT_EXPORT_ROOT=${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT:-${AS_CONTACT_EXPORT_ROOT:-"${DEFAULT_RESUME_FROM_BOX_CONTACT_ROOT}"}}
  RESUME_FROM_BOX_EXPECTED_TOTAL=${RESUME_FROM_BOX_EXPECTED_TOTAL:-169}
fi

if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${AS_SUCCESS133_DATA_DIR}"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${AS_SUCCESS133_DATA_DIR}/_clip_object_urdf_map.json"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-133}
elif [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${RESUME_FROM_BOX_AS_DATA_DIR}"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_OBJECT_MAP}"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-"${RESUME_FROM_BOX_EXPECTED_TOTAL}"}
else
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-45}
fi

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  BOX_RESUME_CKPT="$(normalize_wandb_checkpoint_ref "${BOX_RESUME_CKPT}" "${BOX_RESUME_MODEL_FILE:-}")"
  case "${BOX_RESUME_CKPT}" in
    wandb://*|*.pt)
      ;;
    *)
      echo "[ERROR] BOX_RESUME_CKPT must resolve to a .pt checkpoint. Got: ${BOX_RESUME_CKPT}" >&2
      exit 2
      ;;
  esac
  BOX_RESUME_CKPT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
    --ref "${BOX_RESUME_CKPT}" \
    --cache-root "${BOX_POLICY_INIT_CACHE_ROOT:-${HOME}/.cache/holosoma/policy_init}")
  export BOX_RESUME_CKPT
  if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "[ERROR] RESUME_FROM_BOX initializes policy parameters only; do not also set RESUME_CKPT/RESUME_CHECKPOINT." >&2
    echo "[ERROR] Use BOX_RESUME_CKPT to choose the box policy initializer." >&2
    exit 2
  fi
fi

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
OMOMO_DATA_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")

case "${OMOMO_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_DATA_DIR must be local, not NFS: ${OMOMO_DATA_DIR}" >&2
    if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
      echo "[ERROR] Run ./cp_as.sh first; it copies keep169 and contact_export_from_retarget under ${SCRIPT_DIR}/data/ds_as_data." >&2
    else
      echo "[ERROR] Run ./cp_real.sh first and distill from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
    fi
    exit 2
    ;;
esac
case "${OMOMO_DATA_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_DATA_DIR}" >&2
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_OBJECT_MAP must be local, not NFS: ${OMOMO_OBJECT_MAP}" >&2
    if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
      echo "[ERROR] Run ./cp_as.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    else
      echo "[ERROR] Run ./cp_real.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    fi
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_OBJECT_MAP}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${OMOMO_DATA_DIR}" ]]; then
  echo "[ERROR] OMOMO_DATA_DIR does not exist: ${OMOMO_DATA_DIR}" >&2
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    echo "[ERROR] Run ./cp_as.sh first; it copies the keep169 bank and contact_export_from_retarget under data/ds_as_data/." >&2
  else
    echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  fi
  exit 2
fi

if ! compgen -G "${OMOMO_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in OMOMO_DATA_DIR: ${OMOMO_DATA_DIR}" >&2
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    echo "[ERROR] Run ./cp_as.sh first; it copies the keep169 bank and contact_export_from_retarget under data/ds_as_data/." >&2
  else
    echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  fi
  exit 2
fi

if [[ ! -f "${OMOMO_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${OMOMO_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}}
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
    OBJECT_SPAWN_MODE=single_slot_multi_urdf
    ;;
  *)
    echo "[ERROR] distill_as_perception.sh only supports OBJECT_SPAWN_MODE=single_slot_multi_urdf." >&2
    echo "[ERROR] Legacy urdf bank and primitive/box modes are disabled for AS to prevent object-slot explosion." >&2
    echo "[ERROR] Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] distill_as_perception.sh only supports mesh object geometry." >&2
    echo "[ERROR] Primitive/box/disabled geometry is not allowed for AS real-mesh training." >&2
    echo "[ERROR] Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - "${OMOMO_DATA_DIR}" "${OMOMO_OBJECT_MAP}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
map_path = Path(sys.argv[2]).expanduser().resolve()
expected_raw = sys.argv[3].strip()
expected = int(expected_raw) if expected_raw else None

npz_files = sorted(motion_dir.glob("*.npz"))
if expected is not None and len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} .npz clips under {motion_dir}, found {len(npz_files)}")
if not npz_files:
    raise SystemExit(f"[ERROR] No .npz clips found under {motion_dir}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {map_path}")
if expected is not None and len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} object-map entries in {map_path}, found {len(clips)}")

missing_entries = [p.stem for p in npz_files if p.stem not in clips]
if missing_entries:
    preview = ", ".join(missing_entries[:10])
    raise SystemExit(f"[ERROR] Missing object-map entries for {len(missing_entries)} clip(s): {preview}")


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


bad = []
unique_urdfs = {}
for clip_id, entry in clips.items():
    if not isinstance(entry, dict):
        bad.append(f"{clip_id}: map entry is not a dict")
        continue
    urdf_path = resolve_path(entry.get("object_urdf_path", ""), map_path.parent)
    mesh_path_raw = str(entry.get("object_mesh_path", "")).strip()
    mesh_path = resolve_path(mesh_path_raw, map_path.parent) if mesh_path_raw else None
    if not urdf_path.is_file():
        bad.append(f"{clip_id}: missing URDF {urdf_path}")
        continue
    if mesh_path is not None and not mesh_path.is_file():
        bad.append(f"{clip_id}: missing mesh {mesh_path}")
    unique_urdfs[str(urdf_path)] = clip_id

for urdf_raw, clip_id in sorted(unique_urdfs.items()):
    urdf_path = Path(urdf_raw)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        bad.append(f"{clip_id}: failed to parse URDF {urdf_path}: {exc}")
        continue
    mesh_tags = root.findall(".//mesh")
    if not mesh_tags:
        bad.append(f"{clip_id}: URDF has no <mesh> geometry: {urdf_path}")
        continue
    for tag in mesh_tags:
        filename = str(tag.get("filename", "")).strip()
        if not filename:
            bad.append(f"{clip_id}: URDF mesh tag has empty filename: {urdf_path}")
            continue
        mesh_path = resolve_path(filename, urdf_path.parent)
        if not mesh_path.is_file():
            bad.append(f"{clip_id}: URDF mesh file missing: {mesh_path}")

if bad:
    raise SystemExit("[ERROR] Real-mesh OMOMO validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh OMOMO bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

# AS_SINGLE_SLOT_MOTION_DIR remains a backward-compatible alias for the base;
# the effective payload directory is always the returned by-source/<digest>.
AS_SINGLE_SLOT_MOTION_BASE=${AS_SINGLE_SLOT_MOTION_BASE:-${AS_SINGLE_SLOT_MOTION_DIR:-"${OMOMO_DATA_DIR}/_single_slot_motion_bank"}}
AS_SINGLE_SLOT_MOTION_BASE_ABS=$(realpath -m "${AS_SINGLE_SLOT_MOTION_BASE}")
case "${AS_SINGLE_SLOT_MOTION_BASE_ABS}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] Generated AS single-slot motion-bank base must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_SINGLE_SLOT_MOTION_BASE_ABS}" >&2
    exit 2
    ;;
esac

AS_SINGLE_SLOT_SOURCE_MOTION_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
AS_SINGLE_SLOT_SOURCE_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")
AS_SINGLE_SLOT_MOTION_DIR_ABS=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_immutable_single_slot_bank.py" \
  --source-motion-dir "${AS_SINGLE_SLOT_SOURCE_MOTION_DIR}" \
  --source-object-map "${AS_SINGLE_SLOT_SOURCE_OBJECT_MAP}" \
  --output-base "${AS_SINGLE_SLOT_MOTION_BASE_ABS}")
AS_SINGLE_SLOT_MOTION_DIR_ABS=$(realpath -m "${AS_SINGLE_SLOT_MOTION_DIR_ABS}")
case "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" in
  "${AS_SINGLE_SLOT_MOTION_BASE_ABS}/by-source/"*)
    ;;
  *)
    echo "[ERROR] Immutable AS single-slot bank escaped its content-addressed base." >&2
    echo "[ERROR] base=${AS_SINGLE_SLOT_MOTION_BASE_ABS} output=${AS_SINGLE_SLOT_MOTION_DIR_ABS}" >&2
    exit 2
    ;;
esac

# A direct invocation has no sealed external-AS contract and retains the
# historical behavior.  batch_ne.sh supplies all five values together; if any
# appears, require the complete tuple and bind the view returned by the actual
# wrapper materialization to the controller's all-node barrier.
AS_EXTERNAL_MATERIALIZATION_CONTRACT=0
if [[ -n "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST:-}" \
      || -n "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST:-}" \
      || -n "${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST:-}" \
      || -n "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR:-}" \
      || -n "${HOLOSOMA_EXTERNAL_AS_WORLD_SIZE:-}" ]]; then
  AS_EXTERNAL_MATERIALIZATION_CONTRACT=1
  if ! [[ "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST:-}" =~ ^[0-9a-f]{64}$ \
        && "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST:-}" =~ ^[0-9a-f]{64}$ \
        && "${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST:-}" =~ ^[0-9a-f]{64}$ \
        && "${HOLOSOMA_EXTERNAL_AS_WORLD_SIZE:-}" =~ ^[1-9][0-9]*$ \
        && -n "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR:-}" ]]; then
    echo "[ERROR] Sealed external-AS materialization contract is incomplete or malformed." >&2
    exit 2
  fi
  if ! AS_EXPECTED_SINGLE_SLOT_DIR_ABS=$(realpath -e -- "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR}"); then
    echo "[ERROR] Sealed external-AS single-slot directory is missing: ${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_DIR}" >&2
    exit 2
  fi
  if [[ "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" != "${AS_EXPECTED_SINGLE_SLOT_DIR_ABS}" ]]; then
    echo "[ERROR] Effective AS single-slot directory changed after the all-node barrier: actual=${AS_SINGLE_SLOT_MOTION_DIR_ABS} expected=${AS_EXPECTED_SINGLE_SLOT_DIR_ABS}" >&2
    exit 2
  fi
  AS_SINGLE_SLOT_IDENTITY=$("${PYTHON_BIN}" - "${AS_SINGLE_SLOT_MOTION_DIR_ABS}/manifest.json" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit(f"[ERROR] Immutable AS single-slot manifest is missing or symlinked: {path}")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"[ERROR] Invalid immutable AS single-slot manifest {path}: {exc}") from exc
source = payload.get("source_digest") if isinstance(payload, dict) else None
view = payload.get("view_digest") if isinstance(payload, dict) else None
if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{64}", source) is None:
    raise SystemExit("[ERROR] Immutable AS single-slot manifest has a malformed source digest")
if not isinstance(view, str) or re.fullmatch(r"[0-9a-f]{64}", view) is None:
    raise SystemExit("[ERROR] Immutable AS single-slot manifest has a malformed view digest")
print(f"{source}\t{view}")
PY
  )
  IFS=$'\t' read -r AS_SINGLE_SLOT_SOURCE_DIGEST AS_SINGLE_SLOT_VIEW_DIGEST <<< "${AS_SINGLE_SLOT_IDENTITY}"
  if [[ "${AS_SINGLE_SLOT_SOURCE_DIGEST}" != "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST}" ]]; then
    echo "[ERROR] Effective AS single-slot source changed after the all-node barrier: actual=${AS_SINGLE_SLOT_SOURCE_DIGEST} expected=${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_SOURCE_DIGEST}" >&2
    exit 2
  fi
  if [[ "${AS_SINGLE_SLOT_VIEW_DIGEST}" != "${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST}" ]]; then
    echo "[ERROR] Effective AS single-slot view changed after the all-node barrier: actual=${AS_SINGLE_SLOT_VIEW_DIGEST} expected=${HOLOSOMA_EXTERNAL_AS_SINGLE_SLOT_VIEW_DIGEST}" >&2
    exit 2
  fi
fi

AS_SINGLE_SLOT_OBJECT_MAP="${AS_SINGLE_SLOT_MOTION_DIR_ABS}/_clip_object_urdf_map.json"
OMOMO_OBJECT_MAP=$(realpath -m "${AS_SINGLE_SLOT_OBJECT_MAP}")
case "${OMOMO_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] Generated AS single-slot object map must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_OBJECT_MAP}" >&2
    exit 2
    ;;
esac
OMOMO_DATA_DIR="${AS_SINGLE_SLOT_MOTION_DIR_ABS}"

AS_RANK_LOCAL_SHARDS=${AS_RANK_LOCAL_SHARDS:-1}
case "$(echo "${AS_RANK_LOCAL_SHARDS}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_RANK_LOCAL_SHARDS=1
    ;;
  0|false|no|off)
    AS_RANK_LOCAL_SHARDS=0
    ;;
  *)
    echo "[ERROR] AS_RANK_LOCAL_SHARDS must be a boolean. Got: ${AS_RANK_LOCAL_SHARDS}" >&2
    exit 2
    ;;
esac
if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" \
      && "${AS_RANK_LOCAL_SHARDS}" != "1" ]]; then
  echo "[ERROR] Sealed external-AS materialization requires AS_RANK_LOCAL_SHARDS=1." >&2
  exit 2
fi
if [[ "${AS_RANK_LOCAL_SHARDS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="$(default_cuda_visible_devices_all "${CUDA_VISIBLE_DEVICES:-}")"
  if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
  fi
  if ! [[ "${NPROC}" =~ ^[0-9]+$ ]] || (( NPROC < 1 )); then
    echo "[ERROR] NPROC must be a positive integer. Got: ${NPROC}" >&2
    exit 1
  fi
  AS_VISIBLE_DEVICE_COUNT=$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")
  if (( NPROC > AS_VISIBLE_DEVICE_COUNT )); then
    echo "[ERROR] NPROC=${NPROC} exceeds CUDA_VISIBLE_DEVICES count=${AS_VISIBLE_DEVICE_COUNT}: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  AS_NNODES="${NNODES:-1}"
  if ! [[ "${AS_NNODES}" =~ ^[0-9]+$ ]] || (( AS_NNODES < 1 )); then
    echo "[ERROR] NNODES must be a positive integer when AS_RANK_LOCAL_SHARDS is enabled. Got: ${AS_NNODES}" >&2
    exit 1
  fi
  AS_GLOBAL_WORLD_SIZE=$((NPROC * AS_NNODES))
  export CUDA_VISIBLE_DEVICES
  export NPROC

  if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" \
        && "${AS_GLOBAL_WORLD_SIZE}" != "${HOLOSOMA_EXTERNAL_AS_WORLD_SIZE}" ]]; then
    echo "[ERROR] Effective AS world size changed after the all-node barrier: actual=${AS_GLOBAL_WORLD_SIZE} expected=${HOLOSOMA_EXTERNAL_AS_WORLD_SIZE}" >&2
    exit 2
  fi

  if (( AS_GLOBAL_WORLD_SIZE > 1 )); then
    AS_RANK_SHARD_SOURCE_DIGEST=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_as_rank_shards.py" \
      --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" \
      --object-map "${OMOMO_OBJECT_MAP}" \
      --world-size "${AS_GLOBAL_WORLD_SIZE}" \
      --source-digest-only)
    if ! [[ "${AS_RANK_SHARD_SOURCE_DIGEST}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "[ERROR] Invalid AS rank-shard source digest: ${AS_RANK_SHARD_SOURCE_DIGEST}" >&2
      exit 2
    fi
    if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" \
          && "${AS_RANK_SHARD_SOURCE_DIGEST}" != "${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST}" ]]; then
      echo "[ERROR] Effective AS rank-shard source changed after the all-node barrier: actual=${AS_RANK_SHARD_SOURCE_DIGEST} expected=${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST}" >&2
      exit 2
    fi
    # The default is immutable/content-addressed. A later launch with changed
    # motion, map, URDF, mesh, or world-size inputs publishes to a new root and
    # cannot redirect active workers onto a different shard generation.
    AS_RANK_SHARD_ROOT=${AS_RANK_SHARD_ROOT:-"${AS_SINGLE_SLOT_MOTION_DIR_ABS}/_rank_shards/by-source/${AS_RANK_SHARD_SOURCE_DIGEST}/ws${AS_GLOBAL_WORLD_SIZE}"}
    AS_RANK_SHARD_ROOT_ABS=$(realpath -m "${AS_RANK_SHARD_ROOT}")
    case "${AS_RANK_SHARD_ROOT_ABS}" in
      "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
        ;;
      *)
        echo "[ERROR] AS rank-local shard root must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
        echo "[ERROR] Got: ${AS_RANK_SHARD_ROOT_ABS}" >&2
        exit 2
        ;;
    esac
    if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" ]]; then
      AS_EXPECTED_RANK_SHARD_ROOT="${AS_SINGLE_SLOT_MOTION_DIR_ABS}/_rank_shards/by-source/${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST}/ws${AS_GLOBAL_WORLD_SIZE}"
      if [[ "${AS_RANK_SHARD_ROOT_ABS}" != "${AS_EXPECTED_RANK_SHARD_ROOT}" ]]; then
        echo "[ERROR] Effective AS rank-shard root differs from the sealed content-addressed root: actual=${AS_RANK_SHARD_ROOT_ABS} expected=${AS_EXPECTED_RANK_SHARD_ROOT}" >&2
        exit 2
      fi
    fi
    HOLOSOMA_RANK_LOCAL_MOTION_ROOT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_as_rank_shards.py" \
      --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" \
      --object-map "${OMOMO_OBJECT_MAP}" \
      --output-root "${AS_RANK_SHARD_ROOT_ABS}" \
      --world-size "${AS_GLOBAL_WORLD_SIZE}" \
      --expected-source-digest "${AS_RANK_SHARD_SOURCE_DIGEST}")
    if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" ]]; then
      HOLOSOMA_RANK_LOCAL_MOTION_ROOT=$(realpath -e -- "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT}")
      if [[ "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT}" != "${AS_EXPECTED_RANK_SHARD_ROOT}" ]]; then
        echo "[ERROR] AS rank-shard publisher returned a root outside the sealed contract: actual=${HOLOSOMA_RANK_LOCAL_MOTION_ROOT} expected=${AS_EXPECTED_RANK_SHARD_ROOT}" >&2
        exit 2
      fi
    fi
    export HOLOSOMA_RANK_LOCAL_MOTION_ROOT
    export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=1
    export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0
  else
    if [[ "${AS_EXTERNAL_MATERIALIZATION_CONTRACT}" == "1" ]]; then
      AS_RANK_SHARD_SOURCE_DIGEST=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_as_rank_shards.py" \
        --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" \
        --object-map "${OMOMO_OBJECT_MAP}" \
        --world-size "${AS_GLOBAL_WORLD_SIZE}" \
        --source-digest-only)
      if [[ "${AS_RANK_SHARD_SOURCE_DIGEST}" != "${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST}" ]]; then
        echo "[ERROR] Effective AS rank-shard source changed after the all-node barrier: actual=${AS_RANK_SHARD_SOURCE_DIGEST} expected=${HOLOSOMA_EXTERNAL_AS_RANK_SHARD_SOURCE_DIGEST}" >&2
        exit 2
      fi
    fi
    export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=0
  fi
else
  export HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED=0
fi

CONTACT_EXPORT_ROOT=""
CONTACT_EXPORT_CLIPS_ROOT=""
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  CONTACT_EXPORT_ROOT=$(realpath -m "${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT}")
  CONTACT_EXPORT_CLIPS_ROOT=$(realpath -m "${CONTACT_EXPORT_ROOT}/clips")
  if [[ ! -d "${CONTACT_EXPORT_CLIPS_ROOT}" ]]; then
    CONTACT_EXPORT_CLIPS_ROOT="${CONTACT_EXPORT_ROOT}"
  fi
  if [[ ! -d "${CONTACT_EXPORT_CLIPS_ROOT}" ]]; then
    echo "[ERROR] Contact export root does not exist: ${CONTACT_EXPORT_ROOT}" >&2
    echo "[ERROR] Run ./cp_as.sh first; it copies contact_export_from_retarget into the repo-local keep169 bank." >&2
    exit 2
  fi
  case "${CONTACT_EXPORT_CLIPS_ROOT}" in
    /nfs|/nfs/*)
      echo "[ERROR] Contact export root must be local, not NFS: ${CONTACT_EXPORT_CLIPS_ROOT}" >&2
      echo "[ERROR] Run ./cp_as.sh first; it copies contact_export_from_retarget into the repo-local keep169 bank." >&2
      exit 2
      ;;
    *)
      CONTACT_EXPORT_CLIPS_ROOT=$("${PYTHON_BIN}" - "${OMOMO_DATA_DIR}" "${CONTACT_EXPORT_ROOT}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
contact_root = Path(sys.argv[2]).expanduser().resolve()
expected_raw = sys.argv[3].strip()
expected = int(expected_raw) if expected_raw else None
clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root

if not clips_root.is_dir():
    raise SystemExit(f"[ERROR] Contact export root does not exist: {contact_root}")

motion_ids = {path.stem for path in motion_dir.glob("*.npz")}
if expected is not None and len(motion_ids) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} AS keep clips under {motion_dir}, found {len(motion_ids)}")
if not motion_ids:
    raise SystemExit(f"[ERROR] No .npz clips found under AS keep motion dir: {motion_dir}")

def infer_clip_id(dir_name: str) -> str:
    normalized = dir_name.strip()
    prefix, separator, suffix = normalized.partition("_")
    return suffix.strip() if separator and prefix.isdecimal() and suffix.strip() else normalized

contact_ids: set[str] = set()
missing_files: list[str] = []
required_files = (
    "teacher_rollout_reference.npz",
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)
for clip_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
    clip_id = clip_dir.name if clip_dir.name in motion_ids else infer_clip_id(clip_dir.name)
    if clip_id in contact_ids:
        raise SystemExit(f"[ERROR] Duplicate contact directories resolve to active clip {clip_id!r}")
    contact_ids.add(clip_id)
    for file_name in required_files:
        if not (clip_dir / file_name).is_file():
            missing_files.append(f"{clip_id}:{file_name}")

missing_contacts = sorted(motion_ids.difference(contact_ids))
if missing_contacts:
    preview = ", ".join(missing_contacts[:20])
    raise SystemExit(f"[ERROR] Contact export missing {len(missing_contacts)} active clip(s): {preview}")
if missing_files:
    preview = ", ".join(missing_files[:20])
    raise SystemExit(f"[ERROR] Contact export has incomplete rollout/contact sidecars: {preview}")

print(str(clips_root))
PY
      )
      ;;
  esac
  OFFLINE_WRIST_REGION_NAMES=${OFFLINE_WRIST_REGION_NAMES:-'["left_wrist","right_wrist"]'}
  OFFLINE_CONTACT_REGION_NAMES=${OFFLINE_CONTACT_REGION_NAMES:-'["left_wrist","right_wrist","left_elbow","right_elbow","left_wrist_roll","right_wrist_roll","left_wrist_pitch","right_wrist_pitch","torso"]'}
  export OFFLINE_WRIST_REGION_NAMES
  export OFFLINE_CONTACT_REGION_NAMES
  AS_ROLLOUT_TRACKED_BODY_NAMES=${AS_ROLLOUT_TRACKED_BODY_NAMES:-'["pelvis","left_hip_roll_link","left_knee_link","left_ankle_roll_link","right_hip_roll_link","right_knee_link","right_ankle_roll_link","torso_link","left_shoulder_roll_link","left_elbow_link","left_wrist_yaw_link","right_shoulder_roll_link","right_elbow_link","right_wrist_yaw_link"]'}
  AS_ROLLOUT_REF_BODY_NAME=${AS_ROLLOUT_REF_BODY_NAME:-torso_link}
  CONTACT_VALIDATOR_EXPECTED_ARGS=()
  if [[ -n "${OMOMO_EXPECTED_TOTAL}" ]]; then
    CONTACT_VALIDATOR_EXPECTED_ARGS=(--expected-total "${OMOMO_EXPECTED_TOTAL}")
  fi
  CONTACT_RUNTIME_PREPEND_ARGS=()
  if [[ "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}" == "True" ]]; then
    CONTACT_RUNTIME_PREPEND_ARGS=(
      --runtime-prepend-compensation
      --runtime-prepend-duration-s "${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}"
    )
  fi
  CONTACT_EXPORT_CLIPS_ROOT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/validate_contact_sidecars.py" \
    --motion-dir "${OMOMO_DATA_DIR}" \
    --contact-root "${CONTACT_EXPORT_ROOT}" \
    --motion-end-mode "${STUDENT_MOTION_END_MODE}" \
    "${CONTACT_VALIDATOR_EXPECTED_ARGS[@]}" \
    "${CONTACT_RUNTIME_PREPEND_ARGS[@]}" \
    --tracked-body-names "${AS_ROLLOUT_TRACKED_BODY_NAMES}" \
    --ref-body-name "${AS_ROLLOUT_REF_BODY_NAME}" \
    --offline-contact-region-names "${OFFLINE_CONTACT_REGION_NAMES}" \
    --offline-wrist-region-names "${OFFLINE_WRIST_REGION_NAMES}")
  echo "[INFO] contact_sidecar_contract_verified clips_root=${CONTACT_EXPORT_CLIPS_ROOT} tracked_bodies=${AS_ROLLOUT_TRACKED_BODY_NAMES} ref_body=${AS_ROLLOUT_REF_BODY_NAME}"
  export CONTACT_EXPORT_ROOT
  export ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT="${CONTACT_EXPORT_CLIPS_ROOT}"
fi

# Finalize ambient object/perception controls before provenance is computed.
# The generator preserves their raw spelling (including unset versus explicit
# false), while runtime-asset finalization later records the normalized loader
# closure.  Computing first and exporting later would make those two scientific
# records describe different environments.
export OBJECT_SPAWN_MODE
export OBJECT_GEOMETRY_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS="${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS:-0}"
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}"
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1

PROVENANCE_SHARD_ARGS=()
if [[ -n "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT:-}" && -f "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT}/manifest.json" ]]; then
  PROVENANCE_SHARD_ARGS=(--motion-shard-manifest "${HOLOSOMA_RANK_LOCAL_MOTION_ROOT}/manifest.json")
fi
PROVENANCE_CONTACT_ARGS=()
if [[ -n "${CONTACT_EXPORT_ROOT}" ]]; then
  PROVENANCE_CONTACT_ARGS=(--contact-root "${CONTACT_EXPORT_ROOT}")
fi
PROVENANCE_POLICY_INIT_ARGS=()
EFFECTIVE_POLICY_INIT_CKPT="${POLICY_INIT_CKPT:-}"
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  EFFECTIVE_POLICY_INIT_CKPT="${BOX_RESUME_CKPT}"
fi
if [[ -n "${EFFECTIVE_POLICY_INIT_CKPT}" ]]; then
  PROVENANCE_POLICY_INIT_ARGS=(--policy-init-checkpoint "${EFFECTIVE_POLICY_INIT_CKPT}")
fi
PROVENANCE_TRAINING_RESUME_ARGS=()
if [[ -n "${AS_TRAINING_RESUME_REF:-}" ]]; then
  PROVENANCE_TRAINING_RESUME_ARGS=(--training-resume-checkpoint "${AS_TRAINING_RESUME_REF}")
fi
_contact_compensation_lower=$(echo "${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}" | tr '[:upper:]' '[:lower:]')
HOLOSOMA_TRAINING_PROVENANCE=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/compute_training_provenance.py" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --motion-dir "${OMOMO_DATA_DIR}" \
  --object-map "${OMOMO_OBJECT_MAP}" \
  "${PROVENANCE_CONTACT_ARGS[@]}" \
  "${PROVENANCE_SHARD_ARGS[@]}" \
  "${PROVENANCE_POLICY_INIT_ARGS[@]}" \
  "${PROVENANCE_TRAINING_RESUME_ARGS[@]}" \
  --student-motion-end-mode "${STUDENT_MOTION_END_MODE}" \
  --contact-interval-runtime-prepend-compensation "${_contact_compensation_lower}" \
  --source-root "${SCRIPT_DIR}")
export HOLOSOMA_TRAINING_PROVENANCE
echo "[INFO] training_provenance=${HOLOSOMA_TRAINING_PROVENANCE}"
echo "[INFO] student_motion_end_mode=${STUDENT_MOTION_END_MODE} termination_profile=${STUDENT_TERMINATION_PROFILE}"
echo "[INFO] contact_interval_runtime_prepend_compensation=${CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION}"

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export POLICY_INIT_CKPT="${BOX_RESUME_CKPT}"
  unset RESUME_CKPT
  unset RESUME_CHECKPOINT
  unset WANDB_RUN_ID
  unset RESUME_WANDB_ID
  unset WANDB_RESUME
  export WANDB_RESUME_SAME_RUN=0
fi

export WANDB_PROJECT
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export DATA_MODE="${DATA_MODE:-mix-naive}"
else
  export DATA_MODE=pure-real
fi
export DS_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
export MOTION_DIR="${OMOMO_DATA_DIR}"
export OBJECT_SPEC_PATH="${OMOMO_OBJECT_MAP}"
export OBJECT_URDF="${OMOMO_OBJECT_MAP}"
export AUTO_PREP_DS_BANK=0
export STRICT_DEFAULT_DS_BANK_VALIDATION=0
export USE_LEGACY_DS=0

export VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"

export DEFAULT_TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}"
export TEACHER_CHECKPOINT
export TEACHER_COMPAT_PROFILE="${TEACHER_COMPAT_PROFILE:-none}"
export TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS:-actor_obs}"
case "${TEACHER_CHECKPOINT}" in
  *"zihanw22/carry-any/runs/bcleb5oi"*|*"zihanw22/carry-any/bcleb5oi"*)
    export TEACHER_ACTOR_OBS_HISTORY_LENGTH="${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-1}"
    ;;
esac
export TEACHER_PERCEPTION_PRESET="${TEACHER_PERCEPTION_PRESET:-none}"
export TEACHER_PERCEPTION_OBS_KEY="${TEACHER_PERCEPTION_OBS_KEY:-}"
export TRACKER_PROFILE="${TRACKER_PROFILE:-as-general}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"

if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  export EXP="${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception_init_box}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception_init_box}"
  elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception_contact}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception_contact}"
  else
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_perception_init_box}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_perception_init_box}"
  fi
else
  export EXP="${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception}"
  else
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_perception}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_real_mesh_perception}"
  fi
fi
export TRAINING_PROJECT="${TRAINING_PROJECT:-${WANDB_PROJECT}}"
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
export CAMERA_APPLY_SENSOR_NOISE="${CAMERA_APPLY_SENSOR_NOISE:-True}"
case "${PERCEPTION_PRESET}" in
  camera_depth_d435i|camera_depth_d435i_17x17|camera_depth_d435i_defm_*)
    export CAMERA_WARP_EDGE_NOISE="${CAMERA_WARP_EDGE_NOISE:-True}"
    export CAMERA_WARP_ENABLE_HOLES="${CAMERA_WARP_ENABLE_HOLES:-True}"
    export CAMERA_WARP_HOLE_PROB="${CAMERA_WARP_HOLE_PROB:-0.2}"
    export CAMERA_WARP_ADDITIVE_NOISE_STD="${CAMERA_WARP_ADDITIVE_NOISE_STD:-0.03}"
    export CAMERA_WARP_DEPTH_OFFSET_STD="${CAMERA_WARP_DEPTH_OFFSET_STD:-0.03}"
    ;;
esac
AS_PUSH_INTERVAL_S=${AS_PUSH_INTERVAL_S:-"[0.5,2.0]"}
AS_MAX_PUSH_VEL=${AS_MAX_PUSH_VEL:-"[0.7,0.7,0.25,0.7,0.7,1.0]"}
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
  export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_proprio_with_actions_no_linvel']}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill initialized from an architecture-matched box-button actor. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient.}"
  elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill with contact-aware sparse root. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient.}"
  else
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 real-mesh perception distill initialized from an architecture-matched box-button actor. Contact T1 windows reweight the failure-adaptive timestep density according to the configured multiplicative/target-mass mode; start-at-zero is an explicit reset mixture and AS metrics report the effective distribution. PPO starts at the configured iteration-0 coefficient.}"
  fi
else
  export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root','actor_obs_proprio','actor_obs_actions']}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_step_mix}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill from train_as_general.sh teacher. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. PPO+DAgger hybrid by default; teacher consumes actor_obs without perception, student consumes sparse root command, proprio/action history, and depth perception.}"
  else
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_real_mesh_sparse_root_ppo_first_step_mix}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS/OMOMO real-mesh perception distill from train_as_general.sh teacher. PPO+DAgger uses the delegated ppo_first contract by default: PPO is 0.0 at iteration 0, increases by 0.1 every 700 iterations to 0.9 at iteration 6300, and the effective DAgger BC weight decreases from 1.0 to 0.1. Teacher consumes actor_obs without perception; student consumes sparse root command, proprio/action history, and depth perception.}"
  fi
fi

if [[ "${SCHEDULE_VARIANT}" != "ppo_first" ]]; then
  if [[ "${SCHEDULE_NAME_USER_SET}" -eq 0 ]]; then
    SCHEDULE_NAME="${SCHEDULE_NAME//ppo_first/${SCHEDULE_VARIANT}}"
    export SCHEDULE_NAME
  fi
  if [[ "${SCHEDULE_NOTES_USER_SET}" -eq 0 ]]; then
    case "${SCHEDULE_VARIANT}" in
      dagger_mix)
        SCHEDULE_NOTES="AS perception distillation using the explicit dagger_mix contract: PPO is disabled and teacher-action rollout mixing follows the configured annealing schedule."
        ;;
      dag_first)
        SCHEDULE_NOTES="AS perception distillation using the explicit dag_first contract: pure DAgger precedes the configured PPO blend."
        ;;
      default)
        SCHEDULE_NOTES="AS perception distillation using the explicit default schedule contract selected by the delegated launcher."
        ;;
    esac
    export SCHEDULE_NOTES
  fi
fi
if [[ "${AS_CONTACT_AWARE_HISTORY}" == "1" ]]; then
  export CONTACT_AWARE_HISTORY_LENGTH
  if [[ "${SCHEDULE_NAME_USER_SET}" -eq 0 ]]; then
    SCHEDULE_NAME="${SCHEDULE_NAME}_history${CONTACT_AWARE_HISTORY_LENGTH}"
    export SCHEDULE_NAME
  fi
  if [[ "${SCHEDULE_NOTES_USER_SET}" -eq 0 ]]; then
    SCHEDULE_NOTES="${SCHEDULE_NOTES} Contact-aware-history sets both student actor proprio history and critic proprio history to ${CONTACT_AWARE_HISTORY_LENGTH}."
    export SCHEDULE_NOTES
  fi
fi

echo "[INFO] Launching AS/OMOMO real-mesh perception distillation"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] resume_from_box=${RESUME_FROM_BOX}"
echo "[INFO] as_contact_aware=${AS_CONTACT_AWARE}"
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[INFO] student_policy_init_checkpoint=${POLICY_INIT_CKPT}"
fi
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  echo "[INFO] contact_export_root=${CONTACT_EXPORT_ROOT}"
  echo "[INFO] adaptive_sampling_contact_interval_root=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
fi
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} teacher_perception=${TEACHER_PERCEPTION_PRESET}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] EXP=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] perception_include_robot_mesh=${HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH:-<preset default>}"
echo "[INFO] camera_apply_sensor_noise=${CAMERA_APPLY_SENSOR_NOISE}"
echo "[INFO] camera_warp_freq_ratio=${CAMERA_WARP_FREQ_RATIO:-<preset default>}"
echo "[INFO] camera_warp_edge_noise=${CAMERA_WARP_EDGE_NOISE:-<preset default>} camera_warp_enable_holes=${CAMERA_WARP_ENABLE_HOLES:-<preset default>} camera_warp_hole_prob=${CAMERA_WARP_HOLE_PROB:-<preset default>}"
echo "[INFO] camera_warp_additive_noise_std=${CAMERA_WARP_ADDITIVE_NOISE_STD:-<preset default>} camera_warp_depth_offset_std=${CAMERA_WARP_DEPTH_OFFSET_STD:-<preset default>}"
echo "[INFO] as_push_interval_s=${AS_PUSH_INTERVAL_S} as_max_push_vel=${AS_MAX_PUSH_VEL}"
echo "[INFO] RUN_NAME=${RUN_NAME} TRAINING_PROJECT=${TRAINING_PROJECT}"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] schedule_variant=${SCHEDULE_VARIANT} schedule_name=${SCHEDULE_NAME}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS}"
echo "[INFO] HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}"
echo "[INFO] HOLOSOMA_RANK_LOCAL_MOTION_ROOT=${HOLOSOMA_RANK_LOCAL_MOTION_ROOT:-<disabled>}"
echo "[INFO] HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=${HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS}"
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-}" ]]; then
  echo "[INFO] teacher_actor_obs_history_length=${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
fi

AS_DEPLOYMENT_OBSERVATION_ARGS=()
if [[ "${AS_CONTACT_AWARE_HISTORY}" == "1" ]]; then
  # Pin the other actor group explicitly as well.  The delegated launcher
  # appends the proprio=5 override; the global duplicate-option preflight then
  # rejects any caller attempt to override either half of this 453-D contract.
  AS_DEPLOYMENT_OBSERVATION_ARGS+=(
    --observation.groups.actor_obs_root_contact_aware.history-length=1
  )
fi

exec bash "${SCRIPT_DIR}/distill_box_perception.sh" "$@" \
  "${AS_DEPLOYMENT_OBSERVATION_ARGS[@]}" \
  --randomization.setup_terms.push_randomizer_state.params.push_interval_s="${AS_PUSH_INTERVAL_S}" \
  --randomization.setup_terms.push_randomizer_state.params.max_push_vel="${AS_MAX_PUSH_VEL}" \
  "termination:${STUDENT_TERMINATION_PROFILE}"
