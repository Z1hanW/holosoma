#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_checkpoint_ref() {
    local ref="$1"
    [[ "${ref}" == wandb://* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

POSITIONAL=()
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DISTILL_TEACHER_CHECKPOINT:-}}"
# Optional student resume/init checkpoint. Defaults to empty: no student resume.
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${RESUME_CKPT:-${TRAINING_CHECKPOINT:-}}}"
for arg in "$@"; do
    case "${arg}" in
        corl_128)
            export CORL_128=1
            ;;
        success133|success133_final0p5)
            export AS_SUCCESS133_FINAL0P5=1
            ;;
        resume-from-box-button|resume_from_box_button)
            export RESUME_FROM_BOX=1
            ;;
        *)
            if [[ -z "${TEACHER_CHECKPOINT}" ]] && is_checkpoint_ref "${arg}"; then
                TEACHER_CHECKPOINT="${arg}"
            else
                POSITIONAL+=("${arg}")
            fi
            ;;
    esac
done

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DEFAULT_TEACHER_CHECKPOINT:-}}"

if [[ -z "${AS_SUCCESS133_FINAL0P5+x}" && -z "${RESUME_FROM_BOX+x}" ]]; then
    AS_SUCCESS133_FINAL0P5=1
fi

if [[ "${CORL_128:-0}" == "1" ]]; then
    export HOI_BANK="${HOI_BANK:-${AS_BANK:-corl_128}}"
    export HOI_EXPECTED_TOTAL="${HOI_EXPECTED_TOTAL:-${AS_EXPECTED_TOTAL:-126}}"
elif [[ "${AS_SUCCESS133_FINAL0P5:-0}" == "1" ]]; then
    DEFAULT_SUCCESS133_AS_BANK="carryany_filter_scale_noscale_keep169_20260513"
    DEFAULT_SUCCESS133_AS_BANK+="_plus_box_teacher_rollout_success133_final0p5"
    export HOI_BANK="${HOI_BANK:-${AS_BANK:-${DEFAULT_SUCCESS133_AS_BANK}}}"
    export HOI_EXPECTED_TOTAL="${HOI_EXPECTED_TOTAL:-${AS_EXPECTED_TOTAL:-133}}"
elif [[ "${RESUME_FROM_BOX:-0}" == "1" ]]; then
    export HOI_BANK="${HOI_BANK:-${AS_BANK:-carryany_filter_scale_noscale_keep169_20260513}}"
    export HOI_EXPECTED_TOTAL="${HOI_EXPECTED_TOTAL:-${AS_EXPECTED_TOTAL:-169}}"
fi
export AS_BANK="${AS_BANK:-${HOI_BANK:-}}"
export AS_EXPECTED_TOTAL="${AS_EXPECTED_TOTAL:-${HOI_EXPECTED_TOTAL:-}}"

export AS_CONTACT_AWARE="${AS_CONTACT_AWARE:-1}"
export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
export CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE="${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE:-tracking_error}"
export EXP="${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}"
export OBSERVATION_CONFIG="${OBSERVATION_CONFIG:-g1_29dof_wbt_w_object_distill_sparse_root_cmd}"
export REWARD_CONFIG="${REWARD_CONFIG:-g1_29dof_wbt_w_object_generalist}"
export RUN_NAME="${RUN_NAME:-hoi-button-public-ppo}"
export ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND:-True}"
export DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}"
export HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH="${HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH:-1}"
export HOLOSOMA_MOTION_METRICS_INTERVAL="${HOLOSOMA_MOTION_METRICS_INTERVAL:-16}"
export BAD_TRACKING_THRESHOLD_AUGMENT="${BAD_TRACKING_THRESHOLD_AUGMENT:-1.0}"
DEFAULT_STUDENT_ACTOR_INPUTS="['actor_obs_root_contact_aware','actor_obs_drop_button',"
DEFAULT_STUDENT_ACTOR_INPUTS+="'actor_obs_proprio_with_actions_no_linvel','perception_obs']"
export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-${DEFAULT_STUDENT_ACTOR_INPUTS}}"

if [[ -z "${TEACHER_CHECKPOINT}" && "${DISTILL_REQUIRE_TEACHER:-1}" == "1" ]]; then
    echo "[ERROR] Missing teacher checkpoint for HOI distillation." >&2
    echo "[ERROR] Pass it as the first argument or set TEACHER_CHECKPOINT=/path/to/model.pt." >&2
    exit 2
fi

if [[ "${RESUME_FROM_BOX:-0}" == "1" ]]; then
    export STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${BOX_RESUME_CKPT:-}}"
    if [[ -z "${STUDENT_CHECKPOINT}" ]]; then
        echo "[WARN] RESUME_FROM_BOX=1 requested, but BOX_RESUME_CKPT/STUDENT_CHECKPOINT is empty." >&2
    fi
fi

if [[ -n "${STUDENT_CHECKPOINT}" ]]; then
    export TRAINING_CHECKPOINT="${STUDENT_CHECKPOINT}"
else
    unset TRAINING_CHECKPOINT
fi

DISTILL_ARGS=(
    --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}"
)

if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
    TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS:-actor_obs}"
    TEACHER_ACTOR_OBS_HISTORY_LENGTH="${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-5}"
    DISTILL_ARGS+=(
        --algo.config.distill.enabled=True
        --algo.config.distill.mode="${DISTILL_MODE:-dagger}"
        --algo.config.distill.policy-to-clone="${TEACHER_CHECKPOINT}"
        --algo.config.distill.bc-loss-coef="${BC_LOSS_COEF:-1.0}"
        --algo.config.distill.clip-teacher-actions="${CLIP_TEACHER_ACTIONS:-True}"
        --algo.config.distill.clip-actions-threshold="${CLIP_ACTIONS_THRESHOLD:-8.0}"
        --algo.config.distill.teacher-obs-keys="${TEACHER_OBS_KEYS}"
        --algo.config.distill.strict-teacher-load="${STRICT_TEACHER_LOAD:-True}"
        --algo.config.distill.teacher-action-mix-ratio="${TEACHER_ACTION_MIX_RATIO:-0.0}"
        --algo.config.distill.ppo-start-epoch="${PPO_START_EPOCH:-0}"
        --algo.config.distill.dagger-end-epoch="${DAGGER_END_EPOCH:-4000}"
        --algo.config.distill.ppo-start-coeff="${PPO_START_COEFF:-0.1}"
        --algo.config.distill.ppo-target-coeff="${PPO_TARGET_COEFF:-0.9}"
        --algo.config.distill.ppo-schedule-step-epochs="${PPO_SCHEDULE_STEP_EPOCHS:-500}"
        --algo.config.distill.dagger-loss-coef="${DAGGER_LOSS_COEF:-1.0}"
        --algo.config.distill.distill-loss-type="${DISTILL_LOSS_TYPE:-mse}"
        --algo.config.distill.dagger-ignore-zero-teacher-actions="${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-True}"
        --algo.config.distill.dagger-match-std="${DAGGER_MATCH_STD:-False}"
        --observation.groups.actor_obs.history-length="${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
        --algo.config.actor-learning-rate="${ACTOR_LR:-7e-5}"
        --algo.config.critic-learning-rate="${CRITIC_LR:-7e-5}"
        --algo.config.init-noise-std="${INIT_NOISE_STD:-0.01}"
        --algo.config.entropy-coef="${ENTROPY_COEF:-0.005}"
        --algo.config.module-dict.actor.min-noise-std="${ACTOR_MIN_NOISE_STD:-0.01}"
    )
fi

exec bash "${SCRIPT_DIR}/train_hoi_general.sh" "${DISTILL_ARGS[@]}" "${POSITIONAL[@]}"
