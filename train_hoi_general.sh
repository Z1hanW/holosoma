#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_AS_BANK="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout"
HOI_BANK="${HOI_BANK:-${AS_BANK:-${DEFAULT_AS_BANK}}}"
HOI_DATA_DIR="${HOI_DATA_DIR:-${AS_DATA_DIR:-${OMOMO_DATA_DIR:-data/ds_as_data/${HOI_BANK}}}}"
HOI_OBJECT_MAP="${HOI_OBJECT_MAP:-${AS_OBJECT_MAP:-${OMOMO_OBJECT_MAP:-${HOI_DATA_DIR}/_clip_object_urdf_map.json}}}"
HOI_EXPECTED_TOTAL="${HOI_EXPECTED_TOTAL:-${AS_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL:-195}}}"
AS_BANK="${AS_BANK:-${HOI_BANK}}"
AS_DATA_DIR="${AS_DATA_DIR:-${HOI_DATA_DIR}}"
AS_OBJECT_MAP="${AS_OBJECT_MAP:-${HOI_OBJECT_MAP}}"
AS_EXPECTED_TOTAL="${AS_EXPECTED_TOTAL:-${HOI_EXPECTED_TOTAL}}"
STRICT_HOI_EXPECTED_TOTAL="${STRICT_HOI_EXPECTED_TOTAL:-${STRICT_AS_EXPECTED_TOTAL:-1}}"
REQUIRE_HOI_OBJECT_MAP="${REQUIRE_HOI_OBJECT_MAP:-${REQUIRE_AS_OBJECT_MAP:-1}}"
STRICT_AS_EXPECTED_TOTAL="${STRICT_AS_EXPECTED_TOTAL:-${STRICT_HOI_EXPECTED_TOTAL}}"
REQUIRE_AS_OBJECT_MAP="${REQUIRE_AS_OBJECT_MAP:-${REQUIRE_HOI_OBJECT_MAP}}"

EXP="${EXP:-g1-29dof-wbt-w-object-generalist}"
REWARD_CONFIG="${REWARD_CONFIG:-g1_29dof_wbt_w_object_generalist_offline_contact_guidance}"
OBSERVATION_CONFIG="${OBSERVATION_CONFIG:-}"
COMMAND_CONFIG="${COMMAND_CONFIG:-}"
LOGGER_CONFIG="${LOGGER_CONFIG:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-carry-any}"
RUN_NAME="${RUN_NAME:-hoi-general-real-mesh-cotrack}"
PER_GPU_ENVS="${PER_GPU_ENVS:-8192}"
NUM_GPUS="${NUM_GPUS:-${NPROC_PER_NODE:-1}}"
TOTAL_ENVS="${TOTAL_ENVS:-$((PER_GPU_ENVS * NUM_GPUS))}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SIM_OBJECT_URDF="${SIM_OBJECT_URDF:-holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf}"

CONTACT_AWARE_CARRY_WINDOW_MODE="${CONTACT_AWARE_CARRY_WINDOW_MODE:-rel_z}"
CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE="${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE:-tracking_error}"
ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND:-False}"
DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S:-0.2}"

if [[ ! -d "${AS_DATA_DIR}" ]]; then
    echo "[ERROR] AS_DATA_DIR does not exist: ${AS_DATA_DIR}" >&2
    exit 2
fi

NPZ_COUNT="$(find "${AS_DATA_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if [[ "${AS_EXPECTED_TOTAL}" != "0" && "${STRICT_AS_EXPECTED_TOTAL}" == "1" ]]; then
    if [[ "${NPZ_COUNT}" != "${AS_EXPECTED_TOTAL}" ]]; then
        echo "[ERROR] Expected ${AS_EXPECTED_TOTAL} HOI .npz files, found ${NPZ_COUNT} in ${AS_DATA_DIR}" >&2
        exit 2
    fi
fi

MOTION_DIR="${AS_DATA_DIR}"
if [[ -f "${AS_OBJECT_MAP}" ]]; then
    SINGLE_SLOT_DIR="${AS_SINGLE_SLOT_DIR:-${AS_DATA_DIR}/_single_slot_motion_bank}"
    if [[ "${DRY_RUN:-0}" == "1" && "${PREPARE_AS_BANK_ON_DRY_RUN:-0}" != "1" ]]; then
        AS_SINGLE_SLOT_OBJECT_MAP="${SINGLE_SLOT_DIR}/_clip_object_urdf_map.json"
        echo "[INFO] DRY_RUN: would prepare single-slot HOI bank at ${SINGLE_SLOT_DIR}" >&2
    else
        mkdir -p "${SINGLE_SLOT_DIR}"
        while IFS= read -r -d '' npz_file; do
            ln -sfn "$(realpath "${npz_file}")" "${SINGLE_SLOT_DIR}/$(basename "${npz_file}")"
        done < <(find "${AS_DATA_DIR}" -maxdepth 1 -type f -name '*.npz' -print0)
        AS_SINGLE_SLOT_OBJECT_MAP="$(
            "${PYTHON_BIN}" scripts/prepare_single_slot_object_map.py \
                --motion-dir "${SINGLE_SLOT_DIR}" \
                --object-map "${AS_OBJECT_MAP}" \
                --output-map "${SINGLE_SLOT_DIR}/_clip_object_urdf_map.json"
        )"
    fi
    export OBJECT_SPEC_PATH="${AS_SINGLE_SLOT_OBJECT_MAP}"
    MOTION_DIR="${SINGLE_SLOT_DIR}"
elif [[ "${REQUIRE_AS_OBJECT_MAP}" == "1" ]]; then
    echo "[ERROR] HOI object map is required but missing: ${AS_OBJECT_MAP}" >&2
    exit 2
else
    echo "[WARN] HOI object map missing; running without clip object metadata: ${AS_OBJECT_MAP}" >&2
fi

export AS_DATA_DIR
export AS_OBJECT_MAP
export HOI_BANK
export HOI_DATA_DIR
export HOI_OBJECT_MAP
export HOI_EXPECTED_TOTAL
export DATA_MODE="${DATA_MODE:-mix-naive}"
export DS_DATA_ROOT="${DS_DATA_ROOT:-data/ds_as_data}"
export MOTION_DIR
export HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}"
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS="${HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS:-1}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS="${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS:-0}"
export HOLOSOMA_MOTION_METRICS_INTERVAL="${HOLOSOMA_MOTION_METRICS_INTERVAL:-16}"

MOTION_CFG_ARG_PREFIX="--command.setup_terms.motion_command.params.motion_config"
TRAIN_ARGS=(
    "exp:${EXP}"
    "logger:${LOGGER_CONFIG}"
    "reward:${REWARD_CONFIG}"
    "--training.project=${WANDB_PROJECT}"
    "--training.name=${RUN_NAME}"
    "--training.num-envs=${TOTAL_ENVS}"
    "${MOTION_CFG_ARG_PREFIX}.motion_dir=${MOTION_DIR}"
    "${MOTION_CFG_ARG_PREFIX}.use_adaptive_timesteps_sampler=False"
    "${MOTION_CFG_ARG_PREFIX}.contact_aware_carry_window_mode=${CONTACT_AWARE_CARRY_WINDOW_MODE}"
    "${MOTION_CFG_ARG_PREFIX}.contact_aware_sparse_root_command_mode=${CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE}"
    "${MOTION_CFG_ARG_PREFIX}.enable_default_pose_prepend=${ENABLE_DEFAULT_POSE_PREPEND}"
    "${MOTION_CFG_ARG_PREFIX}.default_pose_prepend_duration_s=${DEFAULT_POSE_PREPEND_DURATION_S}"
    "--robot.object.object_urdf_path=${SIM_OBJECT_URDF}"
)

if [[ -n "${OBSERVATION_CONFIG}" ]]; then
    TRAIN_ARGS+=("observation:${OBSERVATION_CONFIG}")
fi
if [[ -n "${COMMAND_CONFIG}" ]]; then
    TRAIN_ARGS+=("command:${COMMAND_CONFIG}")
fi
if [[ "${LOGGER_CONFIG}" != "disabled" ]]; then
    TRAIN_ARGS+=("--logger.project=${WANDB_PROJECT}" "--logger.name=${RUN_NAME}")
fi
if [[ -n "${TRAINING_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=("--training.checkpoint=${TRAINING_CHECKPOINT}")
fi

TRAIN_ARGS+=("$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${PYTHON_BIN}" src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    exec torchrun --nproc_per_node="${NUM_GPUS}" src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
fi

exec "${PYTHON_BIN}" src/holosoma/holosoma/train_agent.py "${TRAIN_ARGS[@]}"
