#!/usr/bin/env bash
set -euo pipefail

# Debug pure-PPO launcher for the AS drop-button solid-object setup.
#
# This intentionally does not use distillation, teacher actions, DAgger, box
# policy initialization, or full checkpoint resume. It is meant to compare the
# current AS/button environment and reward setup against pure PPO from scratch.
#
# Defaults are conservative and single-node/single-GPU. Override NPROC,
# CUDA_VISIBLE_DEVICES, PER_GPU_ENVS, NNODES, NODE_RANK, MASTER_ADDR, and
# MASTER_PORT when you intentionally want a larger debug launch.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
DEFAULT_BANK_NAME=prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball
DEFAULT_BASE_BANK_NAME=prism_debug30_convexhull_allmesh

MOTION_DIR=${MOTION_DIR:-"data/ds_as_data/${DEFAULT_BANK_NAME}/_single_slot_motion_bank"}
OBJECT_URDF=${OBJECT_URDF:-"${MOTION_DIR}/_clip_object_urdf_map.json"}
REFERENCE_ROOT=${REFERENCE_ROOT:-"data/ds_as_data/${DEFAULT_BASE_BANK_NAME}/contact_export_from_teacher_success133_final0p5"}

MOTION_DIR=$(realpath -m "${MOTION_DIR}")
OBJECT_URDF=$(realpath -m "${OBJECT_URDF}")
REFERENCE_ROOT=$(realpath -m "${REFERENCE_ROOT}")

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR does not exist: ${MOTION_DIR}" >&2
  exit 2
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF map does not exist: ${OBJECT_URDF}" >&2
  exit 2
fi
if [[ ! -d "${REFERENCE_ROOT}" ]]; then
  echo "[ERROR] REFERENCE_ROOT does not exist: ${REFERENCE_ROOT}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${MOTION_DIR}" "${OBJECT_URDF}" <<'PY'
import json
import sys
from pathlib import Path

motion_dir = Path(sys.argv[1]).resolve()
map_path = Path(sys.argv[2]).resolve()
npz = sorted(motion_dir.glob("*.npz"))
if not npz:
    raise SystemExit(f"[ERROR] no .npz clips found under {motion_dir}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload.get("clips", payload) if isinstance(payload, dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] invalid empty object map: {map_path}")

missing = [path.stem for path in npz if path.stem not in clips]
if missing:
    raise SystemExit(f"[ERROR] object map missing {len(missing)} clip(s), first={missing[:8]}")

def resolve(raw: str, base: Path) -> Path:
    p = Path(str(raw)).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()

bad = []
for clip, entry in sorted(clips.items()):
    if not isinstance(entry, dict):
        bad.append(f"{clip}: map entry is not a dict")
        continue
    urdf = resolve(entry.get("object_urdf_path", ""), map_path.parent)
    if not urdf.is_file():
        bad.append(f"{clip}: missing URDF {urdf}")
    for key in ("object_mesh_path", "object_visual_mesh_path", "object_collision_mesh_path"):
        raw = str(entry.get(key, "")).strip()
        if raw and not resolve(raw, map_path.parent).is_file():
            bad.append(f"{clip}: missing {key} {resolve(raw, map_path.parent)}")

if bad:
    raise SystemExit("[ERROR] object-bank validation failed:\n  " + "\n  ".join(bad[:20]))

print(f"[INFO] validated pure-RL bank: clips={len(npz)} map_entries={len(clips)} motion_dir={motion_dir}")
PY

for env_name in RESUME_FROM_BOX RESUME_CKPT RESUME_CHECKPOINT POLICY_INIT_CKPT POLICY_INIT_CHECKPOINT BOX_RESUME_CKPT RESUME_FROM_BOX_CKPT; do
  if [[ -n "${!env_name:-}" ]]; then
    echo "[ERROR] ${env_name} is set, but debug_pure_rl.sh must start from scratch with no policy/checkpoint init." >&2
    echo "[ERROR] Unset ${env_name} or use a different launcher." >&2
    exit 2
  fi
done

export OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
export ACCEPT_EULA=${ACCEPT_EULA:-Y}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=${HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS:-1}
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS=${HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS:-1}
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}
export HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=${HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES:-True}
export HOLOSOMA_SKIP_INITIAL_CHECKPOINT=${HOLOSOMA_SKIP_INITIAL_CHECKPOINT:-1}
export HOLOSOMA_GLOO_SMALL_COLLECTIVES=${HOLOSOMA_GLOO_SMALL_COLLECTIVES:-1}
export HOLOSOMA_GLOO_GRAD_REDUCE=${HOLOSOMA_GLOO_GRAD_REDUCE:-1}
export HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE:-1}
export HOLOSOMA_SYNC_EACH_ITERATION=${HOLOSOMA_SYNC_EACH_ITERATION:-0}
export HOLOSOMA_GLOO_BARRIER=${HOLOSOMA_GLOO_BARRIER:-0}
export TORCH_DIST_TIMEOUT_SEC=${TORCH_DIST_TIMEOUT_SEC:-3600}

NPROC=${NPROC:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29662}
MAX_RESTARTS=${MAX_RESTARTS:-0}
PER_GPU_ENVS=${PER_GPU_ENVS:-1024}
NUM_ENVS=${NUM_ENVS:-$((NPROC * NNODES * PER_GPU_ENVS))}

TRAINING_PROJECT=${TRAINING_PROJECT:-carry-any}
WANDB_ENTITY=${WANDB_ENTITY:-zihanw22}
LOGGER=${LOGGER:-logger:wandb}
RUN_NAME=${RUN_NAME:-debug_pure_rl_as_button_solid_prism_debug30_bigmlp_${RUN_STAMP}}
TRAINING_NAME=${TRAINING_NAME:-debug_pure_rl_as_button_solid_prism_debug30_bigmlp_${RUN_STAMP}}

NUM_LEARNING_ITERATIONS=${NUM_LEARNING_ITERATIONS:-40000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
NUM_MINI_BATCHES=${NUM_MINI_BATCHES:-${NPROC}}
NUM_LEARNING_EPOCHS=${NUM_LEARNING_EPOCHS:-1}
ACTOR_LR=${ACTOR_LR:-1e-3}
CRITIC_LR=${CRITIC_LR:-1e-3}
INIT_NOISE_STD=${INIT_NOISE_STD:-0.01}
ENTROPY_COEF=${ENTROPY_COEF:-0.0}
ACTOR_MIN_NOISE_STD=${ACTOR_MIN_NOISE_STD:-0.01}
CLIP_PARAM=${CLIP_PARAM:-0.2}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
STUDENT_ACTOR_HIDDEN_DIMS=${STUDENT_ACTOR_HIDDEN_DIMS:-"[2048,1024,512,256,128]"}
STUDENT_ACTOR_INPUTS=${STUDENT_ACTOR_INPUTS:-"['actor_obs_root_contact_aware','actor_obs_drop_button','actor_obs_proprio_with_actions_no_linvel']"}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.2}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-1.0}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-8.0}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}
ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT:-"${REFERENCE_ROOT}/clips"}

BAD_REF_POS_THRESHOLD=${BAD_REF_POS_THRESHOLD:-0.5}
BAD_REF_ORI_THRESHOLD=${BAD_REF_ORI_THRESHOLD:-0.8}
BAD_MOTION_BODY_POS_THRESHOLD=${BAD_MOTION_BODY_POS_THRESHOLD:-0.25}
BAD_OBJECT_POS_THRESHOLD=${BAD_OBJECT_POS_THRESHOLD:-0.25}
BAD_OBJECT_ORI_THRESHOLD=${BAD_OBJECT_ORI_THRESHOLD:-0.8}
PUSH_INTERVAL_S=${PUSH_INTERVAL_S:-'[0.5,2.0]'}
PUSH_MAX_VEL=${PUSH_MAX_VEL:-'[0.7,0.7,0.25,0.7,0.7,1.0]'}

PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-301989888}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-301989888}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}

cmd=(
  torchrun
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --nproc_per_node="${NPROC}"
  --max_restarts="${MAX_RESTARTS}"
  --master_port="${MASTER_PORT}"
  src/holosoma/holosoma/train_agent.py
  exp:g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-rollout-ref
  randomization:g1_29dof_wbt_w_object_pure_rl
  --algo.config.distill.enabled=False
  --training.num-envs="${NUM_ENVS}"
  --training.project="${TRAINING_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.multigpu=$([[ "${NPROC}" -gt 1 || "${NNODES}" -gt 1 ]] && echo True || echo False)
  --algo.config.num-learning-iterations="${NUM_LEARNING_ITERATIONS}"
  --algo.config.actor-learning-rate="${ACTOR_LR}"
  --algo.config.critic-learning-rate="${CRITIC_LR}"
  --algo.config.clip-param="${CLIP_PARAM}"
  --algo.config.max-grad-norm="${MAX_GRAD_NORM}"
  --algo.config.init-noise-std="${INIT_NOISE_STD}"
  --algo.config.entropy-coef="${ENTROPY_COEF}"
  --algo.config.module-dict.actor.min-noise-std="${ACTOR_MIN_NOISE_STD}"
  --algo.config.normalize-actor-obs=False
  --algo.config.normalize-critic-obs=False
  --algo.config.save-interval="${SAVE_INTERVAL}"
  --simulator.config.sim.physx.gpu-found-lost-pairs-capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-total-aggregate-pairs-capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu-collision-stack-size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_DIR}"
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion=False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob="${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale="${RESET_NOISE_SCALE}"
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append=False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s=0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend=False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s=0.0
  --command.setup-terms.motion-command.params.motion-config.contact-interval-runtime-prepend-compensation=False
  --robot.object.enabled=True
  --robot.object.object-urdf-path "${OBJECT_URDF}"
  perception:camera_depth_d435i
  --algo.config.module-dict.actor.input-dim "${STUDENT_ACTOR_INPUTS}"
  --algo.config.module-dict.actor.layer-config.hidden-dims="${STUDENT_ACTOR_HIDDEN_DIMS}"
  --algo.config.module-dict.actor.type=MLP
  --algo.config.num-mini-batches="${NUM_MINI_BATCHES}"
  --algo.config.num-learning-epochs="${NUM_LEARNING_EPOCHS}"
  --training.export-onnx=False
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --termination.terms.bad-tracking.params.bad-ref-pos-threshold="${BAD_REF_POS_THRESHOLD}"
  --termination.terms.bad-tracking.params.bad-ref-ori-threshold="${BAD_REF_ORI_THRESHOLD}"
  --termination.terms.bad-tracking.params.bad-motion-body-pos-threshold="${BAD_MOTION_BODY_POS_THRESHOLD}"
  --termination.terms.bad-tracking.params.bad-object-pos-threshold="${BAD_OBJECT_POS_THRESHOLD}"
  --termination.terms.bad-tracking.params.bad-object-ori-threshold="${BAD_OBJECT_ORI_THRESHOLD}"
  --observation.groups.actor_obs.history-length=1
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  --command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root="${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
  --command.setup-terms.motion-command.params.motion-config.contact-aware-carry-window-mode=peak_height
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-alpha=0.91
  --command.setup-terms.motion-command.params.motion-config.contact-aware-peak-height-smoothing-steps=5
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled=True
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-half-width-steps=50
  --command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost=7.0
  --reward.terms.offline-contact-guidance.params.contact-export-root "${REFERENCE_ROOT}"
  --reward.terms.motion-global-ref-position-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.motion-global-ref-orientation-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.motion-relative-body-position-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.motion-relative-body-orientation-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.motion-global-body-lin-vel.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.motion-global-body-ang-vel.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.object-global-ref-position-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --reward.terms.object-global-ref-orientation-error-exp.params.rollout-reference-root "${REFERENCE_ROOT}"
  --perception.camera-pitch-deg=10
  --perception.camera-apply-sensor-noise=True
  --perception.camera-warp-edge-noise=True
  --perception.camera-warp-enable-holes=True
  --perception.camera-warp-hole-prob=0.2
  --perception.camera-warp-additive-noise-std=0.03
  --perception.camera-warp-depth-offset-std=0.03
  --perception.object-geometry-mode=mesh
  --randomization.setup_terms.push_randomizer_state.params.push_interval_s="${PUSH_INTERVAL_S}"
  --randomization.setup_terms.push_randomizer_state.params.max_push_vel="${PUSH_MAX_VEL}"
)

if is_truthy "${ENABLE_CLEAN_NOISY_CURRICULUM:-1}"; then
  cmd+=(
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.enabled=True
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-clip-name-prefixes='["sub"]'
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.stage-start-iterations='[0]'
    --command.setup-terms.motion-command.params.motion-config.clean-noisy-clip-curriculum.clean-group-probabilities='[1.0]'
  )
fi

cmd+=("$@")
cmd+=("${LOGGER}")
if [[ "${LOGGER}" != "logger:disabled" ]]; then
  cmd+=(
    --logger.name="${RUN_NAME}"
    --logger.video.enabled=False
    --logger.headless_recording=False
    --logger.video.upload_to_wandb=False
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    cmd+=(--logger.entity="${WANDB_ENTITY}")
  fi
fi

echo "[INFO] pure_rl=1 distill_enabled=False"
echo "[INFO] motion_transitions prepend=False/0.0s append=False/0.0s"
echo "[INFO] randomization=g1_29dof_wbt_w_object_pure_rl joint_bias_pd_gain_dr=enabled torque_rfi=disabled action_delay=disabled"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
echo "[INFO] reference_root=${REFERENCE_ROOT}"
echo "[INFO] nproc=${NPROC} nnodes=${NNODES} node_rank=${NODE_RANK} num_envs=${NUM_ENVS} per_gpu_envs=${PER_GPU_ENVS}"
echo "[INFO] actor_hidden_dims=${STUDENT_ACTOR_HIDDEN_DIMS}"
echo "[INFO] actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] learning_rate actor=${ACTOR_LR} critic=${CRITIC_LR}"
echo "[INFO] ppo clip_param=${CLIP_PARAM} max_grad_norm=${MAX_GRAD_NORM} init_noise_std=${INIT_NOISE_STD} min_noise_std=${ACTOR_MIN_NOISE_STD} entropy_coef=${ENTROPY_COEF}"
echo "[INFO] distributed gloo_small_collectives=${HOLOSOMA_GLOO_SMALL_COLLECTIVES} gloo_grad_reduce=${HOLOSOMA_GLOO_GRAD_REDUCE} sync_after_grad_allreduce=${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE} sync_each_iteration=${HOLOSOMA_SYNC_EACH_ITERATION} gloo_barrier=${HOLOSOMA_GLOO_BARRIER}"
echo "[INFO] run_name=${RUN_NAME}"

printf '[INFO] final_train_command:'
printf ' %q' \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE}" \
  HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}" \
  HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS="${HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS}" \
  HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES}" \
  HOLOSOMA_GLOO_SMALL_COLLECTIVES="${HOLOSOMA_GLOO_SMALL_COLLECTIVES}" \
  HOLOSOMA_GLOO_GRAD_REDUCE="${HOLOSOMA_GLOO_GRAD_REDUCE}" \
  HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE="${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE}" \
  HOLOSOMA_SYNC_EACH_ITERATION="${HOLOSOMA_SYNC_EACH_ITERATION}" \
  HOLOSOMA_GLOO_BARRIER="${HOLOSOMA_GLOO_BARRIER}" \
  TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" \
  "${cmd[@]}"
printf '\n'

if is_truthy "${DRY_RUN:-0}"; then
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
HOLOSOMA_OBJECT_SPAWN_MODE="${HOLOSOMA_OBJECT_SPAWN_MODE}" \
HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}" \
HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS="${HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS}" \
HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES}" \
HOLOSOMA_GLOO_SMALL_COLLECTIVES="${HOLOSOMA_GLOO_SMALL_COLLECTIVES}" \
HOLOSOMA_GLOO_GRAD_REDUCE="${HOLOSOMA_GLOO_GRAD_REDUCE}" \
HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE="${HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE}" \
HOLOSOMA_SYNC_EACH_ITERATION="${HOLOSOMA_SYNC_EACH_ITERATION}" \
HOLOSOMA_GLOO_BARRIER="${HOLOSOMA_GLOO_BARRIER}" \
TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" \
"${cmd[@]}"
