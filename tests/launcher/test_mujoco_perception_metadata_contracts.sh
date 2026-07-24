#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=""

TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT
MUJOCO_BINDING_SITE=/home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/lib/python3.11/site-packages
MUJOCO_TEST_PYTHONPATH="${REPO_ROOT}/src/holosoma:${REPO_ROOT}/src/holosoma_inference"
if [[ -d "$MUJOCO_BINDING_SITE" ]]; then
  MUJOCO_TEST_PYTHONPATH="${MUJOCO_TEST_PYTHONPATH}:${MUJOCO_BINDING_SITE}"
fi

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

assert_eq() {
  local expected="$1"
  local actual="$2"
  local message="$3"
  [[ "$actual" == "$expected" ]] || fail "${message}: expected=${expected} actual=${actual}"
}

bash -n \
  scripts/mujoco_perception_env.sh \
  scripts/source_mujoco_setup.sh \
  mj_env.sh \
  mj_env_mujoco_render_848.sh \
  mj_depth.sh \
  mj_track.sh \
  mj_box_depth_track.sh

source scripts/mujoco_perception_env.sh

NOISE_VARS=(
  PERCEPTION_CAMERA_WARP_EDGE_NOISE
  PERCEPTION_CAMERA_WARP_ENABLE_HOLES
  PERCEPTION_CAMERA_APPLY_SENSOR_NOISE
  HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE
)

# A launcher fallback is non-explicit and therefore yields to the checkpoint.
for name in "${NOISE_VARS[@]}"; do
  unset "$name" "${name}_EXPLICIT" || true
  holosoma_set_launcher_default "$name" False
  assert_eq False "${!name}" "launcher fallback value"
  marker="${name}_EXPLICIT"
  assert_eq 0 "${!marker}" "launcher fallback marker"
  holosoma_apply_checkpoint_value "$name" True
  assert_eq True "${!name}" "checkpoint must replace launcher fallback"
done

# A value supplied before the wrapper is an explicit user override and wins.
for name in "${NOISE_VARS[@]}"; do
  unset "$name" "${name}_EXPLICIT" || true
  printf -v "$name" '%s' False
  export "$name"
  holosoma_set_launcher_default "$name" True
  marker="${name}_EXPLICIT"
  assert_eq 1 "${!marker}" "user value marker"
  holosoma_apply_checkpoint_value "$name" True
  assert_eq False "${!name}" "explicit user value must beat checkpoint"
done

# An invalid marker must fail closed instead of silently changing precedence.
if (
  TEST_PERCEPTION_VALUE=False
  TEST_PERCEPTION_VALUE_EXPLICIT=not-a-bool
  holosoma_set_launcher_default TEST_PERCEPTION_VALUE True
) >"${TMP_DIR}/invalid_marker.out" 2>&1; then
  fail "invalid explicit marker unexpectedly succeeded"
fi
grep -F 'TEST_PERCEPTION_VALUE_EXPLICIT must be a boolean marker' \
  "${TMP_DIR}/invalid_marker.out" >/dev/null || fail "invalid marker error is not actionable"

# source_mujoco_setup.sh must tag its False values as fallbacks, while values
# present before sourcing remain explicit.
bash -c '
  set -euo pipefail
  for name in \
    PERCEPTION_CAMERA_WARP_EDGE_NOISE \
    PERCEPTION_CAMERA_WARP_ENABLE_HOLES \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE; do
    unset "$name" "${name}_EXPLICIT" || true
  done
  source scripts/source_mujoco_setup.sh box_74
  for name in \
    PERCEPTION_CAMERA_WARP_EDGE_NOISE \
    PERCEPTION_CAMERA_WARP_ENABLE_HOLES \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE; do
    marker="${name}_EXPLICIT"
    [[ "${!name}" == False && "${!marker}" == 0 ]]
  done
' || fail "source_mujoco_setup defaults were not marked non-explicit"

bash -c '
  set -euo pipefail
  export PERCEPTION_CAMERA_WARP_EDGE_NOISE=False
  unset PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT || true
  source scripts/source_mujoco_setup.sh box_74
  [[ "$PERCEPTION_CAMERA_WARP_EDGE_NOISE" == False ]]
  [[ "$PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT" == 1 ]]
' || fail "source_mujoco_setup did not preserve a user override"

# Every tracked wrapper that used to inject False must use the shared origin
# marker helper, and mj_track must apply checkpoint values through it.
python3 - <<'PY'
from pathlib import Path
import re

wrappers = {
    "mj_env.sh": 3,
    "mj_env_mujoco_render_848.sh": 3,
    "mj_depth.sh": 3,
    "scripts/source_mujoco_setup.sh": 4,
}
for path_str, expected_calls in wrappers.items():
    text = Path(path_str).read_text(encoding="utf-8")
    if "mujoco_perception_env.sh" not in text:
        raise SystemExit(f"[FAIL] {path_str} does not load the shared explicit-marker helper")
    calls = len(re.findall(r"^holosoma_set_launcher_default ", text, flags=re.MULTILINE))
    if calls < expected_calls:
        raise SystemExit(
            f"[FAIL] {path_str} marks only {calls} perception fallbacks; expected at least {expected_calls}"
        )

track = Path("mj_track.sh").read_text(encoding="utf-8")
offset_defaults_start = track.index('if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]')
offset_defaults_end = track.index("MODEL_EXPECTS_PERCEPTION_OBS=", offset_defaults_start)
offset_defaults_block = track[offset_defaults_start:offset_defaults_end]
dual_button_presets = (
    "g1-29dof-wbt-object-contact-aware-dual-button-depth-distill",
    "g1-29dof-wbt-contact-aware-dual-button-depth-distill",
    "g1-29dof-wbt-object-contact-aware-pickup-drop-button-depth-distill",
)
for dual_button_preset in dual_button_presets:
    dual_offset_pattern = (
        rf'elif \[\[[^\n]*"{re.escape(dual_button_preset)}"[^\n]*\]\]; then\s*\n'
        r'\s*export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1'
    )
    if re.search(dual_offset_pattern, offset_defaults_block) is None:
        raise SystemExit(
            "[FAIL] mj_track.sh does not apply the training-aligned motion offset to "
            f"the dual-button preset {dual_button_preset!r}"
        )
for name in (
    "PERCEPTION_CAMERA_WARP_EDGE_NOISE",
    "PERCEPTION_CAMERA_WARP_ENABLE_HOLES",
    "PERCEPTION_CAMERA_APPLY_SENSOR_NOISE",
    "HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE",
):
    pattern = rf"{name}\)\s*\n\s*holosoma_apply_checkpoint_value \"\$key\" \"\$value\""
    if re.search(pattern, track) is None:
        raise SystemExit(f"[FAIL] mj_track.sh does not resolve {name} with explicit-marker precedence")

required_normalize_fragments = (
    '"camera_warp_normalize": "PERCEPTION_CAMERA_WARP_NORMALIZE"',
    'PERCEPTION_CAMERA_WARP_NORMALIZE)',
    '--perception.camera-warp-normalize',
    'warp_normalize=${PERCEPTION_CAMERA_WARP_NORMALIZE:-<default>}',
)
for fragment in required_normalize_fragments:
    if fragment not in track:
        raise SystemExit(
            "[FAIL] mj_track.sh does not close the checkpoint-to-publisher depth-normalization contract: "
            + fragment
        )

required_direct_randomization_fragments = (
    "import base64",
    "canonical_camera_func = (",
    'contract_status = "attached-v2-targeted-v2-distribution-verified"',
    'print(f"PERCEPTION_CONTRACT_ENVELOPE_B64=',
    'PERCEPTION_CONTRACT_ENVELOPE_B64="$value"',
    '--perception-randomization.enabled "$PERCEPTION_RANDOMIZATION_ENABLED"',
    'append_run_sim_value --perception-randomization.translation-range "$PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE"',
    'append_run_sim_value --perception-randomization.rotation-range-deg "$PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG"',
    'append_run_sim_value --perception-randomization.noise-std-mult-range "$PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE"',
    'append_run_sim_value --perception-randomization.noise-drop-prob-range "$PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE"',
    '--perception-producer-tick-dt "$PERCEPTION_PRODUCER_TICK_DT"',
    '--perception-allow-mujoco-noise "$PERCEPTION_ALLOW_MUJOCO_NOISE"',
    '--perception-contract-envelope-b64 "$PERCEPTION_CONTRACT_ENVELOPE_B64"',
    '--training.seed "$PERCEPTION_PRODUCER_SEED"',
    'append_run_sim_value --perception.camera-warp-hole-reference-batch-size "$PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE"',
    'RUN_SIM_CMD=(',
    '"${MUJOCO_LAUNCH_PREFIX[@]}" "${RUN_SIM_CMD[@]}"',
)
for fragment in required_direct_randomization_fragments:
    if fragment not in track:
        raise SystemExit(
            "[FAIL] mj_track.sh does not close the direct camera reset contract: " + fragment
        )

required_transition_fragments = (
    "effective_motion_transition_settings_from_metadata",
    "transition_settings = effective_motion_transition_settings_from_metadata(metadata)",
    'transition_settings["prepend"]["applied"]',
    'transition_settings["append"]["applied"]',
    "if effective_transition_applied:",
    "if effective_prepend_applied:",
)
for fragment in required_transition_fragments:
    if fragment not in track:
        raise SystemExit(
            "[FAIL] mj_track.sh does not launch from the authenticated effective motion-transition contract: "
            + fragment
        )
if "if needs_default_pose_transition:" in track:
    raise SystemExit(
        "[FAIL] mj_track.sh still launches transitions from requested MotionConfig flags"
    )
launch_defaults_start = track.index("apply_training_motion_launch_defaults()")
launch_defaults_end = track.index(
    "apply_training_motion_launch_defaults ",
    launch_defaults_start + len("apply_training_motion_launch_defaults()"),
)
launch_defaults_block = track[launch_defaults_start:launch_defaults_end]
if "motion_transition_contract_from_metadata" in launch_defaults_block:
    raise SystemExit(
        "[FAIL] mj_track.sh bypasses the authenticated effective motion-transition helper"
    )

viewer = Path("src/holosoma/holosoma/viser_mujoco_sim_state.py").read_text(encoding="utf-8")
for fragment in (
    "motion_transition_contract_from_metadata",
    "required=raw_transition_requested",
    'transition_contract["prepend"]["applied"]',
):
    if fragment not in viewer:
        raise SystemExit(
            "[FAIL] MuJoCo viewer still infers default-pose initialization from requested flags: "
            + fragment
        )

if 'print(f"HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT=' in track:
    raise SystemExit(
        "[FAIL] mj_track.sh still maps reset-term enablement onto legacy one-shot camera jitter"
    )
if "$( [[" in track:
    raise SystemExit(
        "[FAIL] mj_track.sh still constructs run_sim argv through unquoted command substitution"
    )

box_track = Path("mj_box_depth_track.sh").read_text(encoding="utf-8")
selector = '"$INFER_PYTHON_BIN" "$ROOT_DIR/scripts/mj_infer_inference_config.py" "$MODEL_LOCAL"'
if selector not in box_track:
    raise SystemExit(
        "[FAIL] mj_box_depth_track.sh does not use the metadata-authenticated canonical selector"
    )
if "obs_dim == 96" in box_track or "actor_input_dim =" in box_track:
    raise SystemExit(
        "[FAIL] mj_box_depth_track.sh retains an unsafe inline shape/group preset selector"
    )
if 'INFERENCE_CONFIG:-g1-29dof-wbt-object-distill' in box_track:
    raise SystemExit(
        "[FAIL] mj_box_depth_track.sh silently falls back after canonical preset selection"
    )
PY

# Exercise the real ONNX metadata path when the local launcher runtimes are
# available.  This uses a tracked policy fixture and only writes under /tmp.
INFER_PY=""
for candidate in \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONSAFEPATH=1 "$candidate" -c 'import onnx' >/dev/null 2>&1; then
    INFER_PY="$candidate"
    break
  fi
done

MUJOCO_PY=""
for candidate in \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python \
  /home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python \
  "$(command -v python3 2>/dev/null || true)"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if PYTHONSAFEPATH=1 PYTHONPATH="$MUJOCO_TEST_PYTHONPATH" \
    "$candidate" - <<'PY' >/dev/null 2>&1
import holosoma
import mujoco
import torch
import typeguard
import tyro
from holosoma.config_types.run_sim import RunSimConfig

assert all(hasattr(mujoco, name) for name in ("MjModel", "MjData", "MjSpec", "mj_step"))
PY
  then
    MUJOCO_PY="$candidate"
    break
  fi
done

MODEL_FIXTURE="${REPO_ROOT}/mujoco-web-wobj-depth-demo/public/demo-assets/clips/box_74/policy.onnx"
MOTION_FIXTURE="${REPO_ROOT}/data_demo/box_74.npz"
OBJECT_FIXTURE="${REPO_ROOT}/data_demo/objects/box_74.urdf"
if [[ -n "$INFER_PY" && -n "$MUJOCO_PY" && -f "$MODEL_FIXTURE" && -f "$MOTION_FIXTURE" && -f "$OBJECT_FIXTURE" ]]; then
  MODEL_WITH_NOISE="${TMP_DIR}/policy_all_noise.onnx"
  MODEL_WITH_ATTACHED_CONTRACT="${TMP_DIR}/policy_attached_contract.onnx"
  MODEL_SINGLE_SENSOR_NOISE="${TMP_DIR}/policy_single_sensor_noise.onnx"
  MODEL_ADDITIVE_ONLY_NOISE="${TMP_DIR}/policy_additive_only_noise.onnx"
  MODEL_DUPLICATE_CAMERA_TERM="${TMP_DIR}/policy_duplicate_camera_term.onnx"
  MODEL_BAD_CAMERA_AXIS="${TMP_DIR}/policy_bad_camera_axis.onnx"
  MODEL_NONFINITE_CAMERA_RANGE="${TMP_DIR}/policy_nonfinite_camera_range.onnx"
  MODEL_REVERSED_CAMERA_RANGE="${TMP_DIR}/policy_reversed_camera_range.onnx"
  MODEL_NEGATIVE_NOISE_STD="${TMP_DIR}/policy_negative_noise_std.onnx"
  MODEL_BAD_NOISE_DROP="${TMP_DIR}/policy_bad_noise_drop.onnx"
  MODEL_BAD_TRAINING_FPS="${TMP_DIR}/policy_bad_training_fps.onnx"
  MODEL_LEGACY_HOLES="${TMP_DIR}/policy_legacy_holes.onnx"
  MODEL_MISSING_HOLE_REFERENCE="${TMP_DIR}/policy_missing_hole_reference.onnx"
  MODEL_MISSING_GEOMETRY_SUPPORT="${TMP_DIR}/policy_missing_geometry_support.onnx"
  MODEL_CONTRACT_V1="${TMP_DIR}/policy_contract_v1.onnx"
  MODEL_CONTRACT_TICK_MISMATCH="${TMP_DIR}/policy_contract_tick_mismatch.onnx"
  MODEL_LIFECYCLE_MISMATCH="${TMP_DIR}/policy_lifecycle_mismatch.onnx"
  MODEL_CONTRACT_MISMATCH="${TMP_DIR}/policy_contract_mismatch.onnx"
  "$INFER_PY" - \
    "$MODEL_FIXTURE" \
    "$MODEL_WITH_NOISE" \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    "$MODEL_SINGLE_SENSOR_NOISE" \
    "$MODEL_ADDITIVE_ONLY_NOISE" \
    "$MODEL_DUPLICATE_CAMERA_TERM" \
    "$MODEL_BAD_CAMERA_AXIS" \
    "$MODEL_NONFINITE_CAMERA_RANGE" \
    "$MODEL_REVERSED_CAMERA_RANGE" \
    "$MODEL_NEGATIVE_NOISE_STD" \
    "$MODEL_BAD_NOISE_DROP" \
    "$MODEL_BAD_TRAINING_FPS" \
    "$MODEL_LEGACY_HOLES" \
    "$MODEL_MISSING_HOLE_REFERENCE" \
    "$MODEL_MISSING_GEOMETRY_SUPPORT" \
    "$MODEL_CONTRACT_V1" \
    "$MODEL_CONTRACT_TICK_MISMATCH" \
    "$MODEL_LIFECYCLE_MISMATCH" \
    "$MODEL_CONTRACT_MISMATCH" <<'PY'
import copy
import hashlib
import json
import sys

import onnx

source = onnx.load(sys.argv[1])
outputs = iter(sys.argv[2:])


def camera_summary(config):
    params = config["randomization"]["reset_terms"]["randomize_camera_raycast"]["params"]
    translation = params.get("translation_range")
    rotation = params.get("rotation_range_deg")
    return {
        "enabled": True,
        "translation_xyz": (
            None if translation is None else [translation[axis] for axis in ("x", "y", "z")]
        ),
        "rotation_rpy_deg": (
            None if rotation is None else [rotation[axis] for axis in ("roll", "pitch", "yaw")]
        ),
        "noise_std_mult": params.get("noise_std_mult_range"),
        "noise_drop_prob": params.get("noise_drop_prob_range"),
    }


def save_variant(
    *,
    mutate=None,
    attached=False,
    contract_mismatch=False,
    missing_hole_reference=False,
    contract_version=2,
    contract_tick_dt=0.02,
    lifecycle_mismatch=False,
    missing_geometry_support=False,
):
    model = copy.deepcopy(source)
    config_prop = next(
        (prop for prop in model.metadata_props if prop.key == "experiment_config"),
        None,
    )
    if config_prop is None:
        raise SystemExit("fixture has no experiment_config metadata")
    config = json.loads(config_prop.value)
    perception = config["perception"]
    perception["camera_warp_edge_noise"] = True
    perception["camera_warp_enable_holes"] = bool(attached)
    perception["camera_apply_sensor_noise"] = True
    perception["camera_warp_normalize"] = False
    if attached:
        perception["reset_refresh_semantics"] = "targeted_v2"
    if mutate is not None:
        mutate(config)
    config_prop.value = json.dumps(config, allow_nan=True)
    kept = [
        prop
        for prop in model.metadata_props
        if prop.key
        not in {"perception_observation_contract", "perception_observation_contract_sha256"}
    ]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    if attached:
        summary = camera_summary(config)
        if contract_mismatch:
            summary = copy.deepcopy(summary)
            summary["noise_std_mult"] = [0.0, 0.049]
        hole_schema = None
        if perception["camera_warp_enable_holes"]:
            hole_schema = {
                "normalization_scope": "reference_batch",
                "reference_batch_size": 4096,
            }
            if missing_hole_reference:
                hole_schema.pop("reference_batch_size")
        contract = {
            "version": contract_version,
            "camera_source": "far_tracking_warp",
            "camera_reset_randomization": summary,
            "camera_setup_randomization": {"enabled": False},
            "hole_generator_schema": hole_schema,
            "producer_tick_dt": contract_tick_dt,
            "producer_lifecycle": {
                "reset_refresh_semantics": "targeted_v2",
                "ordinary_manager_update_calls_per_control_tick": 1,
                "initialization_control_ticks_before_first_reset_output": 1,
                "initialization_ordinary_manager_update_calls_before_first_reset_output": 1,
                "reset_output_republished_until_physics_advances": True,
                "reset_output_scope": "reset_env_subset",
                "hole_clock_advances_on_reset_refresh": False,
                "camera_frequency_phase_advances_on_reset_refresh": False,
                "camera_producer_reset_refresh_consumes_process_global_rng": True,
                "future_noise_sample_path_peer_reset_coupled": True,
                "batch_size_invariant_sample_path": False,
                "stochastic_equivalence": "distribution_only",
                "seed_replay_scope": "same_execution_trace_only",
            },
            "training_geometry_support": {
                "version": 1,
                "camera_source": "far_tracking_warp",
                "training_rank_count": 1,
                "robot_mesh_bindings": [],
                "object_mesh_support": [],
            },
        }
        if missing_geometry_support:
            contract.pop("training_geometry_support")
        if lifecycle_mismatch:
            contract["producer_lifecycle"]["camera_producer_reset_refresh_consumes_process_global_rng"] = False
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        contract_prop = model.metadata_props.add()
        contract_prop.key = "perception_observation_contract"
        contract_prop.value = json.dumps(contract, separators=(",", ":"))
        digest_prop = model.metadata_props.add()
        digest_prop.key = "perception_observation_contract_sha256"
        digest_prop.value = json.dumps(hashlib.sha256(payload).hexdigest())
    onnx.save(model, next(outputs))


def camera_params(config):
    return config["randomization"]["reset_terms"]["randomize_camera_raycast"]["params"]


save_variant()
save_variant(attached=True)
save_variant(
    mutate=lambda config: camera_params(config).pop("noise_std_mult_range"),
    attached=True,
)


def additive_only_noise(config):
    perception = config["perception"]
    perception["camera_warp_edge_noise"] = False
    perception["camera_warp_enable_holes"] = False
    perception["camera_apply_sensor_noise"] = False
    perception["camera_warp_additive_noise_std"] = 0.01
    perception["camera_warp_depth_offset_std"] = 0.0


save_variant(mutate=additive_only_noise, attached=True)


def duplicate_term(config):
    terms = config["randomization"]["reset_terms"]
    terms["second_camera_term"] = copy.deepcopy(terms["randomize_camera_raycast"])


save_variant(mutate=duplicate_term)


def bad_axis(config):
    value = camera_params(config)["translation_range"]
    value["sideways"] = value.pop("z")


save_variant(mutate=bad_axis)
save_variant(
    mutate=lambda config: camera_params(config)["translation_range"].__setitem__(
        "x", [0.0, float("inf")]
    )
)
save_variant(
    mutate=lambda config: camera_params(config)["rotation_range_deg"].__setitem__(
        "roll", [1.0, -1.0]
    )
)
save_variant(
    mutate=lambda config: camera_params(config).__setitem__(
        "noise_std_mult_range", [-0.01, 0.05]
    )
)
save_variant(
    mutate=lambda config: camera_params(config).__setitem__(
        "noise_drop_prob_range", [0.0, 1.01]
    )
)
save_variant(
    mutate=lambda config: config["simulator"]["config"]["sim"].__setitem__("fps", 0),
    attached=True,
)
save_variant(
    mutate=lambda config: config["perception"].__setitem__("camera_warp_enable_holes", True)
)
save_variant(attached=True, missing_hole_reference=True)
save_variant(attached=True, missing_geometry_support=True)
save_variant(attached=True, contract_version=1)
save_variant(attached=True, contract_tick_dt=0.01)
save_variant(attached=True, lifecycle_mismatch=True)
save_variant(attached=True, contract_mismatch=True)
PY

  PYTHONSAFEPATH=1 \
    PYTHONPATH="$MUJOCO_TEST_PYTHONPATH" \
    "$MUJOCO_PY" - <<'PY' 2>/dev/null
import tyro

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.tyro_utils import TYRO_CONIFG

args = [
    "simulator:mujoco",
    "robot:g1_29dof_w_object",
    "terrain:terrain_locomotion_plane",
    "perception:camera_depth_d435i",
    "--perception-randomization.enabled",
    "True",
    "--perception-randomization.translation-range",
    '{"x":[-0.025,0.025],"y":[-0.025,0.025],"z":[-0.025,0.025]}',
    "--perception-randomization.rotation-range-deg",
    '{"roll":[-2.5,2.5],"pitch":[-3,3],"yaw":[-2.5,2.5]}',
    "--perception-randomization.noise-std-mult-range",
    "[0,0.05]",
    "--perception-randomization.noise-drop-prob-range",
    "[0,0.025]",
    "--perception-producer-tick-dt",
    "0.02",
    "--perception-allow-mujoco-noise",
    "True",
    "--training.seed",
    "42",
    "--perception.camera-warp-hole-reference-batch-size",
    "4096",
]
config = tyro.cli(RunSimConfig, args=args, config=TYRO_CONIFG)
assert config.perception_randomization.translation_range == {
    "x": [-0.025, 0.025],
    "y": [-0.025, 0.025],
    "z": [-0.025, 0.025],
}
assert config.perception_randomization.noise_std_mult_range == [0, 0.05]
assert config.perception_producer_tick_dt == 0.02
assert config.perception_allow_mujoco_noise is True
assert config.training.seed == 42
assert config.perception.camera_warp_hole_reference_batch_size == 4096
PY

  COMMON_ENV=(
    env
    HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1
    DRY_RUN=1
    RUN_DIR="${TMP_DIR}/run"
    OBJECT_URDF="$OBJECT_FIXTURE"
    INFERENCE_CONFIG=g1-29dof-wbt-object-distill
    MUJOCO_PY="$MUJOCO_PY"
    INFER_PY="$INFER_PY"
    MUJOCO_PYTHONPATH="$MUJOCO_TEST_PYTHONPATH"
    PERCEPTION_CAMERA_WARP_EDGE_NOISE=False
    PERCEPTION_CAMERA_WARP_ENABLE_HOLES=False
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE=False
    PERCEPTION_CAMERA_WARP_NORMALIZE=True
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE=False
  )

  run_checkpoint_model() {
    local model="$1"
    shift
    "${COMMON_ENV[@]}" \
      PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT=0 \
      PERCEPTION_CAMERA_WARP_ENABLE_HOLES_EXPLICIT=0 \
      PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=0 \
      PERCEPTION_CAMERA_WARP_NORMALIZE_EXPLICIT=0 \
      HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE_EXPLICIT=0 \
      "$@" \
      bash mj_track.sh "$MOTION_FIXTURE" "$model"
  }

  expect_model_failure() {
    local label="$1"
    local model="$2"
    local expected="$3"
    shift 3
    local output="${TMP_DIR}/${label}.out"
    if run_checkpoint_model "$model" "$@" >"$output" 2>&1; then
      fail "${label} unexpectedly succeeded"
    fi
    grep -F "$expected" "$output" >/dev/null \
      || fail "${label} did not produce the expected error: ${expected}"
  }

  run_checkpoint_model "$MODEL_WITH_ATTACHED_CONTRACT" >"${TMP_DIR}/checkpoint_wins.out"
  grep -F 'warp_edge_noise=True' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'warp_holes=True' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'sensor_noise=True' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'warp_normalize=False' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'allow_mujoco_noise=True' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'contract=attached-v2-targeted-v2-distribution-verified enabled=True' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'translation={"x":[-0.025,0.025],"y":[-0.025,0.025],"z":[-0.025,0.025]}' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'rotation_deg={"roll":[-2.5,2.5],"pitch":[-3.0,3.0],"yaw":[-2.5,2.5]}' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'noise_std_mult=[0.0,0.05]' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'noise_drop_prob=[0.0,0.025]' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'producer_tick_dt=0.02' "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F 'producer_seed=42 seed_replay_scope=same_execution_trace_only' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F -- '--perception-randomization.enabled True' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F -- '--perception-producer-tick-dt 0.02' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -F -- '--perception-allow-mujoco-noise True' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null
  grep -E -- '--perception-contract-envelope-b64 [A-Za-z0-9+/]+={0,2}([[:space:]]|$)' \
    "${TMP_DIR}/checkpoint_wins.out" >/dev/null \
    || fail "direct run_sim command lacks a non-empty authenticated contract envelope"
  grep -F -- '--training.seed 42' "${TMP_DIR}/checkpoint_wins.out" >/dev/null

  run_checkpoint_model "$MODEL_WITH_ATTACHED_CONTRACT" \
    >"${TMP_DIR}/attached_contract.out"
  grep -F 'contract=attached-v2-targeted-v2-distribution-verified enabled=True' \
    "${TMP_DIR}/attached_contract.out" >/dev/null
  grep -F 'hole_reference_batch_size=4096' \
    "${TMP_DIR}/attached_contract.out" >/dev/null
  grep -F -- '--perception.camera-warp-hole-reference-batch-size 4096' \
    "${TMP_DIR}/attached_contract.out" >/dev/null

  run_checkpoint_model "$MODEL_SINGLE_SENSOR_NOISE" \
    >"${TMP_DIR}/single_sensor_noise.out"
  grep -F -- '--perception-randomization.noise-drop-prob-range' \
    "${TMP_DIR}/single_sensor_noise.out" >/dev/null
  if grep -F -- '--perception-randomization.noise-std-mult-range' \
    "${TMP_DIR}/single_sensor_noise.out" >/dev/null; then
    fail "single-component sensor noise unexpectedly synthesized a multiplicative range"
  fi

  run_checkpoint_model "$MODEL_ADDITIVE_ONLY_NOISE" \
    >"${TMP_DIR}/additive_only_noise.out"
  grep -F -- '--perception-allow-mujoco-noise True' \
    "${TMP_DIR}/additive_only_noise.out" >/dev/null \
    || fail "additive-only camera noise did not enable the authenticated MuJoCo noise path"

  run_box_model() {
    local model="$1"
    env \
      DRY_RUN=1 \
      RUN_DIR="${TMP_DIR}/box_run" \
      MOTION_FILE="$MOTION_FIXTURE" \
      OBJECT_URDF="$OBJECT_FIXTURE" \
      MODEL_INPUT="$model" \
      INFERENCE_CONFIG=g1-29dof-wbt-object-distill \
      INFER_PYTHON_BIN="$INFER_PY" \
      INFER_PY="$INFER_PY" \
      MUJOCO_PY="$MUJOCO_PY" \
      MUJOCO_PYTHONPATH="$MUJOCO_TEST_PYTHONPATH" \
      CUDA_VISIBLE_DEVICES=-1 \
      SIM_DEVICE=cpu \
      MUJOCO_CPUSET= \
      PERCEPTION_CAMERA_WARP_EDGE_NOISE= \
      PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT=0 \
      PERCEPTION_CAMERA_WARP_ENABLE_HOLES= \
      PERCEPTION_CAMERA_WARP_ENABLE_HOLES_EXPLICIT=0 \
      PERCEPTION_CAMERA_APPLY_SENSOR_NOISE= \
      PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=0 \
      HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE= \
      HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE_EXPLICIT=0 \
      bash mj_box_depth_track.sh warp
  }

  run_box_model "$MODEL_WITH_ATTACHED_CONTRACT" >"${TMP_DIR}/box_attached.out"
  grep -F 'contract=attached-v2-targeted-v2-distribution-verified enabled=True' \
    "${TMP_DIR}/box_attached.out" >/dev/null
  grep -F -- '--perception-producer-tick-dt 0.02' \
    "${TMP_DIR}/box_attached.out" >/dev/null
  if run_box_model "$MODEL_WITH_NOISE" >"${TMP_DIR}/box_legacy.out" 2>&1; then
    fail "mj_box_depth_track accepted a legacy perception artifact"
  fi
  grep -F 'requires an attached version-2 observation contract' \
    "${TMP_DIR}/box_legacy.out" >/dev/null \
    || fail "mj_box_depth_track legacy rejection was not actionable"

  run_checkpoint_model "$MODEL_WITH_ATTACHED_CONTRACT" \
    SIM_FPS=2000 \
    SIM_CONTROL_DECIMATION=1 \
    >"${TMP_DIR}/decoupled_tick.out"
  grep -F -- '--simulator.config.sim.fps 2000' "${TMP_DIR}/decoupled_tick.out" >/dev/null
  grep -F -- '--simulator.config.sim.control-decimation 1' \
    "${TMP_DIR}/decoupled_tick.out" >/dev/null
  grep -F -- '--perception-producer-tick-dt 0.02' \
    "${TMP_DIR}/decoupled_tick.out" >/dev/null

  "${COMMON_ENV[@]}" \
    PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT=1 \
    PERCEPTION_CAMERA_WARP_ENABLE_HOLES_EXPLICIT=1 \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=1 \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE=True \
    PERCEPTION_CAMERA_WARP_NORMALIZE_EXPLICIT=0 \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE_EXPLICIT=1 \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE=True \
    PERCEPTION_RANDOMIZATION_ENABLED=True \
    PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE='{"z":[-0.025,0.025],"y":[-0.025,0.025],"x":[-0.025,0.025]}' \
    PERCEPTION_RANDOMIZATION_ROTATION_RANGE_DEG='{"yaw":[-2.5,2.5],"pitch":[-3.0,3.0],"roll":[-2.5,2.5]}' \
    PERCEPTION_RANDOMIZATION_NOISE_STD_MULT_RANGE='[0,0.05]' \
    PERCEPTION_RANDOMIZATION_NOISE_DROP_PROB_RANGE='[0,0.025]' \
    PERCEPTION_PRODUCER_TICK_DT=0.0200000000000000 \
    PERCEPTION_PRODUCER_SEED=42 \
    HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT=False \
    bash mj_track.sh "$MOTION_FIXTURE" "$MODEL_WITH_ATTACHED_CONTRACT" \
    >"${TMP_DIR}/explicit_matching.out"
  grep -F 'sensor_noise=True' "${TMP_DIR}/explicit_matching.out" >/dev/null
  grep -F 'contract=attached-v2-targeted-v2-distribution-verified enabled=True' \
    "${TMP_DIR}/explicit_matching.out" >/dev/null

  if "${COMMON_ENV[@]}" \
    PERCEPTION_CAMERA_WARP_EDGE_NOISE_EXPLICIT=0 \
    PERCEPTION_CAMERA_WARP_ENABLE_HOLES_EXPLICIT=0 \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=0 \
    PERCEPTION_CAMERA_WARP_NORMALIZE_EXPLICIT=1 \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE_EXPLICIT=0 \
    bash mj_track.sh "$MOTION_FIXTURE" "$MODEL_WITH_ATTACHED_CONTRACT" \
    >"${TMP_DIR}/normalize_mismatch.out" 2>&1; then
    fail "explicit depth-normalization drift unexpectedly succeeded"
  fi
  grep -F 'refusing depth-unit drift' "${TMP_DIR}/normalize_mismatch.out" >/dev/null \
    || fail "depth-normalization mismatch did not produce an actionable error"

  expect_model_failure \
    duplicate_camera_term \
    "$MODEL_DUPLICATE_CAMERA_TERM" \
    'Expected at most one enabled canonical randomize_camera_raycast reset term'
  expect_model_failure \
    bad_camera_axis \
    "$MODEL_BAD_CAMERA_AXIS" \
    'translation_range axes must be exactly'
  expect_model_failure \
    nonfinite_camera_range \
    "$MODEL_NONFINITE_CAMERA_RANGE" \
    'translation_range.x[1] must be finite'
  expect_model_failure \
    reversed_camera_range \
    "$MODEL_REVERSED_CAMERA_RANGE" \
    'rotation_range_deg.roll must satisfy low <= high'
  expect_model_failure \
    negative_noise_std \
    "$MODEL_NEGATIVE_NOISE_STD" \
    'noise_std_mult_range must be >= 0.0'
  expect_model_failure \
    bad_noise_drop \
    "$MODEL_BAD_NOISE_DROP" \
    'noise_drop_prob_range must be <= 1.0'
  expect_model_failure \
    bad_training_fps \
    "$MODEL_BAD_TRAINING_FPS" \
    'Training simulator fps and control_decimation must both be positive'
  expect_model_failure \
    legacy_missing_targeted_contract \
    "$MODEL_WITH_NOISE" \
    'requires an attached version-2 observation contract with a targeted_v2 producer_lifecycle'
  expect_model_failure \
    legacy_holes_without_reference \
    "$MODEL_LEGACY_HOLES" \
    'requires an attached version-2 observation contract with a targeted_v2 producer_lifecycle'
  expect_model_failure \
    missing_hole_reference \
    "$MODEL_MISSING_HOLE_REFERENCE" \
    'hole_generator_schema.reference_batch_size must be a positive integer'
  expect_model_failure \
    missing_geometry_support \
    "$MODEL_MISSING_GEOMETRY_SUPPORT" \
    'lacks object-valued training_geometry_support'
  expect_model_failure \
    contract_v1 \
    "$MODEL_CONTRACT_V1" \
    'requires perception observation contract version=2'
  expect_model_failure \
    contract_tick_mismatch \
    "$MODEL_CONTRACT_TICK_MISMATCH" \
    'Attached perception producer_tick_dt conflicts with checkpoint simulator cadence'
  expect_model_failure \
    lifecycle_mismatch \
    "$MODEL_LIFECYCLE_MISMATCH" \
    'requires the authenticated version-2 targeted_v2 producer_lifecycle distribution contract'
  expect_model_failure \
    attached_contract_mismatch \
    "$MODEL_CONTRACT_MISMATCH" \
    'Attached perception camera_reset_randomization conflicts with experiment_config'
  expect_model_failure \
    explicit_randomization_enabled_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_RANDOMIZATION_ENABLED conflicts' \
    PERCEPTION_RANDOMIZATION_ENABLED=False
  expect_model_failure \
    explicit_translation_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE conflicts' \
    PERCEPTION_RANDOMIZATION_TRANSLATION_RANGE='{"x":[0,0],"y":[0,0],"z":[0,0]}'
  expect_model_failure \
    explicit_tick_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_PRODUCER_TICK_DT conflicts with checkpoint training cadence' \
    PERCEPTION_PRODUCER_TICK_DT=0.01
  expect_model_failure \
    explicit_seed_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_PRODUCER_SEED conflicts with checkpoint training seed' \
    PERCEPTION_PRODUCER_SEED=7
  expect_model_failure \
    legacy_setup_jitter_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT enables legacy one-shot jitter' \
    HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT=True
  expect_model_failure \
    explicit_sensor_noise_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_CAMERA_APPLY_SENSOR_NOISE conflicts with checkpoint metadata' \
    PERCEPTION_CAMERA_APPLY_SENSOR_NOISE_EXPLICIT=1
  expect_model_failure \
    explicit_allow_noise_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_ALLOW_MUJOCO_NOISE conflicts with checkpoint perception noise settings' \
    PERCEPTION_ALLOW_MUJOCO_NOISE=False
  expect_model_failure \
    explicit_legacy_allow_noise_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE conflicts with checkpoint perception noise settings' \
    HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE_EXPLICIT=1
  expect_model_failure \
    explicit_hole_reference_conflict \
    "$MODEL_WITH_ATTACHED_CONTRACT" \
    'Explicit PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE conflicts' \
    PERCEPTION_CAMERA_WARP_HOLE_REFERENCE_BATCH_SIZE=1
else
  echo "[SKIP] real ONNX MuJoCo dry-run: compatible local runtimes or tracked fixtures unavailable"
fi

echo "[PASS] MuJoCo perception metadata launcher contracts"
