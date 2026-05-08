#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
checkpoint="${HOLOSOMA_WANDB_CHECKPOINT:-}"
run_ref="${HOLOSOMA_WANDB_RUN:-zihanw22/boxer/w5qostjn}"
motion_init="${HOLOSOMA_MJ_MOTION_INIT:-0}"
auto_start="${HOLOSOMA_RO_AUTO_START:-0}"
auto_motion="${HOLOSOMA_RO_AUTO_MOTION:-0}"
use_sim_state="${HOLOSOMA_RO_USE_SIM_STATE:-1}"
explicit_motion_mode=0
clip_arg_seen=0
if [[ -n "${HOLOSOMA_MJ_MOTION_INIT:-}" ]]; then
  explicit_motion_mode=1
fi
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-init)
      motion_init=1
      explicit_motion_mode=1
      ;;
    --manual)
      motion_init=0
      explicit_motion_mode=1
      ;;
    --auto-start|--rollout)
      auto_start=1
      ;;
    --auto-motion)
      auto_start=1
      auto_motion=1
      ;;
    --use-sim-state)
      use_sim_state=1
      ;;
    --no-sim-state)
      use_sim_state=0
      ;;
    --clip)
      shift
      clip="$1"
      clip_arg_seen=1
      ;;
    --checkpoint)
      shift
      checkpoint="$1"
      ;;
    --run)
      shift
      run_ref="$1"
      ;;
    *)
      positional+=("$1")
      ;;
  esac
  shift
done

if (( ${#positional[@]} >= 1 )); then
  clip="${positional[0]}"
  clip_arg_seen=1
fi
if (( ${#positional[@]} >= 2 )); then
  checkpoint="${positional[1]}"
fi
if (( ${#positional[@]} >= 3 )); then
  run_ref="${positional[2]}"
fi

if [[ "$explicit_motion_mode" == "0" && "$clip_arg_seen" == "1" ]]; then
  motion_init=1
fi

motion_file="$clip"
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
fi

if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

export HOLOSOMA_MJ_MOTION="$motion_file"
export HOLOSOMA_WANDB_CHECKPOINT="$checkpoint"
export HOLOSOMA_WANDB_RUN="$run_ref"
export HOLOSOMA_MJ_MOTION_INIT="$motion_init"
export HOLOSOMA_RO_AUTO_START="$auto_start"
export HOLOSOMA_RO_AUTO_MOTION="$auto_motion"
export HOLOSOMA_RO_USE_SIM_STATE="$use_sim_state"
export SIM_CLOCK_PORT="${SIM_CLOCK_PORT:-5555}"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ "$(basename "$clip" .npz)" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi
export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference:${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"

python_code="$(cat <<'PY'
import os
import json
import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import onnx
import onnxruntime
import wandb
from onnx import numpy_helper

from holosoma_inference.config.config_types.observation import ObservationConfig
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.config.config_values.observation import wbt_object_perception_g1
from holosoma_inference.run_policy import run_policy
from holosoma_inference.utils.wandb import load_checkpoint

DEFAULT_RUN_PATH = "zihanw22/boxer/w5qostjn"
MOTION_FILE = os.environ.get("HOLOSOMA_MJ_MOTION", "data_demo/box_75.npz")
CHECKPOINT = os.environ.get("HOLOSOMA_WANDB_CHECKPOINT", "").strip()
RUN_REF = os.environ.get("HOLOSOMA_WANDB_RUN", DEFAULT_RUN_PATH).strip() or DEFAULT_RUN_PATH
AUTO_START_POLICY = os.environ.get("HOLOSOMA_RO_AUTO_START", "").strip().lower() in {"1", "true", "yes", "on"}
AUTO_START_MOTION = os.environ.get("HOLOSOMA_RO_AUTO_MOTION", "").strip().lower() in {"1", "true", "yes", "on"}
USE_SIM_STATE = os.environ.get("HOLOSOMA_RO_USE_SIM_STATE", "1").strip().lower() in {"1", "true", "yes", "on"}
SIM_CLOCK_PORT = int(os.environ.get("SIM_CLOCK_PORT", "5555") or "5555")
SIM_STATE_PORT = int(os.environ.get("SIM_STATE_PORT", "5557") or "5557")
DOMAIN_ID = int(os.environ.get("HOLOSOMA_DDS_DOMAIN_ID", "0") or "0")


def normalize_run_path(run_ref: str) -> str:
    if run_ref.startswith("https://wandb.ai/"):
        path_parts = [part for part in urlparse(run_ref).path.split("/") if part]
        if len(path_parts) >= 4 and path_parts[2] == "runs":
            return f"{path_parts[0]}/{path_parts[1]}/{path_parts[3]}"
    if run_ref.startswith("wandb://"):
        run_ref = run_ref[len("wandb://") :]
    parts = [part for part in run_ref.split("/") if part]
    if len(parts) == 1:
        return f"zihanw22/boxer/{parts[0]}"
    if len(parts) >= 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}/{parts[2]}"
    raise ValueError(f"Invalid W&B run reference: {run_ref}")


def split_checkpoint_ref(checkpoint: str, default_run_path: str) -> tuple[str, str]:
    if checkpoint.startswith("https://wandb.ai/"):
        parsed = urlparse(checkpoint)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 6 and parts[2] == "runs" and parts[4] == "files":
            return f"{parts[0]}/{parts[1]}/{parts[3]}", parts[-1]
    if checkpoint.startswith("wandb://"):
        parts = [part for part in checkpoint[len("wandb://") :].split("/") if part]
        if len(parts) >= 5 and parts[2] == "runs":
            return f"{parts[0]}/{parts[1]}/{parts[3]}", parts[-1]
        if len(parts) >= 4:
            return f"{parts[0]}/{parts[1]}/{parts[2]}", parts[-1]
    return default_run_path, checkpoint


RUN_PATH, CHECKPOINT = split_checkpoint_ref(CHECKPOINT, normalize_run_path(RUN_REF)) if CHECKPOINT else (
    normalize_run_path(RUN_REF),
    CHECKPOINT,
)
RUN_ID = RUN_PATH.rsplit("/", 1)[-1]
if RUN_ID == "tvtwx4to":
    os.environ["HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND"] = "1"
else:
    os.environ["HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND"] = "0"

run = wandb.Api().run(RUN_PATH)
onnx_files = [file for file in run.files() if file.name.endswith(".onnx")]
if not onnx_files:
    raise RuntimeError(f"No .onnx files found in W&B run {RUN_PATH}")

def checkpoint_step(file_name):
    match = re.search(r"model_(\d+)\.onnx$", file_name)
    return int(match.group(1)) if match else -1


def select_model_name(checkpoint: str) -> str:
    if not checkpoint or checkpoint == "latest":
        latest_file = max(onnx_files, key=lambda file: (checkpoint_step(file.name), file.updated_at or ""))
        return latest_file.name
    if checkpoint.isdigit():
        checkpoint = f"model_{checkpoint}.onnx"
    if checkpoint.startswith("wandb://"):
        return checkpoint.rsplit("/", 1)[-1]
    return checkpoint


def download_model(model_name: str) -> str:
    available = {file.name for file in onnx_files}
    if model_name not in available:
        raise RuntimeError(f"Checkpoint '{model_name}' not found in W&B run {RUN_PATH}")

    model_path = f"wandb://{RUN_PATH}/{model_name}"
    downloaded_path = Path(load_checkpoint(None, model_path, "/tmp"))
    named_path = downloaded_path.with_name(f"{RUN_ID}_{downloaded_path.name}")
    if downloaded_path != named_path:
        downloaded_path.replace(named_path)
    return str(named_path)


selected_model = select_model_name(CHECKPOINT)
model_path = f"wandb://{RUN_PATH}/{selected_model}"
local_model_path = download_model(selected_model)
local_model_file = Path(local_model_path)
assert local_model_file.is_file(), local_model_path
assert local_model_file.name == f"{RUN_ID}_{selected_model}", local_model_path


def decode_names(values):
    decoded = []
    for item in values.tolist():
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def motion_ref_body_name(metadata: dict) -> str:
    motion_cfg = (
        metadata.get("experiment_config", {})
        .get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )
    body_name_ref = motion_cfg.get("body_name_ref", ["torso_link"]) if isinstance(motion_cfg, dict) else ["torso_link"]
    return body_name_ref[0] if isinstance(body_name_ref, list) and body_name_ref else "torso_link"


def root_body_index(body_names: list[str]) -> int:
    for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
        if candidate in body_names:
            return body_names.index(candidate)
    return 0


def load_motion_clip(motion_path: str, dof_names: list[str], ref_body_name: str) -> dict[str, np.ndarray]:
    with np.load(motion_path, allow_pickle=True) as data:
        joint_names = decode_names(np.asarray(data["joint_names"]))
        body_names = decode_names(np.asarray(data["body_names"]))
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        if joint_pos.shape[1] == len(joint_names) + 7:
            joint_pos = joint_pos[:, 7:]
        if joint_vel.shape[1] == len(joint_names) + 6:
            joint_vel = joint_vel[:, 6:]
        joint_indices = [joint_names.index(name) for name in dof_names]
        ref_idx = body_names.index(ref_body_name)
        root_idx = root_body_index(body_names)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
    return {
        "joint_pos": joint_pos[:, joint_indices],
        "joint_vel": joint_vel[:, joint_indices],
        "ref_pos_xyz": body_pos_w[:, ref_idx, :],
        "ref_quat_xyzw": body_quat_w[:, ref_idx, :][:, [1, 2, 3, 0]],
        "root_pos_w": body_pos_w[:, root_idx, :],
        "root_quat_wxyz": body_quat_w[:, root_idx, :],
    }


def find_node_by_output(model: onnx.ModelProto, output_name: str, op_type: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if output_name in node.output and node.op_type == op_type:
            return node
    raise KeyError(f"Could not find {op_type} node producing '{output_name}'")


def find_constant_node(model: onnx.ModelProto, const_output_name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if const_output_name in node.output and node.op_type == "Constant":
            return node
    raise KeyError(f"Could not find Constant node for '{const_output_name}'")


def set_constant_tensor(const_node: onnx.NodeProto, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(np.asarray(value))
    for attr in const_node.attribute:
        if attr.name == "value":
            attr.t.CopyFrom(tensor)
            return
    raise KeyError(f"Constant node '{const_node.name}' has no value attribute")


def patch_onnx_motion_constants(model_path: str, motion_path: str) -> dict[str, np.ndarray]:
    model = onnx.load(model_path)
    metadata = onnx_metadata_from_model(model)
    dof_names = list(metadata["dof_names"])
    motion = load_motion_clip(motion_path, dof_names, motion_ref_body_name(metadata))

    for output_name in ("joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"):
        gather_node = find_node_by_output(model, output_name, "Gather")
        const_node = find_constant_node(model, gather_node.input[0])
        set_constant_tensor(const_node, motion[output_name].astype(np.float32, copy=False))

    joint_gather = find_node_by_output(model, "joint_pos", "Gather")
    clip_node = find_node_by_output(model, joint_gather.input[1], "Clip")
    max_const = find_constant_node(model, clip_node.input[2])
    set_constant_tensor(max_const, np.array([motion["joint_pos"].shape[0] - 1], dtype=np.int64))

    experiment_config = metadata.get("experiment_config")
    if isinstance(experiment_config, dict):
        motion_cfg = (
            experiment_config.setdefault("command", {})
            .setdefault("setup_terms", {})
            .setdefault("motion_command", {})
            .setdefault("params", {})
            .setdefault("motion_config", {})
        )
        if isinstance(motion_cfg, dict):
            motion_cfg["motion_file"] = motion_path
            motion_cfg["motion_clip_id"] = 0
            motion_cfg["motion_clip_name"] = Path(motion_path).stem

    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)

    tmp_path = Path(model_path).with_suffix(".patching.onnx")
    onnx.save(model, tmp_path)
    tmp_path.replace(model_path)
    return motion


def assert_onnx_motion_matches(model_path: str, motion: dict[str, np.ndarray]) -> None:
    session = onnxruntime.InferenceSession(model_path)
    input_feed = {}
    for input_meta in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
        input_feed[input_meta.name] = np.zeros(tuple(shape), dtype=np.float32)
    frame_count = motion["joint_pos"].shape[0]
    sample_frames = sorted({0, min(1, frame_count - 1), min(50, frame_count - 1), min(100, frame_count - 1), frame_count - 1})
    for frame in sample_frames:
        feed = {name: value.copy() for name, value in input_feed.items()}
        if "time_step" in feed:
            feed["time_step"][...] = frame
        outputs = session.run(["joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"], feed)
        for output_name, output_value in zip(("joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"), outputs, strict=True):
            expected = motion[output_name][frame : frame + 1]
            actual = np.asarray(output_value, dtype=np.float32).reshape(expected.shape)
            assert np.allclose(actual, expected, atol=1e-6), (
                f"ONNX motion mismatch for {output_name} frame {frame}: "
                f"maxdiff={float(np.max(np.abs(actual - expected)))}"
            )


def onnx_metadata_from_model(model: onnx.ModelProto):
    metadata = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except Exception:
            metadata[prop.key] = prop.value
    return metadata


def onnx_metadata(path):
    return onnx_metadata_from_model(onnx.load(path))


def onnx_input_dims(path):
    session = onnxruntime.InferenceSession(path)
    input_dims = {}
    for input_meta in session.get_inputs():
        if len(input_meta.shape) >= 2 and isinstance(input_meta.shape[1], int):
            input_dims[input_meta.name] = int(input_meta.shape[1])
    if "obs" not in input_dims:
        raise RuntimeError(f"ONNX model has no fixed-width 'obs' input: {path}")
    return input_dims


def actor_obs_dim(observation: ObservationConfig):
    total = 0
    for group_name, terms in observation.obs_dict.items():
        if not group_name.startswith("actor_obs"):
            continue
        history = observation.history_length_dict.get(group_name, 1)
        total += sum(observation.obs_dims[term] for term in terms) * history
    return total


def group_obs_dim(observation: ObservationConfig, group_name: str):
    terms = observation.obs_dict[group_name]
    history = observation.history_length_dict.get(group_name, 1)
    return sum(observation.obs_dims[term] for term in terms) * history


def observation_for_model(observation: ObservationConfig, expected_obs_dim: int, metadata: dict):
    actor_groups = (
        metadata.get("experiment_config", {})
        .get("algo", {})
        .get("config", {})
        .get("module_dict", {})
        .get("actor", {})
        .get("input_dim", [])
    )
    if not isinstance(actor_groups, list) or not actor_groups:
        actor_groups = ["actor_obs_root_contact_aware", "actor_obs_proprio"]

    if actor_obs_dim(observation) == expected_obs_dim:
        if set(actor_groups) == {"actor_obs_root_contact_aware", "actor_obs_proprio"}:
            return observation

    base_obs_dict = {group: list(terms) for group, terms in observation.obs_dict.items()}
    obs_dims = dict(observation.obs_dims)
    obs_scales = dict(observation.obs_scales)
    if "sparse_target_root_trajectory_command_contact_aware" in obs_dims:
        obs_dims.setdefault(
            "sparse_target_root_trajectory_command",
            obs_dims["sparse_target_root_trajectory_command_contact_aware"],
        )
    if "sparse_target_root_trajectory_command_contact_aware" in obs_scales:
        obs_scales.setdefault(
            "sparse_target_root_trajectory_command",
            obs_scales["sparse_target_root_trajectory_command_contact_aware"],
        )
    candidate_obs_dict = {"perception_obs": base_obs_dict["perception_obs"]}
    candidate_history = {"perception_obs": observation.history_length_dict.get("perception_obs", 1)}

    for group in actor_groups:
        if group == "actor_obs_root":
            candidate_obs_dict[group] = ["sparse_target_root_trajectory_command"]
            candidate_history[group] = 1
        elif group == "actor_obs_root_contact_aware":
            candidate_obs_dict[group] = ["sparse_target_root_trajectory_command_contact_aware"]
            candidate_history[group] = 1
        elif group == "actor_obs_proprio":
            candidate_obs_dict[group] = ["base_lin_vel", "base_ang_vel", "dof_pos", "dof_vel"]
            candidate_history[group] = 5
        elif group == "actor_obs_proprio_no_linvel":
            candidate_obs_dict[group] = ["base_ang_vel", "dof_pos", "dof_vel"]
            candidate_history[group] = 5
        else:
            raise RuntimeError(f"Unsupported ONNX actor observation group '{group}' in {local_model_path}")

    candidate = replace(
        observation,
        obs_dict=candidate_obs_dict,
        obs_dims=obs_dims,
        obs_scales=obs_scales,
        history_length_dict={
            **observation.history_length_dict,
            **candidate_history,
        },
    )
    if actor_obs_dim(candidate) == expected_obs_dim:
        print(f"[mj_ro] ONNX actor groups: {actor_groups}", flush=True)
        return candidate

    raise RuntimeError(
        f"ONNX obs input expects {expected_obs_dim}, but mj_ro observation config produces "
        f"{actor_obs_dim(candidate)} from actor groups {actor_groups}."
    )


def assert_observation_matches_onnx(observation: ObservationConfig, input_dims: dict[str, int]):
    actual_actor_dim = actor_obs_dim(observation)
    assert actual_actor_dim == input_dims["obs"], (
        f"actor obs mismatch: config={actual_actor_dim}, onnx={input_dims['obs']}"
    )
    if "perception_obs" in input_dims:
        actual_perception_dim = group_obs_dim(observation, "perception_obs")
        assert actual_perception_dim == input_dims["perception_obs"], (
            f"perception_obs mismatch: config={actual_perception_dim}, onnx={input_dims['perception_obs']}"
        )


motion_constants = patch_onnx_motion_constants(local_model_path, MOTION_FILE)
assert_onnx_motion_matches(local_model_path, motion_constants)

base_config = DEFAULTS["g1-wbt-distillation"]
input_dims = onnx_input_dims(local_model_path)
metadata = onnx_metadata(local_model_path)
observation = observation_for_model(wbt_object_perception_g1, input_dims["obs"], metadata)
assert_observation_matches_onnx(observation, input_dims)
config = replace(
    base_config,
    observation=observation,
    task=replace(
        base_config.task,
        model_path=local_model_path,
        interface="lo",
        domain_id=DOMAIN_ID,
        rl_rate=50,
        motion_file=MOTION_FILE,
        use_sim_time=True,
        sim_clock_port=SIM_CLOCK_PORT,
        use_sim_state=USE_SIM_STATE,
        sim_state_port=SIM_STATE_PORT,
        prefer_sim_ref_from_sim_state=USE_SIM_STATE,
        auto_start_policy=AUTO_START_POLICY,
        auto_start_motion_clip=AUTO_START_MOTION,
    ),
)

print(f"[mj_ro] Using W&B ONNX: {model_path}", flush=True)
print(f"[mj_ro] Local ONNX: {local_model_path}", flush=True)
print(f"[mj_ro] Patched ONNX motion: {MOTION_FILE}", flush=True)
print(f"[mj_ro] ONNX input dims: {input_dims}", flush=True)
print(f"[mj_ro] DDS domain: {DOMAIN_ID}", flush=True)
print(f"[mj_ro] Use sim state: {USE_SIM_STATE}", flush=True)
print(
    f"[mj_ro] Zero sparse root command: {os.environ.get('HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND', '0')}",
    flush=True,
)
print(f"[mj_ro] Actor obs terms: {observation.obs_dict}", flush=True)
print(f"[mj_ro] Auto start: policy={AUTO_START_POLICY}, motion_clip={AUTO_START_MOTION}", flush=True)
run_policy(config)
PY
)"

if [[ "$auto_start" == "1" ]]; then
  python3 -c "$python_code" </dev/null
elif [[ -t 0 ]]; then
  python3 -c "$python_code"
elif { exec 3</dev/tty; } 2>/dev/null; then
  python3 -c "$python_code" <&3
  exec 3<&-
else
  python3 -c "$python_code"
fi
