from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import json
import math
import os
import shlex
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.terrain import MeshType
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.defm_runtime import set_defm_checkpoint_restore_mode
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.policy_init_preflight import validate_fast_sac_actor_config_identity
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG
from holosoma.utils.visual_motion_transitions import configured_simulator_type


@dataclass(frozen=True)
class Sim2SimConfig:
    enabled: bool = False
    """Enable MuJoCo sim-to-sim workflow (export ONNX + build run_sim/run_policy commands)."""

    auto_launch: bool = False
    """If True, launch run_sim and run_policy as subprocesses."""

    simulator: str = "mujoco"
    """Simulator preset to use for run_sim (mujoco or mjwarp)."""

    interface: str = "lo"
    """Network interface for sim-to-sim bridge/inference."""

    use_joystick: bool = False
    """Enable joystick controls for sim-to-sim inference."""

    use_sim_time: bool | None = None
    """Override use_sim_time; defaults to True for WBT, False for locomotion."""

    rl_rate: float | None = None
    """Override policy inference rate (Hz). Defaults to training control rate."""

    inference_config: str | None = None
    """Override inference config (e.g., inference:g1-29dof-loco)."""

    model_path: str | None = None
    """Override ONNX path for sim-to-sim inference."""

    run_sim_args: str = ""
    """Extra args appended to run_sim.py (shell-style string)."""

    run_sim_robot: str | None = None
    """Override run_sim robot preset (e.g., g1-29dof-stairs)."""

    run_policy_args: str = ""
    """Extra args appended to run_policy.py (shell-style string)."""

    sim_python: str | None = None
    """Python executable used for run_sim.py. Defaults to the current interpreter."""

    policy_python: str | None = None
    """Python executable used for run_policy.py. Defaults to the current interpreter."""


@dataclass(frozen=True)
class EvalCliConfig:
    sim2sim: Sim2SimConfig = Sim2SimConfig()


def _get_export_paths(checkpoint_path: str) -> tuple[Path, Path]:
    checkpoint_file = Path(checkpoint_path)
    export_dir = checkpoint_file.parent / "exported"
    exported_onnx = export_dir / checkpoint_file.with_suffix(".onnx").name
    return export_dir, exported_onnx


def _read_stable_regular_file(path: Path, *, label: str) -> bytes:
    """Read one immutable regular-file snapshot and reject in-place races."""

    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular file: {path}")
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ValueError(f"Could not read {label} {path}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise RuntimeError(f"{label} changed while it was being authenticated: {path}")
    if not payload:
        raise ValueError(f"{label} is empty: {path}")
    return payload


def _preflight_perception_onnx_artifact(
    model_path: str,
    *,
    checkpoint_sha256: str,
    config: ExperimentConfig,
    return_contract_envelope: bool = False,
) -> str | tuple[str, str]:
    """Authenticate an ONNX perception artifact before starting run_sim."""

    from holosoma_inference.utils.policy_contract import (  # noqa: PLC0415
        perception_observation_contract_sha256_from_metadata,
    )
    import onnx  # noqa: PLC0415

    path = _existing_local_path(model_path, label="sim2sim ONNX model")
    payload = _read_stable_regular_file(path, label="sim2sim ONNX model")
    try:
        model = onnx.load_model_from_string(payload)
    except Exception as exc:
        raise ValueError(f"Could not parse sim2sim ONNX model {path}: {exc}") from exc

    def reject_nonfinite_json(constant: str):
        raise ValueError(f"non-finite JSON constant {constant!r}")

    def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    metadata: dict[str, object] = {}
    for prop in model.metadata_props:
        if not prop.key:
            raise ValueError(f"sim2sim ONNX model contains an empty metadata key: {path}")
        if prop.key in metadata:
            raise ValueError(
                f"sim2sim ONNX model contains duplicate metadata key {prop.key!r}: {path}"
            )
        try:
            metadata[prop.key] = json.loads(
                prop.value,
                parse_constant=reject_nonfinite_json,
                object_pairs_hook=reject_duplicate_json_keys,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sim2sim ONNX metadata {prop.key!r} is not strict finite JSON: {path}"
            ) from exc

    attached_source = metadata.get("source_checkpoint_sha256")
    if attached_source != checkpoint_sha256:
        raise ValueError(
            "sim2sim ONNX source checkpoint does not match the checkpoint selected for evaluation: "
            f"attached={attached_source!r}, expected={checkpoint_sha256}."
        )
    contract_sha256 = perception_observation_contract_sha256_from_metadata(metadata)
    if contract_sha256 is None:
        raise ValueError(
            "Perception policy ONNX is missing its authenticated effective observation contract."
        )
    contract = metadata["perception_observation_contract"]
    perception_cfg = getattr(config, "perception", None)
    effective_observation_schema = contract.get("effective_observation_schema")
    holes_active = bool(
        isinstance(effective_observation_schema, Mapping)
        and effective_observation_schema.get("enable_holes") is True
        and float(effective_observation_schema.get("hole_prob", 0.0) or 0.0) > 0.0
    )
    hole_seed_semantics = str(
        getattr(
            perception_cfg,
            "camera_warp_hole_seed_semantics",
            "legacy_fixed_v1",
        )
    )
    hole_schema = contract.get("hole_generator_schema")
    if holes_active and not isinstance(hole_schema, Mapping):
        raise ValueError(
            "Perception policy ONNX is missing its active hole-generator contract."
        )
    if holes_active and hole_seed_semantics == "rank_local_v2":
        expected_rank_zero_seed = getattr(getattr(config, "training", None), "seed", None)
        if (
            isinstance(expected_rank_zero_seed, bool)
            or not isinstance(expected_rank_zero_seed, int)
            or expected_rank_zero_seed < 0
        ):
            raise ValueError(
                "Checkpoint rank-local hole reconstruction requires a non-negative integer training seed."
            )
        if (
            hole_schema.get("seed_semantics") != "rank_local_v2"
            or hole_schema.get("gradient_seed_mixer") != "sha256_u63_be_v1"
            or hole_schema.get("effective_seed") != expected_rank_zero_seed
        ):
            raise ValueError(
                "Perception policy ONNX rank-local hole seed contract does not match the "
                "saved checkpoint's rank-zero training seed."
            )
    elif holes_active and hole_seed_semantics == "legacy_fixed_v1":
        if any(
            field_name in hole_schema
            for field_name in (
                "seed_semantics",
                "effective_seed",
                "gradient_seed_mixer",
                "octave_profile",
            )
        ):
            raise ValueError(
                "Legacy fixed-hole checkpoint is paired with a rank-local ONNX hole contract."
            )
    elif holes_active:
        raise ValueError(
            f"Unsupported checkpoint Perlin hole seed semantics: {hole_seed_semantics!r}."
        )
    expected_tick_dt = (
        float(config.simulator.config.sim.control_decimation)
        / float(config.simulator.config.sim.fps)
    )
    attached_tick_dt = float(contract["producer_tick_dt"])
    if not math.isclose(attached_tick_dt, expected_tick_dt, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "sim2sim ONNX perception producer tick does not match the saved checkpoint config: "
            f"attached={attached_tick_dt}, expected={expected_tick_dt}."
        )
    if not return_contract_envelope:
        return contract_sha256
    envelope_payload = json.dumps(
        {
            "contract": contract,
            "sha256": contract_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return contract_sha256, base64.b64encode(envelope_payload).decode("ascii")


def _is_wbt_experiment(config: ExperimentConfig) -> bool:
    if "wbt" in config.env_class.lower():
        return True
    if config.command is None:
        return False
    for term in config.command.setup_terms.values():
        func = term.func.lower()
        if "motioncommand" in func or "motion_command" in func:
            return True
    return False


def _is_object_distill_experiment(config: ExperimentConfig) -> bool:
    observation_cfg = getattr(config, "observation", None)
    groups = getattr(observation_cfg, "groups", None)
    if not isinstance(groups, dict):
        return False
    return "actor_obs_box" in groups and ("actor_obs_proprio" in groups or "actor_obs_root" in groups)


def _actor_input_groups(config: ExperimentConfig) -> list[str]:
    module_dict = getattr(config.algo.config, "module_dict", None)
    actor = getattr(module_dict, "actor", None)
    input_dim = getattr(actor, "input_dim", None)
    if isinstance(input_dim, (list, tuple)):
        return list(input_dim)
    fast_sac_inputs = getattr(config.algo.config, "actor_obs_keys", None)
    return list(fast_sac_inputs) if isinstance(fast_sac_inputs, (list, tuple)) else []


def _actor_perception_input_name(config: ExperimentConfig) -> str:
    module_dict = getattr(config.algo.config, "module_dict", None)
    actor = getattr(module_dict, "actor", None)
    layer_cfg = getattr(actor, "layer_config", None)
    perception_input = str(getattr(layer_cfg, "perception_input_name", "") or "")
    if perception_input:
        return perception_input
    if bool(getattr(config.algo.config, "use_cnn_encoder", False)):
        return str(getattr(config.algo.config, "encoder_obs_key", "") or "")
    return ""


def _infer_exact_nonperception_wbt_suffix(config: ExperimentConfig, actor_groups: list[str]) -> str:
    """Select a WBT deployment preset from the complete training group contract."""

    if actor_groups != ["actor_obs"]:
        raise ValueError(
            "No safe non-perception WBT inference preset matches actor input groups "
            f"{actor_groups!r}. A flattened dimension is not a sufficient observation contract."
        )
    observation_cfg = getattr(config, "observation", None)
    groups = getattr(observation_cfg, "groups", None)
    if not isinstance(groups, dict):
        raise ValueError("Checkpoint is missing the serialized observation groups required for deployment.")
    actor_group = groups.get("actor_obs")
    if actor_group is None:
        raise ValueError("Checkpoint is missing its serialized actor_obs group.")

    from holosoma.config_values.wbt.g1 import observation as wbt_observation

    exact_groups = {
        # The deployment-side plain WBT preset is the historical history-1
        # contract.  Current history-5 WBT metadata must not be mapped to it.
        "wbt": dataclasses.replace(wbt_observation.actor_obs_shared, history_length=1),
        "wbt-object-generalist": (
            wbt_observation.g1_29dof_wbt_observation_w_object.groups["actor_obs"]
        ),
        "wbt-w-object-history1": dataclasses.replace(
            wbt_observation.g1_29dof_wbt_observation_w_object.groups["actor_obs"],
            history_length=1,
        ),
        "wbt-w-object-legacy": (
            wbt_observation.g1_29dof_wbt_observation_w_object_legacy.groups["actor_obs"]
        ),
    }

    velocity_terms = dict(wbt_observation.actor_obs_w_object_legacy_terms)
    velocity_terms["obj_lin_vel_b"] = wbt_observation.critic_obs_w_object_terms["obj_lin_vel_b"]
    velocity_terms["obj_ang_vel_b"] = (
        wbt_observation.critic_obs_w_object_command_privileged_terms["obj_ang_vel_b"]
    )
    exact_groups["wbt-object-velocity-generalist"] = dataclasses.replace(
        wbt_observation.actor_obs_w_object_legacy,
        history_length=1,
        terms=velocity_terms,
    )

    matches = [suffix for suffix, expected_group in exact_groups.items() if actor_group == expected_group]
    if len(matches) != 1:
        term_names = sorted(getattr(actor_group, "terms", {}))
        history_length = getattr(actor_group, "history_length", None)
        raise ValueError(
            "No unique inference preset matches the complete non-perception WBT actor contract: "
            f"matches={matches}, terms={term_names}, history_length={history_length!r}."
        )
    return matches[0]


def _actor_group_history_lengths(
    config: ExperimentConfig,
    actor_groups: list[str],
) -> tuple[int, ...]:
    """Read the serialized history contract for otherwise ambiguous groups."""

    observation_cfg = getattr(config, "observation", None)
    groups = getattr(observation_cfg, "groups", None)
    if not isinstance(groups, dict):
        raise ValueError(
            "Checkpoint is missing serialized observation groups required to distinguish "
            "AS contact-aware history presets."
        )
    histories: list[int] = []
    for group_name in actor_groups:
        group = groups.get(group_name)
        history = getattr(group, "history_length", None)
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise ValueError(
                f"Checkpoint actor group {group_name!r} has invalid history_length {history!r}."
            )
        histories.append(history)
    return tuple(histories)


def _infer_inference_config(config: ExperimentConfig) -> tuple[str, bool]:
    is_wbt = _is_wbt_experiment(config)
    robot_type = config.robot.asset.robot_type
    robot_map = {
        "g1_29dof": "g1-29dof",
        "t1_29dof": "t1-29dof",
    }
    base = robot_map.get(robot_type)
    if base is None:
        raise ValueError(f"Unsupported robot type for sim2sim inference: {robot_type}")
    if is_wbt and base != "g1-29dof":
        raise ValueError(
            f"No default WBT inference config for robot '{robot_type}'. "
            "Pass --sim2sim.inference-config explicitly."
        )

    actor_groups = _actor_input_groups(config)
    perception_input = _actor_perception_input_name(config)
    as_contact_aware_groups = (
        "actor_obs_root_contact_aware",
        "actor_obs_proprio_with_actions_no_linvel",
    )
    if tuple(actor_groups) == as_contact_aware_groups and perception_input == "perception_obs":
        histories = _actor_group_history_lengths(config, actor_groups)
        as_history_contracts = {
            (1, 1): "wbt-object-as-contact-aware-depth-distill",
            (1, 5): "wbt-object-as-contact-aware-history5-depth-distill",
        }
        exact_suffix = as_history_contracts.get(histories)
        if exact_suffix is None:
            raise ValueError(
                "No safe inference preset matches the serialized AS contact-aware history contract: "
                f"actor_input_groups={actor_groups}, histories={histories}. "
                "Supported histories are command/proprio=(1, 1) and (1, 5)."
            )
        return f"inference:{base}-{exact_suffix}", is_wbt

    exact_wbt_contracts = {
        (
            (
                "actor_obs_root_contact_aware",
                "actor_obs_pickup_button",
                "actor_obs_drop_button",
                "actor_obs_proprio_with_actions_no_linvel",
            ),
            "perception_obs",
        ): "wbt-object-contact-aware-dual-button-depth-distill",
        (
            (
                "actor_obs_root_contact_aware",
                "actor_obs_drop_button",
                "actor_obs_proprio_with_actions_no_linvel",
            ),
            "perception_obs",
        ): "wbt-object-contact-aware-drop-button-depth-distill",
        (
            ("actor_obs_root_contact_aware", "actor_obs_proprio", "actor_obs_actions"),
            "perception_obs",
        ): "wbt-object-contact-aware-depth-distill",
        (
            ("actor_obs_root", "actor_obs_proprio", "actor_obs_actions"),
            "perception_obs",
        ): "wbt-object-as-depth-distill",
        (("actor_obs_root", "actor_obs_proprio_no_linvel"), "perception_obs"): "wbt-object-distill",
        (
            ("actor_obs_root", "actor_obs_proprio_no_linvel", "actor_obs_actions", "actor_obs_box"),
            "",
        ): "wbt-object-mocap-distill",
    }
    exact_suffix = exact_wbt_contracts.get((tuple(actor_groups), perception_input))
    if exact_suffix is not None:
        return f"inference:{base}-{exact_suffix}", is_wbt

    if is_wbt and perception_input:
        raise ValueError(
            "No safe inference preset matches this perception actor contract: "
            f"actor_input_groups={actor_groups}, perception_input={perception_input!r}. "
            "Pass --sim2sim.inference-config only after adding a matching deployment observation preset."
        )

    suffix = (
        _infer_exact_nonperception_wbt_suffix(config, actor_groups)
        if is_wbt
        else "loco"
    )
    return f"inference:{base}-{suffix}", is_wbt


def _validate_eval_policy_contract(saved_config: ExperimentConfig, eval_config: ExperimentConfig) -> None:
    """Reject CLI overrides that silently reinterpret checkpoint actor inputs."""

    if saved_config.algo._target_ != eval_config.algo._target_:
        raise ValueError(
            f"Evaluation algo override changes checkpoint type: {saved_config.algo._target_!r} -> "
            f"{eval_config.algo._target_!r}."
        )

    if _actor_perception_input_name(saved_config):
        saved_sim = saved_config.simulator.config.sim
        eval_sim = eval_config.simulator.config.sim
        saved_timing = (saved_sim.fps, saved_sim.control_decimation)
        eval_timing = (eval_sim.fps, eval_sim.control_decimation)
        if saved_timing != eval_timing:
            raise ValueError(
                "Evaluation CLI overrides changed the perception producer/control timing: "
                f"training fps/decimation={saved_timing}, evaluation={eval_timing}. "
                "A fresh export from the overridden environment would falsely re-authenticate "
                "student weights at a timestep they were not trained on."
            )

    if hasattr(saved_config.algo.config, "actor_obs_keys"):
        validate_fast_sac_actor_config_identity(
            saved_config.to_serializable_dict(),
            eval_config.to_serializable_dict(),
        )
        return

    saved_actor = getattr(getattr(saved_config.algo.config, "module_dict", None), "actor", None)
    eval_actor = getattr(getattr(eval_config.algo.config, "module_dict", None), "actor", None)
    if saved_actor != eval_actor:
        raise ValueError(
            "Evaluation CLI overrides changed the actor architecture/input contract. "
            "Actor module overrides are not safe when loading an existing checkpoint."
        )

    actor_groups = _actor_input_groups(saved_config)
    for group_name in actor_groups:
        saved_group = saved_config.observation.groups.get(group_name)
        eval_group = eval_config.observation.groups.get(group_name)
        if saved_group != eval_group:
            raise ValueError(
                f"Evaluation CLI overrides changed actor observation group {group_name!r}. "
                "Term order, history, scaling, and semantics must match the checkpoint."
            )

    perception_key = str(getattr(getattr(saved_actor, "layer_config", None), "perception_input_name", "") or "")
    if perception_key:
        if saved_config.observation.groups.get(perception_key) != eval_config.observation.groups.get(perception_key):
            raise ValueError(
                f"Evaluation CLI overrides changed actor perception group {perception_key!r}."
            )
        if saved_config.perception != eval_config.perception:
            raise ValueError(
                "Evaluation CLI overrides changed perception preprocessing for a perception policy. "
                "Camera geometry, crop/resize, normalization, and depth conventions must match the checkpoint."
            )
    for field_name in ("normalize_actor_obs", "obs_normalizer_eps", "obs_normalizer_until"):
        if getattr(saved_config.algo.config, field_name, None) != getattr(eval_config.algo.config, field_name, None):
            raise ValueError(f"Evaluation CLI override changed actor normalizer setting {field_name!r}.")

    if list(saved_config.robot.dof_names) != list(eval_config.robot.dof_names):
        raise ValueError("Evaluation robot DOF order does not match the checkpoint.")
    if saved_config.robot.init_state.default_joint_angles != eval_config.robot.init_state.default_joint_angles:
        raise ValueError("Evaluation robot default joint angles do not match the checkpoint.")
    if saved_config.robot.control != eval_config.robot.control:
        raise ValueError("Evaluation robot control/action scaling does not match the checkpoint.")
    if saved_config.action != eval_config.action:
        raise ValueError("Evaluation action-manager contract does not match the checkpoint.")


def _bind_training_perception_reference_batch(
    saved_config: ExperimentConfig,
    eval_config: ExperimentConfig,
) -> ExperimentConfig:
    """Preserve batch-normalized camera noise when eval uses fewer environments."""

    saved_perception = getattr(saved_config, "perception", None)
    eval_perception = getattr(eval_config, "perception", None)
    if saved_perception is None or eval_perception is None:
        return eval_config
    if not bool(getattr(saved_perception, "camera_warp_enable_holes", False)) or float(
        getattr(saved_perception, "camera_warp_hole_prob", 0.0) or 0.0
    ) <= 0.0:
        return eval_config

    reference_batch_size = getattr(
        saved_perception,
        "camera_warp_hole_reference_batch_size",
        None,
    )
    if reference_batch_size is None:
        reference_batch_size = getattr(saved_config.training, "num_envs", None)
    if (
        isinstance(reference_batch_size, bool)
        or not isinstance(reference_batch_size, int)
        or reference_batch_size < 1
    ):
        raise ValueError(
            "Saved camera-hole producer requires a positive integer training num_envs/reference batch, "
            f"got {reference_batch_size!r}."
        )
    if int(reference_batch_size) < int(eval_config.training.num_envs):
        raise ValueError(
            "Saved camera-hole reference batch is smaller than the requested evaluation batch: "
            f"{reference_batch_size} < {eval_config.training.num_envs}."
        )
    return dataclasses.replace(
        eval_config,
        perception=dataclasses.replace(
            eval_perception,
            camera_warp_hole_reference_batch_size=int(reference_batch_size),
        ),
    )


def _resolve_run_sim_robot(config: ExperimentConfig) -> str:
    from holosoma.config_values import robot as robot_values

    robot_type = config.robot.asset.robot_type
    matches = [key for key, cfg in robot_values.DEFAULTS.items() if cfg.asset.robot_type == robot_type]
    if not matches:
        raise ValueError(f"No run_sim robot preset matches robot_type '{robot_type}'")
    if len(matches) > 1:
        object_cfg = getattr(config.robot, "object", None)
        object_preset = f"{robot_type}_w_object"
        if bool(getattr(object_cfg, "enabled", False)) and object_preset in matches:
            return object_preset
        xml_file = config.robot.asset.xml_file
        xml_matches = [key for key in matches if robot_values.DEFAULTS[key].asset.xml_file == xml_file]
        if len(xml_matches) == 1:
            return xml_matches[0]
        if robot_type in matches:
            return robot_type
        raise ValueError(
            f"Multiple run_sim robot presets match robot_type '{robot_type}': {matches}. "
            "Pass --sim2sim.run-sim-robot to pick one."
        )
    return matches[0]


def _resolve_run_sim_terrain_args(config: ExperimentConfig) -> list[str]:
    terrain_term = config.terrain.terrain_term
    if terrain_term.mesh_type == MeshType.LOAD_OBJ and terrain_term.obj_file_path:
        return [
            "terrain:terrain_load_obj",
            "--terrain.terrain-term.obj-file-path",
            terrain_term.obj_file_path,
        ]
    if terrain_term.mesh_type == MeshType.PLANE:
        return ["terrain:terrain_locomotion_plane"]
    if terrain_term.mesh_type == MeshType.TRIMESH:
        return ["terrain:terrain_locomotion_mix"]
    return []


_PERCEPTION_MODEL_ONLY_FIELDS = frozenset(
    {
        "inject_into_policy_modules",
        "inject_into_critic_modules",
        "encoder_output_dim",
        "encoder_type",
        "encoder_fusion",
        "encoder_pretrained",
        "encoder_pretrained_path",
        "encoder_freeze_backbone",
        "encoder_target_size",
        "encoder_patch_size",
    }
)


def _perception_runtime_values(perception_cfg) -> dict[str, object]:
    """Return fields that determine the split-sim perception producer output."""

    if not dataclasses.is_dataclass(perception_cfg):
        raise ValueError("Checkpoint perception config is not a dataclass and cannot be deployed safely.")
    return {
        field.name: getattr(perception_cfg, field.name)
        for field in dataclasses.fields(perception_cfg)
        if field.name not in _PERCEPTION_MODEL_ONLY_FIELDS
    }


def _format_tyro_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple, dict)):
        # RunSimConfig enables UsePythonSyntaxForLiteralCollections.
        return repr(value)
    return str(value)


def _resolve_run_sim_perception_args(config: ExperimentConfig) -> list[str]:
    """Reconstruct the checkpoint's producer-side perception config for run_sim."""

    from holosoma.config_values import perception as perception_values

    perception_cfg = getattr(config, "perception", None)
    if perception_cfg is None or not bool(getattr(perception_cfg, "enabled", False)):
        raise ValueError(
            "The actor expects a perception input, but checkpoint perception is disabled. "
            "Refusing to build a sim2sim command with a missing producer."
        )

    if (
        bool(getattr(perception_cfg, "camera_warp_enable_holes", False))
        and float(getattr(perception_cfg, "camera_warp_hole_prob", 0.0) or 0.0) > 0.0
        and getattr(perception_cfg, "camera_warp_hole_reference_batch_size", None) is None
    ):
        training_cfg = getattr(config, "training", None)
        reference_batch_size = getattr(training_cfg, "num_envs", None)
        if (
            isinstance(reference_batch_size, bool)
            or not isinstance(reference_batch_size, int)
            or reference_batch_size < 1
        ):
            raise ValueError(
                "Camera-hole sim2sim reconstruction requires the checkpoint training num_envs, "
                f"got {reference_batch_size!r}."
            )
        perception_cfg = dataclasses.replace(
            perception_cfg,
            camera_warp_hole_reference_batch_size=int(reference_batch_size),
        )

    target_values = _perception_runtime_values(perception_cfg)
    output_mode = target_values.get("output_mode")
    candidates: list[tuple[int, str, object, dict[str, object]]] = []
    for preset_name, preset_cfg in perception_values.DEFAULTS.items():
        if not bool(getattr(preset_cfg, "enabled", False)):
            continue
        preset_values = _perception_runtime_values(preset_cfg)
        if preset_values.get("output_mode") != output_mode:
            continue
        difference_count = sum(preset_values.get(name) != value for name, value in target_values.items())
        candidates.append((difference_count, preset_name, preset_cfg, preset_values))

    if not candidates:
        raise ValueError(
            f"No run_sim perception preset can represent checkpoint output_mode={output_mode!r}. "
            "Add a deployment preset before using automatic sim2sim."
        )

    _, preset_name, _, preset_values = min(candidates, key=lambda item: (item[0], item[1]))
    args = [f"perception:{preset_name.replace('_', '-')}"]
    for field in dataclasses.fields(perception_cfg):
        name = field.name
        if name in _PERCEPTION_MODEL_ONLY_FIELDS:
            continue
        value = target_values[name]
        if preset_values.get(name) == value:
            continue
        args.extend([f"--perception.{name.replace('_', '-')}", _format_tyro_value(value)])
    return args


def _last_flag_override(args: list[str], option: str, default: bool) -> tuple[bool, bool]:
    """Resolve Tyro's --foo/--no-foo flags, returning (value, was_explicit)."""

    prefix, name = option.rsplit(".", 1)
    negative_option = f"{prefix}.no-{name}"
    value = bool(default)
    explicit = False
    for token in args:
        if token == option:
            value = True
            explicit = True
        elif token == negative_option:
            value = False
            explicit = True
    return value, explicit


def _last_option_value(args: list[str], option: str, default: object) -> tuple[object, bool]:
    value = default
    explicit = False
    for index, token in enumerate(args):
        if token == option:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"Missing value for sim2sim option {option!r}.")
            value = args[index + 1]
            explicit = True
        elif token.startswith(option + "="):
            value = token.split("=", 1)[1]
            explicit = True
    return value, explicit


def _parse_explicit_bool(value: object, *, option: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {option}: {value!r}.")


def _resolve_policy_perception_transport(
    inference_cfg: str, policy_extra_args: list[str]
) -> tuple[bool, int, str]:
    from holosoma_inference.config.config_values import inference as inference_values

    preset_name = inference_cfg.removeprefix("inference:")
    inference_preset = inference_values.DEFAULTS.get(preset_name)
    if inference_preset is None:
        raise ValueError(
            f"Unknown inference preset {inference_cfg!r}; cannot prove the split perception transport contract."
        )

    task_cfg = inference_preset.task
    use_split, split_explicit = _last_flag_override(
        policy_extra_args,
        "--task.use-split-perception-obs",
        bool(getattr(task_cfg, "use_split_perception_obs", False)),
    )
    if not use_split:
        detail = "explicitly disabled" if split_explicit else "disabled by the inference preset"
        raise ValueError(
            f"The checkpoint actor expects perception, but split perception subscription is {detail}. "
            "Select a perception inference preset or remove --task.no-use-split-perception-obs."
        )

    use_shm, _ = _last_flag_override(
        policy_extra_args,
        "--task.use-split-perception-obs-shm",
        bool(getattr(task_cfg, "use_split_perception_obs_shm", False)),
    )
    raw_port, _ = _last_option_value(
        policy_extra_args,
        "--task.perception-obs-port",
        int(getattr(task_cfg, "perception_obs_port", 5558)),
    )
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split perception ZMQ port: {raw_port!r}.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Split perception ZMQ port must be in [1, 65535], got {port}.")

    raw_shm_name, _ = _last_option_value(
        policy_extra_args,
        "--task.perception-obs-shm-name",
        str(getattr(task_cfg, "perception_obs_shm_name", "depth_img_shm")),
    )
    shm_name = str(raw_shm_name).strip()
    if use_shm and not shm_name:
        raise ValueError("Split perception shared-memory transport requires a non-empty shm name.")
    return use_shm, port, shm_name


def _validate_run_sim_perception_extras(
    run_sim_extra_args: list[str],
    *,
    use_shm: bool,
    perception_port: int,
    perception_shm_name: str,
    requires_cuda: bool,
) -> None:
    protected_prefixes = (
        "perception:",
        "--perception.",
        "--perception-randomization.",
        "--perception-producer-tick-dt",
        "--perception-allow-mujoco-noise",
        "--perception-contract-envelope-b64",
        "--training.seed",
    )
    if any(token.startswith(protected_prefixes) for token in run_sim_extra_args):
        raise ValueError(
            "--sim2sim.run-sim-args may not override perception for a perception checkpoint. "
            "Producer randomization, cadence, noise, and seed are reconstructed from the checkpoint "
            "and must remain unchanged."
        )

    expected_bools = {
        "--simulator.config.bridge.publish-perception-obs": not use_shm,
        "--simulator.config.bridge.publish-perception-obs-shm": use_shm,
    }
    for option, expected in expected_bools.items():
        raw_value, explicit = _last_option_value(run_sim_extra_args, option, expected)
        if explicit and _parse_explicit_bool(raw_value, option=option) != expected:
            raise ValueError(
                f"{option} conflicts with the policy perception transport; expected {expected}."
            )

    raw_port, port_explicit = _last_option_value(
        run_sim_extra_args, "--simulator.config.bridge.perception-obs-port", perception_port
    )
    if port_explicit:
        try:
            run_sim_port = int(raw_port)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run_sim perception port: {raw_port!r}.") from exc
        if run_sim_port != perception_port:
            raise ValueError(
                "run_sim and run_policy perception ZMQ ports differ: "
                f"{run_sim_port} != {perception_port}."
            )

    raw_shm_name, shm_explicit = _last_option_value(
        run_sim_extra_args,
        "--simulator.config.bridge.perception-obs-shm-name",
        perception_shm_name,
    )
    if shm_explicit and str(raw_shm_name) != perception_shm_name:
        raise ValueError(
            "run_sim and run_policy perception shared-memory names differ: "
            f"{raw_shm_name!r} != {perception_shm_name!r}."
        )

    raw_device, device_explicit = _last_option_value(run_sim_extra_args, "--device", "cuda:0")
    if requires_cuda and device_explicit and not str(raw_device).strip().lower().startswith("cuda"):
        raise ValueError(
            "far_tracking_warp perception requires a CUDA run_sim device; refusing the silent rendered fallback."
        )


def _resolve_direct_sim_perception_randomization_args(config: ExperimentConfig) -> list[str]:
    """Reconstruct the one canonical camera reset distribution from the checkpoint."""

    perception_cfg = getattr(config, "perception", None)
    if not bool(getattr(perception_cfg, "enabled", False)):
        return []
    sensor_noise_enabled = bool(getattr(perception_cfg, "camera_apply_sensor_noise", False))
    randomization_cfg = getattr(config, "randomization", None)
    reset_terms = getattr(randomization_cfg, "reset_terms", None)
    if reset_terms is None:
        reset_terms = {}
    if not isinstance(reset_terms, dict):
        raise ValueError("Checkpoint randomization.reset_terms must be a mapping.")

    canonical = "holosoma.managers.randomization.terms.locomotion.randomize_camera_raycast"
    allowed_params = {
        "enabled",
        "translation_range",
        "rotation_range_deg",
        "noise_std_mult_range",
        "noise_drop_prob_range",
    }
    enabled_params: list[dict[str, object]] = []
    for term in reset_terms.values():
        raw_func = str(_attr_or_key(term, "func", "") or "")
        normalized_func = raw_func.replace(":", ".")
        if normalized_func.rsplit(".", maxsplit=1)[-1] != "randomize_camera_raycast":
            continue
        if normalized_func != canonical:
            raise ValueError(
                "Checkpoint camera reset randomization must use the canonical "
                f"randomize_camera_raycast implementation, got {raw_func!r}."
            )
        raw_params = _attr_or_key(term, "params", {}) or {}
        if not isinstance(raw_params, dict):
            raise ValueError("Checkpoint camera reset randomization params must be a mapping.")
        unexpected = sorted(set(raw_params) - allowed_params)
        if unexpected:
            raise ValueError(
                "Checkpoint camera reset randomization contains unauthenticated parameters: "
                f"{unexpected}."
            )
        enabled = raw_params.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Checkpoint camera reset randomization enabled must be boolean, got {enabled!r}.")
        if enabled:
            enabled_params.append(dict(raw_params))

    if len(enabled_params) > 1:
        raise ValueError(
            "Checkpoint must contain at most one enabled canonical randomize_camera_raycast term, "
            f"found {len(enabled_params)}."
        )
    if not enabled_params:
        if sensor_noise_enabled:
            raise ValueError(
                "Checkpoint enables camera_apply_sensor_noise, but no enabled canonical camera randomizer "
                "defines its multiplicative-noise/dropout distribution."
            )
        return ["--perception-randomization.enabled", "False"]

    params = enabled_params[0]

    def finite(value: object, *, path: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"Checkpoint camera randomization {path} must be numeric, got boolean {value!r}.")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Checkpoint camera randomization {path} must be a finite real number, got {value!r}."
            ) from exc
        if not math.isfinite(result):
            raise ValueError(f"Checkpoint camera randomization {path} must be finite.")
        return result

    def scalar_range(
        spec: object,
        *,
        path: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> list[float] | None:
        if spec is None:
            return None
        if isinstance(spec, (list, tuple)):
            if len(spec) != 2:
                raise ValueError(f"Checkpoint camera randomization {path} must contain exactly [low, high].")
            low = finite(spec[0], path=f"{path}[0]")
            high = finite(spec[1], path=f"{path}[1]")
        else:
            low = high = finite(spec, path=path)
        if low > high:
            raise ValueError(
                f"Checkpoint camera randomization {path} lower bound {low} exceeds upper bound {high}."
            )
        if minimum is not None and low < minimum:
            raise ValueError(f"Checkpoint camera randomization {path} must be >= {minimum}.")
        if maximum is not None and high > maximum:
            raise ValueError(f"Checkpoint camera randomization {path} must be <= {maximum}.")
        return [low, high]

    def vector_range(
        spec: object,
        *,
        path: str,
        keys: tuple[str, str, str],
    ) -> dict[str, list[float]] | None:
        if spec is None:
            return None
        if isinstance(spec, dict):
            if set(spec) != set(keys):
                raise ValueError(
                    f"Checkpoint camera randomization {path} must declare exactly {list(keys)}, "
                    f"got {sorted(str(key) for key in spec)}."
                )
            return {
                key: scalar_range(spec[key], path=f"{path}.{key}")
                for key in keys
            }
        shared = scalar_range(spec, path=path)
        return {key: list(shared) for key in keys}

    translation = vector_range(
        params.get("translation_range"),
        path="translation_range",
        keys=("x", "y", "z"),
    )
    rotation = vector_range(
        params.get("rotation_range_deg"),
        path="rotation_range_deg",
        keys=("roll", "pitch", "yaw"),
    )
    std_range = scalar_range(
        params.get("noise_std_mult_range"),
        path="noise_std_mult_range",
        minimum=0.0,
    )
    drop_range = scalar_range(
        params.get("noise_drop_prob_range"),
        path="noise_drop_prob_range",
        minimum=0.0,
        maximum=1.0,
    )
    if sensor_noise_enabled and std_range is None and drop_range is None:
        raise ValueError(
            "Checkpoint enables camera_apply_sensor_noise, but its canonical camera randomizer declares "
            "neither noise_std_mult_range nor noise_drop_prob_range."
        )

    args = ["--perception-randomization.enabled", "True"]
    for option, value in (
        ("--perception-randomization.translation-range", translation),
        ("--perception-randomization.rotation-range-deg", rotation),
        ("--perception-randomization.noise-std-mult-range", std_range),
        ("--perception-randomization.noise-drop-prob-range", drop_range),
    ):
        if value is not None:
            args.extend([option, json.dumps(value, sort_keys=True, separators=(",", ":"))])
    return args


def _attr_or_key(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


_DIRECT_PERCEPTION_NOISE_FIELDS = (
    "edge_noise",
    "enable_holes",
    "hole_prob",
    "additive_noise_std",
    "depth_offset_std",
    "apply_sensor_noise",
)


def _decode_authenticated_perception_contract_envelope(envelope_b64: str) -> Mapping[str, object]:
    """Decode the preflight-authenticated contract without accepting alternate encodings."""

    if not isinstance(envelope_b64, str) or not envelope_b64:
        raise ValueError(
            "Direct perception command construction requires a non-empty authenticated ONNX "
            "observation-contract envelope."
        )
    try:
        encoded = envelope_b64.encode("ascii")
        payload = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ValueError(
            "Perception observation-contract envelope must be canonical strict base64."
        ) from exc
    if base64.b64encode(payload) != encoded:
        raise ValueError(
            "Perception observation-contract envelope must use canonical padded base64."
        )

    def reject_nonfinite_json(constant: str):
        raise ValueError(f"non-finite JSON constant {constant!r}")

    def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_nonfinite_json,
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Perception observation-contract envelope must contain strict finite UTF-8 JSON."
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != {"contract", "sha256"}:
        raise ValueError(
            "Perception observation-contract envelope must contain exactly 'contract' and 'sha256'."
        )
    canonical_payload = json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if payload != canonical_payload:
        raise ValueError(
            "Perception observation-contract envelope JSON is not in its canonical encoding."
        )

    from holosoma_inference.utils.policy_contract import (  # noqa: PLC0415
        perception_observation_contract_sha256_from_metadata,
    )

    digest = perception_observation_contract_sha256_from_metadata(
        {
            "perception_observation_contract": decoded["contract"],
            "perception_observation_contract_sha256": decoded["sha256"],
        }
    )
    if digest is None:
        raise ValueError("Perception observation-contract envelope did not authenticate a contract.")
    contract = decoded["contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("Authenticated perception observation contract must be a mapping.")
    return contract


def _configured_perception_noise_profile(config: ExperimentConfig) -> dict[str, object]:
    """Return the noise values requested by the saved perception configuration."""

    perception_cfg = getattr(config, "perception", None)
    if perception_cfg is None:
        raise ValueError("Checkpoint is missing its saved perception configuration.")

    def required_bool(attribute: str) -> bool:
        value = getattr(perception_cfg, attribute, None)
        if not isinstance(value, bool):
            raise ValueError(
                f"Checkpoint perception.{attribute} must be boolean, got {value!r}."
            )
        return value

    def required_nonnegative_float(attribute: str, *, maximum: float | None = None) -> float:
        value = getattr(perception_cfg, attribute, None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (maximum is not None and float(value) > maximum)
        ):
            range_description = "[0, 1]" if maximum == 1.0 else "finite and non-negative"
            raise ValueError(
                f"Checkpoint perception.{attribute} must be {range_description}, got {value!r}."
            )
        return float(value)

    return {
        "edge_noise": required_bool("camera_warp_edge_noise"),
        "enable_holes": required_bool("camera_warp_enable_holes"),
        "hole_prob": required_nonnegative_float("camera_warp_hole_prob", maximum=1.0),
        "additive_noise_std": required_nonnegative_float("camera_warp_additive_noise_std"),
        "depth_offset_std": required_nonnegative_float("camera_warp_depth_offset_std"),
        "apply_sensor_noise": required_bool("camera_apply_sensor_noise"),
    }


def _contract_effective_perception_noise_profile(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Read the effective (post-backend-gating) noise values from an ONNX contract."""

    schema = contract.get("effective_observation_schema")
    if not isinstance(schema, Mapping):
        raise ValueError(
            "Authenticated ONNX perception contract is missing effective_observation_schema."
        )
    missing = sorted(set(_DIRECT_PERCEPTION_NOISE_FIELDS) - set(schema))
    if missing:
        raise ValueError(
            "Authenticated ONNX effective observation schema is missing noise fields: "
            f"{missing}."
        )

    profile: dict[str, object] = {}
    for field_name in ("edge_noise", "enable_holes", "apply_sensor_noise"):
        value = schema[field_name]
        if not isinstance(value, bool):
            raise ValueError(
                "Authenticated ONNX effective observation schema field "
                f"{field_name!r} must be boolean, got {value!r}."
            )
        profile[field_name] = value
    for field_name, maximum in (
        ("hole_prob", 1.0),
        ("additive_noise_std", None),
        ("depth_offset_std", None),
    ):
        value = schema[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or (maximum is not None and float(value) > maximum)
        ):
            range_description = "[0, 1]" if maximum == 1.0 else "finite and non-negative"
            raise ValueError(
                "Authenticated ONNX effective observation schema field "
                f"{field_name!r} must be {range_description}, got {value!r}."
            )
        profile[field_name] = float(value)
    return profile


def _resolve_direct_mujoco_noise_permission(
    config: ExperimentConfig,
    contract: Mapping[str, object],
) -> bool:
    """Reproduce the saved backend's effective noise semantics in direct MuJoCo."""

    backend = configured_simulator_type(config.simulator)
    requested = _configured_perception_noise_profile(config)
    effective = _contract_effective_perception_noise_profile(contract)
    forced_off: dict[str, object] = {
        "edge_noise": False,
        "enable_holes": False,
        "hole_prob": 0.0,
        "additive_noise_std": 0.0,
        "depth_offset_std": 0.0,
        "apply_sensor_noise": False,
    }

    if backend in {"isaacgym", "isaacsim"}:
        if effective != requested:
            raise ValueError(
                "Authenticated ONNX effective perception noise does not match the saved Isaac "
                "training configuration; refusing to guess a direct MuJoCo producer mode."
            )
        # Isaac does not apply MuJoCo's default noise suppression.  Direct
        # MuJoCo must therefore allow the requested pipeline before the live
        # manager authenticates the complete effective observation contract.
        return True

    if backend != "mujoco":
        raise ValueError(f"Unsupported saved simulator backend for direct perception: {backend!r}.")
    if effective == forced_off:
        # This also covers a configuration whose requested profile is already
        # all-off: False is the canonical MuJoCo effective-off reconstruction.
        return False
    if effective == requested:
        # A noisy effective contract proves that MuJoCo training explicitly
        # opted out of its default suppression; reproduce that opt-in.
        return True
    raise ValueError(
        "Authenticated ONNX effective perception noise matches neither the saved MuJoCo "
        "configuration nor MuJoCo's canonical effective-off profile; refusing to guess."
    )


def _resolve_direct_sim_perception_distribution_args(config: ExperimentConfig) -> list[str]:
    perception_cfg = getattr(config, "perception", None)
    reset_refresh_semantics = str(
        getattr(perception_cfg, "reset_refresh_semantics", "legacy_full_v1")
    )
    if reset_refresh_semantics != "targeted_v2":
        raise ValueError(
            "Authenticated one-environment direct perception requires "
            "reset_refresh_semantics='targeted_v2'. "
            f"This checkpoint records {reset_refresh_semantics!r}: its historical producer advanced the full "
            "vectorized perception stream whenever any peer environment reset, so its temporal noise/cadence "
            "cannot be represented by direct RunSim. Retrain or policy-initialize a new student with the "
            "versioned targeted_v2 producer; do not relabel legacy weights as direct-compatible."
        )

    sim_cfg = getattr(getattr(config, "simulator", None), "config", None)
    sim_engine = getattr(sim_cfg, "sim", None)
    fps = getattr(sim_engine, "fps", None)
    decimation = getattr(sim_engine, "control_decimation", None)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        raise ValueError(f"Checkpoint simulator fps must be finite and positive, got {fps!r}.")
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise ValueError(
            f"Checkpoint simulator control_decimation must be a positive integer, got {decimation!r}."
        )
    producer_tick_dt = int(decimation) / float(fps)

    training_cfg = getattr(config, "training", None)
    seed = getattr(training_cfg, "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            "Checkpoint distribution-initialization seed must be a non-negative integer, "
            f"got {seed!r}."
        )

    return [
        "--perception-producer-tick-dt",
        repr(producer_tick_dt),
        "--training.seed",
        str(seed),
        *_resolve_direct_sim_perception_randomization_args(config),
    ]


def _resolve_direct_sim_perception_producer_args(
    config: ExperimentConfig,
    *,
    perception_contract_envelope_b64: str | None,
) -> list[str]:
    distribution_args = _resolve_direct_sim_perception_distribution_args(config)
    if perception_contract_envelope_b64 is None:
        raise ValueError(
            "Direct perception command construction requires the authenticated ONNX effective "
            "observation-contract envelope; the MuJoCo noise mode cannot be guessed from config."
        )
    contract = _decode_authenticated_perception_contract_envelope(
        perception_contract_envelope_b64
    )
    allow_mujoco_noise = _resolve_direct_mujoco_noise_permission(config, contract)
    return [
        *distribution_args[:2],
        "--perception-allow-mujoco-noise",
        str(allow_mujoco_noise),
        *distribution_args[2:],
    ]


def _validate_direct_perception_distribution_source(
    saved_config: ExperimentConfig,
    eval_config: ExperimentConfig,
) -> None:
    """Bind direct stochastic distributions and lifecycle to checkpoint provenance."""

    if not _actor_perception_input_name(saved_config):
        return
    saved_backend = configured_simulator_type(saved_config.simulator)
    eval_backend = configured_simulator_type(eval_config.simulator)
    if saved_backend != eval_backend:
        raise ValueError(
            "Evaluation CLI overrides changed the saved simulator backend used to derive the "
            "effective perception-noise contract: "
            f"{saved_backend!r} -> {eval_backend!r}."
        )
    saved_args = _resolve_direct_sim_perception_distribution_args(saved_config)
    eval_args = _resolve_direct_sim_perception_distribution_args(eval_config)
    if saved_args != eval_args:
        raise ValueError(
            "Evaluation CLI overrides changed the authenticated direct perception producer. "
            "Distribution-initialization seed, camera pose distribution, multiplicative depth noise, dropout, "
            "producer cadence, and reset lifecycle must come from the saved checkpoint config."
        )


def _checkpoint_motion_source(config: ExperimentConfig) -> str | None:
    command_cfg = getattr(config, "command", None)
    setup_terms = getattr(command_cfg, "setup_terms", None)
    if not isinstance(setup_terms, dict):
        return None

    motion_sources: set[str] = set()
    for term in setup_terms.values():
        func = str(_attr_or_key(term, "func", "") or "").lower()
        if "motioncommand" not in func and "motion_command" not in func:
            continue
        params = _attr_or_key(term, "params", {})
        motion_cfg = _attr_or_key(params, "motion_config", None)
        motion_file = str(_attr_or_key(motion_cfg, "motion_file", "") or "").strip()
        if motion_file:
            motion_sources.add(motion_file)

    if len(motion_sources) > 1:
        raise ValueError(
            "Checkpoint contains multiple motion sources; automatic sim2sim cannot prove which one drives the actor."
        )
    return next(iter(motion_sources), None)


def _existing_local_path(raw_path: object, *, label: str, directory: bool = False) -> Path:
    raw = str(raw_path or "").strip()
    if not raw or "://" in raw:
        raise ValueError(f"{label} must be an existing local path, got {raw!r}.")
    path = Path(raw).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise ValueError(f"{label} is not an existing {expected}: {path}")
    return path


def _lexical_absolute_path(raw_path: object, *, label: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw or "://" in raw:
        raise ValueError(f"{label} must be an existing local path, got {raw!r}.")
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    if not path.is_file():
        raise ValueError(f"{label} is not an existing file: {path}")
    return path


def _mapped_object_urdf(object_map: Path, motion_file: Path) -> Path:
    try:
        payload = json.loads(object_map.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read checkpoint clip-object map {object_map}: {exc}") from exc
    clips = payload.get("clips", payload) if isinstance(payload, dict) else None
    if not isinstance(clips, dict):
        raise ValueError(f"Checkpoint clip-object map must contain a 'clips' mapping: {object_map}")
    entry = clips.get(motion_file.stem)
    if not isinstance(entry, dict):
        raise ValueError(
            f"Checkpoint clip-object map has no entry for selected motion {motion_file.stem!r}: {object_map}"
        )
    raw_urdf = str(entry.get("object_urdf_path") or entry.get("urdf_path") or "").strip()
    if not raw_urdf:
        raise ValueError(
            f"Checkpoint clip-object map entry {motion_file.stem!r} has no object_urdf_path: {object_map}"
        )
    urdf = Path(raw_urdf).expanduser()
    if not urdf.is_absolute():
        urdf = object_map.parent / urdf
    urdf = urdf.resolve()
    if not urdf.is_file():
        raise ValueError(
            f"Mapped object URDF for motion {motion_file.stem!r} does not exist: {urdf}"
        )
    return urdf


def _object_motion_selection_error(checkpoint_motion: str | None, checkpoint_object: str) -> ValueError:
    return ValueError(
        "Object-enabled checkpoint uses a motion/object bank, so automatic sim2sim cannot choose a scientific "
        "clip-object pair. Provide the same concrete motion file to both processes and its mapped concrete URDF: "
        "--sim2sim.run-sim-args \"--motion-init.motion-file /path/clip.npz "
        "--robot.object.object-urdf-path /path/object.urdf\" and "
        "--sim2sim.run-policy-args \"--task.motion-file /path/clip.npz\". "
        f"Checkpoint motion source={checkpoint_motion!r}, object source={checkpoint_object!r}."
    )


def _resolve_object_motion_sim2sim_args(
    config: ExperimentConfig,
    run_sim_extra_args: list[str],
    policy_extra_args: list[str],
) -> tuple[list[str], list[str]]:
    object_cfg = getattr(config.robot, "object", None)
    if not bool(getattr(object_cfg, "enabled", False)):
        return [], []

    checkpoint_object_raw = str(getattr(object_cfg, "object_urdf_path", "") or "").strip()
    if not checkpoint_object_raw:
        raise ValueError("Object-enabled checkpoint has no robot.object.object_urdf_path.")
    checkpoint_motion_raw = _checkpoint_motion_source(config)
    if _is_wbt_experiment(config) and not checkpoint_motion_raw:
        raise ValueError(
            "Object-enabled WBT checkpoint has no unique MotionCommand motion_file; "
            "automatic sim2sim cannot initialize policy and simulator from the same motion."
        )

    sim_motion_raw, sim_motion_explicit = _last_option_value(
        run_sim_extra_args, "--motion-init.motion-file", ""
    )
    policy_motion_raw, policy_motion_explicit = _last_option_value(
        policy_extra_args, "--task.motion-file", ""
    )
    object_urdf_raw, object_urdf_explicit = _last_option_value(
        run_sim_extra_args, "--robot.object.object-urdf-path", ""
    )

    checkpoint_motion_path: Path | None = None
    motion_bank: Path | None = None
    if checkpoint_motion_raw:
        unresolved_motion = Path(checkpoint_motion_raw).expanduser().resolve()
        if unresolved_motion.is_dir():
            motion_bank = unresolved_motion
        elif unresolved_motion.is_file():
            checkpoint_motion_path = unresolved_motion
        else:
            raise ValueError(f"Checkpoint motion source does not exist locally: {unresolved_motion}")

    checkpoint_object_path = _existing_local_path(
        checkpoint_object_raw, label="Checkpoint object URDF/map"
    )
    object_map = checkpoint_object_path if checkpoint_object_path.suffix.lower() == ".json" else None
    is_bank_contract = motion_bank is not None or object_map is not None

    if is_bank_contract and not (sim_motion_explicit and policy_motion_explicit and object_urdf_explicit):
        raise _object_motion_selection_error(checkpoint_motion_raw, checkpoint_object_raw)

    if sim_motion_explicit:
        sim_motion_lexical = _lexical_absolute_path(sim_motion_raw, label="run_sim motion file")
        sim_motion = _existing_local_path(sim_motion_raw, label="run_sim motion file")
    elif checkpoint_motion_path is not None:
        sim_motion_lexical = checkpoint_motion_path
        sim_motion = checkpoint_motion_path
    else:
        raise _object_motion_selection_error(checkpoint_motion_raw, checkpoint_object_raw)

    if policy_motion_explicit:
        policy_motion_lexical = _lexical_absolute_path(policy_motion_raw, label="run_policy motion file")
        policy_motion = _existing_local_path(policy_motion_raw, label="run_policy motion file")
    elif checkpoint_motion_path is not None:
        policy_motion_lexical = checkpoint_motion_path
        policy_motion = checkpoint_motion_path
    else:
        raise _object_motion_selection_error(checkpoint_motion_raw, checkpoint_object_raw)

    if sim_motion != policy_motion:
        raise ValueError(
            "run_sim and run_policy must use the same concrete motion file: "
            f"{sim_motion} != {policy_motion}."
        )
    selected_motion = sim_motion
    selected_motion_argument = selected_motion

    if motion_bank is not None:
        if sim_motion_lexical != policy_motion_lexical:
            raise ValueError(
                "run_sim and run_policy must select the same checkpoint-bank entry, not merely symlinks to the "
                f"same target: {sim_motion_lexical} != {policy_motion_lexical}."
            )
        try:
            sim_motion_lexical.relative_to(motion_bank)
        except ValueError as exc:
            raise ValueError(
                f"Selected motion {sim_motion_lexical} is outside checkpoint motion bank {motion_bank}."
            ) from exc
        matching_bank_entries = [
            Path(os.path.abspath(str(candidate)))
            for candidate in motion_bank.iterdir()
            if candidate.is_file() and candidate.stem == sim_motion_lexical.stem
        ]
        if len(matching_bank_entries) != 1:
            raise ValueError(
                "Checkpoint motion bank must contain exactly one top-level entry for selected clip "
                f"{sim_motion_lexical.stem!r}, found {len(matching_bank_entries)} in {motion_bank}."
            )
        canonical_bank_entry = matching_bank_entries[0]
        if sim_motion_lexical != canonical_bank_entry or canonical_bank_entry.resolve() != selected_motion:
            raise ValueError(
                "Selected motion must be the unique checkpoint-bank entry and resolve to its exact target: "
                f"selected={sim_motion_lexical} target={selected_motion}, "
                f"entry={canonical_bank_entry} entry_target={canonical_bank_entry.resolve()}."
            )
        selected_motion_argument = sim_motion_lexical
    elif checkpoint_motion_path is not None and selected_motion != checkpoint_motion_path:
        raise ValueError(
            "Selected motion does not match the checkpoint's single motion contract: "
            f"{selected_motion} != {checkpoint_motion_path}."
        )

    expected_urdf = (
        _mapped_object_urdf(object_map, selected_motion_argument)
        if object_map is not None
        else checkpoint_object_path
    )
    if object_urdf_explicit:
        selected_urdf = _existing_local_path(object_urdf_raw, label="run_sim object URDF")
        if selected_urdf.suffix.lower() == ".json":
            raise ValueError(
                "run_sim requires a concrete object URDF, not a clip-object map; "
                f"selected {selected_urdf}."
            )
    else:
        selected_urdf = expected_urdf
    if selected_urdf != expected_urdf:
        raise ValueError(
            "Selected object URDF does not match the checkpoint clip-object mapping for "
            f"{selected_motion_argument.stem!r}: {selected_urdf} != {expected_urdf}."
        )

    for option in ("--motion-init.enabled", "--robot.object.enabled"):
        raw_enabled, explicit = _last_option_value(run_sim_extra_args, option, True)
        if explicit and not _parse_explicit_bool(raw_enabled, option=option):
            raise ValueError(f"{option} cannot be disabled for an object-enabled WBT checkpoint.")

    return (
        [
            "--motion-init.enabled",
            "True",
            "--motion-init.motion-file",
            str(selected_motion_argument),
            "--motion-init.object-name",
            "object",
            "--robot.object.enabled",
            "True",
            "--robot.object.object-urdf-path",
            str(selected_urdf),
        ],
        ["--task.motion-file", str(selected_motion_argument)],
    )


def _build_sim2sim_commands(
    config: ExperimentConfig,
    sim2sim_cfg: Sim2SimConfig,
    model_path: str,
    *,
    perception_contract_envelope_b64: str | None = None,
) -> tuple[list[str], list[str]]:
    inference_cfg, is_wbt = _infer_inference_config(config)
    if sim2sim_cfg.inference_config:
        inference_cfg = sim2sim_cfg.inference_config

    use_sim_time = sim2sim_cfg.use_sim_time if sim2sim_cfg.use_sim_time is not None else is_wbt
    default_rl_rate = config.simulator.config.sim.fps / config.simulator.config.sim.control_decimation
    rl_rate = sim2sim_cfg.rl_rate if sim2sim_cfg.rl_rate is not None else default_rl_rate

    run_sim_robot = sim2sim_cfg.run_sim_robot or _resolve_run_sim_robot(config)
    if run_sim_robot.startswith("robot:"):
        run_sim_robot = run_sim_robot.split(":", 1)[1]

    actor_perception_input = _actor_perception_input_name(config)
    run_sim_extra_args = shlex.split(sim2sim_cfg.run_sim_args) if sim2sim_cfg.run_sim_args else []
    policy_extra_args = shlex.split(sim2sim_cfg.run_policy_args) if sim2sim_cfg.run_policy_args else []
    object_run_sim_args, object_policy_args = _resolve_object_motion_sim2sim_args(
        config, run_sim_extra_args, policy_extra_args
    )

    run_sim_cmd = [
        sim2sim_cfg.sim_python or sys.executable,
        "src/holosoma/holosoma/run_sim.py",
        f"simulator:{sim2sim_cfg.simulator}",
        f"robot:{run_sim_robot}",
    ]
    run_sim_cmd += _resolve_run_sim_terrain_args(config)
    if sim2sim_cfg.interface:
        run_sim_cmd += ["--simulator.config.bridge.interface", sim2sim_cfg.interface]
    if sim2sim_cfg.use_joystick:
        run_sim_cmd += ["--simulator.config.bridge.use-joystick", "True"]
    run_sim_cmd += object_run_sim_args

    perception_transport: tuple[bool, int, str] | None = None
    if actor_perception_input:
        perception_transport = _resolve_policy_perception_transport(inference_cfg, policy_extra_args)
        use_shm, perception_port, perception_shm_name = perception_transport
        perception_cfg = getattr(config, "perception", None)
        requires_cuda = str(getattr(perception_cfg, "camera_source", "")) == "far_tracking_warp"
        _validate_run_sim_perception_extras(
            run_sim_extra_args,
            use_shm=use_shm,
            perception_port=perception_port,
            perception_shm_name=perception_shm_name,
            requires_cuda=requires_cuda,
        )
        run_sim_cmd += _resolve_run_sim_perception_args(config)
        run_sim_cmd += _resolve_direct_sim_perception_producer_args(
            config,
            perception_contract_envelope_b64=perception_contract_envelope_b64,
        )
        if perception_contract_envelope_b64 is None:  # Defensive type narrowing; resolver rejects first.
            raise RuntimeError("Authenticated perception contract envelope was unexpectedly lost.")
        run_sim_cmd += [
            "--perception-contract-envelope-b64",
            perception_contract_envelope_b64,
        ]
        run_sim_cmd += [
            "--simulator.config.bridge.publish-perception-obs",
            str(not use_shm),
            "--simulator.config.bridge.perception-obs-port",
            str(perception_port),
            "--simulator.config.bridge.publish-perception-obs-shm",
            str(use_shm),
            "--simulator.config.bridge.perception-obs-shm-name",
            perception_shm_name,
        ]
        if requires_cuda:
            run_sim_cmd += ["--device", "cuda:0"]

    run_sim_cmd += run_sim_extra_args

    run_policy_cmd = [
        sim2sim_cfg.policy_python or sys.executable,
        "src/holosoma_inference/holosoma_inference/run_policy.py",
        inference_cfg,
        "--task.model-path",
        model_path,
        "--task.interface",
        sim2sim_cfg.interface,
        "--task.rl-rate",
        str(rl_rate),
    ]
    if use_sim_time:
        run_policy_cmd.append("--task.use-sim-time")
    if sim2sim_cfg.use_joystick:
        run_policy_cmd.append("--task.use-joystick")
    run_policy_cmd += object_policy_args
    if perception_transport is not None:
        use_shm, perception_port, perception_shm_name = perception_transport
        run_policy_cmd += [
            "--task.use-split-perception-obs",
            "--task.perception-obs-port",
            str(perception_port),
        ]
        if use_shm:
            run_policy_cmd += [
                "--task.use-split-perception-obs-shm",
                "--task.perception-obs-shm-name",
                perception_shm_name,
            ]
        else:
            run_policy_cmd.append("--task.no-use-split-perception-obs-shm")
    run_policy_cmd += policy_extra_args

    return run_sim_cmd, run_policy_cmd


def _launch_sim2sim(run_sim_cmd: list[str], run_policy_cmd: list[str]) -> None:
    logger.info("Launching MuJoCo sim-to-sim processes...")
    run_sim_proc = subprocess.Popen(run_sim_cmd)
    run_policy_proc = None

    def stop_process(process) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    try:
        run_policy_proc = subprocess.Popen(run_policy_cmd)
        while True:
            run_sim_status = run_sim_proc.poll()
            run_policy_status = run_policy_proc.poll()

            if run_sim_status is not None:
                if run_sim_status != 0:
                    raise subprocess.CalledProcessError(run_sim_status, run_sim_cmd)
                if run_policy_status is None:
                    raise RuntimeError(
                        "run_sim exited before run_policy completed; the perception/state producer "
                        "is no longer available."
                    )
                if run_policy_status != 0:
                    raise subprocess.CalledProcessError(run_policy_status, run_policy_cmd)
                return

            if run_policy_status is not None:
                if run_policy_status != 0:
                    raise subprocess.CalledProcessError(run_policy_status, run_policy_cmd)
                return

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Sim-to-sim interrupted by user.")
    finally:
        stop_process(run_policy_proc)
        stop_process(run_sim_proc)


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    sim2sim_cfg: Sim2SimConfig,
):
    set_defm_checkpoint_restore_mode()
    tyro_config = apply_observation_overrides(tyro_config)
    tyro_config = apply_perception_overrides(tyro_config)
    _validate_eval_policy_contract(saved_config, tyro_config)
    tyro_config = _bind_training_perception_reference_batch(saved_config, tyro_config)

    # Reject legacy/vector-coupled lifecycle and provenance drift before
    # creating eval directories, loading checkpoints, initializing CUDA/Warp,
    # or exporting an ONNX model that would incorrectly certify the override.
    if sim2sim_cfg.enabled:
        _validate_direct_perception_distribution_source(saved_config, tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)
    export_dir, exported_onnx_path = _get_export_paths(checkpoint_path)

    if sim2sim_cfg.enabled:
        actor_uses_perception = bool(_actor_perception_input_name(tyro_config))
        checkpoint_sha256 = None
        if actor_uses_perception:
            checkpoint_sha256 = hashlib.sha256(
                _read_stable_regular_file(
                    Path(checkpoint_path),
                    label="evaluation checkpoint",
                )
            ).hexdigest()

        explicit_model_path = sim2sim_cfg.model_path is not None
        model_path = sim2sim_cfg.model_path
        perception_contract_envelope_b64: str | None = None
        if model_path is None and exported_onnx_path.exists():
            model_path = str(exported_onnx_path)

        if model_path is not None and actor_uses_perception:
            assert checkpoint_sha256 is not None
            try:
                digest, perception_contract_envelope_b64 = _preflight_perception_onnx_artifact(
                    model_path,
                    checkpoint_sha256=checkpoint_sha256,
                    config=tyro_config,
                    return_contract_envelope=True,
                )
                logger.info(
                    "Authenticated existing perception ONNX before launch: path={} contract_sha256={}",
                    model_path,
                    digest,
                )
            except Exception as exc:
                if explicit_model_path:
                    raise
                logger.warning(
                    "Ignoring stale/unverifiable auto-exported perception ONNX {} and rebuilding it: {}",
                    model_path,
                    exc,
                )
                model_path = None
                perception_contract_envelope_b64 = None

        if model_path is None:
            # Use shared simulation environment setup only if we need to export ONNX.
            env, device, simulation_app = setup_simulation_environment(tyro_config)
            try:
                algo_class = get_class(tyro_config.algo._target_)
                algo: BaseAlgo = algo_class(
                    device=device,
                    env=env,
                    config=tyro_config.algo.config,
                    log_dir=str(eval_log_dir),
                    multi_gpu_cfg=None,
                )
                algo.attach_evaluation_metadata(
                    saved_config,
                    tyro_config,
                    saved_wandb_path,
                )
                algo.setup()
                algo.load_evaluation(checkpoint_path)

                os.makedirs(export_dir, exist_ok=True)
                if not hasattr(algo, "export"):
                    raise AttributeError(
                        f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
                    )
                algo.export(onnx_file_path=str(exported_onnx_path))  # type: ignore[attr-defined]
                logger.info(f"Exported policy as onnx to: {exported_onnx_path}")
            finally:
                if hasattr(env, "close"):
                    env.close()
                if simulation_app:
                    close_simulation_app(simulation_app)

            model_path = str(exported_onnx_path)
            if actor_uses_perception:
                assert checkpoint_sha256 is not None
                digest, perception_contract_envelope_b64 = _preflight_perception_onnx_artifact(
                    model_path,
                    checkpoint_sha256=checkpoint_sha256,
                    config=tyro_config,
                    return_contract_envelope=True,
                )
                logger.info(
                    "Authenticated freshly exported perception ONNX: contract_sha256={}",
                    digest,
                )

        run_sim_cmd, run_policy_cmd = _build_sim2sim_commands(
            tyro_config,
            sim2sim_cfg,
            model_path,
            perception_contract_envelope_b64=perception_contract_envelope_b64,
        )
        logger.info("MuJoCo sim-to-sim commands:")
        logger.info("  run_sim:   " + " ".join(shlex.quote(part) for part in run_sim_cmd))
        logger.info("  run_policy:" + " ".join(shlex.quote(part) for part in run_policy_cmd))

        if sim2sim_cfg.auto_launch:
            _launch_sim2sim(run_sim_cmd, run_policy_cmd)
        return

    # Standard in-simulator evaluation
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.attach_evaluation_metadata(
        saved_config,
        tyro_config,
        saved_wandb_path,
    )
    algo.setup()
    algo.load_evaluation(checkpoint_path)

    if tyro_config.training.export_onnx:
        os.makedirs(export_dir, exist_ok=True)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )
        algo.export(onnx_file_path=str(exported_onnx_path))  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    eval_cli_cfg, remaining_args = tyro.cli(
        EvalCliConfig,
        args=remaining_args,
        return_unknown_args=True,
        add_help=False,
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cli_cfg.sim2sim)


if __name__ == "__main__":
    main()
