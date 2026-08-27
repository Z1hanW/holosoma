from __future__ import annotations

import itertools
import inspect
import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


@dataclass(frozen=True)
class RecordConfig:
    steps: int = 500
    video_episode_id: int = 500
    metrics_path: str = ""
    initial_motion_timestep: int | None = None
    manual_forward_command_m: float | None = None
    manual_forward_after_lift_command_m: float | None = None
    manual_forward_after_lift_rel_z_delta_m: float = 0.3
    manual_forward_after_lift_consecutive_steps: int = 10
    # Command-only ablation: keep the checkpoint-native reference-timed
    # pickup/drop observations while replacing only the three root-command
    # slots.  Default False preserves the historical manual-control contract.
    manual_forward_after_lift_preserve_native_buttons: bool = False
    # More precise command-only controls.  These allow pickup to retain its
    # checkpoint-native phase cue while drop is forced to zero for a persistent
    # post-lift forward command experiment.
    manual_forward_after_lift_preserve_native_pickup_button: bool = False
    manual_forward_after_lift_preserve_native_drop_button: bool = False
    # Declares how the selected checkpoint interprets its three command slots.
    # The default preserves the established sparse-root evaluation contract.
    manual_forward_after_lift_command_semantics: str = "legacy_constant_robot_heading_frame"
    # A missed lift gate is a valid outcome in broad object-position sweeps.
    # When enabled, retain the complete rollout and its explicit not-triggered
    # status instead of turning that policy failure into a recorder failure.
    manual_forward_after_lift_allow_not_triggered: bool = False
    # Diagnostic only.  Formal deployment-faithful evaluation keeps the raw
    # command constant unless heading lock is explicitly requested.
    manual_forward_heading_lock: bool = False
    # Evaluation-only research diagnostic. This permits only the selected
    # object mesh to be outside the checkpoint's authenticated training
    # support; every other policy/perception identity check remains strict.
    allow_ood_object_geometry: bool = False
    # Evaluation-only ablation for a checkpoint trained with immutable
    # precomputed turn/forward commands.  The repository-native pickup latch
    # remains authoritative, but the actor receives only the NPZ dx component:
    # [dx, 0, 0].  Training and ordinary evaluation default to False.
    precomputed_dx_only_after_pickup: bool = False
    # Evaluation-only displacement applied to the physical object at reset.
    # This is deliberately a world-frame offset from the authenticated motion
    # placement: the motion/checkpoint bytes remain unchanged and the exact
    # realized object pose is retained in the per-step metrics.
    manual_object_reset_pos_offset_w: tuple[float, float, float] | None = None
    # Evaluation-only depth audit. Each policy step stores the raw rendered
    # depth, the perception manager's processed/latency depth, and the exact
    # perception tensor passed to the actor in a compressed NPZ.
    depth_trace_path: str = ""


def _save_depth_trace(
    path: Path,
    *,
    actor_perception_key: str,
    raw_depth_frames: list[np.ndarray],
    processed_depth_frames: list[np.ndarray],
    actor_depth_frames: list[np.ndarray],
    camera_positions: list[np.ndarray],
    camera_quaternions: list[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_depth = np.stack(raw_depth_frames).astype(np.float32, copy=False)
    processed_depth = np.stack(processed_depth_frames).astype(np.float32, copy=False)
    actor_depth = np.stack(actor_depth_frames).astype(np.float32, copy=False)
    camera_position = np.stack(camera_positions).astype(np.float32, copy=False)
    camera_quaternion = np.stack(camera_quaternions).astype(np.float32, copy=False)
    if processed_depth.shape != actor_depth.shape:
        raise RuntimeError(
            "Processed depth and exact actor input have different shapes: "
            f"{processed_depth.shape} vs {actor_depth.shape}."
        )
    finite_pair = np.isfinite(processed_depth) & np.isfinite(actor_depth)
    max_abs_error = (
        float(np.max(np.abs(processed_depth[finite_pair] - actor_depth[finite_pair])))
        if bool(np.any(finite_pair))
        else None
    )
    nan_mask_equal = bool(np.array_equal(np.isnan(processed_depth), np.isnan(actor_depth)))
    np.savez_compressed(
        path,
        step=np.arange(actor_depth.shape[0], dtype=np.int32),
        camera_depth_raw=raw_depth,
        camera_depth_processed=processed_depth,
        actor_depth_input=actor_depth,
        camera_pose_position_w=camera_position,
        camera_pose_quaternion_xyzw_w=camera_quaternion,
    )
    metadata = {
        "semantics": "exact_per_step_actor_depth_input_with_synchronous_perception_maps",
        "actor_perception_key": actor_perception_key,
        "num_steps": int(actor_depth.shape[0]),
        "raw_depth_shape": list(raw_depth.shape),
        "processed_depth_shape": list(processed_depth.shape),
        "actor_depth_input_shape": list(actor_depth.shape),
        "processed_vs_actor_finite_max_abs_error": max_abs_error,
        "processed_vs_actor_nan_mask_equal": nan_mask_equal,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _actor_object_state(simulator, device: torch.device) -> dict[str, list[float]]:
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    actor_metadata = getattr(simulator, "_actor_root_metadata", {})
    if isinstance(actor_metadata, dict) and actor_metadata:
        actor_names = [name for name in actor_metadata if name != "robot"]
    else:
        actor_names = list(getattr(simulator, "_object_urdf_by_name", {}).keys()) or ["object"]

    states: dict[str, list[float]] = {}
    for name in actor_names:
        try:
            actor_state = simulator.get_actor_states([name], env_ids)
        except Exception:
            continue
        if actor_state.numel() == 0:
            continue
        states[str(name)] = actor_state[0].detach().cpu().reshape(-1).to(torch.float32).tolist()
    return states


def _write_metric(
    f,
    *,
    step: int,
    actor_state: dict,
    simulator,
    device: torch.device,
    motion_command,
) -> None:
    actions = actor_state.get("actions")
    rewards = actor_state.get("rewards")
    dones = actor_state.get("dones")
    record = {
        "step": int(step),
        "sim_time_s": float(simulator.time()),
        "robot_root_state": simulator.robot_root_states[0].detach().cpu().reshape(-1).to(torch.float32).tolist(),
        "objects": _actor_object_state(simulator, device),
    }
    if motion_command is not None:
        time_steps = getattr(motion_command, "time_steps", None)
        clip_ids = getattr(motion_command, "clip_ids", None)
        if isinstance(time_steps, torch.Tensor) and time_steps.numel() > 0:
            record["motion_timestep"] = int(time_steps[0].item())
        if isinstance(clip_ids, torch.Tensor) and clip_ids.numel() > 0:
            record["motion_clip_index"] = int(clip_ids[0].item())
        if bool(
            getattr(
                motion_command,
                "_evaluation_precomputed_dx_only_after_pickup",
                False,
            )
        ):
            pickup_anchor_set = getattr(motion_command, "pickup_anchor_set", None)
            record["evaluation_precomputed_dx_only_after_pickup"] = {
                "enabled": True,
                "native_pickup_anchor_set": (
                    bool(pickup_anchor_set[0].item())
                    if isinstance(pickup_anchor_set, torch.Tensor)
                    and pickup_anchor_set.numel() > 0
                    else None
                ),
            }
        status_getter = getattr(motion_command, "get_manual_forward_after_lift_status", None)
        if callable(status_getter):
            after_lift_status = status_getter(0)
            if after_lift_status is not None:
                record["manual_forward_after_lift"] = after_lift_status
        if bool(getattr(motion_command, "manual_object_reset_enabled", False)):
            offset = getattr(motion_command, "manual_object_reset_pos_offset_w", None)
            if isinstance(offset, torch.Tensor) and offset.ndim == 2 and offset.shape[0] > 0:
                record["manual_object_reset_pos_offset_w"] = (
                    offset[0].detach().cpu().to(torch.float32).tolist()
                )
    if actions is not None:
        action0 = actions[0].detach().cpu().reshape(-1).to(torch.float32)
        record["action_l2"] = float(torch.linalg.vector_norm(action0).item())
        record["action_abs_max"] = float(action0.abs().max().item())
        record["actions"] = action0.tolist()
    if rewards is not None:
        record["reward"] = float(rewards[0].detach().cpu().item())
    if dones is not None:
        record["done"] = bool(dones[0].detach().cpu().item())
    f.write(json.dumps(record, separators=(",", ":")) + "\n")
    f.flush()


@torch.no_grad()
def run_recording(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    record_cfg: RecordConfig,
) -> None:
    tyro_config = apply_observation_overrides(tyro_config)
    tyro_config = apply_perception_overrides(tyro_config)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    from holosoma.eval_agent import (
        _bind_training_perception_reference_batch,
        _validate_eval_policy_contract,
    )

    _validate_eval_policy_contract(saved_cfg, tyro_config)
    tyro_config = _bind_training_perception_reference_batch(saved_cfg, tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval_record")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

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
            saved_cfg,
            tyro_config,
            saved_wandb_path,
        )
        algo.setup()
        if record_cfg.allow_ood_object_geometry:
            requested_policy = os.environ.get(
                "HOLOSOMA_EVAL_POLICY",
                "checkpoint_actor",
            ).strip().lower()
            if requested_policy not in {"", "checkpoint_actor", "student"}:
                raise ValueError(
                    "OOD object geometry is supported only for the exact checkpoint_actor, "
                    f"not HOLOSOMA_EVAL_POLICY={requested_policy!r}."
                )
            if int(env.num_envs) != 1:
                raise ValueError(
                    "OOD object-geometry evaluation requires exactly one environment, "
                    f"got {env.num_envs}."
                )
            algo.enable_evaluation_only_ood_object_geometry()
        checkpoint_path = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
        algo.load_evaluation(str(checkpoint_path))
        if record_cfg.allow_ood_object_geometry:
            geometry_audit = algo.evaluation_ood_object_geometry_audit()
            if geometry_audit is None:
                raise RuntimeError(
                    "OOD object-geometry evaluation produced no authenticated audit evidence."
                )
            if not record_cfg.metrics_path:
                raise ValueError(
                    "OOD object-geometry evaluation requires metrics_path so its audit "
                    "evidence has an unambiguous output location."
                )
            geometry_audit_path = (
                Path(record_cfg.metrics_path).resolve().parent
                / "ood_geometry_audit.json"
            )
            geometry_audit_path.parent.mkdir(parents=True, exist_ok=True)
            geometry_audit_path.write_text(
                json.dumps(geometry_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        algo._create_eval_callbacks()  # noqa: SLF001 - debug script mirrors evaluate_policy.
        motion_command = env.command_manager.get_state("motion_command")
        if record_cfg.initial_motion_timestep is not None:
            setter = getattr(motion_command, "set_forced_reset_timestep", None)
            if not callable(setter):
                raise RuntimeError("Motion command does not support an exact forced reset timestep.")
            setter(record_cfg.initial_motion_timestep)
        if (
            record_cfg.manual_forward_command_m is not None
            and record_cfg.manual_forward_after_lift_command_m is not None
        ):
            raise ValueError(
                "manual_forward_command_m and manual_forward_after_lift_command_m are mutually exclusive."
            )
        if record_cfg.precomputed_dx_only_after_pickup and (
            record_cfg.manual_forward_command_m is not None
            or record_cfg.manual_forward_after_lift_command_m is not None
            or record_cfg.manual_forward_heading_lock
        ):
            raise ValueError(
                "precomputed_dx_only_after_pickup is mutually exclusive with all manual-forward modes."
            )
        if record_cfg.precomputed_dx_only_after_pickup:
            requested_policy = os.environ.get(
                "HOLOSOMA_EVAL_POLICY",
                "checkpoint_actor",
            ).strip().lower()
            if requested_policy not in {"", "checkpoint_actor", "student"}:
                raise ValueError(
                    "precomputed_dx_only_after_pickup supports only the exact checkpoint actor, "
                    f"not HOLOSOMA_EVAL_POLICY={requested_policy!r}."
                )
            if int(env.num_envs) != 1:
                raise ValueError(
                    "precomputed_dx_only_after_pickup requires exactly one evaluation environment."
                )
            enabled = getattr(
                motion_command,
                "precomputed_turn_then_forward_enabled",
                None,
            )
            if not callable(enabled) or not bool(enabled()):
                raise ValueError(
                    "precomputed_dx_only_after_pickup requires a checkpoint configured with "
                    "precomputed_turn_then_forward."
                )
            motion_command._evaluation_precomputed_dx_only_after_pickup = True
            print(
                "[INFO] Enabled evaluation-only native-pickup-latch NPZ dx-only actor command: "
                "[dx,0,0]; dy=0; dyaw=0."
            )
        if record_cfg.manual_object_reset_pos_offset_w is not None:
            if int(env.num_envs) != 1:
                raise ValueError(
                    "manual_object_reset_pos_offset_w requires exactly one evaluation environment."
                )
            object_offset = tuple(float(value) for value in record_cfg.manual_object_reset_pos_offset_w)
            if len(object_offset) != 3 or not all(math.isfinite(value) for value in object_offset):
                raise ValueError(
                    "manual_object_reset_pos_offset_w must contain exactly three finite values, "
                    f"got {record_cfg.manual_object_reset_pos_offset_w!r}."
                )
            if not hasattr(motion_command, "manual_object_reset_enabled"):
                raise RuntimeError("Motion command does not support a manual object reset offset.")
            motion_command.manual_object_reset_enabled = True
            motion_command.manual_object_reset_pos_offset_w = torch.tensor(
                [object_offset],
                device=device,
                dtype=torch.float32,
            )
            motion_command.manual_object_reset_rpy_offset = torch.zeros(
                (1, 3),
                device=device,
                dtype=torch.float32,
            )
            print(
                "[INFO] Enabled evaluation-only physical-object reset offset in world frame: "
                f"{list(object_offset)} m."
            )
        if record_cfg.manual_forward_command_m is not None:
            forward_command = float(record_cfg.manual_forward_command_m)
            if not math.isfinite(forward_command):
                raise ValueError(f"manual_forward_command_m must be finite, got {forward_command!r}.")
            if record_cfg.manual_forward_heading_lock:
                configure_heading_locked = getattr(
                    motion_command,
                    "configure_manual_heading_locked_forward",
                    None,
                )
                if not callable(configure_heading_locked):
                    raise RuntimeError("Motion command does not support heading-locked manual forward evaluation.")
                configure_heading_locked(command_m=forward_command)
            else:
                motion_command.manual_control_enabled = True
                motion_command.manual_xy_rel = torch.zeros(
                    (env.num_envs, 2),
                    device=device,
                    dtype=torch.float32,
                )
                motion_command.manual_xy_rel[:, 0] = forward_command
                motion_command.manual_yaw_rel = torch.zeros(
                    (env.num_envs, 1),
                    device=device,
                    dtype=torch.float32,
                )
        obs_dict = algo._pre_evaluate_policy()  # noqa: SLF001
        if record_cfg.manual_forward_after_lift_command_m is not None:
            configure_after_lift = getattr(motion_command, "configure_manual_forward_after_lift", None)
            if not callable(configure_after_lift):
                raise RuntimeError("Motion command does not support manual forward-after-lift evaluation.")
            configure_kwargs = dict(
                command_m=record_cfg.manual_forward_after_lift_command_m,
                rel_z_delta_m=record_cfg.manual_forward_after_lift_rel_z_delta_m,
                consecutive_steps=record_cfg.manual_forward_after_lift_consecutive_steps,
                preserve_native_contact_buttons=(
                    record_cfg.manual_forward_after_lift_preserve_native_buttons
                ),
                preserve_native_pickup_button=(
                    record_cfg.manual_forward_after_lift_preserve_native_pickup_button
                ),
                preserve_native_drop_button=(
                    record_cfg.manual_forward_after_lift_preserve_native_drop_button
                ),
                heading_lock=record_cfg.manual_forward_heading_lock,
                command_semantics=record_cfg.manual_forward_after_lift_command_semantics,
            )
            configure_parameters = inspect.signature(configure_after_lift).parameters
            supports_explicit_button_contract = {
                "preserve_native_contact_buttons",
                "preserve_native_pickup_button",
                "preserve_native_drop_button",
            }.issubset(configure_parameters)
            if supports_explicit_button_contract:
                configure_after_lift(**configure_kwargs)
            else:
                # Frozen pre-button-contract sources already implemented the
                # default contract by leaving pickup native and forcing drop
                # to zero. Only permit that exact request, then verify the
                # resulting state instead of silently dropping newer options.
                if (
                    record_cfg.manual_forward_after_lift_preserve_native_buttons
                    or not record_cfg.manual_forward_after_lift_preserve_native_pickup_button
                    or record_cfg.manual_forward_after_lift_preserve_native_drop_button
                ):
                    raise RuntimeError(
                        "Legacy motion command supports only native pickup with drop forced to zero."
                    )
                legacy_kwargs = {
                    key: value
                    for key, value in configure_kwargs.items()
                    if key in configure_parameters
                }
                configure_after_lift(**legacy_kwargs)
                pickup_override = bool(
                    getattr(motion_command, "manual_pickup_button_override_enabled", False)
                )
                drop_override = bool(
                    getattr(motion_command, "manual_drop_button_override_enabled", False)
                )
                manual_drop = getattr(motion_command, "manual_drop_button", None)
                if (
                    pickup_override
                    or not drop_override
                    or not isinstance(manual_drop, torch.Tensor)
                    or bool(torch.count_nonzero(manual_drop).item())
                ):
                    raise RuntimeError(
                        "Legacy motion command failed native-pickup/drop-zero runtime verification."
                    )
                print(
                    "[INFO] Verified legacy forward-after-lift button contract: "
                    "native pickup retained; drop forced to zero."
                )
            hmi_goal_key = "actor_obs_hmi_goal_command"
            if hmi_goal_key in obs_dict:
                get_hmi_command = getattr(
                    motion_command,
                    "get_hmi_object_goal_command",
                    None,
                )
                if not callable(get_hmi_command):
                    raise RuntimeError(
                        "HMI actor observation requires a refreshable HMI goal command."
                    )
                # The initial observation was assembled before the manual latch was
                # configured. Refresh only this command term so frame zero obeys the
                # zero-before-lift contract without advancing perception or clocks.
                obs_dict[hmi_goal_key] = get_hmi_command()
        algo.eval_policy = algo.get_inference_policy()

        init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
        actor_state = algo._create_actor_state()  # noqa: SLF001
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        critic_obs = torch.cat([actor_state["obs"][k] for k in algo.critic_obs_keys], dim=1)
        actor_state["obs"]["critic_obs"] = critic_obs

        recorder = getattr(env.simulator, "video_recorder", None)
        if recorder is None:
            raise RuntimeError("Video recorder is not configured. Pass --logger.video.enabled=True.")
        if recorder.is_recording:
            recorder.stop_recording()
        recorder.start_recording(record_cfg.video_episode_id)

        metrics_path = Path(record_cfg.metrics_path) if record_cfg.metrics_path else eval_log_dir / "record_metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        depth_trace_path = Path(record_cfg.depth_trace_path) if record_cfg.depth_trace_path else None
        raw_depth_frames: list[np.ndarray] = []
        processed_depth_frames: list[np.ndarray] = []
        actor_depth_frames: list[np.ndarray] = []
        camera_positions: list[np.ndarray] = []
        camera_quaternions: list[np.ndarray] = []
        actor_perception_key = str(getattr(algo, "actor_perception_key", "") or "")
        perception_manager = getattr(env, "perception_manager", None)
        if depth_trace_path is not None:
            if not actor_perception_key or actor_perception_key not in actor_state["obs"]:
                raise RuntimeError(
                    "depth_trace_path requires a checkpoint actor with an explicit perception input."
                )
            if perception_manager is None or not bool(getattr(perception_manager, "enabled", False)):
                raise RuntimeError("depth_trace_path requires an enabled perception manager.")
        try:
            with metrics_path.open("w", encoding="utf-8") as f:
                for step in itertools.islice(itertools.count(), int(record_cfg.steps)):
                    actor_state["step"] = step
                    if depth_trace_path is not None:
                        raw_depth = perception_manager.get_camera_depth_map()[0]
                        processed_depth = perception_manager.get_camera_depth_obs_map()[0]
                        actor_depth_flat = actor_state["obs"][actor_perception_key][0]
                        if actor_depth_flat.numel() != processed_depth.numel():
                            raise RuntimeError(
                                "Exact actor perception width does not match the processed depth map: "
                                f"{actor_depth_flat.numel()} vs {processed_depth.numel()}."
                            )
                        actor_depth = actor_depth_flat.reshape_as(processed_depth)
                        env_ids = torch.tensor([0], device=device, dtype=torch.long)
                        camera_position, camera_quaternion = perception_manager.get_camera_pose(
                            env_ids,
                            apply_sensor_offset=True,
                            apply_pitch=True,
                        )
                        raw_depth_frames.append(
                            raw_depth.detach().cpu().to(torch.float32).numpy().copy()
                        )
                        processed_depth_frames.append(
                            processed_depth.detach().cpu().to(torch.float32).numpy().copy()
                        )
                        actor_depth_frames.append(
                            actor_depth.detach().cpu().to(torch.float32).numpy().copy()
                        )
                        camera_positions.append(
                            camera_position[0].detach().cpu().to(torch.float32).numpy().copy()
                        )
                        camera_quaternions.append(
                            camera_quaternion[0].detach().cpu().to(torch.float32).numpy().copy()
                        )
                    actor_state = algo._pre_eval_env_step(actor_state)  # noqa: SLF001
                    actor_state = algo.env_step(actor_state)
                    _write_metric(
                        f,
                        step=step,
                        actor_state=actor_state,
                        simulator=env.simulator,
                        device=device,
                        motion_command=motion_command,
                    )
                    actor_state = algo._post_eval_env_step(actor_state)  # noqa: SLF001
        finally:
            # Failed pickup is a valid evaluation outcome. Finalize the full
            # rollout video before reporting that verdict so failures do not
            # collapse into one-frame startup artifacts.
            if recorder.is_recording:
                recorder.stop_recording()
            if depth_trace_path is not None and actor_depth_frames:
                _save_depth_trace(
                    depth_trace_path,
                    actor_perception_key=actor_perception_key,
                    raw_depth_frames=raw_depth_frames,
                    processed_depth_frames=processed_depth_frames,
                    actor_depth_frames=actor_depth_frames,
                    camera_positions=camera_positions,
                    camera_quaternions=camera_quaternions,
                )
                print(f"[INFO] depth_trace_path={depth_trace_path}")

        if record_cfg.manual_forward_after_lift_command_m is not None:
            status_getter = getattr(motion_command, "get_manual_forward_after_lift_status", None)
            final_status = status_getter(0) if callable(status_getter) else None
            if (
                not isinstance(final_status, dict)
                or not bool(final_status.get("triggered"))
            ) and not record_cfg.manual_forward_after_lift_allow_not_triggered:
                raise RuntimeError(
                    "Object never satisfied the configured stable-lift trigger; "
                    f"final_status={final_status!r}."
                )
            if (
                not isinstance(final_status, dict)
                or not bool(final_status.get("triggered"))
            ):
                print(
                    "[WARN] Object did not satisfy the stable-lift trigger; "
                    f"retaining rollout as not_triggered: {final_status!r}."
                )
        algo._post_evaluate_policy()  # noqa: SLF001
        print(f"[INFO] metrics_path={metrics_path}")
    except BaseException:
        # Isaac/Kit shutdown can be slow or hang after an initialization
        # failure.  Emit the originating exception before entering cleanup so
        # diagnostics are not hidden behind the shutdown path.
        traceback.print_exc()
        raise
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, _ = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    record_cfg, remaining_args = tyro.cli(
        RecordConfig,
        args=remaining_args,
        return_unknown_args=True,
        add_help=False,
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of checkpoint config.",
        config=TYRO_CONIFG,
    )
    run_recording(overwritten_tyro_config, checkpoint_cfg, record_cfg)


if __name__ == "__main__":
    main()
