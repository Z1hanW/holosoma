from __future__ import annotations

import itertools
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

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
        status_getter = getattr(motion_command, "get_manual_forward_after_lift_status", None)
        if callable(status_getter):
            after_lift_status = status_getter(0)
            if after_lift_status is not None:
                record["manual_forward_after_lift"] = after_lift_status
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
        checkpoint_path = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
        algo.load_evaluation(str(checkpoint_path))

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
        if record_cfg.manual_forward_command_m is not None:
            forward_command = float(record_cfg.manual_forward_command_m)
            if not math.isfinite(forward_command):
                raise ValueError(f"manual_forward_command_m must be finite, got {forward_command!r}.")
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
            configure_after_lift(
                command_m=record_cfg.manual_forward_after_lift_command_m,
                rel_z_delta_m=record_cfg.manual_forward_after_lift_rel_z_delta_m,
                consecutive_steps=record_cfg.manual_forward_after_lift_consecutive_steps,
            )
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
        with metrics_path.open("w", encoding="utf-8") as f:
            for step in itertools.islice(itertools.count(), int(record_cfg.steps)):
                actor_state["step"] = step
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

        if record_cfg.manual_forward_after_lift_command_m is not None:
            status_getter = getattr(motion_command, "get_manual_forward_after_lift_status", None)
            final_status = status_getter(0) if callable(status_getter) else None
            if not isinstance(final_status, dict) or not bool(final_status.get("triggered")):
                raise RuntimeError(
                    "Object never satisfied the configured stable-lift trigger; "
                    f"final_status={final_status!r}."
                )
        if recorder.is_recording:
            recorder.stop_recording()
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
