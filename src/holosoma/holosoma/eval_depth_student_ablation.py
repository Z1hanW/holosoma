from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.distill_depth_student import (
    DepthStudentActorCritic,
    DepthStudentPolicy,
    _compute_depth,
    _get_actor_obs_group,
    build_tracking_teacher_policy,
    get_actor_term_slices,
    select_student_lowdim_obs,
)
from holosoma.train_agent import get_device
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, init_sim_imports, load_checkpoint
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app


def _import_torch():
    import torch

    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether a depth-student policy actually depends on depth observations."
    )
    parser.add_argument("--checkpoint", required=True, help="Depth-student checkpoint path.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--modes", nargs="+", default=["normal", "zero", "shuffle"])
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional teacher checkpoint for action L1 diagnostics.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _depth_args_from_checkpoint(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        depth_min_range=ckpt_args.get("depth_min_range", 0.3),
        depth_max_range=ckpt_args.get("depth_max_range", 2.0),
        depth_height=ckpt_args.get("depth_height", 58),
        depth_width=ckpt_args.get("depth_width", 87),
        depth_resize_mode=ckpt_args.get("depth_resize_mode", None),
        depth_camera_self_occlusion=ckpt_args.get("depth_camera_self_occlusion", None),
        depth_sensor_noise=ckpt_args.get("depth_sensor_noise", None),
        depth_pixel_std_dev_multiplier=ckpt_args.get("depth_pixel_std_dev_multiplier", None),
        depth_pixel_dropout_prob=ckpt_args.get("depth_pixel_dropout_prob", None),
        depth_latency_frame_min=ckpt_args.get("depth_latency_frame_min", None),
        depth_latency_frame_max=ckpt_args.get("depth_latency_frame_max", None),
        depth_buffer_len=ckpt_args.get("depth_buffer_len", None),
        student_command_mode=ckpt_args.get("student_command_mode", "root_xy_yaw"),
        student_include_projected_gravity=ckpt_args.get("student_include_projected_gravity", True),
    )


def _make_eval_config(checkpoint: dict[str, Any], args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig(**checkpoint["experiment_config"])
    ckpt_args = checkpoint.get("args", {})
    if args.seed is not None:
        seed = args.seed
    else:
        seed = config.training.seed
    return dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            headless=True,
            num_envs=args.num_envs,
            export_onnx=False,
            seed=seed,
        ),
    )


def _load_student(checkpoint: dict[str, Any], device: str):
    torch = _import_torch()
    ckpt_args = checkpoint.get("args", {})
    training_mode = ckpt_args.get("training_mode", "hybrid")
    if training_mode == "hybrid":
        model = DepthStudentActorCritic(
            lowdim_obs_dim=checkpoint.get("lowdim_obs_dim", checkpoint["proprio_dim"]),
            critic_obs_dim=checkpoint["critic_obs_dim"],
            action_dim=checkpoint["action_dim"],
            depth_shape=tuple(checkpoint["depth_shape"]),
            actor_hidden_dims=ckpt_args.get("student_hidden_dims", [2048, 1024, 512, 256, 128]),
            critic_hidden_dims=ckpt_args.get("critic_hidden_dims", [512, 256, 128]),
            depth_latent_dim=ckpt_args.get("depth_latent_dim", 32),
            init_noise_std=ckpt_args.get("init_noise_std", 0.01),
        ).to(device)
    else:
        model = DepthStudentPolicy(
            lowdim_obs_dim=checkpoint.get("lowdim_obs_dim", checkpoint["proprio_dim"]),
            action_dim=checkpoint["action_dim"],
            depth_shape=tuple(checkpoint["depth_shape"]),
            hidden_dims=ckpt_args.get("student_hidden_dims", [2048, 1024, 512, 256, 128]),
            depth_latent_dim=ckpt_args.get("depth_latent_dim", 32),
        ).to(device)
    model.load_state_dict(checkpoint["student_state_dict"])
    model.eval()
    return model, training_mode


def _student_action(model: Any, training_mode: str, lowdim_obs: Any, depth: Any) -> Any:
    if training_mode == "hybrid":
        action_mean, _ = model(lowdim_obs, depth, None)
        return action_mean
    return model(lowdim_obs, depth)


def _ablate_depth(depth: Any, mode: str) -> Any:
    torch = _import_torch()
    if mode == "normal":
        return depth
    if mode == "teacher":
        return depth
    if mode == "zero":
        return torch.zeros_like(depth)
    if mode == "shuffle":
        return depth[torch.randperm(depth.shape[0], device=depth.device)]
    raise ValueError(f"Unknown depth ablation mode: {mode}")


def _maybe_build_teacher(env: Any, checkpoint: dict[str, Any], args: argparse.Namespace, device: str):
    teacher_checkpoint = args.teacher_checkpoint or checkpoint.get("teacher_checkpoint")
    if not teacher_checkpoint:
        return None
    teacher_path = Path(str(teacher_checkpoint))
    if not teacher_path.exists():
        logger.warning(f"Teacher checkpoint not found; skipping action L1 diagnostics: {teacher_checkpoint}")
        return None
    loaded_teacher = load_checkpoint(str(teacher_path), str(teacher_path.parent))
    teacher_policy, _action_dim, _actor_obs_keys = build_tracking_teacher_policy(
        env,
        loaded_teacher,
        ExperimentConfig(**checkpoint["experiment_config"]).algo.config,
        device,
    )
    return teacher_policy


def evaluate_mode(
    env: Any,
    model: Any,
    training_mode: str,
    term_slices: dict[str, slice],
    depth_args: argparse.Namespace,
    mode: str,
    steps: int,
    device: str,
    teacher_policy: Any | None,
) -> dict[str, float | str]:
    torch = _import_torch()
    obs_dict = env.reset_all()
    reward_sum = torch.zeros(env.num_envs, device=device)
    reward_step_sum = 0.0
    episode_returns: list[float] = []
    done_sum = 0.0
    timeout_sum = 0.0
    non_timeout_done_sum = 0.0
    action_l1_sum = 0.0
    action_l1_count = 0
    depth_mean_sum = 0.0
    depth_std_sum = 0.0

    with torch.no_grad():
        for _ in range(steps):
            actor_obs = _get_actor_obs_group(obs_dict).to(device=device, dtype=torch.float)
            lowdim_obs = select_student_lowdim_obs(env, actor_obs, term_slices, depth_args)
            depth = _compute_depth(env, depth_args, device)
            if mode == "teacher":
                if teacher_policy is None:
                    raise RuntimeError("Mode 'teacher' requires --teacher-checkpoint.")
                actions = teacher_policy(obs_dict).to(device=device, dtype=torch.float)
            else:
                depth_for_policy = _ablate_depth(depth, mode)
                actions = _student_action(model, training_mode, lowdim_obs, depth_for_policy)
            if teacher_policy is not None:
                teacher_actions = teacher_policy(obs_dict).to(device=device, dtype=torch.float)
                action_l1_sum += float((actions - teacher_actions).abs().mean().item())
                action_l1_count += 1
            obs_dict, rewards, dones, extras = env.step({"actions": actions})
            rewards = rewards.detach().float().view(-1)
            dones = dones.detach().bool().view(-1)
            timeouts = extras.get("time_outs")
            if timeouts is None:
                timeouts = torch.zeros_like(dones)
            else:
                timeouts = timeouts.detach().bool().view(-1)
            reward_step_sum += float(rewards.mean().item())
            reward_sum += rewards
            done_sum += float(dones.float().mean().item())
            timeout_sum += float((dones & timeouts).float().mean().item())
            non_timeout_done_sum += float((dones & ~timeouts).float().mean().item())
            if dones.any():
                episode_returns.extend(float(v) for v in reward_sum[dones].detach().cpu())
                reward_sum[dones] = 0.0
            depth_mean_sum += float(depth.mean().item())
            depth_std_sum += float(depth.std(unbiased=False).item())

    return {
        "mode": mode,
        "steps": float(steps),
        "reward_mean": reward_step_sum / max(steps, 1),
        "episode_return_mean": float(sum(episode_returns) / max(len(episode_returns), 1)),
        "episode_return_count": float(len(episode_returns)),
        "done_rate": done_sum / max(steps, 1),
        "timeout_rate": timeout_sum / max(steps, 1),
        "non_timeout_done_rate": non_timeout_done_sum / max(steps, 1),
        "teacher_action_l1": action_l1_sum / max(action_l1_count, 1),
        "depth_mean": depth_mean_sum / max(steps, 1),
        "depth_std": depth_std_sum / max(steps, 1),
    }


def main() -> None:
    init_eval_logging()
    args = parse_args()
    torch = _import_torch()
    simulation_app = None
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = _make_eval_config(checkpoint, args)
        simulation_app = init_sim_imports(config)
        device = get_device(config.training, distributed_conf=None)
        from holosoma.utils.common import seeding

        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        env = get_class(config.env_class)(get_tyro_env_config(config), device=device)
        term_slices = get_actor_term_slices(env, "actor_obs")
        model, training_mode = _load_student(checkpoint, device)
        depth_args = _depth_args_from_checkpoint(checkpoint.get("args", {}))
        teacher_policy = _maybe_build_teacher(env, checkpoint, args, device)

        results = []
        for mode in args.modes:
            result = evaluate_mode(
                env,
                model,
                training_mode,
                term_slices,
                depth_args,
                mode,
                args.steps,
                device,
                teacher_policy,
            )
            results.append(result)
            logger.info(json.dumps(result, sort_keys=True))

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
            logger.info(f"Wrote depth ablation results to {output_path}")
    except Exception as exc:
        logger.error(f"Depth student ablation failed: {exc}")
        traceback.print_exc()
        raise
    finally:
        if simulation_app is not None:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
