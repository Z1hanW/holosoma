from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.agents.fast_sac.fast_sac import Actor, CNNActor
from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization as FastSACEmpiricalNormalization
from holosoma.agents.modules.module_utils import setup_ppo_actor_module
from holosoma.agents.ppo.ppo import EmpiricalNormalization as PPOEmpiricalNormalization
from holosoma.config_types.algo import FastSACConfig, PPOConfig
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.managers.observation.terms.wbt import depth_camera as depth_camera_obs
from holosoma.train_agent import configure_logging, configure_multi_gpu, get_device
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, init_sim_imports, load_checkpoint, load_saved_experiment_config
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app


def _import_torch():
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel

    return torch, dist, nn, F, DistributedDataParallel


class DepthStudentPolicy(_import_torch()[2].Module):
    """Depth student that consumes proprioception plus a normalized depth image."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        depth_shape: tuple[int, int, int],
        hidden_dims: list[int],
        depth_latent_dim: int,
    ):
        torch, _, nn, _, _ = _import_torch()
        super().__init__()
        channels, _, _ = depth_shape
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.depth_shape = depth_shape

        # Matches far-tracking's DepthOnlyFCBackbone58x87Small, with a channel
        # dimension already present in this pipeline.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, depth_latent_dim),
            nn.ELU(),
        )

        layers: list[nn.Module] = []
        last_dim = proprio_dim + depth_latent_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ELU(),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))
        self.action_head = nn.Sequential(*layers)

        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.action_head[-1].weight)
        torch.nn.init.zeros_(self.action_head[-1].bias)

    @staticmethod
    def _init_weights(module: Any) -> None:
        torch, _, nn, _, _ = _import_torch()
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, proprio: Any, depth: Any) -> Any:
        torch, _, _, _, _ = _import_torch()
        depth_latent = self.depth_encoder(depth)
        return self.action_head(torch.cat([proprio, depth_latent], dim=-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill a trained tracking teacher into a depth-based student using true physics rollout."
        )
    )
    parser.add_argument("--teacher-checkpoint", required=True, help="Local checkpoint path or wandb:// checkpoint URI.")
    parser.add_argument("--num-envs", type=int, default=8192, help="Total env count across all GPUs.")
    parser.add_argument("--iterations", type=int, default=20000, help="Number of distillation rollout/update steps.")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint save interval in distill steps.")
    parser.add_argument("--logging-interval", type=int, default=25, help="Metric logging interval in distill steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--student-rollout-prob", type=float, default=0.0)
    parser.add_argument("--student-hidden-dims", type=int, nargs="+", default=[2048, 1024, 512, 256, 128])
    parser.add_argument("--depth-latent-dim", type=int, default=32)
    parser.add_argument("--depth-height", type=int, default=58)
    parser.add_argument("--depth-width", type=int, default=87)
    parser.add_argument("--depth-min-range", type=float, default=0.3)
    parser.add_argument("--depth-max-range", type=float, default=2.0)
    parser.add_argument("--raw-depth-height", type=int, default=60)
    parser.add_argument("--raw-depth-width", type=int, default=106)
    parser.add_argument("--depth-horizontal-fov-deg", type=float, default=101.41)
    parser.add_argument("--depth-camera-body-name", default=None)
    parser.add_argument("--depth-camera-debug-vis", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--project", default="holosomatest")
    parser.add_argument("--log-base-dir", default="logs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="zihanw22")
    parser.add_argument("--wandb-project", default="holosomatest")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _set_adaptive_sampler(config: ExperimentConfig, enabled: bool) -> ExperimentConfig:
    if config.command is None or "motion_command" not in config.command.setup_terms:
        return config

    setup_terms = dict(config.command.setup_terms)
    motion_term = setup_terms["motion_command"]
    params = dict(motion_term.params)
    motion_config = params.get("motion_config")
    if isinstance(motion_config, MotionConfig):
        params["motion_config"] = dataclasses.replace(
            motion_config,
            use_adaptive_timesteps_sampler=enabled,
        )
    elif isinstance(motion_config, dict):
        motion_config = dict(motion_config)
        motion_config["use_adaptive_timesteps_sampler"] = enabled
        params["motion_config"] = motion_config
    setup_terms["motion_command"] = dataclasses.replace(motion_term, params=params)
    return dataclasses.replace(config, command=dataclasses.replace(config.command, setup_terms=setup_terms))


def make_distill_config(saved_config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    depth_cfg = saved_config.simulator.config.depth_camera
    if args.depth_camera_body_name is not None:
        depth_body_name = args.depth_camera_body_name
    else:
        depth_body_name = depth_cfg.body_name

    depth_cfg = dataclasses.replace(
        depth_cfg,
        enabled=True,
        body_name=depth_body_name,
        width=args.raw_depth_width,
        height=args.raw_depth_height,
        horizontal_fov_deg=args.depth_horizontal_fov_deg,
        min_range=args.depth_min_range,
        max_range=args.depth_max_range,
        debug_vis=args.depth_camera_debug_vis,
    )
    simulator_cfg = dataclasses.replace(saved_config.simulator.config, depth_camera=depth_cfg)
    logger_cfg = dataclasses.replace(saved_config.logger, base_dir=args.log_base_dir)
    config = dataclasses.replace(
        saved_config,
        logger=logger_cfg,
        simulator=dataclasses.replace(saved_config.simulator, config=simulator_cfg),
        training=dataclasses.replace(
            saved_config.training,
            headless=True,
            num_envs=args.num_envs,
            export_onnx=False,
            seed=saved_config.training.seed if args.seed is None else args.seed,
            project=args.project,
            name=args.run_name or saved_config.training.name,
        ),
    )
    return _set_adaptive_sampler(config, enabled=False)


def get_actor_term_slices(env: Any, group_name: str = "actor_obs") -> dict[str, slice]:
    group_cfg = env.observation_manager.cfg.groups[group_name]
    if not group_cfg.concatenate:
        raise RuntimeError(f"Expected observation group '{group_name}' to be concatenated.")

    start = 0
    slices: dict[str, slice] = {}
    for term_name in sorted(group_cfg.terms):
        term_cfg = group_cfg.terms[term_name]
        term_obs = env.observation_manager._compute_term(group_name, term_name, term_cfg)
        term_dim = term_obs.reshape(env.num_envs, -1).shape[1]
        term_dim *= getattr(group_cfg, "history_length", 1)
        slices[term_name] = slice(start, start + term_dim)
        start += term_dim
    return slices


def _student_proprio_terms(term_slices: dict[str, slice]) -> list[str]:
    excluded_terms = {"height_scan", "depth_camera"}
    return [name for name in sorted(term_slices) if name not in excluded_terms]


def select_student_proprio(actor_obs: Any, term_slices: dict[str, slice]) -> Any:
    torch, _, _, _, _ = _import_torch()
    parts = [actor_obs[:, term_slices[name]] for name in _student_proprio_terms(term_slices)]
    if not parts:
        raise RuntimeError("No proprioceptive actor observation terms remain after excluding terrain terms.")
    return torch.cat(parts, dim=-1)


def _get_actor_obs_group(obs_dict: dict[str, Any]) -> Any:
    try:
        return obs_dict["actor_obs"]
    except KeyError as exc:
        raise RuntimeError("Depth student distillation expects an 'actor_obs' observation group.") from exc


def _concat_obs(obs_dict: dict[str, Any], keys: list[str]) -> Any:
    torch, _, _, _, _ = _import_torch()
    return torch.cat([obs_dict[key] for key in keys], dim=1)


def _actor_indices_and_dim(env: Any, sac_cfg: FastSACConfig) -> tuple[dict[str, dict[str, int]], int]:
    algo_obs_dim_dict = env.observation_manager.get_obs_dims()
    indices: dict[str, dict[str, int]] = {}
    offset = 0
    for obs_key in sac_cfg.actor_obs_keys:
        obs_dim = algo_obs_dim_dict[obs_key]
        if not isinstance(obs_dim, int):
            raise RuntimeError(f"FastSAC actor observation key '{obs_key}' resolved to non-flat dims: {obs_dim}")
        indices[obs_key] = {
            "start": offset,
            "end": offset + obs_dim,
            "size": obs_dim,
        }
        offset += obs_dim
    return indices, offset


def build_teacher_policy(env: Any, teacher_checkpoint: Path, sac_cfg: FastSACConfig, device: str):
    torch, _, nn, _, _ = _import_torch()
    wrapped_env = FastSACEnv(env, sac_cfg.actor_obs_keys, sac_cfg.critic_obs_keys)
    actor_obs_indices, actor_obs_dim = _actor_indices_and_dim(wrapped_env, sac_cfg)
    action_dim = wrapped_env.robot_config.actions_dim
    action_scale = wrapped_env._action_boundaries if sac_cfg.use_tanh else torch.ones(action_dim, device=device)
    action_bias = torch.zeros(action_dim, device=device)

    if sac_cfg.use_cnn_encoder:
        actor_obs_keys = [key for key in sac_cfg.actor_obs_keys if key != sac_cfg.encoder_obs_key]
        actor_cls = CNNActor
    else:
        actor_obs_keys = list(sac_cfg.actor_obs_keys)
        actor_cls = Actor

    actor = actor_cls(
        obs_indices=actor_obs_indices,
        obs_keys=actor_obs_keys,
        n_act=action_dim,
        num_envs=env.num_envs,
        device=device,
        hidden_dim=sac_cfg.actor_hidden_dim,
        log_std_max=sac_cfg.log_std_max,
        log_std_min=sac_cfg.log_std_min,
        use_tanh=sac_cfg.use_tanh,
        use_layer_norm=sac_cfg.use_layer_norm,
        action_scale=action_scale,
        action_bias=action_bias,
        encoder_obs_key=sac_cfg.encoder_obs_key,
        encoder_obs_shape=sac_cfg.encoder_obs_shape,
    )
    obs_normalizer: nn.Module
    if sac_cfg.obs_normalization:
        obs_normalizer = FastSACEmpiricalNormalization(shape=actor_obs_dim, device=device)
    else:
        obs_normalizer = nn.Identity()

    checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    if sac_cfg.obs_normalization:
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    actor.eval()
    obs_normalizer.eval()

    @torch.no_grad()
    def policy(obs_dict: dict[str, Any]) -> Any:
        actor_obs = _concat_obs(obs_dict, sac_cfg.actor_obs_keys).to(device=device, dtype=torch.float)
        normalized_obs = obs_normalizer(actor_obs, update=False) if sac_cfg.obs_normalization else actor_obs
        return actor(normalized_obs)[0]

    return policy, action_dim, sac_cfg.actor_obs_keys


def _ppo_obs_dim(env: Any, obs_keys: list[str]) -> int:
    algo_obs_dim_dict = env.observation_manager.get_obs_dims()
    obs_dim = 0
    for obs_key in obs_keys:
        key_dim = algo_obs_dim_dict[obs_key]
        if not isinstance(key_dim, int):
            raise RuntimeError(f"PPO actor observation key '{obs_key}' resolved to non-flat dims: {key_dim}")
        obs_dim += key_dim
    return obs_dim


def _ppo_history_length(env: Any) -> dict[str, int]:
    return {
        group_name: group_cfg.history_length
        for group_name, group_cfg in env.observation_manager.cfg.groups.items()
    }


def build_ppo_teacher_policy(env: Any, teacher_checkpoint: Path, ppo_cfg: PPOConfig, device: str):
    torch, _, nn, _, _ = _import_torch()
    obs_dim_dict = env.observation_manager.get_obs_dims()
    actor_obs_keys = list(ppo_cfg.module_dict.actor.input_dim)
    actor_obs_dim = _ppo_obs_dim(env, actor_obs_keys)
    action_dim = env.robot_config.actions_dim

    actor = setup_ppo_actor_module(
        obs_dim_dict=obs_dim_dict,
        module_config=copy.deepcopy(ppo_cfg.module_dict.actor),
        num_actions=action_dim,
        init_noise_std=ppo_cfg.init_noise_std,
        device=device,
        history_length=_ppo_history_length(env),
    )

    if ppo_cfg.empirical_normalization:
        obs_normalizer: nn.Module = PPOEmpiricalNormalization(shape=actor_obs_dim, device=device)
    else:
        obs_normalizer = nn.Identity()

    checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_model_state_dict"])
    if ppo_cfg.empirical_normalization and checkpoint.get("actor_obs_normalizer_state_dict") is not None:
        obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer_state_dict"])

    actor.eval()
    obs_normalizer.eval()

    @torch.no_grad()
    def policy(obs_dict: dict[str, Any]) -> Any:
        actor_obs = _concat_obs(obs_dict, actor_obs_keys).to(device=device, dtype=torch.float)
        normalized_obs = obs_normalizer(actor_obs, update=False) if ppo_cfg.empirical_normalization else actor_obs
        return actor.act_inference({"actor_obs": normalized_obs})

    return policy, action_dim, actor_obs_keys


def build_tracking_teacher_policy(env: Any, teacher_checkpoint: Path, algo_cfg: Any, device: str):
    if isinstance(algo_cfg, FastSACConfig):
        return build_teacher_policy(env, teacher_checkpoint, algo_cfg, device)
    if isinstance(algo_cfg, PPOConfig):
        return build_ppo_teacher_policy(env, teacher_checkpoint, algo_cfg, device)
    raise RuntimeError(f"Depth distillation does not support teacher algo config type: {type(algo_cfg)}")


def _distributed_mean(value: Any, distributed_conf: dict[str, int] | None) -> float:
    _, dist, _, _, _ = _import_torch()
    value_tensor = value.detach().float()
    if distributed_conf is not None:
        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
        value_tensor /= distributed_conf["world_size"]
    return float(value_tensor.item())


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def save_student_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    iteration: int,
    args: argparse.Namespace,
    config: ExperimentConfig,
    teacher_checkpoint: Path,
    proprio_dim: int,
    action_dim: int,
    depth_shape: tuple[int, int, int],
) -> None:
    torch, _, _, _, _ = _import_torch()
    module = _unwrap_model(model)
    payload = {
        "iteration": int(iteration),
        "student_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "teacher_checkpoint": str(teacher_checkpoint),
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "depth_shape": depth_shape,
        "args": vars(args),
        "experiment_config": config.to_serializable_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def export_student_onnx(
    path: Path,
    model: Any,
    proprio_dim: int,
    depth_shape: tuple[int, int, int],
    device: str,
) -> None:
    torch, _, _, _, _ = _import_torch()
    module = copy.deepcopy(_unwrap_model(model)).to(device)
    module.eval()
    dummy_proprio = torch.zeros(1, proprio_dim, device=device)
    dummy_depth = torch.zeros(1, *depth_shape, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (dummy_proprio, dummy_depth),
        str(path),
        input_names=["proprio", "depth"],
        output_names=["actions"],
        dynamic_axes={
            "proprio": {0: "batch"},
            "depth": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=17,
    )


def init_wandb(args: argparse.Namespace, log_dir: Path, config: ExperimentConfig, teacher_checkpoint: Path):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    import wandb

    wandb_dir = log_dir / ".wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.run_name,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config={
            "distill_args": vars(args),
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_experiment": config.to_serializable_dict(),
        },
        tags=["depth-student", "distillation", "physics-rollout"],
    )


def main() -> None:
    args = parse_args()
    torch, dist, nn, F, DistributedDataParallel = _import_torch()
    simulation_app = None
    wandb_run = None

    try:
        saved_config, saved_wandb_path = load_saved_experiment_config(CheckpointConfig(args.teacher_checkpoint))

        if args.run_name is None:
            teacher_name = Path(str(args.teacher_checkpoint)).stem
            args.run_name = f"{get_timestamp()}_depth_student_{teacher_name}"

        distill_config = make_distill_config(saved_config, args)
        simulation_app = init_sim_imports(distill_config)

        distributed_conf = configure_multi_gpu()
        device = get_device(distill_config.training, distributed_conf)
        is_distributed = distributed_conf is not None
        is_main_process = distributed_conf is None or distributed_conf["global_rank"] == 0
        world_size = distributed_conf["world_size"] if distributed_conf is not None else 1

        timestamp = get_timestamp()
        log_dir = get_experiment_dir(
            distill_config.logger,
            distill_config.training,
            timestamp=timestamp,
            task_name="depth-distill",
        )
        configure_logging(distributed_conf=distributed_conf, log_dir=log_dir)

        if is_distributed:
            per_rank_envs = distill_config.training.num_envs // world_size
            distill_config = dataclasses.replace(
                distill_config,
                training=dataclasses.replace(distill_config.training, num_envs=per_rank_envs),
            )
            logger.info(
                f"Distributed depth distillation: rank {distributed_conf['global_rank']} runs {per_rank_envs} envs "
                f"({args.num_envs} total)."
            )

        from holosoma.utils.common import seeding

        seed = distill_config.training.seed + (distributed_conf["global_rank"] if distributed_conf else 0)
        seeding(seed, torch_deterministic=distill_config.training.torch_deterministic)

        if is_main_process:
            log_dir.mkdir(parents=True, exist_ok=True)
            distill_config.save_config(str(log_dir / CONFIG_NAME))
            with open(log_dir / "distill_args.json", "w") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)
            logger.info(f"Saving depth distillation outputs to {log_dir}")
            if saved_wandb_path:
                logger.info(f"Teacher checkpoint originated from W&B run: {saved_wandb_path}")

        env = get_class(distill_config.env_class)(get_tyro_env_config(distill_config), device=device)
        teacher_checkpoint = load_checkpoint(args.teacher_checkpoint, str(log_dir))
        teacher_policy, action_dim, actor_obs_keys = build_tracking_teacher_policy(
            env,
            teacher_checkpoint,
            distill_config.algo.config,
            device,
        )

        obs_dict = env.reset_all()
        actor_obs = _get_actor_obs_group(obs_dict).to(device=device, dtype=torch.float)
        term_slices = get_actor_term_slices(env, "actor_obs")
        proprio = select_student_proprio(actor_obs, term_slices)
        proprio_dim = proprio.shape[1]
        depth_shape = (1, args.depth_height, args.depth_width)

        student = DepthStudentPolicy(
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            depth_shape=depth_shape,
            hidden_dims=args.student_hidden_dims,
            depth_latent_dim=args.depth_latent_dim,
        ).to(device)
        if is_distributed:
            student = DistributedDataParallel(student, device_ids=[distributed_conf["local_rank"]])

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

        if is_main_process:
            wandb_run = init_wandb(args, log_dir, distill_config, teacher_checkpoint)
            logger.info(
                f"Depth student dims: proprio={proprio_dim}, depth={depth_shape}, action={action_dim}, "
                f"student_rollout_prob={args.student_rollout_prob}"
            )
            logger.info(f"Student proprio terms: {_student_proprio_terms(term_slices)}")

        start_time = time.time()
        student.train()
        for iteration in range(1, args.iterations + 1):
            with torch.no_grad():
                teacher_actions = teacher_policy(obs_dict).to(device=device, dtype=torch.float)
                actor_obs = _get_actor_obs_group(obs_dict).to(device=device, dtype=torch.float)
                proprio = select_student_proprio(actor_obs, term_slices)
                depth = depth_camera_obs(
                    env,
                    min_range=args.depth_min_range,
                    max_range=args.depth_max_range,
                    resize_height=args.depth_height,
                    resize_width=args.depth_width,
                    flatten=False,
                ).to(device=device, dtype=torch.float)

            student_actions = student(proprio, depth)
            loss = F.mse_loss(student_actions, teacher_actions)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                rollout_actions = teacher_actions
                if args.student_rollout_prob > 0.0:
                    mask = (torch.rand(teacher_actions.shape[0], 1, device=device) < args.student_rollout_prob).float()
                    rollout_actions = teacher_actions * (1.0 - mask) + student_actions.detach() * mask
                obs_dict, rewards, dones, _extras = env.step({"actions": rollout_actions})

            if iteration % args.logging_interval == 0 or iteration == 1:
                loss_value = _distributed_mean(loss, distributed_conf)
                reward_value = _distributed_mean(rewards.mean(), distributed_conf)
                done_value = _distributed_mean(dones.float().mean(), distributed_conf)
                action_l1 = _distributed_mean(
                    (student_actions.detach() - teacher_actions).abs().mean(),
                    distributed_conf,
                )
                if is_main_process:
                    elapsed_s = time.time() - start_time
                    metrics = {
                        "global_step": iteration,
                        "distill/loss_mse": loss_value,
                        "distill/action_l1": action_l1,
                        "rollout/reward_mean": reward_value,
                        "rollout/done_rate": done_value,
                        "time/elapsed_s": elapsed_s,
                    }
                    logger.info(
                        f"iter={iteration:07d} loss={loss_value:.6f} "
                        f"action_l1={action_l1:.6f} reward={reward_value:.4f} done={done_value:.4f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=iteration)

            if is_main_process and args.save_interval > 0 and iteration % args.save_interval == 0:
                ckpt_path = log_dir / f"student_{iteration:07d}.pt"
                save_student_checkpoint(
                    ckpt_path,
                    student,
                    optimizer,
                    iteration,
                    args,
                    distill_config,
                    teacher_checkpoint,
                    proprio_dim,
                    action_dim,
                    depth_shape,
                )
                if args.export_onnx:
                    export_student_onnx(
                        log_dir / f"student_{iteration:07d}.onnx",
                        student,
                        proprio_dim,
                        depth_shape,
                        device,
                    )

        if is_main_process:
            final_path = log_dir / f"student_{args.iterations:07d}.pt"
            save_student_checkpoint(
                final_path,
                student,
                optimizer,
                args.iterations,
                args,
                distill_config,
                teacher_checkpoint,
                proprio_dim,
                action_dim,
                depth_shape,
            )
            if args.export_onnx:
                export_student_onnx(
                    log_dir / f"student_{args.iterations:07d}.onnx",
                    student,
                    proprio_dim,
                    depth_shape,
                    device,
                )
            logger.info(f"Saved final depth student checkpoint: {final_path}")

        if wandb_run is not None:
            wandb_run.finish()
        if is_distributed:
            dist.destroy_process_group()
    except Exception as exc:
        logger.error(f"Depth distillation failed: {exc}\n{traceback.format_exc()}")
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(1)
    finally:
        close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
