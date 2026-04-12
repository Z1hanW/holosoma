#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "holosoma"))

import holosoma.config_values.logger
from holosoma.observation import apply_observation_overrides
from holosoma.perception import apply_perception_overrides
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from loguru import logger


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _normalize_wandb_checkpoint_arg(value: str) -> str:
    raw = str(value).strip()
    if raw.startswith("https://wandb.ai/"):
        match = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", raw)
        if not match:
            raise ValueError(f"Unsupported W&B run URL: {raw}")
        entity, project, run_id = match.groups()
        return f"wandb://{entity}/{project}/{run_id}/model_03999.pt"
    return raw


def _resolve_output_dir(checkpoint_arg: str, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    run_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", checkpoint_arg).strip("_")[:80] or "eval"
    path = ROOT / "eval_motion_success" / f"{_timestamp()}_{run_slug}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_motion_config(experiment_config: Any) -> Any:
    command_cfg = getattr(experiment_config, "command", None)
    if command_cfg is None:
        raise RuntimeError("Experiment has no command config.")
    setup_terms = getattr(command_cfg, "setup_terms", None)
    if not isinstance(setup_terms, dict):
        raise RuntimeError("Command config has no setup_terms dict.")
    for term_cfg in setup_terms.values():
        if isinstance(term_cfg, dict):
            func = str(term_cfg.get("func", "")).strip().lower()
            params = term_cfg.get("params")
        else:
            func = str(getattr(term_cfg, "func", "")).strip().lower()
            params = getattr(term_cfg, "params", None)
        if "motioncommand" not in func and "motion_command" not in func:
            continue
        if not isinstance(params, dict) or "motion_config" not in params:
            continue
        return params["motion_config"]
    raise RuntimeError("Failed to locate MotionCommand config in command.setup_terms.")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _override_perception_for_eval(cfg: Any) -> Any:
    if cfg is None:
        return None
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(
            cfg,
            camera_warp_edge_noise=False,
            camera_warp_enable_holes=False,
            camera_apply_sensor_noise=False,
        )
    _cfg_set(cfg, "camera_warp_edge_noise", False)
    _cfg_set(cfg, "camera_warp_enable_holes", False)
    _cfg_set(cfg, "camera_apply_sensor_noise", False)
    return cfg


def _force_eval_config(saved_config: Any, *, num_envs: int, headless: bool, max_episode_length_s: float | None) -> Any:
    config = saved_config.get_eval_config()
    config = apply_observation_overrides(config)
    config = apply_perception_overrides(config)
    perception_cfg = _override_perception_for_eval(getattr(config, "perception", None))
    teacher_perception_cfg = _override_perception_for_eval(getattr(config, "teacher_perception", None))
    critic_perception_cfg = _override_perception_for_eval(getattr(config, "critic_perception", None))
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            headless=bool(headless),
            num_envs=int(num_envs),
            enable_viser=False,
            debug=False,
            toy_mode=False,
        ),
        logger=holosoma.config_values.logger.disabled,
        perception=perception_cfg,
        teacher_perception=teacher_perception_cfg,
        critic_perception=critic_perception_cfg,
    )
    if int(num_envs) >= 256:
        physx_cfg = config.simulator.config.sim.physx
        target_patch_count = max(int(getattr(physx_cfg, "gpu_max_rigid_patch_count", 0) or 0), 2_097_152)
        target_contact_count = max(int(getattr(physx_cfg, "gpu_max_rigid_contact_count", 0) or 0), 67_108_864)
        config = dataclasses.replace(
            config,
            simulator=dataclasses.replace(
                config.simulator,
                config=dataclasses.replace(
                    config.simulator.config,
                    sim=dataclasses.replace(
                        config.simulator.config.sim,
                        physx=dataclasses.replace(
                            physx_cfg,
                            gpu_max_rigid_patch_count=target_patch_count,
                            gpu_max_rigid_contact_count=target_contact_count,
                        ),
                    ),
                ),
            ),
        )
        logger.info(
            "Raised PhysX GPU capacities for large eval: gpu_max_rigid_patch_count={}, gpu_max_rigid_contact_count={}",
            target_patch_count,
            target_contact_count,
        )
    if max_episode_length_s is not None:
        config = dataclasses.replace(
            config,
            simulator=dataclasses.replace(
                config.simulator,
                config=dataclasses.replace(
                    config.simulator.config,
                    sim=dataclasses.replace(
                        config.simulator.config.sim,
                        max_episode_length_s=float(max_episode_length_s),
                    ),
                ),
            ),
        )
    motion_cfg = _resolve_motion_config(config)
    _cfg_set(motion_cfg, "motion_clip_id", None)
    _cfg_set(motion_cfg, "motion_clip_name", None)
    _cfg_set(motion_cfg, "use_adaptive_timesteps_sampler", False)
    _cfg_set(motion_cfg, "start_at_timestep_zero_prob", 1.0)
    _cfg_set(motion_cfg, "start_at_timestep_zero_prob_end", None)
    _cfg_set(motion_cfg, "start_at_timestep_zero_prob_start_iter", None)
    _cfg_set(motion_cfg, "start_at_timestep_zero_prob_end_iter", None)
    _cfg_set(motion_cfg, "enable_default_pose_prepend", False)
    _cfg_set(motion_cfg, "default_pose_prepend_duration_s", 0.0)
    sparse_goal_cfg = _cfg_get(motion_cfg, "sparse_object_goal", None)
    if sparse_goal_cfg is not None:
        _cfg_set(sparse_goal_cfg, "enabled", True)
        _cfg_set(sparse_goal_cfg, "eval_external_goal_prob", 1.0)
        _cfg_set(sparse_goal_cfg, "eval_command_only_env_prob", 0.0)
    return config


def _build_motion_object_spec_json(config: Any, output_dir: Path) -> Any:
    motion_cfg = _resolve_motion_config(config)
    motion_file = _cfg_get(motion_cfg, "motion_file", None)
    if not motion_file:
        return config

    motion_path = Path(resolve_data_file_path(str(motion_file))).expanduser().resolve()
    if motion_path.is_file() and motion_path.suffix.lower() == ".npz":
        motion_files = [motion_path]
    elif motion_path.is_dir():
        motion_files = sorted(motion_path.glob("*.npz"))
    else:
        return config

    clip_entries: dict[str, dict[str, str]] = {}
    for clip_file in motion_files:
        with np.load(clip_file, allow_pickle=True) as data:
            object_urdf_path = str(data.get("object_urdf_path", "")).strip()
            if not object_urdf_path:
                continue
            object_name = str(data.get("object_name", "")).strip()
        clip_entries[clip_file.stem] = {
            "object_name": object_name,
            "object_urdf_path": str(Path(resolve_data_file_path(object_urdf_path)).expanduser().resolve()),
        }

    if not clip_entries:
        return config

    object_spec_path = output_dir / "motion_object_spec.json"
    object_spec_path.write_text(json.dumps({"clips": clip_entries}, indent=2), encoding="utf-8")

    robot_cfg = getattr(config, "robot", None)
    object_cfg = getattr(robot_cfg, "object", None) if robot_cfg is not None else None
    if object_cfg is not None:
        if dataclasses.is_dataclass(object_cfg):
            new_object_cfg = dataclasses.replace(object_cfg, object_urdf_path=str(object_spec_path))
        else:
            _cfg_set(object_cfg, "object_urdf_path", str(object_spec_path))
            new_object_cfg = object_cfg

        if dataclasses.is_dataclass(robot_cfg):
            new_robot_cfg = dataclasses.replace(robot_cfg, object=new_object_cfg)
        else:
            _cfg_set(robot_cfg, "object", new_object_cfg)
            new_robot_cfg = robot_cfg

        if dataclasses.is_dataclass(config):
            config = dataclasses.replace(config, robot=new_robot_cfg)
        else:
            setattr(config, "robot", new_robot_cfg)

    logger.info(
        "Generated eval object spec from motion bank: clips={}, unique_urdfs={}, path={}",
        len(clip_entries),
        len({entry['object_urdf_path'] for entry in clip_entries.values()}),
        object_spec_path,
    )
    return config


def _get_motion_command(env: Any) -> Any:
    cmd_mgr = getattr(env, "command_manager", None)
    if cmd_mgr is None:
        raise RuntimeError("Environment has no command_manager.")
    motion_cmd = cmd_mgr.get_state("motion_command")
    if motion_cmd is None:
        raise RuntimeError("motion_command state is unavailable.")
    return motion_cmd


def _require_per_asset_object_mapping(motion_cmd: Any) -> None:
    if not bool(getattr(motion_cmd, "_multi_object_enabled", False)):
        sim_names = list(getattr(motion_cmd, "_sim_object_names", []) or [])
        raise RuntimeError(
            "Per-motion eval requires per-asset simulator objects, but runtime is still using single-slot mapping. "
            f"sim_object_names={sim_names}"
        )


def _build_clip_batches(total_clips: int, batch_size: int) -> list[list[int]]:
    return [list(range(start, min(start + batch_size, total_clips))) for start in range(0, total_clips, batch_size)]


def _assign_fixed_clips(motion_cmd: Any, clip_indices: list[int], num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not clip_indices:
        raise ValueError("clip_indices must not be empty.")
    fixed = torch.full((num_envs,), int(clip_indices[0]), device=motion_cmd.device, dtype=torch.long)
    active = torch.zeros((num_envs,), device=motion_cmd.device, dtype=torch.bool)
    for env_id, clip_idx in enumerate(clip_indices):
        fixed[env_id] = int(clip_idx)
        active[env_id] = True
    motion_cmd._fixed_clip_ids = fixed
    motion_cmd._forced_clip_idx = None
    motion_cmd._forced_start_step = None
    return fixed, active


def _policy_step(algo: Any, actor_state: dict[str, Any]) -> dict[str, Any]:
    actor_state = algo._pre_eval_env_step(actor_state)
    actor_state = algo.env_step(actor_state)
    actor_state = algo._post_eval_env_step(actor_state)
    return actor_state


def _run_batch(
    algo: Any,
    clip_indices: list[int],
    *,
    max_steps: int,
    success_term_name: str,
) -> list[dict[str, Any]]:
    env = algo.env
    motion_cmd = _get_motion_command(env)
    num_envs = int(env.num_envs)
    fixed_clip_ids, active_env_mask = _assign_fixed_clips(motion_cmd, clip_indices, num_envs)

    obs_dict = env.reset_all()
    actor_state = algo._create_actor_state()
    actor_state.update(
        {
            "obs": obs_dict,
            "actions": torch.zeros(env.num_envs, algo.num_act, device=algo.device),
        }
    )

    done_once = torch.zeros((num_envs,), device=env.device, dtype=torch.bool)
    results_by_env: dict[int, dict[str, Any]] = {}

    for step in range(int(max_steps)):
        current_clip_ids = motion_cmd.clip_ids.clone()
        actor_state["step"] = step
        actor_state = _policy_step(algo, actor_state)

        done_mask = actor_state["dones"].to(dtype=torch.bool)
        if not torch.any(done_mask):
            continue

        success_mask = env.termination_manager.get_last_term_result(success_term_name)
        if success_mask is None:
            raise RuntimeError(f"Termination term '{success_term_name}' is not present in the environment.")

        time_outs = actor_state["extras"].get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(done_mask)
        time_outs = time_outs.to(dtype=torch.bool)

        valid_done = done_mask & active_env_mask & (~done_once)
        if not torch.any(valid_done):
            continue

        for env_id_tensor in valid_done.nonzero(as_tuple=False).view(-1):
            env_id = int(env_id_tensor.item())
            clip_idx = int(current_clip_ids[env_id].item())
            results_by_env[env_id] = {
                "env_id": env_id,
                "clip_idx": clip_idx,
                "clip_name": str(motion_cmd.motion.clip_ids[clip_idx]),
                "success": bool(success_mask[env_id].item()),
                "done_step": int(step),
                "timed_out": bool(time_outs[env_id].item()),
            }
        done_once |= valid_done
        if bool(torch.all(done_once[active_env_mask]).item()):
            break

    for env_id, is_active in enumerate(active_env_mask.detach().cpu().tolist()):
        if not is_active or env_id in results_by_env:
            continue
        clip_idx = int(fixed_clip_ids[env_id].item())
        results_by_env[env_id] = {
            "env_id": env_id,
            "clip_idx": clip_idx,
            "clip_name": str(motion_cmd.motion.clip_ids[clip_idx]),
            "success": False,
            "done_step": None,
            "timed_out": False,
            "incomplete": True,
        }

    return [results_by_env[idx] for idx in sorted(results_by_env)]


def _write_outputs(
    output_dir: Path,
    checkpoint: str,
    num_envs: int,
    batch_size: int,
    max_steps: int,
    all_results: list[dict[str, Any]],
) -> None:
    total = len(all_results)
    success_count = sum(1 for row in all_results if row.get("success"))
    incomplete_count = sum(1 for row in all_results if row.get("incomplete"))
    summary = {
        "checkpoint": checkpoint,
        "num_envs": int(num_envs),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "num_clips": total,
        "num_success": success_count,
        "success_rate": (float(success_count) / float(total)) if total > 0 else 0.0,
        "num_incomplete": incomplete_count,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "per_motion_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    with (output_dir / "per_motion_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_idx", "clip_name", "success", "done_step", "timed_out", "incomplete", "env_id"],
        )
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    summary_md_lines = [
        "# Motion Success Summary",
        "",
        "| checkpoint | num_envs | batch_size | max_steps | num_clips | num_success | success_rate | num_incomplete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{checkpoint}` | {int(num_envs)} | {int(batch_size)} | {int(max_steps)} | "
            f"{total} | {success_count} | {summary['success_rate']:.4f} | {incomplete_count} |"
        ),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_md_lines), encoding="utf-8")

    def _table_lines(rows: list[dict[str, Any]], title: str) -> list[str]:
        lines = [
            f"# {title}",
            "",
            "| clip_idx | clip_name | success | done_step | timed_out | incomplete | env_id |",
            "| ---: | --- | --- | ---: | --- | --- | ---: |",
        ]
        for row in rows:
            done_step = "" if row.get("done_step") is None else str(int(row["done_step"]))
            lines.append(
                "| {clip_idx} | `{clip_name}` | {success} | {done_step} | {timed_out} | {incomplete} | {env_id} |".format(
                    clip_idx=int(row["clip_idx"]),
                    clip_name=str(row["clip_name"]),
                    success="true" if row.get("success") else "false",
                    done_step=done_step,
                    timed_out="true" if row.get("timed_out") else "false",
                    incomplete="true" if row.get("incomplete") else "false",
                    env_id=int(row["env_id"]),
                )
            )
        lines.append("")
        return lines

    sorted_rows = sorted(all_results, key=lambda row: int(row["clip_idx"]))
    failed_rows = [row for row in sorted_rows if not row.get("success")]
    success_rows = [row for row in sorted_rows if row.get("success")]
    (output_dir / "per_motion_results.md").write_text(
        "\n".join(_table_lines(sorted_rows, "Per-Motion Results")),
        encoding="utf-8",
    )
    (output_dir / "failed_motions.md").write_text(
        "\n".join(_table_lines(failed_rows, "Failed Motions")),
        encoding="utf-8",
    )
    (output_dir / "successful_motions.md").write_text(
        "\n".join(_table_lines(success_rows, "Successful Motions")),
        encoding="utf-8",
    )

    logger.info(
        "Eval summary: success_rate={:.4f} ({}/{}), incomplete={}",
        summary["success_rate"],
        success_count,
        total,
        incomplete_count,
    )
    logger.info("Saved summary to {}", output_dir / "summary.json")
    logger.info("Saved per-motion CSV to {}", output_dir / "per_motion_results.csv")
    logger.info("Saved markdown tables to {}", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-motion sparse-goal success with batched fixed-clip rollout.")
    parser.add_argument("checkpoint", help="checkpoint path, wandb://..., or https://wandb.ai/.../runs/...")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel envs to launch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Clips per batch. Defaults to num_envs.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum policy steps per batch before unfinished clips are marked incomplete.",
    )
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless.")
    parser.add_argument("--device", default=None, help="Override HOLOSOMA_DEVICE, e.g. cuda:0.")
    parser.add_argument(
        "--success-term",
        default="sparse_goal_success",
        help="Termination term name used as the success mask.",
    )
    parser.add_argument(
        "--max-episode-length-s",
        type=float,
        default=None,
        help="Optional override for simulator max_episode_length_s during eval.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to store JSON/CSV outputs.")
    args = parser.parse_args()

    init_eval_logging()

    checkpoint_arg = _normalize_wandb_checkpoint_arg(args.checkpoint)
    output_dir = _resolve_output_dir(checkpoint_arg, args.output_dir)

    if args.device:
        os.environ["HOLOSOMA_DEVICE"] = str(args.device)
    os.environ["HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT"] = "1"
    os.environ["HOLOSOMA_OBJECT_SPAWN_MODE"] = "urdf"
    os.environ.pop("HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT", None)
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    os.environ.setdefault("PY_LOG_LEVEL", "INFO")

    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_arg)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    config = _force_eval_config(
        saved_cfg,
        num_envs=int(args.num_envs),
        headless=bool(args.headless),
        max_episode_length_s=args.max_episode_length_s,
    )
    config = _build_motion_object_spec_json(config, output_dir)

    checkpoint_path = load_checkpoint(checkpoint_arg, str(output_dir))
    env = None
    simulation_app = None

    try:
        env, device, simulation_app = setup_simulation_environment(config)
        algo_class = get_class(config.algo._target_)
        algo = algo_class(
            device=device,
            env=env,
            config=config.algo.config,
            log_dir=str(output_dir),
            multi_gpu_cfg=None,
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)
        algo.load(str(checkpoint_path))
        algo._eval_mode()
        env.set_is_evaluating()
        algo.eval_policy = algo.get_inference_policy()

        motion_cmd = _get_motion_command(env)
        _require_per_asset_object_mapping(motion_cmd)
        total_clips = int(motion_cmd.motion.num_clips)
        batch_size = int(args.batch_size or args.num_envs)
        batch_size = max(1, min(batch_size, int(args.num_envs)))
        clip_batches = _build_clip_batches(total_clips, batch_size)

        logger.info(
            "Running motion success eval: clips={}, num_envs={}, batch_size={}, max_steps={}, start_mode=t0_per_motion, single_slot_disabled=1",
            total_clips,
            args.num_envs,
            batch_size,
            args.max_steps,
        )

        all_results: list[dict[str, Any]] = []
        for batch_idx, clip_batch in enumerate(clip_batches):
            logger.info(
                "Batch {}/{}: evaluating clips [{}..{}] ({} clips)",
                batch_idx + 1,
                len(clip_batches),
                clip_batch[0],
                clip_batch[-1],
                len(clip_batch),
            )
            batch_results = _run_batch(
                algo,
                clip_batch,
                max_steps=int(args.max_steps),
                success_term_name=str(args.success_term),
            )
            logger.info(
                "Batch {}/{} completed with {} result rows.",
                batch_idx + 1,
                len(clip_batches),
                len(batch_results),
            )
            all_results.extend(batch_results)

        all_results.sort(key=lambda row: int(row["clip_idx"]))
        logger.info("Completed all batches. Writing {} result rows.", len(all_results))
        _write_outputs(
            output_dir,
            checkpoint_arg,
            int(args.num_envs),
            batch_size,
            int(args.max_steps),
            all_results,
        )
    except Exception:
        logger.exception("Motion success eval failed.")
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
