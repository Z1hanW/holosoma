from __future__ import annotations

import dataclasses
import datetime
import os
from pathlib import Path
from typing import Any

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _capture_depth_frame_rgb(env: Any, env_id: int):
    import numpy as np

    depth = env.perception_manager.get_camera_depth_map()[env_id].detach().cpu().numpy()
    cfg = env.perception_manager.cfg
    near = float(getattr(cfg, "camera_near", 0.0) or 0.0)
    max_distance = float(getattr(cfg, "max_distance", getattr(cfg, "camera_far", 1.0)) or 1.0)
    camera_far = float(getattr(cfg, "camera_far", max_distance) or max_distance)
    far = float(min(max_distance, camera_far))
    if far <= near + 1.0e-6:
        far = near + 1.0

    # Match replay logging colormap with Viser depth image settings for faithful comparisons.
    colormap_mode = os.environ.get("VISER_DEPTH_COLORMAP", "fixed").strip().lower()
    flip_vertical = _is_truthy(os.environ.get("VISER_PERCEPTION_FLIP_VERTICAL"), default=False)
    try:
        from holosoma.utils.viser_live import (  # noqa: PLC0415
            _depth_to_rgb as _viser_depth_to_rgb,
        )
        from holosoma.utils.viser_live import (  # noqa: PLC0415
            _depth_to_rgb_fixed_range as _viser_depth_to_rgb_fixed_range,
        )

        if colormap_mode == "fixed":
            frame = _viser_depth_to_rgb_fixed_range(depth, near, far)
        else:
            frame = _viser_depth_to_rgb(depth, near, far)
    except Exception:
        if hasattr(env, "_depth_to_rgb"):
            frame = env._depth_to_rgb(depth)  # type: ignore[attr-defined]
        else:
            depth_safe = np.nan_to_num(depth, nan=far, posinf=far, neginf=0.0)
            depth_safe = np.clip(depth_safe, 0.0, far)
            normalized = depth_safe / far
            gray = np.clip((1.0 - normalized) * 255.0, 0.0, 255.0).astype(np.uint8)
            frame = np.repeat(gray[..., None], 3, axis=-1)

    if flip_vertical:
        frame = np.flipud(frame).copy()
    return frame, depth


def _init_replay_wandb(tyro_config: ExperimentConfig):
    enable_wandb = _is_truthy(os.environ.get("HOLOSOMA_REPLAY_WANDB_ENABLE"), default=False)
    if not enable_wandb:
        return None, None

    try:
        import wandb  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] W&B requested but wandb is unavailable: {exc}")
        return None, None

    logger_cfg = tyro_config.logger
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    training_name = getattr(tyro_config.training, "name", "replay")
    default_name = f"{timestamp}_{training_name}_depth_replay"

    mode = os.environ.get("HOLOSOMA_REPLAY_WANDB_MODE", getattr(logger_cfg, "mode", "online"))
    project = os.environ.get("HOLOSOMA_REPLAY_WANDB_PROJECT", None) or getattr(logger_cfg, "project", None)
    if project is None:
        project = tyro_config.training.project or "holosoma-depth-replay"
    run_name = os.environ.get("HOLOSOMA_REPLAY_WANDB_RUN_NAME", None) or getattr(logger_cfg, "name", None) or default_name
    entity = os.environ.get("HOLOSOMA_REPLAY_WANDB_ENTITY", None) or getattr(logger_cfg, "entity", None)
    group = os.environ.get("HOLOSOMA_REPLAY_WANDB_GROUP", None) or getattr(logger_cfg, "group", None)
    tags_env = os.environ.get("HOLOSOMA_REPLAY_WANDB_TAGS", "").strip()
    tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()] if tags_env else list(
        getattr(logger_cfg, "tags", ())
    )
    wandb_dir = os.environ.get("HOLOSOMA_REPLAY_WANDB_DIR", None) or getattr(logger_cfg, "dir", None)

    try:
        config_dict = dataclasses.asdict(tyro_config)
    except Exception:
        config_dict = {"training": str(getattr(tyro_config, "training", "unknown"))}

    wandb_kwargs = {
        "project": project,
        "name": run_name,
        "mode": mode,
        "config": config_dict,
    }
    if entity:
        wandb_kwargs["entity"] = entity
    if group:
        wandb_kwargs["group"] = group
    if tags:
        wandb_kwargs["tags"] = tags
    if wandb_dir:
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb_kwargs["dir"] = str(wandb_dir)

    try:
        run = wandb.init(**wandb_kwargs)
        return wandb, run
    except Exception as exc:  # pragma: no cover - network/auth dependent
        print(f"[WARN] Failed to initialize W&B run: {exc}")
        return None, None


def replay(tyro_config: ExperimentConfig):
    simulation_app = init_sim_imports(tyro_config)

    import torch
    import numpy as np

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)
    wandb, wandb_run = _init_replay_wandb(tyro_config)
    if wandb_run is not None:
        print(f"[INFO] W&B enabled: {wandb_run.project}/{wandb_run.name}")

    depth_log_every = max(1, int(os.environ.get("HOLOSOMA_REPLAY_WANDB_DEPTH_EVERY", "10")))
    depth_log_video = _is_truthy(os.environ.get("HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO"), default=True)
    depth_video_max_frames = max(1, int(os.environ.get("HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO_MAX_FRAMES", "1200")))
    env_id_max = max(0, int(getattr(env, "num_envs", 1)) - 1)
    depth_env_id = min(max(0, int(os.environ.get("HOLOSOMA_REPLAY_WANDB_ENV_ID", "0"))), env_id_max)
    depth_video_frames: list[np.ndarray] = []
    depth_log_failed = False

    done = False
    step = 0
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
        if getattr(env, "perception_manager", None) is not None:
            env.perception_manager.update()
        if wandb_run is not None and (step % depth_log_every == 0):
            try:
                frame_rgb, depth_map = _capture_depth_frame_rgb(env, depth_env_id)
                cfg = env.perception_manager.cfg
                near = float(getattr(cfg, "camera_near", 0.0) or 0.0)
                max_distance = float(getattr(cfg, "max_distance", getattr(cfg, "camera_far", float("nan"))) or float("nan"))
                camera_far = float(getattr(cfg, "camera_far", max_distance) or max_distance)
                far = float(min(max_distance, camera_far)) if np.isfinite(max_distance) else float(camera_far)

                finite = np.isfinite(depth_map)
                hit = finite & (depth_map < (far - 1.0e-6))
                valid = hit & (depth_map >= near)
                below_near = hit & (depth_map < near)

                if np.any(valid):
                    depth_min = float(np.min(depth_map[valid]))
                    depth_max = float(np.max(depth_map[valid]))
                else:
                    depth_min = float("nan")
                    depth_max = float("nan")

                finite_ratio = float(np.mean(finite))
                hit_ratio = float(np.mean(hit))
                valid_ratio = float(np.mean(valid))
                below_near_ratio = float(np.mean(below_near))
                wandb.log(
                    {
                        "Replay/depth_frame": wandb.Image(frame_rgb),
                        "Replay/depth_min_m": depth_min,
                        "Replay/depth_max_m": depth_max,
                        "Replay/depth_finite_ratio": finite_ratio,
                        "Replay/depth_valid_ratio": valid_ratio,
                        "Replay/depth_hit_ratio": hit_ratio,
                        "Replay/depth_below_near_ratio": below_near_ratio,
                        "Replay/step": step,
                    },
                    step=step,
                )
                if depth_log_video and len(depth_video_frames) < depth_video_max_frames:
                    depth_video_frames.append(frame_rgb)
            except Exception as exc:
                if not depth_log_failed:
                    print(f"[WARN] Failed to log depth frame to W&B: {exc}")
                    depth_log_failed = True
        step += 1

    if wandb_run is not None and depth_log_video and depth_video_frames:
        control_freq = 30.0
        try:
            sim_cfg = env.simulator.simulator_config.sim
            control_freq = float(sim_cfg.fps) / float(sim_cfg.control_decimation)
        except Exception:
            pass
        try:
            from holosoma.utils.video_utils import create_video

            save_dir = Path(os.environ.get("HOLOSOMA_REPLAY_WANDB_VIDEO_DIR", "logs/videos/replay_depth"))
            output_format = os.environ.get("HOLOSOMA_REPLAY_WANDB_VIDEO_FORMAT", "h264")
            create_video(
                video_frames=np.stack(depth_video_frames, axis=0).astype(np.uint8),
                fps=control_freq,
                save_dir=save_dir,
                output_format=output_format,
                wandb_logging=True,
                episode_id=0,
                wandb_key="Replay/depth_video",
            )
        except Exception as exc:
            print(f"[WARN] Failed to create/log replay depth video: {exc}")

    keep_open = _is_truthy(os.environ.get("HOLOSOMA_REPLAY_KEEP_OPEN"), default=False)
    if keep_open:
        print("[INFO] Replay finished. Keeping simulator open. Press Ctrl+C to exit.")
        try:
            while True:
                env.simulator.sim.step()
        except KeyboardInterrupt:
            pass

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
