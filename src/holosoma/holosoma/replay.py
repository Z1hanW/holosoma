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
from holosoma.utils.rotations import quat_apply
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


def _replay_debug_paths() -> tuple[Path, Path]:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = Path(
        os.environ.get(
            "HOLOSOMA_REPLAY_STEP_DEBUG_CSV",
            f"/data/logs_new/replay_depth_debug/replay_step_debug_{timestamp}.csv",
        )
    )
    hits_dir = Path(
        os.environ.get(
            "HOLOSOMA_REPLAY_STEP_DEBUG_HITS_DIR",
            f"/data/logs_new/replay_depth_debug/replay_hits_{timestamp}",
        )
    )
    return csv_path, hits_dir


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
    step_debug_enable = _is_truthy(os.environ.get("HOLOSOMA_REPLAY_STEP_DEBUG"), default=False)
    step_dump_hits = _is_truthy(os.environ.get("HOLOSOMA_REPLAY_STEP_DEBUG_DUMP_HITS"), default=False)
    step_debug_fh = None
    step_debug_csv_path: Path | None = None
    step_hits_dir: Path | None = None
    if step_debug_enable:
        step_debug_csv_path, step_hits_dir = _replay_debug_paths()
        step_debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if step_dump_hits:
            assert step_hits_dir is not None
            step_hits_dir.mkdir(parents=True, exist_ok=True)
        step_debug_fh = step_debug_csv_path.open("w", encoding="utf-8", buffering=1)
        step_debug_fh.write(
            ",".join(
                [
                    "step",
                    "motion_step",
                    "depth_height",
                    "depth_width",
                    "depth_finite_ratio",
                    "depth_hit_ratio",
                    "depth_valid_ratio",
                    "depth_below_near_ratio",
                    "depth_min_valid",
                    "depth_max_valid",
                    "rays_total",
                    "rays_hit_valid",
                    "root_back_ratio",
                    "root_min_dot",
                    "cam_back_ratio",
                    "cam_min_dot",
                    "center_dot_root",
                    "center_dot_cam",
                    "root_x",
                    "root_y",
                    "root_z",
                    "torso_x",
                    "torso_y",
                    "torso_z",
                ]
            )
            + "\n"
        )
        print(f"[INFO] Replay step debug CSV: {step_debug_csv_path}")
        if step_dump_hits:
            print(f"[INFO] Replay step hit dumps: {step_hits_dir}")

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
        if step_debug_enable and step_debug_fh is not None and getattr(env, "perception_manager", None) is not None:
            try:
                pm = env.perception_manager
                if getattr(pm.cfg, "output_mode", "") == "camera_depth":
                    env_ids = torch.tensor([depth_env_id], device=env.device, dtype=torch.long)
                    depth_map_t = pm.get_camera_depth_map()[depth_env_id]

                    cfg = pm.cfg
                    near = float(getattr(cfg, "camera_near", 0.0) or 0.0)
                    max_distance = float(getattr(cfg, "max_distance", getattr(cfg, "camera_far", float("nan"))) or float("nan"))
                    camera_far = float(getattr(cfg, "camera_far", max_distance) or max_distance)
                    far = float(min(max_distance, camera_far)) if np.isfinite(max_distance) else float(camera_far)

                    finite = torch.isfinite(depth_map_t)
                    hit_depth = finite & (depth_map_t < (far - 1.0e-6))
                    valid_depth = hit_depth & (depth_map_t >= near)
                    below_near_depth = hit_depth & (depth_map_t < near)

                    depth_valid_vals = depth_map_t[valid_depth]
                    depth_min = float(depth_valid_vals.min().item()) if depth_valid_vals.numel() > 0 else float("nan")
                    depth_max = float(depth_valid_vals.max().item()) if depth_valid_vals.numel() > 0 else float("nan")

                    hit_mask = torch.empty((0,), device=env.device, dtype=torch.bool)
                    ray_starts = torch.empty((0, 3), device=env.device, dtype=torch.float32)
                    ray_dirs = torch.empty((0, 3), device=env.device, dtype=torch.float32)
                    ray_hits = torch.empty((0, 3), device=env.device, dtype=torch.float32)
                    dots_root = torch.empty((0,), device=env.device, dtype=torch.float32)
                    dots_cam = torch.empty((0,), device=env.device, dtype=torch.float32)
                    root_back_ratio = float("nan")
                    cam_back_ratio = float("nan")
                    root_min_dot = float("nan")
                    cam_min_dot = float("nan")
                    center_dot_root = float("nan")
                    center_dot_cam = float("nan")
                    ray_count = int(depth_map_t.numel())
                    rays_hit_valid = int(valid_depth.to(torch.int32).sum().item())

                    try:
                        sample = pm.get_camera_depth_ray_samples(env_ids, include_misses=False, return_rays=True)
                    except Exception:
                        sample = None

                    if sample is not None:
                        hit_mask = sample[1][0].to(torch.bool)
                        ray_starts = sample[2][0]
                        ray_dirs = sample[3][0]
                        ray_hits = sample[4][0]

                        ray_dirs_norm = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1.0e-6)
                        body_pos, body_quat = pm.get_camera_pose(
                            env_ids=env_ids,
                            apply_sensor_offset=False,
                            apply_pitch=False,
                        )
                        cam_forward = pm._get_camera_forward_axis(body_quat)[0]
                        cam_forward = cam_forward / torch.norm(cam_forward).clamp(min=1.0e-6)

                        root_quat = getattr(env, "base_quat", None)
                        if isinstance(root_quat, torch.Tensor) and root_quat.shape[0] > depth_env_id:
                            root_quat_env = root_quat[depth_env_id : depth_env_id + 1]
                            root_forward = pm._camera_ray_dirs_base.new_tensor([[1.0, 0.0, 0.0]])
                            root_forward = quat_apply(root_quat_env, root_forward, w_last=True)[0]
                        else:
                            root_forward = pm._camera_ray_dirs_base.new_tensor([1.0, 0.0, 0.0])
                        root_forward = root_forward / torch.norm(root_forward).clamp(min=1.0e-6)

                        dots_root = torch.sum(ray_dirs_norm * root_forward.unsqueeze(0), dim=-1)
                        dots_cam = torch.sum(ray_dirs_norm * cam_forward.unsqueeze(0), dim=-1)
                        root_back_ratio = float((dots_root <= 0.0).to(torch.float32).mean().item())
                        cam_back_ratio = float((dots_cam <= 0.0).to(torch.float32).mean().item())
                        root_min_dot = float(dots_root.min().item()) if dots_root.numel() > 0 else 1.0
                        cam_min_dot = float(dots_cam.min().item()) if dots_cam.numel() > 0 else 1.0

                        ray_count = int(ray_dirs_norm.shape[0])
                        rays_hit_valid = int(hit_mask.to(torch.int32).sum().item())
                        width = int(getattr(pm, "_camera_width", 0) or 0)
                        height = int(getattr(pm, "_camera_height", 0) or 0)
                        center_idx = 0
                        if width > 0 and height > 0 and (width * height) == ray_count:
                            center_idx = (height // 2) * width + (width // 2)
                        center_dot_root = float(dots_root[center_idx].item()) if ray_count > 0 else float("nan")
                        center_dot_cam = float(dots_cam[center_idx].item()) if ray_count > 0 else float("nan")

                    motion_step = -1
                    try:
                        motion_cmd = env.command_manager.get_state("motion_command")
                        motion_step = int(motion_cmd.time_steps[depth_env_id].item())
                    except Exception:
                        pass

                    root_pos = env.simulator.robot_root_states[depth_env_id, :3]
                    torso_pos = root_pos
                    try:
                        body_names = getattr(env, "body_names", None)
                        if body_names is not None and "torso_link" in body_names:
                            torso_idx = int(body_names.index("torso_link"))
                            torso_pos = env.simulator._rigid_body_pos[depth_env_id, torso_idx]
                    except Exception:
                        pass

                    step_debug_fh.write(
                        ",".join(
                            [
                                str(step),
                                str(motion_step),
                                str(int(depth_map_t.shape[0])),
                                str(int(depth_map_t.shape[1])),
                                f"{float(finite.to(torch.float32).mean().item()):.6f}",
                                f"{float(hit_depth.to(torch.float32).mean().item()):.6f}",
                                f"{float(valid_depth.to(torch.float32).mean().item()):.6f}",
                                f"{float(below_near_depth.to(torch.float32).mean().item()):.6f}",
                                f"{depth_min:.6f}",
                                f"{depth_max:.6f}",
                                str(ray_count),
                                str(rays_hit_valid),
                                f"{root_back_ratio:.6f}",
                                f"{root_min_dot:.6f}",
                                f"{cam_back_ratio:.6f}",
                                f"{cam_min_dot:.6f}",
                                f"{center_dot_root:.6f}",
                                f"{center_dot_cam:.6f}",
                                f"{float(root_pos[0].item()):.6f}",
                                f"{float(root_pos[1].item()):.6f}",
                                f"{float(root_pos[2].item()):.6f}",
                                f"{float(torso_pos[0].item()):.6f}",
                                f"{float(torso_pos[1].item()):.6f}",
                                f"{float(torso_pos[2].item()):.6f}",
                            ]
                        )
                        + "\n"
                    )

                    if step_dump_hits and step_hits_dir is not None:
                        np.savez_compressed(
                            step_hits_dir / f"step_{step:06d}.npz",
                            step=np.int32(step),
                            motion_step=np.int32(motion_step),
                            depth=depth_map_t.detach().cpu().numpy(),
                            hit_mask=hit_mask.detach().cpu().numpy(),
                            ray_starts=ray_starts.detach().cpu().numpy(),
                            ray_dirs=ray_dirs.detach().cpu().numpy(),
                            ray_hits=ray_hits.detach().cpu().numpy(),
                            dots_root=dots_root.detach().cpu().numpy(),
                            dots_cam=dots_cam.detach().cpu().numpy(),
                        )
            except Exception as exc:
                print(f"[WARN] Replay step debug failed at step={step}: {exc}")
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

            save_dir = Path(os.environ.get("HOLOSOMA_REPLAY_WANDB_VIDEO_DIR", "/data/logs_new/videos/replay_depth"))
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
                viser_live = getattr(env, "_viser_live", None)
                viewer_enabled = bool(viser_live is not None and getattr(viser_live, "enabled", False))
                if viewer_enabled and hasattr(env, "step_visualize_motion"):
                    env.simulator.sim.step()
                    done = env.step_visualize_motion(None)  # type: ignore[attr-defined]
                    if getattr(env, "perception_manager", None) is not None:
                        env.perception_manager.update()
                    if done:
                        play_control = getattr(viser_live, "_play_control", None)
                        if play_control is not None:
                            try:
                                play_control.value = False
                            except Exception:
                                pass
                        try:
                            viser_live._play_last_value = False
                        except Exception:
                            pass
                    continue

                env.simulator.sim.step()
                if getattr(env, "perception_manager", None) is not None:
                    env.perception_manager.update()
                if viewer_enabled:
                    viser_live.record_step()
        except KeyboardInterrupt:
            pass

    if wandb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    if step_debug_fh is not None:
        step_debug_fh.close()
        print(f"[INFO] Replay step debug CSV saved: {step_debug_csv_path}")

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
