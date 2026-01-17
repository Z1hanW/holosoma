from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _resolve_output_dir() -> Path | None:
    out_dir = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_DIR")
    if not out_dir:
        return None
    path = Path(out_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_stride() -> int:
    stride_raw = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_STRIDE", "")
    if not stride_raw:
        return 1
    try:
        stride = int(stride_raw)
    except ValueError:
        return 1
    return max(1, stride)


def _resolve_video_enabled() -> bool:
    raw = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_VIDEO", "")
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_save_npy_enabled() -> bool:
    raw = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_SAVE_NPY", "")
    if not raw:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_save_png_enabled() -> bool:
    raw = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_SAVE_PNG", "")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_video_fps(env) -> float:
    raw = os.environ.get("HOLOSOMA_PREVIS_PERCEPTION_VIDEO_FPS", "")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    sim_config = env.simulator.simulator_config.sim
    return float(sim_config.fps / sim_config.control_decimation)


def replay_perception(tyro_config: ExperimentConfig) -> None:
    simulation_app = init_sim_imports(tyro_config)

    import torch
    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    if env.perception_manager is None:
        raise RuntimeError("Perception is disabled. Use perception:camera_depth_d435i_rendered (or similar).")

    output_dir = _resolve_output_dir()
    stride = _resolve_stride()
    video_enabled = _resolve_video_enabled()
    save_npy = _resolve_save_npy_enabled()
    save_png = _resolve_save_png_enabled()
    if output_dir is None and (video_enabled or save_npy or save_png):
        raise RuntimeError("Depth output requires HOLOSOMA_PREVIS_PERCEPTION_DIR to be set.")
    video_writer = None
    video_fps = _resolve_video_fps(env) if video_enabled else 0.0
    frame_idx = 0

    done = False
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]

        env.perception_manager.update()
        depth = env.perception_manager.get_camera_depth_map()[0].detach().cpu().numpy()
        if output_dir is not None and save_npy and frame_idx % stride == 0:
            np.save(output_dir / f"depth_{frame_idx:06d}.npy", depth)

        depth_rgb = None
        if save_png or video_enabled:
            depth_rgb = env._depth_to_rgb(depth)

        if output_dir is not None and save_png and frame_idx % stride == 0:
            import cv2  # noqa: PLC0415

            png_path = output_dir / f"depth_{frame_idx:06d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

        if video_enabled:
            if video_writer is None:
                import cv2  # noqa: PLC0415

                height, width = depth.shape[:2]
                video_path = output_dir / "depth_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))
            import cv2  # noqa: PLC0415

            video_writer.write(cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    if video_writer is not None:
        video_writer.release()

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_perception(tyro_cfg)


if __name__ == "__main__":
    main()
