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
    frame_idx = 0

    done = False
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]

        env.perception_manager.update()
        if output_dir is not None and frame_idx % stride == 0:
            depth = env.perception_manager.get_camera_depth_map()[0].detach().cpu().numpy()
            np.save(output_dir / f"depth_{frame_idx:06d}.npy", depth)

        frame_idx += 1

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_perception(tyro_cfg)


if __name__ == "__main__":
    main()
