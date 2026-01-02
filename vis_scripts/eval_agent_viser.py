from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.utils.config_utils import CONFIG_NAME  # noqa: E402
from holosoma.utils.eval_utils import (  # noqa: E402
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.sim_utils import (  # noqa: E402
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


@dataclass(frozen=True)
class ViserEvalConfig:
    port: int = 6060
    env_index: int = 0
    update_interval: int = 1
    show_meshes: bool = True
    add_grid: bool = True
    grid_size: float = 10.0


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ViserEvalViewer:
    def __init__(self, robot_config: RobotConfig, cfg: ViserEvalConfig) -> None:
        self.server = viser.ViserServer(port=cfg.port)
        self.robot_root = self.server.scene.add_frame("/robot", show_axes=False)

        urdf_path = _resolve_robot_urdf_path(robot_config)
        self.robot = ViserUrdf(self.server, urdf_or_path=urdf_path, root_node_name="/robot")
        self.robot.show_visual = cfg.show_meshes

        viser_joint_names = list(self.robot.get_actuated_joint_names())
        name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
        if missing:
            raise ValueError(f"Viser joints not found in robot config: {missing}")
        self.joint_order = [name_to_robot_idx[name] for name in viser_joint_names]

        if cfg.add_grid:
            self.server.scene.add_grid(
                "/grid",
                width=cfg.grid_size,
                height=cfg.grid_size,
                position=(0.0, 0.0, 0.0),
            )

        with self.server.gui.add_folder("Display"):
            show_meshes_cb = self.server.gui.add_checkbox("Show meshes", initial_value=cfg.show_meshes)

        @show_meshes_cb.on_update
        def _(_evt) -> None:
            self.robot.show_visual = bool(show_meshes_cb.value)

        logger.info(f"Viser server running on port {cfg.port}")

    def update_from_env(self, env, env_index: int) -> None:
        root_state = _to_numpy(env.simulator.robot_root_states[env_index])
        dof_pos = _to_numpy(env.simulator.dof_pos[env_index]).reshape(-1)

        root_pos = root_state[:3]
        root_quat_xyzw = root_state[3:7]
        root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]]

        self.robot_root.position = root_pos
        self.robot_root.wxyz = root_quat_wxyz
        self.robot.update_cfg(dof_pos[self.joint_order])


class ViserEnvWrapper:
    def __init__(self, env, viewer: ViserEvalViewer, cfg: ViserEvalConfig) -> None:
        self._env = env
        self._viewer = viewer
        self._env_index = int(cfg.env_index)
        self._update_interval = max(1, int(cfg.update_interval))
        self._step_count = 0

        if self._env_index < 0 or self._env_index >= getattr(env, "num_envs", 1):
            raise ValueError(
                f"env_index {self._env_index} is out of range for num_envs={getattr(env, 'num_envs', 'unknown')}"
            )

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _maybe_update(self) -> None:
        if self._step_count % self._update_interval == 0:
            self._viewer.update_from_env(self._env, self._env_index)
        self._step_count += 1

    def reset_all(self):
        obs = self._env.reset_all()
        self._step_count = 0
        self._viewer.update_from_env(self._env, self._env_index)
        return obs

    def reset(self, *args, **kwargs):
        if hasattr(self._env, "reset"):
            obs = self._env.reset(*args, **kwargs)
        else:
            obs = self._env.reset_all()
        self._step_count = 0
        self._viewer.update_from_env(self._env, self._env_index)
        return obs

    def step(self, *args, **kwargs):
        out = self._env.step(*args, **kwargs)
        self._maybe_update()
        return out


def run_eval_with_viser(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    viser_cfg: ViserEvalConfig,
) -> None:
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    viewer = ViserEvalViewer(tyro_config.robot, viser_cfg)
    wrapped_env = ViserEnvWrapper(env, viewer, viser_cfg)

    algo_class = get_class(tyro_config.algo._target_)
    algo = algo_class(
        device=device,
        env=wrapped_env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")

    if tyro_config.training.export_onnx:
        exported_onnx_path = os.path.join(exported_policy_dir_path, exported_onnx_name)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )

        algo.export(onnx_file_path=exported_onnx_path)  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    eval_cfg_overrides, remaining_args = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        return_unknown_args=True,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    viser_cfg = tyro.cli(
        ViserEvalConfig,
        args=remaining_args,
        description="Viser viewer configuration.",
    )
    run_eval_with_viser(eval_cfg_overrides, checkpoint_cfg, saved_cfg, saved_wandb_path, viser_cfg)


if __name__ == "__main__":
    main()
