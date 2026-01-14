from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.terrain import MeshType
from holosoma.perception import apply_perception_overrides
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


@dataclass(frozen=True)
class Sim2SimConfig:
    enabled: bool = False
    """Enable MuJoCo sim-to-sim workflow (export ONNX + build run_sim/run_policy commands)."""

    auto_launch: bool = False
    """If True, launch run_sim and run_policy as subprocesses."""

    simulator: str = "mujoco"
    """Simulator preset to use for run_sim (mujoco or mjwarp)."""

    interface: str = "lo"
    """Network interface for sim-to-sim bridge/inference."""

    use_joystick: bool = False
    """Enable joystick controls for sim-to-sim inference."""

    use_sim_time: bool | None = None
    """Override use_sim_time; defaults to True for WBT, False for locomotion."""

    rl_rate: float | None = None
    """Override policy inference rate (Hz). Defaults to training control rate."""

    inference_config: str | None = None
    """Override inference config (e.g., inference:g1-29dof-loco)."""

    model_path: str | None = None
    """Override ONNX path for sim-to-sim inference."""

    run_sim_args: str = ""
    """Extra args appended to run_sim.py (shell-style string)."""

    run_sim_robot: str | None = None
    """Override run_sim robot preset (e.g., g1-29dof-stairs)."""

    run_policy_args: str = ""
    """Extra args appended to run_policy.py (shell-style string)."""


@dataclass(frozen=True)
class EvalCliConfig:
    sim2sim: Sim2SimConfig = Sim2SimConfig()


def _get_export_paths(checkpoint_path: str) -> tuple[Path, Path]:
    checkpoint_file = Path(checkpoint_path)
    export_dir = checkpoint_file.parent / "exported"
    exported_onnx = export_dir / checkpoint_file.with_suffix(".onnx").name
    return export_dir, exported_onnx


def _is_wbt_experiment(config: ExperimentConfig) -> bool:
    if "wbt" in config.env_class.lower():
        return True
    if config.command is None:
        return False
    for term in config.command.setup_terms.values():
        func = term.func.lower()
        if "motioncommand" in func or "motion_command" in func:
            return True
    return False


def _infer_inference_config(config: ExperimentConfig) -> tuple[str, bool]:
    is_wbt = _is_wbt_experiment(config)
    robot_type = config.robot.asset.robot_type
    robot_map = {
        "g1_29dof": "g1-29dof",
        "t1_29dof": "t1-29dof",
    }
    base = robot_map.get(robot_type)
    if base is None:
        raise ValueError(f"Unsupported robot type for sim2sim inference: {robot_type}")
    if is_wbt and base != "g1-29dof":
        raise ValueError(
            f"No default WBT inference config for robot '{robot_type}'. "
            "Pass --sim2sim.inference-config explicitly."
        )
    suffix = "wbt" if is_wbt else "loco"
    return f"inference:{base}-{suffix}", is_wbt


def _resolve_run_sim_robot(config: ExperimentConfig) -> str:
    from holosoma.config_values import robot as robot_values

    robot_type = config.robot.asset.robot_type
    matches = [key for key, cfg in robot_values.DEFAULTS.items() if cfg.asset.robot_type == robot_type]
    if not matches:
        raise ValueError(f"No run_sim robot preset matches robot_type '{robot_type}'")
    if len(matches) > 1:
        xml_file = config.robot.asset.xml_file
        xml_matches = [key for key in matches if robot_values.DEFAULTS[key].asset.xml_file == xml_file]
        if len(xml_matches) == 1:
            return xml_matches[0]
        if robot_type in matches:
            return robot_type
        raise ValueError(
            f"Multiple run_sim robot presets match robot_type '{robot_type}': {matches}. "
            "Pass --sim2sim.run-sim-robot to pick one."
        )
    return matches[0]


def _resolve_run_sim_terrain_args(config: ExperimentConfig) -> list[str]:
    terrain_term = config.terrain.terrain_term
    if terrain_term.mesh_type == MeshType.LOAD_OBJ and terrain_term.obj_file_path:
        return [
            "terrain:terrain_load_obj",
            "--terrain.terrain-term.obj-file-path",
            terrain_term.obj_file_path,
        ]
    if terrain_term.mesh_type == MeshType.PLANE:
        return ["terrain:terrain_locomotion_plane"]
    if terrain_term.mesh_type == MeshType.TRIMESH:
        return ["terrain:terrain_locomotion_mix"]
    return []


def _build_sim2sim_commands(
    config: ExperimentConfig, sim2sim_cfg: Sim2SimConfig, model_path: str
) -> tuple[list[str], list[str]]:
    inference_cfg, is_wbt = _infer_inference_config(config)
    if sim2sim_cfg.inference_config:
        inference_cfg = sim2sim_cfg.inference_config

    use_sim_time = sim2sim_cfg.use_sim_time if sim2sim_cfg.use_sim_time is not None else is_wbt
    default_rl_rate = config.simulator.config.sim.fps / config.simulator.config.sim.control_decimation
    rl_rate = sim2sim_cfg.rl_rate if sim2sim_cfg.rl_rate is not None else default_rl_rate

    run_sim_robot = sim2sim_cfg.run_sim_robot or _resolve_run_sim_robot(config)
    if run_sim_robot.startswith("robot:"):
        run_sim_robot = run_sim_robot.split(":", 1)[1]

    run_sim_cmd = [
        sys.executable,
        "src/holosoma/holosoma/run_sim.py",
        f"simulator:{sim2sim_cfg.simulator}",
        f"robot:{run_sim_robot}",
    ]
    run_sim_cmd += _resolve_run_sim_terrain_args(config)
    if sim2sim_cfg.interface:
        run_sim_cmd += ["--simulator.config.bridge.interface", sim2sim_cfg.interface]
    if sim2sim_cfg.use_joystick:
        run_sim_cmd += ["--simulator.config.bridge.use-joystick", "True"]
    if sim2sim_cfg.run_sim_args:
        run_sim_cmd += shlex.split(sim2sim_cfg.run_sim_args)

    run_policy_cmd = [
        sys.executable,
        "src/holosoma_inference/holosoma_inference/run_policy.py",
        inference_cfg,
        "--task.model-path",
        model_path,
        "--task.interface",
        sim2sim_cfg.interface,
        "--task.rl-rate",
        str(rl_rate),
    ]
    if use_sim_time:
        run_policy_cmd.append("--task.use-sim-time")
    if sim2sim_cfg.use_joystick:
        run_policy_cmd.append("--task.use-joystick")
    if sim2sim_cfg.run_policy_args:
        run_policy_cmd += shlex.split(sim2sim_cfg.run_policy_args)

    return run_sim_cmd, run_policy_cmd


def _launch_sim2sim(run_sim_cmd: list[str], run_policy_cmd: list[str]) -> None:
    logger.info("Launching MuJoCo sim-to-sim processes...")
    run_sim_proc = subprocess.Popen(run_sim_cmd)
    try:
        run_policy_proc = subprocess.Popen(run_policy_cmd)
        run_policy_proc.wait()
    except KeyboardInterrupt:
        logger.info("Sim-to-sim interrupted by user.")
    finally:
        if run_sim_proc.poll() is None:
            run_sim_proc.terminate()


def run_eval_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
    sim2sim_cfg: Sim2SimConfig,
):
    tyro_config = apply_perception_overrides(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)
    export_dir, exported_onnx_path = _get_export_paths(checkpoint_path)

    if sim2sim_cfg.enabled:
        model_path = sim2sim_cfg.model_path
        if model_path is None and exported_onnx_path.exists():
            model_path = str(exported_onnx_path)

        if model_path is None:
            # Use shared simulation environment setup only if we need to export ONNX.
            env, device, simulation_app = setup_simulation_environment(tyro_config)

            algo_class = get_class(tyro_config.algo._target_)
            algo: BaseAlgo = algo_class(
                device=device,
                env=env,
                config=tyro_config.algo.config,
                log_dir=str(eval_log_dir),
                multi_gpu_cfg=None,
            )
            algo.setup()
            algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
            algo.load(checkpoint_path)

            os.makedirs(export_dir, exist_ok=True)
            if not hasattr(algo, "export"):
                raise AttributeError(
                    f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
                )
            algo.export(onnx_file_path=str(exported_onnx_path))  # type: ignore[attr-defined]
            logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

            if hasattr(env, "close"):
                env.close()
            if simulation_app:
                close_simulation_app(simulation_app)

            model_path = str(exported_onnx_path)

        run_sim_cmd, run_policy_cmd = _build_sim2sim_commands(tyro_config, sim2sim_cfg, model_path)
        logger.info("MuJoCo sim-to-sim commands:")
        logger.info("  run_sim:   " + " ".join(shlex.quote(part) for part in run_sim_cmd))
        logger.info("  run_policy:" + " ".join(shlex.quote(part) for part in run_policy_cmd))

        if sim2sim_cfg.auto_launch:
            _launch_sim2sim(run_sim_cmd, run_policy_cmd)
        return

    # Standard in-simulator evaluation
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(checkpoint_path)

    if tyro_config.training.export_onnx:
        os.makedirs(export_dir, exist_ok=True)
        if not hasattr(algo, "export"):
            raise AttributeError(
                f"{algo_class.__name__} is missing an `export` method required for ONNX export during evaluation."
            )
        algo.export(onnx_file_path=str(exported_onnx_path))  # type: ignore[attr-defined]
        logger.info(f"Exported policy as onnx to: {exported_onnx_path}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    eval_cli_cfg, remaining_args = tyro.cli(
        EvalCliConfig,
        args=remaining_args,
        return_unknown_args=True,
        add_help=False,
    )
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg, saved_cfg, saved_wandb_path, eval_cli_cfg.sim2sim)


if __name__ == "__main__":
    main()
