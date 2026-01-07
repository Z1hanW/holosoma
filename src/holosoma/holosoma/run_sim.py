#!/usr/bin/env python3
"""
Simulation Runner Script

This script provides a direct simulation runner for holosoma with bridge support without training
or evaluation environments.
"""

import dataclasses
import math
import sys
import traceback
from pathlib import Path

import tyro
from loguru import logger

from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.sim_utils import DirectSimulation, setup_simulation_environment
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _coerce_triplet_args(argv: list[str]) -> list[str]:
    """Allow --flag x y z in addition to list literal syntax."""
    triplet_flags = {
        "--robot.init-state.pos",
        "--robot.init_state.pos",
        "--simulator.config.virtual-gantry.point",
        "--simulator.config.virtual_gantry.point",
    }

    def _is_number(token: str) -> bool:
        try:
            float(token)
        except ValueError:
            return False
        return True

    i = 0
    while i < len(argv):
        if argv[i] in triplet_flags:
            if i + 1 < len(argv) and argv[i + 1].lstrip().startswith("["):
                i += 1
                continue
            if i + 3 < len(argv):
                t1, t2, t3 = argv[i + 1 : i + 4]
                if _is_number(t1) and _is_number(t2) and _is_number(t3):
                    argv[i + 1 : i + 4] = [f"[{t1}, {t2}, {t3}]"]
                    i += 1
                    continue
        i += 1
    return argv


def _load_onnx_metadata(onnx_path: Path) -> dict:
    try:
        import json
        import onnx
    except ImportError as exc:  # pragma: no cover - dependency optional for run_sim
        logger.warning("ONNX support not available: {}", exc)
        return {}

    model = onnx.load(str(onnx_path))
    metadata: dict[str, object] = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except json.JSONDecodeError:
            metadata[prop.key] = prop.value
    return metadata


def _extract_motion_config(metadata: dict) -> dict | None:
    motion_cfg = metadata.get("motion_config")
    if isinstance(motion_cfg, dict):
        return motion_cfg

    exp_cfg = metadata.get("experiment_config")
    if not isinstance(exp_cfg, dict):
        return None

    motion_cfg = (
        exp_cfg.get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )
    return motion_cfg if isinstance(motion_cfg, dict) else None


def _resolve_motion_file(motion_file: str, onnx_path: Path) -> Path | None:
    motion_path = Path(motion_file).expanduser()
    if motion_path.is_file():
        return motion_path

    candidate = onnx_path.parent / motion_file
    if candidate.is_file():
        return candidate

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / motion_file
    if candidate.is_file():
        return candidate

    logger.warning("Motion file not found: {}", motion_file)
    return None


def _load_motion_first_frame_yaw(motion_path: Path) -> float | None:
    if motion_path.suffix.lower() != ".npz":
        logger.warning("Only .npz motion files are supported for yaw alignment: {}", motion_path)
        return None
    try:
        import numpy as np
        import torch
        from holosoma.utils.rotations import get_euler_xyz
    except ImportError as exc:  # pragma: no cover - dependency optional for run_sim
        logger.warning("Motion yaw alignment dependencies missing: {}", exc)
        return None

    with np.load(motion_path, allow_pickle=True) as data:
        if "body_quat_w" not in data:
            logger.warning("Motion file missing body_quat_w: {}", motion_path)
            return None
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)

    if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
        logger.warning("Unexpected body_quat_w shape {} in {}", body_quat_w.shape, motion_path)
        return None

    root_quat_xyzw = body_quat_w[0, 0, [1, 2, 3, 0]]  # wxyz -> xyzw
    quat_tensor = torch.tensor(root_quat_xyzw, dtype=torch.float32).unsqueeze(0)
    _, _, yaw = get_euler_xyz(quat_tensor, w_last=True)
    return float(yaw.squeeze(0))


def _apply_motion_init_from_onnx(config: RunSimConfig) -> RunSimConfig:
    if not config.motion_init_onnx:
        return config

    onnx_path = Path(config.motion_init_onnx).expanduser()
    if not onnx_path.is_file():
        logger.warning("ONNX file not found for motion init: {}", onnx_path)
        return config

    metadata = _load_onnx_metadata(onnx_path)
    motion_cfg = _extract_motion_config(metadata)
    if not motion_cfg:
        logger.warning("No motion_config found in ONNX metadata: {}", onnx_path)
        return config

    motion_file = motion_cfg.get("motion_file")
    if not motion_file:
        logger.warning("motion_config.motion_file missing in ONNX metadata: {}", onnx_path)
        return config

    motion_path = _resolve_motion_file(str(motion_file), onnx_path)
    if motion_path is None:
        return config

    motion_yaw = _load_motion_first_frame_yaw(motion_path)
    if motion_yaw is None:
        return config

    try:
        import torch
        from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz
    except ImportError as exc:  # pragma: no cover - dependency optional for run_sim
        logger.warning("Quaternion helpers unavailable for motion init: {}", exc)
        return config

    init_rot = torch.tensor(config.robot.init_state.rot, dtype=torch.float32).unsqueeze(0)
    init_roll, init_pitch, _ = get_euler_xyz(init_rot, w_last=True)
    yaw_tensor = torch.tensor(motion_yaw, dtype=torch.float32)
    new_quat = quat_from_euler_xyz(init_roll.squeeze(0), init_pitch.squeeze(0), yaw_tensor)
    new_rot = [float(value) for value in new_quat.squeeze(0).tolist()]

    init_state = dataclasses.replace(config.robot.init_state, rot=new_rot)
    robot_cfg = dataclasses.replace(config.robot, init_state=init_state)
    logger.info("Aligned init yaw to motion clip frame 0 ({:.1f} deg).", math.degrees(motion_yaw))
    return dataclasses.replace(config, robot=robot_cfg)


def run_simulation(config: RunSimConfig):
    """Run simulation with direct simulator control.

    This function provides direct access to the simulator for continuous simulation
    with bridge support using the DirectSimulation class.

    Parameters
    ----------
    config : RunSimConfig
        Configuration containing all simulation settings.
    """
    config = _apply_motion_init_from_onnx(config)

    # Auto-set device for GPU-accelerated backends if still on default CPU
    if config.device == "cpu":
        # Check if using Warp backend (requires CUDA)
        if hasattr(config.simulator.config, "mujoco_backend"):
            from holosoma.config_types.simulator import MujocoBackend  # noqa: PLC0415 -- deferred

            if config.simulator.config.mujoco_backend == MujocoBackend.WARP:
                logger.info("Auto-detected MuJoCo Warp backend - setting device to cuda:0")
                config = dataclasses.replace(config, device="cuda:0")

    config = dataclasses.replace(config, device=config.device)

    logger.info("Starting Holosoma Direct Simulation...")
    logger.info(f"Robot: {config.robot.asset.robot_type}")
    logger.info(f"Simulator: {config.simulator._target_}")
    logger.info(f"Terrain: {config.terrain.terrain_term.mesh_type} ({config.terrain.terrain_term.func})")

    try:
        # Use shared utils for setup
        env, device, simulation_app = setup_simulation_environment(config, device=config.device)

        # Create and run direct simulation using context manager for automatic clean-up
        with DirectSimulation(config, env, device, simulation_app) as sim:
            sim.run()

    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main function using tyro configuration with compositional subcommands."""
    # Initialize logging
    init_eval_logging()

    logger.info("Holosoma Direct Simulation Runner")
    logger.info("Compositional configuration via subcommands (like eval_agent.py)")

    # Parse configuration with tyro - same pattern as ExperimentConfig
    config = tyro.cli(
        RunSimConfig,
        args=_coerce_triplet_args(sys.argv[1:]),
        description="Run simulation with direct simulator control and bridge support.\n\n"
        "Usage: python -m holosoma.run_sim simulator:<sim> robot:<robot> terrain:<terrain>\n"
        "Examples:\n"
        "  python -m holosoma.run_sim # defaults \n"
        "  python -m holosoma.run_sim simulator:mujoco robot:t1_29dof_waist_wrist terrain:terrain_locomotion_plane\n"
        "  python -m holosoma.run_sim simulator:isaacgym robot:g1_29dof terrain:terrain_locomotion_mix",
        config=TYRO_CONIFG,
    )

    # Run simulation directly with parsed config
    run_simulation(config)


if __name__ == "__main__":
    main()
