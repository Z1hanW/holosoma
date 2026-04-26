"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.perception import PerceptionManager
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch


def _parse_debug_float_list_env(name: str, *, expected_len: int) -> list[float] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    text = raw
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{name} expected {expected_len} comma-separated floats, got: {raw}")
    return [float(part) for part in parts]


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    print("\n\n\nsimulator type: ", config.simulator)
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco  # noqa: PLC0415

        assert mujoco is not None
    elif simulator_type == SimulatorType.ISAACGYM:
        import isaacgym  # noqa: PLC0415

        assert isaacgym is not None

    # IsaacSim imports handled in setup_isaaclab_launcher


def setup_isaaclab_launcher(config: ExperimentConfig | RunSimConfig, device: str | None = None) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
    parser.add_argument("--output_dir", type=str, default="/data/logs_new", help="Directory to store the output.")
    AppLauncher.add_app_launcher_args(parser)

    # Parse known arguments to get argparse params
    args_cli, unknown_args = parser.parse_known_args()

    # Set values from config
    args_cli.num_envs = config.training.num_envs
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.simulator.config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        args_cli.device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    # Check if video recording is enabled and add --enable_cameras flag
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    perception_cfg = getattr(config, "perception", None)
    if perception_cfg is not None:
        if getattr(perception_cfg, "enabled", False) and getattr(perception_cfg, "camera_source", "") in {
            "rendered",
            "rendered_depth_sensor",
        }:
            video_enabled = True
    if video_enabled:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    logger.info(f"IsaacSim args_cli: {args_cli}")
    logger.info(f"IsaacSim unknown_args: {unknown_args}")
    sys.argv = [sys.argv[0]] + unknown_args

    return simulation_app


def setup_keyboard_listener(env) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key, env):
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env):
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard  # noqa: PLC0415

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig, device: str | None = None
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup
    if device is None:
        device = os.environ.get("HOLOSOMA_DEVICE")
    if device is None or not str(device).strip():
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(config, device)

    # Set random seed if specified. Direct RunSimConfig also carries a TrainingConfig,
    # and sim2sim rollouts need deterministic terrain/contact initialization.
    training_cfg = getattr(config, "training", None)
    seed = getattr(training_cfg, "seed", None)
    if seed is not None:
        seeding(seed, torch_deterministic=getattr(training_cfg, "torch_deterministic", False))
        logger.info(f"Seed: {seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Create FullSimConfig from RunSimConfig
        # Extract SimulatorInitConfig from SimulatorConfig
        full_config = FullSimConfig(
            simulator=config.simulator.config,  # Extract .config from SimulatorConfig
            robot=config.robot,
            training=config.training,
            logger=config.logger,
            command=None,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env
        class EnvProxy:
            def __init__(self, device):
                self.num_envs = 1
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator):
                self.sim = simulator

            def reset(self):
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self):
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_manager = TerrainManager(config.terrain, env=EnvProxy(device), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env, device, simulation_app


def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        logger.info("Shutting down simulation app...")
        try:
            # Work-around for IsaacLab hanging headless.
            # Patch the close_stage method to avoid hanging
            import omni.usd  # noqa: PLC0415

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(self, config: RunSimConfig, env: Any, device: str, simulation_app: Any):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim
        self._perception_env_proxy: _DirectPerceptionEnvProxy | None = None
        self._perception_manager: PerceptionManager | None = None

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        self.cleanup()

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Need to manually set headless since it's stored in training config.
        self.simulator.set_headless(bool(self.config.training.headless))

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Create environments (need to provide required parameters)
        # Create env_origins (single environment at origin)
        env_origins = torch.zeros(1, 3, device=self.device)

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")

        # Step 5.5: Initialize episode (positions virtual gantry, etc.)
        self.simulator.on_episode_start(env_id=0)
        logger.debug("simulator.on_episode_start() completed")

        # Optional clip-driven initialization for split sim2sim verification.
        self._maybe_apply_motion_initial_state()
        self._maybe_setup_split_sim_perception()

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        logger.info("Simulator initialized")

        # Step 7: Toggle start recording if enabled
        if self.simulator.video_recorder and self.simulator.video_recorder.enabled:
            # arbitrary episode ID given this is sim2sim, we may want to
            # actually support toggling recording and with better filenames too
            self.simulator.video_recorder.start_recording(episode_id=0)

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and FPS logging.

        Manages the complete simulation loop including rate limiting,
        viewer synchronization, FPS logging, and error handling.
        """
        # Setup rate limiting
        sim_frequency = self.config.simulator.config.sim.fps
        rate_limiter = RateLimiter(sim_frequency)

        # Calculate viewer sync frequency
        viewer_steps = self._calculate_viewer_steps()
        should_render = not self.config.training.headless

        logger.info(f"Simulation rate: {sim_frequency} Hz ({1.0 / sim_frequency * 1000:.2f} ms)")
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")
        logger.info("Starting direct simulation loop...")
        logger.info("Press Ctrl+C to stop simulation")

        # Determine refresh strategy based on simulator type
        # IsaacGym/IsaacSim: need pre-step to refresh tensors to sync simulator state
        # MuJoCo direct split-sim still needs fresh rigid body tensors before the bridge
        # publishes sim-state/perception observations.
        simulator_type = get_simulator_type()
        if simulator_type in [SimulatorType.ISAACGYM, SimulatorType.ISAACSIM]:
            pre_step_refresh = self.simulator.refresh_sim_tensors
        elif simulator_type == SimulatorType.MUJOCO and self._perception_manager is not None:
            pre_step_refresh = self.simulator.refresh_sim_tensors
        else:
            pre_step_refresh = lambda: None  # noqa: E731  (No-op for MuJoCo)

        # Direct simulation loop (like holosoma_inference's simulation_thread)
        step_count = 0
        start_time = time.time()
        fps_start_time = start_time

        while True:
            try:
                # Refresh tensors if needed (no-op for MuJoCo)
                pre_step_refresh()

                # Direct simulator step - this triggers bridge.step() inside simulate_at_each_physics_step()
                self.simulator.simulate_at_each_physics_step()

                # Update viewer at display rate
                if should_render and step_count % viewer_steps == 0:
                    self.simulator.render()

                # Periodic FPS logging (every 1000 steps)
                if step_count > 0 and step_count % 1000 == 0:
                    fps_start_time = self._log_fps(step_count, fps_start_time)

                step_count += 1
                rate_limiter.sleep()

            except KeyboardInterrupt:  # noqa: PERF203
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during simulation step {step_count}: {e}")
                traceback.print_exc()
                break

        # Final statistics
        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        if hasattr(self.simulator, "_split_sim_perception_provider"):
            self.simulator._split_sim_perception_provider = None

        # Cleanup environment
        if hasattr(self.env, "close"):
            self.env.close()

        if self.simulator.video_recorder:
            self.simulator.video_recorder.cleanup()

        # Cleanup simulation app
        if self.simulation_app:
            close_simulation_app(self.simulation_app)

    @staticmethod
    def _decode_names(values: np.ndarray) -> list[str]:
        decoded: list[str] = []
        for item in values.tolist():
            if isinstance(item, (bytes, bytearray, np.bytes_)):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    @staticmethod
    def _resolve_root_body_index(body_names: list[str]) -> int:
        for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
            if candidate in body_names:
                return body_names.index(candidate)
        for idx, name in enumerate(body_names):
            if name.lower() != "world":
                return idx
        return 0

    def _maybe_setup_split_sim_perception(self) -> None:
        bridge_cfg = self.config.simulator.config.bridge
        wants_publish = bool(getattr(bridge_cfg, "publish_perception_obs", False)) or bool(
            getattr(bridge_cfg, "publish_perception_obs_shm", False)
        )
        if not self.config.perception.enabled:
            if wants_publish:
                raise ValueError(
                    "split sim2sim perception publishing requires a perception config, "
                    "e.g. perception:camera_depth_d435i"
                )
            return

        perception_cfg = self.config.perception
        device_type = torch.device(str(self.device)).type
        if (
            get_simulator_type() == SimulatorType.MUJOCO
            and perception_cfg.output_mode == "camera_depth"
            and perception_cfg.camera_source == "far_tracking_warp"
            and device_type != "cuda"
        ):
            strict_source = os.environ.get("HOLOSOMA_STRICT_PERCEPTION_CAMERA_SOURCE", "").strip().lower()
            if strict_source in {"1", "true", "yes", "on"}:
                raise RuntimeError(
                    "MuJoCo split perception requested camera_source=far_tracking_warp, "
                    f"but the simulator is running on device={self.device}. "
                    "The bundled far-tracking warp sensor requires CUDA; set SIM_DEVICE=cuda:0 "
                    "or explicitly choose PERCEPTION_CAMERA_SOURCE=rendered for the approximate renderer path."
                )
            logger.warning(
                "Split sim perception requested camera_source=far_tracking_warp on device={} for MuJoCo; "
                "falling back to camera_source=rendered because the bundled warp sensor requires CUDA.",
                self.device,
            )
            perception_cfg = replace(perception_cfg, camera_source="rendered")

        self._perception_env_proxy = _DirectPerceptionEnvProxy(
            simulator=self.simulator,
            terrain_manager=self.simulator.terrain_manager,
            robot_config=self.config.robot,
            dt=float(self.simulator.sim_dt),
            device=self.device,
        )
        self._perception_manager = PerceptionManager(perception_cfg, self._perception_env_proxy, self.device)
        self._perception_manager.setup()
        self._perception_manager.reset()
        self.simulator.refresh_sim_tensors()
        self._perception_manager.update()
        logger.info(
            "Split sim perception initialized: mode={} camera_source={}",
            perception_cfg.output_mode,
            perception_cfg.camera_source,
        )

        if wants_publish:
            self.simulator._split_sim_perception_provider = self._get_split_sim_perception_obs

    def _get_split_sim_perception_obs(self) -> list[float] | None:
        if self._perception_manager is None:
            return None

        self.simulator.refresh_sim_tensors()
        self._perception_manager.update()
        perception_obs = self._perception_manager.get_obs()
        if perception_obs.ndim != 2 or perception_obs.shape[0] < 1:
            raise RuntimeError(f"Unexpected perception observation shape: {tuple(perception_obs.shape)}")
        return perception_obs[0].detach().cpu().to(torch.float32).tolist()

    def _maybe_apply_motion_initial_state(self) -> None:
        motion_init_cfg = self.config.motion_init
        if not motion_init_cfg.enabled:
            return
        if not motion_init_cfg.motion_file:
            raise ValueError("run_sim.motion_init.enabled=True requires --motion-init.motion-file")

        motion_path = Path(motion_init_cfg.motion_file).expanduser().resolve()
        if motion_path.suffix.lower() != ".npz":
            raise ValueError(f"run_sim.motion_init currently supports only .npz clips, got: {motion_path}")

        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        desired_root_state_np: np.ndarray | None = None
        desired_dof_pos_np: np.ndarray | None = None
        desired_object_state_np: np.ndarray | None = None
        has_object_motion = False
        with np.load(motion_path, allow_pickle=True) as data:
            body_names = self._decode_names(np.asarray(data["body_names"]))
            joint_names = self._decode_names(np.asarray(data["joint_names"]))
            frame_count = int(np.asarray(data["joint_pos"]).shape[0])
            frame_idx = int(np.clip(motion_init_cfg.frame_idx, 0, max(frame_count - 1, 0)))
            init_mode = str(getattr(motion_init_cfg, "mode", "raw_motion")).strip().lower().replace("-", "_")

            joint_pos = np.asarray(data["joint_pos"][frame_idx], dtype=np.float32)
            if joint_pos.shape[0] == len(joint_names) + 7:
                joint_pos = joint_pos[7:]
            joint_vel = np.asarray(data["joint_vel"][frame_idx], dtype=np.float32)
            if joint_vel.shape[0] == len(joint_names) + 6:
                joint_vel = joint_vel[6:]

            root_idx = self._resolve_root_body_index(body_names)
            body_pos_w = np.asarray(data["body_pos_w"][frame_idx], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"][frame_idx], dtype=np.float32)
            body_lin_vel_w = np.asarray(data["body_lin_vel_w"][frame_idx], dtype=np.float32)
            body_ang_vel_w = np.asarray(data["body_ang_vel_w"][frame_idx], dtype=np.float32)

            base_root_pos = np.asarray(body_pos_w[root_idx], dtype=np.float32)
            base_root_quat_wxyz = np.asarray(body_quat_w[root_idx], dtype=np.float32)
            base_root_lin_vel = np.asarray(body_lin_vel_w[root_idx], dtype=np.float32)
            base_root_ang_vel = np.asarray(body_ang_vel_w[root_idx], dtype=np.float32)
            has_object_motion = "object_pos_w" in data and motion_init_cfg.object_name
            joint_name_to_index = {name: i for i, name in enumerate(joint_names)}

            def _build_motion_init_state(init_mode_name: str) -> dict[str, Any]:
                root_pos = np.array(base_root_pos, dtype=np.float32, copy=True)
                root_quat_wxyz = np.array(base_root_quat_wxyz, dtype=np.float32, copy=True)
                root_lin_vel = np.array(base_root_lin_vel, dtype=np.float32, copy=True)
                root_ang_vel = np.array(base_root_ang_vel, dtype=np.float32, copy=True)
                dof_pos = np.array(self.simulator.dof_pos[0].detach().cpu().numpy(), dtype=np.float32, copy=True)
                dof_vel = np.array(self.simulator.dof_vel[0].detach().cpu().numpy(), dtype=np.float32, copy=True)

                if init_mode_name == "raw_motion":
                    for sim_idx, sim_name in enumerate(self.simulator.dof_names):
                        clip_idx = joint_name_to_index.get(sim_name)
                        if clip_idx is None:
                            continue
                        dof_pos[sim_idx] = float(joint_pos[clip_idx])
                        dof_vel[sim_idx] = float(joint_vel[clip_idx])
                    root_quat_xyzw = np.array(
                        [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]],
                        dtype=np.float32,
                    )
                elif init_mode_name == "training_default_pose":
                    init_state = self.config.robot.init_state
                    default_joint_angles = getattr(init_state, "default_joint_angles", {}) or {}
                    for sim_idx, sim_name in enumerate(self.simulator.dof_names):
                        if sim_name in default_joint_angles:
                            dof_pos[sim_idx] = float(default_joint_angles[sim_name])
                    init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
                    init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)
                    motion_root_quat = torch.tensor(root_quat_wxyz, dtype=torch.float32, device=self.device).unsqueeze(0)
                    _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=False)
                    default_root_quat_xyzw = quat_from_euler_xyz(
                        init_roll.squeeze(0),
                        init_pitch.squeeze(0),
                        motion_yaw.squeeze(0),
                    )
                    root_pos = np.array([root_pos[0], root_pos[1], init_state.pos[2]], dtype=np.float32)
                    root_quat_xyzw = (
                        default_root_quat_xyzw.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    root_lin_vel = np.asarray(init_state.lin_vel, dtype=np.float32)
                    root_ang_vel = np.asarray(init_state.ang_vel, dtype=np.float32)
                    dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
                else:
                    raise ValueError(
                        f"Unsupported motion-init.mode='{init_mode_name}'. Expected 'raw_motion' or 'training_default_pose'."
                    )

                root_pos_delta = _parse_debug_float_list_env(
                    "HOLOSOMA_MOTION_INIT_ROOT_POS_DELTA",
                    expected_len=3,
                )
                if root_pos_delta is not None:
                    root_pos = root_pos + np.asarray(root_pos_delta, dtype=np.float32)

                yaw_delta_deg_raw = os.environ.get("HOLOSOMA_MOTION_INIT_YAW_DELTA_DEG", "").strip()
                if yaw_delta_deg_raw:
                    yaw_delta_rad = np.deg2rad(float(yaw_delta_deg_raw))
                    root_quat_t = torch.tensor(root_quat_xyzw, dtype=torch.float32, device=self.device).unsqueeze(0)
                    roll_t, pitch_t, yaw_t = get_euler_xyz(root_quat_t, w_last=True)
                    root_quat_xyzw = (
                        quat_from_euler_xyz(roll_t.squeeze(0), pitch_t.squeeze(0), yaw_t.squeeze(0) + yaw_delta_rad)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                zero_init_velocities = _truthy_env("HOLOSOMA_MOTION_INIT_ZERO_VELOCITIES")
                if zero_init_velocities:
                    root_lin_vel = np.zeros_like(root_lin_vel, dtype=np.float32)
                    root_ang_vel = np.zeros_like(root_ang_vel, dtype=np.float32)
                    dof_vel = np.zeros_like(dof_vel, dtype=np.float32)

                root_state = torch.tensor(
                    [[*root_pos.tolist(), *root_quat_xyzw.tolist(), *root_lin_vel.tolist(), *root_ang_vel.tolist()]],
                    device=self.device,
                    dtype=torch.float32,
                )
                dof_state = torch.stack(
                    [
                        torch.tensor(dof_pos, device=self.device, dtype=torch.float32),
                        torch.tensor(dof_vel, device=self.device, dtype=torch.float32),
                    ],
                    dim=-1,
                ).unsqueeze(0)

                object_state = None
                desired_object_state = None
                actor_states: dict[str, torch.Tensor] = {}
                if has_object_motion:
                    object_pos = np.asarray(data["object_pos_w"][frame_idx], dtype=np.float32)
                    object_quat_wxyz = np.asarray(data["object_quat_w"][frame_idx], dtype=np.float32)
                    object_quat_xyzw = np.array(
                        [object_quat_wxyz[1], object_quat_wxyz[2], object_quat_wxyz[3], object_quat_wxyz[0]],
                        dtype=np.float32,
                    )
                    object_lin_vel = np.asarray(data["object_lin_vel_w"][frame_idx], dtype=np.float32)
                    object_ang_vel = np.zeros(3, dtype=np.float32)
                    if zero_init_velocities:
                        object_lin_vel = np.zeros_like(object_lin_vel, dtype=np.float32)
                        object_ang_vel = np.zeros_like(object_ang_vel, dtype=np.float32)
                    object_state = torch.tensor(
                        [[
                            *object_pos.tolist(),
                            *object_quat_xyzw.tolist(),
                            *object_lin_vel.tolist(),
                            *object_ang_vel.tolist(),
                        ]],
                        device=self.device,
                        dtype=torch.float32,
                    )
                    desired_object_state = object_state[0].detach().cpu().numpy().astype(np.float32, copy=False)
                    actor_states[str(motion_init_cfg.object_name)] = object_state.detach().clone()

                return {
                    "root_state": root_state,
                    "dof_state": dof_state,
                    "actor_states": actor_states,
                    "desired_root_state_np": root_state[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "desired_dof_pos_np": dof_state[0, :, 0].detach().cpu().numpy().astype(np.float32, copy=False),
                    "desired_object_state_np": desired_object_state,
                    "object_state": object_state,
                }

            reset_states_by_mode = {
                "raw_motion": _build_motion_init_state("raw_motion"),
                "training_default_pose": _build_motion_init_state("training_default_pose"),
            }
            if init_mode not in reset_states_by_mode:
                raise ValueError(
                    f"Unsupported motion-init.mode='{motion_init_cfg.mode}'. "
                    "Expected 'raw_motion' or 'training_default_pose'."
                )
            active_state = reset_states_by_mode[init_mode]
            root_state = active_state["root_state"]
            dof_state = active_state["dof_state"]
            object_state = active_state["object_state"]
            desired_root_state_np = active_state["desired_root_state_np"]
            desired_dof_pos_np = active_state["desired_dof_pos_np"]
            desired_object_state_np = active_state["desired_object_state_np"]

            self.simulator.robot_root_states[0] = root_state[0]
            self.simulator.set_dof_state_tensor_robots(env_ids, dof_state)
            self.simulator.set_actor_root_state_tensor_robots(env_ids, root_state)
            if has_object_motion and object_state is not None:
                self.simulator.set_actor_states([motion_init_cfg.object_name], env_ids, object_state)

            self.simulator._motion_init_reset_states_by_mode = {
                str(mode_name): {
                    "root_state": state["root_state"].detach().clone(),
                    "dof_state": state["dof_state"].detach().clone(),
                    "actor_states": {
                        str(actor_name): actor_state.detach().clone()
                        for actor_name, actor_state in dict(state["actor_states"]).items()
                    },
                }
                for mode_name, state in reset_states_by_mode.items()
            }
            self.simulator._motion_init_reset_mode = init_mode
            self.simulator._motion_init_reset_root_state = active_state["root_state"].detach().clone()
            self.simulator._motion_init_reset_dof_state = active_state["dof_state"].detach().clone()
            self.simulator._motion_init_reset_actor_states = {
                str(actor_name): actor_state.detach().clone()
                for actor_name, actor_state in dict(active_state["actor_states"]).items()
            }

        if hasattr(self.simulator, "write_state_updates"):
            self.simulator.write_state_updates()
        else:
            self.simulator.refresh_sim_tensors()
        try:
            robot_root_state = self.simulator.robot_root_states[0].detach().cpu().numpy().astype(np.float32, copy=False)
            dof_pos_readback = self.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float32, copy=False)
            object_state = None
            if has_object_motion:
                object_env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
                object_state = (
                    self.simulator.get_actor_states([motion_init_cfg.object_name], object_env_ids)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            root_err = None
            dof_err = None
            object_err = None
            if desired_root_state_np is not None:
                root_err = float(np.max(np.abs(robot_root_state - desired_root_state_np)))
            if desired_dof_pos_np is not None:
                dof_err = float(np.max(np.abs(dof_pos_readback - desired_dof_pos_np)))
            if desired_object_state_np is not None and object_state is not None:
                object_err = float(np.max(np.abs(object_state - desired_object_state_np)))
            logger.info(
                "Motion-init readback: robot_root_state={}, dof_pos_max_abs_err={}{}{}",
                np.array2string(robot_root_state, precision=4),
                "n/a" if dof_err is None else f"{dof_err:.6f}",
                "" if root_err is None else f", root_state_max_abs_err={root_err:.6f}",
                ""
                if object_state is None
                else (
                    f", object_state={np.array2string(object_state, precision=4)}"
                    + ("" if object_err is None else f", object_state_max_abs_err={object_err:.6f}")
                ),
            )
        except Exception as exc:
            logger.warning("Failed to read back motion-init state: {}", exc)
        logger.info(
            "Initialized direct simulation from motion frame {} using mode '{}': {}",
            frame_idx,
            init_mode,
            motion_path.name,
        )

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        """Log FPS statistics for simulation performance monitoring.

        Parameters
        ----------
        step_count : int
            Current step count.
        fps_start_time : float
            Start time for FPS measurement.

        Returns
        -------
        float
            New start time for next FPS measurement.
        """
        elapsed = time.time() - fps_start_time
        fps = 1000 / elapsed
        logger.info(f"Simulation FPS: {fps:.1f}")
        return time.time()


class _DirectPerceptionEnvProxy:
    """Minimal env facade required by PerceptionManager during direct sim runs."""

    def __init__(self, simulator: Any, terrain_manager: Any, robot_config: Any, dt: float, device: str):
        self.simulator = simulator
        self.terrain_manager = terrain_manager
        self.robot_config = robot_config
        self.dt = dt
        self.device = device
        self.num_envs = int(getattr(simulator, "num_envs", 1))
        self.logger = logger
        self._perception_camera_offset_pos = None
        self._perception_camera_offset_quat = None

    @property
    def body_names(self) -> list[str]:
        return list(getattr(self.simulator, "body_names", []))

    @property
    def base_quat(self):
        return self.simulator.base_quat
