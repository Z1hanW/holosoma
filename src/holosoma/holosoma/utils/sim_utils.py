"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.utils.common import (
    rank_training_seed,
    seeding,
    validate_deterministic_runtime,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.rotations import get_euler_xyz, quat_from_euler_xyz
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch

if TYPE_CHECKING:
    from holosoma.managers.perception import PerceptionManager


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


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc


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
    world_size = _int_env("WORLD_SIZE", 1)
    if world_size > 1:
        # AppLauncher's distributed branch also limits per-rank Kit/Carbonite/OpenBLAS threads.
        args_cli.distributed = True
        args_cli.device = f"cuda:{_int_env('LOCAL_RANK', 0)}"
    elif device is not None:
        # Use the resolved device
        args_cli.device = device
    else:  # AppLauncher auto-detects
        pass

    kit_args = os.environ.get("HOLOSOMA_ISAACSIM_KIT_ARGS") or os.environ.get("ISAACSIM_KIT_ARGS")
    if kit_args and not getattr(args_cli, "kit_args", ""):
        args_cli.kit_args = kit_args

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

    # Reject an invalid NumPy seed before AppLauncher can create a simulator or
    # CUDA context.  The later seeding() call remains immediately before
    # environment construction.
    training_cfg = getattr(config, "training", None)
    seed = getattr(training_cfg, "seed", None)
    if seed is not None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
        seed = rank_training_seed(
            seed,
            world_size=world_size,
            global_rank=global_rank,
        )
        validate_deterministic_runtime(
            bool(getattr(training_cfg, "torch_deterministic", False))
        )

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
        # Terrain/perception managers transitively initialize Warp. Keep those
        # imports behind the seed/deterministic-runtime preflight above so a
        # rejected direct API call cannot create a CUDA/Warp context first.
        from holosoma.managers.terrain.manager import TerrainManager  # noqa: PLC0415

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
        self._perception_randomization_manager: Any | None = None
        self._perception_producer_steps: int | None = None
        self._perception_last_update_completed_steps: int | None = None
        self._perception_publish_pending = False

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
                # Scientific/contract failures must propagate to run_sim's
                # process status.  Treating them like a normal end of the
                # interactive loop makes launchers report a broken producer as
                # successful and can leave the policy waiting on stale input.
                raise

        # Final statistics
        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        if hasattr(self.simulator, "_split_sim_perception_provider"):
            self.simulator._split_sim_perception_provider = None
        if hasattr(self.simulator, "_split_sim_perception_contract_provider"):
            self.simulator._split_sim_perception_contract_provider = None
        if hasattr(self.simulator, "_reset_split_sim_perception_provider"):
            self.simulator._reset_split_sim_perception_provider = None

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
    def _decode_perception_contract_envelope(encoded: str) -> tuple[dict[str, Any], str]:
        """Decode one bounded, duplicate-key-free ONNX contract handoff."""

        if not isinstance(encoded, str) or not encoded or len(encoded) > 2_000_000:
            raise ValueError(
                "perception_contract_envelope_b64 must be a non-empty bounded base64 string."
            )
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("perception_contract_envelope_b64 is not canonical base64.") from exc
        if base64.b64encode(payload).decode("ascii") != encoded:
            raise ValueError("perception_contract_envelope_b64 is not canonical base64.")
        if not payload or len(payload) > 1_500_000:
            raise ValueError("Decoded perception contract envelope is empty or too large.")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Perception contract envelope contains duplicate key {key!r}.")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"Perception contract envelope contains non-finite JSON value {value}.")

        try:
            envelope = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("Perception contract envelope is not strict unique-key JSON.") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"contract", "sha256"}:
            raise ValueError("Perception contract envelope must contain exactly contract and sha256.")
        contract = envelope.get("contract")
        digest = envelope.get("sha256")
        if not isinstance(contract, dict):
            raise ValueError("Perception contract envelope contract must be a mapping.")
        canonical = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        computed = hashlib.sha256(canonical).hexdigest()
        if (
            not isinstance(digest, str)
            or digest != digest.lower()
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or digest != computed
        ):
            raise ValueError("Perception contract envelope SHA-256 is invalid or mismatched.")
        return contract, digest

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

        producer_tick_dt = self.config.perception_producer_tick_dt
        if producer_tick_dt is None:
            if wants_publish:
                raise ValueError(
                    "Authenticated split perception publishing requires "
                    "--perception-producer-tick-dt from the training control-step contract."
                )
            producer_tick_dt = float(self.simulator.sim_dt)
        if (
            isinstance(producer_tick_dt, bool)
            or not isinstance(producer_tick_dt, (int, float))
            or not math.isfinite(float(producer_tick_dt))
            or float(producer_tick_dt) <= 0.0
        ):
            raise ValueError(
                f"perception_producer_tick_dt must be a finite positive number, got {producer_tick_dt!r}."
            )
        sim_dt = float(self.simulator.sim_dt)
        producer_ratio = float(producer_tick_dt) / sim_dt
        producer_steps = int(round(producer_ratio))
        if producer_steps < 1 or not math.isclose(
            producer_ratio,
            float(producer_steps),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "perception_producer_tick_dt must be an integer multiple of the direct physics dt: "
                f"tick_dt={producer_tick_dt}, sim_dt={sim_dt}, ratio={producer_ratio}."
            )
        if wants_publish and not hasattr(self.simulator, "completed_physics_steps"):
            raise RuntimeError(
                "Authenticated split perception cadence requires a simulator completed_physics_steps counter."
            )
        bridge = getattr(self.simulator, "bridge", None)
        if bool(getattr(bridge, "_reset_perception_on_first_lowcmd", False)):
            raise ValueError(
                "HOLOSOMA_RESET_PERCEPTION_ON_FIRST_LOWCMD is incompatible with episode-authenticated "
                "direct perception; physical simulator reset is the only valid episode boundary."
            )

        self._perception_producer_steps = producer_steps
        self._perception_env_proxy = _DirectPerceptionEnvProxy(
            simulator=self.simulator,
            terrain_manager=self.simulator.terrain_manager,
            robot_config=self.config.robot,
            training_config=self.config.training,
            dt=float(producer_tick_dt),
            device=self.device,
            allow_mujoco_perception_noise=bool(self.config.perception_allow_mujoco_noise),
        )
        from holosoma.managers.perception import PerceptionManager  # noqa: PLC0415
        from holosoma.config_types.randomization import (  # noqa: PLC0415
            RandomizationManagerCfg,
            RandomizationTermCfg,
        )
        from holosoma.managers.randomization import RandomizationManager  # noqa: PLC0415

        self._perception_manager = PerceptionManager(perception_cfg, self._perception_env_proxy, self.device)
        self._perception_env_proxy.perception_manager = self._perception_manager
        direct_randomization = self.config.perception_randomization
        if not direct_randomization.enabled and any(
            value is not None
            for value in (
                direct_randomization.translation_range,
                direct_randomization.rotation_range_deg,
                direct_randomization.noise_std_mult_range,
                direct_randomization.noise_drop_prob_range,
            )
        ):
            raise ValueError(
                "Direct perception randomization ranges were supplied while "
                "--perception-randomization.enabled is false."
            )
        reset_terms = {}
        if direct_randomization.enabled:
            reset_terms["direct_camera_raycast"] = RandomizationTermCfg(
                func=(
                    "holosoma.managers.randomization.terms.locomotion:"
                    "randomize_camera_raycast"
                ),
                params={
                    "enabled": True,
                    "translation_range": direct_randomization.translation_range,
                    "rotation_range_deg": direct_randomization.rotation_range_deg,
                    "noise_std_mult_range": direct_randomization.noise_std_mult_range,
                    "noise_drop_prob_range": direct_randomization.noise_drop_prob_range,
                },
            )
        self._perception_randomization_manager = RandomizationManager(
            RandomizationManagerCfg(reset_terms=reset_terms),
            self._perception_env_proxy,
            self.device,
        )
        self._perception_env_proxy.randomization_manager = self._perception_randomization_manager
        self._perception_randomization_manager.setup()
        self._perception_manager.setup()
        contract_envelope = self.config.perception_contract_envelope_b64
        if wants_publish and not contract_envelope:
            raise ValueError(
                "Authenticated split perception publishing requires "
                "--perception-contract-envelope-b64 from the selected ONNX artifact."
            )
        if contract_envelope:
            contract, declared_digest = self._decode_perception_contract_envelope(
                contract_envelope
            )
            authenticated_digest = self._perception_manager.authenticate_observation_contract(
                contract,
                declared_sha256=declared_digest,
            )
            logger.info(
                "Authenticated direct perception contract against live geometry: sha256={}",
                authenticated_digest,
            )
        # BaseTask.reset_all() advances one zero-action control tick, performs
        # one ordinary all-environment producer update, then installs the
        # targeted reset output seen by the first policy observation.  Replay
        # that authenticated initialization sequence before publication.
        self._reset_split_sim_perception(initialization_warmup=True)
        logger.info(
            "Split sim perception initialized: mode={} camera_source={} producer_tick_dt={} "
            "physics_steps_per_tick={} camera_reset_randomization={} mujoco_noise={}",
            perception_cfg.output_mode,
            perception_cfg.camera_source,
            float(producer_tick_dt),
            producer_steps,
            bool(direct_randomization.enabled),
            bool(self.config.perception_allow_mujoco_noise),
        )

        self.simulator._reset_split_sim_perception_provider = self._reset_split_sim_perception
        if wants_publish:
            self.simulator._split_sim_perception_provider = self._get_split_sim_perception_obs
            contract_provider = getattr(self._perception_manager, "get_observation_contract_sha256", None)
            if not callable(contract_provider):
                raise RuntimeError(
                    "split sim2sim perception publishing requires an effective observation-contract digest"
                )
            self.simulator._split_sim_perception_contract_provider = contract_provider

    def _get_split_sim_perception_obs(self) -> list[float] | None:
        if self._perception_manager is None:
            return None
        producer_steps = self._perception_producer_steps
        if producer_steps is None:
            raise RuntimeError("Split perception producer cadence was not initialized.")
        completed_steps = int(getattr(self.simulator, "completed_physics_steps"))
        last_update = self._perception_last_update_completed_steps
        if last_update is None:
            raise RuntimeError("Split perception producer update state was not initialized.")
        if completed_steps < last_update:
            raise RuntimeError(
                "Simulator physics-step counter moved backwards without an authenticated perception reset: "
                f"completed={completed_steps}, last_update={last_update}."
            )
        if self._perception_publish_pending and completed_steps == last_update:
            # Re-publish the already-computed reset frame without another
            # manager update or RNG draw until physics advances.  This makes a
            # late ZMQ subscriber compatible with freeze-until-first-command
            # and keeps shared-memory publication equally well defined.
            pass
        else:
            self._perception_publish_pending = False
            if completed_steps == last_update or completed_steps % producer_steps != 0:
                return None
            if completed_steps - last_update != producer_steps:
                raise RuntimeError(
                    "Split perception provider skipped an authenticated producer tick: "
                    f"completed={completed_steps}, last_update={last_update}, "
                    f"physics_steps_per_tick={producer_steps}."
                )
            self.simulator.refresh_sim_tensors()
            self._perception_manager.update()
            self._perception_last_update_completed_steps = completed_steps
        perception_obs = self._perception_manager.get_obs()
        if perception_obs.ndim != 2 or perception_obs.shape[0] < 1:
            raise RuntimeError(f"Unexpected perception observation shape: {tuple(perception_obs.shape)}")
        return perception_obs[0].detach().cpu().to(torch.float32).tolist()

    def _advance_initialization_perception_warmup(self) -> None:
        """Advance the one zero-action control tick used by training reset_all()."""

        producer_steps = self._perception_producer_steps
        if producer_steps is None or producer_steps < 1:
            raise RuntimeError("Perception initialization warm-up cadence is unavailable.")
        completed_before = int(getattr(self.simulator, "completed_physics_steps", 0))
        if completed_before != 0:
            raise RuntimeError(
                "Perception initialization warm-up must start at physical step zero, "
                f"got {completed_before}."
            )

        bridge = getattr(self.simulator, "bridge", None)
        reset_control_phase = getattr(bridge, "reset_control_phase", None)
        apply_initialization_hold = getattr(
            bridge,
            "apply_initialization_default_pose_hold",
            None,
        )
        if bridge is not None and (
            not callable(reset_control_phase)
            or not callable(apply_initialization_hold)
        ):
            raise RuntimeError(
                "Authenticated perception warm-up requires an isolatable simulator bridge."
            )

        # Do not let an already-connected DDS/ZMQ sender replace training's
        # zero-action initialization tick.  The detached bridge applies only
        # the nominal default-pose PD hold and cannot drain control/reset input,
        # publish a premature state, or advance its runtime decimation phase.
        if callable(reset_control_phase):
            reset_control_phase()
        if bridge is not None:
            self.simulator.bridge = None
        try:
            for expected_step in range(1, producer_steps + 1):
                self.simulator.refresh_sim_tensors()
                if callable(apply_initialization_hold):
                    apply_initialization_hold()
                self.simulator.simulate_at_each_physics_step()
                completed = int(getattr(self.simulator, "completed_physics_steps", 0))
                if completed != expected_step:
                    raise RuntimeError(
                        "Direct initialization failed to advance exactly one authenticated "
                        "physics step during its training-equivalent warm-up: "
                        f"expected={expected_step}, completed={completed}."
                    )
        finally:
            if bridge is not None:
                self.simulator.bridge = bridge
            if callable(reset_control_phase):
                reset_control_phase()

    def _reset_split_sim_perception(
        self,
        *,
        initialization_warmup: bool = False,
    ) -> None:
        if self._perception_manager is None:
            return
        env_ids = torch.arange(
            self._perception_env_proxy.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._perception_manager.reset(env_ids)
        if self._perception_randomization_manager is not None:
            self._perception_randomization_manager.reset(env_ids)
        if initialization_warmup:
            self._advance_initialization_perception_warmup()
        self.simulator.refresh_sim_tensors()
        if initialization_warmup:
            self._perception_manager.update()
            # Training restores/refreshes the reset state after its warm-up
            # control tick and before the targeted producer refresh.
            self.simulator.refresh_sim_tensors()
        update_env_ids = (
            None
            if self._perception_manager.uses_legacy_full_reset_refresh()
            else env_ids
        )
        self._perception_manager.update(update_env_ids)
        completed_steps = int(getattr(self.simulator, "completed_physics_steps", 0))
        expected_completed_steps = (
            int(self._perception_producer_steps or 0)
            if initialization_warmup
            else 0
        )
        if completed_steps != expected_completed_steps:
            raise RuntimeError(
                "Perception reset completed-physics-step boundary is inconsistent: "
                f"initialization_warmup={initialization_warmup}, "
                f"expected={expected_completed_steps}, got={completed_steps}."
            )
        self._perception_last_update_completed_steps = completed_steps
        self._perception_publish_pending = True

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

                if init_mode_name in {"raw_motion", "raw_motion_grounded"}:
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
                    if init_mode_name == "raw_motion_grounded":
                        root_pos[2] = float(self.config.robot.init_state.pos[2])
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
                        f"Unsupported motion-init.mode='{init_mode_name}'. "
                        "Expected 'raw_motion', 'raw_motion_grounded', or 'training_default_pose'."
                    )

                root_pos_delta = _parse_debug_float_list_env(
                    "HOLOSOMA_MOTION_INIT_ROOT_POS_DELTA",
                    expected_len=3,
                )
                if root_pos_delta is not None:
                    root_pos = root_pos + np.asarray(root_pos_delta, dtype=np.float32)

                root_lin_vel_override = _parse_debug_float_list_env(
                    "HOLOSOMA_MOTION_INIT_ROOT_LIN_VEL",
                    expected_len=3,
                )
                if root_lin_vel_override is not None:
                    root_lin_vel = np.asarray(root_lin_vel_override, dtype=np.float32)

                root_ang_vel_override = _parse_debug_float_list_env(
                    "HOLOSOMA_MOTION_INIT_ROOT_ANG_VEL",
                    expected_len=3,
                )
                if root_ang_vel_override is not None:
                    root_ang_vel = np.asarray(root_ang_vel_override, dtype=np.float32)

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
                "raw_motion_grounded": _build_motion_init_state("raw_motion_grounded"),
                "training_default_pose": _build_motion_init_state("training_default_pose"),
            }
            if init_mode not in reset_states_by_mode:
                raise ValueError(
                    f"Unsupported motion-init.mode='{motion_init_cfg.mode}'. "
                    "Expected 'raw_motion', 'raw_motion_grounded', or 'training_default_pose'."
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
            self._maybe_align_virtual_gantry_to_motion_init(root_state, frame_idx=frame_idx)

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

    def _maybe_align_virtual_gantry_to_motion_init(self, root_state: torch.Tensor, *, frame_idx: int) -> None:
        gantry = getattr(self.simulator, "virtual_gantry", None)
        if gantry is None:
            return
        gantry_cfg = getattr(getattr(self.simulator, "simulator_config", None), "virtual_gantry", None)
        if gantry_cfg is not None and not bool(getattr(gantry_cfg, "follow_robot_on_episode_start", True)):
            return

        root_pos = root_state[0, :3].detach().cpu().numpy().astype(np.float32, copy=False)
        height = float(getattr(gantry, "height", 0.0))
        gantry.point = np.array([float(root_pos[0]), float(root_pos[1]), float(root_pos[2]) + height], dtype=float)
        logger.info(
            "Virtual gantry aligned above motion-init frame {}: [{:.3f}, {:.3f}, {:.3f}]",
            frame_idx,
            float(gantry.point[0]),
            float(gantry.point[1]),
            float(gantry.point[2]),
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

    def __init__(
        self,
        simulator: Any,
        terrain_manager: Any,
        robot_config: Any,
        training_config: Any,
        dt: float,
        device: str,
        *,
        allow_mujoco_perception_noise: bool,
    ):
        self.simulator = simulator
        self.terrain_manager = terrain_manager
        self.robot_config = robot_config
        self.training_config = training_config
        self.dt = dt
        self.device = device
        self.num_envs = int(getattr(simulator, "num_envs", 1))
        self.logger = logger
        self._allow_mujoco_perception_noise = allow_mujoco_perception_noise
        self._perception_camera_offset_pos = None
        self._perception_camera_offset_quat = None

    @property
    def body_names(self) -> list[str]:
        return list(getattr(self.simulator, "body_names", []))

    @property
    def base_quat(self):
        return self.simulator.base_quat
