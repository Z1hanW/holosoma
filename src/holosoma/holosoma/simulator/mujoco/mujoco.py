"""MuJoCo simulator implementation.

The simulator follows the BaseSimulator interface while providing MuJoCo-specific
implementations for terrain rendering, contact detection, and physics simulation, etc.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import glfw
import numpy as np
import torch
from loguru import logger

from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.simulator import MujocoBackend
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.mujoco.backends import WARP_AVAILABLE, ClassicBackend, WarpBackend
from holosoma.simulator.mujoco.command_registry import CommandRegistry
from holosoma.simulator.mujoco.fields import prepare_fields, prepare_manager_fields
from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager
from holosoma.simulator.mujoco.tensor_views import (
    create_base_linear_acceleration_view,
    quat_apply_mujoco,
    quat_rotate_inverse_mujoco,
)
from holosoma.simulator.mujoco.mjw_views import quat_apply_wxyz_torch
from holosoma.simulator.mujoco.video_recorder import MuJoCoVideoRecorder
from holosoma.simulator.shared.object_registry import ObjectType
from holosoma.simulator.shared.virtual_gantry import create_virtual_gantry
from holosoma.simulator.types import ActorIndices, ActorNames, ActorPoses, ActorStates, EnvIds
from holosoma.utils.adapters import mujoco_draw_adapter


class MuJoCoScene:
    """MuJoCo Scene implementation following SceneInterface protocol.

    Provides a scene interface for MuJoCo simulations that manages environment
    origins and provides compatibility with the holosoma scene system.
    """

    def __init__(self, env_origins: torch.Tensor, device: str) -> None:
        """Initialize MuJoCo Scene.

        Parameters
        ----------
        env_origins : torch.Tensor
            Environment origins tensor with shape [num_envs, 3].
        device : str
            Device string ('cpu' or 'cuda').

        Raises
        ------
        TypeError
            If env_origins is not a torch.Tensor.
        ValueError
            If env_origins doesn't have the correct shape.
        """
        logger.info(f"Initializing MuJoCo Scene with env_origins shape: {env_origins.shape}, device: {device}")

        # Validate input tensor
        if not isinstance(env_origins, torch.Tensor):
            raise TypeError(f"env_origins must be torch.Tensor, got {type(env_origins)}")

        if env_origins.dim() != 2 or env_origins.shape[1] != 3:
            raise ValueError(f"env_origins must have shape [num_envs, 3], got {env_origins.shape}")

        # Ensure tensor is on correct device with correct dtype
        self._env_origins = env_origins.to(device=device, dtype=torch.float32)
        self._device = device

        logger.info(f"MuJoCo Scene initialized successfully - {self._env_origins.shape[0]} environments")

    @property
    def env_origins(self) -> torch.Tensor:
        """Get environment origins tensor.

        Returns
        -------
        torch.Tensor
            Environment origins with shape [num_envs, 3].
        """
        return self._env_origins


class MuJoCo(BaseSimulator):
    """MuJoCo physics simulator with terrain support.

    This class provides a MuJoCo-based physics simulator that provides compatibility with
    the holosoma simulator interface with unified state access and the shared terrain system.
    """

    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str) -> None:
        """Initialize MuJoCo simulator.

        Parameters
        ----------
        tyro_config : FullSimConfig
            Tyro configuration containing simulator, robot, and terrain settings.
        device : str
            Device type for simulation ('cpu' or 'cuda').

        Raises
        ------
        ValueError
            If robot configuration is missing from tyro_config.
        """
        simulator_config = tyro_config.simulator

        logger.info("=== MuJoCo Simulator Initialization Started ===")
        logger.info(f"Device: {device}")
        logger.info(f"Simulator config: {simulator_config}")

        super().__init__(tyro_config, terrain_manager, device)

        # Set robot config for consistency with Isaac simulators
        if not hasattr(tyro_config, "robot"):
            raise ValueError("Robot configuration is required but missing from tyro_config")

        # Store full config for backend access
        self.tyro_config = tyro_config
        self.device = device
        self.robot_config = tyro_config.robot

        # Save num_envs on init() rather than create_envs() so other modules can rely on it
        self.num_envs = self.training_config.num_envs

        # MuJoCo-specific attributes
        self.root_model: mujoco.MjModel | None = None
        self.root_data: mujoco.MjData | None = None

        # Name mapping for prefix handling, because the robot is placed at a named site within
        # Mujoco.
        self.clean_to_prefixed_names: dict[str, str] = {}  # "hip_joint" -> "robot_hip_joint"
        self.prefixed_to_clean_names: dict[str, str] = {}  # "robot_hip_joint" -> "hip_joint"

        # Minimal state tensors (placeholders)
        self.dof_pos = torch.zeros(0, device=device)
        self.dof_vel = torch.zeros(0, device=device)
        self.contact_forces = torch.zeros(0, device=device)
        self.object_contact_forces = torch.zeros(0, device=device)
        self.object_contact_forces_history = torch.zeros(0, device=device)
        self._object_urdf_by_name: dict[str, str] = {}
        self._object_body_name_by_name: dict[str, str] = {}
        self._object_mujoco_body_ids: set[int] = set()
        self._actor_root_metadata: dict[str, dict[str, int | str]] = {}
        self._rigid_body_mujoco_ids: list[int] = []

        # Viewer
        self.viewer: mujoco.viewer.Handle | None = None

        # World ID for multi-environment visualization (which environment to view)
        self.current_world_id: int = 0

        # Text overlay visibility toggle
        self.show_text_overlay: bool = True
        self._pending_reset: bool = False
        self._pending_delayed_resets: list[float] = []
        self._pending_policy_control_actions: list[tuple[float, str]] = []
        self._backspace_policy_control = None
        self._backspace_policy_control_enabled = os.getenv("HOLOSOMA_MUJOCO_BACKSPACE_POLICY_CONTROL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._backspace_policy_control_port = int(
            os.getenv("POLICY_CONTROL_PORT", os.getenv("HOLOSOMA_POLICY_CONTROL_PORT", "5662") or "5662") or "5662"
        )
        self._backspace_autorestart_policy = os.getenv(
            "HOLOSOMA_MUJOCO_BACKSPACE_AUTORESTART_POLICY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._guard_default_viewer_reset = os.getenv("HOLOSOMA_MUJOCO_GUARD_DEFAULT_RESET", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_default_viewer_reset_guard_time = 0.0
        self._policy_command_overlay_enabled = os.getenv(
            "HOLOSOMA_MUJOCO_POLICY_COMMAND_OVERLAY", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._policy_overlay_port = int(
            os.getenv("HOLOSOMA_POLICY_OVERLAY_PORT", os.getenv("POLICY_OVERLAY_PORT", "5663") or "5663") or "5663"
        )
        self._policy_overlay_sub = None
        self._latest_policy_overlay_payload: dict | None = None
        self._last_policy_command_overlay_text: str | None = None
        self._last_policy_overlay_poll_time = 0.0
        self.show_object_collision_geoms: bool = bool(
            getattr(self.simulator_config, "mujoco_show_object_collision", False)
        )
        self.hide_object_visuals_when_showing_collision: bool = bool(
            getattr(self.simulator_config, "mujoco_hide_object_visuals_when_showing_collision", False)
        )
        self._original_geom_rgba: np.ndarray | None = None
        self._object_collision_geom_ids = np.zeros((0,), dtype=np.int32)
        self._object_visual_geom_ids = np.zeros((0,), dtype=np.int32)
        snapshot_path = os.getenv("HOLOSOMA_MUJOCO_OBJECT_GEOM_SNAPSHOT_PATH", "").strip()
        self._mujoco_object_geom_snapshot_path: str | None = snapshot_path or None
        self._mujoco_object_geom_snapshot_written: bool = False

        # Command system for keyboard/joystick controls
        # Initialize commands tensor matching IsaacGym format:
        #    [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        # Shape: [num_envs, 9] to match IsaacGym command structure
        self.commands: torch.Tensor | None = None  # Will be initialized in create_envs when num_envs is known

        logger.info("=== MuJoCo Simulator Initialization Completed ===")

    def _ensure_policy_overlay_sub(self):
        if not self._policy_command_overlay_enabled:
            return None
        if self._policy_overlay_sub is not None:
            return self._policy_overlay_sub
        if self._policy_overlay_port <= 0:
            self._policy_command_overlay_enabled = False
            return None
        try:
            from holosoma_inference.utils.policy_overlay import PolicyOverlaySub

            sub = PolicyOverlaySub(port=self._policy_overlay_port)
            sub.start()
            self._policy_overlay_sub = sub
            return sub
        except Exception as exc:
            logger.warning("MuJoCo policy-command overlay unavailable: {}", exc)
            self._policy_command_overlay_enabled = False
            return None

    @staticmethod
    def _format_sparse_command(command: object) -> str | None:
        try:
            values = np.asarray(command, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 3:
            return None
        return "x={:.3f} y={:.3f} yaw={:.1f}deg".format(
            float(values[0]),
            float(values[1]),
            float(np.rad2deg(float(values[2]))),
        )

    def _policy_command_overlay_lines(self) -> list[str]:
        if not self._policy_command_overlay_enabled:
            return []
        payload = self._latest_policy_overlay_payload
        if not isinstance(payload, dict):
            return [
                "Policy input cmd: waiting for policy",
                "Terminal: ] start, Space motion, W/S/A/D/Q/E command",
            ]

        source = str(payload.get("sparse_command_source", "unknown"))
        mode = str(payload.get("sparse_command_mode", "unknown"))
        manual_enabled = bool(payload.get("sparse_manual_enabled", False))
        effective = self._format_sparse_command(payload.get("sparse_effective_command"))
        manual = self._format_sparse_command(payload.get("sparse_manual_command"))

        lines = []
        if effective is not None:
            lines.append(f"Policy input cmd: {effective}")
        else:
            lines.append("Policy input cmd: waiting for command")
        if manual is not None and manual_enabled:
            lines.append(f"Manual cmd: {manual} ({source}, {mode})")
        else:
            lines.append(f"Command source: {source}, mode={mode}")
        return lines

    def _poll_policy_overlay_for_text(self) -> None:
        if self.viewer is None or not self.show_text_overlay or not self._policy_command_overlay_enabled:
            return
        now = time.monotonic()
        if now - self._last_policy_overlay_poll_time < 0.05:
            return
        self._last_policy_overlay_poll_time = now

        sub = self._ensure_policy_overlay_sub()
        if sub is None:
            return
        payload = sub.get_payload()
        if not isinstance(payload, dict):
            return
        self._latest_policy_overlay_payload = payload
        overlay_text = "\n".join(self._policy_command_overlay_lines())
        if overlay_text != self._last_policy_command_overlay_text:
            self._last_policy_command_overlay_text = overlay_text
            self._update_text_overlay()

    def _ensure_backspace_policy_control(self):
        if not self._backspace_policy_control_enabled:
            return None
        if self._backspace_policy_control is not None:
            return self._backspace_policy_control
        if self._backspace_policy_control_port <= 0:
            self._backspace_policy_control_enabled = False
            return None
        try:
            from holosoma_inference.utils.sim_control import PolicyControlPush

            publisher = PolicyControlPush(port=self._backspace_policy_control_port)
            publisher.start()
            if not publisher.enabled:
                self._backspace_policy_control_enabled = False
                return None
            self._backspace_policy_control = publisher
            return publisher
        except Exception as exc:
            logger.warning("MuJoCo Backspace policy-control publisher unavailable: {}", exc)
            self._backspace_policy_control_enabled = False
            return None

    def _publish_backspace_policy_action(self, action: str) -> bool:
        publisher = self._ensure_backspace_policy_control()
        if publisher is None:
            return False
        try:
            return bool(publisher.publish(action, source="mujoco_backspace"))
        except Exception as exc:
            logger.warning("MuJoCo Backspace policy-control action '{}' failed: {}", action, exc)
            return False

    def _queue_backspace_policy_action(self, action: str, delay_s: float) -> None:
        self._pending_policy_control_actions.append((time.monotonic() + max(float(delay_s), 0.0), str(action)))

    def _queue_delayed_reset(self, delay_s: float) -> None:
        self._pending_delayed_resets.append(time.monotonic() + max(float(delay_s), 0.0))

    def _publish_due_backspace_policy_actions(self) -> None:
        if not self._pending_policy_control_actions:
            return
        now = time.monotonic()
        pending: list[tuple[float, str]] = []
        for due_time, action in self._pending_policy_control_actions:
            if due_time <= now:
                self._publish_backspace_policy_action(action)
            else:
                pending.append((due_time, action))
        self._pending_policy_control_actions = pending

    def _reset_bridge_runtime_state(self) -> None:
        reset_bridge = getattr(self.bridge, "_reset_zmq_lowcmd_runtime_state", None) if self.bridge is not None else None
        if callable(reset_bridge):
            drop_sec = float(os.getenv("HOLOSOMA_ZMQ_LOWCMD_DROP_AFTER_RESET_SEC", "0.22") or "0.22")
            try:
                reset_bridge(drop_lowcmd_sec=max(drop_sec, 0.0))
            except TypeError:
                reset_bridge()

    def _reset_perception_runtime_state(self) -> None:
        reset_perception = (
            getattr(self.bridge, "reset_perception_obs_runtime_state", None) if self.bridge is not None else None
        )
        if callable(reset_perception):
            reset_perception()

    def _consume_due_reset_request(self) -> bool:
        reset_requested = False
        if self._pending_reset:
            self._pending_reset = False
            reset_requested = True
        if self._pending_delayed_resets:
            now = time.monotonic()
            pending: list[float] = []
            for due_time in self._pending_delayed_resets:
                if due_time <= now:
                    reset_requested = True
                else:
                    pending.append(due_time)
            self._pending_delayed_resets = pending
        return reset_requested

    def _perform_coordinated_reset(self) -> None:
        self._reset_bridge_runtime_state()
        self.reset()
        self._reset_perception_runtime_state()
        self._reset_bridge_runtime_state()
        self._update_text_overlay()

    def _looks_like_default_viewer_reset_pose(self) -> bool:
        if self.root_model is None or self.root_data is None or self.robot_qpos_addr is None:
            return False
        motion_init_root_state = getattr(self, "_motion_init_reset_root_state", None)
        if motion_init_root_state is None:
            return False
        current_pos = np.asarray(self.root_data.qpos[self.robot_qpos_addr : self.robot_qpos_addr + 3], dtype=np.float64)
        default_pos = np.asarray(self.root_model.qpos0[self.robot_qpos_addr : self.robot_qpos_addr + 3], dtype=np.float64)
        motion_pos = motion_init_root_state[0, :3].detach().cpu().numpy().astype(np.float64, copy=False)
        return bool(np.linalg.norm(current_pos - default_pos) < 0.08 and np.linalg.norm(current_pos - motion_pos) > 0.25)

    def _guard_default_viewer_reset_pose(self) -> bool:
        if not self._guard_default_viewer_reset:
            return False
        now = time.monotonic()
        if now - self._last_default_viewer_reset_guard_time < 0.25:
            return False
        if not self._looks_like_default_viewer_reset_pose():
            return False
        self._last_default_viewer_reset_guard_time = now
        logger.info("Detected MuJoCo viewer default reset pose; restoring motion-init reset state")
        self._perform_coordinated_reset()
        return True

    def _lift_reset_objects_out_of_scene_contact(self) -> None:
        if os.getenv("HOLOSOMA_MUJOCO_RESET_LIFT_OBJECTS", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        if self.root_model is None or self.root_data is None or not self._object_mujoco_body_ids:
            return
        clearance = float(os.getenv("HOLOSOMA_MUJOCO_RESET_OBJECT_GROUND_CLEARANCE", "0.002") or "0.002")
        max_penetration = 0.0
        for contact_idx in range(int(self.root_data.ncon)):
            contact = self.root_data.contact[contact_idx]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            body1_id = int(self.root_model.geom_bodyid[geom1_id])
            body2_id = int(self.root_model.geom_bodyid[geom2_id])
            body1_is_object = body1_id in self._object_mujoco_body_ids
            body2_is_object = body2_id in self._object_mujoco_body_ids
            if body1_is_object == body2_is_object:
                continue
            penetration = max(0.0, float(-contact.dist))
            max_penetration = max(max_penetration, penetration)
        if max_penetration <= 0.0:
            return
        lift = max_penetration + max(clearance, 0.0)
        if lift <= 0.0:
            return
        for actor_name in self._object_body_name_by_name:
            metadata = self._actor_root_metadata.get(str(actor_name))
            if metadata is None:
                continue
            qpos_addr = int(metadata["qpos_addr"])
            qvel_addr = int(metadata["qvel_addr"])
            self.root_data.qpos[qpos_addr + 2] += lift
            self.root_data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        mujoco.mj_forward(self.root_model, self.root_data)
        logger.info("Lifted reset object state by {:.6f}m to clear scene contact", lift)

    def _build_name_maps(self) -> None:
        """Build bidirectional name maps for clean <-> prefixed name translation.

        Creates mapping dictionaries to translate between clean names (used by holosoma)
        and prefixed names (used internally by MuJoCo) for joints, bodies, and actuators.
        """
        self.clean_to_prefixed_names.clear()
        self.prefixed_to_clean_names.clear()

        prefix = self.scene_manager.robot_prefix

        # Build joint name maps
        assert self.root_model
        for joint_id in range(self.root_model.njnt):
            prefixed_name = self.root_model.joint(joint_id).name
            if prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        # Build body name maps
        for body_id in range(self.root_model.nbody):
            prefixed_name = self.root_model.body(body_id).name
            if prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        # Build actuator name maps
        for actuator_id in range(self.root_model.nu):
            prefixed_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if prefixed_name and prefixed_name.startswith(prefix):
                clean_name = prefixed_name[len(prefix) :]
                self.clean_to_prefixed_names[clean_name] = prefixed_name
                self.prefixed_to_clean_names[prefixed_name] = clean_name

        logger.info(f"Built name maps: {len(self.clean_to_prefixed_names)} clean->prefixed mappings")

    def _build_body_index_mapping(self) -> None:
        """Build MuJoCo body ID to holosoma body index mapping for contact forces.

        Creates a mapping from MuJoCo's internal body IDs to holosoma's body indices,
        which is essential for correctly attributing contact forces to the right
        bodies in the contact force tensor.
        """
        self.mujoco_to_holosoma_body_map: dict[int, int] = {}

        logger.info("=== Building MuJoCo body ID to holosoma index mapping ===")

        # holosoma body_names excludes world, so index 0 = first robot body
        for holosoma_idx, body_name in enumerate(self.body_names):
            # Find corresponding MuJoCo body ID
            prefixed_name = self._get_prefixed_name(body_name)
            mujoco_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
            if mujoco_body_id != -1:
                self.mujoco_to_holosoma_body_map[mujoco_body_id] = holosoma_idx
                logger.info(
                    f"Body mapping: '{body_name}' -> '{prefixed_name}' | MuJoCo ID {mujoco_body_id} -> "
                    f"holosoma idx {holosoma_idx}"
                )
            else:
                logger.warning(f"Body mapping FAILED: '{body_name}' -> '{prefixed_name}' | MuJoCo ID not found")

        logger.info(f"=== Body mapping complete: {len(self.mujoco_to_holosoma_body_map)} mappings created ===")

    def _get_prefixed_name(self, clean_name: str) -> str:
        """Get prefixed name from clean name using map lookup.

        Parameters
        ----------
        clean_name : str
            Clean name without prefix.

        Returns
        -------
        str
            Prefixed name for MuJoCo lookup, or original name if not found.
        """
        return self.clean_to_prefixed_names.get(clean_name, clean_name)

    def _get_clean_name(self, prefixed_name: str) -> str:
        """Get clean name from prefixed name using map lookup.

        Parameters
        ----------
        prefixed_name : str
            Prefixed name from MuJoCo.

        Returns
        -------
        str
            Clean name for holosoma use, or original name if not found.
        """
        return self.prefixed_to_clean_names.get(prefixed_name, prefixed_name)

    def set_headless(self, headless: bool) -> None:
        """Set headless mode for the simulator.

        Parameters
        ----------
        headless : bool
            Whether to run in headless mode (no visualization).
        """
        super().set_headless(headless)
        self.headless = headless

    def setup(self) -> None:
        """Initialize simulator parameters and environment."""
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self) -> None:
        """Configure terrain - deferred until load_assets."""
        return

    def clear_lines(self) -> None:
        """Clear debug visualization lines."""
        mujoco_draw_adapter.clear_lines(self)

    def draw_sphere(
        self, pos: torch.Tensor, radius: float, color: torch.Tensor, env_id: int, pos_id: int | None = None
    ) -> None:
        """Draw a debug sphere at the specified position.

        Parameters
        ----------
        pos : torch.Tensor
            Position of the sphere.
        radius : float
            Radius of the sphere.
        color : torch.Tensor
            Color of the sphere.
        env_id : int
            Environment ID.
        pos_id : Optional[int]
            Position ID for the sphere.
        """
        mujoco_draw_adapter.draw_sphere(self, pos, radius, color, env_id, pos_id=pos_id)

    def draw_line(self, start_point: torch.Tensor, end_point: torch.Tensor, color: torch.Tensor, env_id: int) -> None:
        """Draw a debug line between two points.

        Parameters
        ----------
        start_point : torch.Tensor
            Starting point of the line.
        end_point : torch.Tensor
            Ending point of the line.
        color : torch.Tensor
            Color of the line.
        env_id : int
            Environment ID.
        """
        mujoco_draw_adapter.draw_line(self, start_point, end_point, color, env_id)

    def load_assets(self):
        """Load assets using compositional MjSpec approach.

        Creates the scene manager, sets up the scene components (terrain, lighting,
        materials, robot), compiles the final model, and initializes robot properties
        and joint addressing for simulation.
        """
        logger.info("=== Loading assets ===")

        # Create scene manager
        self.scene_manager = MujocoSceneManager(self.simulator_config)
        self._setup_scene()
        self._object_urdf_by_name = dict(getattr(self.scene_manager, "_object_urdf_by_name", {}))
        self._object_body_name_by_name = dict(getattr(self.scene_manager, "_object_body_name_by_name", {}))

        # Compile once at the end
        self.root_model = self.scene_manager.compile()
        self.root_data = mujoco.MjData(self.root_model)
        self._cache_object_body_ids()
        self._initialize_object_collision_view_state()

        # Apply post-compilation settings
        self.root_model.opt.timestep = self.sim_dt
        self._maybe_export_compiled_model_xml()

        # Backend selection based on configuration
        if self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            if not WARP_AVAILABLE:
                raise RuntimeError(
                    "WarpBackend requested (mujoco_backend='warp') but dependencies not available.\n\n"
                    "To enable GPU acceleration, reinstall with warp support:\n"
                    "  bash scripts/setup_mujoco.sh --with-warp\n\n"
                    "Or install dependencies manually:\n"
                    "  pip install warp-lang mujoco-warp\n\n"
                    "System requirements: CUDA-capable GPU required"
                )
            logger.info("Initializing WarpBackend (GPU multi-environment)")
            self.backend = WarpBackend(self.root_model, self.root_data, self.tyro_config, self.device)
            # Sync CPU initial state (set by _set_initial_joint_angles) to GPU
            self.backend.initialize_state(self.root_model, self.root_data)
        else:
            logger.info("Initializing ClassicBackend (CPU single-environment)")
            self.backend = ClassicBackend(self.root_model, self.root_data, self.tyro_config, self.device)

        # Setup robot indexes, etc
        self._set_robot_properties()
        self._set_robot_joint_addressing()
        self._set_initial_joint_angles()

        # Initialize virtual gantry after the robot using config
        gantry_cfg = self.simulator_config.virtual_gantry
        self.virtual_gantry = create_virtual_gantry(
            sim=self,
            enable=gantry_cfg.enabled,
            attachment_body_names=gantry_cfg.attachment_body_names,
            cfg=gantry_cfg,
        )

        # Initialize bridge system using base class helper
        self._init_bridge()

        if self.video_config.enabled:
            self.video_recorder = MuJoCoVideoRecorder(self.video_config, self)
            self.video_recorder.setup_recording()

        # For debugging
        self.print_mujoco_model_tree()

        logger.info(f"Assets loaded - num_dof: {self.num_dof}, num_bodies: {self.num_bodies}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")

    def _maybe_export_compiled_model_xml(self) -> None:
        export_path = os.environ.get("HOLOSOMA_MUJOCO_EXPORT_XML_PATH", "").strip()
        if not export_path:
            return
        assert self.root_model is not None
        path = Path(export_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            mujoco.mj_saveLastXML(str(path), self.root_model)
        except Exception as exc:
            logger.warning("mj_saveLastXML failed for '{}': {}", path, exc)
            if getattr(self, "scene_manager", None) is None or not hasattr(self.scene_manager.world_spec, "to_xml"):
                raise
            path.write_text(self.scene_manager.world_spec.to_xml(), encoding="utf-8")
        logger.info("Exported compiled MuJoCo XML to {}", path)

    def _setup_scene(self) -> None:
        """Setup scene by composing terrain, lighting, materials, and robot components.

        Follows a specific composition order: terrain first (if not 'none' or 'fake'),
        then lighting and materials, and finally the robot. This ensures proper
        collision configuration and scene element integration.
        """
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state.mesh_type not in ["none", "fake"]:
            # For now, use mesh type to decide whether to programmatically
            # setup scene, terrain, etc. Cannot use "none" since env code relies on none
            # to literally mean none, so we use "fake"
            # This also means robot self_collisions are ignored because we're not in control
            # of the terrain/floor/ground, etc. In this case, the robot MJCF XML needs to handle
            # for collisions (or not).
            self.scene_manager.add_terrain(terrain_state, self.training_config.num_envs)
            self.scene_manager.add_lighting()
            self.scene_manager.add_materials()

        # Always add robot after terrain, in case it references ground/floor, etc for contacts
        self.scene_manager.add_robot(
            terrain_state, self.robot_config, xml_filter=self.simulator_config.robot_mjcf_filter
        )

    def _set_robot_properties(self) -> None:
        """Set robot properties including DOF names, body names, and index mappings.

        Extracts robot joint and body information from the compiled MuJoCo model,
        filters out non-robot elements, and creates the necessary mappings for
        holosoma compatibility.
        """
        # Get all joint names
        assert self.root_model
        all_joint_names = [self.root_model.joint(i).name for i in range(self.root_model.njnt)]
        prefix = self.scene_manager.robot_prefix
        exclude_names = {
            f"{prefix}freejoint",
            f"{prefix}floating_base_joint",
        }

        robot_joint_names = [n for n in all_joint_names if n and n.startswith(prefix) and n not in exclude_names]

        # Build name maps first
        self._build_name_maps()

        self.num_dof = len(robot_joint_names)
        # Use map lookup for clean names
        self.dof_names = [self._get_clean_name(name) for name in robot_joint_names]

        self._rigid_body_mujoco_ids = [
            body_id
            for body_id in range(self.root_model.nbody)
            if self.root_model.body(body_id).name and self.root_model.body(body_id).name.startswith(prefix)
        ]
        self.body_names = [self._get_clean_name(self.root_model.body(body_id).name) for body_id in self._rigid_body_mujoco_ids]
        self.num_bodies = len(self.body_names)

        # Build body index mapping for contact forces (after body_names is defined)
        self._build_body_index_mapping()

        # Motion loading/reset logic expects the configured robot body list, not composite object bodies.
        self._body_list = list(self.robot_config.body_names)

        logger.info(f"Total joints: {len(all_joint_names)}, Robot DOFs: {self.num_dof}")
        logger.info(f"Robot joint names (prefixed): {robot_joint_names}")
        logger.info(f"DOF names: {self.dof_names}")
        logger.info(f"Body names: {self.body_names}")

    def _set_robot_joint_addressing(self) -> None:
        """Setup proper joint addressing using named freejoint and MuJoCo APIs.

        Configures addressing for the robot's freejoint (for root body control)
        and all DOF joints, storing the necessary qpos and qvel addresses for
        efficient state access during simulation.
        """
        logger.info("=== Setting up robot joint addressing ===")

        # Find the named freejoint for robot root control (use prefixed name)
        assert self.root_model
        freejoint_name = self._get_prefixed_name("floating_base_joint")
        self.robot_freejoint_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
        if self.robot_freejoint_id != -1 and self.root_model.jnt_type[self.robot_freejoint_id] != mujoco.mjtJoint.mjJNT_FREE:
            joint_type = self.root_model.jnt_type[self.robot_freejoint_id]
            raise ValueError(f"Joint '{freejoint_name}' is not a freejoint, got type {joint_type}")

        if self.robot_freejoint_id != -1:
            self.robot_qpos_addr = self.root_model.jnt_qposadr[self.robot_freejoint_id]
            self.robot_qvel_addr = self.root_model.jnt_dofadr[self.robot_freejoint_id]
        else:
            logger.warning(f"Robot freejoint '{freejoint_name}' not found in model; resolving from robot root body")
            root_body_candidates: list[str] = []
            if self.body_names:
                root_body_candidates.append(self.body_names[0])
            root_body_candidates.extend(["pelvis", "pelvis_link", "base_link", "torso_link"])

            resolved = None
            seen: set[str] = set()
            for body_name in root_body_candidates:
                if body_name in seen:
                    continue
                seen.add(body_name)
                try:
                    resolved = self._resolve_freejoint_for_body(body_name)
                    break
                except Exception:
                    continue

            if resolved is None:
                raise ValueError(
                    "Robot freejoint not found by name and could not be resolved from a robot root body. "
                    f"Tried joint '{freejoint_name}' and bodies {root_body_candidates}."
                )

            self.robot_freejoint_id = int(resolved["joint_id"])
            self.robot_qpos_addr = int(resolved["qpos_addr"])
            self.robot_qvel_addr = int(resolved["qvel_addr"])
            logger.info(
                "Resolved robot freejoint from body '{}': joint_id={}, qpos_addr={}, qvel_addr={}",
                resolved["body_name"],
                self.robot_freejoint_id,
                self.robot_qpos_addr,
                self.robot_qvel_addr,
            )

        logger.info(
            f"Robot freejoint addressing: ID={self.robot_freejoint_id}, "
            f"qpos_addr={self.robot_qpos_addr}, qvel_addr={self.robot_qvel_addr}"
        )

        # Setup DOF joint addressing using proper MuJoCo APIs
        self.dof_qpos_addrs = []
        self.dof_qvel_addrs = []

        for dof_name in self.dof_names:
            # Add prefix for MuJoCo lookup (dof_names are clean, need prefixed version)
            joint_name = self._get_prefixed_name(dof_name)
            joint_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if joint_id == -1:
                raise ValueError(f"DOF joint '{joint_name}' (clean name: '{dof_name}') not found in model")

            qpos_addr = self.root_model.jnt_qposadr[joint_id]
            qvel_addr = self.root_model.jnt_dofadr[joint_id]

            self.dof_qpos_addrs.append(qpos_addr)
            self.dof_qvel_addrs.append(qvel_addr)

        logger.info(f"Setup {len(self.dof_qpos_addrs)} DOF joint addresses")
        logger.info("=== Robot joint addressing setup completed ===")

    def _set_initial_joint_angles(self) -> None:
        """Set initial joint angles from robot configuration.

        Applies the default joint angles specified in the robot configuration
        to the MuJoCo model's initial state, then performs forward kinematics
        to update body positions.
        """
        logger.info("Setting initial joint angles from robot config")

        assert self.root_model
        assert self.root_data

        default_joint_angles = self.robot_config.init_state.default_joint_angles
        joint_angles_set = 0
        joint_angles_failed = 0
        for joint_name, angle in default_joint_angles.items():
            # Add prefix for MuJoCo lookup
            mujoco_joint_name = self._get_prefixed_name(joint_name)
            joint_id = None
            for i in range(self.root_model.njnt):
                if self.root_model.joint(i).name == mujoco_joint_name:
                    joint_id = i
                    break

            if joint_id is None:
                logger.warning(f"Joint '{joint_name}' (MuJoCo name: '{mujoco_joint_name}') not found in model")
                joint_angles_failed += 1
                continue

            try:
                # Get the qpos address for this joint
                joint_qposadr = self.root_model.jnt_qposadr[joint_id]
                self.root_data.qpos[joint_qposadr] = angle
                joint_angles_set += 1
                logger.info(
                    f"Set joint '{joint_name}' -> '{mujoco_joint_name}' (ID: {joint_id}, "
                    f"qpos_addr: {joint_qposadr}) to angle {angle}"
                )
            except Exception as e:
                logger.warning(f"Failed to set angle for joint '{joint_name}': {e}")
                joint_angles_failed += 1

        if joint_angles_failed > 0:
            raise RuntimeError("Failed to set joint angles")

        logger.info(
            f"Joint angle setting complete: {joint_angles_set} set, {joint_angles_failed} "
            f"failed out of {len(default_joint_angles)} total"
        )

        # Forward kinematics to update body positions based on joint angles
        mujoco.mj_forward(self.root_model, self.root_data)
        logger.info("Applied forward kinematics with initial joint angles")

    def get_supported_scene_formats(self) -> list[str]:
        """Get supported scene formats.

        Returns
        -------
        list[str]
            List of supported scene formats (currently empty).
        """
        return []  # not yet supported

    def create_envs(self, num_envs, env_origins, base_init_state):
        """Create environments - enhanced implementation with robot support.

        Parameters
        ----------
        num_envs : int
            Number of environments to create (currently limited to 1).
        env_origins : torch.Tensor
            Environment origin positions.
        base_init_state : dict[str, Any]
            Initial state configuration for the base.

        Raises
        ------
        ValueError
            If num_envs > 1 (multiple environments not yet supported).
        """
        if num_envs > 1 and self.simulator_config.mujoco_backend != MujocoBackend.WARP:
            raise ValueError(
                f"MuJoCo ClassicBackend only supports single environment, got {num_envs}. "
                f"Use --simulator.config.mujoco-backend=warp for multi-environment support."
            )
        if self._object_urdf_by_name and num_envs > 1:
            raise ValueError(
                "MuJoCo object-carry verification currently supports a single environment only. "
                f"Got num_envs={num_envs}."
            )

        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        # Create Scene following SceneInterface protocol
        self.scene = MuJoCoScene(self.env_origins, self.sim_device)

        # Initialize state tensors based on actual DOF count
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.sim_device)

        # Initialize contact forces tensor with correct shape [num_envs, num_bodies, 3]
        # This matches the interface expected by holosoma (IsaacGym/IsaacSim pattern)
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.sim_device)
        self.object_contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.sim_device)

        # Initialize contact forces history tensor to match IsaacGym/IsaacSim pattern
        # Shape: [num_envs, history_length, num_bodies, 3]
        history_length = self.simulator_config.contact_sensor_history_length
        self.contact_forces_history = torch.zeros(
            self.num_envs, history_length, self.num_bodies, 3, device=self.sim_device
        )
        self.object_contact_forces_history = torch.zeros(
            self.num_envs, history_length, self.num_bodies, 3, device=self.sim_device
        )

        # Initialize command system (Phase 1)
        # Command tensor format matching IsaacGym: [vx, vy, vz, yaw_rate, walk_stand, waist_yaw, ..., height, ...]
        self.commands = torch.zeros(self.num_envs, 9, device=self.sim_device, dtype=torch.float32)
        logger.info(f"Initialized command system with shape: {self.commands.shape}")

    def _set_robot_initial_state(self) -> None:
        """Set complete initial robot state (position, orientation, velocities).

        Applies the robot's initial state configuration to the MuJoCo model,
        including root body position, orientation, and velocities.
        """
        assert self.root_data
        assert self.robot_config
        assert self.robot_qpos_addr is not None
        assert self.robot_qvel_addr is not None

        # Set complete initial robot state (position, orientation, velocities)
        initial_pos = self.robot_config.init_state.pos
        initial_rot = self.robot_config.init_state.rot  # [x,y,z,w] quaternion
        initial_lin_vel = self.robot_config.init_state.lin_vel
        initial_ang_vel = self.robot_config.init_state.ang_vel

        # Apply initial state to robot root body if it exists

        # Convert quaternion: holosoma [x,y,z,w] → MuJoCo [w,x,y,z]
        initial_rot_mj = [initial_rot[3], initial_rot[0], initial_rot[1], initial_rot[2]]
        initial_ang_vel_local = quat_rotate_inverse_mujoco(np.asarray(initial_rot_mj, dtype=np.float64), initial_ang_vel)
        # Use the existing _set_robot_joint_addressing() results
        # Set position: [x, y, z, qw, qx, qy, qz] (7 elements)
        self.root_data.qpos[self.robot_qpos_addr : self.robot_qpos_addr + 3] = initial_pos
        self.root_data.qpos[self.robot_qpos_addr + 3 : self.robot_qpos_addr + 7] = initial_rot_mj

        # Set velocity: [vx, vy, vz, wx, wy, wz] (6 elements)
        self.root_data.qvel[self.robot_qvel_addr : self.robot_qvel_addr + 3] = initial_lin_vel
        self.root_data.qvel[self.robot_qvel_addr + 3 : self.robot_qvel_addr + 6] = initial_ang_vel_local

    def _register_actor_root_metadata(self, name: str, body_name: str, qpos_addr: int, qvel_addr: int) -> None:
        assert self.root_model
        prefixed_body_name = self._get_prefixed_name(body_name)
        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' (MuJoCo name: '{prefixed_body_name}') not found in model")
        self._actor_root_metadata[name] = {
            "body_name": body_name,
            "prefixed_body_name": prefixed_body_name,
            "body_id": body_id,
            "qpos_addr": qpos_addr,
            "qvel_addr": qvel_addr,
        }

    def _resolve_freejoint_for_body(self, body_name: str) -> dict[str, int | str]:
        assert self.root_model

        prefixed_body_name = self._get_prefixed_name(body_name)
        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' (MuJoCo name: '{prefixed_body_name}') not found in model")

        joint_count = int(self.root_model.body_jntnum[body_id])
        if joint_count <= 0:
            raise ValueError(f"Body '{body_name}' does not have a root joint in the MuJoCo model")

        joint_id = int(self.root_model.body_jntadr[body_id])
        if self.root_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            joint_type = self.root_model.jnt_type[joint_id]
            raise ValueError(f"Body '{body_name}' root joint is not freejoint (type={joint_type})")

        return {
            "body_id": body_id,
            "joint_id": joint_id,
            "qpos_addr": int(self.root_model.jnt_qposadr[joint_id]),
            "qvel_addr": int(self.root_model.jnt_dofadr[joint_id]),
            "prefixed_body_name": prefixed_body_name,
            "body_name": body_name,
        }

    def _forward_backend_state(self) -> None:
        forward = getattr(self.backend, "forward", None)
        if callable(forward):
            forward()
            return
        mujoco.mj_forward(self.root_model, self.root_data)

    def _actor_state_from_qpos(self, name: str, env_id: int = 0) -> torch.Tensor:
        assert self.root_data is not None

        metadata = self._actor_root_metadata.get(name)
        if metadata is None:
            raise KeyError(f"Actor '{name}' is not registered in MuJoCo root metadata")

        qpos_addr = int(metadata["qpos_addr"])
        qvel_addr = int(metadata["qvel_addr"])

        qpos_t = getattr(self.backend, "qpos_t", None)
        qvel_t = getattr(self.backend, "qvel_t", None)
        if qpos_t is not None and qvel_t is not None:
            qpos = qpos_t[int(env_id)]
            qvel = qvel_t[int(env_id)]
            pos = qpos[qpos_addr : qpos_addr + 3].to(device=self.sim_device, dtype=torch.float32)
            quat_mj = qpos[qpos_addr + 3 : qpos_addr + 7].to(device=self.sim_device, dtype=torch.float32)
            quat = quat_mj[[1, 2, 3, 0]]
            lin_vel = qvel[qvel_addr : qvel_addr + 3].to(device=self.sim_device, dtype=torch.float32)
            ang_vel_local = qvel[qvel_addr + 3 : qvel_addr + 6].to(device=self.sim_device, dtype=torch.float32)
            ang_vel = quat_apply_wxyz_torch(quat_mj, ang_vel_local)
            return torch.cat([pos, quat, lin_vel, ang_vel], dim=0)

        pos = torch.tensor(self.root_data.qpos[qpos_addr : qpos_addr + 3], device=self.sim_device, dtype=torch.float32)
        quat_mj = self.root_data.qpos[qpos_addr + 3 : qpos_addr + 7]
        quat = torch.tensor([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]], device=self.sim_device, dtype=torch.float32)
        lin_vel = torch.tensor(self.root_data.qvel[qvel_addr : qvel_addr + 3], device=self.sim_device, dtype=torch.float32)
        ang_vel_local = self.root_data.qvel[qvel_addr + 3 : qvel_addr + 6]
        ang_vel_world = quat_apply_mujoco(quat_mj, ang_vel_local)
        ang_vel = torch.tensor(ang_vel_world, device=self.sim_device, dtype=torch.float32)
        return torch.cat([pos, quat, lin_vel, ang_vel], dim=0)

    def _actor_pose_from_qpos(self, name: str) -> torch.Tensor:
        assert self.root_data is not None

        metadata = self._actor_root_metadata.get(name)
        if metadata is None:
            raise KeyError(f"Actor '{name}' is not registered in MuJoCo root metadata")

        qpos_addr = int(metadata["qpos_addr"])
        pos = torch.tensor(self.root_data.qpos[qpos_addr : qpos_addr + 3], device=self.sim_device, dtype=torch.float32)
        quat_mj = self.root_data.qpos[qpos_addr + 3 : qpos_addr + 7]
        quat = torch.tensor([quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]], device=self.sim_device, dtype=torch.float32)
        return torch.cat([pos, quat], dim=0)

    def _write_actor_state(self, name: str, state: torch.Tensor) -> None:
        assert self.root_data is not None

        metadata = self._actor_root_metadata.get(name)
        if metadata is None:
            raise KeyError(f"Actor '{name}' is not registered in MuJoCo root metadata")

        qpos_addr = int(metadata["qpos_addr"])
        qvel_addr = int(metadata["qvel_addr"])

        pos = state[:3].detach().cpu().numpy()
        quat_holosoma = state[3:7].detach().cpu().numpy()
        lin_vel = state[7:10].detach().cpu().numpy()
        ang_vel_world = state[10:13].detach().cpu().numpy()
        quat_mj = np.array([quat_holosoma[3], quat_holosoma[0], quat_holosoma[1], quat_holosoma[2]], dtype=np.float64)
        ang_vel_local = quat_rotate_inverse_mujoco(quat_mj, ang_vel_world)

        self.root_data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.root_data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_mj
        self.root_data.qvel[qvel_addr : qvel_addr + 3] = lin_vel
        self.root_data.qvel[qvel_addr + 3 : qvel_addr + 6] = ang_vel_local

    def _has_registered_dynamic_objects(self) -> bool:
        return bool(self._object_urdf_by_name)

    @staticmethod
    def _geom_type_name(geom_type: int) -> str:
        geom_type_map = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
            int(mujoco.mjtGeom.mjGEOM_SDF): "sdf",
        }
        return geom_type_map.get(int(geom_type), f"geom_{int(geom_type)}")

    def _body_belongs_to_root(self, body_id: int, root_body_id: int) -> bool:
        assert self.root_model is not None
        current_body_id = int(body_id)
        root_body_id = int(root_body_id)
        while current_body_id > 0:
            if current_body_id == root_body_id:
                return True
            current_body_id = int(self.root_model.body_parentid[current_body_id])
        return root_body_id == 0

    def _extract_mesh_payload(self, geom_id: int) -> dict[str, object] | None:
        assert self.root_model is not None
        mesh_id = int(self.root_model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None

        vert_start = int(self.root_model.mesh_vertadr[mesh_id])
        vert_count = int(self.root_model.mesh_vertnum[mesh_id])
        face_start = int(self.root_model.mesh_faceadr[mesh_id])
        face_count = int(self.root_model.mesh_facenum[mesh_id])
        if vert_count <= 0 or face_count <= 0:
            return None

        vertices = np.asarray(self.root_model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32)
        faces = np.asarray(self.root_model.mesh_face[face_start : face_start + face_count], dtype=np.int32)
        if faces.size > 0:
            face_min = int(np.min(faces))
            face_max = int(np.max(faces))
            if face_min >= vert_start and face_max < vert_start + vert_count:
                faces = faces - vert_start

        return {
            "mesh_id": mesh_id,
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
        }

    def _maybe_write_object_geom_snapshot(self) -> str | None:
        if self._mujoco_object_geom_snapshot_written or not self._mujoco_object_geom_snapshot_path:
            return self._mujoco_object_geom_snapshot_path

        assert self.root_model is not None
        assert self.root_data is not None
        if not self._object_body_name_by_name:
            return None

        actors_payload: dict[str, object] = {}
        for actor_name, body_name in self._object_body_name_by_name.items():
            prefixed_root_body_name = self._get_prefixed_name(body_name)
            root_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_root_body_name)
            if root_body_id == -1:
                continue

            root_world_pos = np.asarray(self.root_data.xpos[root_body_id], dtype=np.float64)
            root_world_rot = np.asarray(self.root_data.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
            root_world_rot_inv = root_world_rot.T
            geoms_payload: list[dict[str, object]] = []
            for geom_id in range(int(self.root_model.ngeom)):
                geom_body_id = int(self.root_model.geom_bodyid[geom_id])
                if not self._body_belongs_to_root(geom_body_id, root_body_id):
                    continue

                geom_world_pos = np.asarray(self.root_data.geom_xpos[geom_id], dtype=np.float64)
                geom_world_rot = np.asarray(self.root_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                geom_rel_pos = root_world_rot_inv @ (geom_world_pos - root_world_pos)
                geom_rel_rot = root_world_rot_inv @ geom_world_rot
                geom_rel_quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(geom_rel_quat, geom_rel_rot.reshape(-1))
                geom_rgba = np.asarray(self.root_model.geom_rgba[geom_id], dtype=np.float32)
                geom_entry: dict[str, object] = {
                    "id": int(geom_id),
                    "name": str(mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"),
                    "body_name": str(
                        mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, geom_body_id) or f"body_{geom_body_id}"
                    ),
                    "type": self._geom_type_name(int(self.root_model.geom_type[geom_id])),
                    "relative_pos": [float(v) for v in geom_rel_pos],
                    "relative_quat_wxyz": [float(v) for v in geom_rel_quat],
                    "size": [float(v) for v in self.root_model.geom_size[geom_id]],
                    "rgba": [float(v) for v in geom_rgba],
                    "is_collision": bool(
                        int(self.root_model.geom_contype[geom_id]) != 0 and int(self.root_model.geom_conaffinity[geom_id]) != 0
                    ),
                    "contype": int(self.root_model.geom_contype[geom_id]),
                    "conaffinity": int(self.root_model.geom_conaffinity[geom_id]),
                }
                if int(self.root_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh_payload = self._extract_mesh_payload(geom_id)
                    if mesh_payload is not None:
                        geom_entry["mesh"] = mesh_payload
                geoms_payload.append(geom_entry)

            actors_payload[str(actor_name)] = {
                "root_body_name": body_name,
                "root_body_name_prefixed": prefixed_root_body_name,
                "geoms": geoms_payload,
            }

        if not actors_payload:
            return None

        snapshot_dir = os.path.dirname(self._mujoco_object_geom_snapshot_path)
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
        tmp_path = f"{self._mujoco_object_geom_snapshot_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "actors": actors_payload}, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_path, self._mujoco_object_geom_snapshot_path)
        self._mujoco_object_geom_snapshot_written = True
        logger.info("Wrote MuJoCo object geom snapshot to {}", self._mujoco_object_geom_snapshot_path)
        return self._mujoco_object_geom_snapshot_path

    def _get_split_sim_state_extra_payload(self) -> dict[str, object]:
        """Publish MuJoCo-measured reference-body pose for split sim2sim alignment."""
        assert self.root_model is not None
        assert self.root_data is not None
        include_object_contact_details = os.getenv("HOLOSOMA_SIM_STATE_INCLUDE_OBJECT_CONTACT_DETAILS", "0") == "1"
        include_robot_contact_details = os.getenv("HOLOSOMA_SIM_STATE_INCLUDE_ROBOT_CONTACT_DETAILS", "0") == "1"
        include_key_body_states = os.getenv("HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES", "0") == "1"

        ref_body_name = getattr(self.robot_config, "torso_name", None) or (self.body_names[0] if self.body_names else None)
        if ref_body_name is None:
            return {}

        prefixed_body_name = self._get_prefixed_name(ref_body_name)
        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body_name)
        if body_id == -1:
            return {}

        ref_pos = self.root_data.xpos[body_id]
        ref_quat_mj = self.root_data.xquat[body_id]  # [w, x, y, z]
        ref_quat_xyzw = [ref_quat_mj[1], ref_quat_mj[2], ref_quat_mj[3], ref_quat_mj[0]]

        body_vel = np.zeros(6, dtype=np.float64)  # [angular_vel, linear_vel]
        mujoco.mj_objectVelocity(self.root_model, self.root_data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0)

        object_geom_ids: set[int] = set()
        object_body_ids: set[int] = set()
        for body_name in getattr(self, "_object_body_name_by_name", {}).values():
            prefixed_body = self._get_prefixed_name(body_name)
            object_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body)
            if object_body_id != -1:
                object_body_ids.add(int(object_body_id))

        robot_body_ids: set[int] = {
            int(body_id)
            for body_id in range(1, int(self.root_model.nbody))
            if int(body_id) not in object_body_ids
        }

        for geom_id in range(int(self.root_model.ngeom)):
            body_for_geom = int(self.root_model.geom_bodyid[geom_id])
            if body_for_geom in object_body_ids:
                object_geom_ids.add(geom_id)

        object_robot_contact_count = 0
        object_scene_contact_count = 0
        object_robot_max_pen = 0.0
        object_scene_max_pen = 0.0
        object_robot_contact_bodies: set[str] = set()
        object_robot_contact_geoms: set[str] = set()
        robot_scene_contact_count = 0
        robot_self_contact_count = 0
        robot_scene_max_pen = 0.0
        robot_self_max_pen = 0.0
        robot_scene_contact_bodies: set[str] = set()
        robot_scene_contact_geoms: set[str] = set()
        robot_self_contact_bodies: set[str] = set()
        robot_self_contact_geoms: set[str] = set()
        if object_geom_ids:
            for contact_idx in range(int(self.root_data.ncon)):
                contact = self.root_data.contact[contact_idx]
                geom1_id = int(contact.geom1)
                geom2_id = int(contact.geom2)
                involves_object = geom1_id in object_geom_ids or geom2_id in object_geom_ids
                if not involves_object:
                    continue

                other_geom_id = geom2_id if geom1_id in object_geom_ids else geom1_id
                other_body_id = int(self.root_model.geom_bodyid[other_geom_id])
                penetration = max(0.0, float(-contact.dist))

                if other_body_id in robot_body_ids:
                    object_robot_contact_count += 1
                    object_robot_max_pen = max(object_robot_max_pen, penetration)
                    if include_object_contact_details:
                        body_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
                        if body_name:
                            object_robot_contact_bodies.add(str(body_name))
                        object_geom_id = geom1_id if geom1_id in object_geom_ids else geom2_id
                        other_geom_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)
                        object_geom_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_id)
                        if other_geom_name:
                            object_robot_contact_geoms.add(str(other_geom_name))
                        if object_geom_name:
                            object_robot_contact_geoms.add(str(object_geom_name))
                elif other_body_id not in object_body_ids:
                    object_scene_contact_count += 1
                    object_scene_max_pen = max(object_scene_max_pen, penetration)
        if include_robot_contact_details:
            for contact_idx in range(int(self.root_data.ncon)):
                contact = self.root_data.contact[contact_idx]
                geom1_id = int(contact.geom1)
                geom2_id = int(contact.geom2)
                body1_id = int(self.root_model.geom_bodyid[geom1_id])
                body2_id = int(self.root_model.geom_bodyid[geom2_id])
                if body1_id in object_body_ids or body2_id in object_body_ids:
                    continue
                geom1_is_robot = body1_id in robot_body_ids
                geom2_is_robot = body2_id in robot_body_ids
                if not geom1_is_robot and not geom2_is_robot:
                    continue

                penetration = max(0.0, float(-contact.dist))
                geom1_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
                geom2_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
                body1_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mujoco.mj_id2name(self.root_model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

                if geom1_is_robot and geom2_is_robot:
                    robot_self_contact_count += 1
                    robot_self_max_pen = max(robot_self_max_pen, penetration)
                    if body1_name:
                        robot_self_contact_bodies.add(str(body1_name))
                    if body2_name:
                        robot_self_contact_bodies.add(str(body2_name))
                    if geom1_name:
                        robot_self_contact_geoms.add(str(geom1_name))
                    if geom2_name:
                        robot_self_contact_geoms.add(str(geom2_name))
                    continue

                robot_scene_contact_count += 1
                robot_scene_max_pen = max(robot_scene_max_pen, penetration)
                robot_body_name = body1_name if geom1_is_robot else body2_name
                robot_geom_name = geom1_name if geom1_is_robot else geom2_name
                scene_geom_name = geom2_name if geom1_is_robot else geom1_name
                if robot_body_name:
                    robot_scene_contact_bodies.add(str(robot_body_name))
                if robot_geom_name:
                    robot_scene_contact_geoms.add(str(robot_geom_name))
                if scene_geom_name:
                    robot_scene_contact_geoms.add(str(scene_geom_name))

        payload = {
            "robot_ref_body_name": ref_body_name,
            "robot_ref_state": [
                float(ref_pos[0]),
                float(ref_pos[1]),
                float(ref_pos[2]),
                float(ref_quat_xyzw[0]),
                float(ref_quat_xyzw[1]),
                float(ref_quat_xyzw[2]),
                float(ref_quat_xyzw[3]),
                float(body_vel[3]),
                float(body_vel[4]),
                float(body_vel[5]),
                float(body_vel[0]),
                float(body_vel[1]),
                float(body_vel[2]),
            ],
            "object_robot_contact_count": int(object_robot_contact_count),
            "object_scene_contact_count": int(object_scene_contact_count),
            "object_robot_max_pen": float(object_robot_max_pen),
            "object_scene_max_pen": float(object_scene_max_pen),
        }
        object_geom_snapshot_path = self._maybe_write_object_geom_snapshot()
        if object_geom_snapshot_path:
            payload["mujoco_object_geom_snapshot_path"] = object_geom_snapshot_path
        if include_object_contact_details:
            payload["object_robot_contact_bodies"] = sorted(object_robot_contact_bodies)
            payload["object_robot_contact_geoms"] = sorted(object_robot_contact_geoms)
        if include_robot_contact_details:
            payload["robot_scene_contact_count"] = int(robot_scene_contact_count)
            payload["robot_self_contact_count"] = int(robot_self_contact_count)
            payload["robot_scene_max_pen"] = float(robot_scene_max_pen)
            payload["robot_self_max_pen"] = float(robot_self_max_pen)
            payload["robot_scene_contact_bodies"] = sorted(robot_scene_contact_bodies)
            payload["robot_scene_contact_geoms"] = sorted(robot_scene_contact_geoms)
            payload["robot_self_contact_bodies"] = sorted(robot_self_contact_bodies)
            payload["robot_self_contact_geoms"] = sorted(robot_self_contact_geoms)
        if include_key_body_states:
            key_body_states: dict[str, list[float]] = {}
            raw_key_names = os.getenv("HOLOSOMA_SIM_STATE_KEY_BODY_NAMES", "").strip()
            if raw_key_names:
                key_body_names = [name.strip() for name in raw_key_names.split(",") if name.strip()]
            else:
                key_body_names = [
                    "torso_link",
                    "left_shoulder_roll_link",
                    "right_shoulder_roll_link",
                    "left_elbow_link",
                    "right_elbow_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                    "left_sphere_hand_link",
                    "right_sphere_hand_link",
                ]
            for body_name in key_body_names:
                prefixed_name = self._get_prefixed_name(body_name)
                key_body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
                if key_body_id == -1:
                    continue
                key_pos = self.root_data.xpos[key_body_id]
                key_quat_mj = self.root_data.xquat[key_body_id]
                key_body_states[body_name] = [
                    float(key_pos[0]),
                    float(key_pos[1]),
                    float(key_pos[2]),
                    float(key_quat_mj[1]),
                    float(key_quat_mj[2]),
                    float(key_quat_mj[3]),
                    float(key_quat_mj[0]),
                ]
            payload["key_body_states"] = key_body_states
        return payload

    def _sync_robot_rows_into_all_root_states(self, env_ids: torch.Tensor) -> None:
        if self.all_root_states is self.robot_root_states or env_ids.numel() == 0:
            return
        robot_indices = self.object_registry.get_object_indices("robot", env_ids)
        self.all_root_states[robot_indices] = self.robot_root_states[env_ids]

    def _interleaved_actor_indices_for_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.sim_device)
        per_env = int(self.object_registry.objects_per_env)
        flat_indices = []
        for env_id in env_ids.tolist():
            start = int(env_id) * per_env
            flat_indices.append(torch.arange(start, start + per_env, device=self.sim_device, dtype=torch.long))
        return torch.cat(flat_indices, dim=0)

    def _refresh_all_root_states(self) -> None:
        if self.all_root_states is self.robot_root_states:
            return

        env_ids = torch.arange(self.num_envs, device=self.sim_device, dtype=torch.long)
        for name, _, _, _, _ in self.object_registry.objects:
            actor_indices = self.object_registry.get_object_indices(name, env_ids)
            self.all_root_states[actor_indices] = self.get_actor_states_by_index(actor_indices)

    def reset(self) -> None:
        """Reset simulation state and optionally align robot XY with the gantry anchor."""
        if self.root_model is None or self.root_data is None:
            return

        mujoco.mj_resetData(self.root_model, self.root_data)

        motion_init_root_state = getattr(self, "_motion_init_reset_root_state", None)
        motion_init_dof_state = getattr(self, "_motion_init_reset_dof_state", None)
        motion_init_actor_states = getattr(self, "_motion_init_reset_actor_states", None)
        if motion_init_root_state is not None and motion_init_dof_state is not None:
            env_ids = torch.tensor([0], device=self.sim_device, dtype=torch.long)
            logger.info(
                "Reset restoring motion-init state: have_object_states={} actor_names={}",
                isinstance(motion_init_actor_states, dict) and bool(motion_init_actor_states),
                sorted(motion_init_actor_states.keys()) if isinstance(motion_init_actor_states, dict) else [],
            )
            self.set_actor_root_state_tensor_robots(env_ids, motion_init_root_state)
            self.set_dof_state_tensor_robots(env_ids, motion_init_dof_state)
            if isinstance(motion_init_actor_states, dict):
                for actor_name, actor_state in motion_init_actor_states.items():
                    self.set_actor_states([str(actor_name)], env_ids, actor_state)
            try:
                robot_state_readback = self.get_actor_states(["robot"], env_ids)[0].detach().cpu().numpy()
                object_state_readback = None
                if isinstance(motion_init_actor_states, dict) and motion_init_actor_states:
                    actor_names = [str(actor_name) for actor_name in motion_init_actor_states]
                    object_state_readback = self.get_actor_states(actor_names, env_ids).detach().cpu().numpy()
                logger.info(
                    "Reset motion-init readback: robot_state={}{}",
                    np.array2string(robot_state_readback, precision=4),
                    ""
                    if object_state_readback is None
                    else f", object_state={np.array2string(object_state_readback, precision=4)}",
                )
            except Exception as exc:
                logger.warning("Failed to read back motion-init reset state: {}", exc)
        else:
            self._set_robot_initial_state()

        snapped_to_gantry = self.virtual_gantry is not None and self.virtual_gantry.point is not None
        if snapped_to_gantry:
            self.root_data.qpos[self.robot_qpos_addr : self.robot_qpos_addr + 2] = [
                float(self.virtual_gantry.point[0]),
                float(self.virtual_gantry.point[1]),
            ]

        self._zero_commands()
        mujoco.mj_forward(self.root_model, self.root_data)
        self._lift_reset_objects_out_of_scene_contact()
        env_ids = torch.arange(self.num_envs, device=self.sim_device, dtype=torch.long)
        self.clear_contact_forces_history(env_ids)
        if self._has_registered_dynamic_objects():
            self._refresh_all_root_states()
        if snapped_to_gantry:
            logger.info("Simulation reset (robot XY aligned to virtual gantry anchor)")
        else:
            logger.info("Simulation reset (virtual gantry disabled)")

    def prepare_sim(self) -> None:
        """Prepare simulation - enhanced implementation with ObjectRegistry integration.

        Resets simulation data, sets initial robot state, configures the object registry,
        and creates tensor views for efficient state access during simulation.
        """
        # Reset simulation data
        assert self.root_data
        mujoco.mj_resetData(self.root_model, self.root_data)

        self._set_robot_initial_state()
        self._actor_root_metadata = {}
        self._register_actor_root_metadata(
            "robot",
            self.robot_config.body_names[0],
            self.robot_qpos_addr,
            self.robot_qvel_addr,
        )

        if self._has_registered_dynamic_objects() and self.simulator_config.mujoco_backend == MujocoBackend.WARP:
            raise NotImplementedError(
                "MuJoCo object-carry verification currently supports ClassicBackend only. "
                "Set --simulator.config.mujoco-backend classic."
            )

        object_names = list(self._object_urdf_by_name.keys())
        self.object_registry.setup_ranges(
            self.num_envs,
            robot_count=1,
            scene_count=0,
            individual_count=len(object_names),
        )

        # Register robot with initial pose
        robot_poses = self._actor_pose_from_qpos("robot").unsqueeze(0).repeat(self.num_envs, 1)
        self.object_registry.register_object("robot", ObjectType.ROBOT, 0, robot_poses)

        for position_in_type, object_name in enumerate(object_names):
            body_name = self._object_body_name_by_name.get(object_name, object_name)
            metadata = self._resolve_freejoint_for_body(body_name)
            self._register_actor_root_metadata(
                object_name,
                str(metadata["body_name"]),
                int(metadata["qpos_addr"]),
                int(metadata["qvel_addr"]),
            )
            object_poses = self._actor_pose_from_qpos(object_name).unsqueeze(0).repeat(self.num_envs, 1)
            self.object_registry.register_object(object_name, ObjectType.INDIVIDUAL, position_in_type, object_poses)

        self.object_registry.finalize_registration()

        # Calculate indices for robot freejoint components
        pos_indices = slice(self.robot_qpos_addr, self.robot_qpos_addr + 3)
        quat_indices = slice(self.robot_qpos_addr + 3, self.robot_qpos_addr + 7)
        vel_indices = slice(self.robot_qvel_addr, self.robot_qvel_addr + 3)
        ang_vel_indices = slice(self.robot_qvel_addr + 3, self.robot_qvel_addr + 6)

        # Create robot root states proxy via backend factory
        root_addrs = {
            "pos_indices": pos_indices,
            "quat_indices": quat_indices,
            "vel_indices": vel_indices,
            "ang_vel_indices": ang_vel_indices,
        }
        self.robot_root_states = self.backend.create_root_view(root_addrs)  # type: ignore[assignment]

        if object_names:
            self.all_root_states = torch.zeros(
                self.num_envs * self.object_registry.objects_per_env,
                13,
                device=self.sim_device,
                dtype=torch.float32,
            )
            self._refresh_all_root_states()
        else:
            # Create all_root_states as a view of robot_root_states (single robot case)
            self.all_root_states = self.robot_root_states

        # Calculate indices for DOF positions and velocities
        dof_pos_indices = (
            slice(min(self.dof_qpos_addrs), max(self.dof_qpos_addrs) + 1) if self.dof_qpos_addrs else slice(0, 0)
        )
        dof_vel_indices = (
            slice(min(self.dof_qvel_addrs), max(self.dof_qvel_addrs) + 1) if self.dof_qvel_addrs else slice(0, 0)
        )
        dof_acc_indices = (
            slice(min(self.dof_qvel_addrs), max(self.dof_qvel_addrs) + 1) if self.dof_qvel_addrs else slice(0, 0)
        )

        # Create DOF state proxy via backend factory
        dof_addrs = {"dof_pos_indices": dof_pos_indices, "dof_vel_indices": dof_vel_indices}
        self.dof_state = self.backend.create_dof_state_view(dof_addrs, self.num_dof)  # type: ignore[assignment]

        # Create individual DOF views via backend factories
        self.dof_pos = self.backend.create_dof_pos_view(dof_pos_indices, self.num_dof)  # type: ignore[assignment]
        self.dof_vel = self.backend.create_dof_vel_view(dof_vel_indices, self.num_dof)  # type: ignore[assignment]
        self.dof_acc = self.backend.create_dof_acc_view(dof_acc_indices, self.num_dof)  # type: ignore[assignment]

        # Create contact forces via backend factory
        self.contact_forces = self.backend.create_force_view(self.num_bodies)  # type: ignore[assignment]

        # Create unified applied forces accessor for external force application (e.g., virtual gantry)
        self.applied_forces = self.backend.get_applied_forces_view()

        # Create base_quat, base_angular_vel, base_linear_acc views via backend
        self.base_quat = self.backend.create_quaternion_view(quat_indices)  # type: ignore[assignment]
        self.base_angular_vel = self.backend.create_angular_velocity_view(ang_vel_indices)  # type: ignore[assignment]

        # Base linear acceleration: backend-specific handling
        base_lin_acc_indices = slice(0, 3)
        if WarpBackend is not None and isinstance(self.backend, WarpBackend):
            # WarpBackend: direct GPU tensor access
            self.base_linear_acc = self.backend.qacc_t[:, base_lin_acc_indices]  # type: ignore[assignment,attr-defined]
        else:
            # ClassicBackend: use view system
            self.base_linear_acc = create_base_linear_acceleration_view(  # type: ignore[assignment]
                qacc_array=self.root_data.qacc,
                indices=base_lin_acc_indices,
                num_envs=self.num_envs,
                device=self.sim_device,
            )

        # Initialize rigid body state tensors (required by BaseTask)
        self._rigid_body_pos = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_rot = torch.zeros(
            self.num_envs, self.num_bodies, 4, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )
        self._rigid_body_ang_vel = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.sim_device, dtype=torch.float32
        )

    def prepare_randomization_fields(self, field_names: list[str]) -> None:
        """Prepare model fields for per-environment randomization.

        Delegates to field_preparation.prepare_fields().

        Parameters
        ----------
        field_names : list[str]
            List of MuJoCo field names to expand for per-environment use.
        """
        prepare_fields(self, field_names)

    def prepare_manager_fields(self, **managers) -> None:
        """Scan managers for field requirements and prepare them.

        Delegates to field_preparation.prepare_manager_fields().

        Parameters
        ----------
        **managers : Any
            Manager instances to scan for field requirements.
        """
        prepare_manager_fields(self, **managers)

    def refresh_sim_tensors(self) -> None:
        """Refresh simulation tensors with actual robot data.

        Updates rigid body state tensors and contact forces from the current
        MuJoCo simulation state. Most state tensors use proxy views that
        automatically reflect the current state.
        """
        if self.num_bodies <= 0:
            logger.info("No bodies to refresh (empty world)")
            return

        # NOTE: With the proxy system, most state tensors (dof_pos, dof_vel, dof_state, robot_root_states)
        # automatically reflect the current MuJoCo state, so we only need to update the non-proxy tensors.

        # Try to get rigid body states via backend (zero-copy for WarpBackend)
        rigid_body_views = self.backend.get_rigid_body_state_views()

        if rigid_body_views is not None:
            # Fast path: zero-copy GPU tensors (WarpBackend)
            # Eliminates 132 tensor allocations per frame for G1 robot (33 bodies x 4 tensors)
            positions, orientations, linear_vel, angular_vel = rigid_body_views
            self._rigid_body_pos[:] = positions
            self._rigid_body_rot[:] = orientations
            self._rigid_body_vel[:] = linear_vel
            self._rigid_body_ang_vel[:] = angular_vel
        else:
            # Slow path: CPU loop with tensor allocation (ClassicBackend)
            assert self.root_model
            assert self.root_data
            for holosoma_body_idx, body_id in enumerate(self._rigid_body_mujoco_ids):
                assert body_id < self.root_model.nbody, f"Body ID {body_id} exceeds model bodies {self.root_model.nbody}"

                # Positions (direct access to global coordinates)
                self._rigid_body_pos[0, holosoma_body_idx] = (
                    torch.from_numpy(self.root_data.xpos[body_id]).float().to(self.sim_device)
                )

                # Quaternions (convert MuJoCo w,x,y,z to holosoma x,y,z,w)
                mj_quat = self.root_data.xquat[body_id]  # [w, x, y, z]
                holosoma_quat = [mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]]  # [x, y, z, w]
                self._rigid_body_rot[0, holosoma_body_idx] = torch.tensor(
                    holosoma_quat, device=self.sim_device, dtype=torch.float32
                )

                # Velocities using mj_objectVelocity (recommended approach)
                body_vel = np.zeros(6)  # [angular_vel, linear_vel]
                mujoco.mj_objectVelocity(
                    self.root_model, self.root_data, mujoco.mjtObj.mjOBJ_BODY, body_id, body_vel, 0
                )

                # Extract angular and linear velocities
                self._rigid_body_ang_vel[0, holosoma_body_idx] = (
                    torch.from_numpy(body_vel[:3]).float().to(self.sim_device)
                )
                self._rigid_body_vel[0, holosoma_body_idx] = (
                    torch.from_numpy(body_vel[3:]).float().to(self.sim_device)
                )

        # Update contact forces and history via backend delegation
        if hasattr(self, "contact_forces_history") and hasattr(self, "contact_forces"):
            self.backend.refresh_sim_tensors(self.contact_forces_history)
        if hasattr(self, "object_contact_forces_history") and hasattr(self, "object_contact_forces"):
            self._update_object_contact_forces()
            self.object_contact_forces_history[:] = torch.cat(
                [
                    self.object_contact_forces.clone().unsqueeze(1),
                    self.object_contact_forces_history[:, :-1],
                ],
                dim=1,
            )
        if self._has_registered_dynamic_objects():
            self._refresh_all_root_states()

    def clear_contact_forces_history(self, env_ids: torch.Tensor) -> None:
        """Clear contact forces history for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor
            Tensor of environment IDs to clear history for.
        """
        if len(env_ids) > 0:
            self.contact_forces_history[env_ids, :, :, :] = 0.0
            if self.object_contact_forces_history.numel() > 0:
                self.object_contact_forces_history[env_ids, :, :, :] = 0.0

    def apply_torques_at_dof(self, torques: torch.Tensor) -> None:
        """Apply torques with backend-specific optimization.

        Parameters
        ----------
        torques : torch.Tensor
            Torques to apply to each DOF.

        Raises
        ------
        ValueError
            If torque count doesn't match actuator count or actuator not found.
        """
        assert self.root_model
        assert self.root_data

        if self.root_model.nu == 0:
            logger.warning("No actuators found in MuJoCo model")
            return

        # Check if backend supports direct tensor writes
        ctrl_tensor = self.backend.get_ctrl_tensor()

        if ctrl_tensor is not None:
            # Fast path: Direct zero-copy write (WarpBackend)
            ctrl_tensor[:] = torques
        else:
            # Slow path: Loop-based write (ClassicBackend)
            torques_np = torques.detach().cpu().numpy().flatten()

            # Verify we have the expected number of actuators
            if len(torques_np) != self.root_model.nu:
                raise ValueError(f"Torque count mismatch: got {len(torques_np)}, expected {self.root_model.nu}")

            # Map holosoma DOF indices to MuJoCo actuator indices
            for i, dof_name in enumerate(self.dof_names):
                # Add prefix for MuJoCo actuator lookup (dof_names are clean, need prefixed version)
                actuator_name = self._get_prefixed_name(dof_name)
                actuator_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                if actuator_id == -1:
                    raise ValueError(f"Actuator for DOF '{dof_name}' (MuJoCo name: '{actuator_name}') not found")
                self.root_data.ctrl[actuator_id] = torques_np[i]

    def draw_debug_viz(self):
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()
        self._draw_contact_forces(env_id=self.current_world_id)

    def simulate_at_each_physics_step(self) -> None:
        """Advance simulation by one step."""
        self._publish_due_backspace_policy_actions()

        if self._consume_due_reset_request():
            self._perform_coordinated_reset()
            return

        if self._guard_default_viewer_reset_pose():
            return

        if self.virtual_gantry:
            # Apply virtual gantry forces before step
            self.virtual_gantry.step()

        # Step bridge for updated torques before step using base class helper
        self._step_bridge()

        if (
            self.bridge is not None
            and bool(getattr(self.bridge.bridge_config, "freeze_until_first_command", False))
            and not self.bridge.has_received_external_active_command()
        ):
            if not getattr(self, "_logged_freeze_until_first_command", False):
                logger.info("Freezing MuJoCo physics at initialized state until first external active lowcmd arrives")
                self._logged_freeze_until_first_command = True
            return

        should_hold_physics = getattr(self.bridge, "should_hold_physics", None) if self.bridge is not None else None
        if callable(should_hold_physics) and should_hold_physics():
            return

        # Delegate simulation step to backend
        self.backend.step()

        # Call video recorder capture frame if recording is active
        if self.video_recorder and self.video_recorder.is_recording:
            self.capture_video_frame()

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """Get actor states using the shared ObjectRegistry address space.

        Parameters
        ----------
        indices : ActorIndices
            Actor indices to get states for.

        Returns
        -------
        ActorStates
            Actor states tensor with shape [num_actors, 13] containing
            [x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz] for each actor.

        """
        if len(indices) == 0:
            return torch.empty(0, 13, device=self.sim_device)

        per_env = int(self.object_registry.objects_per_env)
        pos_in_env = indices % per_env
        env_ids = indices // per_env
        output = torch.empty(len(indices), 13, device=self.sim_device, dtype=torch.float32)
        for row, (env_id, pos) in enumerate(zip(env_ids.tolist(), pos_in_env.tolist(), strict=True)):
            object_name = self.object_registry._position_to_name[int(pos)]  # noqa: SLF001
            output[row] = self._actor_state_from_qpos(object_name, int(env_id))
        return output

    def set_actor_states_by_index(self, indices: ActorIndices, states: ActorStates, write_updates: bool = True) -> None:
        """Set actor states using the shared ObjectRegistry address space.

        Parameters
        ----------
        indices : ActorIndices
            Actor indices to set states for.
        states : ActorStates
            Actor states to set with shape [num_actors, 13].
        write_updates : bool
            Whether to apply forward kinematics after setting states.

        """
        assert self.root_data is not None

        if len(indices) != len(states):
            raise ValueError(f"indices/states length mismatch: {len(indices)} indices vs {len(states)} states")

        per_env = int(self.object_registry.objects_per_env)
        pos_in_env = indices % per_env
        env_ids_for_indices = indices // per_env
        use_backend_root_state = hasattr(self.backend, "qpos_t")
        for pos in torch.unique(pos_in_env):
            object_name = self.object_registry._position_to_name[int(pos.item())]  # noqa: SLF001
            mask = pos_in_env == pos
            obj_states = states[mask]
            obj_env_ids = env_ids_for_indices[mask].to(device=self.sim_device, dtype=torch.long)
            for state in obj_states:
                self._write_actor_state(object_name, state)
            if use_backend_root_state and obj_env_ids.numel() > 0:
                metadata = self._actor_root_metadata.get(object_name)
                if metadata is None:
                    raise KeyError(f"Actor '{object_name}' is not registered in MuJoCo root metadata")
                self.backend.set_root_state(
                    obj_env_ids,
                    obj_states,
                    {
                        "robot_qpos_addr": int(metadata["qpos_addr"]),
                        "robot_qvel_addr": int(metadata["qvel_addr"]),
                    },
                )

        if write_updates:
            self._forward_backend_state()
            if self._has_registered_dynamic_objects():
                self._refresh_all_root_states()

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """Get actor indices using ObjectRegistry.

        Parameters
        ----------
        names : Union[str, ActorNames]
            Actor name(s) to get indices for.
        env_ids : Optional[EnvIds]
            Environment IDs to get indices for (None = all environments).

        Returns
        -------
        ActorIndices
            Actor indices for the specified names and environments.

        """
        if isinstance(names, str):
            names = [names]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_initial_poses(self, names: list[str], env_ids: EnvIds | None = None) -> ActorPoses:
        """Get initial poses using ObjectRegistry.

        Parameters
        ----------
        names : list[str]
            Actor names to get initial poses for.
        env_ids : Optional[EnvIds]
            Environment IDs to get poses for (None = all environments).

        Returns
        -------
        ActorPoses
            Initial poses for the specified actors and environments.

        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        return self.object_registry.get_initial_poses_batch(names, env_ids)

    def write_state_updates(self) -> None:
        """Flush staged root-state and DOF updates into the MuJoCo model state."""
        env_ids = torch.arange(self.num_envs, device=self.sim_device)
        self.set_actor_root_state_tensor(env_ids, self.all_root_states)
        self.set_dof_state_tensor_robots(env_ids, self.dof_state)
        self._forward_backend_state()
        if self._has_registered_dynamic_objects():
            self._refresh_all_root_states()

    def set_actor_root_state_tensor(self, set_env_ids: torch.Tensor | None, root_states: torch.Tensor | None) -> None:
        """Legacy compatibility method for LeggedRobotBase.

        This method provides backward compatibility with the existing LeggedRobotBase code
        that calls set_actor_root_state_tensor. It delegates to the robot-specific method.

        Parameters
        ----------
        set_env_ids : Optional[torch.Tensor]
            Which environments to update (None = all).
        root_states : Optional[torch.Tensor]
            Root states tensor (can be all_root_states or robot_root_states).
        """
        if set_env_ids is None:
            set_env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if root_states is not None and root_states is self.all_root_states:
            if self.all_root_states is self.robot_root_states:
                self.set_actor_root_state_tensor_robots(set_env_ids, self.robot_root_states[set_env_ids])
                return

            self._sync_robot_rows_into_all_root_states(set_env_ids)
            actor_indices = self._interleaved_actor_indices_for_envs(set_env_ids)
            self.set_actor_states_by_index(actor_indices, self.all_root_states[actor_indices])
            return

        self.set_actor_root_state_tensor_robots(set_env_ids, root_states)

    def set_dof_state_tensor(self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None) -> None:
        """Legacy compatibility method for LeggedRobotBase.

        This method provides backward compatibility with the existing LeggedRobotBase code
        that calls set_dof_state_tensor. It delegates to the robot-specific method.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        dof_states : Optional[torch.Tensor]
            DOF states tensor (flattened IsaacGym format).
        """
        self.set_dof_state_tensor_robots(env_ids, dof_states)

    def set_actor_root_state_tensor_robots(
        self, env_ids: EnvIds | None = None, root_states: torch.Tensor | None = None
    ) -> None:
        """Set robot root states via backend delegation.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        root_states : Optional[torch.Tensor]
            Robot states to set. Can be either:
            - Pre-sliced tensor [len(env_ids), 13] matching env_ids
            - Full global tensor [num_envs, 13] (will be sliced automatically)
            Format: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz].
            If None, uses current robot_root_states.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if root_states is None:
            root_states = self.robot_root_states[env_ids]
        # CRITICAL: Normalize calling convention - if caller passes full global tensor
        # but only updating subset of envs, slice it to match env_ids dimension
        elif len(root_states) != len(env_ids):
            if len(root_states) == self.num_envs:
                # Full global tensor provided, slice to match env_ids
                root_states = root_states[env_ids]
            else:
                raise ValueError(
                    f"root_states dimension mismatch: got {len(root_states)}, "
                    f"expected either {len(env_ids)} (pre-sliced) or {self.num_envs} (global)"
                )

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.num_dof == 0:
            logger.info("No robot DOFs available - skipping root state update")
            return

        # Delegate to backend
        root_addrs = {"robot_qpos_addr": self.robot_qpos_addr, "robot_qvel_addr": self.robot_qvel_addr}
        self.backend.set_root_state(env_ids, root_states, root_addrs)
        if self.all_root_states is not self.robot_root_states:
            self._sync_robot_rows_into_all_root_states(env_ids)

    def set_dof_state_tensor_robots(
        self, env_ids: EnvIds | None = None, dof_states: torch.Tensor | None = None
    ) -> None:
        """Set robot DOF states via backend delegation.

        Parameters
        ----------
        env_ids : Optional[EnvIds]
            Which environments to update (None = all).
        dof_states : Optional[torch.Tensor]
            DOF states to set. Format depends on tensor shape:
            - 3D [num_selected_envs, num_dofs, 2]: IsaacSim format [pos, vel] per DOF
            - 2D [num_selected_envs * num_dofs, 2]: IsaacGym flattened format
            If None, uses current dof_state.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state  # type: ignore[assignment]

        # Validate inputs
        if len(env_ids) == 0:
            logger.info("No environments to update")
            return

        if self.num_dof == 0:
            logger.info("No robot DOFs available - skipping DOF state update")
            return

        if getattr(dof_states, "_is_tensor_proxy", False):
            dof_states = dof_states.clone()

        selected_env_count = len(env_ids)
        total_dof_rows = self.num_envs * self.num_dof
        selected_dof_rows = selected_env_count * self.num_dof
        env_offsets = env_ids.unsqueeze(1) * self.num_dof
        dof_offsets = torch.arange(self.num_dof, device=env_ids.device).unsqueeze(0)
        selected_indices = (env_offsets + dof_offsets).reshape(-1)

        if dof_states.dim() == 3:
            if dof_states.shape[1:] != (self.num_dof, 2):
                raise ValueError(
                    f"Unsupported 3D dof_states tensor format: {dof_states.shape}. "
                    f"Expected [num_envs, {self.num_dof}, 2]"
                )
            if dof_states.shape[0] == self.num_envs:
                normalized_dof_states = dof_states.reshape(total_dof_rows, 2)
            elif dof_states.shape[0] == selected_env_count:
                normalized_dof_states = self.dof_state.clone()
                normalized_dof_states[selected_indices] = dof_states.reshape(selected_dof_rows, 2)
            else:
                raise ValueError(
                    f"Unsupported 3D dof_states tensor format: {dof_states.shape}. "
                    f"Expected first dimension {selected_env_count} (selected envs) or {self.num_envs} (all envs)"
                )
        elif dof_states.dim() == 2 and dof_states.shape[1] == 2:
            if dof_states.shape[0] == total_dof_rows:
                normalized_dof_states = dof_states
            elif dof_states.shape[0] == selected_dof_rows:
                normalized_dof_states = self.dof_state.clone()
                normalized_dof_states[selected_indices] = dof_states
            else:
                raise ValueError(
                    f"Unsupported dof_states tensor format: {dof_states.shape}. "
                    f"Expected [{self.num_envs} * {self.num_dof}, 2] or [{selected_env_count} * {self.num_dof}, 2]"
                )
        else:
            raise ValueError(
                f"Unsupported dof_states tensor format: {dof_states.shape}. "
                f"Expected [num_envs, num_dofs, 2] or [num_envs * num_dofs, 2]"
            )

        # Delegate to backend
        dof_addrs = {"dof_qpos_addrs": self.dof_qpos_addrs, "dof_qvel_addrs": self.dof_qvel_addrs}
        self.backend.set_dof_state(env_ids, normalized_dof_states, dof_addrs)

    def get_dof_limits_properties(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get DOF limits properties - simplified IsaacSim pattern.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (dof_pos_limits, dof_vel_limits, torque_limits).
        """
        # Initialize tensors directly in method (like IsaacSim)
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)

        # Populate from robot config (like IsaacSim)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]

            # Apply soft limits (like IsaacSim)
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit

        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name: str) -> int:
        """Find rigid body index in body_names list.

        Parameters
        ----------
        body_name : str
            Name of the body to find.

        Returns
        -------
        int
            Index of the body in the body_names list.

        Raises
        ------
        RuntimeError
            If the body name is not found.
        """
        # Returns MuJoCo body ID that works with apply_force()
        prefixed_name = self._get_prefixed_name(body_name)
        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name)
        if body_id >= 0:
            return body_id

        raise RuntimeError(f"Body '{body_name}' not found in body_names: {self.body_names}")

    def setup_viewer(self) -> None:
        """Set up MuJoCo viewer using official mujoco.viewer API with keyboard callback."""
        logger.info("=== Setting up MuJoCo viewer ===")

        if self.headless:
            logger.info("Headless mode enabled - skipping viewer setup")
            self.viewer = None
            return

        self.viewer = mujoco.viewer.launch_passive(self.root_model, self.root_data, key_callback=self._key_callback)
        self._focus_viewer_on_robot()
        self._apply_object_collision_view()
        self._update_text_overlay()
        logger.info("=== Viewer setup completed with keyboard callback ===")

    def _focus_viewer_on_robot(self) -> None:
        if self.viewer is None or self.root_model is None or self.root_data is None:
            return

        body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, self._get_prefixed_name("pelvis"))
        if body_id < 0 and self.root_model.nbody > 1:
            body_id = 1
        if body_id < 0:
            return

        self.viewer.cam.lookat[:] = self.root_data.xpos[body_id]
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 135.0
        self.viewer.cam.elevation = -20.0

    def _initialize_object_collision_view_state(self) -> None:
        if self.root_model is None:
            return

        self._original_geom_rgba = np.array(self.root_model.geom_rgba, dtype=np.float32, copy=True)
        object_body_ids: set[int] = set()
        for body_name in self._object_body_name_by_name.values():
            prefixed_body = self._get_prefixed_name(body_name)
            body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body)
            if body_id != -1:
                object_body_ids.add(int(body_id))

        collision_geom_ids: list[int] = []
        visual_geom_ids: list[int] = []
        for geom_id in range(int(self.root_model.ngeom)):
            body_id = int(self.root_model.geom_bodyid[geom_id])
            if body_id not in object_body_ids:
                continue
            contype = int(self.root_model.geom_contype[geom_id])
            conaffinity = int(self.root_model.geom_conaffinity[geom_id])
            if contype != 0 and conaffinity != 0:
                collision_geom_ids.append(geom_id)
            else:
                visual_geom_ids.append(geom_id)

        self._object_collision_geom_ids = np.asarray(collision_geom_ids, dtype=np.int32)
        self._object_visual_geom_ids = np.asarray(visual_geom_ids, dtype=np.int32)
        if collision_geom_ids:
            logger.info(
                "Detected {} object collision geom(s) and {} object visual geom(s) for MuJoCo collision view",
                len(collision_geom_ids),
                len(visual_geom_ids),
            )
        self._apply_object_collision_view()

    def _apply_object_collision_view(self) -> None:
        if self.root_model is None or self._original_geom_rgba is None:
            return

        if self._object_collision_geom_ids.size > 0:
            self.root_model.geom_rgba[self._object_collision_geom_ids] = self._original_geom_rgba[
                self._object_collision_geom_ids
            ]
        if self._object_visual_geom_ids.size > 0:
            self.root_model.geom_rgba[self._object_visual_geom_ids] = self._original_geom_rgba[self._object_visual_geom_ids]

        if not self.show_object_collision_geoms or self._object_collision_geom_ids.size == 0:
            return

        highlight_rgba = np.array([1.0, 0.15, 0.15, 0.45], dtype=np.float32)
        self.root_model.geom_rgba[self._object_collision_geom_ids] = highlight_rgba

        if self.hide_object_visuals_when_showing_collision and self._object_visual_geom_ids.size > 0:
            visual_rgba = np.array(self._original_geom_rgba[self._object_visual_geom_ids], copy=True)
            visual_rgba[:, 3] = np.minimum(visual_rgba[:, 3], 0.05)
            self.root_model.geom_rgba[self._object_visual_geom_ids] = visual_rgba

    def _add_text_overlay(
        self,
        text: str,
        font: int | None = None,
        gridpos: int | None = None,
        text2: str = "",
    ) -> None:
        """Add screen-space text overlay (HUD) to the MuJoCo viewer.

        This creates a fixed screen-space overlay that doesn't move with the camera,
        similar to a heads-up display (HUD).

        Parameters
        ----------
        text : str
            Primary text to display (left column).
        font : Optional[int]
            Font scale from mujoco.mjtFontScale enum. If None, uses default (150% scale).
            Options: mjFONTSCALE_50, mjFONTSCALE_100, mjFONTSCALE_150, etc.
        gridpos : Optional[int]
            Grid position from mujoco.mjtGridPos enum. If None, uses TOPLEFT.
            Options: mjGRID_TOPLEFT, mjGRID_TOPRIGHT, mjGRID_BOTTOMLEFT, mjGRID_BOTTOMRIGHT.
        text2 : str
            Secondary text to display (right column), defaults to empty string.
        """
        if self.viewer is None:
            return

        # Use the passive viewer's set_texts method for screen-space HUD overlay
        # Format: (font, gridpos, text1, text2)
        self.viewer.set_texts((font, gridpos, text, text2))

    def render(self, sync_frame_time: bool = True) -> None:
        """Render simulation to the viewer

        Parameters
        ----------
        sync_frame_time : bool
            Whether to synchronize frame time (currently unused).
        """
        if self.viewer is None:
            logger.warning("Cannot render, no viewer")
            return

        # Sync GPU -> CPU for WarpBackend with current world_id
        # (no-op for ClassicBackend which returns same data)
        self.root_data = self.backend.get_render_data(world_id=self.current_world_id)

        self._poll_policy_overlay_for_text()
        self.viewer.sync()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    def time(self) -> float:
        """Get current simulation time in seconds.

        Returns the MuJoCo simulation time, used for clock synchronization
        in sim2sim setups. This allows policies to stay synchronized with
        the simulation state.

        Returns
        -------
        float
            Current MuJoCo simulation time in seconds.
        """
        assert self.root_data is not None
        return self.root_data.time

    def get_dof_forces(self, env_id: int = 0) -> torch.Tensor:
        """Get DOF forces for a specific environment.

        Returns actuator forces from MuJoCo's force sensors, providing
        measured joint forces for bridge system sim2sim force feedback.

        Parameters
        ----------
        env_id : int, default=0
            Environment index (currently only supports env 0).

        Returns
        -------
        torch.Tensor
            Tensor of shape [num_dof] with measured joint forces, dtype torch.float32.

        Raises
        ------
        RuntimeError
            If multiple environments requested (not yet supported).
        """
        if env_id != 0:
            raise RuntimeError(f"MuJoCo classic currently only supports single environment (env_id=0), got {env_id}")

        assert self.root_data is not None
        return torch.from_numpy(self.root_data.actuator_force[: self.num_dof]).float().to(self.sim_device)

    def _update_text_overlay(self) -> None:
        """Update text overlay based on current state (event-driven).

        This method is called only when state changes occur (e.g., key presses),
        not on every render frame. This prevents the viewer's keyboard input
        system from being disrupted by frequent set_texts() calls.
        """
        if self.viewer is None:
            return

        if not self.show_text_overlay:
            # Clear text overlays when disabled
            self.viewer.set_texts([])
            return

        # Determine virtual gantry status
        if self.virtual_gantry and self.virtual_gantry.enabled:
            gantry_status = "active"
        else:
            gantry_status = "inactive"

        text_lines = []
        command_lines = self._policy_command_overlay_lines()
        if command_lines:
            text_lines.extend(command_lines)
            text_lines.append("")

        text_lines.extend([
            f"Virtual gantry is {gantry_status}",
            "Press '7' to raise it",
            "Press '8' to lower it",
            "Press '9' to toggle it",
            "Use arrow keys to move gantry (XY)",
            "Press backspace to reset the environment",
        ])
        if self._object_collision_geom_ids.size > 0:
            text_lines.append(
                f"Press 'c' to toggle object collision view ({'on' if self.show_object_collision_geoms else 'off'})"
            )
            text_lines.append(
                "Press 'h' to toggle object visual dimming "
                f"({'on' if self.hide_object_visuals_when_showing_collision else 'off'})"
            )
        text_lines.append("Press 'g' to hide this menu")
        text = " \n".join(text_lines)

        # Use default font and position (None values will use MuJoCo defaults)
        self._add_text_overlay(text)

    def _key_callback(self, keycode: int) -> None:
        """Handle keyboard input with unified command registry and world_id toggling.

        Parameters
        ----------
        keycode : int
            GLFW keycode for the pressed key.
        """
        if self.commands is None:
            return

        # Handle text overlay toggle
        # G key (71): Toggle text overlay visibility
        if keycode == 71:  # 'G' key
            self.show_text_overlay = not self.show_text_overlay
            status = "ON" if self.show_text_overlay else "OFF"
            logger.info(f"Text overlay: {status}")
            # Update overlay immediately when toggled
            self._update_text_overlay()
            return

        if keycode == glfw.KEY_C and self._object_collision_geom_ids.size > 0:
            self.show_object_collision_geoms = not self.show_object_collision_geoms
            self._apply_object_collision_view()
            logger.info("Object collision view: {}", "ON" if self.show_object_collision_geoms else "OFF")
            self._update_text_overlay()
            return

        if keycode == glfw.KEY_H and self._object_collision_geom_ids.size > 0:
            self.hide_object_visuals_when_showing_collision = not self.hide_object_visuals_when_showing_collision
            self._apply_object_collision_view()
            logger.info(
                "Object visual dimming while showing collisions: {}",
                "ON" if self.hide_object_visuals_when_showing_collision else "OFF",
            )
            self._update_text_overlay()
            return

        if keycode in {glfw.KEY_BACKSPACE, glfw.KEY_R}:
            self._pending_policy_control_actions.clear()
            self._pending_delayed_resets.clear()
            reset_sent = self._publish_backspace_policy_action("reset")
            self._pending_reset = True
            # The passive MuJoCo viewer also has built-in Backspace reset behavior.
            # Re-apply our motion-init reset on later sim ticks so the viewer's
            # qpos0 reset cannot leave the robot/object at the model default pose.
            for delay_s in (0.05, 0.15, 0.30):
                self._queue_delayed_reset(delay_s)
                self._queue_backspace_policy_action("reset", delay_s + 0.01)
            if self._backspace_autorestart_policy:
                self._queue_backspace_policy_action("start", 0.55)
                policy_auto_motion = os.getenv(
                    "HOLOSOMA_POLICY_RESET_REARM_AUTO_MOTION_CLIP", "1"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if not policy_auto_motion:
                    self._queue_backspace_policy_action("space", 0.65)
            logger.info(
                "MuJoCo key reset requested coordinated reset (keycode={} policy_reset_sent={} policy_autorestart={})",
                keycode,
                reset_sent,
                self._backspace_autorestart_policy,
            )
            return

        # Handle world_id toggling for multi-environment visualization (WarpBackend only)
        # LEFT ARROW (263): Previous environment
        # RIGHT ARROW (262): Next environment
        # Numbers 0-9 (48-57): Jump to specific environment
        if self.num_envs > 1:
            if keycode == 263:  # LEFT ARROW - Previous environment
                self.current_world_id = (self.current_world_id - 1) % self.num_envs
                logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                return
            if keycode == 262:  # RIGHT ARROW - Next environment
                self.current_world_id = (self.current_world_id + 1) % self.num_envs
                logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                return
            if 48 <= keycode <= 57:  # Number keys 0-9
                requested_id = keycode - 48  # Convert keycode to number (0-9)
                if requested_id < self.num_envs:
                    self.current_world_id = requested_id
                    logger.info(f"Viewing environment: {self.current_world_id + 1}/{self.num_envs}")
                else:
                    logger.warning(f"Environment {requested_id} does not exist (max: {self.num_envs - 1})")
                return

        # Use unified command registry
        if not hasattr(self, "_command_registry"):
            self._command_registry = CommandRegistry(self)
            # Register callback for UI updates on command execution
            self._command_registry.on_command_executed = self._update_text_overlay

        # Single call handles both gantry and robot commands
        if self._command_registry.execute_command(keycode):
            return  # Command handled

        # Log unhandled keys
        logger.debug(f"Unhandled keycode: {keycode}")

    def _zero_commands(self) -> None:
        """Zero all commands (Phase 1 helper method)."""
        if hasattr(self, "commands") and self.commands is not None:
            self.commands.fill_(0.0)
            logger.info("Zeroed all commands")

    def __del__(self) -> None:
        """Cleanup viewer on simulator destruction."""
        logger.info("=== MuJoCo Simulator Cleanup Started ===")
        if getattr(self, "_policy_overlay_sub", None) is not None:
            try:
                self._policy_overlay_sub.close()
                self._policy_overlay_sub = None
            except Exception as e:
                logger.warning(f"Error closing policy overlay subscriber: {e}")
        if getattr(self, "_backspace_policy_control", None) is not None:
            try:
                self._backspace_policy_control.close()
                self._backspace_policy_control = None
            except Exception as e:
                logger.warning(f"Error closing Backspace policy-control publisher: {e}")
        if hasattr(self, "viewer") and self.viewer is not None:
            try:
                logger.info("Closing MuJoCo viewer")
                # Official mujoco.viewer handles cleanup automatically, set to None to release reference
                self.viewer = None
                logger.info("MuJoCo viewer reference released")
            except Exception as e:
                logger.warning(f"Error during viewer cleanup: {e}")
        logger.info("=== MuJoCo Simulator Cleanup Completed ===")

    def _cache_object_body_ids(self) -> None:
        self._object_mujoco_body_ids = set()
        if self.root_model is None:
            return

        for body_name in self._object_body_name_by_name.values():
            prefixed_body_name = self._get_prefixed_name(body_name)
            body_id = mujoco.mj_name2id(self.root_model, mujoco.mjtObj.mjOBJ_BODY, prefixed_body_name)
            if body_id != -1:
                self._object_mujoco_body_ids.add(int(body_id))

    def _update_object_contact_forces(self) -> None:
        """Accumulate only robot<->object contact forces at the robot-body level."""
        assert self.root_model is not None
        assert self.root_data is not None

        if self.object_contact_forces.numel() == 0:
            return

        self.object_contact_forces.fill_(0.0)
        if not self._object_mujoco_body_ids or self.root_data.ncon == 0:
            return

        forcetorque = np.zeros(6, dtype=np.float64)
        for contact_idx in range(self.root_data.ncon):
            contact = self.root_data.contact[contact_idx]
            mujoco.mj_contactForce(self.root_model, self.root_data, contact_idx, forcetorque)
            contact_force = torch.from_numpy(forcetorque[:3]).float().to(self.sim_device)

            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            mj_body1_id = int(self.root_model.geom_bodyid[geom1_id])
            mj_body2_id = int(self.root_model.geom_bodyid[geom2_id])

            body1_is_object = mj_body1_id in self._object_mujoco_body_ids
            body2_is_object = mj_body2_id in self._object_mujoco_body_ids
            if body1_is_object == body2_is_object:
                continue

            holosoma_body1_idx = self.mujoco_to_holosoma_body_map.get(mj_body1_id)
            holosoma_body2_idx = self.mujoco_to_holosoma_body_map.get(mj_body2_id)

            if body1_is_object and holosoma_body2_idx is not None:
                self.object_contact_forces[0, holosoma_body2_idx] += contact_force
            elif body2_is_object and holosoma_body1_idx is not None:
                self.object_contact_forces[0, holosoma_body1_idx] -= contact_force

    def get_object_contact_force_history(self, body_names: list[str] | tuple[str, ...]) -> torch.Tensor:
        if self.object_contact_forces_history.numel() == 0:
            raise RuntimeError("Object-contact history is not initialized for this MuJoCo simulator.")

        if not body_names:
            history_len = int(self.object_contact_forces_history.shape[1])
            return torch.zeros((self.num_envs, history_len, 0, 3), device=self.sim_device, dtype=torch.float32)

        body_indexes = []
        for body_name in body_names:
            if body_name not in self.body_names:
                raise ValueError(f"Body '{body_name}' not found in MuJoCo body_names: {self.body_names}")
            body_indexes.append(self.body_names.index(body_name))

        index_tensor = torch.tensor(body_indexes, dtype=torch.long, device=self.sim_device)
        return self.object_contact_forces_history.index_select(2, index_tensor)

    def _update_contact_forces(self) -> None:
        """Update contact forces tensor using MuJoCo's canonical mj_contactForce() API.

        This method extracts contact forces from MuJoCo's contact detection system and
        accumulates them per body to match holosoma's expected interface.

        Key concepts:
        - MuJoCo contacts are detected between geoms (collision geometries)
        - Multiple geoms can belong to the same body (e.g., robot foot with multiple collision shapes)
        - holosoma expects forces per body, so we need to aggregate geom-level forces to body-level
        - mj_contactForce() returns the 6D force/torque that geom1 exerts on geom2
        - We only use the first 3 components (forces), ignoring torques for now

        Shape: self.contact_forces = [num_envs, num_bodies, 3] = [1, num_bodies, 3]
        """
        assert self.root_model
        assert self.root_data

        # Reset contact forces to zero before accumulating new forces
        # This is essential because we accumulate forces from multiple contacts per body
        self.contact_forces.fill_(0.0)

        # Early return if no contacts detected
        if self.root_data.ncon == 0:
            return

        # Temporary buffer for mj_contactForce() output: [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        forcetorque = np.zeros(6, dtype=np.float64)

        # Iterate through all active contacts in the simulation
        # Each contact represents a collision between two geoms
        for contact_idx in range(self.root_data.ncon):
            contact = self.root_data.contact[contact_idx]

            # Extract the 6D force/torque vector for this contact using MuJoCo's canonical API
            # This gives us the force that geom1 exerts on geom2 at the contact point
            mujoco.mj_contactForce(self.root_model, self.root_data, contact_idx, forcetorque)

            # Extract only the force components (first 3 elements), ignoring torques
            contact_force = forcetorque[:3]  # [force_x, force_y, force_z]

            # Map geoms to their parent bodies using MuJoCo's geom_bodyid mapping
            # This is necessary because contacts are geom-level but holosoma expects body-level forces
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            mj_body1_id = self.root_model.geom_bodyid[geom1_id]
            mj_body2_id = self.root_model.geom_bodyid[geom2_id]

            # Map MuJoCo body IDs to holosoma indices using pre-built mapping
            holosoma_body1_idx = self.mujoco_to_holosoma_body_map.get(mj_body1_id)
            holosoma_body2_idx = self.mujoco_to_holosoma_body_map.get(mj_body2_id)

            # Contact logging is now handled centrally in legged_robot_base._log_contact_forces()

            # Apply Newton's 3rd law: mj_contactForce() result is geom1 exerts on geom2, so geom2's
            # body gets +force, geom1's body gets -force. Note: skips bodies not in our map
            if holosoma_body1_idx is not None:
                self.contact_forces[0, holosoma_body1_idx] -= (
                    torch.from_numpy(contact_force).float().to(self.sim_device)
                )
            if holosoma_body2_idx is not None:
                self.contact_forces[0, holosoma_body2_idx] += (
                    torch.from_numpy(contact_force).float().to(self.sim_device)
                )

    def print_mujoco_model_tree(self) -> None:
        """Print comprehensive MuJoCo model structure for debugging."""
        assert self.root_model
        assert self.root_data

        model_path = self.scene_manager.robot_model_path
        print(f"Analyzing compiled model (robot source: {model_path})")

        model = self.root_model  # Use compiled model instead of reloading from XML
        data = self.root_data  # Use existing data instead of creating new

        print("=" * 80)
        print("MUJOCO MODEL STRUCTURE ANALYSIS")
        print("=" * 80)

        # 1. BASIC MODEL INFO
        print("\n📊 MODEL OVERVIEW:")
        print(f"   Model file: {model_path}")
        print(f"   Total bodies: {model.nbody}")
        print(f"   Total joints: {model.njnt}")
        print(f"   Total DOFs: {model.nv}")
        print(f"   Total qpos elements: {model.nq}")
        print(f"   Total actuators: {model.nu}")
        print(f"   Total geoms: {model.ngeom}")

        # 2. BODY LIST (Simple, no hierarchy to avoid infinite loops)
        print("\n🏗️  BODY LIST:")
        print(f"   {'ID':<3} {'Name':<30} {'Parent ID':<9} {'Parent Name'}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 9} {'-' * 20}")

        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            parent_id = model.body_parentid[body_id]
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id != -1 else "WORLD"
            print(f"   {body_id:<3} {body_name:<30} {parent_id:<9} {parent_name}")

        # 3. JOINT DETAILS (This is the most important part!)
        print("\n🔗 JOINT STRUCTURE:")
        print(f"   {'ID':<3} {'Name':<30} {'Type':<8} {'Body':<20} {'qpos_addr':<9} {'qvel_addr':<9}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 8} {'-' * 20} {'-' * 9} {'-' * 9}")

        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            joint_type = model.jnt_type[joint_id]
            body_id = model.jnt_bodyid[joint_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]

            # Joint type names
            type_names = {0: "FREE", 1: "BALL", 2: "SLIDE", 3: "HINGE"}
            type_name = type_names.get(joint_type, f"TYPE_{joint_type}")

            print(f"   {joint_id:<3} {joint_name:<30} {type_name:<8} {body_name:<20} {qpos_addr:<9} {qvel_addr:<9}")

        # 4. DOF ANALYSIS (What holosoma expects)
        print("\n🎯 DOF ANALYSIS (holosoma perspective):")

        # Get all non-freejoint joints
        dof_joints = []
        for joint_id in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            joint_type = model.jnt_type[joint_id]

            # Skip freejoint (type 0) and floating_base joints
            if joint_type != 0 and "floating_base" not in joint_name.lower():
                dof_joints.append((joint_id, joint_name))

        print(f"   Expected DOF count: {len(dof_joints)}")
        print(f"\n   {'Idx':<3} {'DOF Name':<30} {'MJ_ID':<5} {'qpos_addr':<9} {'qvel_addr':<9}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 5} {'-' * 9} {'-' * 9}")

        for idx, (joint_id, joint_name) in enumerate(dof_joints):
            qpos_addr = model.jnt_qposadr[joint_id]
            qvel_addr = model.jnt_dofadr[joint_id]
            print(f"   {idx:<3} {joint_name:<30} {joint_id:<5} {qpos_addr:<9} {qvel_addr:<9}")

        # 5. ACTUATOR MAPPING
        print("\n⚙️  ACTUATOR MAPPING:")
        print(f"   {'ID':<3} {'Name':<30} {'Joint':<30}")
        print(f"   {'-' * 3} {'-' * 30} {'-' * 30}")

        for actuator_id in range(model.nu):
            actuator_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or f"actuator_{actuator_id}"
            )
            # Get the joint this actuator controls
            joint_id = model.actuator_trnid[actuator_id, 0]  # First transmission element
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
            print(f"   {actuator_id:<3} {actuator_name:<30} {joint_name:<30}")

        # 6. CURRENT STATE SNAPSHOT
        print("\n📸 CURRENT STATE SNAPSHOT:")
        print(f"   qpos (first 10): {data.qpos[:10]}")
        print(f"   qvel (first 10): {data.qvel[:10]}")
        print(f"   ctrl (all): {data.ctrl}")

        print("\n" + "=" * 80)
        print("END OF MODEL ANALYSIS")
        print("=" * 80)
