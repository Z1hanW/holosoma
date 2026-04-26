from __future__ import annotations

import builtins
import copy
from dataclasses import fields, is_dataclass
import json
import math
import os
import pathlib
import re
import zipfile
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import trimesh

from holosoma.config_types.full_sim import FullSimConfig
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import omni.log
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import EventManager, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.timer import Timer
from loguru import logger
from omegaconf import DictConfig, ListConfig

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.config_types.command import MotionConfig
from holosoma.config_types.simulator import SimulatorInitConfig, SceneConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.isaacsim.event_cfg import EventCfg
from holosoma.simulator.isaacsim.events import randomize_body_com, randomize_rigid_body_inertia
from holosoma.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG
from holosoma.simulator.isaacsim.usd_file_loader import USDFileLoader
from holosoma.simulator.isaacsim.registry_utils import register_objects
from holosoma.simulator.isaacsim.proxy_utils import AllRootStatesProxy, RootStatesProxy
from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter
from holosoma.simulator.isaacsim.state_utils import fullstate_xyzw_to_wxyz
from holosoma.simulator.isaacsim.prim_utils import (
    log_robot_properties,
    print_prim_tree,
    UsdSceneLoaderCfg,
    create_usd_scene_loader,
)
from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder
from holosoma.simulator.shared.virtual_gantry import (
    VirtualGantry,
    create_virtual_gantry,
    GantryCommand,
    GantryCommandData,
)
from holosoma.simulator.shared.urdf_topology import extract_urdf_topology_signature
from holosoma.utils.object_geometry import UrdfBoxPrimitiveMetadata, load_urdf_box_primitive_metadata

from holosoma.simulator.types import ActorNames, ActorIndices, EnvIds, ActorStates, ActorPoses

_OBJECT_CONTACT_MONITOR_BODY_NAMES = (
    "left_foot_contact_point",
    "right_foot_contact_point",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "torso_link",
)
_OBJECT_CONTACT_SENSOR_FORCE_THRESHOLD = 0.0
_OBJECT_CONTACT_OBSERVATION_FUNC_PATHS = frozenset(
    {
        "holosoma.managers.observation.terms.wbt:body_contact_force_magnitude",
        "holosoma.managers.observation.terms.wbt:body_contact_binary_flag",
    }
)
_OBJECT_CONTACT_REWARD_FUNC_PATHS = frozenset(
    {
        "holosoma.managers.reward.terms.wbt:body_object_contact_reward",
        "holosoma.managers.reward.terms.wbt:ObjectUndesiredContacts",
    }
)


def _repo_root_from_holosoma_package() -> pathlib.Path:
    return pathlib.Path(get_holosoma_root()).resolve().parents[2]


def _object_urdf_compat_fallbacks(path: pathlib.Path) -> list[pathlib.Path]:
    """Current-repo fallbacks for object URDF paths embedded in older motion banks."""
    repo_root = _repo_root_from_holosoma_package()
    candidates: list[pathlib.Path] = []

    parts = path.expanduser().parts
    if "data" in parts:
        data_idx = parts.index("data")
        candidates.append(repo_root.joinpath(*parts[data_idx:]))

    stem = path.stem
    if stem:
        if "__" in stem:
            names = [stem]
        else:
            names = [f"{stem}__eff10", f"{stem}__eff09", f"{stem}__baseline"]
        base = repo_root / "data/ds_box_data/scale_mix_all/train_g1_w_obj_prepared/_generated_urdfs"
        candidates.extend(base / f"{name}.urdf" for name in names)

    return candidates


def _resolve_existing_object_urdf_path(path_like: str | pathlib.Path) -> pathlib.Path:
    resolved = pathlib.Path(resolve_data_file_path(str(path_like))).expanduser().resolve()
    if resolved.is_file():
        return resolved

    if not resolved.exists():
        for fallback in _object_urdf_compat_fallbacks(resolved):
            if fallback.is_file() and fallback.suffix.lower() == ".urdf":
                logger.warning("Resolved missing object URDF '{}' to compatibility fallback '{}'", resolved, fallback)
                return fallback.resolve()

    return resolved


def _iter_config_nodes(config: Any, *, seen: set[int] | None = None):
    if config is None or isinstance(config, (str, bytes, int, float, bool, pathlib.Path)):
        return

    if seen is None:
        seen = set()

    config_id = id(config)
    if config_id in seen:
        return
    seen.add(config_id)
    yield config

    if is_dataclass(config):
        for config_field in fields(config):
            yield from _iter_config_nodes(getattr(config, config_field.name), seen=seen)
        return

    if isinstance(config, (dict, DictConfig)):
        for value in config.values():
            yield from _iter_config_nodes(value, seen=seen)
        return

    if isinstance(config, (list, tuple, set, ListConfig)):
        for value in config:
            yield from _iter_config_nodes(value, seen=seen)


class IsaacSim(BaseSimulator):
    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str):
        super().__init__(tyro_config, terrain_manager, device)

        # Add device attribute for base simulator compatibility
        self.device = device
        self._object_urdf_by_name: dict[str, str] = {}
        self._resolved_training_object_specs: list[tuple[str, str]] = []
        self._env_object_urdf_paths: list[str] = []
        self._heterogeneous_object_env_assignment = False
        self._heterogeneous_object_single_slot_enabled = False
        self._training_object_use_box_primitives = False
        self._training_object_box_metadata_by_urdf: dict[str, UrdfBoxPrimitiveMetadata] = {}
        self._object_contact_filter_prim_paths_expr: list[str] = []
        self._object_contact_sensors: dict[str, ContactSensor] = {}
        self._required_object_contact_sensor_body_names = self._resolve_required_object_contact_sensor_body_names(
            tyro_config
        )

        # Patch buffer overflow in PhysX GPU narrow phase is sensitive to contact density.
        # Keep a safer default than IsaacLab's default for large multi-env object training.
        gpu_max_rigid_contact_count = getattr(self.simulator_config.sim.physx, "gpu_max_rigid_contact_count", None)
        if gpu_max_rigid_contact_count is None:
            gpu_max_rigid_contact_count = 2**25  # 33554432
        gpu_max_rigid_patch_count = getattr(self.simulator_config.sim.physx, "gpu_max_rigid_patch_count", None)
        if gpu_max_rigid_patch_count is None:
            gpu_max_rigid_patch_count = 20 * 2**15  # 655360
        gpu_found_lost_pairs_capacity = getattr(
            self.simulator_config.sim.physx,
            "gpu_found_lost_pairs_capacity",
            None,
        )
        if gpu_found_lost_pairs_capacity is None:
            gpu_found_lost_pairs_capacity = 2**27  # 134217728
        gpu_found_lost_aggregate_pairs_capacity = getattr(
            self.simulator_config.sim.physx,
            "gpu_found_lost_aggregate_pairs_capacity",
            None,
        )
        if gpu_found_lost_aggregate_pairs_capacity is None:
            gpu_found_lost_aggregate_pairs_capacity = 2**27  # 134217728
        gpu_total_aggregate_pairs_capacity = getattr(
            self.simulator_config.sim.physx,
            "gpu_total_aggregate_pairs_capacity",
            None,
        )
        if gpu_total_aggregate_pairs_capacity is None:
            gpu_total_aggregate_pairs_capacity = 2**24  # 16777216
        gpu_collision_stack_size = getattr(self.simulator_config.sim.physx, "gpu_collision_stack_size", None)
        if gpu_collision_stack_size is None:
            gpu_collision_stack_size = 2**26  # 67108864
        gpu_heap_capacity = getattr(self.simulator_config.sim.physx, "gpu_heap_capacity", None)
        if gpu_heap_capacity is None:
            gpu_heap_capacity = 2**26  # 67108864
        gpu_temp_buffer_capacity = getattr(self.simulator_config.sim.physx, "gpu_temp_buffer_capacity", None)
        if gpu_temp_buffer_capacity is None:
            gpu_temp_buffer_capacity = 2**24  # 16777216
        physx_gpu_buffer_config = {
            "gpu_max_rigid_contact_count": int(gpu_max_rigid_contact_count),
            "gpu_max_rigid_patch_count": int(gpu_max_rigid_patch_count),
            "gpu_found_lost_pairs_capacity": int(gpu_found_lost_pairs_capacity),
            "gpu_found_lost_aggregate_pairs_capacity": int(gpu_found_lost_aggregate_pairs_capacity),
            "gpu_total_aggregate_pairs_capacity": int(gpu_total_aggregate_pairs_capacity),
            "gpu_collision_stack_size": int(gpu_collision_stack_size),
            "gpu_heap_capacity": int(gpu_heap_capacity),
            "gpu_temp_buffer_capacity": int(gpu_temp_buffer_capacity),
        }

        sim_config: SimulationCfg = SimulationCfg(
            dt=1.0 / self.simulator_config.sim.fps,
            render_interval=self.simulator_config.sim.render_interval,
            device=self.sim_device,
            physx=PhysxCfg(
                bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                solver_type=self.simulator_config.sim.physx.solver_type,
                max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations,
                gpu_max_rigid_contact_count=physx_gpu_buffer_config["gpu_max_rigid_contact_count"],
                gpu_max_rigid_patch_count=physx_gpu_buffer_config["gpu_max_rigid_patch_count"],
                gpu_found_lost_pairs_capacity=physx_gpu_buffer_config["gpu_found_lost_pairs_capacity"],
                gpu_found_lost_aggregate_pairs_capacity=physx_gpu_buffer_config[
                    "gpu_found_lost_aggregate_pairs_capacity"
                ],
                gpu_total_aggregate_pairs_capacity=physx_gpu_buffer_config["gpu_total_aggregate_pairs_capacity"],
                gpu_collision_stack_size=physx_gpu_buffer_config["gpu_collision_stack_size"],
                gpu_heap_capacity=physx_gpu_buffer_config["gpu_heap_capacity"],
                gpu_temp_buffer_capacity=physx_gpu_buffer_config["gpu_temp_buffer_capacity"],
            ),
            # Global physics material, can be overridden by the individual articulation
            # Can be inspected by:
            # materials = self._robot.root_physx_view.get_material_properties()
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # default is 0.5
                dynamic_friction=1.0,  # default is 0.5
                restitution=0.0,
            ),
        )
        for config_name, config_value in physx_gpu_buffer_config.items():
            logger.info("PhysX {} set to {}", config_name, config_value)

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1.0 / self.simulator_config.sim.fps}")
        logger.info(
            f"\tRendering step-size   : {1.0 / self.simulator_config.sim.fps * self.simulator_config.sim.substeps}"
        )

        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each "
                "environment step. If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        replicate_physics = self.simulator_config.scene.replicate_physics
        object_cfg = getattr(self.robot_config, "object", None)
        object_path_spec = str(getattr(object_cfg, "object_urdf_path", "") or "").strip()
        if getattr(object_cfg, "enabled", False) and object_path_spec:
            self._resolved_training_object_specs = self._resolve_training_object_specs(object_path_spec)
            self._training_object_use_box_primitives = self._should_use_box_primitives(
                self._resolved_training_object_specs
            )
            self._heterogeneous_object_env_assignment = len(self._resolved_training_object_specs) > 1
            disable_single_slot = os.environ.get(
                "HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            self._heterogeneous_object_single_slot_enabled = (
                self._heterogeneous_object_env_assignment
                and not disable_single_slot
                and (
                    self._training_object_use_box_primitives
                    or self._can_use_single_slot_heterogeneous_objects(self._resolved_training_object_specs)
                )
            )
            if disable_single_slot and self._heterogeneous_object_env_assignment:
                logger.info("Disabled heterogeneous single-slot object spawning via env override.")
            if self._heterogeneous_object_single_slot_enabled and replicate_physics:
                logger.warning(
                    "Detected {} training objects for object generalist. "
                    "Forcing InteractiveScene.replicate_physics=False so each env can keep a single fixed object.",
                    len(self._resolved_training_object_specs),
                )
                replicate_physics = False
            elif self._heterogeneous_object_env_assignment:
                logger.info(
                    "Detected {} training objects for object generalist. "
                    "Using one object slot per asset across all envs because shared-slot multi-asset spawning "
                    "is not stable for mixed URDF banks.",
                    len(self._resolved_training_object_specs),
                )

        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.training_config.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            replicate_physics=replicate_physics,
        )
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
            self._apply_physx_gpu_collision_stack_size()
        print("[INFO]: Scene manager: ", self.scene)

        viewer_config: ViewerCfg = ViewerCfg()
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller: ViewportCameraController | None = ViewportCameraController(
                self, viewer_config
            )
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore[attr-defined]
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()

        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        self.base_com_bias = torch.zeros((self.training_config.num_envs, 3), dtype=torch.float, device="cpu")

        self.events_cfg = EventCfg()

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)

        # # extend UI elements
        # # we need to do this here after all the managers are initialized
        # # this is because they dictate the sensors and commands right now
        # if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
        #     self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        # else:
        #     # if no window, then we don't need to store the window
        #     self._window = None

        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of
        # the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation

        self._sim_step_counter = 0

        if self.video_config.enabled:
            self.video_recorder = IsaacSimVideoRecorder(self.video_config, self)

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()

        # print the environment information

        logger.info("Completed setting up the environment...")

    @staticmethod
    def _sanitize_object_name(name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())
        cleaned = cleaned.strip("_")
        return cleaned or "object"

    def _resolve_object_specs(self, object_path_spec: str) -> list[tuple[str, str]]:
        """Resolve object asset specification into unique (object_name, urdf_path) pairs."""
        resolved = resolve_data_file_path(object_path_spec)
        path = pathlib.Path(resolved)

        raw_specs: list[tuple[str, str]] = []
        if path.is_file() and path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
                payload = payload["clips"]
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid object spec json: {path}")
            for entry in payload.values():
                if isinstance(entry, str):
                    urdf_path = entry.strip()
                    if urdf_path:
                        if not pathlib.Path(urdf_path).is_absolute() and not urdf_path.startswith("holosoma/data"):
                            urdf_path = str((path.parent / urdf_path).resolve())
                        raw_specs.append(("", urdf_path))
                    continue
                if not isinstance(entry, dict):
                    continue
                urdf_path = str(entry.get("object_urdf_path", "")).strip()
                obj_name = str(entry.get("object_name", "")).strip()
                if urdf_path:
                    if not pathlib.Path(urdf_path).is_absolute() and not urdf_path.startswith("holosoma/data"):
                        urdf_path = str((path.parent / urdf_path).resolve())
                    raw_specs.append((obj_name, urdf_path))
        elif path.is_dir():
            for urdf in sorted(list(path.rglob("*.urdf")) + list(path.rglob("*.URDF"))):
                raw_specs.append((urdf.stem, str(urdf)))
        else:
            if path.suffix.lower() != ".urdf":
                raise ValueError(f"Object path must be a URDF file, directory, or json map: {resolved}")
            raw_specs.append((path.stem, str(path)))

        unique_specs: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        used_names: set[str] = set()
        for obj_name, urdf_path in raw_specs:
            urdf_resolved = _resolve_existing_object_urdf_path(urdf_path).resolve()
            urdf_key = str(urdf_resolved)
            if urdf_key in seen_paths:
                continue
            seen_paths.add(urdf_key)

            base_name = self._sanitize_object_name(obj_name if obj_name else urdf_resolved.stem)
            name = base_name
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base_name}_{suffix}"
            used_names.add(name)
            unique_specs.append((name, urdf_key))

        return unique_specs

    @staticmethod
    def _scalar_str(value: Any) -> str:
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.shape == ():
            item = arr.item()
        else:
            item = arr.reshape(-1)[0]
            if hasattr(item, "item"):
                item = item.item()
        return str(item).strip()

    @staticmethod
    def _decode_h5_strings(values: np.ndarray) -> list[str]:
        decoded: list[str] = []
        for item in values:
            if isinstance(item, (bytes, np.bytes_)):
                decoded.append(item.decode("utf-8"))
            else:
                decoded.append(str(item))
        return decoded

    @staticmethod
    def _resolve_motion_object_urdf_path(raw_path: str, *, base_dir: pathlib.Path) -> str:
        path_str = str(raw_path).strip()
        if not path_str:
            return ""
        candidate = pathlib.Path(path_str)
        if not candidate.is_absolute() and not path_str.startswith("holosoma/data"):
            candidate = (base_dir / path_str).resolve()
            if candidate.exists():
                return str(candidate)

            # OMOMO carry banks may store retargeting asset paths like
            # "models/largebox/largebox.urdf" relative to the repo retargeting root
            # instead of the converted clip directory.
            repo_root = pathlib.Path(__file__).resolve().parents[5]
            fallback_candidates = (
                repo_root / "src" / "holosoma_retargeting" / path_str,
                repo_root / "src" / "holosoma_retargeting_my" / path_str,
            )
            for fallback in fallback_candidates:
                fallback_resolved = fallback.resolve()
                if fallback_resolved.exists():
                    return str(fallback_resolved)

            return str(_resolve_existing_object_urdf_path(candidate))
        return str(_resolve_existing_object_urdf_path(path_str))

    @classmethod
    def _load_clip_object_metadata_map(cls, motion_dir: pathlib.Path) -> dict[str, dict[str, str]]:
        candidate_files = (
            motion_dir / "_clip_object_urdf_map.json",
            motion_dir / "clip_object_urdf_map.json",
        )
        for path in candidate_files:
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to parse clip-object metadata map '{}': {}", path, exc)
                return {}

            if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
                payload = payload["clips"]
            if not isinstance(payload, dict):
                logger.warning("Invalid clip-object metadata map format in '{}': expected dict.", path)
                return {}

            normalized: dict[str, dict[str, str]] = {}
            for clip_id, entry in payload.items():
                if not isinstance(clip_id, str):
                    continue
                if isinstance(entry, str):
                    normalized[clip_id] = {"object_name": "", "object_urdf_path": entry.strip()}
                elif isinstance(entry, dict):
                    normalized[clip_id] = {
                        "object_name": str(entry.get("object_name", "")).strip(),
                        "object_urdf_path": str(entry.get("object_urdf_path", "")).strip(),
                    }
            logger.info("Loaded clip-object metadata map '{}' ({} entries).", path, len(normalized))
            return normalized
        return {}

    @classmethod
    def _extract_object_clip_metadata(
        cls,
        *,
        data: Any,
        clip_id: str,
        clip_map: dict[str, dict[str, str]] | None,
        base_dir: pathlib.Path,
    ) -> tuple[str, str]:
        object_name = cls._scalar_str(data["object_name"]) if "object_name" in data else ""
        object_urdf_path = cls._scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""

        if clip_map is not None and clip_id in clip_map:
            mapped = clip_map[clip_id]
            if not object_name:
                object_name = mapped.get("object_name", "").strip()
            if not object_urdf_path:
                object_urdf_path = mapped.get("object_urdf_path", "").strip()

        if object_urdf_path:
            object_urdf_path = cls._resolve_motion_object_urdf_path(object_urdf_path, base_dir=base_dir)
        if not object_name and object_urdf_path:
            object_name = pathlib.Path(object_urdf_path).stem
        if not object_name:
            object_name = "object"
        return object_name, object_urdf_path

    @staticmethod
    def _unique_preserve_order(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))

    def _resolve_motion_config(self) -> MotionConfig | None:
        command_cfg = getattr(self, "command_config", None)
        if command_cfg is None:
            return None
        setup_terms = getattr(command_cfg, "setup_terms", None)
        if not isinstance(setup_terms, dict):
            return None
        for term_cfg in setup_terms.values():
            func = str(getattr(term_cfg, "func", "")).strip()
            if "MotionCommand" not in func:
                continue
            params = getattr(term_cfg, "params", None)
            if not isinstance(params, dict) or "motion_config" not in params:
                continue
            motion_cfg = params["motion_config"]
            if isinstance(motion_cfg, MotionConfig):
                return motion_cfg
            if isinstance(motion_cfg, dict):
                return MotionConfig(**motion_cfg)
        return None

    @staticmethod
    def _resolve_selected_motion_paths(
        motion_dir: pathlib.Path,
        *,
        motion_clip_id: int | None,
        motion_clip_name: str | None,
    ) -> list[pathlib.Path]:
        files = sorted(motion_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in motion directory: {motion_dir}")
        if motion_clip_name is not None:
            matches = [path for path in files if path.stem == motion_clip_name]
            if not matches:
                raise ValueError(f"Clip name '{motion_clip_name}' not found in {motion_dir}")
            return matches
        if motion_clip_id is not None:
            clip_idx = int(motion_clip_id)
            if clip_idx < 0 or clip_idx >= len(files):
                raise IndexError(f"Clip index {clip_idx} out of range for {motion_dir}")
            return [files[clip_idx]]
        return files

    def _resolve_motion_subset_object_urdfs_from_npz(self, motion_path: pathlib.Path, motion_cfg: MotionConfig) -> list[str]:
        if motion_path.is_dir():
            selected_paths = self._resolve_selected_motion_paths(
                motion_path,
                motion_clip_id=motion_cfg.motion_clip_id,
                motion_clip_name=motion_cfg.motion_clip_name,
            )
            clip_map = self._load_clip_object_metadata_map(motion_path)
        else:
            selected_paths = [motion_path]
            clip_map = self._load_clip_object_metadata_map(motion_path.parent)

        object_urdfs: list[str] = []
        for clip_path in selected_paths:
            if not zipfile.is_zipfile(clip_path):
                if len(selected_paths) == 1:
                    raise zipfile.BadZipFile(f"Invalid motion npz archive: {clip_path}")
                logger.warning("Skipping invalid motion npz archive while resolving object metadata: {}", clip_path)
                continue
            try:
                with np.load(clip_path, allow_pickle=True) as data:
                    _object_name, object_urdf = self._extract_object_clip_metadata(
                        data=data,
                        clip_id=clip_path.stem,
                        clip_map=clip_map,
                        base_dir=clip_path.parent,
                    )
            except Exception as exc:
                if len(selected_paths) == 1:
                    raise
                logger.warning(
                    "Skipping motion clip '{}' while resolving object metadata: {}",
                    clip_path,
                    exc,
                )
                continue
            if object_urdf:
                object_urdfs.append(object_urdf)
        return self._unique_preserve_order(object_urdfs)

    @staticmethod
    def _get_h5_attr_or_dataset(h5f: Any, name: str) -> np.ndarray | None:
        if name in h5f.attrs:
            return np.asarray(h5f.attrs[name])
        if f"/{name}" in h5f.attrs:
            return np.asarray(h5f.attrs[f"/{name}"])
        if name in h5f:
            return np.asarray(h5f[name])
        if f"/{name}" in h5f:
            return np.asarray(h5f[f"/{name}"])
        return None

    @classmethod
    def _resolve_h5_clip_metadata_values(
        cls,
        h5f: Any,
        *,
        clip_ids: list[str],
        selected_clip_indices: list[int],
        field_names: tuple[str, ...],
    ) -> list[str]:
        containers = []
        if "clips" in h5f:
            containers.append(h5f["clips"])
        if "meta" in h5f:
            containers.append(h5f["meta"])
        containers.append(h5f)

        raw_values = None
        for container in containers:
            for field_name in field_names:
                raw_values = cls._get_h5_attr_or_dataset(container, field_name)
                if raw_values is not None:
                    break
            if raw_values is not None:
                break

        if raw_values is not None:
            arr = np.asarray(raw_values)
            if arr.shape == ():
                return [cls._scalar_str(arr)] * len(selected_clip_indices)
            flat = arr.reshape(-1)
            if flat.shape[0] >= max(selected_clip_indices, default=0) + 1:
                return [cls._scalar_str(flat[idx]) for idx in selected_clip_indices]

        clips_group = h5f["clips"] if "clips" in h5f else None
        if clips_group is not None:
            nested_values: list[str] = []
            for clip_idx in selected_clip_indices:
                clip_id = clip_ids[clip_idx]
                clip_group = clips_group.get(clip_id, None)
                if clip_group is None:
                    return []
                clip_value = None
                for field_name in field_names:
                    clip_value = cls._get_h5_attr_or_dataset(clip_group, field_name)
                    if clip_value is not None:
                        break
                if clip_value is None:
                    return []
                nested_values.append(cls._scalar_str(clip_value))
            return nested_values

        return []

    def _resolve_motion_subset_object_urdfs_from_h5(self, motion_path: pathlib.Path, motion_cfg: MotionConfig) -> list[str] | None:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("h5py is required to load HDF5 motion metadata.") from exc

        with h5py.File(motion_path, "r") as h5f:
            clip_ids: list[str]
            selected_clip_indices: list[int]
            if "clips" in h5f and "clip_ids" in h5f["clips"]:
                clip_ids = self._decode_h5_strings(np.asarray(h5f["clips"]["clip_ids"]))
                if motion_cfg.motion_clip_name is not None:
                    if motion_cfg.motion_clip_name not in clip_ids:
                        raise ValueError(f"Clip name '{motion_cfg.motion_clip_name}' not found in HDF5 motion file.")
                    selected_clip_indices = [clip_ids.index(motion_cfg.motion_clip_name)]
                elif motion_cfg.motion_clip_id is not None:
                    clip_idx = int(motion_cfg.motion_clip_id)
                    if clip_idx < 0 or clip_idx >= len(clip_ids):
                        raise IndexError(f"Clip index {clip_idx} out of range for HDF5 motion file.")
                    selected_clip_indices = [clip_idx]
                else:
                    selected_clip_indices = list(range(len(clip_ids)))
            else:
                clip_ids = [motion_path.stem]
                selected_clip_indices = [0]

            object_urdfs_raw = self._resolve_h5_clip_metadata_values(
                h5f,
                clip_ids=clip_ids,
                selected_clip_indices=selected_clip_indices,
                field_names=("object_urdf_path", "object_urdf_paths"),
            )
            if not object_urdfs_raw:
                return None

            resolved_urdfs = [
                self._resolve_motion_object_urdf_path(raw_urdf, base_dir=motion_path.parent)
                for raw_urdf in object_urdfs_raw
                if str(raw_urdf).strip()
            ]
            return self._unique_preserve_order(resolved_urdfs)

    def _resolve_motion_subset_object_urdfs(self) -> list[str] | None:
        motion_cfg = self._resolve_motion_config()
        if motion_cfg is None:
            return None

        motion_path = pathlib.Path(resolve_data_file_path(motion_cfg.motion_file))
        suffix = motion_path.suffix.lower()
        if motion_path.is_dir() or suffix == ".npz":
            return self._resolve_motion_subset_object_urdfs_from_npz(motion_path, motion_cfg)
        if suffix in {".h5", ".hdf5"}:
            return self._resolve_motion_subset_object_urdfs_from_h5(motion_path, motion_cfg)
        return None

    def _resolve_training_object_specs(self, object_path_spec: str) -> list[tuple[str, str]]:
        object_specs = self._resolve_object_specs(object_path_spec)
        motion_object_urdfs = self._resolve_motion_subset_object_urdfs()
        if not motion_object_urdfs:
            return object_specs

        object_spec_by_urdf = {urdf_path: (object_name, urdf_path) for object_name, urdf_path in object_specs}
        filtered_specs: list[tuple[str, str]] = []
        missing_motion_urdfs: list[str] = []
        for urdf_path in motion_object_urdfs:
            spec = object_spec_by_urdf.get(urdf_path)
            if spec is None:
                missing_motion_urdfs.append(urdf_path)
                continue
            filtered_specs.append(spec)

        if not filtered_specs:
            raise RuntimeError(
                "Failed to resolve any active training objects from the intersection of "
                f"object spec '{object_path_spec}' and motion metadata."
            )
        if missing_motion_urdfs:
            logger.warning(
                "Ignoring {} motion-bank URDF(s) that are absent from object spec '{}'. Sample: {}",
                len(missing_motion_urdfs),
                object_path_spec,
                missing_motion_urdfs[:4],
            )
        if len(filtered_specs) != len(object_specs):
            logger.info(
                "Filtered training object bank from {} to {} URDF(s) using motion-file metadata.",
                len(object_specs),
                len(filtered_specs),
            )
        return filtered_specs

    @staticmethod
    def _build_env_object_urdf_assignment(
        object_specs: list[tuple[str, str]],
        *,
        num_envs: int,
    ) -> list[str]:
        if not object_specs:
            return []
        return [object_specs[env_id % len(object_specs)][1] for env_id in range(num_envs)]

    @staticmethod
    def _resolve_object_spawn_mode() -> tuple[str, bool]:
        raw_mode = os.environ.get("HOLOSOMA_OBJECT_SPAWN_MODE")
        raw_mode_normalized = "" if raw_mode is None else raw_mode.strip().lower()
        if raw_mode_normalized in {"", "primitive", "primitives", "box", "cuboid"}:
            return "primitive", bool(raw_mode_normalized)
        if raw_mode_normalized == "auto":
            return "auto", True
        if raw_mode_normalized in {"urdf", "mesh", "off", "disable", "disabled"}:
            return "urdf", True
        logger.warning(
            "Unknown HOLOSOMA_OBJECT_SPAWN_MODE='{}'. Falling back to 'primitive'.",
            raw_mode,
        )
        return "primitive", bool(raw_mode_normalized)

    def _should_use_box_primitives(self, object_specs: list[tuple[str, str]]) -> bool:
        self._training_object_box_metadata_by_urdf = {}
        spawn_mode, spawn_mode_explicit = self._resolve_object_spawn_mode()
        if spawn_mode == "urdf" or not object_specs:
            return False

        metadata_by_urdf: dict[str, UrdfBoxPrimitiveMetadata] = {}
        for _object_name, object_path in object_specs:
            metadata = load_urdf_box_primitive_metadata(object_path)
            if metadata is None:
                if spawn_mode == "primitive" and spawn_mode_explicit:
                    raise ValueError(
                        "HOLOSOMA_OBJECT_SPAWN_MODE=primitive requires every training object URDF to be a "
                        f"simple box-like asset. Failed on: {object_path}"
                    )
                return False
            metadata_by_urdf[object_path] = metadata

        self._training_object_box_metadata_by_urdf = metadata_by_urdf
        logger.info(
            "Using Isaac Sim cuboid primitives for {} training object URDF(s).",
            len(metadata_by_urdf),
        )
        return True

    @staticmethod
    def _apply_object_scale_to_extents(
        extents: tuple[float, float, float],
        object_scale: tuple[float, float, float] | None,
    ) -> tuple[float, float, float]:
        if object_scale is None:
            return extents
        return tuple(float(extents[idx]) * float(object_scale[idx]) for idx in range(3))

    def _build_box_primitive_spawn_cfg(
        self,
        object_asset_urdf_path: str,
        *,
        object_scale: tuple[float, float, float] | None,
    ) -> sim_utils.CuboidCfg | None:
        metadata = self._training_object_box_metadata_by_urdf.get(object_asset_urdf_path)
        if metadata is None:
            return None
        size = self._apply_object_scale_to_extents(metadata.extents, object_scale)
        visual_color = metadata.visual_color if metadata.visual_color is not None else (0.7, 0.8, 0.9)
        return sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=visual_color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=metadata.static_friction,
                dynamic_friction=metadata.dynamic_friction,
                restitution=metadata.restitution,
                compliant_contact_stiffness=metadata.compliant_contact_stiffness,
                compliant_contact_damping=metadata.compliant_contact_damping,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=metadata.mass),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        )

    def _build_object_spawn_cfg(
        self,
        object_asset_urdf_path: str,
        *,
        object_scale: tuple[float, float, float] | None,
    ) -> sim_utils.UrdfFileCfg | sim_utils.CuboidCfg:
        if self._training_object_use_box_primitives:
            primitive_cfg = self._build_box_primitive_spawn_cfg(
                object_asset_urdf_path,
                object_scale=object_scale,
            )
            if primitive_cfg is not None:
                return primitive_cfg

        return sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=object_asset_urdf_path,
            scale=object_scale,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        )

    @staticmethod
    def _can_use_single_slot_heterogeneous_objects(object_specs: list[tuple[str, str]]) -> bool:
        """Whether heterogeneous objects can safely share one `RigidObject` slot.

        Mixed URDF banks often expand to different rigid-body prim hierarchies after import. IsaacLab's
        `RigidObject` view resolution assumes that every matched instance shares the same hierarchy as env_0.
        When that assumption is false, the resulting PhysX view only binds a subset of environments, which then
        explodes later as an env/object instance-count mismatch during state reads.

        However, many generated box banks differ only in mesh extents and inertia while keeping the same
        rigid-body/link hierarchy (for example, a single `baseLink` box URDF). Those banks are safe to spawn
        through a single shared object slot, which is much cheaper than instantiating every object in every env.
        """
        disable_flag = os.environ.get("HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT", "").strip().lower()
        if disable_flag in {"1", "true", "yes", "on"}:
            logger.info("Disabled heterogeneous single-slot object spawning via env override.")
            return False

        force_flag = os.environ.get("HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT", "").strip().lower()
        if force_flag in {"1", "true", "yes", "on"}:
            logger.warning("Forcing heterogeneous single-slot object spawning via env override.")
            return True

        if len(object_specs) <= 1:
            return False

        reference_name, reference_path = object_specs[0]
        try:
            reference_signature = extract_urdf_topology_signature(reference_path)
        except Exception as exc:
            logger.warning(
                "Failed to inspect heterogeneous object URDF '{}' ({}); falling back to per-asset object slots.",
                reference_path,
                exc,
            )
            return False

        for object_name, object_path in object_specs[1:]:
            try:
                current_signature = extract_urdf_topology_signature(object_path)
            except Exception as exc:
                logger.warning(
                    "Failed to inspect heterogeneous object URDF '{}' ({}); falling back to per-asset object slots.",
                    object_path,
                    exc,
                )
                return False

            if current_signature != reference_signature:
                logger.info(
                    "Heterogeneous object bank requires per-asset slots: topology mismatch '{}' ({}) vs '{}' ({}).",
                    reference_name,
                    reference_path,
                    object_name,
                    object_path,
                )
                return False

        logger.info(
            "Heterogeneous object bank is topology-compatible across {} URDFs; enabling single-slot spawning.",
            len(object_specs),
        )
        return True

    def _setup_scene(self) -> None:
        self._load_scene_config()

        robot_asset_cfg = self.robot_config.asset

        asset_root = robot_asset_cfg.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())

        robot_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=robot_asset_cfg.linear_damping,
            angular_damping=robot_asset_cfg.angular_damping,
            max_linear_velocity=robot_asset_cfg.max_linear_velocity,
            max_angular_velocity=robot_asset_cfg.max_angular_velocity,
            max_depenetration_velocity=1.0,
        )

        robot_articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=robot_asset_cfg.enable_self_collisions,
            # NOTE: (4, 0) -> (8, 4) necessary for reproducing FAR-tracking-implementation
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )

        if robot_asset_cfg.usd_file is None:
            # convert from urdf dynamically
            asset_path = robot_asset_cfg.urdf_file
            full_urdf_path = os.path.abspath(os.path.join(asset_root, asset_path))

            # Get local rank to avoid race conditions in multi-GPU setups
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            usd_conversion_dir = os.path.abspath(os.path.join(asset_root, f"converted_rank{local_rank}"))

            spawn = sim_utils.UrdfFileCfg(
                usd_dir=usd_conversion_dir,
                asset_path=full_urdf_path,
                fix_base=robot_asset_cfg.fix_base_link,
                merge_fixed_joints=robot_asset_cfg.collapse_fixed_joints,
                replace_cylinders_with_capsules=robot_asset_cfg.replace_cylinder_with_capsule,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0,
                        damping=0,
                    ),
                    target_type="none",
                ),
                activate_contact_sensors=True,
                rigid_props=robot_rigid_props,
                articulation_props=robot_articulation_props,
            )
        else:
            asset_path = robot_asset_cfg.usd_file
            spawn = sim_utils.UsdFileCfg(
                usd_path=os.path.abspath(os.path.join(asset_root, asset_path)),
                activate_contact_sensors=True,
                rigid_props=robot_rigid_props,
                articulation_props=robot_articulation_props,
            )

        # prepare to override the articulation configuration in
        # holosoma/holosoma/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()},
            joint_vel={".*": 0.0},
        )

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping

        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict:
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")

        # ImplicitActuatorCfg IdealPDActuatorCfg
        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                # effort_limit_sim=dof_effort_limit_list[i],
                # velocity_limit_sim=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            )
            for i in range(len(dof_names_list))
        }

        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(
            prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators
        )

        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=self.simulator_config.contact_sensor_history_length,
            update_period=0.005,
            track_air_time=True,
            force_threshold=10.0,
            debug_vis=True,
        )

        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        terrain_prim_path = "/World/ground"
        ground_plane_collision_prim_path = "/World/ground_plane_collision"
        height_scanner_mesh_paths = [terrain_prim_path]
        if terrain_state.mesh_type == "load_obj" and bool(getattr(terrain_state, "add_ground_plane_collision", False)):
            height_scanner_mesh_paths.append(ground_plane_collision_prim_path)
        height_scanner_config = None
        if terrain_state.mesh_type not in ["fake", None]:
            # Add a height scanner to the torso to detect the height of the terrain mesh
            # TODO: Scene USD files need ground mapping
            height_scanner_config = RayCasterCfg(
                prim_path=f"/World/envs/env_.*/Robot/{self.robot_config.body_names[0]}",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                attach_yaw_only=True,
                # Apply a grid pattern that is smaller than the resolution to only return one height value.
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
                debug_vis=False,
                mesh_prim_paths=height_scanner_mesh_paths,
            )

        global_collision_prims = []
        if terrain_state.mesh_type == "plane":
            terrain_config = TerrainImporterCfg(
                prim_path=terrain_prim_path,
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=terrain_state.static_friction,
                    dynamic_friction=terrain_state.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
            terrain_config.class_type(terrain_config)
            global_collision_prims.append(terrain_config.prim_path)
            self._add_debug_grid(terrain_state.mesh.bounds if terrain_state.mesh is not None else None)
        elif terrain_state.mesh_type in ["trimesh", "load_obj"]:
            self.terrain = self.terrain_manager.get_state("locomotion_terrain").terrain
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=terrain_state.static_friction,
                dynamic_friction=terrain_state.dynamic_friction,
                restitution=terrain_state.restitution,
            )

            create_prim_from_mesh(
                terrain_prim_path,
                self.terrain.mesh,
                visual_material=visual_material,
                physics_material=physics_material,
                translation=(0.0, 0.0, 0.0),
            )
            global_collision_prims.append(terrain_prim_path)
            print("[INFO] Successfully created custom terrain mesh")
            self._add_debug_grid(self.terrain.mesh.bounds if self.terrain is not None else None)

            if terrain_state.mesh_type == "load_obj" and bool(
                getattr(terrain_state, "add_ground_plane_collision", False)
            ):
                ground_plane_config = TerrainImporterCfg(
                    prim_path=ground_plane_collision_prim_path,
                    terrain_type="plane",
                    collision_group=-1,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=terrain_state.static_friction,
                        dynamic_friction=terrain_state.dynamic_friction,
                        restitution=0.0,
                    ),
                    debug_vis=False,
                )
                ground_plane_config.num_envs = self.scene.cfg.num_envs
                ground_plane_config.env_spacing = self.scene.cfg.env_spacing
                ground_plane_config.class_type(ground_plane_config)
                global_collision_prims.append(ground_plane_collision_prim_path)
                logger.info("Added fallback ground plane collision under load_obj terrain and exposed it to the height scanner.")
        else:
            raise ValueError(f"Unsupported terrain mesh type: {terrain_state.mesh_type}")

        self._robot = Articulation(robot_articulation_config)

        if os.environ.get("HOLOSOMA_DEBUG_ROBOT_PRIMS") == "1":
            print_prim_tree("/World/envs/env_0/Robot")
            log_robot_properties("/World/envs/env_0/Robot", "*")

        self.scene.articulations["robot"] = self._robot

        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        if height_scanner_config:
            self._height_scanner = RayCaster(height_scanner_config)
            self.scene.sensors["height_scanner"] = self._height_scanner

        # add training object(s) before collision filtering so they are included in env isolation.
        if getattr(self.robot_config.object, "enabled", False) and self.robot_config.object.object_urdf_path:
            object_specs = self._resolved_training_object_specs
            if not object_specs:
                object_specs = self._resolve_object_specs(self.robot_config.object.object_urdf_path)
            if not object_specs:
                raise ValueError(
                    f"No valid object URDFs resolved from: {self.robot_config.object.object_urdf_path}"
                )

            object_scale = None
            object_scale_raw = getattr(self.robot_config.object, "scale", None)
            if object_scale_raw is not None:
                if len(object_scale_raw) == 1:
                    value = float(object_scale_raw[0])
                    object_scale = (value, value, value)
                elif len(object_scale_raw) == 3:
                    object_scale = tuple(float(v) for v in object_scale_raw)
                else:
                    raise ValueError(
                        "robot.object.scale must have length 1 or 3. "
                        f"Got: {object_scale_raw}"
                    )

            self._object_urdf_by_name = {}
            self._env_object_urdf_paths = []
            self._object_contact_filter_prim_paths_expr = []
            if self._heterogeneous_object_env_assignment and self._heterogeneous_object_single_slot_enabled:
                object_assets_cfg = [
                    self._build_object_spawn_cfg(object_asset_urdf_path, object_scale=object_scale)
                    for _, object_asset_urdf_path in object_specs
                ]
                multi_asset_cfg = sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=object_assets_cfg,
                    random_choice=False,
                    activate_contact_sensors=True,
                )
                object_cfg = RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Object",
                    spawn=multi_asset_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.5),
                    ),
                )
                rigid_object = RigidObject(object_cfg)
                self.scene.rigid_objects["object"] = rigid_object
                self._object_urdf_by_name["object"] = ""
                self._env_object_urdf_paths = self._build_env_object_urdf_assignment(
                    object_specs,
                    num_envs=self.training_config.num_envs,
                )
                if self._training_object_use_box_primitives:
                    self._object_contact_filter_prim_paths_expr.append("{ENV_REGEX_NS}/Object")
                else:
                    self._object_contact_filter_prim_paths_expr.append("{ENV_REGEX_NS}/Object/baseLink")
                logger.info(
                    "Loaded heterogeneous training object bank: {} unique URDF(s) assigned across {} envs.",
                    len(object_specs),
                    self.training_config.num_envs,
                )
            else:
                use_single_name = len(object_specs) == 1
                for idx, (raw_name, object_asset_urdf_path) in enumerate(object_specs):
                    object_name = "object" if use_single_name else raw_name
                    prim_suffix = "Object" if use_single_name else f"Object_{idx}_{object_name}"
                    object_cfg = RigidObjectCfg(
                        prim_path=f"/World/envs/env_.*/{prim_suffix}",
                        spawn=self._build_object_spawn_cfg(object_asset_urdf_path, object_scale=object_scale),
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(0.0, 0.0, 0.5),
                        ),
                    )
                    rigid_object = RigidObject(object_cfg)
                    self.scene.rigid_objects[object_name] = rigid_object
                    self._object_urdf_by_name[object_name] = str(pathlib.Path(object_asset_urdf_path).resolve())
                    if self._training_object_use_box_primitives:
                        self._object_contact_filter_prim_paths_expr.append(f"{{ENV_REGEX_NS}}/{prim_suffix}")
                    else:
                        # The current URDF box assets expose a single rigid body under `baseLink`.
                        # Filter against that rigid body prim instead of the Xform root or collision child.
                        self._object_contact_filter_prim_paths_expr.append(f"{{ENV_REGEX_NS}}/{prim_suffix}/baseLink")

                logger.info(
                    "Loaded {} training object URDF(s): {}",
                    len(self._object_urdf_by_name),
                    list(self._object_urdf_by_name.keys()),
                )
            self._setup_object_contact_sensors()

        # clone, filter, and replicate
        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)

        if hasattr(self.simulator_config.scene, "usd_file"):
            # Activate collisions with the entire scene
            global_collision_prims.append("/World/scene")

        self.scene.filter_collisions(global_prim_paths=global_collision_prims)

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))

    def _apply_physx_gpu_collision_stack_size(self) -> None:
        size_bytes = getattr(self.simulator_config.sim.physx, "gpu_collision_stack_size", None)
        if size_bytes is None:
            return
        try:
            import omni.usd  # noqa: PLC0415
            from pxr import PhysxSchema, Sdf, UsdPhysics  # noqa: PLC0415
        except Exception as exc:
            logger.warning("PhysX collision stack size not applied (missing USD bindings): {}", exc)
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            logger.warning("PhysX collision stack size not applied (USD stage unavailable).")
            return

        scene_prim = None
        for prim in stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                scene_prim = prim
                break

        if scene_prim is None:
            logger.warning("PhysX collision stack size not applied (PhysicsScene not found).")
            return

        physx_scene_api = PhysxSchema.PhysxSceneAPI(scene_prim)
        if not physx_scene_api:
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

        try:
            attr = physx_scene_api.CreateGpuCollisionStackSizeAttr()
        except Exception:
            attr = scene_prim.CreateAttribute("physxScene:gpuCollisionStackSize", Sdf.ValueTypeNames.Int)

        attr.Set(int(size_bytes))
        logger.info("PhysX gpuCollisionStackSize set to {}", int(size_bytes))

    def _add_debug_grid(self, bounds: np.ndarray | None) -> None:
        """Add a visual-only checkerboard at z=0 for alignment debugging."""
        if bounds is None:
            return
        if not self.simulator_config.debug_viz:
            return
        if self.training_config.headless:
            return
        if self.sim.render_mode < self.sim.RenderMode.PARTIAL_RENDERING:
            return

        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except Exception:
            return

        min_corner = np.asarray(bounds[0], dtype=np.float64)
        max_corner = np.asarray(bounds[1], dtype=np.float64)
        span = max_corner - min_corner
        if span[0] <= 0.0 or span[1] <= 0.0:
            return

        stage = omni.usd.get_context().get_stage()
        root_path = "/World/ground_chessboard"
        if stage.GetPrimAtPath(root_path).IsValid():
            return
        stage.DefinePrim(root_path, "Xform")

        tile_size = 1.0
        max_tiles = 40
        tiles_x = int(math.ceil(span[0] / tile_size))
        tiles_y = int(math.ceil(span[1] / tile_size))
        scale = max(tiles_x / max_tiles, tiles_y / max_tiles, 1.0)
        if scale > 1.0:
            tile_size *= scale
            tiles_x = int(math.ceil(span[0] / tile_size))
            tiles_y = int(math.ceil(span[1] / tile_size))

        start_x = math.floor(min_corner[0] / tile_size) * tile_size
        start_y = math.floor(min_corner[1] / tile_size) * tile_size

        thickness = 0.02
        z = thickness * 0.5
        color_dark = (0.04, 0.18, 0.06)
        color_light = (0.06, 0.22, 0.08)

        for ix in range(tiles_x):
            x = start_x + (ix + 0.5) * tile_size
            for iy in range(tiles_y):
                y = start_y + (iy + 0.5) * tile_size
                color = color_dark if (ix + iy) % 2 == 0 else color_light
                prim_path = f"{root_path}/tile_{ix}_{iy}"
                prim = stage.DefinePrim(prim_path, "Cube")
                xform = UsdGeom.Xformable(prim)
                xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
                xform.AddScaleOp().Set(Gf.Vec3f(tile_size * 0.5, tile_size * 0.5, thickness * 0.5))
                gprim = UsdGeom.Gprim(prim)
                gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
                gprim.CreateDisplayOpacityAttr().Set([1.0])


    def _get_base_body_name(self, preference_order: list[str]) -> str:
        """Get the base body name with fallback logic.

        Args:
            preference_order: List of body names to try in order

        Returns:
            The first body name found in the robot's body list

        Raises:
            ValueError: If none of the preferred body names are found
        """
        _, body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        for preferred_name in preference_order:
            if preferred_name in body_names:
                return preferred_name

        raise ValueError(
            f"None of the preferred base body names {preference_order} found in robot body names: {body_names}"
        )

    def get_supported_scene_formats(self) -> list[str]:
        """See base class.

        IsaacSim-specific notes:
        - Supports USD only currently

        Returns
        -------
        List[str]
            ["usd" ]
        """
        return ["usd"]

    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            try:
                from isaacsim.util.debug_draw import _debug_draw
            except (ImportError, ModuleNotFoundError) as exc:
                logger.warning(
                    "Isaac Sim debug draw is unavailable in this environment; "
                    "continuing without debug draw. Error: {}",
                    exc,
                )
                self.draw = None
            else:
                self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def _load_scene_config(self) -> None:
        """Load scene configuration with proper separation of concerns.

        Handles both scene files (collections) and individual rigid objects.
        Replaces the previous _load_scene_usd method with a more flexible approach
        that supports multiple scene file formats and individual object loading.
        """
        if self.simulator_config.scene is None:
            return

        scene_config = self.simulator_config.scene

        # Load scene files (USD/URDF scene files as collections) - NEW APPROACH
        if scene_config.scene_files:
            self._load_scene_files(scene_config)

        # Load individual rigid objects
        if scene_config.rigid_objects:
            self._load_rigid_objects(scene_config)

    def _load_scene_files(self, scene_config: SceneConfig) -> None:
        """Load scene files (USD/URDF scene files as collections).

        Loads scene files as collections using the USDFileLoader. This is the new
        approach that replaces direct USD file loading with a more flexible system
        that supports multiple scene file formats.

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration containing scene files and asset root path

        Raises
        ------
        ValueError
            If scene_files is an empty list
        """
        if not scene_config.scene_files:  # Empty list
            raise ValueError("scene.scene_files is empty list - remove field or provide scene files")

        usd_loader = USDFileLoader(self.sim, self.scene, self.sim_device)
        scene_collection = usd_loader.load_scene_files(scene_config.scene_files, scene_config.asset_root)

        if scene_collection is not None:
            self.scene.rigid_objects["usd_scene_objects"] = scene_collection

    def _load_rigid_objects(self, scene_config: SceneConfig) -> None:
        """Load individual rigid objects from configuration.

        Loads individual rigid objects using the USDFileLoader and adds them
        to the scene using their configuration names as keys.

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration containing rigid objects and asset root path

        Raises
        ------
        ValueError
            If rigid_objects is an empty list
        """
        if not scene_config.rigid_objects:  # Empty list
            raise ValueError("scene.rigid_objects is empty list - remove field or provide objects")

        usd_loader = USDFileLoader(self.sim, self.scene, self.sim_device)
        individual_objects = usd_loader.load_rigid_objects(scene_config.rigid_objects, scene_config.asset_root)

        # Add individual objects to scene using direct config names
        for obj_name, rigid_object in individual_objects.items():
            self.scene.rigid_objects[obj_name] = rigid_object

    def setup(self):
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self):
        pass

    def load_assets(self):
        """
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        """

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        # isaacsim only support matching joint names without "joint" postfix

        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 1.05),
        #     joint_pos={
        #         ".*_hip_yaw": 0.0,
        #         ".*_hip_roll": 0.0,
        #         ".*_hip_pitch": -0.28,  # -16 degrees
        #         ".*_knee": 0.79,  # 45 degrees
        #         ".*_ankle": -0.52,  # -30 degrees
        #         "torso": 0.0,
        #         ".*_shoulder_pitch": 0.28,
        #         ".*_shoulder_roll": 0.0,
        #         ".*_shoulder_yaw": 0.0,
        #         ".*_elbow": 0.52,
        #     },
        #     joint_vel={".*": 0.0},
        # ),

        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        #     ),
        # ),

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, holosoma config)
        # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        """
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19],
        ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link',
        'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',
        'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link',
        'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link',
        'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link',
        'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link',
        'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link',
        'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link',
        'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        """

        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning(
                "The order of the joint_names in the robot_config does not match the "
                "order of the joint_ids in IsaacSim."
            )

        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), (
            "Number of bodies must be equal to number of body names"
        )
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

        self._contact_to_robot_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(body_name) for body_name in self.body_names],
            device=self.sim_device,
        )

        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(self, num_envs, env_origins, base_init_state):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        return self.scene, self._robot

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        """
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        """
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        if len(indices) == 1:
            return indices[0]
        # multiple bodies found
        logger.warning(f"Multiple bodies found for {body_name}.")
        return indices

    def _setup_object_contact_sensors(self) -> None:
        """Create box-filtered contact sensors for selected robot support bodies."""
        self._object_contact_sensors = {}
        env_regex_ns = getattr(self.scene, "env_regex_ns", "/World/envs/env_.*")
        filter_prim_paths_expr = [
            path.format(ENV_REGEX_NS=env_regex_ns) for path in dict.fromkeys(self._object_contact_filter_prim_paths_expr)
        ]
        if not filter_prim_paths_expr:
            return

        available_body_names = set(getattr(self.robot_config, "body_names", []))
        target_body_names = [
            body_name
            for body_name in self._required_object_contact_sensor_body_names
            if body_name in available_body_names
        ]
        if not target_body_names:
            logger.info("Skipping object-filtered contact sensors; current config does not request any monitored bodies.")
            return

        for body_name in target_body_names:

            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body_name}",
                history_length=self.simulator_config.contact_sensor_history_length,
                update_period=0.005,
                track_air_time=False,
                # Let downstream reward/prior code own the thresholding. A high sensor-side
                # threshold can zero out valid but moderate box-contact forces before they are
                # ever observed by rewards or online contact-prior estimation.
                force_threshold=_OBJECT_CONTACT_SENSOR_FORCE_THRESHOLD,
                debug_vis=False,
                filter_prim_paths_expr=filter_prim_paths_expr,
            )
            sensor_name = f"object_contact_sensor_{body_name}"
            sensor = ContactSensor(sensor_cfg)
            self.scene.sensors[sensor_name] = sensor
            self._object_contact_sensors[body_name] = sensor

        if self._object_contact_sensors:
            logger.info(
                "Created {} object-filtered contact sensor(s): {} with filter paths {}",
                len(self._object_contact_sensors),
                sorted(self._object_contact_sensors.keys()),
                filter_prim_paths_expr,
            )

    def get_object_contact_force_history(self, body_names: list[str] | tuple[str, ...]) -> torch.Tensor:
        """Return object-only contact force history for requested robot bodies.

        Shape: [num_envs, history_length, len(body_names), 3]
        """
        history_len = int(self.simulator_config.contact_sensor_history_length)
        if self._object_contact_sensors:
            first_sensor = next(iter(self._object_contact_sensors.values()))
            first_history = first_sensor.data.force_matrix_w_history
            if first_history is not None:
                history_len = int(first_history.shape[1])
        if not body_names:
            return torch.zeros((self.num_envs, history_len, 0, 3), device=self.device, dtype=torch.float32)

        result = torch.zeros((self.num_envs, history_len, len(body_names), 3), device=self.device, dtype=torch.float32)
        for body_idx, body_name in enumerate(body_names):
            sensor = self._object_contact_sensors.get(body_name)
            if sensor is None:
                continue

            matrix_history = sensor.data.force_matrix_w_history
            if matrix_history is None:
                continue

            # Shape: [E, T, 1, M, 3] -> sum across filtered object prims => [E, T, 3]
            result[:, :, body_idx, :] = matrix_history[:, :, 0, :, :].sum(dim=2)

        return result

    def prepare_sim(self):
        # Wait until play so rigid object collections are initialized
        register_objects(self)

        # Create before state adapter, needs a reference
        self.robot_root_states = RootStatesProxy(self._robot.data.root_state_w)  # (num_envs, 13)

        # Create state adapter after object registry and robot root states are set
        self._state_adapter = IsaacSimStateAdapter(
            device=self.device,
            object_registry=self.object_registry,
            scene=self.scene,
            robot=self._robot,
            robot_states=self.robot_root_states,
        )

        # Create unified access proxy using the state adapter
        self.all_root_states = AllRootStatesProxy(self._state_adapter)

        self.contact_forces_history = torch.zeros(
            self.num_envs, self.simulator_config.contact_sensor_history_length, self.num_bodies, 3, device=self.device
        )

        # Initialize virtual gantry system after object registry setup
        # Initialize virtual gantry using config
        gantry_cfg = self.simulator_config.virtual_gantry
        self.virtual_gantry = create_virtual_gantry(
            sim=self,
            enable=gantry_cfg.enabled,
            attachment_body_names=gantry_cfg.attachment_body_names,
            cfg=gantry_cfg,
        )

        # Initialize bridge system using base class helper
        self._init_bridge()

        # Setup video recording after scene is ready
        if self.video_recorder:
            self.video_recorder.setup_recording()

        # Initialize robot tensors
        self.refresh_sim_tensors()

        if self._object_contact_sensors:
            try:
                first_sensor_name, first_sensor = next(iter(self._object_contact_sensors.items()))
                logger.info(
                    "Object contact sensor '{}' initialized with filter_count={} across {} envs.",
                    first_sensor_name,
                    int(first_sensor.contact_physx_view.filter_count),
                    self.num_envs,
                )
            except Exception as exc:
                logger.warning("Failed to inspect object contact sensor filter_count: {}", exc)

        # Initialize acceleration tensors ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            logger.info("Bridge enabled: initializing acceleration computation tensors")
            self.dof_acc = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.prev_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.base_linear_acc = torch.zeros(self.num_envs, 3, device=self.device)
            self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            logger.debug("Bridge disabled: skipping acceleration computation tensors")

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        # Apply reset to recache new wyxz -> xyzw tensor
        self.robot_root_states.reset(self._robot.data.root_state_w)  # (num_envs, 13)

        self.base_quat = self.robot_root_states[:, 3:7]  # (num_envs, 4), xyzw
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # The body ordering of contact_sensor is different from the body ordering of the robot.
        self.contact_forces = self.contact_sensor.data.net_forces_w[
            :, self._contact_to_robot_body_ids
        ]  # (num_envs, num_bodies, 3)

        # Issue: data.net_forces_w_history is not cleared after a reset.
        # Solution: We only read the most recent decimation_factor steps.
        control_decimation = self.simulator_config.sim.control_decimation
        effective_history_length = min(control_decimation, self.simulator_config.contact_sensor_history_length)
        self.contact_forces_history[:, :effective_history_length, :, :] = self.contact_sensor.data.net_forces_w_history[
            :, :effective_history_length, self._contact_to_robot_body_ids
        ]  # (num_envs, history_length, num_bodies, 3), the first index is the most recent

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][
            :, :, [1, 2, 3, 0]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]

    def clear_contact_forces_history(self, env_id):
        if len(env_id) > 0:
            self.contact_forces_history[env_id, :, :, :] = 0.0
            env_reset_ids = env_id.detach().cpu().tolist() if isinstance(env_id, torch.Tensor) else env_id
            for sensor in self._object_contact_sensors.values():
                sensor.reset(env_reset_ids)

    def _resolve_required_object_contact_sensor_body_names(self, config_root: Any) -> tuple[str, ...]:
        available_body_names = list(getattr(self.robot_config, "body_names", []))
        monitorable_body_names = set(_OBJECT_CONTACT_MONITOR_BODY_NAMES).intersection(available_body_names)
        if not monitorable_body_names:
            return ()

        configured_body_names = list(
            getattr(self.simulator_config, "object_filtered_contact_sensor_body_names", []) or []
        )
        if configured_body_names:
            return tuple(
                body_name
                for body_name in _OBJECT_CONTACT_MONITOR_BODY_NAMES
                if body_name in monitorable_body_names and body_name in configured_body_names
            )

        command_setup_terms = getattr(self.command_config, "setup_terms", {}) if self.command_config is not None else {}
        if "motion_command" in command_setup_terms:
            return ()

        required_body_names: set[str] = set()
        for node in _iter_config_nodes(config_root):
            func_path: str | None = None
            params: dict[str, Any] | DictConfig | None = None
            reward_weight: Any = None

            if is_dataclass(node):
                func_path = getattr(node, "func", None)
                params = getattr(node, "params", None)
                reward_weight = getattr(node, "weight", None)
            elif isinstance(node, (dict, DictConfig)):
                func_path = node.get("func")
                params = node.get("params")
                reward_weight = node.get("weight")

            if not isinstance(func_path, str) or not isinstance(params, (dict, DictConfig)):
                continue

            needs_filtered_sensor = False
            if func_path in _OBJECT_CONTACT_OBSERVATION_FUNC_PATHS:
                needs_filtered_sensor = bool(params.get("object_only") or params.get("non_object_only"))
            elif func_path in _OBJECT_CONTACT_REWARD_FUNC_PATHS:
                needs_filtered_sensor = reward_weight is None or float(reward_weight) != 0.0

            if not needs_filtered_sensor:
                continue

            required_body_names.update(
                monitorable_body_names.intersection(
                    self._resolve_contact_sensor_body_names_from_params(params, available_body_names)
                )
            )

        return tuple(body_name for body_name in _OBJECT_CONTACT_MONITOR_BODY_NAMES if body_name in required_body_names)

    def _resolve_contact_sensor_body_names_from_params(
        self,
        params: dict[str, Any] | DictConfig,
        available_body_names: list[str],
    ) -> list[str]:
        body_names = params.get("body_names")
        if body_names is not None:
            return [body_name for body_name in body_names if body_name in available_body_names]

        body_name_pattern = params.get("body_name_pattern")
        if body_name_pattern is None:
            body_name_pattern = params.get("undesired_contacts_body_names")
        if body_name_pattern:
            regex = re.compile(body_name_pattern)
            return [body_name for body_name in available_body_names if regex.match(body_name)]

        return list(available_body_names)

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)

    def draw_debug_viz(self):
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()
        self._draw_contact_forces()

    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        # Only render if actively recording (not just if video recorder exists)
        has_video_recording = self.video_recorder is not None and self.video_recorder.is_recording
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors() or has_video_recording

        # Apply virtual gantry forces before physics step
        if self.virtual_gantry:
            self.virtual_gantry.step()

        # Step bridge for updated torques before physics step using base class helper
        self._step_bridge()

        self.scene.write_data_to_sim()

        # simulate
        self.sim.step(render=False)

        # Render between steps only IF the GUI or sensor need it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.render()

        # update buffers at sim
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

        # Need to update these tensors after each step, since they are used in `_apply_force_in_physics_step`
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # Update accelerations ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            # Update DOF acceleration using numerical differentiation
            self.dof_acc = (self.dof_vel - self.prev_dof_vel) / self.sim_dt
            self.prev_dof_vel = self.dof_vel.clone()

            # Update base linear acceleration using numerical differentiation
            current_base_vel = self.robot_root_states[:, 7:10]
            self.base_linear_acc = (current_base_vel - self.prev_base_lin_vel) / self.sim_dt
            self.prev_base_lin_vel = current_base_vel.clone()

        # Call video recorder capture frame if recording is active
        if self.video_recorder:
            self.capture_video_frame()

    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller

        # Initialize commands tensor if not already done
        if not hasattr(self, "commands"):
            self.commands = torch.zeros((self.training_config.num_envs, 12), device=self.sim_device)

        # Set up keyboard handling
        if self.viewport_camera_controller is not None:
            self._setup_keyboard_controls()

    def _setup_keyboard_controls(self):
        """Set up keyboard controls for the simulator."""
        try:
            # Import necessary modules
            import carb.input
            import omni.appwindow

            # Get the input interface
            self.input_interface = carb.input.acquire_input_interface()
            self.appwindow = omni.appwindow.get_default_app_window()
            self.keyboard = self.appwindow.get_keyboard()

            # Define key mappings
            self.key_commands = {
                "W": "forward_command",
                "S": "backward_command",
                "A": "left_command",
                "D": "right_command",
                "Q": "heading_left_command",
                "E": "heading_right_command",
                "Z": "zero_command",
                "X": "walk_stand_toggle",
                "U": "height_up",
                "L": "height_down",
                "I": "waist_yaw_up",
                "K": "waist_yaw_down",
                "P": "push_robots",
                # Virtual gantry controls (using enum)
                "KEY_7": GantryCommand.LENGTH_ADJUST,  # decrease
                "KEY_8": GantryCommand.LENGTH_ADJUST,  # increase
                "KEY_9": GantryCommand.TOGGLE,
                "KEY_0": GantryCommand.FORCE_ADJUST,
                "MINUS": GantryCommand.FORCE_SIGN_TOGGLE,
            }

            # Initialize push_requested flag
            self.push_requested = False

            # Register keyboard callback
            def keyboard_callback(event, *args, **kwargs):
                # Only process key press events
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input.name in self.key_commands:
                        command = self.key_commands[event.input.name]
                        if command == "forward_command":
                            self.commands[:, 0] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "backward_command":
                            self.commands[:, 0] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "left_command":
                            self.commands[:, 1] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "right_command":
                            self.commands[:, 1] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "heading_left_command":
                            self.commands[:, 3] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "heading_right_command":
                            self.commands[:, 3] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "zero_command":
                            self.commands[:, :4] = 0
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "walk_stand_toggle":
                            self.commands[:, 4] = 1 - self.commands[:, 4]
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "height_up":
                            self.commands[:, 8] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "height_down":
                            self.commands[:, 8] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "waist_yaw_up":
                            self.commands[:, 5] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "waist_yaw_down":
                            self.commands[:, 5] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "push_robots":
                            logger.info("Push Robots Requested")
                            self.push_requested = True
                        # Virtual gantry commands (using enum)
                        elif command == GantryCommand.LENGTH_ADJUST:
                            if self.virtual_gantry:
                                # Differentiate between KEY_7 (decrease) and KEY_8 (increase)
                                amount = -0.1 if event.input.name == "KEY_7" else 0.1
                                command_data = GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": amount})
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.TOGGLE:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.TOGGLE)
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.FORCE_ADJUST:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.FORCE_ADJUST)
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.FORCE_SIGN_TOGGLE:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE)
                                self.virtual_gantry.handle_command(command_data)
                        return True
                return False

            self.keyboard_sub = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard,
                lambda event, *args: keyboard_callback(event, *args),
            )
            logger.info("Keyboard controls initialized")

        except Exception as e:
            logger.warning(f"Could not initialize keyboard controls: {e}")

    def render(self, sync_frame_time=True):
        self.sim.render()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    # debug visualization - delegate to draw adapter
    def clear_lines(self):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import clear_lines

        clear_lines(self)

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_sphere

        draw_sphere(self, pos, radius, color, env_id, pos_id)

    def draw_line(self, start_point, end_point, color, env_id):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_line

        draw_line(self, start_point, end_point, color, env_id)

    def set_actor_root_state_tensor_robots(self, env_ids=None, root_states=None):
        """See base class.

        IsaacSim-specific notes:
        - Quaternions converted from (x,y,z,w) to (w,x,y,z) format for IsaacSim compatibility
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if root_states is None:
            robot_root_states_wxyz = self.robot_root_states._get_wxyz(env_ids)
        elif isinstance(root_states, AllRootStatesProxy):
            robot_root_states_wxyz = self.robot_root_states._get_wxyz(env_ids)
        elif isinstance(root_states, RootStatesProxy):
            # assumes the user passed in robot_root_states directly
            robot_root_states_wxyz = root_states._get_wxyz(env_ids)
        elif isinstance(root_states, torch.Tensor):
            root_states = root_states.to(device=self.sim_device, dtype=torch.float32)
            if root_states.ndim != 2 or root_states.shape[1] != 13:
                raise ValueError(
                    f"Expected root_states shape [N, 13], got {tuple(root_states.shape)}"
                )
            if root_states.shape[0] == getattr(self, "num_envs", self.training_config.num_envs):
                root_states = root_states[env_ids]
            elif root_states.shape[0] != len(env_ids):
                raise ValueError(
                    f"Expected root_states batch size to match env_ids ({len(env_ids)}) "
                    f"or num_envs ({getattr(self, 'num_envs', self.training_config.num_envs)}), "
                    f"got {root_states.shape[0]}"
                )
            robot_root_states_wxyz = fullstate_xyzw_to_wxyz(root_states)
        else:
            raise ValueError(f"Unexpected root states type: {type(root_states)}")

        self._robot.write_root_pose_to_sim(robot_root_states_wxyz[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(robot_root_states_wxyz[:, 7:], env_ids)

    def set_dof_state_tensor_robots(self, env_ids=None, dof_states=None):
        """See base class.

        IsaacSim-specific notes:
        - Tensor format: 3D [num_envs, num_dofs, 2] (differs from IsaacGym's flattened format)

        Examples
        --------
        >>> # IsaacSim format: 3D [num_envs, num_dofs, 2]
        >>> env_ids = torch.tensor([0, 1], device=device)
        >>> dof_states = torch.zeros(len(env_ids), sim.num_dof, 2, device=device)
        >>> dof_states[:, :, 0] = default_joint_positions  # 2D positions [envs, dofs]
        >>> dof_states[:, :, 1] = 0.0  # Zero velocities
        >>> sim.set_dof_state_tensor_robots(env_ids, dof_states)
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state

        dof_pos, dof_vel = dof_states[env_ids, :, 0], dof_states[env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, env_ids)

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """See base class."""
        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_states_by_index(self, indices: ActorIndices) -> ActorStates:
        """Get actor states by pre-computed indices.

        IsaacSim stores robot/object root states in a unified proxy (`all_root_states`)
        using the same flattened indexing scheme as ObjectRegistry.
        """
        return self.all_root_states[indices, :13]

    def set_actor_states_by_index(
        self, indices: ActorIndices, states: ActorStates, write_updates: bool = True
    ) -> None:
        """Set actor states by pre-computed indices."""
        self.all_root_states[indices, :13] = states
        if write_updates:
            self.write_state_updates()

    def set_actor_states(self, names: ActorNames, env_ids: EnvIds, states: ActorStates):
        """See base class.

        IsaacSim-specific notes:
        - Uses AllRootStatesProxy for unified tensor access
        - Automatically calls write_state_updates() for immediate sync
        """
        actor_indices = self.get_actor_indices(names, env_ids)
        self.all_root_states[actor_indices, :13] = states
        self.write_state_updates()

    def _select_env_rows(self, rows: torch.Tensor, env_ids: torch.Tensor, *, label: str) -> torch.Tensor:
        """Select per-env rows safely from IsaacLab asset tensors.

        Some asset tensors are expected to be shaped `[num_envs, ...]`, but in failure modes we want a
        synchronous, explicit error instead of a CUDA device-side assert from an out-of-bounds advanced index.
        """
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)

        if rows.shape[0] == 1 and env_ids.numel() > 1:
            return rows.expand(env_ids.numel(), *rows.shape[1:]).clone()

        if env_ids.numel() == 0:
            return rows[:0]

        env_ids = env_ids.to(device=rows.device, dtype=torch.long)
        max_env_id = int(env_ids.max().item())
        if max_env_id >= rows.shape[0]:
            raise RuntimeError(
                f"{label} has only {rows.shape[0]} row(s), but env_ids request index {max_env_id}. "
                f"This indicates an env/object instance-count mismatch."
            )

        return rows.index_select(0, env_ids)

    def get_actor_initial_poses(self, names: ActorNames, env_ids: EnvIds | None = None) -> ActorPoses:
        """See base class."""
        if not names:
            return torch.empty(0, 7, device=self.sim_device, dtype=torch.float32)

        # Determine which environments to use
        if env_ids is None:
            num_envs = getattr(self, "num_envs", self.scene.num_envs)
            env_ids = torch.arange(num_envs, device=self.sim_device)

        pose_batches: list[torch.Tensor] = []
        for obj_name in names:
            if obj_name == "robot":
                # Get robot base pose from configuration
                pos = torch.tensor(self.robot_config.init_state.pos, device=self.sim_device, dtype=torch.float32)
                rot = torch.tensor(self.robot_config.init_state.rot, device=self.sim_device, dtype=torch.float32)
                pose = torch.cat([pos, rot])  # [7] - [x,y,z,qx,qy,qz,qw]
                pose_batches.append(pose.unsqueeze(0).expand(len(env_ids), -1).clone())

            elif self._is_scene_object(obj_name):
                # Get scene object pose from scene collection
                scene_collection = self.scene.rigid_objects["usd_scene_objects"]
                object_index = self._get_object_index_in_collection(obj_name, scene_collection)
                world_state = self._select_env_rows(
                    scene_collection.data.object_state_w[:, object_index],
                    env_ids,
                    label=f"scene object '{obj_name}' world state",
                )
                pose_batches.append(world_state[:, [0, 1, 2, 4, 5, 6, 3]])

            elif obj_name in self.scene.rigid_objects:
                # Get individual object pose from rigid object.
                # Use the live world-frame state after simulation reset instead of default_root_state so the
                # registry reflects the actual instantiated per-env object poses in heterogeneous scenes.
                rigid_object = self.scene.rigid_objects[obj_name]
                world_state = self._select_env_rows(
                    rigid_object.data.root_link_state_w,
                    env_ids,
                    label=f"individual object '{obj_name}' root state",
                )
                pose_batches.append(world_state[:, [0, 1, 2, 4, 5, 6, 3]])

            else:
                available_objects = ["robot"] + list(self.scene.rigid_objects.keys())
                raise KeyError(f"Object '{obj_name}' not found. Available: {available_objects}")

        if not pose_batches:
            return torch.empty(0, 7, device=self.sim_device, dtype=torch.float32)
        # Object-major flattening: [obj0_env0, obj0_env1, obj1_env0, obj1_env1, ...]
        return torch.cat(pose_batches, dim=0)

    def _is_scene_object(self, object_name: str) -> bool:
        """Check if an object is part of the USD scene collection - IsaacSim implementation.

        Uses IsaacLab's native RigidObjectCollection methods to determine if an object
        belongs to a scene collection rather than being an individual rigid object.

        Parameters
        ----------
        object_name : str
            Name of the object to check, e.g., "obj0_0"

        Returns
        -------
        bool
            True if object is in a scene collection, False otherwise

        Notes
        -----
        - Currently assumes single scene collection named 'usd_scene_objects'
        - Uses full path name with "/world/" prefix for IsaacLab compatibility
        - Scene collections are loaded from USD/URDF scene files
        """
        collection_name = "usd_scene_objects"  # TODO fix assumption for one scene collection
        scene_collection = self.scene.rigid_objects.get(collection_name, None)
        full_path_name = f"/world/{object_name}"
        return scene_collection and full_path_name in scene_collection.object_names

    def _get_object_index_in_collection(self, object_name: str, scene_collection) -> int:
        """Get object index within scene collection - IsaacSim implementation.

        Uses IsaacLab's native RigidObjectCollection.find_objects() method to locate
        an object within a scene collection and return its internal index.

        Parameters
        ----------
        object_name : str
            Name of the object, e.g., "obj0_0"
        scene_collection : RigidObjectCollection
            The USD scene collection to search within

        Returns
        -------
        int
            Index of the object within the collection

        Raises
        ------
        KeyError
            If object not found in collection

        Notes
        -----
        - Uses full path name with "/world/" prefix for IsaacLab compatibility
        - Returns the first match if multiple objects found
        - Index is used for tensor access within the collection
        """
        # TODO: Fix remove /world prefix due to USD loader coupling
        full_path_name = f"/world/{object_name}"
        obj_indices, obj_names = scene_collection.find_objects(full_path_name)

        if len(obj_indices) == 0:
            available_names = scene_collection.object_names
            raise KeyError(f"Object '{object_name}' not found in collection. Available: {available_names}")

        return obj_indices[0].item()

    def _get_scene_default_object_state(self, scene_collection, object_name: str) -> torch.Tensor:
        """Get initial object state from scene collection - IsaacSim implementation.

        Retrieves the default/initial state for an object within a scene collection
        using IsaacLab's tensor data after simulation initialization.

        NOTE: Returns quat in IsaacSim wxyz format, not holosoma xyzw format (internal function)

        Parameters
        ----------
        scene_collection : RigidObjectCollection
            The scene collection containing the object
        object_name : str
            Name of the object to get state for

        Returns
        -------
        torch.Tensor
            Default object state [13] containing position, quaternion, and velocities

        Notes
        -----
        - Must be called after sim.play() when tensor data is available
        - Returns full 13-element state vector from IsaacLab's default_object_state
        - Used internally for initial pose extraction and reset operations
        """
        object_index = self._get_object_index_in_collection(object_name, scene_collection)
        return scene_collection.data.default_object_state[0, object_index]  # [13]

    def _get_object_states(self, object_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states for any object type - delegates to state adapter.

        Parameters
        ----------
        object_name : str
            Name of the object to query
        env_ids : torch.Tensor
            Environment IDs to query, shape [num_envs], dtype torch.long

        Returns
        -------
        torch.Tensor
            Object states [len(env_ids), 13] containing position, quaternion, and velocities
            in xyzw format (converted by state adapter)
        """
        return self._state_adapter.get_object_states(object_name, env_ids)

    def _write_object_state_unified(self, object_name: str, states: torch.Tensor, env_ids: torch.Tensor):
        """Write object states for any object type - delegates to state adapter."""
        self._state_adapter.write_object_states(object_name, states, env_ids)

    def time(self) -> float:
        """Get current simulation time.

        Returns:
            float: Current simulation time in seconds
        """
        return self.sim.current_time

    def _get_split_sim_state_extra_payload(self) -> dict[str, object]:
        """Publish IsaacSim-measured reference-body pose for split sim2sim alignment."""
        if not hasattr(self, "_rigid_body_pos") or not hasattr(self, "_rigid_body_rot"):
            return {}
        if len(getattr(self, "body_names", [])) == 0:
            return {}

        ref_body_name = getattr(self.robot_config, "torso_name", None) or self.body_names[0]
        try:
            ref_body_idx = self.body_names.index(ref_body_name)
        except ValueError:
            ref_body_idx = 0
            ref_body_name = self.body_names[0]

        ref_pos = self._rigid_body_pos[0, ref_body_idx].detach().cpu().tolist()
        ref_quat = self._rigid_body_rot[0, ref_body_idx].detach().cpu().tolist()
        ref_lin_vel = self._rigid_body_vel[0, ref_body_idx].detach().cpu().tolist()
        ref_ang_vel = self._rigid_body_ang_vel[0, ref_body_idx].detach().cpu().tolist()

        payload: dict[str, object] = {
            "robot_ref_body_name": str(ref_body_name),
            "robot_ref_state": [
                float(ref_pos[0]),
                float(ref_pos[1]),
                float(ref_pos[2]),
                float(ref_quat[0]),
                float(ref_quat[1]),
                float(ref_quat[2]),
                float(ref_quat[3]),
                float(ref_lin_vel[0]),
                float(ref_lin_vel[1]),
                float(ref_lin_vel[2]),
                float(ref_ang_vel[0]),
                float(ref_ang_vel[1]),
                float(ref_ang_vel[2]),
            ],
        }

        include_key_body_states = os.getenv("HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES", "0") == "1"
        if include_key_body_states:
            requested_names = [
                name.strip()
                for name in os.getenv("HOLOSOMA_SIM_STATE_KEY_BODY_NAMES", "").split(",")
                if name.strip()
            ]
            key_body_names = requested_names or [str(ref_body_name)]
            key_body_states: dict[str, list[float]] = {}
            for body_name in key_body_names:
                try:
                    body_idx = self.body_names.index(body_name)
                except ValueError:
                    continue
                pos = self._rigid_body_pos[0, body_idx].detach().cpu().tolist()
                quat = self._rigid_body_rot[0, body_idx].detach().cpu().tolist()
                key_body_states[str(body_name)] = [
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ]
            if key_body_states:
                payload["key_body_states"] = key_body_states

        return payload

    def get_dof_forces(self, env_id: int = 0):
        """Get DOF forces for a specific environment.

        This method provides access to measured joint forces. For IsaacSim,
        joint forces are computed from applied torques since direct force
        sensing is not available in the same way as IsaacGym.

        Args:
            env_id: Environment index (default: 0)

        Returns:
            torch.Tensor: Tensor of shape [num_dof] with computed joint forces

        Note:
            IsaacSim doesn't have the same DOF force sensor infrastructure as IsaacGym.
            This implementation returns the applied torques as an approximation.
            For actual force sensing, consider using contact sensors or force/torque sensors.
        """
        # IsaacSim doesn't have direct DOF force sensors like IsaacGym
        # Return the applied torques (which are the commanded forces)
        # This matches the bridge's usage pattern where forces are used for feedback
        if not hasattr(self._robot, "data") or not hasattr(self._robot.data, "applied_torque"):
            logger.warning(
                "DOF forces not directly available in IsaacSim. "
                "Returning zeros. For force feedback, the bridge will use commanded torques."
            )
            return torch.zeros(self.num_dof, device=self.device)

        # Get applied torques which represent the forces being applied to joints
        applied_torques = self._robot.data.applied_torque[env_id, self.dof_ids]
        return applied_torques

    def write_state_updates(self):
        """See base class.

        IsaacSim-specific notes:
        - Root-state writes already call IsaacLab/PhysX setters immediately
        - This method exists for API compatibility with deferred-update simulators
        """
        debug_state_sync = os.environ.get("HOLOSOMA_DEBUG_STATE_SYNC", "").lower() not in ("", "0", "false", "no")
        if not self._state_adapter.is_dirty():
            if debug_state_sync:
                logger.info("State sync skipped: no object state changes to sync")
            return

        if debug_state_sync:
            logger.info("State sync begin (compatibility no-op)")

        # Root/object state setters already write through to PhysX immediately via IsaacLab views.
        # Keep only the dirty-flag clear here to avoid redundant whole-scene write passes.
        self._state_adapter.clear_dirty()

        if debug_state_sync:
            logger.info("State sync end")
