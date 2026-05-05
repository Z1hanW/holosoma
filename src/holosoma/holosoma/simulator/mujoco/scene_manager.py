"""MuJoCo scene manager."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, List

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger

from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import MujocoXMLFilterCfg, SimulatorConfig
from holosoma.managers.camera import CameraManager
from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path

_GT_MUJOCO_PHYSICS_ENV = "GT_MUJOCO_PHYSICS"
_HOLOSOMA_GT_MUJOCO_PHYSICS_ENV = "HOLOSOMA_GT_MUJOCO_PHYSICS"
_MUJOCO_OBJECT_MASS_OVERRIDE_ENV = "MUJOCO_OBJECT_MASS_OVERRIDE"
_MUJOCO_OBJECT_GEOM_FRICTION_ENV = "MUJOCO_OBJECT_GEOM_FRICTION"
_MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION_ENV = "MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION"

_GT_TERRAIN_FRICTION = [1.0, 0.005, 0.001]
_GT_TERRAIN_SOLREF = [0.01, 1.0]
_GT_OBJECT_GEOM_FRICTION = [0.6, 0.02, 0.005]
_GT_OBJECT_MASS = 1.4
_GT_HAND_OBJECT_FRICTION = [0.8, 0.02, 0.005]
_GT_HAND_OBJECT_SOLREF = [0.01, 1.0]


def _euler_xyz_to_quat_wxyz(euler_rad: np.ndarray) -> np.ndarray:
    """Convert intrinsic XYZ Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).

    Uses the same convention as quat_from_euler_xyz in the warp camera sensor code,
    but returns MuJoCo's (w, x, y, z) ordering.
    """
    roll, pitch, yaw = euler_rad
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return np.array([w, x, y, z])


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class MujocoSceneManager:
    """Compositional world builder using MjSpec for MuJoCo simulations.

    This class provides a compositional approach to building MuJoCo simulation worlds
    by combining terrain, lighting, materials, and robots using the MjSpec API.
    It handles terrain generation, collision configuration, and robot integration
    while maintaining proper scene composition order.

    The scene manager supports multiple terrain types (plane, heightfield, trimesh)
    and provides automatic collision configuration based on robot self-collision settings.
    """

    def __init__(self, simulator_config: SimulatorConfig) -> None:
        """Initialize the scene manager with simulator configuration.

        Parameters
        ----------
        simulator_config : SimulatorConfig
            Simulator configuration containing physics and rendering parameters.
        """
        self.world_spec = mujoco.MjSpec()
        self.world_spec.copy_during_attach = True
        self._setup_world_options(simulator_config)
        self.robot_config: RobotConfig | None = None  # Set when adding robot
        self._terrain_geom_name = "floor"

    def _setup_world_options(self, simulator_config: SimulatorConfig) -> None:
        """Configure world specification options from simulator config.

        Parameters
        ----------
        simulator_config : SimulatorConfig
            Simulator configuration containing physics parameters.
        """
        # TODO: expose to Mujoco-specific config
        self.world_spec.option.gravity = [0, 0, -9.81]
        self.world_spec.option.timestep = 1.0 / simulator_config.sim.fps  # type: ignore[attr-defined]

    def add_materials(self) -> None:
        """Add standard materials and textures to the world specification.

        Creates a chequered texture and grid material that can be applied
        to terrain and other geometric elements for visual enhancement.
        """

        self.world_spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            width=512,
            height=3072,
            rgb1=[0.3, 0.5, 0.7],  # Light blue
            rgb2=[0.0, 0.0, 0.0],  # Black
        )

        # Add chequered texture
        self.world_spec.add_texture(
            name="chequered",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
        )

        grid_material = self.world_spec.add_material(name="grid", texrepeat=[5, 5], reflectance=0.2)
        grid_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "chequered"

        # Add a solid gray material with moderate specular response for meshes without textures
        self.world_spec.add_material(
            name="solid_gray",
            rgba=[0.3, 0.3, 0.3, 1.0],
            specular=0.2,
            reflectance=0.2,
            shininess=0.2,
            metallic=0.1,
            emission=1.0,
        )

    def add_lighting(self, lighting_config: Any | None = None) -> None:
        """Add lighting configuration to the world specification.

        Parameters
        ----------
        lighting_config : Any | None
            Lighting configuration parameters (currently unused, uses defaults).
        """
        # Arbitrary headlight ambient lighting
        self.world_spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        self.world_spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
        self.world_spec.visual.headlight.specular = [0.0, 0.0, 0.0]

        # Add global lighting orientation
        self.world_spec.visual.global_.azimuth = -130
        self.world_spec.visual.global_.elevation = -20
        self.world_spec.visual.global_.offwidth = 1920
        self.world_spec.visual.global_.offheight = 1080

        # Match our existing scene files
        self.world_spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]

        # Uncomment to increase to reduce shadow pixelation for larger terrain.
        # Slows down rendering dramatically...
        # self.world_spec.visual.quality.shadowsize = 1024

        # Arbitrary lights (offset XY to avoid gantry shadows)
        self.world_spec.worldbody.add_light(
            pos=[2, 0, 5.0],
            dir=[0, 0, -1],
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
            # castshadow=True,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

        # Second light for extra shadows, commented out a little experience performance.
        # self.world_spec.worldbody.add_light(
        #    pos=[-2, 0, 4.0], dir=[0, 0, -1],
        #    diffuse=[0.6, 0.6, 0.6],
        #    specular=[0.2, 0.2, 0.2],
        #    castshadow=True,
        #    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        # )

    def add_camera(self, camera_manager: CameraManager, num_envs: int) -> None:
        """Add cameras to the world specification from camera manager config.

        Reads camera terms from the CameraManager config and creates MuJoCo camera
        elements attached to the appropriate robot body links. Converts camera pose
        from the warp/IsaacGym convention (camera views along +Z) to MuJoCo convention
        (camera views along -Z) using offset_rot_base and a frame flip.

        Parameters
        ----------
        camera_manager : CameraManager
            Camera manager for the simulation.
        num_envs : int
            Number of environments (affects camera layout planning).
        """
        if not camera_manager.cfg.terms:
            return

        prefix = getattr(self, "robot_prefix", "robot_")

        for term_name, term_cfg in camera_manager.cfg.terms.items():
            params = term_cfg.params
            pose = params.get("pose")
            props = params.get("props")

            if pose is None:
                continue

            # Find parent body in world spec (robot bodies are prefixed after attach)
            body_name = f"{prefix}{pose.camera_body_link}"
            body = self.world_spec.body(body_name)
            if body is None:
                logger.warning(f"Camera '{term_name}': parent body '{body_name}' not found, skipping")
                continue

            # Camera name follows the convention used by sim_utils / image_server
            camera_name = f"{prefix}cam_{term_name}"

            # Remove existing camera with same name from XML (avoid duplicates)
            try:
                existing = self.world_spec.camera(camera_name)
                if existing is not None:
                    self.world_spec.delete(existing)
                    logger.info(f"Replaced existing XML camera '{camera_name}'")
            except (AttributeError, ValueError):
                pass  # find_camera may not exist in older MuJoCo versions

            # Convert orientation from warp/IsaacGym convention to MuJoCo.
            #
            # In warp:   camera views along +Z in its data frame.
            #            offset_rot_base = [-90, 0, -90] (roll, pitch, yaw deg) converts
            #            from the data frame to the physical sensor frame.
            #            Final local orientation = q_user * q_base
            #
            # In MuJoCo: camera views along -Z in its local frame.
            #            We apply Rx(180°) to flip +Z to -Z:
            #            q_mujoco = q_user * q_base * q_flip
            user_rot_rad = np.deg2rad(list(pose.camera_rotation))  # (roll, pitch, yaw)
            base_rot_rad = np.deg2rad([-90.0, 0.0, -90.0])  # offset_rot_base

            q_user = _euler_xyz_to_quat_wxyz(user_rot_rad)
            q_base = _euler_xyz_to_quat_wxyz(base_rot_rad)
            q_flip = np.array([0.0, 1.0, 0.0, 0.0])  # Rx(180°) in (w,x,y,z)

            q_mj = _quat_mul_wxyz(_quat_mul_wxyz(q_user, q_base), q_flip)
            q_mj = q_mj / np.linalg.norm(q_mj)

            # Vertical FOV from camera properties
            fovy = props.vertical_fov if props is not None else 45.0

            body.add_camera(
                name=camera_name,
                pos=list(pose.camera_offset),
                quat=q_mj.tolist(),
                fovy=fovy,
            )

            logger.info(
                f"Added camera '{camera_name}' to '{body_name}': "
                f"pos={list(pose.camera_offset)}, fovy={fovy}"
            )

    def add_terrain(self, terrain_state: TerrainTermBase, num_envs: int) -> None:
        """Add terrain to the world specification with extensible dispatch.

        Creates terrain using the TerrainTermBase class and converts it to the
        appropriate MuJoCo representation (plane, heightfield, or trimesh).
        Automatically configures collision properties for robot interaction.

        Parameters
        ----------
        cfg : TerrainConfig
            Terrain configuration specifying mesh type, dimensions, and properties.
        num_envs : int
            Number of environments (affects terrain layout planning).
        """

        geom: mujoco.MjSpec.Geom | None = None
        if terrain_state.mesh_type == "plane":
            geom = self._create_ground_plane(terrain_state)
        elif terrain_state.mesh_type in ["trimesh"]:
            # Use heightfield to reduce penetrations (vs. trimesh/geom mesh)
            geom = self._create_hfield(terrain_state)
        elif terrain_state.mesh_type in ["load_obj"]:
            geom = self._create_trimesh(terrain_state)
        elif terrain_state.mesh_type is None:
            logger.info("Terrain is none")
        else:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        if geom is not None:
            # Monkey-patch Mujoco geom into our terrain manager for convenience
            terrain_state.geom = geom  # type: ignore[attr-defined]
            self._terrain_geom_name = str(geom.name or terrain_state.name or "floor")

            # Set environment collision properties so robot self_collision flag works
            # Environment collision class
            terrain_state.geom.contype = 2  # type: ignore[attr-defined]
            # Only collide with robot (class 1)
            terrain_state.geom.conaffinity = 1  # type: ignore[attr-defined]

    def _create_ground_plane(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create a ground plane terrain geometry.

        Returns
        -------
        mujoco.MjSpec.Geom
            Ground plane geometry with configured physics properties.
        """
        friction = [0.7, 0.005, 0.001]
        solref = [0.001, 1.0]
        if self._gt_mujoco_physics_enabled():
            friction = list(_GT_TERRAIN_FRICTION)
            solref = list(_GT_TERRAIN_SOLREF)

        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            # Size=0 is rendered infinitely. Collision plane is always infinite.
            # Note: size.z is actually the rendered spacing betweeh the grid
            #       subdivisions (to improve lighting, shadows).
            size=[0, 0, 0.05],
            pos=[0, 0, 0],
            material="grid",
            friction=friction,  # [sliding, torsional, rolling]
            solimp=[0.99, 0.99, 0.01, 0.5, 2],  # 5 elements: [dmin, dmax, width, midpoint, power]
            solref=solref,  # 2 elements: [timeconst, dampratio]
        )

    def _create_trimesh(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create MuJoCo mesh terrain matching shared Terrain class behavior.

        Splits the mesh into connected components so that each component gets its
        own convex hull for collision. This avoids the issue where MuJoCo computes
        a single convex hull over all vertices, filling in concave regions like steps.
        """

        if terrain_state.mesh is None:
            raise ValueError("Terrain mesh data is required when using trimesh terrain type.")

        components = terrain_state.mesh.split()
        if not components:
            raise ValueError("Terrain mesh is empty and cannot be used to create a mesh geom.")

        logger.info(f"Splitting terrain mesh into {len(components)} convex component(s)")

        first_geom = None
        for idx, component in enumerate(components):
            vertices = np.asarray(component.vertices, dtype=np.float32)
            faces = np.asarray(component.faces, dtype=np.int32)

            if vertices.size == 0 or faces.size == 0:
                continue

            mesh_name = f"terrain_{idx}"
            geom_name = terrain_state.name if idx == 0 else f"{terrain_state.name}_{idx}"

            mesh_spec = self.world_spec.add_mesh(name=mesh_name)
            mesh_spec.uservert = vertices.flatten(order="C")
            mesh_spec.userface = faces.flatten(order="C")
            mesh_spec.smoothnormal = False

            geom = self.world_spec.worldbody.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_spec.name,
                pos=[0.0, 0.0, 0.0],
                material="solid_gray",
                friction=[
                    0.7,  # reasonable default
                    0.005,  # reasonable default
                    0.001,  # reasonable default
                ],  # [sliding, torsional, rolling]
                solimp=[0.99, 0.99, 0.01, 0.5, 2],
                solref=[0.001, 1],
            )

            # Set collision properties on each component so all interact with the robot
            geom.contype = 2
            geom.conaffinity = 1

            if first_geom is None:
                first_geom = geom

        return first_geom

    def _create_hfield(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create MuJoCo heightfield terrain from procedural terrain data.

        Converts the heightfield data from the terrain generator into a MuJoCo
        heightfield asset and geom. This avoids the convex hull simplification
        that occurs with trimesh terrain.

        Returns
        -------
        mujoco.MjSpec.Geom
            Heightfield geometry with configured physics properties.
        """
        terrain = terrain_state.terrain
        if not hasattr(terrain, "_height_field_raw"):
            raise ValueError("Terrain does not have heightfield data")

        # Get heightfield parameters from terrain
        height_data = np.asarray(terrain._height_field_raw, dtype=np.float32)
        vertical_scale = terrain._vertical_scale
        border_size = terrain._border_size
        total_length = terrain._total_length
        total_width = terrain._total_width

        # Apply vertical scaling to height data (convert from int16 indices to meters)
        height_data_scaled = height_data * vertical_scale

        # Handle negative heights: shift to make non-negative (MuJoCo requirement)
        min_height = height_data_scaled.min()
        z_offset = 0.0
        # min_height needs to be positive. zero is not allowed.
        if min_height < 1e-9:
            height_data_scaled = height_data_scaled - min_height + 1e-9
            z_offset = min_height
            logger.info(f"Shifted heightfield by {-min_height:.3f}m to ensure non-negative heights")

        max_height = height_data_scaled.max()
        min_height_final = height_data_scaled.min()

        # Calculate size parameters for MuJoCo hfield
        # size = [x_half, y_half, HEIGHT_RANGE, z_baseline]
        # Note: nrow/ncol are swapped for correct orientation
        height_range = max_height - min_height_final
        if height_range < 1e-9:
            height_range = 1e-9

        # Create heightfield asset
        hfield_spec = self.world_spec.add_hfield(name="terrain")
        hfield_spec.nrow = height_data.shape[1]  # swap: cols become rows
        hfield_spec.ncol = height_data.shape[0]  # swap: rows become cols
        hfield_spec.size = [0.5 * total_length, 0.5 * total_width, height_range, min_height_final]
        # MuJoCo expects raw elevation data in column-major (Fortran) order
        hfield_spec.userdata = height_data_scaled.flatten(order="F").tolist()

        logger.info(
            f"Created heightfield: {hfield_spec.nrow}x{hfield_spec.ncol},"
            f" size=[{0.5 * total_length:.2f}, {0.5 * total_width:.2f}, {height_range:.3f}, {min_height_final:.3f}]"
        )

        # Create heightfield geom, positioned to match terrain coordinate system
        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hfield_spec.name,
            pos=[
                0.5 * total_length - border_size,
                0.5 * total_width - border_size,
                z_offset if z_offset < 0 else 0.0,
            ],
            friction=[
                # Ignore terrain config until we expose Mujoco-specific parameters
                0.7,  # reasonable default
                0.005,  # reasonable default
                0.001,  # reasonable default
            ],  # [sliding, torsional, rolling]
            solimp=[0.99, 0.99, 0.01, 0.5, 2],
            solref=[0.001, 1],
        )

    def add_robot(
        self,
        terrain_state: TerrainTermBase,
        robot_config: RobotConfig,
        xml_filter: MujocoXMLFilterCfg | None = None,
        prefix: str = "robot_",
    ) -> None:
        """Add robot from XML file with namespace prefix and optional filtering.

        Loads a robot from its XML specification, applies optional filtering to
        remove scene elements (lights, ground), configures collision settings,
        and attaches it to the world with a namespace prefix.

        Parameters
        ----------
        robot_config : RobotConfig
            Robot configuration containing asset path and collision settings.
        xml_filter : MujocoXMLFilterCfg | None
            Optional XML filtering configuration to remove unwanted elements.
        prefix : str
            Namespace prefix for robot elements (default: "robot_").
        """
        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        robot_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)

        logger.info(f"Adding robot from: {robot_xml_path} with prefix: {prefix}")
        self.robot_model_path = robot_xml_path
        robot_spec = mujoco.MjSpec.from_file(robot_xml_path)

        if xml_filter and getattr(xml_filter, "enable", False):
            # Remove worldbody lights and ground|floor|plane geoms because they're added dynamically
            robot_spec = self._filter_robot_worldbody(robot_spec, xml_filter)

        if hasattr(terrain_state, "geom") and terrain_state.geom:
            # Apply collision settings based on unified self_collisions flag in config
            # Only modifies collision groups if we have programmatically added terrain, otherwise
            # assumes the robot XML knows what it's doing
            self._apply_collision_settings(robot_spec, robot_config)

        # Create a spawn site for robot. This is not the initial body state from config,
        # which is set later
        robot_pos = [0, 0, 0.0]
        robot_rot = [1, 0, 0, 0]
        site = self.world_spec.worldbody.add_site(pos=robot_pos, quat=robot_rot)
        self.world_spec.attach(robot_spec, site=site, prefix=prefix)
        self._add_object_from_robot_config(robot_config, terrain_geom_name=str(getattr(terrain_state, "name", "floor")))

        # Store prefix for later use by simulator
        self.robot_prefix = prefix

    @staticmethod
    def _configure_urdf_meshdir(spec: mujoco.MjSpec, urdf_path: Path) -> None:
        mesh_files = [Path(str(mesh.file)) for mesh in spec.meshes if str(getattr(mesh, "file", "")).strip()]
        meshdir_candidates = [urdf_path.parent, urdf_path.parent / "meshes"]
        for candidate in meshdir_candidates:
            if not candidate.is_dir():
                continue
            if mesh_files and not all(
                mesh_file.is_absolute() or (candidate / mesh_file).is_file() for mesh_file in mesh_files
            ):
                continue
            spec.compiler.meshdir = str(candidate.resolve())
            return
        spec.compiler.meshdir = str(urdf_path.parent.resolve())

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"", "0", "false", "no", "off"}

    @classmethod
    def _gt_mujoco_physics_enabled(cls) -> bool:
        return cls._env_flag(_GT_MUJOCO_PHYSICS_ENV) or cls._env_flag(_HOLOSOMA_GT_MUJOCO_PHYSICS_ENV)

    @staticmethod
    def _parse_float_env(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return float(default)
        return float(raw)

    @staticmethod
    def _parse_float_triplet_env(name: str, default: list[float]) -> list[float]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return [float(value) for value in default]

        try:
            parsed = ast.literal_eval(raw)
            values = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            values = raw.strip("[]()").replace(",", " ").split()

        triplet = [float(value) for value in values]
        if len(triplet) != 3:
            raise ValueError(f"{name} must provide exactly 3 values [slide, spin, roll], got {triplet}")
        return triplet

    @staticmethod
    def _expand_pair_friction(friction_triplet: list[float]) -> list[float]:
        slide, spin, roll = [float(value) for value in friction_triplet]
        return [slide, slide, spin, roll, roll]

    @staticmethod
    def _condim_from_friction_triplet(friction_triplet: list[float]) -> int:
        _, spin, roll = [abs(float(value)) for value in friction_triplet]
        if roll > 0.0:
            return 6
        if spin > 0.0:
            return 4
        return 3

    @staticmethod
    def _iter_bodies(container: Any):
        for body in getattr(container, "bodies", []):
            yield body
            yield from MujocoSceneManager._iter_bodies(body)

    @staticmethod
    def _iter_body_geoms(container: Any):
        for body in MujocoSceneManager._iter_bodies(container):
            for geom in getattr(body, "geoms", []):
                yield body, geom

    @staticmethod
    def _select_object_root_body(object_spec: mujoco.MjSpec) -> mujoco.MjSpec.Body:
        for preferred_name in ("baseLink", "base_link"):
            for body in object_spec.bodies:
                if body.name == preferred_name:
                    return body
        for body in object_spec.bodies:
            if body.name:
                return body
        raise ValueError("Could not resolve a named object root body from the MuJoCo object URDF")

    def _add_object_from_robot_config(self, robot_config: RobotConfig, *, terrain_geom_name: str) -> None:
        object_cfg = getattr(robot_config, "object", None)
        object_urdf_path = getattr(object_cfg, "object_urdf_path", None)
        if not object_urdf_path:
            return

        resolved_object_path = Path(resolve_data_file_path(str(object_urdf_path))).expanduser().resolve()
        if not resolved_object_path.is_file():
            raise FileNotFoundError(f"MuJoCo object URDF not found: {resolved_object_path}")

        object_spec = mujoco.MjSpec.from_file(str(resolved_object_path))
        self._configure_urdf_meshdir(object_spec, resolved_object_path)
        object_root_body = self._select_object_root_body(object_spec)
        object_root_body.pos = [0.75, 0.0, 0.18]
        self._apply_gt_object_properties(object_spec)

        object_site = self.world_spec.worldbody.add_site(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
        self.world_spec.attach(object_spec, site=object_site, prefix="object_")
        self._add_gt_object_contact_pairs(terrain_geom_name=terrain_geom_name)
        logger.info(f"Added MuJoCo object from: {resolved_object_path}")

    def _apply_gt_object_properties(self, object_spec: mujoco.MjSpec) -> None:
        if not self._gt_mujoco_physics_enabled():
            return

        geom_friction = self._parse_float_triplet_env(_MUJOCO_OBJECT_GEOM_FRICTION_ENV, _GT_OBJECT_GEOM_FRICTION)
        mass_override = self._parse_float_env(_MUJOCO_OBJECT_MASS_OVERRIDE_ENV, _GT_OBJECT_MASS)
        condim = self._condim_from_friction_triplet(geom_friction)
        existing_geom_names = {str(geom.name) for _, geom in self._iter_body_geoms(object_spec) if geom.name}

        def _unique_geom_name(base_name: str) -> str:
            candidate = base_name
            suffix = 2
            while candidate in existing_geom_names:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            existing_geom_names.add(candidate)
            return candidate

        updated_bodies = 0
        updated_geoms = 0
        for body in object_spec.bodies:
            if not body.name:
                continue

            original_mass = float(body.mass)
            if mass_override > 0.0 and original_mass > 0.0:
                mass_ratio = mass_override / original_mass
                body.mass = mass_override
                fullinertia = np.asarray(getattr(body, "fullinertia", []), dtype=np.float64)
                if fullinertia.size == 6 and np.all(np.isfinite(fullinertia)):
                    body.fullinertia = (fullinertia * mass_ratio).tolist()
                else:
                    inertia = np.asarray(body.inertia, dtype=np.float64)
                    if inertia.size == 3 and np.all(np.isfinite(inertia)):
                        body.inertia = (inertia * mass_ratio).tolist()
                updated_bodies += 1

            for geom in body.geoms:
                if int(geom.contype) == 0 or int(geom.conaffinity) == 0:
                    continue
                if not geom.name:
                    geom.name = _unique_geom_name("collision")
                geom.contype = 4
                geom.conaffinity = 11
                geom.condim = max(int(geom.condim), condim)
                geom.friction = list(geom_friction)
                geom.solref = list(_GT_HAND_OBJECT_SOLREF)
                updated_geoms += 1

        logger.info(
            "Applied GT MuJoCo object material: bodies={}, geoms={}, mass={}, geom_friction={}",
            updated_bodies,
            updated_geoms,
            mass_override,
            geom_friction,
        )

    def _add_gt_object_contact_pairs(self, *, terrain_geom_name: str) -> None:
        if not self._gt_mujoco_physics_enabled():
            return

        object_terrain_friction = self._parse_float_triplet_env(
            _MUJOCO_OBJECT_TERRAIN_PAIR_FRICTION_ENV,
            _GT_OBJECT_GEOM_FRICTION,
        )
        object_terrain_pair_friction = self._expand_pair_friction(object_terrain_friction)
        object_terrain_condim = self._condim_from_friction_triplet(object_terrain_friction)
        hand_pair_friction = self._expand_pair_friction(list(_GT_HAND_OBJECT_FRICTION))

        existing_pair_keys = {
            (str(pair.geomname1), str(pair.geomname2))
            for pair in self.world_spec.pairs
            if pair.geomname1 and pair.geomname2
        }
        existing_pair_names = {str(pair.name) for pair in self.world_spec.pairs if pair.name}

        object_collision_geoms: list[str] = []
        hand_collision_geoms: list[str] = []
        for body, geom in self._iter_body_geoms(self.world_spec.worldbody):
            if int(geom.contype) == 0 or int(geom.conaffinity) == 0:
                continue
            body_name = str(body.name or "")
            geom_name = str(geom.name or "")
            mesh_name = str(getattr(geom, "meshname", "") or "")
            combined_name = f"{body_name} {geom_name} {mesh_name}".lower()

            if body_name.startswith("object_") or geom_name.startswith("object_"):
                object_collision_geoms.append(geom_name)
            elif (
                "rubber_hand" in combined_name
                or "sphere_hand" in combined_name
                or "left_hand_collision" in combined_name
                or "right_hand_collision" in combined_name
            ):
                geom.contype = 1
                geom.conaffinity = 6
                geom.condim = 6
                geom.friction = list(_GT_HAND_OBJECT_FRICTION)
                geom.solref = list(_GT_HAND_OBJECT_SOLREF)
                hand_collision_geoms.append(geom_name)

        added_pairs = 0
        for object_geom_name in sorted(set(object_collision_geoms)):
            pair_name = f"{object_geom_name}__{terrain_geom_name}__sim2sim"
            pair_key = (object_geom_name, terrain_geom_name)
            reverse_pair_key = (terrain_geom_name, object_geom_name)
            if (
                pair_name not in existing_pair_names
                and pair_key not in existing_pair_keys
                and reverse_pair_key not in existing_pair_keys
            ):
                self.world_spec.add_pair(
                    name=pair_name,
                    geomname1=object_geom_name,
                    geomname2=terrain_geom_name,
                    condim=object_terrain_condim,
                    friction=object_terrain_pair_friction,
                    solref=list(_GT_HAND_OBJECT_SOLREF),
                )
                existing_pair_names.add(pair_name)
                existing_pair_keys.add(pair_key)
                added_pairs += 1

        for hand_geom_name in sorted(set(hand_collision_geoms)):
            for object_geom_name in sorted(set(object_collision_geoms)):
                pair_name = f"{hand_geom_name}_{object_geom_name}"
                pair_key = (hand_geom_name, object_geom_name)
                reverse_pair_key = (object_geom_name, hand_geom_name)
                if (
                    pair_name in existing_pair_names
                    or pair_key in existing_pair_keys
                    or reverse_pair_key in existing_pair_keys
                ):
                    continue
                self.world_spec.add_pair(
                    name=pair_name,
                    geomname1=hand_geom_name,
                    geomname2=object_geom_name,
                    condim=6,
                    friction=hand_pair_friction,
                    solref=list(_GT_HAND_OBJECT_SOLREF),
                )
                existing_pair_names.add(pair_name)
                existing_pair_keys.add(pair_key)
                added_pairs += 1

        logger.info(
            "Applied GT MuJoCo object contact pairs: added={}, hands={}, objects={}, terrain={}, "
            "object_terrain_friction={}, hand_object_friction={}",
            added_pairs,
            sorted(set(hand_collision_geoms)),
            sorted(set(object_collision_geoms)),
            terrain_geom_name,
            object_terrain_friction,
            _GT_HAND_OBJECT_FRICTION,
        )

    def _apply_collision_settings(self, robot_spec: mujoco.MjSpec, robot_config: RobotConfig) -> None:
        """Apply collision settings based on unified self_collisions configuration.

        This matches IsaacGym/IsaacSim behavior programmatically by configuring
        MuJoCo collision classes based on the robot's self_collisions setting.

        Parameters
        ----------
        robot_spec : mujoco.MjSpec
            Robot specification to modify collision settings for.
        robot_config : RobotConfig
            Robot configuration containing self_collisions setting.
        """
        self._configure_robot_collisions(robot_spec, robot_config.asset.enable_self_collisions)

    def _configure_robot_collisions(self, robot_spec: mujoco.MjSpec, enable_self_collisions: bool) -> None:
        """Configure robot collision behavior using MuJoCo collision classes.

        Parameters
        ----------
        robot_spec : mujoco.MjSpec
            Robot specification to configure collisions for.
        enable_self_collisions : bool
            If True, robot parts collide with each other + environment.
            If False, robot parts only collide with environment.

        Notes
        -----
        Collision class system:
        - Robot parts: contype=1
        - Environment: contype=2, conaffinity=1
        - Robot conaffinity: 3 (both) if self_collisions, 2 (env only) if not
        """
        if enable_self_collisions:
            robot_conaffinity = 3  # Collide with robot (1) + environment (2) = 3
            collision_mode = "self + environment"
        else:
            robot_conaffinity = 2  # Only collide with environment (2)
            collision_mode = "environment only"

        bodies_processed = 0
        geoms_processed = 0

        # Apply collision settings to all robot bodies
        for body in robot_spec.bodies:
            if not body.name:
                # Skip unnamed bodies
                continue

            bodies_processed += 1
            for geom in body.geoms:
                # Skip geoms that have been explicitly configured away from defaults
                # Visual meshes typically have contype=0, conaffinity=0
                if geom.contype == 0 or geom.conaffinity == 0:
                    continue  # Skip visual/disabled collision geoms

                # Apply collision settings to geoms using default collision behavior
                # (contype=1, conaffinity=1 are MuJoCo defaults)
                if geom.contype == 1 and geom.conaffinity == 1:
                    geom.contype = 1  # Robot collision class
                    geom.conaffinity = robot_conaffinity  # Configurable based on self_collisions
                    geoms_processed += 1
                    logger.debug(f"Set {body.name} geom: contype=1, conaffinity={robot_conaffinity} ({collision_mode})")

        logger.info(f"Applied collision settings to {geoms_processed} geoms across {bodies_processed} bodies")

    def _filter_robot_worldbody(self, robot_spec: mujoco.MjSpec, cfg: MujocoXMLFilterCfg) -> mujoco.MjSpec:
        """Remove lights and ground elements from robot worldbody.

        Helper work-around while robot XMLs contain scene elements that should
        be managed by the scene manager instead.

        Parameters
        ----------
        robot_spec : mujoco.MjSpec
            Robot specification to filter.
        cfg : MujocoXMLFilterCfg
            Filtering configuration specifying what to remove.

        Returns
        -------
        mujoco.MjSpec
            Filtered robot specification.
        """
        # Remove lights if configured
        if cfg.remove_lights:
            for light in robot_spec.worldbody.lights:
                robot_spec.delete(light)

        # Remove ground geoms if configured
        if cfg.remove_ground:
            for geom in robot_spec.worldbody.geoms:
                if self._is_ground_geom(geom, cfg.ground_names):
                    robot_spec.delete(geom)

        return robot_spec

    def _is_ground_geom(self, geom: mujoco.MjSpec.Geom, ground_names: List[str]) -> bool:
        """Determine if a geometry represents ground/floor.

        Parameters
        ----------
        geom : mujoco.MjSpec.Geom
            Geometry to check.
        ground_names : List[str]
            List of names that indicate ground geometries.

        Returns
        -------
        bool
            True if the geometry represents ground/floor.
        """
        # Check by name
        if geom.name and any(name in geom.name.lower() for name in ground_names):
            return True

        return geom.type == mujoco.mjtGeom.mjGEOM_PLANE

    def compile(self) -> mujoco.MjModel:
        """Compile the final world model from the specification.

        Returns
        -------
        mujoco.MjModel
            Compiled MuJoCo model ready for simulation.
        """
        logger.info("Compiling world model using MjSpec")
        return self.world_spec.compile()
