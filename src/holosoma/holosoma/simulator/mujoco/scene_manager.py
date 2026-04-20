"""MuJoCo scene manager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from loguru import logger

from holosoma.config_types.robot import RobotConfig
from holosoma.config_types.simulator import MujocoXMLFilterCfg, SimulatorConfig
from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path


_SUPPORTED_OBJECT_SCENE_SPECS: dict[str, dict[str, tuple[str, str]]] = {
    "g1_29dof": {
        "objects_largebox": ("src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml", "largebox_link"),
        "largebox": ("src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml", "largebox_link"),
        "boxlarge": ("src/holosoma_retargeting/models/g1/g1_29dof_w_boxlarge.xml", "boxlarge"),
        "boxmedium": ("src/holosoma_retargeting/models/g1/g1_29dof_w_boxmedium.xml", "boxmedium"),
        "boxsmall": ("src/holosoma_retargeting/models/g1/g1_29dof_w_boxsmall.xml", "boxsmall"),
        "boxtiny": ("src/holosoma_retargeting/models/g1/g1_29dof_w_boxtiny.xml", "boxtiny"),
        "boxlong": ("src/holosoma_retargeting/models/g1/g1_29dof_w_boxlong.xml", "boxlong"),
    }
}

HOLOSOMA_PERCEPTION_CAMERA_NAME = "holosoma_perception_camera"
_CAMERA_TERRAIN_PROXY_ENV = "HOLOSOMA_ENABLE_CAMERA_TERRAIN_PROXY"
_CAMERA_TERRAIN_PROXY_SUFFIX = "_camera_proxy"
_LOAD_ROBOT_VISUAL_MESHES_ENV = "HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES"
_LOAD_OBJECT_VISUAL_MESHES_ENV = "HOLOSOMA_MUJOCO_LOAD_OBJECT_VISUAL_MESHES"
_WEB_DEMO_OBJECT_CONTACTS_ENV = "HOLOSOMA_MUJOCO_WEB_DEMO_OBJECT_CONTACTS"


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
        self._add_perception_camera_placeholder()
        self.robot_config: RobotConfig | None = None  # Set when adding robot
        self._object_urdf_by_name: dict[str, str] = {}
        self._object_body_name_by_name: dict[str, str] = {}

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

    def _add_perception_camera_placeholder(self) -> None:
        """Add a dedicated camera slot for runtime MuJoCo perception rendering."""
        self.world_spec.worldbody.add_camera(
            name=HOLOSOMA_PERCEPTION_CAMERA_NAME,
            pos=[0.0, 0.0, 1.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            fovy=60.0,
        )

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
            self._maybe_add_camera_render_proxy_geom(terrain_state)
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

            # Set environment collision properties so robot self_collision flag works
            # Environment collision class
            terrain_state.geom.contype = 2  # type: ignore[attr-defined]
            # Only collide with robot (class 1)
            terrain_state.geom.conaffinity = 1  # type: ignore[attr-defined]

    def _camera_render_proxy_enabled(self) -> bool:
        raw = os.environ.get(_CAMERA_TERRAIN_PROXY_ENV, "0").strip().lower()
        return raw not in {"", "0", "false", "no", "off"}

    def _create_ground_plane(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create a ground plane terrain geometry.

        Returns
        -------
        mujoco.MjSpec.Geom
            Ground plane geometry with configured physics properties.
        """
        # Create ground plane with hardcoded parameters and physics properties
        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            # Size=0 is rendered infinitely. Collision plane is always infinite.
            # Note: size.z is actually the rendered spacing betweeh the grid
            #       subdivisions (to improve lighting, shadows).
            size=[0, 0, 0.05],
            pos=[0, 0, 0],
            material="grid",
            friction=self._terrain_friction_triplet_from_state(terrain_state),
            solimp=[0.99, 0.99, 0.01, 0.5, 2],  # 5 elements: [dmin, dmax, width, midpoint, power]
            solref=[0.001, 1],  # 2 elements: [timeconst, dampratio]
        )

    def _maybe_add_camera_render_proxy_geom(self, terrain_state: TerrainTermBase) -> None:
        setattr(terrain_state, "camera_render_proxy_geom_name", None)
        if not self._camera_render_proxy_enabled():
            return
        if terrain_state.mesh is None:
            logger.warning("Camera terrain proxy requested but terrain mesh is unavailable.")
            return

        bounds = np.asarray(terrain_state.mesh.bounds, dtype=np.float64)
        if bounds.shape != (2, 3):
            logger.warning("Camera terrain proxy skipped: unexpected terrain bounds shape {}", bounds.shape)
            return

        mins = bounds[0]
        maxs = bounds[1]
        half_x = 0.5 * float(maxs[0] - mins[0])
        half_y = 0.5 * float(maxs[1] - mins[1])
        if half_x <= 0.0 or half_y <= 0.0:
            logger.warning(
                "Camera terrain proxy skipped: invalid terrain extents half_x={} half_y={}",
                half_x,
                half_y,
            )
            return

        top_z = float(maxs[2])
        thickness = 0.002
        geom = self.world_spec.worldbody.add_geom(
            name=f"{terrain_state.name}{_CAMERA_TERRAIN_PROXY_SUFFIX}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half_x, half_y, thickness],
            pos=[0.5 * float(mins[0] + maxs[0]), 0.5 * float(mins[1] + maxs[1]), top_z - thickness],
            material="grid",
            friction=self._terrain_friction_triplet_from_state(terrain_state),
            contype=0,
            conaffinity=0,
        )
        setattr(terrain_state, "camera_render_proxy_geom_name", geom.name)
        logger.info(
            "Added MuJoCo camera terrain proxy geom '{}' with size=({}, {}, {})",
            geom.name,
            half_x,
            half_y,
            thickness,
        )

    def _create_trimesh(self, terrain_state: TerrainTermBase) -> mujoco.MjSpec.Geom:
        """Create MuJoCo mesh terrain matching shared Terrain class behavior."""

        if terrain_state.mesh is None:
            raise ValueError("Terrain mesh data is required when using trimesh terrain type.")

        vertices = np.asarray(terrain_state.mesh.vertices, dtype=np.float32)
        faces = np.asarray(terrain_state.mesh.faces, dtype=np.int32)

        if vertices.size == 0 or faces.size == 0:
            raise ValueError("Terrain mesh is empty and cannot be used to create a mesh geom.")

        mesh_spec = self.world_spec.add_mesh(name="terrain")
        mesh_spec.uservert = vertices.flatten(order="C")
        mesh_spec.userface = faces.flatten(order="C")
        mesh_spec.smoothnormal = False

        return self.world_spec.worldbody.add_geom(
            name=terrain_state.name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_spec.name,
            pos=[0.0, 0.0, 0.0],
            material="solid_gray",
            friction=self._terrain_friction_triplet_from_state(terrain_state),
            solimp=[0.99, 0.99, 0.01, 0.5, 2],
            solref=[0.001, 1],
        )

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
        if min_height < 0:
            height_data_scaled = height_data_scaled - min_height + 1e-9
            z_offset = min_height
            logger.info(f"Shifted heightfield by {-min_height:.3f}m to ensure non-negative heights")

        max_height = height_data_scaled.max()
        min_height_final = height_data_scaled.min()

        # Calculate size parameters for MuJoCo hfield
        # size = [x_half, y_half, HEIGHT_RANGE, z_baseline]
        # Note: nrow/ncol are swapped for correct orientation
        height_range = max_height - min_height_final

        # Create heightfield asset
        hfield_spec = self.world_spec.add_hfield(name="terrain")
        hfield_spec.nrow = height_data.shape[1]  # swap: cols become rows
        hfield_spec.ncol = height_data.shape[0]  # swap: rows become cols
        hfield_spec.size = [0.5 * total_length, 0.5 * total_width, height_range, min_height_final]
        # MuJoCo expects raw elevation data in column-major (Fortran) order
        hfield_spec.userdata = height_data_scaled.flatten(order="F").tolist()

        logger.info(
            f"Created heightfield: {hfield_spec.nrow}x{hfield_spec.ncol},"
            " size=[{0.5 * total_length:.2f}, {0.5 * total_width:.2f}, {height_range:.3f}, {min_height_final:.3f}]"
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
            friction=self._terrain_friction_triplet_from_state(terrain_state),
            solimp=[0.99, 0.99, 0.01, 0.5, 2],
            solref=[0.001, 1],
        )

    def _terrain_friction_triplet_from_state(self, terrain_state: TerrainTermBase) -> list[float]:
        """Map shared terrain friction settings onto MuJoCo's [slide, spin, roll] tuple."""
        slide_friction = float(getattr(terrain_state, "static_friction", 1.0))
        dynamic_friction = float(getattr(terrain_state, "dynamic_friction", slide_friction))
        if not np.isclose(dynamic_friction, slide_friction):
            logger.info(
                "MuJoCo terrain geom uses [slide, spin, roll] friction; mapping Isaac-style "
                "static/dynamic terrain friction ({:.4f}, {:.4f}) to slide only.",
                slide_friction,
                dynamic_friction,
            )
        return [slide_friction, 0.005, 0.001]

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
        object_spec_to_attach: mujoco.MjSpec | None = None
        external_object_body_names: set[str] = set()
        if self._should_use_training_urdf_object_scene(robot_config):
            (
                robot_spec,
                object_spec_to_attach,
                robot_xml_path,
                object_urdf_by_name,
                object_body_name_by_name,
                external_object_body_names,
            ) = self._build_training_urdf_object_scene(robot_config)
            using_composite_object_scene = False
            self._object_urdf_by_name = object_urdf_by_name
            self._object_body_name_by_name = object_body_name_by_name
            logger.info(
                "Using MuJoCo training-URDF object scene '{}' with object URDF(s): {}",
                robot_xml_path,
                list(object_urdf_by_name.values()),
            )
        else:
            asset_root = robot_config.asset.asset_root
            if asset_root.startswith("@holosoma/"):
                asset_root = asset_root.replace("@holosoma", get_holosoma_root())
            robot_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)

            resolved_object_scene = self._resolve_supported_object_scene(robot_config)
            using_composite_object_scene = resolved_object_scene is not None
            if resolved_object_scene is not None:
                robot_xml_path, object_urdf_by_name, object_body_name_by_name = resolved_object_scene
                self._object_urdf_by_name = object_urdf_by_name
                self._object_body_name_by_name = object_body_name_by_name
                logger.info(
                    "Using MuJoCo composite object scene '{}' for actor(s): {}",
                    robot_xml_path,
                    list(object_urdf_by_name.keys()),
                )
            else:
                self._object_urdf_by_name = {}
                self._object_body_name_by_name = {}

            robot_spec = mujoco.MjSpec.from_file(robot_xml_path)

        logger.info(f"Adding robot from: {robot_xml_path} with prefix: {prefix}")
        self.robot_model_path = robot_xml_path

        if xml_filter and getattr(xml_filter, "enable", False):
            # Remove worldbody lights and ground|floor|plane geoms because they're added dynamically
            robot_spec = self._filter_robot_worldbody(robot_spec, xml_filter)

        self._maybe_align_composite_body_inertials_with_training_urdf(
            robot_spec,
            robot_config,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_copy_joint_defaults_from_reference_robot_xml(
            robot_spec,
            robot_config,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_copy_collision_geoms_from_reference_robot_xml(
            robot_spec,
            robot_config,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_replace_composite_collision_geoms_with_reference_robot_xml(
            robot_spec,
            robot_config,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_align_composite_hand_collision_geoms_with_training_urdf(
            robot_spec,
            robot_config,
            robot_xml_path=robot_xml_path,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_copy_tendons_from_reference_robot_xml(
            robot_spec,
            robot_config,
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_copy_contact_pairs_from_reference_robot_xml(
            robot_spec,
            robot_config,
            terrain_geom_name=str(getattr(terrain_state, "name", "floor")),
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_override_composite_object_properties(
            robot_spec,
            robot_config,
            terrain_geom_name=str(getattr(terrain_state, "name", "floor")),
            using_composite_object_scene=using_composite_object_scene,
        )
        self._maybe_add_default_actuators(robot_spec, robot_config)

        if object_spec_to_attach is not None:
            self._configure_object_collisions(object_spec_to_attach)
            self._maybe_override_object_properties(
                object_spec_to_attach,
                robot_config,
                terrain_geom_name=str(getattr(terrain_state, "name", "floor")),
                target_body_names=external_object_body_names,
            )

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
        if object_spec_to_attach is not None:
            object_site = self.world_spec.worldbody.add_site(name="object_spawn", pos=[0, 0, 0.0], quat=[1, 0, 0, 0])
            self.world_spec.attach(object_spec_to_attach, site=object_site, prefix="object_")
            self._maybe_add_web_demo_object_contact_pairs()

        # Store prefix for later use by simulator
        self.robot_prefix = prefix

    @staticmethod
    def _resolve_robot_asset_root(robot_config: RobotConfig) -> Path:
        asset_root = str(robot_config.asset.asset_root)
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        return Path(asset_root).expanduser().resolve()

    @classmethod
    def _resolve_robot_urdf_path(cls, robot_config: RobotConfig) -> Path:
        return (cls._resolve_robot_asset_root(robot_config) / str(robot_config.asset.urdf_file)).resolve()

    @staticmethod
    def _configure_urdf_meshdir(spec: mujoco.MjSpec, urdf_path: Path) -> None:
        mesh_files = [Path(str(mesh.file)) for mesh in spec.meshes if str(getattr(mesh, "file", "")).strip()]
        meshdir_candidates = [urdf_path.parent, urdf_path.parent / "meshes"]
        for candidate in meshdir_candidates:
            if not candidate.is_dir():
                continue
            if mesh_files and not all(mesh_file.is_absolute() or (candidate / mesh_file).is_file() for mesh_file in mesh_files):
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
    def _load_urdf_spec(cls, urdf_path: Path, *, load_visual_meshes: bool = False) -> mujoco.MjSpec:
        if not load_visual_meshes:
            return mujoco.MjSpec.from_file(str(urdf_path))

        root = ET.parse(urdf_path).getroot()
        mujoco_elem = root.find("mujoco")
        if mujoco_elem is None:
            mujoco_elem = ET.Element("mujoco")
            root.insert(0, mujoco_elem)
        compiler_elem = mujoco_elem.find("compiler")
        if compiler_elem is None:
            compiler_elem = ET.SubElement(mujoco_elem, "compiler")

        compiler_elem.set("discardvisual", "false")
        meshdir_raw = str(compiler_elem.get("meshdir", "") or "").strip()
        mesh_names = {
            Path(str(mesh_tag.get("filename") or "")).name
            for mesh_tag in root.findall(".//mesh")
            if str(mesh_tag.get("filename") or "").strip()
        }
        meshdir_candidates: list[Path] = []
        if meshdir_raw:
            meshdir_candidate = Path(meshdir_raw).expanduser()
            if not meshdir_candidate.is_absolute():
                meshdir_candidate = urdf_path.parent / meshdir_candidate
            meshdir_candidates.append(meshdir_candidate)
        meshdir_candidates += [urdf_path.parent / "meshes", urdf_path.parent]
        for meshdir_candidate in meshdir_candidates:
            if meshdir_candidate.is_dir() and (
                not mesh_names or all((meshdir_candidate / mesh_name).is_file() for mesh_name in mesh_names)
            ):
                compiler_elem.set("meshdir", str(meshdir_candidate.resolve()))
                break
        else:
            fallback_meshdir = meshdir_candidates[0] if meshdir_candidates else urdf_path.parent
            compiler_elem.set("meshdir", str(fallback_meshdir.resolve()))

        return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))

    @staticmethod
    def _find_spec_body(spec: mujoco.MjSpec, body_name: str) -> mujoco.MjSpec.Body:
        for body in spec.bodies:
            if body.name == body_name:
                return body
        raise ValueError(f"Body '{body_name}' not found in MuJoCo spec")

    @staticmethod
    def _select_object_root_body_name(object_spec: mujoco.MjSpec) -> str:
        for preferred_name in ("baseLink", "base_link"):
            for body in object_spec.bodies:
                if body.name == preferred_name:
                    return preferred_name
        for body in object_spec.bodies:
            if body.name:
                return str(body.name)
        raise ValueError("Could not resolve a named object root body from the MuJoCo object URDF")

    @classmethod
    def _should_use_training_urdf_object_scene(cls, robot_config: RobotConfig) -> bool:
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "enabled", False):
            return False
        if not getattr(object_cfg, "object_urdf_path", None):
            return False
        return bool(getattr(object_cfg, "mujoco_use_training_urdf_scene", False))

    @classmethod
    def _build_training_urdf_object_scene(
        cls,
        robot_config: RobotConfig,
    ) -> tuple[mujoco.MjSpec, mujoco.MjSpec, str, dict[str, str], dict[str, str], set[str]]:
        robot_urdf = cls._resolve_robot_urdf_path(robot_config)
        load_robot_visual_meshes = cls._env_flag(_LOAD_ROBOT_VISUAL_MESHES_ENV)
        robot_spec = cls._load_urdf_spec(robot_urdf, load_visual_meshes=load_robot_visual_meshes)
        cls._configure_urdf_meshdir(robot_spec, robot_urdf)
        if load_robot_visual_meshes:
            logger.info(
                "Loaded robot URDF visual meshes for MuJoCo scene via {}=1: {} geom(s), {} mesh asset(s).",
                _LOAD_ROBOT_VISUAL_MESHES_ENV,
                len(robot_spec.geoms),
                len(robot_spec.meshes),
            )

        robot_root_body_name = str(robot_config.body_names[0])
        robot_root_body = cls._find_spec_body(robot_spec, robot_root_body_name)
        robot_root_body.add_freejoint(name="floating_base_joint")

        object_cfg = getattr(robot_config, "object", None)
        assert object_cfg is not None
        object_urdf = cls._resolve_single_object_urdf(str(object_cfg.object_urdf_path))
        load_object_visual_meshes = cls._env_flag(_LOAD_OBJECT_VISUAL_MESHES_ENV)
        object_spec = cls._load_urdf_spec(object_urdf, load_visual_meshes=load_object_visual_meshes)
        cls._configure_urdf_meshdir(object_spec, object_urdf)
        if load_object_visual_meshes:
            logger.info(
                "Loaded object URDF visual meshes for MuJoCo scene via {}=1: {} geom(s), {} mesh asset(s).",
                _LOAD_OBJECT_VISUAL_MESHES_ENV,
                len(object_spec.geoms),
                len(object_spec.meshes),
            )

        object_root_body_name = cls._select_object_root_body_name(object_spec)
        object_root_body = cls._find_spec_body(object_spec, object_root_body_name)
        object_root_body.add_freejoint(name="object_freejoint")

        object_actor_name = "object"
        object_prefix = "object_"
        object_body_names = {
            str(body.name)
            for body in object_spec.bodies
            if body.name and str(body.name).lower() not in {"world", "universe"}
        }
        return (
            robot_spec,
            object_spec,
            str(robot_urdf),
            {object_actor_name: str(object_urdf)},
            {object_actor_name: f"{object_prefix}{object_root_body_name}"},
            object_body_names,
        )

    def _maybe_add_default_actuators(self, robot_spec: mujoco.MjSpec, robot_config: RobotConfig) -> None:
        """Inject default torque actuators for MuJoCo-only scenes when explicitly requested."""
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_add_default_actuators", False):
            return

        if len(robot_spec.actuators) > 0:
            logger.info("Skipping default actuator injection because scene already defines {} actuators", len(robot_spec.actuators))
            return

        if len(robot_config.dof_names) != len(robot_config.dof_effort_limit_list):
            raise ValueError(
                "Cannot inject MuJoCo default actuators because DOF names and effort limits have different lengths: "
                f"{len(robot_config.dof_names)} vs {len(robot_config.dof_effort_limit_list)}"
            )

        for dof_name, effort_limit in zip(robot_config.dof_names, robot_config.dof_effort_limit_list):
            torque_limit = float(abs(effort_limit))
            robot_spec.add_actuator(
                name=dof_name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=dof_name,
                gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dyntype=mujoco.mjtDyn.mjDYN_NONE,
                gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                biastype=mujoco.mjtBias.mjBIAS_NONE,
                ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
                ctrlrange=[-torque_limit, torque_limit],
            )

        logger.info("Injected {} default torque actuators into MuJoCo scene for '{}'", len(robot_spec.actuators), robot_config.asset.robot_type)

    def _maybe_copy_joint_defaults_from_reference_robot_xml(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        using_composite_object_scene: bool,
    ) -> None:
        """Copy standalone MuJoCo joint defaults into composite object scenes when explicitly requested.

        Composite MuJoCo scenes used for object carry can omit joint defaults that exist in the
        standalone robot XML. This creates a control/dynamics mismatch specific to MuJoCo sim2sim.
        Keep this behind a MuJoCo-only object config flag so Isaac Sim behavior is untouched.
        """

        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_copy_joint_defaults_from_robot_xml", False):
            return
        using_training_urdf_object_scene = self._should_use_training_urdf_object_scene(robot_config)
        if not using_composite_object_scene and not using_training_urdf_object_scene:
            logger.info("Skipping joint-default copy because current MuJoCo scene does not use object verification")
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)
        reference_joints = {joint.name: joint for joint in reference_spec.joints if joint.name}

        missing_reference_joints: list[str] = []
        updated_joint_count = 0
        updated_fields_count = 0
        dof_name_set = set(robot_config.dof_names)

        def _copy_numeric_joint_field(joint: Any, reference_joint: Any, field_name: str) -> int:
            current_value = np.asarray(getattr(joint, field_name), dtype=np.float64)
            reference_value = np.asarray(getattr(reference_joint, field_name), dtype=np.float64)
            if current_value.shape == reference_value.shape and np.allclose(current_value, reference_value):
                return 0

            flattened_reference = reference_value.reshape(-1)
            setattr(
                joint,
                field_name,
                float(flattened_reference[0]) if reference_value.ndim == 0 or reference_value.size == 1 else reference_value.tolist(),
            )
            return 1

        for joint in robot_spec.joints:
            if not joint.name or joint.name not in dof_name_set:
                continue

            reference_joint = reference_joints.get(joint.name)
            if reference_joint is None:
                missing_reference_joints.append(joint.name)
                continue

            changed_fields = 0
            changed_fields += _copy_numeric_joint_field(joint, reference_joint, "armature")
            changed_fields += _copy_numeric_joint_field(joint, reference_joint, "damping")
            changed_fields += _copy_numeric_joint_field(joint, reference_joint, "frictionloss")

            joint_solimp_limit = np.asarray(joint.solimp_limit, dtype=np.float64)
            reference_solimp_limit = np.asarray(reference_joint.solimp_limit, dtype=np.float64)
            if not np.allclose(joint_solimp_limit, reference_solimp_limit):
                joint.solimp_limit = reference_solimp_limit.tolist()
                changed_fields += 1

            if changed_fields > 0:
                updated_joint_count += 1
                updated_fields_count += changed_fields

        if missing_reference_joints:
            raise ValueError(
                "Cannot copy MuJoCo joint defaults because these DOFs were not found in the standalone robot XML: "
                f"{missing_reference_joints}"
            )

        logger.info(
            "Copied MuJoCo joint defaults from '{}' into MuJoCo object scene for {} joint(s) across {} field update(s)",
            reference_xml_path,
            updated_joint_count,
            updated_fields_count,
        )

    def _maybe_copy_tendons_from_reference_robot_xml(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        using_composite_object_scene: bool,
    ) -> None:
        """Copy standalone MuJoCo tendons into composite object scenes when explicitly requested."""

        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_copy_tendons_from_robot_xml", False):
            return
        using_training_urdf_object_scene = self._should_use_training_urdf_object_scene(robot_config)
        if not using_composite_object_scene and not using_training_urdf_object_scene:
            logger.info("Skipping tendon copy because current MuJoCo scene does not use object verification")
            return
        if len(robot_spec.tendons) > 0:
            logger.info("Skipping tendon copy because current MuJoCo robot spec already defines {} tendons", len(robot_spec.tendons))
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)
        if len(reference_spec.tendons) == 0:
            logger.info("Reference robot XML '{}' defines no tendons; skipping tendon copy", reference_xml_path)
            return

        copied_tendon_count = 0
        copied_wrap_count = 0

        def _seq_or_none(values: Any) -> list[float] | None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return None
            return arr.tolist()

        def _scalar_or_seq_or_none(values: Any) -> float | list[float] | None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[0]) if arr.ndim == 0 or arr.size == 1 else arr.tolist()

        for tendon_idx, reference_tendon in enumerate(reference_spec.tendons):
            tendon = robot_spec.add_tendon(
                name=reference_tendon.name or None,
                stiffness=_scalar_or_seq_or_none(reference_tendon.stiffness),
                springlength=_seq_or_none(reference_tendon.springlength),
                damping=_scalar_or_seq_or_none(reference_tendon.damping),
                frictionloss=float(reference_tendon.frictionloss),
                solref_friction=_seq_or_none(reference_tendon.solref_friction),
                solimp_friction=_seq_or_none(reference_tendon.solimp_friction),
                armature=float(reference_tendon.armature),
                limited=int(reference_tendon.limited),
                actfrclimited=int(reference_tendon.actfrclimited),
                range=_seq_or_none(reference_tendon.range),
                actfrcrange=_seq_or_none(reference_tendon.actfrcrange),
                margin=float(reference_tendon.margin),
                solref_limit=_seq_or_none(reference_tendon.solref_limit),
                solimp_limit=_seq_or_none(reference_tendon.solimp_limit),
                material=reference_tendon.material or None,
                width=float(reference_tendon.width),
                rgba=_seq_or_none(reference_tendon.rgba),
                group=int(reference_tendon.group),
                userdata=_seq_or_none(reference_tendon.userdata),
                info=reference_tendon.info or None,
            )

            for wrap in reference_tendon.path:
                target_name = getattr(getattr(wrap, "target", None), "name", None)
                if wrap.type == mujoco.mjtWrap.mjWRAP_JOINT:
                    if not target_name:
                        raise ValueError(
                            f"Reference tendon {tendon_idx} contains a joint wrap without a target joint name"
                        )
                    tendon.wrap_joint(str(target_name), float(wrap.coef))
                elif wrap.type == mujoco.mjtWrap.mjWRAP_PULLEY:
                    tendon.wrap_pulley(float(wrap.divisor))
                elif wrap.type == mujoco.mjtWrap.mjWRAP_SITE:
                    if not target_name:
                        raise ValueError(
                            f"Reference tendon {tendon_idx} contains a site wrap without a target site name"
                        )
                    tendon.wrap_site(str(target_name))
                elif wrap.type == mujoco.mjtWrap.mjWRAP_GEOM:
                    side_site_name = getattr(getattr(wrap, "sidesite", None), "name", None)
                    if not target_name or not side_site_name:
                        raise ValueError(
                            f"Reference tendon {tendon_idx} contains a geom wrap without target/side site names"
                        )
                    tendon.wrap_geom(str(target_name), str(side_site_name))
                else:
                    raise ValueError(f"Unsupported tendon wrap type in reference robot XML: {wrap.type}")
                copied_wrap_count += 1

            copied_tendon_count += 1

        logger.info(
            "Copied {} MuJoCo tendon(s) with {} wrap(s) from '{}' into MuJoCo object scene",
            copied_tendon_count,
            copied_wrap_count,
            reference_xml_path,
        )

    def _maybe_copy_collision_geoms_from_reference_robot_xml(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        using_composite_object_scene: bool,
    ) -> None:
        """Copy standalone MuJoCo collision geoms into object scenes when explicitly requested.

        Training-URDF object scenes are assembled from the Isaac-style robot URDF plus a
        separate object URDF. The URDF foot collision set does not expose the named MuJoCo
        foot capsules/contact pairs from ``g1_29dof.xml``, so direct MuJoCo replay uses a
        different support model and can fall before object interaction. For that path we
        replace the robot's active URDF collision geoms with the reference XML collision set,
        while preserving rubber-hand geoms that are intentionally added to the URDF.
        """

        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_copy_collision_geoms_from_robot_xml", False):
            return
        using_training_urdf_object_scene = self._should_use_training_urdf_object_scene(robot_config)
        if not using_composite_object_scene and not using_training_urdf_object_scene:
            logger.info("Skipping collision-geom copy because current MuJoCo scene is not an object scene")
            return
        urdf_file = str(getattr(robot_config.asset, "urdf_file", "") or "")
        if using_composite_object_scene and urdf_file.endswith("main_mesh_collision_halfspherehand.urdf"):
            logger.info(
                "Skipping collision-geom copy from reference MuJoCo XML because training URDF '{}' already "
                "defines the intended carry colliders and the reference XML uses a mismatched collision set.",
                urdf_file,
            )
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)

        target_bodies = {body.name: body for body in robot_spec.bodies if body.name}
        object_body_names = set(getattr(self, "_object_body_name_by_name", {}).values())

        disabled_geom_count = 0
        if using_training_urdf_object_scene:
            for body in robot_spec.bodies:
                if not body.name or body.name in object_body_names:
                    continue
                if "rubber_hand" in str(body.name).lower():
                    continue
                for geom in body.geoms:
                    if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                        continue
                    geom.contype = 0
                    geom.conaffinity = 0
                    disabled_geom_count += 1

        existing_geom_names = {geom.name for body in robot_spec.bodies for geom in body.geoms if geom.name}
        skip_reference_hand_collision_names: set[str] = set()
        if using_training_urdf_object_scene:
            active_rubber_hand_bodies = {
                str(body.name)
                for body in robot_spec.bodies
                if body.name
                and "rubber_hand" in str(body.name).lower()
                and any(int(geom.contype) != 0 and int(geom.conaffinity) != 0 for geom in body.geoms)
            }
            if "left_rubber_hand" in active_rubber_hand_bodies:
                skip_reference_hand_collision_names.add("left_hand_collision")
            if "right_rubber_hand" in active_rubber_hand_bodies:
                skip_reference_hand_collision_names.add("right_hand_collision")

        copied_geom_count = 0
        skipped_reference_hand_collision_count = 0

        def _seq(values: Any, *, allow_zero: bool = True) -> list[float] | None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return None
            if not allow_zero and np.allclose(arr, 0.0):
                return None
            return arr.tolist()

        for reference_body in reference_spec.bodies:
            if not reference_body.name or reference_body.name not in target_bodies:
                continue

            target_body = target_bodies[reference_body.name]
            for reference_geom in reference_body.geoms:
                if int(reference_geom.contype) == 0 and int(reference_geom.conaffinity) == 0:
                    continue
                if reference_geom.name and reference_geom.name in skip_reference_hand_collision_names:
                    skipped_reference_hand_collision_count += 1
                    continue
                if reference_geom.name and reference_geom.name in existing_geom_names:
                    continue

                kwargs = {
                    "name": reference_geom.name or None,
                    "type": reference_geom.type,
                    "size": _seq(reference_geom.size),
                    "pos": _seq(reference_geom.pos),
                    "quat": _seq(reference_geom.quat),
                    "fromto": _seq(reference_geom.fromto, allow_zero=False),
                    "contype": int(reference_geom.contype),
                    "conaffinity": int(reference_geom.conaffinity),
                    "condim": int(reference_geom.condim),
                    "group": int(reference_geom.group),
                    "density": float(reference_geom.density),
                    "friction": _seq(reference_geom.friction),
                    "solref": _seq(reference_geom.solref),
                    "solimp": _seq(reference_geom.solimp),
                    "rgba": _seq(reference_geom.rgba),
                    "meshname": reference_geom.meshname or None,
                    "material": reference_geom.material or None,
                    "priority": int(reference_geom.priority),
                    "margin": float(reference_geom.margin),
                    "gap": float(reference_geom.gap),
                }
                kwargs = {key: value for key, value in kwargs.items() if value is not None}
                target_body.add_geom(**kwargs)
                copied_geom_count += 1
                if reference_geom.name:
                    existing_geom_names.add(reference_geom.name)

        logger.info(
            "Copied {} MuJoCo collision geom(s) from '{}' into MuJoCo object scene; disabled {} URDF geom(s); "
            "skipped {} reference hand geom(s) because rubber-hand collision is active",
            copied_geom_count,
            reference_xml_path,
            disabled_geom_count,
            skipped_reference_hand_collision_count,
        )

    def _maybe_replace_composite_collision_geoms_with_reference_robot_xml(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        using_composite_object_scene: bool,
    ) -> None:
        """Replace composite-scene robot collisions with the simplified reference MuJoCo set.

        The retargeting composite ``g1_29dof_w_*.xml`` scenes expose per-link mesh geoms as
        active collisions. Training uses the simplified robot collision set from
        ``g1_29dof.xml`` plus the half-sphere hand colliders from
        ``main_mesh_collision_halfspherehand.urdf``. When we keep the composite mesh collisions
        active, MuJoCo gets much denser contacts than training, which destabilizes carry rollouts.
        """

        if not using_composite_object_scene:
            return

        urdf_file = str(getattr(robot_config.asset, "urdf_file", "") or "")
        if not urdf_file.endswith("main_mesh_collision_halfspherehand.urdf"):
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)

        target_bodies = {body.name: body for body in robot_spec.bodies if body.name}
        object_body_names = set(getattr(self, "_object_body_name_by_name", {}).values())
        existing_geom_names = {geom.name for body in robot_spec.bodies for geom in body.geoms if geom.name}

        disabled_geom_count = 0
        for body in robot_spec.bodies:
            if not body.name or body.name in object_body_names:
                continue
            for geom in body.geoms:
                if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
                    continue
                geom.contype = 0
                geom.conaffinity = 0
                disabled_geom_count += 1

        copied_geom_count = 0

        def _seq(values: Any, *, allow_zero: bool = True) -> list[float] | None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return None
            if not allow_zero and np.allclose(arr, 0.0):
                return None
            return arr.tolist()

        for reference_body in reference_spec.bodies:
            if not reference_body.name or reference_body.name not in target_bodies:
                continue
            if reference_body.name in object_body_names:
                continue

            target_body = target_bodies[reference_body.name]
            for reference_geom in reference_body.geoms:
                if int(reference_geom.contype) == 0 and int(reference_geom.conaffinity) == 0:
                    continue
                if reference_geom.name and reference_geom.name in existing_geom_names:
                    continue

                kwargs = {
                    "name": reference_geom.name or None,
                    "type": reference_geom.type,
                    "size": _seq(reference_geom.size),
                    "pos": _seq(reference_geom.pos),
                    "quat": _seq(reference_geom.quat),
                    "fromto": _seq(reference_geom.fromto, allow_zero=False),
                    "contype": int(reference_geom.contype),
                    "conaffinity": int(reference_geom.conaffinity),
                    "condim": int(reference_geom.condim),
                    "group": int(reference_geom.group),
                    "density": float(reference_geom.density),
                    "friction": _seq(reference_geom.friction),
                    "solref": _seq(reference_geom.solref),
                    "solimp": _seq(reference_geom.solimp),
                    "rgba": _seq(reference_geom.rgba),
                    "meshname": reference_geom.meshname or None,
                    "material": reference_geom.material or None,
                    "priority": int(reference_geom.priority),
                    "margin": float(reference_geom.margin),
                    "gap": float(reference_geom.gap),
                }
                kwargs = {key: value for key, value in kwargs.items() if value is not None}
                target_body.add_geom(**kwargs)
                copied_geom_count += 1
                if reference_geom.name:
                    existing_geom_names.add(reference_geom.name)

        logger.info(
            "Replaced composite robot collisions with {} simplified geom(s) from '{}'; disabled {} composite geom(s)",
            copied_geom_count,
            reference_xml_path,
            disabled_geom_count,
        )

    def _maybe_align_composite_body_inertials_with_training_urdf(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        using_composite_object_scene: bool,
    ) -> None:
        """Align composite MuJoCo body inertials with the training Isaac URDF when possible.

        The object-carry MuJoCo scenes under ``holosoma_retargeting/models`` were produced from a
        different robot model than the Isaac training path. For ``g1_29dof_w_object``, this leads to
        materially different body masses/inertias on key carry links (torso, wrists, etc.), so copy
        the inertial parameters from the training URDF into the composite scene before compiling it.
        """

        if not using_composite_object_scene:
            return

        urdf_file = str(getattr(robot_config.asset, "urdf_file", "") or "")
        if not urdf_file.endswith("main_mesh_collision_halfspherehand.urdf"):
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        training_urdf_path = Path(asset_root) / urdf_file
        training_urdf_path = training_urdf_path.resolve()
        if not training_urdf_path.is_file():
            raise FileNotFoundError(
                "Expected training URDF for MuJoCo carry inertial alignment at "
                f"'{training_urdf_path}', but it was not found."
            )

        def _parse_xyz(raw: str | None) -> list[float]:
            values = [float(v) for v in (raw or "0 0 0").split()]
            if len(values) != 3:
                raise ValueError(f"Expected 3 inertial-origin values, got {values}")
            return values

        def _rpy_to_quat_wxyz(rpy: list[float]) -> list[float]:
            roll, pitch, yaw = [float(v) for v in rpy]
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            return [
                float(cr * cp * cy + sr * sp * sy),
                float(sr * cp * cy - cr * sp * sy),
                float(cr * sp * cy + sr * cp * sy),
                float(cr * cp * sy - sr * sp * cy),
            ]

        def _quat_to_matrix(quat_wxyz: list[float]) -> np.ndarray:
            w, x, y, z = [float(v) for v in quat_wxyz]
            return np.array(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                    [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                    [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
                ],
                dtype=np.float64,
            )

        def _matrix_to_quat_wxyz(rot: np.ndarray) -> list[float]:
            trace = float(np.trace(rot))
            if trace > 0.0:
                s = np.sqrt(trace + 1.0) * 2.0
                w = 0.25 * s
                x = (rot[2, 1] - rot[1, 2]) / s
                y = (rot[0, 2] - rot[2, 0]) / s
                z = (rot[1, 0] - rot[0, 1]) / s
            elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
                s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif rot[1, 1] > rot[2, 2]:
                s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
            quat = np.array([w, x, y, z], dtype=np.float64)
            quat /= np.linalg.norm(quat)
            return quat.tolist()

        root = ET.parse(training_urdf_path).getroot()
        inertials_by_link: dict[str, dict[str, list[float] | float]] = {}
        for link in root.findall("link"):
            link_name = link.attrib.get("name")
            inertial = link.find("inertial")
            if not link_name or inertial is None:
                continue
            mass_elem = inertial.find("mass")
            inertia_elem = inertial.find("inertia")
            if mass_elem is None or inertia_elem is None:
                continue
            origin_elem = inertial.find("origin")
            xyz = _parse_xyz(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
            rpy = _parse_xyz(origin_elem.attrib.get("rpy") if origin_elem is not None else None)
            origin_rot = _quat_to_matrix(_rpy_to_quat_wxyz(rpy))
            inertia_body = np.array(
                [
                    [
                        float(inertia_elem.attrib.get("ixx", "0")),
                        float(inertia_elem.attrib.get("ixy", "0")),
                        float(inertia_elem.attrib.get("ixz", "0")),
                    ],
                    [
                        float(inertia_elem.attrib.get("ixy", "0")),
                        float(inertia_elem.attrib.get("iyy", "0")),
                        float(inertia_elem.attrib.get("iyz", "0")),
                    ],
                    [
                        float(inertia_elem.attrib.get("ixz", "0")),
                        float(inertia_elem.attrib.get("iyz", "0")),
                        float(inertia_elem.attrib.get("izz", "0")),
                    ],
                ],
                dtype=np.float64,
            )
            inertia_in_origin = origin_rot.T @ inertia_body @ origin_rot
            eigvals, eigvecs = np.linalg.eigh(inertia_in_origin)
            if np.linalg.det(eigvecs) < 0.0:
                eigvecs[:, 0] *= -1.0
            principal_rot = origin_rot @ eigvecs
            inertials_by_link[link_name] = {
                "mass": float(mass_elem.attrib["value"]),
                "ipos": xyz,
                "iquat": _matrix_to_quat_wxyz(principal_rot),
                "inertia": np.maximum(eigvals, 0.0).tolist(),
            }

        updated_body_count = 0
        for body in robot_spec.bodies:
            body_name = body.name
            if not body_name or body_name not in inertials_by_link:
                continue
            aligned = inertials_by_link[body_name]
            body.mass = float(aligned["mass"])
            body.ipos = list(aligned["ipos"])
            body.iquat = list(aligned["iquat"])
            body.inertia = list(aligned["inertia"])
            updated_body_count += 1

        logger.info(
            "Aligned {} composite MuJoCo body inertials from training URDF '{}'.",
            updated_body_count,
            training_urdf_path,
        )

    def _maybe_align_composite_hand_collision_geoms_with_training_urdf(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        robot_xml_path: str,
        using_composite_object_scene: bool,
    ) -> None:
        """Align composite MuJoCo hand collision geoms with the training Isaac URDF when possible.

        The object-carry training path uses ``main_mesh_collision_halfspherehand.urdf`` in Isaac Sim,
        which provides a half-sphere palm collider on each wrist and does not expose the composite
        scene's extra rubber-hand/thumb/pinky collision geoms. The MuJoCo split path loads a
        composite XML instead, so reconcile the hand colliders here without changing body topology.
        """

        if not using_composite_object_scene:
            return

        urdf_file = str(getattr(robot_config.asset, "urdf_file", "") or "")
        if not urdf_file.endswith("main_mesh_collision_halfspherehand.urdf"):
            return

        existing_geom_names = {geom.name for body in robot_spec.bodies for geom in body.geoms if geom.name}
        hand_targets = (
            (
                "left_wrist_yaw_link",
                "left_sphere_hand",
                "left_hand_collision",
                ("left_rubber_hand_link", "left_thumb_link", "left_pinky_link"),
            ),
            (
                "right_wrist_yaw_link",
                "right_sphere_hand",
                "right_hand_collision",
                ("right_rubber_hand_link", "right_thumb_link", "right_pinky_link"),
            ),
        )

        # Reuse the same half-sphere asset that the retargeting scene already ships with.
        half_sphere_asset = Path(robot_xml_path).resolve().parent / "assets" / "half_sphere.obj"
        if not half_sphere_asset.is_file():
            raise FileNotFoundError(
                "Expected half-sphere hand mesh for MuJoCo carry alignment at "
                f"'{half_sphere_asset}', but it was not found."
            )

        half_sphere_mesh_name = "halfsphere_hand_mesh"
        existing_mesh_names = {mesh.name for mesh in robot_spec.meshes if mesh.name}
        if half_sphere_mesh_name not in existing_mesh_names:
            robot_spec.add_mesh(name=half_sphere_mesh_name, file=str(half_sphere_asset))

        target_bodies = {body.name: body for body in robot_spec.bodies if body.name}
        disabled_names_global = {
            "left_hand_collision",
            "right_hand_collision",
            "left_rubber_hand_link",
            "right_rubber_hand_link",
            "left_thumb_link",
            "right_thumb_link",
            "left_pinky_link",
            "right_pinky_link",
        }
        added_geom_count = 0
        disabled_geom_count = 0
        for body_name, sphere_geom_name, legacy_capsule_name, extra_geom_names in hand_targets:
            body = target_bodies.get(body_name)
            if body is None:
                raise ValueError(
                    "Cannot align MuJoCo half-sphere hand collisions because body "
                    f"'{body_name}' is missing from composite scene '{robot_xml_path}'."
                )

            if sphere_geom_name not in existing_geom_names:
                body.add_geom(
                    name=sphere_geom_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=half_sphere_mesh_name,
                    pos=[0.029, -0.003, 0.0],
                    quat=[0.707107, 0.0, 0.707107, 0.0],
                )
                existing_geom_names.add(sphere_geom_name)
                added_geom_count += 1

        for body in robot_spec.bodies:
            for geom in body.geoms:
                if geom.name in disabled_names_global and (int(geom.contype) != 0 or int(geom.conaffinity) != 0):
                    geom.contype = 0
                    geom.conaffinity = 0
                    disabled_geom_count += 1

        logger.info(
            "Aligned MuJoCo composite hand collisions with training URDF '{}': added {} half-sphere geom(s), "
            "disabled {} mismatched hand geom(s)",
            urdf_file,
            added_geom_count,
            disabled_geom_count,
        )

    def _maybe_copy_contact_pairs_from_reference_robot_xml(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        terrain_geom_name: str,
        using_composite_object_scene: bool,
    ) -> None:
        """Copy standalone MuJoCo contact pairs into composite object scenes when explicitly requested."""

        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_copy_contact_pairs_from_robot_xml", False):
            return
        using_training_urdf_object_scene = self._should_use_training_urdf_object_scene(robot_config)
        if not using_composite_object_scene and not using_training_urdf_object_scene:
            logger.info("Skipping contact-pair copy because current MuJoCo scene does not use object verification")
            return
        if len(robot_spec.pairs) > 0:
            logger.info("Skipping contact-pair copy because current MuJoCo robot spec already defines {} pair(s)", len(robot_spec.pairs))
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)

        existing_geom_names = {geom.name for body in robot_spec.bodies for geom in body.geoms if geom.name}
        existing_pair_names = {pair.name for pair in robot_spec.pairs if pair.name}
        copied_pair_count = 0

        def _seq(values: Any) -> list[float] | None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return None
            return arr.tolist()

        for reference_pair in reference_spec.pairs:
            if reference_pair.name and reference_pair.name in existing_pair_names:
                continue

            geom1 = str(reference_pair.geomname1)
            geom2 = str(reference_pair.geomname2)
            if geom1 not in existing_geom_names:
                continue
            if geom2 == "floor":
                geom2 = terrain_geom_name

            kwargs = {
                "name": reference_pair.name or None,
                "geomname1": geom1,
                "geomname2": geom2,
                "condim": int(reference_pair.condim),
                "friction": _seq(reference_pair.friction),
                "solref": _seq(reference_pair.solref),
                "solreffriction": _seq(reference_pair.solreffriction),
                "solimp": _seq(reference_pair.solimp),
                "margin": float(reference_pair.margin),
                "gap": float(reference_pair.gap),
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            robot_spec.add_pair(**kwargs)
            copied_pair_count += 1
            if reference_pair.name:
                existing_pair_names.add(reference_pair.name)

        logger.info(
            "Copied {} MuJoCo contact pair(s) from '{}' into MuJoCo object scene (terrain geom '{}')",
            copied_pair_count,
            reference_xml_path,
            terrain_geom_name,
        )

    def _configure_object_collisions(self, object_spec: mujoco.MjSpec) -> None:
        web_demo_contacts = self._env_flag(_WEB_DEMO_OBJECT_CONTACTS_ENV)
        object_contype = 4
        object_conaffinity = 11 if web_demo_contacts else 3
        collision_geom_index = 0
        visual_geom_index = 0
        updated_geoms = 0
        visual_geoms = 0
        for body in object_spec.bodies:
            for geom in body.geoms:
                geom_contype = int(geom.contype)
                geom_conaffinity = int(geom.conaffinity)
                if geom_contype == 0 or geom_conaffinity == 0:
                    if web_demo_contacts:
                        visual_geom_index += 1
                        if not geom.name:
                            geom.name = "visual" if visual_geom_index == 1 else f"visual_{visual_geom_index}"
                        geom.contype = 0
                        geom.conaffinity = 0
                        geom.group = 1
                        geom.density = 0.0
                        visual_geoms += 1
                    continue

                if web_demo_contacts or (geom_contype == 1 and geom_conaffinity == 1):
                    geom.contype = object_contype
                    geom.conaffinity = object_conaffinity
                    if web_demo_contacts:
                        collision_geom_index += 1
                        if not geom.name:
                            geom.name = (
                                "collision" if collision_geom_index == 1 else f"collision_{collision_geom_index}"
                            )
                        geom.condim = 6
                        geom.solref = [0.01, 1.0]
                    updated_geoms += 1
        logger.info(
            "Applied MuJoCo object collision settings to {} collision geom(s), {} visual geom(s); web_demo_contacts={}",
            updated_geoms,
            visual_geoms,
            web_demo_contacts,
        )

    def _maybe_override_object_properties(
        self,
        target_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        terrain_geom_name: str,
        target_body_names: set[str],
    ) -> None:
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None:
            return

        mass_scale = getattr(object_cfg, "mujoco_object_mass_scale", None)
        mass_override = getattr(object_cfg, "mujoco_object_mass_override", None)
        geom_friction = getattr(object_cfg, "mujoco_object_geom_friction", None)
        terrain_pair_friction = getattr(object_cfg, "mujoco_object_terrain_pair_friction", None)
        if mass_scale is None and mass_override is None and geom_friction is None and terrain_pair_friction is None:
            return
        if not target_body_names:
            return

        friction_override: list[float] | None = None
        if geom_friction is not None:
            friction_override = [float(value) for value in geom_friction]
            if len(friction_override) != 3:
                raise ValueError(
                    "robot.object.mujoco_object_geom_friction must provide exactly 3 values "
                    f"[slide, spin, roll], got {friction_override}"
                )
        terrain_pair_friction_override: list[float] | None = None
        if terrain_pair_friction is not None:
            terrain_pair_friction_override = [float(value) for value in terrain_pair_friction]
            if len(terrain_pair_friction_override) != 3:
                raise ValueError(
                    "robot.object.mujoco_object_terrain_pair_friction must provide exactly 3 values "
                    f"[slide, spin, roll], got {terrain_pair_friction_override}"
                )

        updated_bodies = 0
        updated_geoms = 0
        updated_pairs = 0
        existing_pair_names = {pair.name for pair in target_spec.pairs if pair.name}
        for body in target_spec.bodies:
            if not body.name or body.name not in target_body_names:
                continue

            body_ratio: float | None = None
            original_mass = float(body.mass)
            if mass_override is not None:
                target_mass = float(mass_override)
                if target_mass <= 0.0:
                    raise ValueError(f"robot.object.mujoco_object_mass_override must be > 0, got {target_mass}")
                if original_mass <= 0.0:
                    logger.warning(
                        "Skipping MuJoCo object mass override for non-positive-mass body '{}': mass={}",
                        body.name,
                        original_mass,
                    )
                else:
                    body_ratio = target_mass / original_mass
                    body.mass = target_mass
            elif mass_scale is not None:
                scale = float(mass_scale)
                if scale <= 0.0:
                    raise ValueError(f"robot.object.mujoco_object_mass_scale must be > 0, got {scale}")
                body_ratio = scale
                body.mass = original_mass * scale

            if body_ratio is not None:
                inertia = np.asarray(body.inertia, dtype=np.float64)
                if inertia.size == 3 and np.all(np.isfinite(inertia)):
                    body.inertia = (inertia * body_ratio).tolist()
                updated_bodies += 1

            if friction_override is not None:
                for geom in body.geoms:
                    if int(geom.contype) == 0 or int(geom.conaffinity) == 0:
                        continue
                    geom.friction = friction_override
                    updated_geoms += 1
            if terrain_pair_friction_override is not None:
                for geom in body.geoms:
                    if int(geom.contype) == 0 or int(geom.conaffinity) == 0:
                        continue
                    if not geom.name:
                        continue
                    pair_name = f"{geom.name}__{terrain_geom_name}__sim2sim"
                    if pair_name in existing_pair_names:
                        continue
                    target_spec.add_pair(
                        name=pair_name,
                        geomname1=str(geom.name),
                        geomname2=str(terrain_geom_name),
                        condim=3,
                        friction=self._expand_pair_friction(terrain_pair_friction_override),
                    )
                    existing_pair_names.add(pair_name)
                    updated_pairs += 1

        if updated_bodies > 0:
            logger.info(
                "Overrode MuJoCo object mass/inertia for {} body(s): mass_scale={}, mass_override={}",
                updated_bodies,
                mass_scale,
                mass_override,
            )
        if updated_geoms > 0:
            logger.info("Overrode MuJoCo object geom friction for {} geom(s): {}", updated_geoms, friction_override)
        if updated_pairs > 0:
            logger.info(
                "Added {} MuJoCo object-terrain pair override(s) against terrain geom '{}': {}",
                updated_pairs,
                terrain_geom_name,
                terrain_pair_friction_override,
            )

    def _maybe_add_web_demo_object_contact_pairs(self) -> None:
        """Match the web demo's rubber-hand/object contact overrides when requested."""
        if not self._env_flag(_WEB_DEMO_OBJECT_CONTACTS_ENV):
            return

        def _iter_bodies(container: Any):
            for body in getattr(container, "bodies", []):
                yield body
                yield from _iter_bodies(body)

        existing_geom_names = {
            str(geom.name)
            for body in _iter_bodies(self.world_spec.worldbody)
            for geom in body.geoms
            if geom.name
        }

        def _unique_geom_name(base_name: str) -> str:
            candidate = base_name
            suffix = 2
            while candidate in existing_geom_names:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            existing_geom_names.add(candidate)
            return candidate

        rubber_hand_geoms: list[str] = []
        object_collision_geoms: list[str] = []
        for body in _iter_bodies(self.world_spec.worldbody):
            body_name = str(body.name or "")
            for geom in body.geoms:
                if int(geom.contype) == 0 or int(geom.conaffinity) == 0:
                    continue

                geom_name = str(geom.name or "")
                mesh_name = str(getattr(geom, "meshname", "") or "")
                combined_name = f"{body_name} {geom_name} {mesh_name}".lower()
                if "rubber_hand" in combined_name:
                    if not geom_name:
                        if "left" in combined_name:
                            geom_name = _unique_geom_name("left_rubber_hand_collision")
                        elif "right" in combined_name:
                            geom_name = _unique_geom_name("right_rubber_hand_collision")
                        else:
                            geom_name = _unique_geom_name("rubber_hand_collision")
                        geom.name = geom_name
                    geom.contype = 1
                    geom.conaffinity = 6
                    geom.condim = 6
                    geom.friction = [0.8, 0.02, 0.005]
                    geom.solref = [0.01, 1.0]
                    rubber_hand_geoms.append(geom_name)
                if body_name.startswith("object_") or geom_name.startswith("object_"):
                    object_collision_geoms.append(geom_name)

        if not rubber_hand_geoms or not object_collision_geoms:
            logger.warning(
                "Could not add web-demo object contact pairs: rubber_hand_geoms={}, object_collision_geoms={}",
                sorted(rubber_hand_geoms),
                sorted(object_collision_geoms),
            )
            return

        existing_pair_names = {str(pair.name) for pair in self.world_spec.pairs if pair.name}
        existing_pair_keys = {
            (str(pair.geomname1), str(pair.geomname2))
            for pair in self.world_spec.pairs
            if pair.geomname1 and pair.geomname2
        }
        added_pairs = 0
        for hand_geom_name in sorted(set(rubber_hand_geoms)):
            for object_geom_name in sorted(set(object_collision_geoms)):
                pair_name = f"{hand_geom_name}_{object_geom_name}"
                if pair_name in existing_pair_names:
                    continue
                if (hand_geom_name, object_geom_name) in existing_pair_keys:
                    continue
                if (object_geom_name, hand_geom_name) in existing_pair_keys:
                    continue
                self.world_spec.add_pair(
                    name=pair_name,
                    geomname1=hand_geom_name,
                    geomname2=object_geom_name,
                    condim=6,
                    friction=[0.8, 0.8, 0.02, 0.005, 0.005],
                    solref=[0.01, 1.0],
                )
                existing_pair_names.add(pair_name)
                existing_pair_keys.add((hand_geom_name, object_geom_name))
                added_pairs += 1

        logger.info(
            "Added {} web-demo rubber-hand/object contact pair(s): hands={}, objects={}",
            added_pairs,
            sorted(set(rubber_hand_geoms)),
            sorted(set(object_collision_geoms)),
        )

    def _maybe_override_composite_object_properties(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
        *,
        terrain_geom_name: str,
        using_composite_object_scene: bool,
    ) -> None:
        """Override composite object mass/inertia/contact properties for MuJoCo-only sim2sim."""
        if not using_composite_object_scene:
            return
        self._maybe_override_object_properties(
            robot_spec,
            robot_config,
            terrain_geom_name=terrain_geom_name,
            target_body_names=set(getattr(self, "_object_body_name_by_name", {}).values()),
        )

    @staticmethod
    def _expand_pair_friction(friction_triplet: list[float]) -> list[float]:
        """Expand [slide, spin, roll] into MuJoCo pair.friction's 5D contact basis."""
        slide, spin, roll = [float(value) for value in friction_triplet]
        return [slide, slide, spin, roll, roll]

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
        self._configure_robot_collisions(
            robot_spec,
            robot_config.asset.enable_self_collisions,
            object_body_names=set(self._object_body_name_by_name.values()),
            object_contact_body_names=self._get_object_contact_enabled_body_names(robot_spec, robot_config),
        )

    def _get_object_contact_enabled_body_names(
        self,
        robot_spec: mujoco.MjSpec,
        robot_config: RobotConfig,
    ) -> set[str] | None:
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None:
            return None

        configured_markers = getattr(object_cfg, "mujoco_object_contact_body_name_markers", None)
        if configured_markers is not None:
            carry_body_markers = tuple(
                str(marker).strip().lower() for marker in configured_markers if str(marker).strip()
            )
            if not carry_body_markers:
                return None
        elif getattr(object_cfg, "mujoco_limit_object_contacts_to_carry_bodies", False):
            carry_body_markers = (
                "waist",
                "torso",
                "shoulder",
                "elbow",
                "wrist",
                "hand",
            )
        else:
            return None

        allowed_body_names = {
            str(body.name)
            for body in robot_spec.bodies
            if body.name and any(marker in str(body.name).lower() for marker in carry_body_markers)
        }
        logger.info(
            "Limiting MuJoCo object contacts to {} carry body(ies) using markers {}: {}",
            len(allowed_body_names),
            list(carry_body_markers),
            sorted(allowed_body_names),
        )
        return allowed_body_names

    def _configure_robot_collisions(
        self,
        robot_spec: mujoco.MjSpec,
        enable_self_collisions: bool,
        object_body_names: set[str] | None = None,
        object_contact_body_names: set[str] | None = None,
    ) -> None:
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
        - Object parts: contype=4
        - Robot conaffinity must include object class 4 when object bodies are present,
          otherwise robot-object contacts are silently disabled in one direction.
        """
        has_object_bodies = bool(object_body_names)
        carry_robot_contype = 1
        noncarry_robot_contype = 8 if object_contact_body_names else carry_robot_contype
        robot_conaffinity = 2  # Environment
        if enable_self_collisions:
            robot_conaffinity |= carry_robot_contype
            if object_contact_body_names:
                robot_conaffinity |= noncarry_robot_contype
        robot_conaffinity_with_object = robot_conaffinity
        if has_object_bodies:
            robot_conaffinity_with_object |= 4  # Object

        object_body_names = object_body_names or set()
        object_contact_body_names = object_contact_body_names or set()
        object_contype = 4
        object_conaffinity = 3  # Collide with robot (1) and terrain (2)

        bodies_processed = 0
        geoms_processed = 0

        # Apply collision settings to all robot bodies
        for body in robot_spec.bodies:
            if not body.name:
                # Skip unnamed bodies
                continue

            bodies_processed += 1
            is_object_body = body.name in object_body_names
            for geom in body.geoms:
                # Skip geoms that have been explicitly configured away from defaults
                # Visual meshes typically have contype=0, conaffinity=0
                if geom.contype == 0 or geom.conaffinity == 0:
                    continue  # Skip visual/disabled collision geoms

                # Apply collision settings to geoms using default collision behavior
                # (contype=1, conaffinity=1 are MuJoCo defaults)
                if geom.contype == 1 and geom.conaffinity == 1:
                    if is_object_body:
                        geom.contype = object_contype
                        geom.conaffinity = object_conaffinity
                    else:
                        geom.contype = carry_robot_contype
                        if object_contact_body_names:
                            if body.name in object_contact_body_names:
                                geom.contype = carry_robot_contype
                                geom.conaffinity = robot_conaffinity_with_object
                            else:
                                geom.contype = noncarry_robot_contype
                                geom.conaffinity = robot_conaffinity
                        else:
                            geom.conaffinity = robot_conaffinity_with_object
                    geoms_processed += 1
                    logger.debug(
                        "Set {} geom collision to contype={}, conaffinity={}",
                        body.name,
                        geom.contype,
                        geom.conaffinity,
                    )

        logger.info(f"Applied collision settings to {geoms_processed} geoms across {bodies_processed} bodies")

    @staticmethod
    def _resolve_single_object_urdf(object_path_spec: str) -> Path:
        resolved = Path(resolve_data_file_path(object_path_spec)).expanduser().resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".urdf":
            return resolved

        if resolved.is_dir():
            urdfs = sorted(list(resolved.rglob("*.urdf")) + list(resolved.rglob("*.URDF")))
            if len(urdfs) == 1:
                return urdfs[0].resolve()
            raise ValueError(
                f"MuJoCo object verification expects exactly one object URDF, found {len(urdfs)} in '{resolved}'."
            )

        if resolved.is_file() and resolved.suffix.lower() == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
                payload = payload["clips"]
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid clip-object map json: {resolved}")

            urdf_paths: list[Path] = []
            for entry in payload.values():
                raw_path = ""
                if isinstance(entry, str):
                    raw_path = entry.strip()
                elif isinstance(entry, dict):
                    raw_path = str(entry.get("object_urdf_path", "")).strip()
                if not raw_path:
                    continue
                raw_candidate = Path(raw_path)
                if not raw_candidate.is_absolute() and not raw_path.startswith("holosoma/data"):
                    candidate = (resolved.parent / raw_candidate).resolve()
                else:
                    candidate = Path(resolve_data_file_path(raw_path)).resolve()
                urdf_paths.append(candidate.resolve())

            unique_urdfs = sorted({path for path in urdf_paths if path.suffix.lower() == ".urdf"})
            if len(unique_urdfs) == 1:
                return unique_urdfs[0]
            raise ValueError(
                "MuJoCo object verification expects a single resolved object URDF when using a clip map. "
                f"Found {len(unique_urdfs)} unique URDFs in '{resolved}'."
            )

        raise ValueError(f"Unsupported object specification for MuJoCo object verification: {resolved}")

    @staticmethod
    def _repo_root() -> Path:
        return Path(get_holosoma_root()).resolve().parents[2]

    @classmethod
    def _resolve_supported_object_scene(
        cls,
        robot_config: RobotConfig,
    ) -> tuple[str, dict[str, str], dict[str, str]] | None:
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "enabled", False):
            return None

        object_path_spec = getattr(object_cfg, "object_urdf_path", None)
        if not object_path_spec:
            return None

        object_urdf = cls._resolve_single_object_urdf(str(object_path_spec))
        supported_specs = _SUPPORTED_OBJECT_SCENE_SPECS.get(robot_config.asset.robot_type, {})
        if not supported_specs:
            raise ValueError(
                f"MuJoCo object verification is not configured for robot '{robot_config.asset.robot_type}'."
            )

        object_candidates = (
            object_urdf.stem.lower(),
            object_urdf.parent.name.lower(),
            object_urdf.name.lower(),
        )
        for candidate in object_candidates:
            for key, (xml_rel_path, object_body_name) in supported_specs.items():
                if candidate == key or candidate.endswith(f"_{key}") or key in candidate:
                    xml_path = (cls._repo_root() / xml_rel_path).resolve()
                    if not xml_path.is_file():
                        raise FileNotFoundError(f"Resolved MuJoCo composite scene not found: {xml_path}")
                    return (
                        str(xml_path),
                        {"object": str(object_urdf)},
                        {"object": object_body_name},
                    )

        supported_keys = ", ".join(sorted(supported_specs.keys()))
        raise ValueError(
            f"Unsupported object URDF '{object_urdf}' for robot '{robot_config.asset.robot_type}'. "
            f"Supported MuJoCo object keys: {supported_keys}"
        )

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
