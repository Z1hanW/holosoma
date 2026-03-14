"""MuJoCo scene manager."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

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

        logger.info(f"Adding robot from: {robot_xml_path} with prefix: {prefix}")
        self.robot_model_path = robot_xml_path
        robot_spec = mujoco.MjSpec.from_file(robot_xml_path)

        if xml_filter and getattr(xml_filter, "enable", False):
            # Remove worldbody lights and ground|floor|plane geoms because they're added dynamically
            robot_spec = self._filter_robot_worldbody(robot_spec, xml_filter)

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

        # Store prefix for later use by simulator
        self.robot_prefix = prefix

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
        if not using_composite_object_scene:
            logger.info("Skipping joint-default copy because current MuJoCo scene is not a composite object scene")
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
        for joint in robot_spec.joints:
            if not joint.name or joint.name not in dof_name_set:
                continue

            reference_joint = reference_joints.get(joint.name)
            if reference_joint is None:
                missing_reference_joints.append(joint.name)
                continue

            changed_fields = 0
            if not np.isclose(float(joint.armature), float(reference_joint.armature)):
                joint.armature = float(reference_joint.armature)
                changed_fields += 1
            if not np.isclose(float(joint.damping), float(reference_joint.damping)):
                joint.damping = float(reference_joint.damping)
                changed_fields += 1
            if not np.isclose(float(joint.frictionloss), float(reference_joint.frictionloss)):
                joint.frictionloss = float(reference_joint.frictionloss)
                changed_fields += 1

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
            "Copied MuJoCo joint defaults from '{}' into composite scene for {} joint(s) across {} field update(s)",
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
        if not using_composite_object_scene:
            logger.info("Skipping tendon copy because current MuJoCo scene is not a composite object scene")
            return
        if len(robot_spec.tendons) > 0:
            logger.info("Skipping tendon copy because composite scene already defines {} tendons", len(robot_spec.tendons))
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

        for tendon_idx, reference_tendon in enumerate(reference_spec.tendons):
            tendon = robot_spec.add_tendon(
                name=reference_tendon.name or None,
                stiffness=float(reference_tendon.stiffness),
                springlength=_seq_or_none(reference_tendon.springlength),
                damping=float(reference_tendon.damping),
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
            "Copied {} MuJoCo tendon(s) with {} wrap(s) from '{}' into composite scene",
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
        """Copy standalone MuJoCo collision geoms into composite object scenes when explicitly requested."""

        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not getattr(object_cfg, "mujoco_copy_collision_geoms_from_robot_xml", False):
            return
        if not using_composite_object_scene:
            logger.info("Skipping collision-geom copy because current MuJoCo scene is not a composite object scene")
            return

        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        reference_xml_path = os.path.join(asset_root, robot_config.asset.xml_file)
        reference_spec = mujoco.MjSpec.from_file(reference_xml_path)

        target_bodies = {body.name: body for body in robot_spec.bodies if body.name}
        existing_geom_names = {geom.name for body in robot_spec.bodies for geom in body.geoms if geom.name}

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
            "Copied {} MuJoCo collision geom(s) from '{}' into composite scene",
            copied_geom_count,
            reference_xml_path,
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
        if not using_composite_object_scene:
            logger.info("Skipping contact-pair copy because current MuJoCo scene is not a composite object scene")
            return
        if len(robot_spec.pairs) > 0:
            logger.info("Skipping contact-pair copy because composite scene already defines {} pair(s)", len(robot_spec.pairs))
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
            "Copied {} MuJoCo contact pair(s) from '{}' into composite scene (terrain geom '{}')",
            copied_pair_count,
            reference_xml_path,
            terrain_geom_name,
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
        object_cfg = getattr(robot_config, "object", None)
        if object_cfg is None or not using_composite_object_scene:
            return

        mass_scale = getattr(object_cfg, "mujoco_object_mass_scale", None)
        mass_override = getattr(object_cfg, "mujoco_object_mass_override", None)
        geom_friction = getattr(object_cfg, "mujoco_object_geom_friction", None)
        terrain_pair_friction = getattr(object_cfg, "mujoco_object_terrain_pair_friction", None)
        if mass_scale is None and mass_override is None and geom_friction is None and terrain_pair_friction is None:
            return

        target_body_names = set(getattr(self, "_object_body_name_by_name", {}).values())
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
        existing_pair_names = {pair.name for pair in robot_spec.pairs if pair.name}
        for body in robot_spec.bodies:
            if not body.name or body.name not in target_body_names:
                continue

            body_ratio: float | None = None
            original_mass = float(body.mass)
            if mass_override is not None:
                target_mass = float(mass_override)
                if target_mass <= 0.0:
                    raise ValueError(f"robot.object.mujoco_object_mass_override must be > 0, got {target_mass}")
                if original_mass <= 0.0:
                    raise ValueError(
                        "Cannot override MuJoCo object mass because the composite scene body has non-positive mass: "
                        f"{body.name} mass={original_mass}"
                    )
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
                    geom.friction = friction_override
                    updated_geoms += 1
            if terrain_pair_friction_override is not None:
                for geom in body.geoms:
                    if not geom.name:
                        continue
                    pair_name = f"{geom.name}__{terrain_geom_name}__sim2sim"
                    if pair_name in existing_pair_names:
                        continue
                    robot_spec.add_pair(
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
                "Overrode MuJoCo composite object mass/inertia for {} body(s): mass_scale={}, mass_override={}",
                updated_bodies,
                mass_scale,
                mass_override,
            )
        if updated_geoms > 0:
            logger.info(
                "Overrode MuJoCo composite object geom friction for {} geom(s): {}",
                updated_geoms,
                friction_override,
            )
        if updated_pairs > 0:
            logger.info(
                "Added {} MuJoCo object-terrain pair override(s) against terrain geom '{}': {}",
                updated_pairs,
                terrain_geom_name,
                terrain_pair_friction_override,
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
        )

    def _configure_robot_collisions(
        self,
        robot_spec: mujoco.MjSpec,
        enable_self_collisions: bool,
        object_body_names: set[str] | None = None,
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
        robot_conaffinity = 2  # Environment
        if enable_self_collisions:
            robot_conaffinity |= 1  # Robot
        if has_object_bodies:
            robot_conaffinity |= 4  # Object

        object_body_names = object_body_names or set()
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
                        geom.contype = 1  # Robot collision class
                        geom.conaffinity = robot_conaffinity  # Configurable based on self_collisions
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
