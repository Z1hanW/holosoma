from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.config_types.retargeter import FootLockConfig, SelfCollisionConfig

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _mesh_local_vf,
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)
from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found,no-redef]  # noqa: E402


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    HARD_CONTACT_MIN_WEIGHT_SCALE = 0.999

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        terrain_collision_geom_prefix: str = "",
        terrain_collision_foot_only: bool = True,
        terrain_support_mesh_file: str = "",
        terrain_support_mesh_scale: float = 1.0,
        terrain_support_min_normal_z: float = 0.9,
        terrain_support_clearance: float = 2e-4,
        terrain_support_sphere_radius: float = 0.005,
        terrain_support_activation_margin: float = 0.1,
        terrain_support_max_sqp_iterations: int = 50,
        terrain_support_feasibility_tolerance: float = 5e-5,
        activate_joint_limits: bool = True,
        ground_initial_robot: bool = False,
        initial_ground_clearance: float = 0.0,
        foot_sticking_pin_z: bool = False,
        foot_sticking_z_floor: float = 0.005,
        foot_grounding_weight: float = 0.0,
        foot_grounding_mode: str = "sticking",
        foot_grounding_schedule: str = "all_frames",
        foot_grounding_ramp_frames: int = 0,
        step_size: float = 0.2,
        max_frame_root_translation: float = 0.0,
        max_frame_root_quaternion_delta: float = 0.0,
        max_frame_joint_delta: float = 0.0,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        enforce_exact_nonpenetration: bool = False,
        exact_nonpenetration_max_sqp_iterations: int = 50,
        exact_nonpenetration_feasibility_tolerance: float = 1e-6,
        exact_nonpenetration_interior_margin: float = 0.0,
        exact_nonpenetration_qp_safety_margin: float = 0.0,
        exact_nonpenetration_restore_infeasible_start: bool = False,
        exact_nonpenetration_restoration_max_iterations: int = 20,
        exact_nonpenetration_backtracking_steps: int = 24,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        w_keypoint_tracking: float = 0.0,
        activate_hand_contact: bool = False,
        hand_contact_mode: str = "soft",
        hand_contact_weight: float = 0.0,
        hand_contact_tolerance: float = 0.0,
        hand_contact_point_offset: tuple[float, float, float] = (0.09, 0.0, 0.0),
        hand_contact_point_mode: str = "fixed_offset",
        replace_source_wrist_with_contact: bool = False,
        save_partial_on_failure: bool = False,
        partial_checkpoint_interval_frames: int = 0,
        resume_partial_file: str = "",
        initial_qpos_file: str = "",
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            enforce_exact_nonpenetration: require the nonlinear MuJoCo distances after each SQP update
                to satisfy the configured penetration tolerance before accepting a frame.
            exact_nonpenetration_max_sqp_iterations: maximum per-frame SQP iterations while the
                nonlinear non-penetration check remains infeasible.
            exact_nonpenetration_feasibility_tolerance: numerical residual allowed beyond the
                configured penetration tolerance.
            exact_nonpenetration_interior_margin: stricter exact-feasibility margin inside
                the linearized penetration limit, used to leave room for the next SQP frame.
            exact_nonpenetration_qp_safety_margin: additional margin applied only to the
                linearized QP collision target to absorb local linearization error.
            exact_nonpenetration_restore_infeasible_start: before the main SQP objective,
                run a collision-only projection if current object motion makes the prior
                robot qpos fail the nonlinear collision gate.
            exact_nonpenetration_restoration_max_iterations: maximum collision-only SQP
                projections used to find a feasible current-frame initialization.
            exact_nonpenetration_backtracking_steps: bisection steps used to retain the largest
                nonlinear-feasible fraction of an SQP update.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            hand_contact_point_mode: fixed_offset uses one configured body-local point;
                nearest_collision_surface selects the closest point on the actual hand
                collision mesh for each active target and SQP linearization point.
            ground_initial_robot: lower the initial floating base until a foot collision sphere reaches the floor.
            initial_ground_clearance: target minimum foot-to-floor clearance after initial grounding.
            foot_sticking_pin_z: add a floor-height Z constraint when a foot is sticking.
            foot_sticking_z_floor: target Z of the mapped toe-sphere center for sticking feet.
            foot_grounding_weight: soft floor-height objective weight for sticking feet.
            foot_grounding_mode: choose detected sticking feet, continuous support,
                the lower robot foot, or only the lowest sphere on each support foot.
            foot_grounding_schedule: apply the objective on all frames or only before contact.
            foot_grounding_ramp_frames: linearly decay the pre-contact objective to zero at t1.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.ground_initial_robot = bool(ground_initial_robot)
        self.initial_ground_clearance = float(initial_ground_clearance)
        self.foot_sticking_pin_z = bool(foot_sticking_pin_z)
        self.foot_sticking_z_floor = float(foot_sticking_z_floor)
        self.foot_grounding_weight = float(foot_grounding_weight)
        self.foot_grounding_mode = str(foot_grounding_mode)
        self.foot_grounding_schedule = str(foot_grounding_schedule)
        self.foot_grounding_ramp_frames = int(foot_grounding_ramp_frames)
        self._foot_grounding_contact_start_idx = -1
        if not np.isfinite(self.initial_ground_clearance) or self.initial_ground_clearance < 0.0:
            raise ValueError("initial_ground_clearance must be finite and non-negative")
        if not np.isfinite(self.foot_sticking_z_floor):
            raise ValueError("foot_sticking_z_floor must be finite")
        if not np.isfinite(self.foot_grounding_weight) or self.foot_grounding_weight < 0.0:
            raise ValueError("foot_grounding_weight must be finite and non-negative")
        if self.foot_grounding_mode not in {
            "sticking",
            "continuous_support",
            "continuous_lowest_sphere",
            "lowest_foot",
        }:
            raise ValueError(
                "foot_grounding_mode must be 'sticking', 'continuous_support', "
                "'continuous_lowest_sphere', or 'lowest_foot'"
            )
        if self.foot_grounding_schedule not in {"all_frames", "before_contact"}:
            raise ValueError(
                "foot_grounding_schedule must be 'all_frames' or 'before_contact'"
            )
        if self.foot_grounding_ramp_frames < 0:
            raise ValueError("foot_grounding_ramp_frames must be non-negative")
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.terrain_collision_geom_prefix = terrain_collision_geom_prefix.strip()
        self.terrain_collision_foot_only = bool(terrain_collision_foot_only)
        self.terrain_support_mesh_file = terrain_support_mesh_file.strip()
        self.terrain_support_mesh_scale = float(terrain_support_mesh_scale)
        self.terrain_support_min_normal_z = float(terrain_support_min_normal_z)
        self.terrain_support_clearance = float(terrain_support_clearance)
        self.terrain_support_sphere_radius = float(terrain_support_sphere_radius)
        self.terrain_support_activation_margin = float(terrain_support_activation_margin)
        self.terrain_support_max_sqp_iterations = int(terrain_support_max_sqp_iterations)
        self.terrain_support_feasibility_tolerance = float(
            terrain_support_feasibility_tolerance
        )
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = float(penetration_tolerance)
        self.enforce_exact_nonpenetration = bool(enforce_exact_nonpenetration)
        self.exact_nonpenetration_max_sqp_iterations = int(
            exact_nonpenetration_max_sqp_iterations
        )
        self.exact_nonpenetration_feasibility_tolerance = float(
            exact_nonpenetration_feasibility_tolerance
        )
        self.exact_nonpenetration_interior_margin = float(
            exact_nonpenetration_interior_margin
        )
        self.exact_nonpenetration_qp_safety_margin = float(
            exact_nonpenetration_qp_safety_margin
        )
        self.exact_nonpenetration_restore_infeasible_start = bool(
            exact_nonpenetration_restore_infeasible_start
        )
        self.exact_nonpenetration_restoration_max_iterations = int(
            exact_nonpenetration_restoration_max_iterations
        )
        self.exact_nonpenetration_backtracking_steps = int(
            exact_nonpenetration_backtracking_steps
        )
        if (
            not np.isfinite(self.penetration_tolerance)
            or self.penetration_tolerance < 0.0
            or self.exact_nonpenetration_max_sqp_iterations <= 0
            or not np.isfinite(self.exact_nonpenetration_feasibility_tolerance)
            or self.exact_nonpenetration_feasibility_tolerance < 0.0
            or not np.isfinite(self.exact_nonpenetration_interior_margin)
            or self.exact_nonpenetration_interior_margin < 0.0
            or self.exact_nonpenetration_interior_margin > self.penetration_tolerance
            or not np.isfinite(self.exact_nonpenetration_qp_safety_margin)
            or self.exact_nonpenetration_qp_safety_margin < 0.0
            or self.exact_nonpenetration_interior_margin
            + self.exact_nonpenetration_qp_safety_margin
            > self.penetration_tolerance
            or self.exact_nonpenetration_restoration_max_iterations <= 0
            or self.exact_nonpenetration_backtracking_steps <= 0
        ):
            raise ValueError(
                "Invalid penetration tolerance, exact non-penetration iteration count, "
                "exact feasibility tolerance, exact interior/QP margin, or restoration count"
            )
        self.step_size = step_size
        self.max_frame_root_translation = float(max_frame_root_translation)
        self.max_frame_root_quaternion_delta = float(
            max_frame_root_quaternion_delta
        )
        self.max_frame_joint_delta = float(max_frame_joint_delta)
        if min(
            self.max_frame_root_translation,
            self.max_frame_root_quaternion_delta,
            self.max_frame_joint_delta,
        ) < 0.0:
            raise ValueError("Per-frame retarget limits must be non-negative")
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = dict(task_constants.JOINTS_MAPPING)
        self.task_constants = task_constants
        self.activate_hand_contact = activate_hand_contact
        self.hand_contact_mode = hand_contact_mode.lower().strip()
        if self.hand_contact_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported hand_contact_mode: {hand_contact_mode!r}")
        self.hand_contact_weight = float(hand_contact_weight)
        self.hand_contact_tolerance = float(hand_contact_tolerance)
        self.hand_contact_point_offset = np.asarray(hand_contact_point_offset, dtype=float)
        self.hand_contact_point_mode = str(hand_contact_point_mode).strip().lower()
        if self.hand_contact_point_mode not in {
            "fixed_offset",
            "nearest_collision_surface",
        }:
            raise ValueError(
                "hand_contact_point_mode must be 'fixed_offset' or "
                "'nearest_collision_surface'"
            )
        self.replace_source_wrist_with_contact = bool(replace_source_wrist_with_contact)
        self.save_partial_on_failure = bool(save_partial_on_failure)
        self.partial_checkpoint_interval_frames = int(partial_checkpoint_interval_frames)
        if self.partial_checkpoint_interval_frames < 0:
            raise ValueError("partial_checkpoint_interval_frames must be non-negative")
        self.resume_partial_file = resume_partial_file.strip()
        self.initial_qpos_file = initial_qpos_file.strip()
        if self.resume_partial_file and self.initial_qpos_file:
            raise ValueError("resume_partial_file and initial_qpos_file are mutually exclusive")

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.2
        self.w_keypoint_tracking = w_keypoint_tracking
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        explicit_scene_xml = getattr(self.task_constants, "SCENE_XML_FILE", "")
        if explicit_scene_xml:
            robot_xml_path = explicit_scene_xml
        elif self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        self.laplacian_match_links = self._resolve_mapped_hand_links(self.laplacian_match_links)
        self.hand_contact_links = self._resolve_hand_contact_links()
        self._init_hand_contact_surface_meshes()
        self._init_terrain_support()
        self._init_self_collision(self._self_collision_config)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False

        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        self._initial_grounding_offset_m = 0.0
        self._initial_grounding_clearance_before_m = np.nan
        self._initial_grounding_clearance_after_m = np.nan

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            for window in windows:
                if len(window) != 2:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                start, end = int(window[0]), int(window[1])
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
            self._foot_lock_windows[side] = tuple(normalized_windows)

    def _ground_initial_robot_configuration(self, q: np.ndarray) -> np.ndarray:
        """Translate the floating base so the lowest foot sphere reaches z=0."""
        grounded = np.asarray(q, dtype=np.float64).copy()
        if grounded.shape != (self.nq,) or not np.isfinite(grounded).all():
            raise ValueError(f"Invalid initial qpos for grounding: {grounded.shape}")
        if 2 not in self.q_a_indices:
            raise ValueError("Initial robot grounding requires root Z in the optimized qpos slice")

        ground_id = int(
            mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "ground",
            )
        )
        if ground_id < 0 or self.robot_model.geom_type[ground_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Initial robot grounding requires a plane geom named 'ground'")

        foot_geom_ids = []
        for geom_id in range(self.robot_model.ngeom):
            geom_name = (
                mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or ""
            )
            if "ankle_roll_sphere_" not in geom_name:
                continue
            if (
                self.robot_model.geom_contype[geom_id] == 0
                and self.robot_model.geom_conaffinity[geom_id] == 0
            ):
                continue
            foot_geom_ids.append(geom_id)
        if len(foot_geom_ids) != 10:
            names = [
                mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                for geom_id in foot_geom_ids
            ]
            raise ValueError(f"Expected 10 collision foot spheres for grounding, got {names}")

        def minimum_clearance(candidate: np.ndarray) -> float:
            self.robot_data.qpos[:] = candidate
            mujoco.mj_forward(self.robot_model, self.robot_data)
            distances = []
            fromto = np.zeros(6, dtype=np.float64)
            for geom_id in foot_geom_ids:
                fromto[:] = 0.0
                distances.append(
                    float(
                        mujoco.mj_geomDistance(
                            self.robot_model,
                            self.robot_data,
                            geom_id,
                            ground_id,
                            10.0,
                            fromto,
                        )
                    )
                )
            return float(min(distances))

        clearance_before = minimum_clearance(grounded)
        offset = float(self.initial_ground_clearance - clearance_before)
        grounded[2] += offset
        clearance_after = minimum_clearance(grounded)
        if abs(clearance_after - self.initial_ground_clearance) > 1.0e-6:
            raise RuntimeError(
                "Initial foot grounding did not reach the requested clearance: "
                f"target={self.initial_ground_clearance:.9g}, actual={clearance_after:.9g}"
            )

        self._initial_grounding_offset_m = offset
        self._initial_grounding_clearance_before_m = clearance_before
        self._initial_grounding_clearance_after_m = clearance_after
        print(
            "[RobotGrounding] initial foot clearance "
            f"{clearance_before:.6f} -> {clearance_after:.6f} m; root_z_offset={offset:.6f} m"
        )
        return grounded

    def _init_terrain_support(self) -> None:
        """Precompute walkable CRISP top-face planes for upward foot constraints."""
        self._terrain_support_enabled = False
        self._terrain_support_foot_geoms: list[tuple[int, int, str]] = []
        self._terrain_support_geom_component_ids: dict[int, int] = {}
        self._terrain_support_component_bottom_planes: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._terrain_support_latched_components: dict[int, set[int]] = {}
        self._terrain_projection_frames: list[int] = []
        self._terrain_projection_max_joint_delta: list[float] = []
        if not self.terrain_support_mesh_file:
            return
        if (
            not np.isfinite(self.terrain_support_mesh_scale)
            or self.terrain_support_mesh_scale <= 0.0
            or not 0.0 < self.terrain_support_min_normal_z <= 1.0
            or self.terrain_support_clearance < 0.0
            or self.terrain_support_sphere_radius <= 0.0
            or self.terrain_support_activation_margin <= 0.0
            or self.terrain_support_max_sqp_iterations <= 0
            or self.terrain_support_feasibility_tolerance < 0.0
        ):
            raise ValueError(
                "Invalid terrain support scale, normal, clearance, radius, activation margin, "
                "iteration count, or feasibility tolerance"
            )

        mesh_path = Path(self.terrain_support_mesh_file).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Terrain support mesh does not exist: {mesh_path}")
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.size == 0 or loaded.faces.size == 0:
            raise ValueError(f"Terrain support mesh is empty: {mesh_path}")
        components = list(loaded.split(only_watertight=False))
        # The scene builder assigns crisp_terrain_part_NNN after this exact
        # ordering. trimesh's raw split order is not stable across exports.
        components.sort(key=lambda mesh: tuple(np.round(mesh.centroid, decimals=9)))
        invalid = [index for index, component in enumerate(components) if not component.is_watertight]
        if invalid:
            raise ValueError(f"Terrain support components must be watertight; invalid indices: {invalid}")

        triangle_groups = []
        normal_groups = []
        lower_bound_groups = []
        component_id_groups = []
        scaled_component_bounds = []
        for component_id, component in enumerate(components):
            scaled = component.copy()
            scaled.apply_scale(self.terrain_support_mesh_scale)
            scaled.fix_normals()
            scaled_component_bounds.append(np.asarray(scaled.bounds, dtype=np.float64))
            all_normals = np.asarray(scaled.face_normals, dtype=np.float64)
            all_offsets = np.einsum(
                "ni,ni->n", all_normals, np.asarray(scaled.triangles[:, 0], dtype=np.float64)
            )
            downward = all_normals[:, 2] < -1e-8
            if not np.any(downward):
                raise ValueError(f"Terrain component {component_id} has no downward-facing plane")
            self._terrain_support_component_bottom_planes[component_id] = (
                all_normals[downward],
                all_offsets[downward],
            )
            face_mask = scaled.face_normals[:, 2] >= self.terrain_support_min_normal_z
            triangles = np.asarray(scaled.triangles[face_mask], dtype=np.float64)
            if triangles.size == 0:
                continue
            triangle_groups.append(triangles)
            normal_groups.append(np.asarray(scaled.face_normals[face_mask], dtype=np.float64))
            lower_bound_groups.append(np.full(triangles.shape[0], float(scaled.bounds[0, 2])))
            component_id_groups.append(np.full(triangles.shape[0], component_id, dtype=np.int32))
        if not triangle_groups:
            raise ValueError(
                f"Terrain support mesh has no faces with normal_z >= {self.terrain_support_min_normal_z}"
            )

        triangles = np.concatenate(triangle_groups, axis=0)
        normals = np.concatenate(normal_groups, axis=0)
        lower_bounds = np.concatenate(lower_bound_groups, axis=0)
        component_ids = np.concatenate(component_id_groups, axis=0)
        xy_origins = triangles[:, 0, :2]
        xy_edges = np.stack(
            (triangles[:, 1, :2] - xy_origins, triangles[:, 2, :2] - xy_origins), axis=-1
        )
        determinants = np.linalg.det(xy_edges)
        valid_projection = np.abs(determinants) > 1e-12
        if not np.any(valid_projection):
            raise ValueError("Terrain support top faces have degenerate XY projections")
        triangles = triangles[valid_projection]
        normals = normals[valid_projection]
        lower_bounds = lower_bounds[valid_projection]
        component_ids = component_ids[valid_projection]
        xy_origins = xy_origins[valid_projection]
        xy_edges = xy_edges[valid_projection]

        self._terrain_support_triangles = triangles
        self._terrain_support_normals = normals
        self._terrain_support_plane_offsets = np.einsum("ni,ni->n", normals, triangles[:, 0])
        self._terrain_support_component_lower_z = lower_bounds
        self._terrain_support_component_ids = component_ids
        self._terrain_support_xy_origins = xy_origins
        self._terrain_support_xy_inverse_edges = np.linalg.inv(xy_edges)

        for geom_id in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_id = int(self.robot_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "ankle_roll_sphere_" in f"{geom_name} {body_name}":
                self._terrain_support_foot_geoms.append((geom_id, body_id, body_name or geom_name))
        if len(self._terrain_support_foot_geoms) != 10:
            names = [entry[2] for entry in self._terrain_support_foot_geoms]
            raise ValueError(f"Expected 10 G1 foot sphere geoms for terrain support, got {names}")
        if self.terrain_collision_geom_prefix:
            component_pattern = re.compile(
                rf"^{re.escape(self.terrain_collision_geom_prefix)}(\d+)$"
            )
            for geom_id in range(self.robot_model.ngeom):
                geom_name = (
                    mujoco.mj_id2name(
                        self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    or ""
                )
                match = component_pattern.fullmatch(geom_name)
                if match is not None:
                    self._terrain_support_geom_component_ids[geom_id] = int(match.group(1))
            expected_ids = set(range(len(components)))
            actual_ids = set(self._terrain_support_geom_component_ids.values())
            if actual_ids != expected_ids:
                raise ValueError(
                    "Terrain geom/component ids do not match support mesh components: "
                    f"expected={sorted(expected_ids)}, actual={sorted(actual_ids)}"
                )
            mujoco.mj_forward(self.robot_model, self.robot_data)
            for geom_id, component_id in self._terrain_support_geom_component_ids.items():
                if self.robot_model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                    geom_name = mujoco.mj_id2name(
                        self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    raise ValueError(
                        f"Terrain geom {geom_name!r} is not a mesh"
                    )
                mesh_id = int(self.robot_model.geom_dataid[geom_id])
                vertex_start = int(self.robot_model.mesh_vertadr[mesh_id])
                vertex_count = int(self.robot_model.mesh_vertnum[mesh_id])
                vertices = np.asarray(
                    self.robot_model.mesh_vert[
                        vertex_start : vertex_start + vertex_count
                    ],
                    dtype=np.float64,
                )
                rotation = self.robot_data.geom_xmat[geom_id].reshape(3, 3)
                vertices_world = (
                    rotation @ vertices.T
                ).T + self.robot_data.geom_xpos[geom_id]
                scene_bounds = np.stack(
                    (vertices_world.min(axis=0), vertices_world.max(axis=0))
                )
                support_bounds = scaled_component_bounds[component_id]
                bounds_error = float(np.max(np.abs(scene_bounds - support_bounds)))
                if bounds_error > 1.0e-5:
                    geom_name = mujoco.mj_id2name(
                        self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    )
                    raise ValueError(
                        "Terrain scene/support component mismatch for "
                        f"{geom_name}: max bounds error={bounds_error:.6g} m"
                    )
        self._terrain_support_enabled = True
        print(
            f"[TerrainSupport] Loaded {len(components)} components, "
            f"{len(triangles)} walkable triangles, scale={self.terrain_support_mesh_scale:.9g}"
        )

    def _terrain_support_planes(
        self, point_world: np.ndarray, foot_geom_id: int | None = None
    ) -> list[tuple[np.ndarray, float, float, int]]:
        if not self._terrain_support_enabled:
            return []
        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        delta_xy = point[:2] - self._terrain_support_xy_origins
        barycentric = np.einsum("nij,nj->ni", self._terrain_support_xy_inverse_edges, delta_xy)
        inside = (
            (barycentric[:, 0] >= -1e-8)
            & (barycentric[:, 1] >= -1e-8)
            & (barycentric.sum(axis=1) <= 1.0 + 1e-8)
        )
        triangles = self._terrain_support_triangles
        support_z = (
            triangles[:, 0, 2]
            + barycentric[:, 0] * (triangles[:, 1, 2] - triangles[:, 0, 2])
            + barycentric[:, 1] * (triangles[:, 2, 2] - triangles[:, 0, 2])
        )
        relevant = inside & (
            point[2]
            <= support_z + self.terrain_support_sphere_radius + self.terrain_support_activation_margin
        )
        candidates = np.flatnonzero(relevant)
        if candidates.size == 0:
            return []
        latched = (
            self._terrain_support_latched_components.setdefault(foot_geom_id, set())
            if foot_geom_id is not None
            else set()
        )
        supports: list[tuple[np.ndarray, float, float, int]] = []
        for component_id in np.unique(self._terrain_support_component_ids[candidates]):
            component_candidates = candidates[
                self._terrain_support_component_ids[candidates] == component_id
            ]
            index = int(
                component_candidates[np.argmax(support_z[component_candidates])]
            )
            bottom_normals, bottom_offsets = self._terrain_support_component_bottom_planes[
                int(component_id)
            ]
            bottom_z = np.max(
                (
                    bottom_offsets
                    - bottom_normals[:, 0] * point[0]
                    - bottom_normals[:, 1] * point[1]
                )
                / bottom_normals[:, 2]
            )
            normally_relevant = (
                point[2]
                >= max(
                    bottom_z,
                    self._terrain_support_component_lower_z[index],
                )
                - self.terrain_support_sphere_radius
                - 1e-8
            )
            if not normally_relevant and int(component_id) not in latched:
                continue
            if normally_relevant and foot_geom_id is not None:
                latched.add(int(component_id))
            supports.append(
                (
                    self._terrain_support_normals[index],
                    float(self._terrain_support_plane_offsets[index]),
                    float(support_z[index]),
                    int(component_id),
                )
            )
        supports.sort(key=lambda support: support[2], reverse=True)
        return supports

    def _terrain_support_plane(
        self, point_world: np.ndarray, foot_geom_id: int | None = None
    ):
        supports = self._terrain_support_planes(point_world, foot_geom_id)
        return supports[0] if supports else None

    def _begin_terrain_support_frame(self, q: np.ndarray) -> None:
        self._terrain_support_latched_components = {
            geom_id: set() for geom_id, _, _ in self._terrain_support_foot_geoms
        }
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        for geom_id, _, _ in self._terrain_support_foot_geoms:
            self._terrain_support_planes(self.robot_data.geom_xpos[geom_id], geom_id)

    def _build_terrain_support_constraints(self, q: np.ndarray):
        if not self._terrain_support_enabled:
            return []
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        rows = []
        for geom_id, body_id, body_name in self._terrain_support_foot_geoms:
            point_world = np.asarray(self.robot_data.geom_xpos[geom_id], dtype=np.float64)
            jacobian = self._calc_contact_jacobian_from_point(body_id, point_world, input_world=True)
            for normal, plane_offset, support_z, component_id in self._terrain_support_planes(
                point_world, geom_id
            ):
                signed_plane_distance = float(normal @ point_world - plane_offset)
                rows.append(
                    (body_name, normal @ jacobian, signed_plane_distance, support_z, component_id)
                )
        return rows

    def _terrain_feasibility_violations(self, q: np.ndarray) -> tuple[float, float]:
        """Return final nonlinear top-plane and remaining terrain collision residuals."""
        if not self._terrain_support_enabled:
            return 0.0, 0.0
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        top_violation = 0.0
        collision_violation = 0.0
        target_plane_distance = self.terrain_support_sphere_radius + self.terrain_support_clearance
        fromto = np.zeros(6, dtype=float)
        for foot_geom_id, _, _ in self._terrain_support_foot_geoms:
            point_world = np.asarray(self.robot_data.geom_xpos[foot_geom_id], dtype=np.float64)
            supports = self._terrain_support_planes(point_world, foot_geom_id)
            support_component_ids = {support[3] for support in supports}
            for normal, plane_offset, _, _ in supports:
                signed_plane_distance = float(normal @ point_world - plane_offset)
                top_violation = max(
                    top_violation, target_plane_distance - signed_plane_distance
                )
            for terrain_geom_id, component_id in self._terrain_support_geom_component_ids.items():
                if component_id in support_component_ids:
                    continue
                distance = mujoco.mj_geomDistance(
                    self.robot_model,
                    self.robot_data,
                    foot_geom_id,
                    terrain_geom_id,
                    self.collision_detection_threshold,
                    fromto,
                )
                collision_violation = max(
                    collision_violation, -self.penetration_tolerance - float(distance)
                )
        return max(0.0, top_violation), max(0.0, collision_violation)

    def _terrain_feasibility_details(self, q: np.ndarray) -> str:
        """Describe terrain residuals that remain above the configured tolerance."""
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        target_distance = self.terrain_support_sphere_radius + self.terrain_support_clearance
        fromto = np.zeros(6, dtype=float)
        details = []
        for foot_geom_id, _, foot_name in self._terrain_support_foot_geoms:
            point_world = np.asarray(
                self.robot_data.geom_xpos[foot_geom_id], dtype=np.float64
            )
            supports = self._terrain_support_planes(point_world, foot_geom_id)
            support_component_ids = {support[3] for support in supports}
            for normal, plane_offset, support_z, component_id in supports:
                signed_distance = float(normal @ point_world - plane_offset)
                residual = target_distance - signed_distance
                if residual > self.terrain_support_feasibility_tolerance:
                    details.append(
                        f"top foot={foot_name} component={component_id} "
                        f"residual={residual:.6g} support_z={support_z:.6g} "
                        f"point={point_world.tolist()}"
                    )
            for terrain_geom_id, component_id in self._terrain_support_geom_component_ids.items():
                if component_id in support_component_ids:
                    continue
                distance = float(
                    mujoco.mj_geomDistance(
                        self.robot_model,
                        self.robot_data,
                        foot_geom_id,
                        terrain_geom_id,
                        self.collision_detection_threshold,
                        fromto,
                    )
                )
                residual = -self.penetration_tolerance - distance
                if residual > self.terrain_support_feasibility_tolerance:
                    details.append(
                        f"collision foot={foot_name} component={component_id} "
                        f"residual={residual:.6g} distance={distance:.6g}"
                    )
        return "; ".join(details) if details else "no residual details"

    def _terrain_leg_projection_indices(
        self, side: str
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos_indices = []
        for joint_id in range(self.robot_model.njnt):
            joint_name = (
                mujoco.mj_id2name(
                    self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                or ""
            )
            if not joint_name.startswith(f"{side}_"):
                continue
            if not any(token in joint_name for token in ("hip_", "knee_", "ankle_")):
                continue
            if self.robot_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qpos_indices.append(int(self.robot_model.jnt_qposadr[joint_id]))
        if len(qpos_indices) != 6:
            raise ValueError(
                f"Expected six {side} leg hinge joints, got qpos indices {qpos_indices}"
            )
        qpos_indices_array = np.asarray(qpos_indices, dtype=np.int32)
        q_a_positions = []
        for qpos_index in qpos_indices_array:
            matches = np.flatnonzero(self.q_a_indices == qpos_index)
            if len(matches) != 1:
                raise ValueError(
                    f"Terrain leg projection requires qpos index {qpos_index} to be optimized"
                )
            q_a_positions.append(int(matches[0]))
        return qpos_indices_array, np.asarray(q_a_positions, dtype=np.int32)

    def _project_terrain_feasibility(
        self,
        q: np.ndarray,
        *,
        q_t_last: np.ndarray,
        max_iterations: int = 10,
    ) -> tuple[np.ndarray, bool, list[tuple[int, float, float]], float]:
        """Project residual terrain violations using only the affected leg joints."""
        q_projected = np.asarray(q, dtype=np.float64).copy()
        q_start = q_projected.copy()
        history: list[tuple[int, float, float]] = []
        for projection_iteration in range(max_iterations + 1):
            top_violation, collision_violation = self._terrain_feasibility_violations(
                q_projected
            )
            history.append(
                (projection_iteration, top_violation, collision_violation)
            )
            if (
                max(top_violation, collision_violation)
                <= self.terrain_support_feasibility_tolerance
            ):
                max_delta = float(
                    np.max(np.abs(q_projected[self.q_a_indices] - q_start[self.q_a_indices]))
                )
                return q_projected, True, history, max_delta
            if projection_iteration == max_iterations:
                break

            changed = False
            for side in ("left", "right"):
                leg_qpos_indices, leg_q_a_positions = self._terrain_leg_projection_indices(
                    side
                )
                rows = []
                lower_bounds = []
                for body_name, jacobian, signed_distance, _, _ in self._build_terrain_support_constraints(
                    q_projected
                ):
                    if side not in body_name.lower():
                        continue
                    rows.append(jacobian[leg_qpos_indices])
                    lower_bounds.append(
                        self.terrain_support_sphere_radius
                        + self.terrain_support_clearance
                        - signed_distance
                    )

                collision_jacobians, collision_distances = (
                    self._update_jacobians_and_phis_from_q(q_projected)
                )
                for key, distance in collision_distances.items():
                    if len(key) != 2 or not all(isinstance(value, (int, np.integer)) for value in key):
                        continue
                    geom_a, geom_b = int(key[0]), int(key[1])
                    terrain_a = self._geom_names[geom_a].startswith(
                        self.terrain_collision_geom_prefix
                    )
                    terrain_b = self._geom_names[geom_b].startswith(
                        self.terrain_collision_geom_prefix
                    )
                    if terrain_a == terrain_b:
                        continue
                    foot_geom = geom_b if terrain_a else geom_a
                    foot_label = (
                        f"{self._geom_names[foot_geom]} "
                        f"{self._geom_body_names[foot_geom]}"
                    ).lower()
                    if side not in foot_label or "ankle_roll_sphere_" not in foot_label:
                        continue
                    rows.append(collision_jacobians[key][leg_qpos_indices])
                    lower_bounds.append(-float(distance) - self.penetration_tolerance)

                if not rows:
                    continue
                constraint_matrix = np.asarray(rows, dtype=np.float64)
                constraint_lower = np.asarray(lower_bounds, dtype=np.float64)
                if float(np.max(constraint_lower)) <= self.terrain_support_feasibility_tolerance:
                    continue

                delta = cp.Variable(len(leg_qpos_indices), name=f"terrain_{side}_leg_delta")
                delta_lower = (
                    self.q_a_lb[leg_q_a_positions]
                    - q_projected[leg_qpos_indices]
                )
                delta_upper = (
                    self.q_a_ub[leg_q_a_positions]
                    - q_projected[leg_qpos_indices]
                )
                if self.max_frame_joint_delta > 0.0:
                    delta_lower = np.maximum(
                        delta_lower,
                        q_t_last[leg_qpos_indices]
                        - self.max_frame_joint_delta
                        - q_projected[leg_qpos_indices],
                    )
                    delta_upper = np.minimum(
                        delta_upper,
                        q_t_last[leg_qpos_indices]
                        + self.max_frame_joint_delta
                        - q_projected[leg_qpos_indices],
                    )
                constraints = [
                    constraint_matrix @ delta >= constraint_lower,
                    delta >= delta_lower,
                    delta <= delta_upper,
                    cp.norm(delta, 2) <= min(0.05, self.step_size),
                ]
                problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), constraints)
                problem.solve(solver=cp.CLARABEL)
                if problem.status != cp.OPTIMAL:
                    continue
                linear_violation = max(
                    float(np.max(np.asarray(constraint.violation(), dtype=np.float64)))
                    for constraint in constraints
                )
                if linear_violation > self.terrain_support_feasibility_tolerance:
                    continue
                q_projected[leg_qpos_indices] += np.asarray(delta.value, dtype=np.float64)
                changed = True
            if not changed:
                break

        max_delta = float(
            np.max(np.abs(q_projected[self.q_a_indices] - q_start[self.q_a_indices]))
        )
        return q_projected, False, history, max_delta

    def _clip_final_configuration(
        self, q: np.ndarray, *, q_t_last: np.ndarray, init_t: bool
    ) -> tuple[np.ndarray, float]:
        """Remove small solver violations of joint and temporal trust bounds."""
        clipped = np.asarray(q, dtype=np.float64).copy()
        before = clipped.copy()
        if self.activate_joint_limits:
            clipped[self.q_a_indices] = np.clip(
                clipped[self.q_a_indices], self.q_a_lb, self.q_a_ub
            )
        if not init_t:
            if self.max_frame_root_translation > 0.0:
                translation_delta = clipped[:3] - q_t_last[:3]
                translation_norm = float(np.linalg.norm(translation_delta))
                if translation_norm > self.max_frame_root_translation:
                    clipped[:3] = q_t_last[:3] + translation_delta * (
                        self.max_frame_root_translation / translation_norm
                    )
            if self.max_frame_root_quaternion_delta > 0.0:
                if float(np.dot(clipped[3:7], q_t_last[3:7])) < 0.0:
                    clipped[3:7] *= -1.0
                quaternion_delta = clipped[3:7] - q_t_last[3:7]
                quaternion_delta_norm = float(np.linalg.norm(quaternion_delta))
                if quaternion_delta_norm > self.max_frame_root_quaternion_delta:
                    clipped[3:7] = q_t_last[3:7] + quaternion_delta * (
                        self.max_frame_root_quaternion_delta
                        / quaternion_delta_norm
                    )
            if self.max_frame_joint_delta > 0.0:
                joint_indices = self.q_a_indices[self.q_a_indices >= 7]
                clipped[joint_indices] = np.clip(
                    clipped[joint_indices],
                    q_t_last[joint_indices] - self.max_frame_joint_delta,
                    q_t_last[joint_indices] + self.max_frame_joint_delta,
                )
        clipped[3:7] /= np.linalg.norm(clipped[3:7]) + 1.0e-12
        return clipped, float(np.max(np.abs(clipped - before)))

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
        hand_contact_points_local=None,
        hand_contact_valid=None,
        hand_contact_weight_scale=None,
        contact_start_idx: int | None = None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array in MuJoCo order (trans, quat).
            object_poses_augmented (np.ndarray): (num_frames, 7) array in MuJoCo order (trans, quat).
            object_points_local_demo (np.ndarray): Demo object points in local frame.
            object_points_local (np.ndarray): Robot-side object points in local frame.
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
        """
        num_frames = human_joint_motions.shape[0]
        self._exact_nonpenetration_frame_iterations: list[int] = []
        self._exact_nonpenetration_frame_min_distance_m: list[float] = []
        self._exact_nonpenetration_frame_max_violation_m: list[float] = []
        self._exact_nonpenetration_frame_backtrack_count: list[int] = []
        self._exact_nonpenetration_frame_min_backtrack_alpha: list[float] = []
        self._exact_nonpenetration_restoration_frames: list[int] = []
        self._exact_nonpenetration_restoration_iterations: list[int] = []
        self._exact_nonpenetration_restoration_start_min_distance_m: list[float] = []
        self._exact_nonpenetration_restoration_final_min_distance_m: list[float] = []
        self._exact_nonpenetration_restoration_max_qpos_delta: list[float] = []
        self._exact_nonpenetration_restoration_success: list[bool] = []
        self._exact_nonpenetration_restoration_selected_alpha: list[float] = []
        self._exact_nonpenetration_failed_candidate_qpos = None
        self._exact_nonpenetration_failed_sqp_history = None
        self._exact_nonpenetration_failed_details = ""
        self._exact_nonpenetration_last_restoration_history = np.empty(
            (0, 8), dtype=np.float64
        )
        if self.foot_grounding_schedule == "before_contact":
            if contact_start_idx is None:
                raise ValueError(
                    "before_contact foot grounding requires contact_start_idx from the input NPZ"
                )
            resolved_contact_start_idx = int(contact_start_idx)
            if resolved_contact_start_idx != contact_start_idx:
                raise ValueError("contact_start_idx must be an integer")
            if not 0 <= resolved_contact_start_idx < num_frames:
                raise ValueError(
                    f"contact_start_idx={resolved_contact_start_idx} is outside [0, {num_frames})"
                )
            self._foot_grounding_contact_start_idx = resolved_contact_start_idx
        else:
            self._foot_grounding_contact_start_idx = -1
        if q_nominal_list is not None:
            q_locked_list = np.asarray(q_nominal_list, dtype=np.float64).copy()
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        if self.has_dynamic_object:
            q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        if self.initial_qpos_file:
            if self.ground_initial_robot:
                raise ValueError(
                    "initial_qpos_file cannot be combined with ground_initial_robot"
                )
            initial_path = Path(self.initial_qpos_file).expanduser().resolve()
            if not initial_path.is_file():
                raise FileNotFoundError(f"Initial qpos NPZ does not exist: {initial_path}")
            with np.load(initial_path, allow_pickle=True) as data:
                if "qpos" not in data:
                    raise KeyError(f"Initial qpos NPZ has no qpos array: {initial_path}")
                initial_qpos = np.asarray(data["qpos"], dtype=np.float64)
            if initial_qpos.shape != (1, self.nq) or not np.isfinite(initial_qpos).all():
                raise ValueError(
                    f"Initial qpos must have finite shape (1, {self.nq}), got {initial_qpos.shape}"
                )
            q = initial_qpos[0].copy()
            if self.has_dynamic_object:
                object_error = float(np.max(np.abs(q[-7:] - q_locked_list[0, -7:])))
                if object_error > 1.0e-8:
                    raise ValueError(
                        "Initial qpos object pose disagrees with current frame 0: "
                        f"max_abs_error={object_error:.6g}"
                    )
            q_locked_list[0, self.q_a_indices] = q[self.q_a_indices]
            self._initial_qpos_file_resolved = str(initial_path)
            print(f"[InitialQpos] Loaded frame-0 SQP initialization from {initial_path}")
        if self.ground_initial_robot and not self.resume_partial_file:
            q = self._ground_initial_robot_configuration(q)
            q_locked_list[0, self.q_a_indices] = q[self.q_a_indices]
        retargeted_motions = [q]
        resume_prefix = np.empty((0, self.nq), dtype=np.float64)
        resume_payload: dict[str, np.ndarray] = {}
        start_frame = 0
        if self.resume_partial_file:
            resume_path = Path(self.resume_partial_file).expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume partial NPZ does not exist: {resume_path}")
            with np.load(resume_path, allow_pickle=True) as data:
                resume_payload = {key: np.asarray(data[key]).copy() for key in data.files}
            if not bool(np.asarray(resume_payload.get("retarget_partial", False)).item()):
                raise ValueError(f"Resume input is not marked retarget_partial: {resume_path}")
            if "qpos" not in resume_payload:
                raise KeyError(f"Resume input has no qpos array: {resume_path}")
            resume_prefix = np.asarray(resume_payload["qpos"], dtype=np.float64)
            if (
                resume_prefix.ndim != 2
                or resume_prefix.shape[1] != self.nq
                or not 0 < len(resume_prefix) < num_frames
                or not np.isfinite(resume_prefix).all()
            ):
                raise ValueError(
                    f"Invalid resume qpos shape/content {resume_prefix.shape}; "
                    f"expected (1..{num_frames - 1}, {self.nq})"
                )
            start_frame = len(resume_prefix)
            failed_frame = int(
                np.asarray(resume_payload.get("retarget_failed_frame", start_frame)).item()
            )
            if failed_frame != start_frame:
                raise ValueError(
                    f"Resume failed frame {failed_frame} disagrees with prefix length {start_frame}"
                )
            if self.has_dynamic_object:
                object_error = float(
                    np.max(np.abs(resume_prefix[:, -7:] - q_locked_list[:start_frame, -7:]))
                )
                if object_error > 1.0e-8:
                    raise ValueError(
                        "Resume object trajectory disagrees with current input: "
                        f"max_abs_error={object_error:.6g}"
                    )
            q = np.copy(resume_prefix[-1])
            retargeted_motions = [q]
            self._resume_prefix_frames = start_frame
            print(f"[Resume] Loaded {start_frame} accepted frames from {resume_path}")

        tetrahedra = []
        obj_pts_demo_list = []
        obj_pts_list = []
        contact_errors = np.full((num_frames, 2), np.nan, dtype=np.float32)
        contact_point_offsets = np.full((num_frames, 2, 3), np.nan, dtype=np.float32)
        if start_frame and self.activate_hand_contact:
            prior_point_mode = str(
                np.asarray(
                    resume_payload.get("hand_contact_point_mode", "")
                ).reshape(-1)[0]
            )
            if prior_point_mode and prior_point_mode != self.hand_contact_point_mode:
                raise ValueError(
                    "Resume hand contact point mode disagrees with the current run: "
                    f"{prior_point_mode!r} vs {self.hand_contact_point_mode!r}"
                )
            if self.hand_contact_point_mode == "nearest_collision_surface":
                if prior_point_mode != self.hand_contact_point_mode:
                    raise ValueError(
                        "Nearest-surface hand contact cannot resume a prefix without an "
                        "explicit matching hand_contact_point_mode"
                    )
                if "hand_contact_point_offsets_local" not in resume_payload:
                    raise KeyError(
                        "Nearest-surface hand contact resume is missing "
                        "hand_contact_point_offsets_local"
                    )
        if start_frame and self.enforce_exact_nonpenetration:
            prior_exact_enabled = bool(
                np.asarray(
                    resume_payload.get("retarget_exact_nonpenetration_enabled", False)
                ).item()
            )
            if not prior_exact_enabled:
                raise ValueError(
                    "Exact non-penetration cannot resume a prefix that was not accepted "
                    "by the exact nonlinear gate"
                )
            resume_exact_config = (
                (
                    "retarget_penetration_tolerance_m",
                    self.penetration_tolerance,
                    np.nan,
                ),
                (
                    "retarget_exact_nonpenetration_feasibility_tolerance_m",
                    self.exact_nonpenetration_feasibility_tolerance,
                    np.nan,
                ),
                (
                    "retarget_exact_nonpenetration_interior_margin_m",
                    self.exact_nonpenetration_interior_margin,
                    0.0,
                ),
                (
                    "retarget_exact_nonpenetration_qp_safety_margin_m",
                    self.exact_nonpenetration_qp_safety_margin,
                    0.0,
                ),
            )
            for key, expected, default in resume_exact_config:
                prior = float(np.asarray(resume_payload.get(key, default)).item())
                if not np.isfinite(prior) or not np.isclose(
                    prior, expected, rtol=0.0, atol=1.0e-15
                ):
                    raise ValueError(
                        f"Exact non-penetration resume config {key} disagrees with "
                        f"the current run: {prior} vs {expected}"
                    )
            resume_exact_keys = (
                "retarget_exact_nonpenetration_sqp_iterations",
                "retarget_exact_nonpenetration_min_distance_m",
                "retarget_exact_nonpenetration_max_violation_m",
                "retarget_exact_nonpenetration_backtrack_count",
                "retarget_exact_nonpenetration_min_backtrack_alpha",
            )
            missing_exact_keys = [key for key in resume_exact_keys if key not in resume_payload]
            if missing_exact_keys:
                raise KeyError(
                    "Exact non-penetration resume metadata is incomplete: "
                    f"{missing_exact_keys}"
                )
            resume_exact_values = [
                np.asarray(resume_payload[key]).reshape(-1) for key in resume_exact_keys
            ]
            if any(len(values) != start_frame for values in resume_exact_values):
                raise ValueError(
                    "Exact non-penetration resume metadata length disagrees with the prefix"
                )
            self._exact_nonpenetration_frame_iterations = [
                int(value) for value in resume_exact_values[0]
            ]
            self._exact_nonpenetration_frame_min_distance_m = [
                float(value) for value in resume_exact_values[1]
            ]
            self._exact_nonpenetration_frame_max_violation_m = [
                float(value) for value in resume_exact_values[2]
            ]
            self._exact_nonpenetration_frame_backtrack_count = [
                int(value) for value in resume_exact_values[3]
            ]
            self._exact_nonpenetration_frame_min_backtrack_alpha = [
                float(value) for value in resume_exact_values[4]
            ]
            restoration_keys = (
                "retarget_exact_nonpenetration_restoration_frames",
                "retarget_exact_nonpenetration_restoration_iterations",
                "retarget_exact_nonpenetration_restoration_start_min_distance_m",
                "retarget_exact_nonpenetration_restoration_final_min_distance_m",
                "retarget_exact_nonpenetration_restoration_max_qpos_delta",
                "retarget_exact_nonpenetration_restoration_success",
            )
            present_restoration_keys = [
                key for key in restoration_keys if key in resume_payload
            ]
            if present_restoration_keys:
                missing_restoration_keys = [
                    key for key in restoration_keys if key not in resume_payload
                ]
                if missing_restoration_keys:
                    raise KeyError(
                        "Exact non-penetration restoration resume metadata is incomplete: "
                        f"{missing_restoration_keys}"
                    )
                restoration_values = [
                    np.asarray(resume_payload[key]).reshape(-1)
                    for key in restoration_keys
                ]
                restoration_event_count = len(restoration_values[0])
                if any(
                    len(values) != restoration_event_count
                    for values in restoration_values[1:]
                ):
                    raise ValueError(
                        "Exact non-penetration restoration resume metadata lengths disagree"
                    )
                restoration_frames = np.asarray(
                    restoration_values[0], dtype=np.int64
                )
                if (
                    np.any(restoration_frames < 0)
                    or np.any(restoration_frames >= start_frame)
                ):
                    raise ValueError(
                        "Exact non-penetration restoration event frame is outside the "
                        "accepted resume prefix"
                    )
                self._exact_nonpenetration_restoration_frames = [
                    int(value) for value in restoration_values[0]
                ]
                self._exact_nonpenetration_restoration_iterations = [
                    int(value) for value in restoration_values[1]
                ]
                self._exact_nonpenetration_restoration_start_min_distance_m = [
                    float(value) for value in restoration_values[2]
                ]
                self._exact_nonpenetration_restoration_final_min_distance_m = [
                    float(value) for value in restoration_values[3]
                ]
                self._exact_nonpenetration_restoration_max_qpos_delta = [
                    float(value) for value in restoration_values[4]
                ]
                self._exact_nonpenetration_restoration_success = [
                    bool(value) for value in restoration_values[5]
                ]
                selected_alpha = np.asarray(
                    resume_payload.get(
                        "retarget_exact_nonpenetration_restoration_selected_alpha",
                        np.full(restoration_event_count, np.nan, dtype=np.float64),
                    )
                ).reshape(-1)
                if len(selected_alpha) != restoration_event_count:
                    raise ValueError(
                        "Exact non-penetration restoration selected-alpha length "
                        "disagrees with restoration event count"
                    )
                self._exact_nonpenetration_restoration_selected_alpha = [
                    float(value) for value in selected_alpha
                ]
        if start_frame and "hand_contact_error_m" in resume_payload:
            prefix_contact_error = np.asarray(
                resume_payload["hand_contact_error_m"], dtype=np.float32
            )
            if prefix_contact_error.shape != (start_frame, 2):
                raise ValueError(
                    "Resume hand_contact_error_m shape disagrees with prefix: "
                    f"{prefix_contact_error.shape} vs {(start_frame, 2)}"
                )
            contact_errors[:start_frame] = prefix_contact_error
        if start_frame and "hand_contact_point_offsets_local" in resume_payload:
            prefix_contact_offsets = np.asarray(
                resume_payload["hand_contact_point_offsets_local"], dtype=np.float32
            )
            if prefix_contact_offsets.shape != (start_frame, 2, 3):
                raise ValueError(
                    "Resume hand_contact_point_offsets_local shape disagrees with prefix: "
                    f"{prefix_contact_offsets.shape} vs {(start_frame, 2, 3)}"
                )
            contact_point_offsets[:start_frame] = prefix_contact_offsets
        hand_contact_points_local, hand_contact_valid, hand_contact_weight_scale = self._normalize_hand_contact_inputs(
            hand_contact_points_local,
            hand_contact_valid,
            num_frames,
            hand_contact_weight_scale,
        )
        last_cost = np.nan
        completed_frames = start_frame

        def completed_qpos() -> np.ndarray:
            suffix = np.asarray(retargeted_motions[1:], dtype=np.float64)
            if suffix.size == 0:
                return resume_prefix.copy()
            return np.concatenate((resume_prefix, suffix), axis=0)

        def save_partial_result(error: str) -> None:
            if dest_res_path is None:
                return
            self._save_results(
                dest_res_path,
                completed_qpos(),
                human_joint_motions[:completed_frames],
                last_cost,
                hand_contact_points_local=(
                    hand_contact_points_local[:completed_frames]
                    if hand_contact_points_local is not None
                    else None
                ),
                hand_contact_valid=(
                    hand_contact_valid[:completed_frames]
                    if hand_contact_valid is not None
                    else None
                ),
                hand_contact_error=contact_errors[:completed_frames],
                hand_contact_point_offsets_local=contact_point_offsets[:completed_frames],
                partial=True,
                failed_frame=completed_frames,
                error=error,
                hand_contact_weight_scale=(
                    hand_contact_weight_scale[:completed_frames]
                    if hand_contact_weight_scale is not None
                    else None
                ),
            )

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        try:
            with tqdm(
                range(start_frame, num_frames), initial=start_frame, total=num_frames
            ) as pbar:
                for i in pbar:
                    object_quat_demo = object_poses[i, 3:]
                    object_trans_demo = object_poses[i, :3]
                    human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]

                    if self.object_name == "ground":
                        human_mapped_joints_in_object = human_mapped_joints
                    else:
                        human_mapped_joints_in_object = transform_points_world_to_local(
                            object_quat_demo, object_trans_demo, human_mapped_joints
                        )

                    frame_contact_points = None if hand_contact_points_local is None else hand_contact_points_local[i]
                    frame_contact_valid = None if hand_contact_valid is None else hand_contact_valid[i]
                    frame_contact_weight_scale = (
                        None if hand_contact_weight_scale is None else hand_contact_weight_scale[i]
                    )
                    human_mapped_joints_in_object = self._replace_wrist_targets_with_contact(
                        human_mapped_joints_in_object,
                        frame_contact_points,
                        frame_contact_valid,
                        frame_contact_weight_scale,
                    )

                    source_vertices, source_tetrahedra = create_interaction_mesh(
                        np.vstack([human_mapped_joints_in_object, object_points_local_demo])
                    )
                    tetrahedra.append(source_tetrahedra)

                    if self.debug:
                        object_quat = object_poses_augmented[i, 3:]
                        object_trans = object_poses_augmented[i, :3]
                        obj_pts_demo = transform_points_local_to_world(
                            object_quat_demo, object_trans_demo, object_points_local_demo
                        )
                        obj_pts = transform_points_local_to_world(object_quat, object_trans, object_points_local)

                        obj_pts_demo_list.append(obj_pts_demo)
                        obj_pts_list.append(obj_pts)
                        human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")
                        obj_kpts_demo_handle_list = self.draw_keypoints(
                            obj_pts_demo, name="object_demo_kpts", rgba=(1, 0, 0, 1)
                        )
                        obj_kpts_handle_list = self.draw_keypoints(obj_pts, name="object_kpts", rgba=(0, 1, 1, 1))

                    adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                    target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)

                    if original:
                        w_nominal_tracking = self.w_nominal_tracking_init
                    else:
                        w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                    q, cost = self.iterate(
                        q_locked=q_locked_list[i],
                        q_n=q,
                        q_t_last=retargeted_motions[-1],
                        target_laplacian=target_laplacian,
                        adj_list=adj_list,
                        obj_pts_local=object_points_local,
                        foot_sticking=foot_sticking_sequences[i],
                        target_robot_pts_local=human_mapped_joints_in_object,
                        hand_contact_targets_local=frame_contact_points,
                        hand_contact_valid=frame_contact_valid,
                        hand_contact_weight_scale=frame_contact_weight_scale,
                        w_nominal_tracking=w_nominal_tracking,
                        q_a_nominal=(q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None),
                        init_t=i == 0,
                        n_iter=50 if i == 0 else 10,
                        frame_idx=i,
                        foot_grounding_weight_scale=self._foot_grounding_schedule_scale(
                            frame_idx=i,
                            schedule=self.foot_grounding_schedule,
                            contact_start_idx=(
                                self._foot_grounding_contact_start_idx
                                if self._foot_grounding_contact_start_idx >= 0
                                else None
                            ),
                            ramp_frames=self.foot_grounding_ramp_frames,
                        ),
                    )
                    last_cost = cost
                    self._record_hand_contact_error(
                        q,
                        frame_contact_points,
                        frame_contact_valid,
                        contact_errors[i],
                        frame_contact_weight_scale,
                        contact_point_offsets[i],
                    )

                    if self.debug:
                        robot_link_positions = self._get_robot_link_positions(q, self.laplacian_match_links.values())
                        robot_kpts_handle_list = self.draw_keypoints(
                            robot_link_positions, name="robot_kpts", rgba=(0, 1, 0, 1)
                        )

                    retargeted_motions.append(q)
                    completed_frames += 1
                    checkpoint_interval = int(
                        getattr(self, "partial_checkpoint_interval_frames", 0)
                    )
                    if (
                        checkpoint_interval > 0
                        and completed_frames < num_frames
                        and completed_frames % checkpoint_interval == 0
                    ):
                        save_partial_result(
                            "in-progress periodic checkpoint after "
                            f"{completed_frames} accepted frames"
                        )
                        print(
                            "[Checkpoint] Saved "
                            f"{completed_frames} accepted frames to {dest_res_path}"
                        )
                    if self.visualize and self.debug:
                        self.draw_q(q)

                    pbar.set_postfix(cost=cost)
        except Exception as exc:
            if self.save_partial_on_failure and dest_res_path is not None:
                save_partial_result(str(exc))
            raise

        if self.debug:
            for name in (
                "human_kpts_handle_list",
                "obj_kpts_demo_handle_list",
                "obj_kpts_handle_list",
                "robot_kpts_handle_list",
            ):
                for handle in locals().get(name, []) or []:
                    handle.remove()

        self._save_results(
            dest_res_path,
            completed_qpos(),
            human_joint_motions,
            last_cost,
            hand_contact_points_local=hand_contact_points_local,
            hand_contact_valid=hand_contact_valid,
            hand_contact_error=contact_errors,
            hand_contact_point_offsets_local=contact_point_offsets,
            hand_contact_weight_scale=hand_contact_weight_scale,
        )
        print("Saving results to path:", dest_res_path)

        if self.visualize:
            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            np.array(retargeted_motions)[1:],
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    @staticmethod
    def _foot_grounding_schedule_scale(
        frame_idx: int,
        schedule: str,
        contact_start_idx: int | None,
        ramp_frames: int,
    ) -> float:
        """Return the per-frame multiplier for the soft grounding objective."""

        if schedule == "all_frames":
            return 1.0
        if schedule != "before_contact":
            raise ValueError(f"Unsupported foot grounding schedule: {schedule}")
        if contact_start_idx is None:
            raise ValueError("before_contact schedule requires contact_start_idx")
        if ramp_frames < 0:
            raise ValueError("ramp_frames must be non-negative")
        if frame_idx >= contact_start_idx:
            return 0.0
        if ramp_frames == 0:
            return 1.0
        return float(min(1.0, (contact_start_idx - frame_idx) / ramp_frames))

    @staticmethod
    def _select_foot_grounding_keys(
        foot_positions: dict[str, np.ndarray],
        foot_sticking: dict[str, bool],
        left_sticking_key: str,
        right_sticking_key: str,
        mode: str,
    ) -> list[str]:
        """Select the foot points used by the soft grounding objective."""

        left_keys = [key for key in foot_positions if "left" in key]
        right_keys = [key for key in foot_positions if "right" in key]
        if not left_keys or not right_keys:
            raise ValueError("foot positions must include left and right links")

        ground_left = bool(foot_sticking[left_sticking_key])
        ground_right = bool(foot_sticking[right_sticking_key])
        continuous_modes = {"continuous_support", "continuous_lowest_sphere"}
        if mode == "lowest_foot" or (
            mode in continuous_modes and not ground_left and not ground_right
        ):
            left_z = min(float(np.asarray(foot_positions[key])[2]) for key in left_keys)
            right_z = min(float(np.asarray(foot_positions[key])[2]) for key in right_keys)
            ground_left = left_z <= right_z
            ground_right = not ground_left

        selected = []
        if ground_left:
            selected.extend(left_keys)
        if ground_right:
            selected.extend(right_keys)

        if mode != "continuous_lowest_sphere":
            return selected

        # Ground one active contact point per selected foot. Pulling all five
        # sole spheres to the same Z forces the foot flat and can fight the
        # ankle pose even though a single sphere already establishes support.
        result = []
        for keys in (left_keys if ground_left else [], right_keys if ground_right else []):
            if keys:
                result.append(
                    min(
                        keys,
                        key=lambda key: (
                            float(np.asarray(foot_positions[key])[2]),
                            key,
                        ),
                    )
                )
        return result

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        target_robot_pts_local: np.ndarray | None = None,
        hand_contact_targets_local: np.ndarray | None = None,
        hand_contact_valid: np.ndarray | None = None,
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        hand_contact_weight_scale: np.ndarray | None = None,
        foot_grounding_weight_scale: float = 1.0,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a
        effective_foot_grounding_weight = (
            self.foot_grounding_weight * float(foot_grounding_weight_scale)
        )
        if (
            not np.isfinite(effective_foot_grounding_weight)
            or effective_foot_grounding_weight < 0.0
        ):
            raise ValueError("effective foot grounding weight must be finite and non-negative")

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last
        hand_contact_point_offsets = None
        if (
            self.activate_hand_contact
            and hand_contact_targets_local is not None
            and hand_contact_valid is not None
        ):
            hand_contact_point_offsets = self._hand_contact_point_offsets(
                q,
                hand_contact_targets_local,
                hand_contact_valid,
                hand_contact_weight_scale,
            )

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q,
            links=self.laplacian_match_links,
            obj_frame=(self.object_name != "ground"),
            point_offsets=(
                hand_contact_point_offsets
                if self.replace_source_wrist_with_contact
                else None
            ),
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []
        terrain_linear_constraints = []

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot constraints and optional soft grounding residuals.
        foot_grounding_residuals = []
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_grounding = apply_foot_sticking and effective_foot_grounding_weight > 0.0
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        apply_initial_ground_lock = self.ground_initial_robot and frame_idx == 0
        if apply_foot_sticking or apply_foot_grounding or apply_foot_lock or apply_initial_ground_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            left_key = right_key = None
            for key in foot_sticking:
                if key.lower().startswith("l"):
                    left_key = key
                elif key.lower().startswith("r"):
                    right_key = key
            if (apply_foot_sticking or apply_foot_grounding) and (
                left_key is None or right_key is None
            ):
                raise ValueError("foot_sticking must include one left* and one right* key")

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        constraints += [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]
                        if self.foot_sticking_pin_z:
                            z_delta = self.foot_sticking_z_floor - p_WF_dict[key][2]
                            Jz = J_WF[2, self.q_a_indices]
                            constraints += [
                                Jz @ dqa >= z_delta - self.foot_sticking_tolerance,
                                Jz @ dqa <= z_delta + self.foot_sticking_tolerance,
                            ]

            if apply_foot_grounding:
                grounding_keys = self._select_foot_grounding_keys(
                    p_WF_dict,
                    foot_sticking,
                    left_key,
                    right_key,
                    self.foot_grounding_mode,
                )
                for key in grounding_keys:
                    Jz = J_WF_dict[key][2, self.q_a_indices]
                    foot_grounding_residuals.append(
                        Jz @ dqa + p_WF_dict[key][2] - self.foot_sticking_z_floor
                    )

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    if not self._is_foot_locked_in_window(key, frame_idx):
                        continue

                    z_anchor = self.foot_lock.z_floor
                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]

            # Keep the grounded initializer grounded through the frame-0 SQP.
            # This is separate from velocity-based sticking, which may be false
            # on frame 0 and is intentionally XY-only by default.
            if apply_initial_ground_lock:
                for key, J_WF in J_WF_dict.items():
                    z_delta = self.foot_sticking_z_floor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    constraints += [
                        Jz @ dqa >= z_delta - self.foot_sticking_tolerance,
                        Jz @ dqa <= z_delta + self.foot_sticking_tolerance,
                    ]

        for _, jacobian, signed_distance, _, _ in self._build_terrain_support_constraints(q):
            target_distance = self.terrain_support_sphere_radius + self.terrain_support_clearance
            constraint = (
                jacobian[self.q_a_indices] @ dqa
                >= target_distance - signed_distance
            )
            constraints.append(constraint)
            terrain_linear_constraints.append(constraint)

        contact_residuals = self._build_hand_contact_residuals(
            q,
            dqa,
            hand_contact_targets_local,
            hand_contact_valid,
            hand_contact_weight_scale,
            hand_contact_point_offsets,
        )
        if self.activate_hand_contact and contact_residuals and self.hand_contact_mode == "hard":
            for _, _, residual in contact_residuals:
                constraints += [
                    residual >= -self.hand_contact_tolerance,
                    residual <= self.hand_contact_tolerance,
                ]

        # Non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        required_distance = -self.penetration_tolerance
        if self.enforce_exact_nonpenetration:
            required_distance = self._linearized_nonpenetration_required_distance()
        for key, phi in phis.items():
            Ja_n_full = Js[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            rhs = required_distance - phi
            constraints += [Ja_n @ dqa >= rhs]

        # Self-collision constraints
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
            rhs = self._self_collision_tolerance - phi
            constraints += [Ja_n @ dqa >= rhs]

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            constraints += [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]

        # Step size constraints (Lorentz cone)
        constraints += [cp.SOC(self.step_size, dqa)]

        if not init_t:
            frame_delta = dqa + q_a_n_last - q_t_last[self.q_a_indices]
            root_translation_positions = np.flatnonzero(self.q_a_indices < 3)
            if self.max_frame_root_translation > 0.0 and root_translation_positions.size:
                constraints.append(
                    cp.norm(frame_delta[root_translation_positions], 2)
                    <= self.max_frame_root_translation
                )
            root_quaternion_positions = np.flatnonzero(
                (self.q_a_indices >= 3) & (self.q_a_indices < 7)
            )
            if (
                self.max_frame_root_quaternion_delta > 0.0
                and root_quaternion_positions.size
            ):
                constraints.append(
                    cp.norm(frame_delta[root_quaternion_positions], 2)
                    <= self.max_frame_root_quaternion_delta
                )
            joint_positions = np.flatnonzero(self.q_a_indices >= 7)
            if self.max_frame_joint_delta > 0.0 and joint_positions.size:
                constraints.extend(
                    [
                        frame_delta[joint_positions] >= -self.max_frame_joint_delta,
                        frame_delta[joint_positions] <= self.max_frame_joint_delta,
                    ]
                )

        # Objective
        obj_terms = []

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))
        if self.w_keypoint_tracking > 0.0 and target_robot_pts_local is not None:
            robot_position_residual = (
                J_V[: 3 * V_r, :] @ dqa
                + robot_pts_local.reshape(-1)
                - np.asarray(target_robot_pts_local, dtype=float).reshape(-1)
            )
            obj_terms.append(self.w_keypoint_tracking * cp.sum_squares(robot_position_residual))
        if self.activate_hand_contact and contact_residuals and self.hand_contact_mode == "soft":
            obj_terms.extend(self._soft_hand_contact_objective_terms(contact_residuals))
        if foot_grounding_residuals:
            obj_terms.append(
                effective_foot_grounding_weight
                * cp.sum_squares(cp.hstack(foot_grounding_residuals))
            )

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # Strict production solve: retain every hard constraint and accept only
        # an accurate Clarabel optimum. Infeasible/inaccurate frames must fail so
        # the caller can preserve them as explicitly marked partial results.
        try:
            problem.solve(solver=cp.CLARABEL, verbose=verbose)
        except cp.error.SolverError as exc:
            raise RuntimeError(f"CVXPY strict solve failed: CLARABEL {type(exc).__name__}") from exc

        self._last_solver_notes = f"CLARABEL: {problem.status}"
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f"CVXPY strict solve failed: {self._last_solver_notes}")
        self._last_terrain_linear_constraint_violation = max(
            (
                float(np.max(np.asarray(constraint.violation(), dtype=np.float64)))
                for constraint in terrain_linear_constraints
            ),
            default=0.0,
        )

        dqa_star = np.asarray(dqa.value, dtype=np.float64)
        if dqa_star.shape != (len(self.q_a_indices),) or not np.isfinite(dqa_star).all():
            raise RuntimeError("CVXPY strict solve returned an invalid dqa")
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        return q_star, cost

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> bool:
        """Check whether a foot link is locked by configured frame windows."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return False

        return any(start <= frame_idx <= end for start, end in self._foot_lock_windows.get(side, ()))

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def _exact_nonpenetration_required_distance(self) -> float:
        """Return the nonlinear exact distance target."""
        return (
            -self.penetration_tolerance
            + float(getattr(self, "exact_nonpenetration_interior_margin", 0.0))
        )

    def _linearized_nonpenetration_required_distance(self) -> float:
        """Return the QP target with extra room for linearization error."""
        return self._exact_nonpenetration_required_distance() + float(
            getattr(self, "exact_nonpenetration_qp_safety_margin", 0.0)
        )

    def _exact_nonpenetration_feasibility(
        self, q: np.ndarray
    ) -> tuple[float, float, tuple[int, int] | None]:
        """Return nonlinear violation, minimum distance, and limiting geom pair."""
        _, phis = self._update_jacobians_and_phis_from_q(
            q, compute_jacobians=False
        )
        if not phis:
            return 0.0, np.inf, None
        limiting_pair = min(phis, key=phis.__getitem__)
        minimum_distance = float(phis[limiting_pair])
        required_minimum_distance = self._exact_nonpenetration_required_distance()
        violation = max(0.0, required_minimum_distance - minimum_distance)
        return violation, minimum_distance, limiting_pair

    def _exact_nonpenetration_details(
        self, minimum_distance: float, limiting_pair: tuple[int, int] | None
    ) -> str:
        if limiting_pair is None:
            return "no active object/ground collision pair"
        geom_a, geom_b = limiting_pair
        return (
            f"pair={self._geom_names[geom_a]}::{self._geom_names[geom_b]}, "
            f"distance_m={minimum_distance:.9g}, "
            "required_min_m="
            f"{self._exact_nonpenetration_required_distance():.9g}"
        )

    @staticmethod
    def _interpolate_qpos(q_from: np.ndarray, q_to: np.ndarray, alpha: float) -> np.ndarray:
        """Interpolate qpos while keeping the floating-base quaternion normalized."""
        start = np.asarray(q_from, dtype=np.float64)
        end = np.asarray(q_to, dtype=np.float64).copy()
        if start.shape != end.shape:
            raise ValueError(f"qpos interpolation shape mismatch: {start.shape} vs {end.shape}")
        if len(start) >= 7 and float(np.dot(start[3:7], end[3:7])) < 0.0:
            end[3:7] *= -1.0
        trial = start + float(alpha) * (end - start)
        if len(trial) >= 7:
            quaternion_norm = float(np.linalg.norm(trial[3:7]))
            if not np.isfinite(quaternion_norm) or quaternion_norm <= 1.0e-12:
                raise RuntimeError("Backtracked floating-base quaternion is invalid")
            trial[3:7] /= quaternion_norm
        return trial

    def _backtrack_exact_nonpenetration(
        self, q_feasible: np.ndarray, q_candidate: np.ndarray
    ) -> tuple[np.ndarray, float, float, tuple[int, int] | None, float]:
        """Keep the largest bisection fraction that remains nonlinearly feasible."""
        (
            feasible_violation,
            feasible_minimum_distance,
            feasible_limiting_pair,
        ) = self._exact_nonpenetration_feasibility(q_feasible)
        if feasible_violation > self.exact_nonpenetration_feasibility_tolerance:
            raise ValueError("Exact non-penetration backtracking requires a feasible start")

        lower = 0.0
        upper = 1.0
        best_q = np.asarray(q_feasible, dtype=np.float64).copy()
        best_violation = feasible_violation
        best_minimum_distance = feasible_minimum_distance
        best_limiting_pair = feasible_limiting_pair
        for _ in range(self.exact_nonpenetration_backtracking_steps):
            alpha = 0.5 * (lower + upper)
            trial = self._interpolate_qpos(q_feasible, q_candidate, alpha)
            violation, minimum_distance, limiting_pair = (
                self._exact_nonpenetration_feasibility(trial)
            )
            if violation <= self.exact_nonpenetration_feasibility_tolerance:
                lower = alpha
                best_q = trial
                best_violation = violation
                best_minimum_distance = minimum_distance
                best_limiting_pair = limiting_pair
            else:
                upper = alpha
        return (
            best_q,
            best_violation,
            best_minimum_distance,
            best_limiting_pair,
            lower,
        )

    def _restore_exact_nonpenetration_feasibility(
        self,
        q_start: np.ndarray,
        *,
        q_t_last: np.ndarray,
        init_t: bool,
    ) -> tuple[np.ndarray, bool, np.ndarray, float]:
        """Project an infeasible current-frame seed before running the main SQP objective."""
        q_restored = np.asarray(q_start, dtype=np.float64).copy()
        q_initial = q_restored.copy()
        history: list[tuple[float, ...]] = []
        line_search_steps = max(2, int(self.exact_nonpenetration_backtracking_steps))

        for restoration_iteration in range(
            self.exact_nonpenetration_restoration_max_iterations
        ):
            current_violation, current_minimum, _ = (
                self._exact_nonpenetration_feasibility(q_restored)
            )
            if current_violation <= self.exact_nonpenetration_feasibility_tolerance:
                max_delta = float(
                    np.max(
                        np.abs(
                            q_restored[self.q_a_indices]
                            - q_initial[self.q_a_indices]
                        )
                    )
                )
                return q_restored, True, np.asarray(history, dtype=np.float64), max_delta

            jacobians, distances = self._update_jacobians_and_phis_from_q(q_restored)
            if not distances:
                break

            dqa = cp.Variable(len(self.q_a_indices), name="exact_restoration_dqa")
            constraints = []
            required_distance = self._linearized_nonpenetration_required_distance()
            for key, distance in distances.items():
                jacobian = np.asarray(jacobians[key], dtype=np.float64)[
                    self.q_a_indices
                ]
                constraints.append(
                    jacobian @ dqa >= required_distance - float(distance)
                )

            q_a_current = q_restored[self.q_a_indices]
            if self.activate_joint_limits:
                constraints.extend(
                    [
                        dqa >= self.q_a_lb - q_a_current,
                        dqa <= self.q_a_ub - q_a_current,
                    ]
                )
            constraints.append(cp.SOC(self.step_size, dqa))

            if not init_t:
                frame_delta = dqa + q_a_current - q_t_last[self.q_a_indices]
                root_translation_positions = np.flatnonzero(self.q_a_indices < 3)
                if self.max_frame_root_translation > 0.0 and root_translation_positions.size:
                    constraints.append(
                        cp.norm(frame_delta[root_translation_positions], 2)
                        <= self.max_frame_root_translation
                    )
                root_quaternion_positions = np.flatnonzero(
                    (self.q_a_indices >= 3) & (self.q_a_indices < 7)
                )
                if (
                    self.max_frame_root_quaternion_delta > 0.0
                    and root_quaternion_positions.size
                ):
                    constraints.append(
                        cp.norm(frame_delta[root_quaternion_positions], 2)
                        <= self.max_frame_root_quaternion_delta
                    )
                joint_positions = np.flatnonzero(self.q_a_indices >= 7)
                if self.max_frame_joint_delta > 0.0 and joint_positions.size:
                    constraints.extend(
                        [
                            frame_delta[joint_positions] >= -self.max_frame_joint_delta,
                            frame_delta[joint_positions] <= self.max_frame_joint_delta,
                        ]
                    )

            problem = cp.Problem(cp.Minimize(cp.sum_squares(dqa)), constraints)
            try:
                problem.solve(solver=cp.CLARABEL)
            except cp.error.SolverError:
                break
            if problem.status != cp.OPTIMAL or dqa.value is None:
                break
            delta = np.asarray(dqa.value, dtype=np.float64)
            if delta.shape != (len(self.q_a_indices),) or not np.isfinite(delta).all():
                break

            q_candidate = q_restored.copy()
            q_candidate[self.q_a_indices] += delta
            if len(q_candidate) >= 7:
                quaternion_norm = float(np.linalg.norm(q_candidate[3:7]))
                if not np.isfinite(quaternion_norm) or quaternion_norm <= 1.0e-12:
                    break
                q_candidate[3:7] /= quaternion_norm
            candidate_violation, candidate_minimum, _ = (
                self._exact_nonpenetration_feasibility(q_candidate)
            )

            if (
                candidate_violation
                <= self.exact_nonpenetration_feasibility_tolerance
            ):
                history.append(
                    (
                        restoration_iteration + 1,
                        current_minimum,
                        current_violation,
                        candidate_minimum,
                        candidate_violation,
                        candidate_minimum,
                        candidate_violation,
                        1.0,
                    )
                )
                max_delta = float(
                    np.max(
                        np.abs(
                            q_candidate[self.q_a_indices]
                            - q_initial[self.q_a_indices]
                        )
                    )
                )
                return (
                    q_candidate,
                    True,
                    np.asarray(history, dtype=np.float64),
                    max_delta,
                )

            best_q = q_restored
            best_violation = current_violation
            best_minimum = current_minimum
            best_alpha = 0.0
            for alpha in np.linspace(1.0, 1.0 / line_search_steps, line_search_steps):
                trial = self._interpolate_qpos(q_restored, q_candidate, float(alpha))
                trial_violation, trial_minimum, _ = (
                    self._exact_nonpenetration_feasibility(trial)
                )
                if trial_violation < best_violation - 1.0e-12:
                    best_q = trial
                    best_violation = trial_violation
                    best_minimum = trial_minimum
                    best_alpha = float(alpha)
                    if (
                        trial_violation
                        <= self.exact_nonpenetration_feasibility_tolerance
                    ):
                        break

            history.append(
                (
                    restoration_iteration + 1,
                    current_minimum,
                    current_violation,
                    candidate_minimum,
                    candidate_violation,
                    best_minimum,
                    best_violation,
                    best_alpha,
                )
            )
            if best_alpha == 0.0:
                break
            q_restored = np.asarray(best_q, dtype=np.float64).copy()

        final_violation, _, _ = self._exact_nonpenetration_feasibility(q_restored)
        max_delta = float(
            np.max(
                np.abs(
                    q_restored[self.q_a_indices] - q_initial[self.q_a_indices]
                )
            )
        )
        return (
            q_restored,
            final_violation <= self.exact_nonpenetration_feasibility_tolerance,
            np.asarray(history, dtype=np.float64),
            max_delta,
        )

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        target_robot_pts_local: np.ndarray | None = None,
        hand_contact_targets_local: np.ndarray | None = None,
        hand_contact_valid: np.ndarray | None = None,
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        frame_idx: int = 0,
        hand_contact_weight_scale: np.ndarray | None = None,
        foot_grounding_weight_scale: float = 1.0,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        terrain_enabled = self._terrain_support_enabled
        exact_nonpenetration_enabled = self.enforce_exact_nonpenetration
        if terrain_enabled:
            self._begin_terrain_support_frame(q_n)
        max_iterations = n_iter
        if terrain_enabled:
            max_iterations = max(max_iterations, self.terrain_support_max_sqp_iterations)
        if exact_nonpenetration_enabled:
            max_iterations = max(
                max_iterations, self.exact_nonpenetration_max_sqp_iterations
            )
        terrain_feasible = not terrain_enabled
        exact_nonpenetration_feasible = not exact_nonpenetration_enabled
        top_violation = collision_violation = 0.0
        exact_violation = 0.0
        exact_minimum_distance = np.inf
        exact_limiting_pair = None
        sqp_history = []
        exact_sqp_history = []
        frame_backtrack_count = 0
        frame_min_backtrack_alpha = 1.0
        self._terrain_failed_candidate_qpos = None
        self._terrain_failed_sqp_history = None
        self._terrain_failed_details = ""
        self._terrain_failed_projection_history = None
        self._exact_nonpenetration_last_restoration_history = np.empty(
            (0, 8), dtype=np.float64
        )
        if (
            exact_nonpenetration_enabled
            and self.exact_nonpenetration_restore_infeasible_start
        ):
            q_current_locked = np.asarray(q_locked, dtype=np.float64).copy()
            q_current_locked[self.q_a_indices] = np.asarray(q_n, dtype=np.float64)[
                self.q_a_indices
            ]
            start_violation, start_minimum_distance, _ = (
                self._exact_nonpenetration_feasibility(q_current_locked)
            )
            if start_violation > self.exact_nonpenetration_feasibility_tolerance:
                (
                    q_n,
                    restoration_success,
                    restoration_history,
                    restoration_max_delta,
                ) = self._restore_exact_nonpenetration_feasibility(
                    q_current_locked,
                    q_t_last=q_t_last,
                    init_t=init_t,
                )
                _, restoration_final_minimum, _ = (
                    self._exact_nonpenetration_feasibility(q_n)
                )
                self._exact_nonpenetration_last_restoration_history = (
                    restoration_history
                )
                self._exact_nonpenetration_restoration_frames.append(int(frame_idx))
                self._exact_nonpenetration_restoration_iterations.append(
                    int(len(restoration_history))
                )
                self._exact_nonpenetration_restoration_start_min_distance_m.append(
                    float(start_minimum_distance)
                )
                self._exact_nonpenetration_restoration_final_min_distance_m.append(
                    float(restoration_final_minimum)
                )
                self._exact_nonpenetration_restoration_max_qpos_delta.append(
                    float(restoration_max_delta)
                )
                self._exact_nonpenetration_restoration_success.append(
                    bool(restoration_success)
                )
                self._exact_nonpenetration_restoration_selected_alpha.append(
                    float(restoration_history[-1, 7])
                    if len(restoration_history)
                    else 0.0
                )
                print(
                    "[ExactNonPenetration] Infeasible-start restoration "
                    f"frame={frame_idx} success={int(restoration_success)} "
                    f"iterations={len(restoration_history)} "
                    f"distance={start_minimum_distance:.9g}->"
                    f"{restoration_final_minimum:.9g} "
                    f"max_qpos_delta={restoration_max_delta:.9g}"
                )
        for iteration in range(max_iterations):
            q_before_iteration = np.asarray(q_n, dtype=np.float64).copy()
            q_before_iteration_locked = np.asarray(q_locked, dtype=np.float64).copy()
            q_before_iteration_locked[self.q_a_indices] = q_before_iteration[
                self.q_a_indices
            ]
            q_a_n_last = q_n[self.q_a_indices]
            q_candidate, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                target_robot_pts_local=target_robot_pts_local,
                hand_contact_targets_local=hand_contact_targets_local,
                hand_contact_valid=hand_contact_valid,
                hand_contact_weight_scale=hand_contact_weight_scale,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                foot_grounding_weight_scale=foot_grounding_weight_scale,
            )
            q_n = q_candidate
            candidate_minimum_distance = np.inf
            candidate_violation = 0.0
            backtrack_alpha = 1.0
            if exact_nonpenetration_enabled:
                (
                    candidate_violation,
                    candidate_minimum_distance,
                    exact_limiting_pair,
                ) = self._exact_nonpenetration_feasibility(q_candidate)
                exact_violation = candidate_violation
                exact_minimum_distance = candidate_minimum_distance
                if (
                    candidate_violation
                    > self.exact_nonpenetration_feasibility_tolerance
                ):
                    previous_violation, _, _ = self._exact_nonpenetration_feasibility(
                        q_before_iteration_locked
                    )
                    if (
                        previous_violation
                        <= self.exact_nonpenetration_feasibility_tolerance
                    ):
                        (
                            q_n,
                            exact_violation,
                            exact_minimum_distance,
                            exact_limiting_pair,
                            backtrack_alpha,
                        ) = self._backtrack_exact_nonpenetration(
                            q_before_iteration_locked, q_candidate
                        )
                        frame_backtrack_count += 1
                        frame_min_backtrack_alpha = min(
                            frame_min_backtrack_alpha, backtrack_alpha
                        )
                exact_nonpenetration_feasible = (
                    exact_violation
                    <= self.exact_nonpenetration_feasibility_tolerance
                )
                exact_sqp_history.append(
                    (
                        iteration + 1,
                        float(cost),
                        candidate_minimum_distance,
                        candidate_violation,
                        exact_minimum_distance,
                        exact_violation,
                        backtrack_alpha,
                    )
                )
            if terrain_enabled:
                top_violation, collision_violation = self._terrain_feasibility_violations(q_n)
                terrain_feasible = (
                    max(top_violation, collision_violation)
                    <= self.terrain_support_feasibility_tolerance
                )
                sqp_history.append(
                    (
                        iteration + 1,
                        float(cost),
                        top_violation,
                        collision_violation,
                        float(
                            getattr(
                                self,
                                "_last_terrain_linear_constraint_violation",
                                np.nan,
                            )
                        ),
                    )
                )
            cost_converged = np.isclose(cost, last_cost)
            if (
                terrain_feasible
                and exact_nonpenetration_feasible
                and (cost_converged or iteration + 1 >= n_iter)
            ):
                break
            if (
                not terrain_feasible
                and exact_nonpenetration_feasible
                and iteration + 1 >= n_iter
                and len(sqp_history) >= 5
            ):
                recent_residuals = np.asarray(sqp_history[-5:], dtype=np.float64)[:, 2:4]
                if float(np.max(np.ptp(recent_residuals, axis=0))) <= 1.0e-6:
                    break
            last_cost = cost
        if terrain_enabled and not terrain_feasible:
            (
                projected_q,
                projection_feasible,
                projection_history,
                projection_max_delta,
            ) = self._project_terrain_feasibility(q_n, q_t_last=q_t_last)
            self._terrain_failed_projection_history = np.asarray(
                projection_history, dtype=np.float64
            )
            if projection_feasible:
                q_n = projected_q
                top_violation, collision_violation = self._terrain_feasibility_violations(
                    q_n
                )
                terrain_feasible = True
                if (
                    self._terrain_projection_frames
                    and self._terrain_projection_frames[-1] == int(frame_idx)
                ):
                    self._terrain_projection_max_joint_delta[-1] = max(
                        self._terrain_projection_max_joint_delta[-1],
                        projection_max_delta,
                    )
                else:
                    self._terrain_projection_frames.append(int(frame_idx))
                    self._terrain_projection_max_joint_delta.append(
                        projection_max_delta
                    )
                print(
                    f"[TerrainSupport] Leg projection repaired frame {frame_idx}: "
                    f"max_joint_delta={projection_max_delta:.6g}"
                )
        if terrain_feasible:
            q_n, clip_delta = self._clip_final_configuration(
                q_n, q_t_last=q_t_last, init_t=init_t
            )
            if terrain_enabled:
                top_violation, collision_violation = self._terrain_feasibility_violations(
                    q_n
                )
                terrain_feasible = (
                    max(top_violation, collision_violation)
                    <= self.terrain_support_feasibility_tolerance
                )
                if not terrain_feasible:
                    (
                        q_n,
                        terrain_feasible,
                        projection_history,
                        projection_max_delta,
                    ) = self._project_terrain_feasibility(q_n, q_t_last=q_t_last)
                    self._terrain_failed_projection_history = np.asarray(
                        projection_history, dtype=np.float64
                    )
                    if terrain_feasible:
                        top_violation, collision_violation = (
                            self._terrain_feasibility_violations(q_n)
                        )
                        if (
                            self._terrain_projection_frames
                            and self._terrain_projection_frames[-1] == int(frame_idx)
                        ):
                            self._terrain_projection_max_joint_delta[-1] = max(
                                self._terrain_projection_max_joint_delta[-1],
                                projection_max_delta,
                            )
                        else:
                            self._terrain_projection_frames.append(int(frame_idx))
                            self._terrain_projection_max_joint_delta.append(
                                projection_max_delta
                            )
            if clip_delta > 1.0e-9:
                print(
                    f"[RetargetBounds] Clipped frame {frame_idx}: "
                    f"max_delta={clip_delta:.6g}"
                )
        if terrain_enabled and not terrain_feasible:
            self._terrain_failed_candidate_qpos = np.asarray(q_n, dtype=np.float64).copy()
            self._terrain_failed_sqp_history = np.asarray(sqp_history, dtype=np.float64)
            self._terrain_failed_details = self._terrain_feasibility_details(q_n)
            print(
                f"[TerrainSupport] Failure details: {self._terrain_failed_details}; "
                f"solver={getattr(self, '_last_solver_notes', 'unknown')}"
            )
            raise RuntimeError(
                f"Terrain constraints did not converge at frame {frame_idx} after "
                f"{max_iterations} SQP iterations: top_violation={top_violation:.6g}, "
                f"collision_violation={collision_violation:.6g}"
            )
        if exact_nonpenetration_enabled:
            (
                exact_violation,
                exact_minimum_distance,
                exact_limiting_pair,
            ) = self._exact_nonpenetration_feasibility(q_n)
            exact_nonpenetration_feasible = (
                exact_violation <= self.exact_nonpenetration_feasibility_tolerance
            )
            if not exact_nonpenetration_feasible:
                self._exact_nonpenetration_failed_candidate_qpos = np.asarray(
                    q_n, dtype=np.float64
                ).copy()
                self._exact_nonpenetration_failed_sqp_history = np.asarray(
                    exact_sqp_history, dtype=np.float64
                )
                self._exact_nonpenetration_failed_details = (
                    self._exact_nonpenetration_details(
                        exact_minimum_distance, exact_limiting_pair
                    )
                )
                print(
                    "[ExactNonPenetration] Failure details: "
                    f"{self._exact_nonpenetration_failed_details}; "
                    f"solver={getattr(self, '_last_solver_notes', 'unknown')}"
                )
                raise RuntimeError(
                    f"Exact nonlinear non-penetration did not converge at frame {frame_idx} "
                    f"after {max_iterations} SQP iterations: "
                    f"minimum_distance={exact_minimum_distance:.9g}, "
                    f"maximum_violation={exact_violation:.9g}"
                )
            if not hasattr(self, "_exact_nonpenetration_frame_iterations"):
                self._exact_nonpenetration_frame_iterations = []
                self._exact_nonpenetration_frame_min_distance_m = []
                self._exact_nonpenetration_frame_max_violation_m = []
                self._exact_nonpenetration_frame_backtrack_count = []
                self._exact_nonpenetration_frame_min_backtrack_alpha = []
            self._exact_nonpenetration_frame_iterations.append(len(exact_sqp_history))
            self._exact_nonpenetration_frame_min_distance_m.append(
                exact_minimum_distance
            )
            self._exact_nonpenetration_frame_max_violation_m.append(exact_violation)
            self._exact_nonpenetration_frame_backtrack_count.append(
                frame_backtrack_count
            )
            self._exact_nonpenetration_frame_min_backtrack_alpha.append(
                frame_min_backtrack_alpha
            )
        return q_n, cost

    def _resolve_hand_contact_links(self) -> dict[str, str]:
        candidates = {
            "left": ("left_rubber_hand_link", "left_sphere_hand_link", "left_hand_sphere_link", "left_hand_link"),
            "right": ("right_rubber_hand_link", "right_sphere_hand_link", "right_hand_sphere_link", "right_hand_link"),
        }
        resolved: dict[str, str] = {}
        for side, names in candidates.items():
            for name in names:
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id != -1:
                    resolved[side] = name
                    break
        return resolved

    def _init_hand_contact_surface_meshes(self) -> None:
        self._hand_contact_surface_meshes: dict[
            str, tuple[int, int, trimesh.Trimesh, trimesh.proximity.ProximityQuery]
        ] = {}
        if self.hand_contact_point_mode != "nearest_collision_surface":
            return
        for side, link_name in self.hand_contact_links.items():
            geom_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, link_name
            )
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name
            )
            if geom_id == -1 or body_id == -1:
                raise ValueError(
                    f"Nearest-surface hand contact requires named body and geom {link_name!r}"
                )
            if int(self.robot_model.geom_dataid[geom_id]) < 0:
                raise ValueError(
                    f"Nearest-surface hand contact requires mesh geom {link_name!r}"
                )
            vertices, faces = _mesh_local_vf(self.robot_model, geom_id)
            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int32),
                process=False,
            )
            if mesh.vertices.size == 0 or mesh.faces.size == 0:
                raise ValueError(f"Hand collision mesh is empty: {link_name!r}")
            self._hand_contact_surface_meshes[side] = (
                geom_id,
                body_id,
                mesh,
                trimesh.proximity.ProximityQuery(mesh),
            )
        missing = sorted(set(self.hand_contact_links) - set(self._hand_contact_surface_meshes))
        if missing:
            raise ValueError(
                f"Nearest-surface hand contact is missing collision meshes for {missing}"
            )

    def _hand_contact_point_offsets(
        self,
        q: np.ndarray,
        targets: np.ndarray,
        valid: np.ndarray,
        weight_scale: np.ndarray | None = None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.hand_contact_point_mode == "fixed_offset":
            return self.hand_contact_point_offset

        scales = self._normalize_hand_contact_weight_scales(weight_scale, 1)[0]
        links = self._active_hand_contact_links(valid, scales)
        self.robot_data.qpos[:] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        if self.object_name != "ground" and self.has_dynamic_object:
            object_position = np.asarray(q[-7:-4], dtype=np.float64)
            object_quaternion = np.asarray(q[-4:], dtype=np.float64)
            object_rotation = Rotation.from_quat(
                [
                    object_quaternion[1],
                    object_quaternion[2],
                    object_quaternion[3],
                    object_quaternion[0],
                ]
            ).as_matrix()
        else:
            object_position = np.zeros(3, dtype=np.float64)
            object_rotation = np.eye(3, dtype=np.float64)

        offsets: dict[str, np.ndarray] = {}
        for side, target_idx in (("left", 0), ("right", 1)):
            if side not in links:
                continue
            if side not in self._hand_contact_surface_meshes:
                raise ValueError(f"No collision surface cached for active {side} hand")
            geom_id, body_id, _, proximity = self._hand_contact_surface_meshes[side]
            target_local = np.asarray(targets[target_idx], dtype=np.float64)
            target_world = object_rotation @ target_local + object_position
            geom_rotation = self.robot_data.geom_xmat[geom_id].reshape(3, 3)
            geom_position = self.robot_data.geom_xpos[geom_id]
            target_geom = geom_rotation.T @ (target_world - geom_position)
            closest_geom, distances, _ = proximity.on_surface(target_geom[None])
            if (
                closest_geom.shape != (1, 3)
                or distances.shape != (1,)
                or not np.isfinite(closest_geom).all()
                or not np.isfinite(distances).all()
            ):
                raise RuntimeError(f"Invalid nearest hand-surface query for {side}")
            closest_world = geom_rotation @ closest_geom[0] + geom_position
            body_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
            body_position = self.robot_data.xpos[body_id]
            body_offset = body_rotation.T @ (closest_world - body_position)
            body_offset = np.asarray(body_offset, dtype=np.float64)
            offsets[side] = body_offset
            offsets[links[side]] = body_offset
        return offsets

    def _resolve_mapped_hand_links(self, mapping: dict[str, str]) -> dict[str, str]:
        aliases = {
            "left_rubber_hand_link": ("left_sphere_hand_link", "left_hand_sphere_link", "left_hand_link"),
            "right_rubber_hand_link": ("right_sphere_hand_link", "right_hand_sphere_link", "right_hand_link"),
            "left_sphere_hand_link": ("left_rubber_hand_link", "left_hand_sphere_link", "left_hand_link"),
            "right_sphere_hand_link": ("right_rubber_hand_link", "right_hand_sphere_link", "right_hand_link"),
        }
        resolved = dict(mapping)
        for source_joint, robot_link in mapping.items():
            if mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, robot_link) != -1:
                continue
            for alias in aliases.get(robot_link, ()):
                if mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, alias) != -1:
                    resolved[source_joint] = alias
                    break
        return resolved

    @staticmethod
    def _normalize_hand_contact_weight_scales(weight_scale, num_frames: int) -> np.ndarray:
        if weight_scale is None:
            scales = np.ones((num_frames, 2), dtype=float)
        else:
            scales = np.asarray(weight_scale, dtype=float)
            if scales.ndim == 0:
                scales = np.full((num_frames, 2), float(scales), dtype=float)
            elif scales.shape == (2,):
                scales = np.broadcast_to(scales[None, :], (num_frames, 2)).copy()
            elif scales.shape == (num_frames, 1):
                scales = np.repeat(scales, 2, axis=1)
            elif scales.shape != (num_frames, 2):
                raise ValueError(
                    f"Hand contact weight scale must be scalar or have shape (2,), (T, 1), or (T, 2); "
                    f"got {scales.shape} for T={num_frames}"
                )
        if not np.all(np.isfinite(scales)):
            raise ValueError("Hand contact weight scale must contain only finite values")
        if np.any(scales < 0.0):
            raise ValueError("Hand contact weight scale must be non-negative")
        return scales

    def _normalize_hand_contact_inputs(self, points, valid, num_frames: int, weight_scale=None):
        if not self.activate_hand_contact:
            return None, None, None
        if points is None or valid is None:
            print("[HandContact] enabled but no contact points were provided")
            return None, None, None
        points = np.asarray(points, dtype=float)
        valid = np.asarray(valid).astype(bool)
        if points.ndim != 3 or points.shape[1:] != (2, 3):
            raise ValueError(f"Hand contact points must have shape (T, 2, 3), got {points.shape}")
        if valid.shape != points.shape[:2]:
            raise ValueError(f"Hand contact valid shape {valid.shape} does not match points {points.shape[:2]}")
        if points.shape[0] == 0:
            raise ValueError("Hand contact inputs must contain at least one frame")

        weight_scale = self._normalize_hand_contact_weight_scales(weight_scale, points.shape[0])
        if points.shape[0] != num_frames:
            n = min(points.shape[0], num_frames)
            points = points[:n]
            valid = valid[:n]
            weight_scale = weight_scale[:n]
            if n < num_frames:
                pad_points = np.repeat(points[-1:], num_frames - n, axis=0)
                pad_valid = np.zeros((num_frames - n, 2), dtype=bool)
                pad_weight_scale = np.zeros((num_frames - n, 2), dtype=float)
                points = np.concatenate([points, pad_points], axis=0)
                valid = np.concatenate([valid, pad_valid], axis=0)
                weight_scale = np.concatenate([weight_scale, pad_weight_scale], axis=0)
        print(f"[HandContact] Loaded hand contact targets ({int(valid.sum())} valid points)")
        return points, valid, weight_scale

    def _replace_wrist_targets_with_contact(
        self,
        human_targets,
        contact_points,
        contact_valid,
        weight_scale=None,
    ):
        if not self.replace_source_wrist_with_contact or contact_points is None or contact_valid is None:
            return human_targets
        targets = np.array(human_targets, dtype=float, copy=True)
        scales = self._normalize_hand_contact_weight_scales(weight_scale, 1)[0]
        robot_link_keys = list(self.laplacian_match_links.keys())
        for side, joint_name, contact_idx in (("left", "L_Wrist", 0), ("right", "R_Wrist", 1)):
            if side not in self.hand_contact_links or not bool(contact_valid[contact_idx]):
                continue
            if joint_name in robot_link_keys:
                target_idx = robot_link_keys.index(joint_name)
                scale = float(scales[contact_idx])
                targets[target_idx] = (1.0 - scale) * targets[target_idx] + scale * contact_points[contact_idx]
        return targets

    def _active_hand_contact_links(self, valid, weight_scale=None):
        if valid is None:
            return {}
        scales = self._normalize_hand_contact_weight_scales(weight_scale, 1)[0]
        links = {}
        hard_mode = getattr(self, "hand_contact_mode", "soft") == "hard"
        left_active = not hard_mode or scales[0] >= self.HARD_CONTACT_MIN_WEIGHT_SCALE
        right_active = not hard_mode or scales[1] >= self.HARD_CONTACT_MIN_WEIGHT_SCALE
        if len(valid) > 0 and bool(valid[0]) and left_active and "left" in self.hand_contact_links:
            links["left"] = self.hand_contact_links["left"]
        if len(valid) > 1 and bool(valid[1]) and right_active and "right" in self.hand_contact_links:
            links["right"] = self.hand_contact_links["right"]
        return links

    def _build_hand_contact_residuals(
        self,
        q,
        dqa,
        targets,
        valid,
        weight_scale=None,
        point_offsets=None,
    ):
        if (not self.activate_hand_contact) or targets is None or valid is None:
            return []
        scales = self._normalize_hand_contact_weight_scales(weight_scale, 1)[0]
        links = self._active_hand_contact_links(valid, scales)
        if not links:
            return []
        if point_offsets is None:
            point_offsets = self._hand_contact_point_offsets(
                q, targets, valid, weight_scale
            )
        J_dict, p_dict, _ = self._calc_manipulator_jacobians(
            q,
            links=links,
            obj_frame=(self.object_name != "ground"),
            point_offsets=point_offsets,
        )
        residuals = []
        for side, idx in (("left", 0), ("right", 1)):
            if side not in links:
                continue
            residual = J_dict[side] @ dqa + p_dict[side] - np.asarray(targets[idx], dtype=float)
            residuals.append((idx, float(scales[idx]), residual))
        return residuals

    def _soft_hand_contact_objective_terms(self, contact_residuals):
        return [
            self.hand_contact_weight * scale * cp.sum_squares(residual)
            for _, scale, residual in contact_residuals
            if scale > 0.0
        ]

    def _record_hand_contact_error(
        self, q, targets, valid, error_row, weight_scale=None, point_offset_row=None
    ):
        if (not self.activate_hand_contact) or targets is None or valid is None:
            return
        links = self._active_hand_contact_links(valid, weight_scale)
        if not links:
            return
        point_offsets = self._hand_contact_point_offsets(
            q, targets, valid, weight_scale
        )
        _, p_dict, _ = self._calc_manipulator_jacobians(
            q,
            links=links,
            obj_frame=(self.object_name != "ground"),
            point_offsets=point_offsets,
        )
        for side, idx in (("left", 0), ("right", 1)):
            if side in links:
                error_row[idx] = float(np.linalg.norm(p_dict[side] - targets[idx]))
                if point_offset_row is not None:
                    if isinstance(point_offsets, dict):
                        point_offset_row[idx] = point_offsets[side]
                    else:
                        point_offset_row[idx] = point_offsets

    def _save_results(
        self,
        dest_res_path,
        qpos,
        human_joints,
        cost,
        hand_contact_points_local=None,
        hand_contact_valid=None,
        hand_contact_error=None,
        hand_contact_point_offsets_local=None,
        partial: bool = False,
        failed_frame: int = -1,
        error: str = "",
        hand_contact_weight_scale=None,
    ):
        payload = {
            "qpos": qpos,
            "human_joints": human_joints,
            "fps": 30,
            "cost": cost,
            "retarget_solver_policy": np.asarray("strict_clarabel_optimal_only"),
            "retarget_partial_checkpoint_interval_frames": np.asarray(
                int(getattr(self, "partial_checkpoint_interval_frames", 0)),
                dtype=np.int32,
            ),
            "retarget_exact_nonpenetration_enabled": np.asarray(
                bool(getattr(self, "enforce_exact_nonpenetration", False))
            ),
            "retarget_exact_nonpenetration_revision": np.asarray(
                "post_sqp_mujoco_geom_distance_locked_qpos_restoration_v4"
            ),
            "retarget_exact_nonpenetration_max_sqp_iterations": np.asarray(
                int(getattr(self, "exact_nonpenetration_max_sqp_iterations", 0)),
                dtype=np.int32,
            ),
            "retarget_exact_nonpenetration_feasibility_tolerance_m": np.asarray(
                float(
                    getattr(
                        self,
                        "exact_nonpenetration_feasibility_tolerance",
                        np.nan,
                    )
                ),
                dtype=np.float64,
            ),
            "retarget_exact_nonpenetration_interior_margin_m": np.asarray(
                float(
                    getattr(
                        self,
                        "exact_nonpenetration_interior_margin",
                        0.0,
                    )
                ),
                dtype=np.float64,
            ),
            "retarget_exact_nonpenetration_qp_safety_margin_m": np.asarray(
                float(
                    getattr(
                        self,
                        "exact_nonpenetration_qp_safety_margin",
                        0.0,
                    )
                ),
                dtype=np.float64,
            ),
            "retarget_exact_nonpenetration_restore_infeasible_start": np.asarray(
                bool(
                    getattr(
                        self,
                        "exact_nonpenetration_restore_infeasible_start",
                        False,
                    )
                )
            ),
            "retarget_exact_nonpenetration_restoration_max_iterations": np.asarray(
                int(
                    getattr(
                        self,
                        "exact_nonpenetration_restoration_max_iterations",
                        0,
                    )
                ),
                dtype=np.int32,
            ),
            "retarget_exact_nonpenetration_backtracking_steps": np.asarray(
                int(getattr(self, "exact_nonpenetration_backtracking_steps", 0)),
                dtype=np.int32,
            ),
            "retarget_penetration_tolerance_m": np.asarray(
                float(getattr(self, "penetration_tolerance", np.nan)),
                dtype=np.float64,
            ),
            "retarget_initial_robot_grounding_enabled": np.asarray(
                bool(getattr(self, "ground_initial_robot", False))
            ),
            "retarget_initial_ground_clearance_target_m": np.asarray(
                float(getattr(self, "initial_ground_clearance", 0.0)), dtype=np.float32
            ),
            "retarget_initial_grounding_offset_m": np.asarray(
                float(getattr(self, "_initial_grounding_offset_m", 0.0)), dtype=np.float32
            ),
            "retarget_initial_grounding_clearance_before_m": np.asarray(
                float(getattr(self, "_initial_grounding_clearance_before_m", np.nan)),
                dtype=np.float32,
            ),
            "retarget_initial_grounding_clearance_after_m": np.asarray(
                float(getattr(self, "_initial_grounding_clearance_after_m", np.nan)),
                dtype=np.float32,
            ),
            "retarget_foot_sticking_pin_z": np.asarray(
                bool(getattr(self, "foot_sticking_pin_z", False))
            ),
            "retarget_foot_sticking_z_floor_m": np.asarray(
                float(getattr(self, "foot_sticking_z_floor", 0.005)), dtype=np.float32
            ),
            "retarget_foot_grounding_weight": np.asarray(
                float(getattr(self, "foot_grounding_weight", 0.0)), dtype=np.float32
            ),
            "retarget_foot_grounding_mode": np.asarray(
                str(getattr(self, "foot_grounding_mode", "sticking"))
            ),
            "retarget_foot_grounding_schedule": np.asarray(
                str(getattr(self, "foot_grounding_schedule", "all_frames"))
            ),
            "retarget_foot_grounding_ramp_frames": np.asarray(
                int(getattr(self, "foot_grounding_ramp_frames", 0)), dtype=np.int32
            ),
            "retarget_foot_grounding_contact_start_idx": np.asarray(
                int(getattr(self, "_foot_grounding_contact_start_idx", -1)), dtype=np.int32
            ),
            "retarget_source_ground_reference_mode": np.asarray(
                str(getattr(self, "source_ground_reference_mode", "unknown"))
            ),
            "retarget_source_ground_z_m": np.asarray(
                float(getattr(self, "source_ground_z_m", np.nan)), dtype=np.float32
            ),
            "retarget_foot_sticking_mode": np.asarray(
                str(getattr(self, "source_foot_sticking_mode", "unknown"))
            ),
            "retarget_replace_source_wrist_with_contact": np.asarray(
                bool(getattr(self, "replace_source_wrist_with_contact", False))
            ),
        }
        if getattr(self, "enforce_exact_nonpenetration", False):
            exact_iterations = np.asarray(
                self._exact_nonpenetration_frame_iterations, dtype=np.int32
            )
            exact_minimum_distances = np.asarray(
                self._exact_nonpenetration_frame_min_distance_m, dtype=np.float64
            )
            exact_maximum_violations = np.asarray(
                self._exact_nonpenetration_frame_max_violation_m, dtype=np.float64
            )
            exact_backtrack_counts = np.asarray(
                self._exact_nonpenetration_frame_backtrack_count, dtype=np.int32
            )
            exact_min_backtrack_alphas = np.asarray(
                self._exact_nonpenetration_frame_min_backtrack_alpha, dtype=np.float64
            )
            if not (
                len(exact_iterations)
                == len(exact_minimum_distances)
                == len(exact_maximum_violations)
                == len(exact_backtrack_counts)
                == len(exact_min_backtrack_alphas)
                == len(qpos)
            ):
                raise RuntimeError(
                    "Exact non-penetration provenance length disagrees with saved qpos: "
                    f"iterations={len(exact_iterations)}, distances={len(exact_minimum_distances)}, "
                    f"violations={len(exact_maximum_violations)}, "
                    f"backtracks={len(exact_backtrack_counts)}, "
                    f"alphas={len(exact_min_backtrack_alphas)}, qpos={len(qpos)}"
                )
            payload.update(
                {
                    "retarget_exact_nonpenetration_sqp_iterations": exact_iterations,
                    "retarget_exact_nonpenetration_min_distance_m": exact_minimum_distances,
                    "retarget_exact_nonpenetration_max_violation_m": exact_maximum_violations,
                    "retarget_exact_nonpenetration_backtrack_count": exact_backtrack_counts,
                    "retarget_exact_nonpenetration_min_backtrack_alpha": exact_min_backtrack_alphas,
                    "retarget_exact_nonpenetration_restoration_frames": np.asarray(
                        getattr(self, "_exact_nonpenetration_restoration_frames", []),
                        dtype=np.int32,
                    ),
                    "retarget_exact_nonpenetration_restoration_iterations": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_iterations",
                            [],
                        ),
                        dtype=np.int32,
                    ),
                    "retarget_exact_nonpenetration_restoration_start_min_distance_m": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_start_min_distance_m",
                            [],
                        ),
                        dtype=np.float64,
                    ),
                    "retarget_exact_nonpenetration_restoration_final_min_distance_m": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_final_min_distance_m",
                            [],
                        ),
                        dtype=np.float64,
                    ),
                    "retarget_exact_nonpenetration_restoration_max_qpos_delta": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_max_qpos_delta",
                            [],
                        ),
                        dtype=np.float64,
                    ),
                    "retarget_exact_nonpenetration_restoration_success": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_success",
                            [],
                        ),
                        dtype=bool,
                    ),
                    "retarget_exact_nonpenetration_restoration_selected_alpha": np.asarray(
                        getattr(
                            self,
                            "_exact_nonpenetration_restoration_selected_alpha",
                            [],
                        ),
                        dtype=np.float64,
                    ),
                }
            )
        if getattr(self, "_terrain_support_enabled", False):
            payload.update(
                {
                    "retarget_terrain_support_revision": np.asarray(
                        "crisp-top-plane-component-mapped-sqp-leg-projection-v3"
                    ),
                    "retarget_terrain_support_mesh_file": np.asarray(self.terrain_support_mesh_file),
                    "retarget_terrain_support_mesh_scale": np.asarray(
                        self.terrain_support_mesh_scale, dtype=np.float32
                    ),
                    "retarget_terrain_support_min_normal_z": np.asarray(
                        self.terrain_support_min_normal_z, dtype=np.float32
                    ),
                    "retarget_terrain_support_clearance_m": np.asarray(
                        self.terrain_support_clearance, dtype=np.float32
                    ),
                    "retarget_terrain_support_sphere_radius_m": np.asarray(
                        self.terrain_support_sphere_radius, dtype=np.float32
                    ),
                    "retarget_terrain_support_max_sqp_iterations": np.asarray(
                        self.terrain_support_max_sqp_iterations, dtype=np.int32
                    ),
                    "retarget_terrain_support_feasibility_tolerance_m": np.asarray(
                        self.terrain_support_feasibility_tolerance, dtype=np.float32
                    ),
                    "retarget_terrain_collision_geom_prefix": np.asarray(
                        self.terrain_collision_geom_prefix
                    ),
                    "retarget_terrain_collision_foot_only": np.asarray(
                        self.terrain_collision_foot_only
                    ),
                    "retarget_max_frame_root_translation_m": np.asarray(
                        self.max_frame_root_translation, dtype=np.float64
                    ),
                    "retarget_max_frame_root_quaternion_delta": np.asarray(
                        self.max_frame_root_quaternion_delta, dtype=np.float64
                    ),
                    "retarget_max_frame_joint_delta_rad": np.asarray(
                        self.max_frame_joint_delta, dtype=np.float64
                    ),
                    "retarget_terrain_projection_frames": np.asarray(
                        self._terrain_projection_frames, dtype=np.int32
                    ),
                    "retarget_terrain_projection_max_joint_delta_rad": np.asarray(
                        self._terrain_projection_max_joint_delta, dtype=np.float64
                    ),
                }
            )
        if partial:
            payload["retarget_partial"] = np.asarray(True)
            payload["retarget_failed_frame"] = np.asarray(int(failed_frame))
            payload["retarget_error"] = np.asarray(error)
            failed_candidate = getattr(self, "_terrain_failed_candidate_qpos", None)
            failed_history = getattr(self, "_terrain_failed_sqp_history", None)
            if failed_candidate is not None:
                payload["retarget_failed_candidate_qpos"] = np.asarray(
                    failed_candidate, dtype=np.float64
                )
            if failed_history is not None:
                payload["retarget_failed_sqp_history"] = np.asarray(
                    failed_history, dtype=np.float64
                )
                payload["retarget_failed_sqp_history_columns"] = np.asarray(
                    [
                        "iteration",
                        "cost",
                        "top_violation_m",
                        "collision_violation_m",
                        "terrain_linear_constraint_violation_m",
                    ]
                )
            payload["retarget_failed_details"] = np.asarray(
                getattr(self, "_terrain_failed_details", "")
            )
            exact_failed_candidate = getattr(
                self, "_exact_nonpenetration_failed_candidate_qpos", None
            )
            exact_failed_history = getattr(
                self, "_exact_nonpenetration_failed_sqp_history", None
            )
            if exact_failed_candidate is not None:
                payload["retarget_exact_nonpenetration_failed_candidate_qpos"] = np.asarray(
                    exact_failed_candidate, dtype=np.float64
                )
            if exact_failed_history is not None:
                payload["retarget_exact_nonpenetration_failed_sqp_history"] = np.asarray(
                    exact_failed_history, dtype=np.float64
                )
                payload["retarget_exact_nonpenetration_failed_sqp_history_columns"] = np.asarray(
                    [
                        "iteration",
                        "cost",
                        "candidate_minimum_distance_m",
                        "candidate_maximum_violation_m",
                        "accepted_minimum_distance_m",
                        "accepted_maximum_violation_m",
                        "backtrack_alpha",
                    ]
                )
            payload["retarget_exact_nonpenetration_failed_details"] = np.asarray(
                getattr(self, "_exact_nonpenetration_failed_details", "")
            )
            restoration_history = np.asarray(
                getattr(
                    self,
                    "_exact_nonpenetration_last_restoration_history",
                    np.empty((0, 8), dtype=np.float64),
                ),
                dtype=np.float64,
            )
            if restoration_history.size:
                payload["retarget_exact_nonpenetration_failed_restoration_history"] = (
                    restoration_history
                )
                payload[
                    "retarget_exact_nonpenetration_failed_restoration_history_columns"
                ] = np.asarray(
                    [
                        "iteration",
                        "start_minimum_distance_m",
                        "start_maximum_violation_m",
                        "candidate_minimum_distance_m",
                        "candidate_maximum_violation_m",
                        "selected_minimum_distance_m",
                        "selected_maximum_violation_m",
                        "selected_alpha",
                    ]
                )
            failed_projection_history = getattr(
                self, "_terrain_failed_projection_history", None
            )
            if failed_projection_history is not None:
                payload["retarget_failed_projection_history"] = np.asarray(
                    failed_projection_history, dtype=np.float64
                )
                payload["retarget_failed_projection_history_columns"] = np.asarray(
                    ["iteration", "top_violation_m", "collision_violation_m"]
                )
        initial_qpos_file = getattr(self, "_initial_qpos_file_resolved", "")
        if initial_qpos_file:
            payload["retarget_initial_qpos_file"] = np.asarray(initial_qpos_file)
            payload["retarget_initial_qpos_applied"] = np.asarray(True)
        resume_partial_file = getattr(self, "resume_partial_file", "")
        if resume_partial_file:
            payload["retarget_resume_partial_file"] = np.asarray(
                str(Path(resume_partial_file).expanduser().resolve())
            )
            payload["retarget_resume_prefix_frames"] = np.asarray(
                int(getattr(self, "_resume_prefix_frames", 0)), dtype=np.int32
            )
        if self.activate_hand_contact and hand_contact_points_local is not None and hand_contact_valid is not None:
            err = np.asarray(hand_contact_error, dtype=np.float32)
            point_mode = str(
                getattr(self, "hand_contact_point_mode", "fixed_offset")
            )
            if hand_contact_point_offsets_local is None:
                if point_mode != "fixed_offset":
                    raise ValueError(
                        "Nearest-surface hand contact results require recorded per-frame "
                        "hand_contact_point_offsets_local"
                    )
                fixed_offset = np.asarray(
                    self.hand_contact_point_offset, dtype=np.float32
                )
                point_offsets = np.broadcast_to(
                    fixed_offset, (len(qpos), 2, 3)
                ).copy()
            else:
                point_offsets = np.asarray(
                    hand_contact_point_offsets_local, dtype=np.float32
                )
            expected_offset_shape = (len(qpos), 2, 3)
            if point_offsets.shape != expected_offset_shape:
                raise ValueError(
                    "Saved hand contact point offsets disagree with qpos: "
                    f"{point_offsets.shape} vs {expected_offset_shape}"
                )
            finite = np.isfinite(err)
            if hand_contact_weight_scale is None:
                hand_contact_weight_scale = np.ones(np.asarray(hand_contact_valid).shape, dtype=float)
            payload.update(
                {
                    "hand_contact_points_local": np.asarray(hand_contact_points_local, dtype=np.float32),
                    "hand_contact_valid": np.asarray(hand_contact_valid).astype(bool),
                    "hand_contact_weight_scale": np.asarray(hand_contact_weight_scale, dtype=np.float32),
                    "hand_contact_error_m": err,
                    "hand_contact_mode": np.asarray(self.hand_contact_mode),
                    "hand_contact_point_mode": np.asarray(point_mode),
                    "hand_contact_tolerance": np.asarray(np.float32(self.hand_contact_tolerance)),
                    "hand_contact_point_offset": np.asarray(self.hand_contact_point_offset, dtype=np.float32),
                    "hand_contact_point_offsets_local": point_offsets,
                    "hand_contact_mean_error_m": np.asarray(
                        np.float32(np.nanmean(err)) if finite.any() else np.float32(np.nan)
                    ),
                    "hand_contact_max_error_m": np.asarray(
                        np.float32(np.nanmax(err)) if finite.any() else np.float32(np.nan)
                    ),
                }
            )
        destination = Path(dest_res_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}.tmp-{os.getpid()}.npz"
        )
        try:
            np.savez(temporary, **payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]
        self._geom_body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "" for g in range(ngeom)
        ]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(
        self, q: np.ndarray, compute_jacobians: bool = True
    ):
        """Evaluate active collision distances and optionally their Jacobians."""
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            terrain_1 = bool(self.terrain_collision_geom_prefix) and self._geom_names[g1].startswith(
                self.terrain_collision_geom_prefix
            )
            terrain_2 = bool(self.terrain_collision_geom_prefix) and self._geom_names[g2].startswith(
                self.terrain_collision_geom_prefix
            )
            if terrain_1 or terrain_2:
                if terrain_1 and terrain_2:
                    return False
                other = g2 if terrain_1 else g1
                if self.object_name in self._geom_names[other]:
                    return False
                foot_name = f"{self._geom_names[other]} {self._geom_body_names[other]}"
                is_foot_sphere = "ankle_roll_sphere_" in foot_name
                if self.terrain_collision_foot_only and not is_foot_sphere:
                    return False
                if is_foot_sphere and getattr(self, "_terrain_support_enabled", False):
                    supports = self._terrain_support_planes(d.geom_xpos[other], other)
                    terrain_geom = g1 if terrain_1 else g2
                    terrain_component = self._terrain_support_geom_component_ids.get(
                        terrain_geom
                    )
                    if any(terrain_component == support[3] for support in supports):
                        return False
                return True
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                if compute_jacobians:
                    J_rel = self._compute_jacobian_for_contact_relative(
                        m.geom(g1),
                        m.geom(g2),
                        self._geom_names[g1],
                        self._geom_names[g2],
                        fromto,
                        dist,
                    )
                    Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | dict[str, np.ndarray] | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name!r} for mapped joint {name!r} not found in MuJoCo model")

            if isinstance(point_offsets, dict):
                pC_B = point_offsets.get(name, point_offsets.get(link_name, np.zeros(3)))
            elif point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            body_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
            pos_world = self.robot_data.xpos[body_id] + body_rotation @ pC_B

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
