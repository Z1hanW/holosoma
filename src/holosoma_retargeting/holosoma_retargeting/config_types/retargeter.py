"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Example: {"L_Toe": [(30, 60)], "R_Toe": [(10, 20), (80, 95)]}"""

    z_floor: float = 0.0
    """Floor height used by Z pinning constraints."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce object non-penetration constraints."""

    terrain_collision_geom_prefix: str = ""
    """Static terrain geom prefix. Empty disables terrain-specific constraints."""

    terrain_collision_foot_only: bool = True
    """Whether terrain constraints are limited to G1 foot collision spheres."""

    terrain_support_mesh_file: str = ""
    """CRISP terrain mesh used to build upward foot support-plane constraints."""

    terrain_support_mesh_scale: float = 1.0
    """Uniform scale applied to the terrain support mesh."""

    terrain_support_min_normal_z: float = 0.9
    """Minimum upward face-normal component for a walkable support surface."""

    terrain_support_clearance: float = 2e-4
    """Clearance above a terrain support plane in addition to the sphere radius."""

    terrain_support_sphere_radius: float = 0.005
    """Radius of each G1 foot collision sphere."""

    terrain_support_activation_margin: float = 0.1
    """Maximum height above a support plane at which its constraint is activated."""

    terrain_support_max_sqp_iterations: int = 50
    """Maximum per-frame SQP iterations while terrain constraints remain infeasible."""

    terrain_support_feasibility_tolerance: float = 5e-5
    """Allowed final nonlinear terrain-constraint residual in meters."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    ground_initial_robot: bool = False
    """Whether to lower the initial robot root until a foot collision sphere touches the floor."""

    initial_ground_clearance: float = 0.0
    """Target minimum foot collision clearance after initial grounding, in meters."""

    foot_sticking_pin_z: bool = False
    """Whether a sticking foot is also pinned to a configured floor height in Z."""

    foot_sticking_z_floor: float = 0.005
    """World Z for the sticking toe-sphere center; 0.005 m places a 5 mm sphere on z=0."""

    foot_grounding_weight: float = 0.0
    """Soft Z-grounding objective weight for feet classified as sticking; zero disables it."""

    foot_grounding_mode: str = "sticking"
    """Soft grounding policy: sticking, continuous_support, continuous_lowest_sphere, or lowest_foot."""

    foot_grounding_schedule: str = "all_frames"
    """Soft grounding frame schedule: all_frames or before_contact."""

    foot_grounding_ramp_frames: int = 0
    """Frames over which before_contact grounding weight linearly decays to zero at t1."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    enforce_exact_nonpenetration: bool = False
    """Require the final nonlinear MuJoCo distances to satisfy the penetration limit."""

    exact_nonpenetration_max_sqp_iterations: int = 50
    """Maximum per-frame SQP iterations while exact non-penetration remains infeasible."""

    exact_nonpenetration_feasibility_tolerance: float = 1e-6
    """Numerical residual allowed beyond the configured penetration tolerance."""

    exact_nonpenetration_interior_margin: float = 0.0
    """Extra nonlinear acceptance margin inside the penetration limit."""

    exact_nonpenetration_qp_safety_margin: float = 0.0
    """Additional linearized-QP margin beyond the nonlinear acceptance target."""

    exact_nonpenetration_restore_infeasible_start: bool = False
    """Run an explicit feasibility restoration when object motion makes the prior qpos infeasible."""

    exact_nonpenetration_restoration_max_iterations: int = 20
    """Maximum collision-only SQP projections used by infeasible-start restoration."""

    exact_nonpenetration_backtracking_steps: int = 24
    """Bisection steps used to keep an SQP update inside the nonlinear feasible set."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    max_frame_root_translation: float = 0.0
    """Maximum floating-base translation change per frame; zero disables it."""

    max_frame_root_quaternion_delta: float = 0.0
    """Maximum root-quaternion Euclidean change per frame; zero disables it."""

    max_frame_joint_delta: float = 0.0
    """Maximum absolute actuated-joint change per frame; zero disables it."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""

    w_keypoint_tracking: float = 0.0
    """Weight for direct mapped robot-link to human-joint tracking cost."""

    activate_hand_contact: bool = False
    """Whether to add hand-to-object contact anchoring from staged contact points."""

    hand_contact_mode: str = "soft"
    """Hand contact mode: 'soft' adds a weighted objective, 'hard' adds box constraints."""

    hand_contact_weight: float = 0.0
    """Weight for the soft hand contact objective."""

    hand_contact_tolerance: float = 0.0
    """Per-axis tolerance for hard hand contact constraints."""

    hand_contact_point_offset: tuple[float, float, float] = (0.09, 0.0, 0.0)
    """Point on each hand link, in the hand body frame, used for contact anchoring."""

    hand_contact_point_mode: str = "fixed_offset"
    """Hand point policy: fixed_offset or nearest_collision_surface."""

    replace_source_wrist_with_contact: bool = False
    """Whether to replace source wrist vertices with contact targets in the Laplacian source mesh."""

    save_partial_on_failure: bool = False
    """Whether to save partial qpos if retargeting fails part-way through."""

    partial_checkpoint_interval_frames: int = 0
    """Write an atomic resumable partial result every N accepted frames; 0 disables it."""

    resume_partial_file: str = ""
    """Partial retarget NPZ whose accepted prefix should be resumed."""

    initial_qpos_file: str = ""
    """One-frame qpos NPZ used only as the frame-0 SQP initialization."""

    activate_source_global_offset: bool = False
    """Compatibility flag for older CRISP scripts; currently parsed but not applied."""

    source_global_offset_weight: float = 0.0
    """Compatibility knob for older CRISP source-global-offset experiments."""

    source_global_offset_smooth_weight: float = 0.0
    """Compatibility knob for older CRISP source-global-offset experiments."""

    source_global_offset_max: float = 0.0
    """Compatibility knob for older CRISP source-global-offset experiments."""
