"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class SparseObjectGoalConfig:
    """Sparse object-goal and command-observation curriculum configuration."""

    enabled: bool = False
    """Enable mixed sparse-goal sampling and command-only observation curriculum."""

    command_only_env_prob_start: float = 0.0
    """Initial probability of sampling command-only observation episodes."""

    command_only_env_prob_end: float = 0.0
    """Final probability of sampling command-only observation episodes."""

    command_only_env_prob_start_iter: int | None = None
    """Training iteration where the command-only probability schedule activates."""

    command_only_env_prob_end_iter: int | None = None
    """Training iteration where the command-only probability schedule reaches its final value."""

    command_only_env_prob_ramp_resets: int | None = None
    """Optional ramp horizon for command-only episode probability; defaults to external_goal_prob_ramp_resets."""

    eval_command_only_env_prob: float | None = None
    """Optional command-only episode probability used during evaluation; defaults to end prob when None."""

    clip_goal_delta_min_steps: int = 60
    """Legacy clip-goal waypoint setting kept for compatibility; final clip-goal placement no longer uses it."""

    clip_goal_delta_max_steps: int = 180
    """Legacy clip-goal waypoint setting kept for compatibility; final clip-goal placement no longer uses it."""

    external_goal_prob_start: float = 0.0
    """Initial probability of sampling external random goals."""

    external_goal_prob_end: float = 1.0
    """Final probability of sampling external random goals."""

    external_goal_prob_start_iter: int | None = None
    """Training iteration where the external-goal probability schedule activates."""

    external_goal_prob_end_iter: int | None = None
    """Training iteration where the external-goal probability schedule reaches its final value."""

    external_goal_prob_ramp_resets: int = 200000
    """Linear ramp horizon in reset-events (env-wise), from start prob to end prob."""

    eval_external_goal_prob: float | None = None
    """Optional external-goal probability used during evaluation; defaults to end prob when None."""

    carry_extension_prob_start: float = 0.0
    """Initial probability of sampling clip-anchored carry-extension goals."""

    carry_extension_prob_end: float = 0.0
    """Final probability of sampling clip-anchored carry-extension goals."""

    carry_extension_prob_start_iter: int | None = None
    """Training iteration where the carry-extension probability schedule activates."""

    carry_extension_prob_end_iter: int | None = None
    """Training iteration where the carry-extension probability schedule reaches its final value."""

    carry_extension_prob_ramp_resets: int | None = None
    """Optional ramp horizon for carry-extension goal probability; defaults to external_goal_prob_ramp_resets."""

    eval_carry_extension_prob: float | None = None
    """Optional carry-extension probability used during evaluation; defaults to end prob when None."""

    external_goal_range_ramp_resets: int | None = None
    """Optional ramp horizon for external-goal sampling range; defaults to external_goal_prob_ramp_resets."""

    external_goal_range_start_iter: int | None = None
    """Training iteration where the external-goal range curriculum activates."""

    external_goal_range_end_iter: int | None = None
    """Training iteration where the external-goal range curriculum reaches its final span."""

    external_goal_sampling_mode: str = "box"
    """XY sampling mode for external goals: 'box' or 'annulus'."""

    external_goal_radius_min_start: float | None = None
    """Optional initial inner radius for annulus external-goal sampling in local XY."""

    external_goal_radius_max_start: float | None = None
    """Optional initial outer radius for annulus external-goal sampling in local XY."""

    external_goal_radius_min: float = 1.0
    """Final inner radius for annulus external-goal sampling in local XY."""

    external_goal_radius_max: float = 3.4
    """Final outer radius for annulus external-goal sampling in local XY."""

    carry_extension_range_ramp_resets: int | None = None
    """Optional ramp horizon for carry-extension sampling range; defaults to carry_extension_prob_ramp_resets."""

    carry_extension_range_start_iter: int | None = None
    """Training iteration where the carry-extension range curriculum activates."""

    carry_extension_range_end_iter: int | None = None
    """Training iteration where the carry-extension range curriculum reaches its final span."""

    external_goal_pos_local_min_start: list[float] | None = None
    """Optional initial lower bounds [x, y, z] for external goal position in local frame around env origin."""

    external_goal_pos_local_max_start: list[float] | None = None
    """Optional initial upper bounds [x, y, z] for external goal position in local frame around env origin."""

    external_goal_pos_local_min: list[float] = field(default_factory=lambda: [1.0, -0.8, 0.7])
    """External goal local position lower bounds [x, y, z]; in annulus mode only z is used for position sampling."""

    external_goal_pos_local_max: list[float] = field(default_factory=lambda: [1.75, 0.8, 1.0])
    """External goal local position upper bounds [x, y, z]; in annulus mode only z is used for position sampling."""

    carry_extension_pos_local_min_start: list[float] | None = None
    """Optional initial lower bounds [x, y, z] for carry-extension goal in the clip-final object frame."""

    carry_extension_pos_local_max_start: list[float] | None = None
    """Optional initial upper bounds [x, y, z] for carry-extension goal in the clip-final object frame."""

    carry_extension_pos_local_min: list[float] = field(default_factory=lambda: [0.10, -0.10, 0.0])
    """Carry-extension goal lower bounds [x, y, z] in the clip-final object frame."""

    carry_extension_pos_local_max: list[float] = field(default_factory=lambda: [0.50, 0.10, 0.0])
    """Carry-extension goal upper bounds [x, y, z] in the clip-final object frame."""

    external_goal_rpy_min_start: list[float] | None = None
    """Optional initial orientation lower bounds [roll, pitch, yaw] in radians."""

    external_goal_rpy_max_start: list[float] | None = None
    """Optional initial orientation upper bounds [roll, pitch, yaw] in radians."""

    external_goal_rpy_min: list[float] = field(default_factory=lambda: [0.0, 0.0, -3.1415926])
    """External goal orientation lower bounds [roll, pitch, yaw] in radians."""

    external_goal_rpy_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 3.1415926])
    """External goal orientation upper bounds [roll, pitch, yaw] in radians."""

    carry_extension_rpy_min_start: list[float] | None = None
    """Optional initial orientation lower bounds [roll, pitch, yaw] relative to the clip-final object pose."""

    carry_extension_rpy_max_start: list[float] | None = None
    """Optional initial orientation upper bounds [roll, pitch, yaw] relative to the clip-final object pose."""

    carry_extension_rpy_min: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.35])
    """Carry-extension orientation lower bounds [roll, pitch, yaw] relative to the clip-final object pose."""

    carry_extension_rpy_max: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.35])
    """Carry-extension orientation upper bounds [roll, pitch, yaw] relative to the clip-final object pose."""


@dataclass(frozen=True)
class CleanNoisyClipCurriculumConfig:
    """Iteration-driven clean/noisy clip mixing schedule."""

    enabled: bool = False
    """Enable clip-group sampling curriculum."""

    clean_clip_name_prefixes: list[str] = field(default_factory=lambda: ["sub"])
    """Clip-name prefixes treated as clean examples. Other clips are treated as noisy."""

    stage_start_iterations: list[int] = field(default_factory=lambda: [0, 1500, 2000, 2500, 3000, 4000])
    """Iteration milestones for the piecewise-constant clean/noisy sampling schedule."""

    clean_group_probabilities: list[float] = field(default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    """Target total sampling probability assigned to the clean clip group at each milestone."""


@dataclass(frozen=True)
class FixedClipGroupAssignmentConfig:
    """Fixed env-to-clip-group assignment for mixed motion banks."""

    enabled: bool = False
    """Enable fixed environment assignment to a named clip group and its complement."""

    group_clip_name_prefixes: list[str] = field(default_factory=lambda: ["sub"])
    """Clip-name prefixes for the fixed group. Non-matching clips form the complement group."""

    group_env_fraction: float = 0.25
    """Fraction of environments assigned to the fixed group."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz or .h5) that contains motion clips to track."""

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_clip_id: int | None = None
    """Optional clip index inside an HDF5 motion bank."""

    motion_clip_name: str | None = None
    """Optional clip name inside an HDF5 motion bank (overrides motion_clip_id)."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    adaptive_sampling_contact_interval_root: str | None = None
    """Optional root directory of exported per-clip ``contact_intervals.json`` files.

    When set, adaptive timestep sampling logs additional contact-relative probability masses
    to W&B/TensorBoard, and uniform ``t1`` window sampling can use the same bank. The bank is
    matched by clip id, so it can live outside the motion bank.
    """

    uniform_t1_window_sampling_enabled: bool = False
    """Whether uniform timestep resets should density-boost a window around contact start ``t1``."""

    uniform_t1_window_half_width_steps: int = 50
    """Half-width, in motion steps, of the density-boosted ``[t1 - width, t1 + width]`` window."""

    uniform_t1_window_density_boost: float = 1.0
    """Per-timestep density multiplier for the uniform ``t1`` window relative to other nonzero steps."""

    start_at_timestep_zero_prob: float = 0.2
    """Probability of starting at timestep zero."""

    start_at_timestep_zero_prob_end: float | None = None
    """Optional final probability of starting at timestep zero after curriculum ramp."""

    start_at_timestep_zero_prob_start_iter: int | None = None
    """Training iteration where the start-at-zero curriculum begins."""

    start_at_timestep_zero_prob_end_iter: int | None = None
    """Training iteration where the start-at-zero curriculum reaches its final probability."""

    freeze_at_timestep_zero_prob: float = 0.95
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    freeze_at_timestep_zero_prob_end: float | None = None
    """Optional final freeze probability after curriculum ramp."""

    freeze_at_timestep_zero_prob_start_iter: int | None = None
    """Training iteration where the freeze-at-zero curriculum begins."""

    freeze_at_timestep_zero_prob_end_iter: int | None = None
    """Training iteration where the freeze-at-zero curriculum reaches its final probability."""

    clip_weighting_strategy: str = "uniform_clip"
    """Sampling strategy across clips when a motion bank is loaded.

    Options:
      - "uniform_clip": each clip gets equal probability.
      - "uniform_step": clips are weighted by valid start steps (length-aware).
      - "success_rate_adaptive": clip weights adapt inversely to success rate.
    """

    min_weight_factor: float = 1.0 / 3.0
    """Minimum weight multiplier for adaptive clip weighting."""

    max_weight_factor: float = 3.0
    """Maximum weight multiplier for adaptive clip weighting."""

    enable_default_pose_prepend: bool = True
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = True
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    align_motion_to_init_yaw: bool = False
    """If True, rotate the motion so frame-0 yaw matches the robot init yaw.

    This also applies a translation so the frame-0 root position matches the robot
    init position (plus env origin), keeping a consistent world frame across episodes.
    """

    pair_terrain_with_motion: bool = False
    """If True, align each motion clip with a terrain tile/column.

    Preferred mode uses named OBJ tiles (mesh_type=load_obj) where each OBJ stem
    matches a motion clip name. If OBJ names are not available but the terrain
    exposes a tile origin grid (for example, procedural/heightmap terrains), clips
    are paired to terrain columns by clip order.
    """

    object_size_scale: list[float] | None = None
    """Optional scale applied to motion-provided object-size metadata.

    This is useful when the simulator object asset is scaled at launch time via
    ``robot.object.scale`` and observations should expose the scaled dimensions
    instead of the raw motion-bank metadata.
    """

    clean_noisy_clip_curriculum: CleanNoisyClipCurriculumConfig = field(default_factory=CleanNoisyClipCurriculumConfig)
    """Optional clip-group curriculum for mixed clean/noisy motion banks."""

    fixed_clip_group_assignment: FixedClipGroupAssignmentConfig = field(default_factory=FixedClipGroupAssignmentConfig)
    """Optional fixed env-to-clip-group assignment for mixed motion banks."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)

    # sparse object-goal distillation
    sparse_object_goal: SparseObjectGoalConfig = field(default_factory=SparseObjectGoalConfig)

    # future target pose (MotionTracking-style)
    num_future_steps: int = 0
    """Number of future motion steps to expose as target poses. Set to 0 to disable."""

    target_pose_type: str | None = None
    """Target pose encoding type (e.g., 'max-coords-future-rel' or 'max-coords-future-rel-with-time')."""
