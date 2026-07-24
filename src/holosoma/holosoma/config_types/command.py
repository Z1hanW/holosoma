"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Annotated, Any, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


# These bounds are intentionally generous relative to normal motion clips,
# but prevent malformed scientific configs from allocating enormous smoothing
# kernels or overflowing segment-index arithmetic before runtime validation.
MAX_CONTACT_AWARE_SMOOTHING_STEPS = 4096
MAX_CONTACT_AWARE_SEGMENT_STEPS = 1_000_000


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

    dof_vel: float = 0.0
    """Noise scale for the initial dof velocity in rad/s."""

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

    allowed_object_categories: list[str] = field(default_factory=list)
    """Optional object-category allowlist used when loading a motion directory.

    Supported coarse categories are inferred from clip/object metadata strings
    and currently include values such as ``box``, ``bin``, ``barrel``, and
    ``ball``. Empty means all clips are loaded.
    """

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    adaptive_sampling_contact_interval_root: str | None = None
    """Optional root directory of exported per-clip ``contact_intervals.json`` files.

    When set, adaptive timestep sampling logs additional contact-relative probability masses
    to W&B/TensorBoard, uniform ``t1`` window sampling can use the same bank, and contact-aware
    command/button observations retain the exported training timebase even when reset sampling
    is disabled (for example during evaluation). The bank is matched by clip id, so it can live
    outside the motion bank.
    """

    contact_interval_runtime_prepend_compensation: bool = False
    """Convert exported wall-clock contact intervals into runtime motion time.

    Contact exporters index physical rollout steps, while the multi-clip
    runtime prepend holds motion time at zero.  New runs that combine those
    two features must enable this conversion explicitly.  The default remains
    ``False`` so checkpoints trained with the legacy, uncompensated timeline
    keep their original observation/reward semantics when evaluated or
    resumed.
    """

    contact_aware_button_window_mode: Literal[
        "contact_interval",
        "kinematic_lift",
    ] = "contact_interval"
    """How automatic pickup/drop button transition labels are derived.

    ``contact_interval`` preserves the legacy exported-sidecar behavior.
    ``kinematic_lift`` uses the source motion's object-to-root relative-height
    trace and is deliberately independent of the root-command carry window.
    """

    contact_aware_carry_window_mode: Literal["rel_z", "peak_height"] = "rel_z"
    """How contact-aware root-command active windows are derived.

    ``rel_z`` preserves the original object-root relative-height rule. ``peak_height`` uses
    the full object world-height trace and keeps command active only while the object is
    stably near the clip's peak carry height.
    """

    contact_aware_peak_height_alpha: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.91
    """Peak-height fraction used by ``contact_aware_carry_window_mode='peak_height'``.

    The threshold is ``min_height + alpha * (max_height - min_height)``.
    """

    contact_aware_peak_height_smoothing_steps: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CONTACT_AWARE_SMOOTHING_STEPS),
    ] = 5
    """Centered edge-padded moving-average window for peak-height carry-window detection."""

    contact_aware_sparse_root_command_mode: Literal[
        "tracking_error",
        "t1_aligned_segment",
    ] = "tracking_error"
    """Root command used by ``actor_obs_root_contact_aware``.

    ``tracking_error`` preserves the original robot-to-target root delta. ``t1_aligned_segment``
    uses a non-overlap motion segment command anchored at carry-window start ``t1``.
    """

    contact_aware_sparse_root_segment_steps: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CONTACT_AWARE_SEGMENT_STEPS),
    ] = 30
    """Segment length, in motion frames, for ``contact_aware_sparse_root_command_mode='t1_aligned_segment'``."""

    contact_aware_sparse_root_zero_yaw_threshold_deg: Annotated[
        float,
        Field(strict=True, ge=0.0, le=180.0, allow_inf_nan=False),
    ] = 0.0
    """Zero sparse yaw commands whose absolute value is at or below this threshold in degrees."""

    uniform_t1_window_sampling_enabled: bool = False
    """Whether uniform timestep resets should density-boost a window around contact start ``t1``."""

    uniform_t1_window_half_width_steps: int = 50
    """Half-width, in motion steps, of the density-boosted ``[t1 - width, t1 + width]`` window."""

    uniform_t1_window_density_boost: float = 1.0
    """Per-timestep density multiplier for the uniform ``t1`` window relative to other nonzero steps."""

    uniform_t1_window_target_sample_frac: float | None = None
    """Optional target fraction of reset samples that should land in the ``t1`` window.

    When set, this overrides ``uniform_t1_window_density_boost`` for clips with
    loaded contact windows and accounts for ``start_at_timestep_zero_prob``. The
    overall reset fraction can be lower if some sampled clips have no loaded contact
    window. Clips whose valid nonzero range is entirely inside the ``t1`` window may
    exceed the target because there is no outside window to sample.
    """

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

    # future target pose (MotionTracking-style)
    num_future_steps: int = 0
    """Number of future motion steps to expose as target poses. Set to 0 to disable."""

    target_pose_type: str | None = None
    """Target pose encoding type (e.g., 'max-coords-future-rel' or 'max-coords-future-rel-with-time')."""
