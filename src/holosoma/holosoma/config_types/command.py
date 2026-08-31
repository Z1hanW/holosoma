"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
import math
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
class HMIGoalPoseNoiseConfig:
    """Clipped Gaussian terminal-goal noise for Hybrid Motion Imitation.

    This is the shape-preserving HoloSoma adapter for the goal-noise contract
    introduced by Hybrid-Motion-Imitation.  Position noise is expressed in the
    world frame and rotation noise is roll/pitch/yaw in radians.
    """

    pos_std_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pos_clip_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_clip: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self) -> None:
        for name in ("pos_std_xyz", "pos_clip_xyz", "rpy_std", "rpy_clip"):
            values = getattr(self, name)
            if len(values) != 3:
                raise ValueError(f"HMIGoalPoseNoiseConfig.{name} must contain exactly three values.")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
                raise ValueError(
                    f"HMIGoalPoseNoiseConfig.{name} values must be finite and non-negative."
                )


@dataclass(frozen=True)
class HMIMotionConfig:
    """Hybrid Motion Imitation track/generation training contract.

    Environment identity is fixed for the process lifetime.  Tracking rows use
    dense reference imitation; generation rows hide reference targets from the
    actor/critic and receive only terminal-goal task reward plus shared
    regularization.
    """

    upstream_repository: Literal[
        "https://github.com/jiashunwang/Hybrid-Motion-Imitation"
    ] = "https://github.com/jiashunwang/Hybrid-Motion-Imitation"
    upstream_commit: Literal[
        "c353731999b3578c41ad5a00f896415b45e6a9f5"
    ] = "c353731999b3578c41ad5a00f896415b45e6a9f5"
    actor_interface_semantics: Literal[
        "actor94_depth5046_action29_terminal_object_xy_yaw_v1",
        "actor93_depth5046_action29_terminal_object_xy_v2",
        "actor93_depth5046_action29_terminal_root_xy_v2",
    ] = "actor94_depth5046_action29_terminal_object_xy_yaw_v1"
    goal_target: Literal["object_xy_yaw", "object_xy", "root_xy"] = (
        "object_xy_yaw"
    )
    track_ratio: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.5
    env_partition_seed: Annotated[int, Field(strict=True, ge=0)] = 0
    gen_start_at_timestep_zero_prob: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] | None = 0.1
    object_goal_noise: HMIGoalPoseNoiseConfig = field(default_factory=HMIGoalPoseNoiseConfig)
    root_goal_noise: HMIGoalPoseNoiseConfig = field(default_factory=HMIGoalPoseNoiseConfig)
    gen_step_zero_root_pos_std_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gen_step_zero_root_pos_clip_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gen_step_zero_root_rpy_std: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gen_step_zero_root_rpy_clip: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    goal_noise_initial_scale: Annotated[
        float,
        Field(strict=True, ge=0.0, allow_inf_nan=False),
    ] = 0.3
    goal_noise_min_scale: Annotated[
        float,
        Field(strict=True, ge=0.0, allow_inf_nan=False),
    ] = 0.3
    goal_noise_max_scale: Annotated[
        float,
        Field(strict=True, ge=0.0, allow_inf_nan=False),
    ] = 1.0
    goal_noise_scale_step: Annotated[
        float,
        Field(strict=True, ge=0.0, allow_inf_nan=False),
    ] = 0.01
    goal_noise_success_threshold_up: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.90
    goal_noise_success_threshold_down: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.80
    goal_noise_update_interval: Annotated[int, Field(strict=True, ge=1)] = 20
    goal_noise_ema_alpha: Annotated[
        float,
        Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.5

    def __post_init__(self) -> None:
        expected_interface = {
            "object_xy_yaw": "actor94_depth5046_action29_terminal_object_xy_yaw_v1",
            "object_xy": "actor93_depth5046_action29_terminal_object_xy_v2",
            "root_xy": "actor93_depth5046_action29_terminal_root_xy_v2",
        }[self.goal_target]
        if self.actor_interface_semantics != expected_interface:
            raise ValueError(
                "HMIMotionConfig goal_target/interface mismatch: "
                f"goal_target={self.goal_target!r}, "
                f"actor_interface_semantics={self.actor_interface_semantics!r}, "
                f"expected={expected_interface!r}."
            )
        if self.goal_target in {"object_xy", "root_xy"}:
            xy_noise = (
                self.object_goal_noise
                if self.goal_target == "object_xy"
                else self.root_goal_noise
            )
            if (
                float(xy_noise.pos_std_xyz[2]) != 0.0
                or float(xy_noise.pos_clip_xyz[2]) != 0.0
                or any(float(value) != 0.0 for value in xy_noise.rpy_std)
                or any(float(value) != 0.0 for value in xy_noise.rpy_clip)
            ):
                raise ValueError(
                    "HMI XY-only goal interfaces require zero Z and RPY goal noise."
                )
        for name in (
            "gen_step_zero_root_pos_std_xyz",
            "gen_step_zero_root_pos_clip_xyz",
            "gen_step_zero_root_rpy_std",
            "gen_step_zero_root_rpy_clip",
        ):
            values = getattr(self, name)
            if len(values) != 3:
                raise ValueError(f"HMIMotionConfig.{name} must contain exactly three values.")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
                raise ValueError(
                    f"HMIMotionConfig.{name} values must be finite and non-negative."
                )
        if not (
            self.goal_noise_min_scale
            <= self.goal_noise_initial_scale
            <= self.goal_noise_max_scale
        ):
            raise ValueError(
                "HMI goal-noise scales must satisfy min <= initial <= max."
            )
        if self.goal_noise_success_threshold_down > self.goal_noise_success_threshold_up:
            raise ValueError(
                "HMI goal-noise thresholds must satisfy down <= up."
            )


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
        "rolling_reference_delta",
        "precomputed_turn_then_forward",
    ] = "tracking_error"
    """Root command used by ``actor_obs_root_contact_aware``.

    ``tracking_error`` preserves the original robot-to-target root delta. ``t1_aligned_segment``
    uses a non-overlap motion segment command anchored at carry-window start ``t1``.
    ``rolling_reference_delta`` recomputes the reference-to-reference displacement from the
    current motion frame to ``contact_aware_sparse_root_segment_steps`` frames ahead on every
    policy step, expressed in the current reference-root heading frame and without robot-state
    feedback.
    ``precomputed_turn_then_forward`` reads an immutable per-frame command from the motion NPZ;
    each active row must contain either forward ``dx`` or ``dyaw``, never both, and is gated by
    the same runtime pickup latch as the pure-RL deployment command.
    """

    zero_root_command_when_drop_active: Annotated[
        bool,
        Field(strict=True),
    ] = False
    """Zero the actor's three root-command dimensions while its effective drop button is active.

    The default preserves historical checkpoint semantics. New runs may enable
    this explicitly to enforce the mutually exclusive actor input contract
    ``[root_x, root_y, root_yaw, drop] == [0, 0, 0, 1]`` during release.
    """

    contact_aware_sparse_root_segment_steps: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CONTACT_AWARE_SEGMENT_STEPS),
    ] = 30
    """Segment length or rolling lookahead, in motion frames, for reference-delta commands."""

    contact_aware_sparse_root_zero_yaw_threshold_deg: Annotated[
        float,
        Field(strict=True, ge=0.0, le=180.0, allow_inf_nan=False),
    ] = 0.0
    """Zero sparse yaw commands whose absolute value is at or below this threshold in degrees."""

    hybrid_stage2_enabled: bool = False
    """Enable the HIL-inspired parallel tracking/task environment contract.

    Tracking environments keep the configured reference command. Task
    environments receive zero root command until the runtime pickup latch is
    set, then receive a deployment-faithful constant local-frame forward
    command. Reward and termination terms use the same immutable environment
    assignment.
    """

    hybrid_stage2_task_env_fraction: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.5
    """Fraction of environments assigned to the stage-2 task objective."""

    hybrid_stage2_forward_command_m: Annotated[
        float,
        Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False),
    ] = 0.15
    """Constant raw ``dx`` command exposed after the task environment lifts."""

    hybrid_velocity_enabled: bool = False
    """Enable the isolated velocity-conditioned tracking/task curriculum.

    The actor command is always ``[vx, vy, yaw_rate]``. Tracking rows receive
    the reference root velocity while task rows receive zero before the
    runtime pickup latch and a constant forward velocity afterwards.
    """

    hybrid_velocity_command_frame: Literal["heading", "world"] = "heading"
    """Coordinate frame for hybrid velocity commands and task rewards.

    ``heading`` preserves the original behavior: reference XY velocity and the
    task objective are expressed in the robot/pickup heading frame. ``world``
    exposes the canonical per-frame world XY velocity and defines the fixed
    task command along world +X.
    """

    hybrid_velocity_task_env_fraction_start: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.0
    """Task-row fraction at the beginning of the hybrid curriculum."""

    hybrid_velocity_task_env_fraction_end: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.5
    """Task-row fraction after the hybrid curriculum ramp completes."""

    hybrid_velocity_task_env_fraction_start_iter: Annotated[
        int,
        Field(strict=True, ge=0),
    ] = 0
    """First PPO iteration of the task-row fraction ramp."""

    hybrid_velocity_task_env_fraction_end_iter: Annotated[
        int,
        Field(strict=True, ge=0),
    ] = 5000
    """PPO iteration at which the final task-row fraction is reached."""

    hybrid_velocity_forward_command_mps: Annotated[
        float,
        Field(strict=True, gt=0.0, le=5.0, allow_inf_nan=False),
    ] = 0.5
    """Task-row forward command after pickup, in metres per second."""

    hybrid_velocity_lift_height_m: Annotated[
        float,
        Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False),
    ] = 0.10
    """Object world-height increase that saturates task lift progress."""

    hmi: HMIMotionConfig | None = None
    """Optional Hybrid Motion Imitation track/generation contract.

    This is intentionally separate from the older ``hybrid_stage2`` and
    ``hybrid_velocity`` experiments, whose objectives and actor commands are
    different.
    """

    pure_rl_policy_command_after_lift_enabled: bool = False
    """Replace only the actor's sparse root command around the runtime lift latch.

    This option does not alter rewards, terminations, motion tracking, or any
    other observation term.
    """

    pure_rl_policy_forward_command_m: Annotated[
        float,
        Field(strict=True, gt=0.0, le=10.0, allow_inf_nan=False),
    ] = 0.5
    """Constant post-lift ``dx`` value in the actor's ``[dx, dy, dyaw]`` input."""

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
