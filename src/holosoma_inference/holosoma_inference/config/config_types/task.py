"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    use_joystick_goal: bool = False
    """Use joystick to generate torso position/yaw goals for WBT policies."""

    joystick_goal_scale: float = 0.5
    """Scale (meters) applied to joystick XY for torso_xy_rel goals."""

    joystick_yaw_scale: float = 0.5
    """Scale (radians) applied to joystick yaw for torso_yaw_rel goals."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    sim_clock_port: int = 5555
    """ZMQ port used to receive split sim2sim simulator clock."""

    auto_start_policy: bool = False
    """Automatically enable policy actions at startup."""

    auto_start_motion: bool = False
    """Automatically start WBT motion progression after policy startup."""

    auto_start_motion_clip: bool = False
    """Automatically stiff-hold, then start policy + motion clip together."""

    auto_start_stiff_hold_sec: float = 1.0
    """Minimum stiff-hold duration before starting auto-start WBT rollout."""

    auto_start_stiff_pose_tolerance: float = 0.12
    """Maximum joint-position error (rad) to treat stiff-hold target as reached."""

    auto_start_stiff_max_wait_sec: float = 4.0
    """Maximum wait before auto-start WBT rollout proceeds even if pose error stays high."""

    defer_policy_start_until_valid_state: bool = False
    """In non-interactive mode, wait for the first valid robot state before enabling policy actions."""

    motion_file: str | None = None
    """Optional single motion clip path used for split sim2sim WBT inference."""

    apply_training_motion_transitions: bool = False
    """Apply training-time default-pose prepend/append transitions to split sim2sim motion data."""

    use_sim_state: bool = False
    """Subscribe to simulator root/object state over ZMQ for split sim2sim inference."""

    sim_state_port: int = 5557
    """ZMQ port used to receive split sim2sim simulator state."""

    sim_control_port: int = 5559
    """ZMQ port used to send split sim2sim simulator control requests such as reset."""

    sim_reset_restart_delay_sec: float = 0.5
    """Delay between requesting a simulator reset and restarting policy+motion playback."""

    sim_object_name: str = "object"
    """Simulator actor name used for the carried object in split sim2sim inference."""

    use_sim_perception: bool = False
    """Subscribe to simulator perception observations over ZMQ for split sim2sim inference."""

    sim_perception_port: int = 5558
    """ZMQ port used to receive split sim2sim perception observations."""

    include_motion_future_target_poses: bool = False
    """Include motion_future_target_poses in the actor observation (WBT motion-tracking policies)."""

    motion_future_target_poses_dim: int | None = None
    """Override dimension for motion_future_target_poses when metadata is unavailable."""

    motion_future_target_poses_motion_file: str | None = None
    """Optional motion file path override for motion_future_target_poses computation."""

    use_root_reference_at_clip_start: bool = False
    """For split WBT tracking, mimic training step-0 observations by using robot root instead of ref body."""

    prefer_sim_ref_from_sim_state: bool = False
    """Prefer simulator-measured ref-body pose from split sim-state over Pinocchio FK when available."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""
