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

    auto_start_policy: bool = False
    """Automatically enable policy action at startup (equivalent to pressing start)."""

    auto_start_motion_clip: bool = False
    """Automatically start WBT motion clip playback at startup."""

    auto_start_stiff_hold_sec: float = 1.0
    """Duration (seconds) to stay in stiff-hold before auto-starting policy + motion clip."""

    auto_start_stiff_pose_tolerance: float = 0.12
    """Max joint-position error (rad) to consider stiff-hold at target pose reached."""

    auto_start_stiff_max_wait_sec: float = 4.0
    """Maximum wait (seconds) before forcing auto-start if stiff target is not reached."""

    include_motion_future_target_poses: bool = False
    """Include motion_future_target_poses in the actor observation (WBT motion-tracking policies)."""

    motion_future_target_poses_dim: int | None = None
    """Override dimension for motion_future_target_poses when metadata is unavailable."""

    motion_future_target_poses_motion_file: str | None = None
    """Optional motion file path override for motion_future_target_poses computation."""

    object_state_sub_enabled: bool = False
    """Subscribe to MuJoCo object-state stream for object-aware WBT observations."""

    object_state_sub_port: int = 5557
    """ZMQ SUB port used when object_state_sub_enabled=True."""

    tracking_viz_pub_enabled: bool = False
    """Publish tracking-debug trajectories (future keypoints/object) for MuJoCo GUI overlay."""

    tracking_viz_pub_port: int = 5560
    """ZMQ PUB port used when tracking_viz_pub_enabled=True."""

    tracking_viz_future_steps: int = 10
    """Number of future steps to publish for tracking-debug visualization."""

    tracking_viz_pub_every_n: int = 1
    """Publish tracking-debug payload every N policy ticks."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""
