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
    """Automatically start WBT policy + motion clip in non-interactive mode."""

    auto_start_stiff_hold_sec: float = 1.0
    """Minimum stiff-hold duration before auto-start WBT rollout."""

    auto_start_stiff_pose_tolerance: float = 0.12
    """Maximum joint-position error treated as stiff-hold target reached."""

    auto_start_stiff_max_wait_sec: float = 4.0
    """Maximum wait before auto-start WBT rollout proceeds anyway."""

    defer_policy_start_until_valid_state: bool = False
    """In non-interactive mode, wait for a valid robot state before enabling policy."""

    motion_file: str | None = None
    """Optional single motion clip path used for split sim2sim WBT inference."""

    apply_training_motion_transitions: bool = False
    """Apply training-time default-pose prepend/append transitions to split sim2sim motion data."""

    use_sim_state: bool = False
    """Subscribe to simulator root/object state over ZMQ for split sim2sim inference."""

    sim_state_port: int = 5557
    """ZMQ port used to receive split sim2sim simulator state."""

    sim_control_port: int = 5559
    """ZMQ port used to send split sim2sim simulator control requests."""

    use_zmq_lowcmd: bool = False
    """Send lowcmd over the split sim-control ZMQ channel instead of Unitree DDS."""

    sim_object_name: str = "object"
    """Simulator actor name used for the carried object in split sim2sim inference."""

    use_root_reference_at_clip_start: bool = False
    """Use robot root instead of ref body at clip start to match training step-0 semantics."""

    prefer_sim_ref_from_sim_state: bool = False
    """Prefer simulator-measured ref-body pose from split sim-state when available."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""
