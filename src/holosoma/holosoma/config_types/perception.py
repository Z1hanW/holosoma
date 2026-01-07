"""Configuration types for perception sensors."""

from __future__ import annotations

from dataclasses import field

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for perception sensing and fusion."""

    enabled: bool = False
    """Enable perception sensors and policy fusion."""

    output_mode: str = "heightmap"
    """Perception output type: 'heightmap' or 'camera_depth'."""

    camera_source: str = "raycast"
    """Camera source for camera_depth output: 'raycast' or 'rendered'."""

    grid_size: int = 11
    """Number of samples per dimension for the heightmap grid."""

    grid_interval: float = 0.1
    """Grid spacing in meters between samples."""

    ray_start_height: float = 0.6
    """Height above the sampling plane to start rays (meters)."""

    max_distance: float = 5.0
    """Clamp distance for missed rays (meters)."""

    update_hz: float = 50.0
    """Perception update rate in Hz."""

    use_heading_only: bool = True
    """Rotate grid/rays using yaw only when True."""

    camera_pitch_deg: float = -20.0
    """Virtual camera pitch in degrees (negative tilts down)."""

    camera_body_name: str | None = None
    """Body name to anchor the camera pose (defaults to robot root when None)."""

    camera_env_id: int = 0
    """Environment index to render from when using rendered cameras."""

    camera_width: int | None = None
    """Camera image width in pixels (defaults to grid_size when None)."""

    camera_height: int | None = None
    """Camera image height in pixels (defaults to grid_size when None)."""

    camera_vfov_deg: float = 90.0
    """Camera vertical field of view in degrees."""

    camera_hfov_deg: float | None = None
    """Camera horizontal field of view in degrees (optional override)."""

    camera_fx: float | None = None
    """Camera focal length fx in pixels (overrides FOV if provided)."""

    camera_fy: float | None = None
    """Camera focal length fy in pixels (overrides FOV if provided)."""

    camera_cx: float | None = None
    """Camera principal point cx in pixels."""

    camera_cy: float | None = None
    """Camera principal point cy in pixels."""

    camera_fps: float = 30.0
    """Camera frame rate in Hz."""

    camera_near: float = 0.1
    """Camera near clipping plane in meters."""

    camera_far: float = 10.0
    """Camera far clipping plane in meters."""

    camera_distortion: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    """Camera distortion coefficients (k1, k2, p1, p2, k3)."""

    sensor_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Sensor offset from robot root in base frame (meters)."""

    encoder_output_dim: int = 512
    """Output dimension for the perception encoder."""
