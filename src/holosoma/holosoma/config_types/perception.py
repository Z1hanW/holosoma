"""Configuration types for perception sensors."""

from __future__ import annotations

from dataclasses import field
from typing import Dict

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionConfig:
    """Configuration for perception sensing and fusion."""

    enabled: bool = False
    """Enable perception sensors and policy fusion."""

    inject_into_policy_modules: bool = True
    """Whether to inject perception encoder branches into actor/critic modules."""

    inject_into_critic_modules: bool = True
    """Whether to inject perception into the critic as well as the actor."""

    output_mode: str = "heightmap"
    """Perception output type: 'heightmap' or 'camera_depth'."""

    camera_source: str = "far_tracking_warp"
    """Camera source for camera_depth output. Supported: 'far_tracking_warp', 'rendered', 'rendered_depth_sensor'."""

    camera_strict_warp: bool | None = None
    """When set, force strict warp camera ray convention instead of env-var defaults."""

    camera_auto_fix_backward: bool | None = None
    """When set, enable/disable automatic backward-ray correction (ignored in strict warp mode)."""

    grid_size: int = 11
    """Number of samples per dimension for the heightmap grid."""

    grid_interval: float = 0.1
    """Grid spacing in meters between samples."""

    heightmap_size: tuple[float, float] | None = None
    """Optional (length, width) in meters for heightmap grid (VideoMimic-style)."""

    heightmap_resolution: float | None = None
    """Optional resolution (meters) for heightmap grid (VideoMimic-style)."""

    ray_start_height: float = 0.0
    """Height above the sampling plane to start rays (meters)."""

    max_distance: float = 5.0
    """Clamp distance for missed rays (meters)."""

    heightmap_obs_offset: float = 0.0
    """Offset subtracted from heightmap observations (far-tracking parity uses 0.5)."""

    update_hz: float = 50.0
    """Perception update rate in Hz."""

    use_heading_only: bool = True
    """Rotate grid/rays using yaw only when True."""

    heightmap_body_name: str | None = None
    """Body name to anchor heightmap rays (defaults to robot root when None)."""

    camera_pitch_deg: float = -20.0
    """Virtual camera pitch in degrees (negative tilts down)."""
    camera_mount_quat: list[float] | None = None
    """Optional local camera mount rotation (xyzw), typically link->sensor mount."""
    camera_frame_quat: list[float] | None = None
    """Optional camera frame rotation (xyzw). Matches simulator camera frame when provided."""

    camera_target_pitch_deg: float | None = None
    """Optional target pitch in degrees for auto-tilting raycast cameras when no hits are detected."""

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

    camera_warp_preprocess: bool = True
    """Apply warp-image-style depth preprocessing (crop/resize/noise/latency/normalization)."""

    camera_warp_freq_ratio: int = 1
    """Update preprocessed depth every N perception updates (1 = every update)."""

    camera_warp_latency_frame: int = 0
    """Return delayed frame index from temporal buffer."""

    camera_warp_buffer_len: int = 1
    """Temporal depth buffer length for latency modeling."""

    camera_warp_resize: tuple[int, int] | None = None
    """Optional (height, width) resize after crop."""

    camera_warp_crop_top: int = 0
    """Pixels cropped from top before resize."""

    camera_warp_crop_bottom: int = 0
    """Pixels cropped from bottom before resize."""

    camera_warp_crop_left: int = 0
    """Pixels cropped from left before resize."""

    camera_warp_crop_right: int = 0
    """Pixels cropped from right before resize."""

    camera_warp_min_valid_depth: float = 0.15
    """Depth below this threshold is treated as empty (set to max_distance)."""

    camera_warp_normalize: bool = False
    """Normalize depth to [-0.5, 0.5] using [camera_near, max_distance]."""

    camera_warp_edge_noise: bool = False
    """Enable Sobel edge perturbation to mimic depth discontinuity artifacts."""

    camera_warp_edge_border: int = 3
    """Ignore edge perturbation near image borders by this many pixels."""

    camera_warp_edge_shuffle_prob: float = 0.9
    """Probability for edge pixels to be replaced by shuffled local neighbors."""

    camera_warp_edge_empty_prob: float = 0.7
    """Probability for selected edge pixels to be set to empty depth."""

    camera_warp_edge_thresh_primary: float = 1.0
    """Primary Sobel magnitude threshold used for neighbor shuffling."""

    camera_warp_edge_thresh_secondary: float = 0.6
    """Secondary Sobel magnitude threshold used for empty-pixel injection."""

    camera_warp_edge_far_depth_thresh: float = 2.5
    """Depth threshold used by secondary edge-empty mask."""

    camera_warp_enable_holes: bool = False
    """Enable synthetic hole dropout (coarse approximation of perlin-style holes)."""

    camera_warp_hole_prob: float = 0.0
    """Probability threshold for synthetic hole masks when enabled."""

    camera_apply_sensor_noise: bool = True
    """Apply runtime camera multiplicative/dropout noise when provided by environment randomization."""

    camera_scandots_stride: int = 4
    """Pixel stride for legacy scandots depth path (currently unused)."""

    camera_scandots_width: int | None = None
    """Optional scandots sample count along image width (overrides stride when set)."""

    camera_scandots_height: int | None = None
    """Optional scandots sample count along image height (overrides stride when set)."""

    camera_scandots_upsample: str = "bilinear"
    """Upsampling mode for legacy scandots depth path (currently unused)."""

    camera_include_robot_mesh: bool = False
    """Include robot visual meshes in camera raycasting."""

    camera_mesh_allowlist: list[str] | None = None
    """Optional robot link allowlist for camera raycast self-meshes (uses all links when None)."""

    camera_mesh_file_map: Dict[str, str] | None = None
    """Optional explicit link->mesh filename map for camera self-mesh raycasting (far-tracking parity)."""

    depth_sensor_asset_path: str | None = None
    """Optional IsaacSim depth sensor asset path (e.g., /Isaac/Sensors/Intel/RealSense/rsd455.usd)."""

    depth_sensor_depth_prim: str | None = None
    """Optional depth sensor prim path for SingleViewDepthSensorAsset depth output."""

    sensor_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Sensor offset from robot root in base frame (meters)."""

    encoder_output_dim: int = 512
    """Output dimension for the perception encoder."""

    encoder_type: str = "gated_linear"
    """Perception encoder type: 'gated_linear', 'attention', 'far_tracking_cnn_small', or 'time_gru'."""

    encoder_fusion: str = "extra_input_to_hidden"
    """How encoded perception is fused into policy modules: 'extra_input_to_hidden' or 'concat'."""

    encoder_pretrained: bool = True
    """Whether to load pretrained weights for external perception backbones when supported."""

    encoder_pretrained_path: str | None = None
    """Optional local checkpoint path for external perception backbones."""

    encoder_freeze_backbone: bool = True
    """Freeze external perception backbones and train only the projection head when supported."""

    encoder_target_size: int | tuple[int, int] | None = None
    """Optional target size used by external perception preprocessors."""

    encoder_patch_size: int | None = None
    """Optional patch-size alignment used by external perception preprocessors."""
