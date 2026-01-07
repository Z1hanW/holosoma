"""Perception configuration presets."""

from holosoma.config_types.perception import PerceptionConfig

none = PerceptionConfig(enabled=False)

heightmap = PerceptionConfig(
    enabled=True,
    output_mode="heightmap",
    grid_size=11,
    grid_interval=0.1,
    update_hz=50.0,
    encoder_output_dim=512,
)

camera_depth_d435i = PerceptionConfig(
    enabled=True,
    output_mode="camera_depth",
    camera_source="raycast",
    grid_size=11,
    grid_interval=0.1,
    update_hz=30.0,
    camera_width=1280,
    camera_height=720,
    camera_vfov_deg=55.2,
    camera_pitch_deg=-47.6,
    camera_fps=30.0,
    camera_near=0.1,
    camera_far=10.0,
    camera_distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    encoder_output_dim=512,
)

camera_depth_d435i_rendered = PerceptionConfig(
    enabled=True,
    output_mode="camera_depth",
    camera_source="rendered",
    grid_size=11,
    grid_interval=0.1,
    update_hz=30.0,
    camera_width=1280,
    camera_height=720,
    camera_vfov_deg=55.2,
    camera_pitch_deg=-47.6,
    camera_body_name="torso_link",
    camera_env_id=0,
    camera_fps=30.0,
    camera_near=0.1,
    camera_far=10.0,
    camera_distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    encoder_output_dim=512,
)

DEFAULTS = {
    "none": none,
    "heightmap": heightmap,
    "camera_depth_d435i": camera_depth_d435i,
    "camera_depth_d435i_rendered": camera_depth_d435i_rendered,
}

__all__ = ["none", "heightmap", "camera_depth_d435i", "camera_depth_d435i_rendered", "DEFAULTS"]
