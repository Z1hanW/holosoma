"""Perception configuration presets."""

from holosoma.config_types.perception import PerceptionConfig


WARP_SENSORS_G1_D435_MESH_ALLOWLIST = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

none = PerceptionConfig(enabled=False)

heightmap = PerceptionConfig(
    enabled=True,
    output_mode="heightmap",
    grid_size=11,
    grid_interval=0.1,
    heightmap_size=(1.0, 1.0),
    heightmap_resolution=0.1,
    heightmap_body_name="torso_link",
    max_distance=5.0,
    ray_start_height=0.0,
    update_hz=50.0,
    encoder_output_dim=512,
    encoder_type="attention",
)

camera_depth_d435i = PerceptionConfig(
    enabled=True,
    output_mode="camera_depth",
    camera_source="mesh_raycast",
    max_distance=3.0,
    grid_size=11,
    grid_interval=0.1,
    update_hz=30.0,
    camera_width=106,
    camera_height=60,
    camera_vfov_deg=58.6,
    camera_hfov_deg=89.5,
    camera_pitch_deg=0.0,
    camera_target_pitch_deg=None,
    # Strict warp_sensors chain:
    # world_cam = world_torso * mount(offset_rot=[1,27,1]) * sensor_frame(offset_rot_base=[-90,0,-90]).
    camera_mount_quat=[0.00644801, 0.23350163, 0.00644801, 0.97231365],
    camera_frame_quat=[-0.5, 0.5, -0.5, 0.5],
    camera_body_name="torso_link",
    sensor_offset=[0.01, 0.01, 0.44],
    camera_include_robot_mesh=True,
    camera_mesh_allowlist=WARP_SENSORS_G1_D435_MESH_ALLOWLIST,
    camera_fps=30.0,
    camera_near=0.3,
    camera_far=3.0,
    camera_distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    encoder_output_dim=512,
    encoder_type="attention",
)

DEFAULTS = {
    "none": none,
    "heightmap": heightmap,
    "camera_depth_d435i": camera_depth_d435i,
}

__all__ = [
    "none",
    "heightmap",
    "camera_depth_d435i",
    "DEFAULTS",
]
