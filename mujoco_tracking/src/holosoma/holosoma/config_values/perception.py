"""Perception configuration presets."""

from dataclasses import replace

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

WARP_SENSORS_G1_D435_MESH_FILE_MAP = {
    "pelvis": "combined_pelvis.STL",
    "left_hip_pitch_link": "left_hip_pitch_link.STL",
    "left_hip_roll_link": "left_hip_roll_link.STL",
    "left_hip_yaw_link": "left_hip_yaw_link.STL",
    "left_knee_link": "left_knee_link.STL",
    "left_ankle_pitch_link": "left_ankle_pitch_link.STL",
    "left_ankle_roll_link": "left_ankle_roll_link.STL",
    "right_hip_pitch_link": "right_hip_pitch_link.STL",
    "right_hip_roll_link": "right_hip_roll_link.STL",
    "right_hip_yaw_link": "right_hip_yaw_link.STL",
    "right_knee_link": "right_knee_link.STL",
    "right_ankle_pitch_link": "right_ankle_pitch_link.STL",
    "right_ankle_roll_link": "right_ankle_roll_link.STL",
    "waist_yaw_link": "waist_yaw_link_rev_1_0.STL",
    "waist_roll_link": "waist_roll_link_rev_1_0.STL",
    "left_shoulder_pitch_link": "left_shoulder_pitch_link.STL",
    "left_shoulder_roll_link": "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link": "left_shoulder_yaw_link.STL",
    "left_elbow_link": "left_elbow_link.STL",
    "left_wrist_roll_link": "left_wrist_roll_link.STL",
    "left_wrist_pitch_link": "left_wrist_pitch_link.STL",
    "left_wrist_yaw_link": "combined_left_wrist_spherehand.STL",
    "right_shoulder_pitch_link": "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link": "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link": "right_shoulder_yaw_link.STL",
    "right_elbow_link": "right_elbow_link.STL",
    "right_wrist_roll_link": "right_wrist_roll_link.STL",
    "right_wrist_pitch_link": "right_wrist_pitch_link.STL",
    "right_wrist_yaw_link": "combined_right_wrist_spherehand.STL",
}

none = PerceptionConfig(enabled=False)

heightmap = PerceptionConfig(
    enabled=True,
    output_mode="heightmap",
    # Match far-tracking HeightmapSceneCfg:
    # GridPatternCfg(resolution=0.1, size=[1.6, 1.6]), attach on pelvis with z-offset.
    grid_size=17,
    grid_interval=0.1,
    heightmap_size=(1.6, 1.6),
    heightmap_resolution=0.1,
    heightmap_body_name="pelvis",
    max_distance=30.0,
    ray_start_height=20.0,
    heightmap_obs_offset=0.5,
    update_hz=50.0,
    encoder_output_dim=512,
    encoder_type="attention",
)

camera_depth_d435i = PerceptionConfig(
    enabled=True,
    inject_into_critic_modules=False,
    output_mode="camera_depth",
    camera_source="far_tracking_warp",
    camera_strict_warp=True,
    camera_auto_fix_backward=False,
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
    camera_mesh_file_map=WARP_SENSORS_G1_D435_MESH_FILE_MAP,
    camera_fps=30.0,
    camera_near=0.1,
    camera_far=3.0,
    camera_distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    # Align observation preprocessing with far-tracking warp_image_features.
    camera_warp_preprocess=True,
    camera_warp_freq_ratio=1,
    camera_warp_latency_frame=0,
    camera_warp_buffer_len=3,
    camera_warp_resize=(58, 87),
    camera_warp_crop_top=2,
    camera_warp_crop_bottom=0,
    camera_warp_crop_left=4,
    camera_warp_crop_right=4,
    camera_warp_min_valid_depth=0.15,
    camera_warp_normalize=True,
    camera_warp_edge_noise=True,
    camera_warp_edge_border=3,
    camera_warp_edge_shuffle_prob=0.9,
    camera_warp_edge_empty_prob=0.7,
    camera_warp_edge_thresh_primary=1.0,
    camera_warp_edge_thresh_secondary=0.6,
    camera_warp_edge_far_depth_thresh=2.5,
    camera_warp_enable_holes=False,
    camera_warp_hole_prob=0.0,
    # Match far-tracking defaults: placement randomization stays on, sensor noise stays off.
    camera_apply_sensor_noise=False,
    # Match far-tracking student depth path: 58x87 -> CNN -> 32d, concatenated to actor inputs.
    encoder_output_dim=32,
    encoder_type="far_tracking_cnn_small",
    encoder_fusion="concat",
)

camera_depth_d435i_17x17 = replace(
    camera_depth_d435i,
    inject_into_critic_modules=True,
    camera_width=17,
    camera_height=17,
    camera_warp_resize=(17, 17),
    camera_warp_crop_top=0,
    camera_warp_crop_bottom=0,
    camera_warp_crop_left=0,
    camera_warp_crop_right=0,
    camera_apply_sensor_noise=True,
    encoder_output_dim=512,
    encoder_type="attention",
    encoder_fusion="extra_input_to_hidden",
)

camera_depth_d435i_defm_vit_s14 = replace(
    camera_depth_d435i,
    # Keep the same far-tracking-style 58x87 depth preprocessing, but swap the encoder.
    encoder_output_dim=384,
    encoder_type="defm_vit_s14",
    encoder_fusion="concat",
    encoder_pretrained=True,
    encoder_pretrained_path=None,
    encoder_freeze_backbone=True,
    encoder_target_size=224,
    encoder_patch_size=14,
)

camera_depth_d435i_defm_regnet_y_800mf = replace(
    camera_depth_d435i,
    # Keep the same far-tracking-style 58x87 depth preprocessing, but swap the encoder.
    encoder_output_dim=784,
    encoder_type="defm_regnet_y_800mf",
    encoder_fusion="concat",
    encoder_pretrained=True,
    encoder_pretrained_path=None,
    encoder_freeze_backbone=True,
    encoder_target_size=224,
    encoder_patch_size=None,
)

DEFAULTS = {
    "none": none,
    "heightmap": heightmap,
    "camera_depth_d435i": camera_depth_d435i,
    "camera_depth_d435i_17x17": camera_depth_d435i_17x17,
    "camera_depth_d435i_defm_vit_s14": camera_depth_d435i_defm_vit_s14,
    "camera_depth_d435i_defm_regnet_y_800mf": camera_depth_d435i_defm_regnet_y_800mf,
}

__all__ = [
    "none",
    "heightmap",
    "camera_depth_d435i",
    "camera_depth_d435i_17x17",
    "camera_depth_d435i_defm_vit_s14",
    "camera_depth_d435i_defm_regnet_y_800mf",
    "DEFAULTS",
]
