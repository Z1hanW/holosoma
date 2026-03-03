#!/usr/bin/env bash

# Shared data path manifest for data_in.sh / data_out.sh.
# Paths are absolute and grouped by destination bucket.

# object-related -> /nfs/zzzihanw/box3r
BOX3R_PATHS=(
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/behave_sq_obj"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/behave_sq_carry"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/behave"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_aug_mix_ml"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_mix_ml"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/behave_objects"
  "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
)

# terrain-related -> /nfs/zzzihanw/crisp
CRISP_PATHS=(
  "/data/terrain/___crisp_clean_geometry"
  "/data/terrain/___crisp_clean_motion"
  "/home/ubuntu/FAR/holosoma/multi-terrain"
  "/home/ubuntu/FAR/holosoma/multi-motion"
  "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz"
  "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj"
  "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz"
)
