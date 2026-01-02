import numpy as np
from motion_tracking.utils.urdfpoints import save_points,add_noise_to_urdf

noise_levels = (0.05, 0.1)  # Adjust these values as needed
size_scale_std = 0.05  # Standard deviation for size scaling, adjust as needed

urdf_path = "motion_tracking/data/assets/urdf/parkour_line1_5_fix_10_horizon_1228_eval"
for i in range(10):
    add_noise_to_urdf(urdf_path+'.urdf', noise_levels, size_scale_std, urdf_path+f'_random_{i}.urdf')
