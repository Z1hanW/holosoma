import numpy as np
import os
from place_scene_motion_utils import *

seq_list = [
            "MPH1Library",    
        ]

date = "0921"
method = "ours"
for seq in seq_list:

    motion_filename =  f"motion_data/{date}/{seq}_{method}.npz"
    motion_outpath = f"motion_data/{date}/{seq}_{method}.npy"

    motion_data = np.load(motion_filename)

    end_root = get_smpl_robot(motion_data, motion_outpath, [0,0,0], 0, need_rotate=False)





        



