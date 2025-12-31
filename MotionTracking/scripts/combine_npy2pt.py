import numpy as np
import os
from place_scene_motion_utils import *


source_data = {}
path = f"motion_data/emdb"
files = ["110hurdle1trim_tram.npy","110hurdle2trim_tram.npy","110hurdle1trim.npy","110hurdle2trim.npy"]

import copy
for ii in range(len(files)):
    from motion_tracking.utils.motion_lib import MotionLib
    motion_lib = MotionLib(motion_file=f'{path}/{files[ii]}', 
                        dof_body_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 
                        dof_offsets=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69], 
                        key_body_ids=torch.tensor([ 7,  3, 18, 23]))
    video_id = '_'.join(files[ii].split('_')[-2:]).replace('.npy', '')
    ref_height = 0
    source_data[ii] = [copy.copy(motion_lib.state),ref_height]

motion_data = []
height_adjust = []
for i in range(len(source_data)):
    motion_data.append(source_data[i][0])
    height_adjust.append(source_data[i][1])

combined_motions = combine_loaded_motions(copy.deepcopy(motion_data), height_adjust, resample=False)


torch.save(combined_motions, f"motion_data/110hurdle_4motions.pt")


        



