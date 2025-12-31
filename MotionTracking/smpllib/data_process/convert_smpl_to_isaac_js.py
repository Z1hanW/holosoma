import numpy as np
import os
from pathlib import Path
import typer
import shutil
from scipy.spatial.transform import Rotation as sRot
import torch
from tqdm import tqdm

from smpllib.smpllib.utils import TMP_SMPL_DIR
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from smpllib.utils import flags

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import (
    SkeletonMotion,
    SkeletonState,
    SkeletonTree
)

humanoid_type = 'smpl'

assert humanoid_type in ["smpl", "smplx", "smplh"], "Humanoid type must be one of smpl, smplx, smplh"
append_name = humanoid_type

if humanoid_type == "smpl":
    mujoco_joint_names = SMPL_MUJOCO_NAMES
    joint_names = SMPL_BONE_ORDER_NAMES
elif humanoid_type == "smplx" or humanoid_type == "smplh":
    mujoco_joint_names = SMPLH_MUJOCO_NAMES
    joint_names = SMPLH_BONE_ORDER_NAMES
else:
    raise NotImplementedError

upright_start = True

name = "video_3017_3120_0"
filename =  f"smpllib/parkour_motion/{name}.npz"
outputpath = f"motion_data/{name}_isaac.npy"

need_rotate=True

robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

# smpl_local_robot = LocalRobot(
#     robot_cfg,
#     data_dir="smpllib/data/smpl",
# )

smpl_local_robot = LocalRobot(robot_cfg,)




motion_data = np.load(filename)

# betas = motion_data["betas"]
# gender = motion_data["gender"]


amass_pose = motion_data["poses"]
amass_trans = motion_data["transl"]
betas = np.zeros(16)

if need_rotate:
    from scipy.spatial.transform import Rotation as sRot
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False)
    new_root_rot = (transform * sRot.from_rotvec(amass_pose[:, :3])).as_rotvec()
    amass_pose[:, :3] = new_root_rot
    amass_trans = amass_trans.dot(transform.as_matrix().T)

amass_trans[:,0] -= amass_trans[0:1,0]
amass_trans[:,1] -= amass_trans[0:1,1]

mocap_fr = 30
pose_aa = torch.tensor(amass_pose)  
amass_trans = torch.tensor(amass_trans) 
betas = torch.from_numpy(betas)
batch_size = pose_aa.shape[0]



gender = "neutral"

motion_data = {
    "pose_aa": pose_aa.numpy(),
    "trans": amass_trans.numpy(),
    "beta": betas.numpy(),
    "gender": gender,
}

smpl_2_mujoco = [
    joint_names.index(q) for q in mujoco_joint_names if q in joint_names
]

if humanoid_type == "smpl":
    pose_aa = np.concatenate([motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))], axis=1)  # TODO: need to extract correct handle rotations instead of zero
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]
else:
    pose_aa = np.concatenate([motion_data['pose_aa'][:, :66], motion_data['pose_aa'][:, 75:]], axis=-1)
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 52, 4)[..., smpl_2_mujoco, :]

if isinstance(gender, np.ndarray):
    gender = gender.item()

if isinstance(gender, bytes):
    gender = gender.decode("utf-8")
if gender == "neutral":
    gender_number = [0]
elif gender == "male":
    gender_number = [1]
elif gender == "female":
    gender_number = [2]
else:
    import ipdb
    ipdb.set_trace()
    raise Exception("Gender Not Supported!!")


smpl_local_robot.load_from_skeleton(betas=betas[None, ], gender=gender_number, objs_info=None)
smpl_local_robot.write_xml(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")
skeleton_tree = SkeletonTree.from_mjcf(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")

root_trans_offset = torch.from_numpy(motion_data["trans"] + skeleton_tree.local_translation[0].numpy())

new_sk_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree, # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
    torch.from_numpy(pose_quat),
    root_trans_offset,
    is_local=True
)
sk_motion = SkeletonMotion.from_skeleton_state(
    new_sk_state,
    fps=mocap_fr
)

if robot_cfg["upright_start"]:
    B = pose_aa.shape[0]
    pose_quat_global = (sRot.from_quat(sk_motion.global_rotation.reshape(-1, 4).numpy()) *sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)



    global_translations = sk_motion.global_translation
    min_height = torch.min(global_translations[:, :, 0])  # take x axis (up axis) across all frames and joints

    trans = root_trans_offset.clone()
    trans[:, 0] -= min_height  # subtract offset from root x axis for all frames

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        trans,
        is_local=False)

    new_sk_motion = SkeletonMotion.from_skeleton_state(
        new_sk_state,
        fps=mocap_fr
    )
 
    print(f"Saving to {outputpath}")
    new_sk_motion.to_file(str(outputpath))
        



