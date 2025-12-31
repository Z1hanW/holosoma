import copy
import typer

import torch
import numpy as np
from smpllib.visualizer.vis_smpl import SMPLVisualizer
from smpllib.smpllib.smpl_parser import SMPL_Parser
from poselib.skeleton.skeleton3d import (
    SkeletonMotion,
    SkeletonState,
    SkeletonTree
)
from scipy.spatial.transform import Rotation as sRot
from isaac_utils import torch_utils
from smpllib.smpllib.utils import TMP_SMPL_DIR
from smpllib.smpllib.smpl_local_robot import Robot as LocalRobot
from smpllib.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from pathlib import Path
from parkour_anim.scripts.recordings.make_video_grid import assert_and_run


mujoco_joint_names = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
    'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder',
    'R_Elbow', 'R_Wrist', 'R_Hand'
]

def convert_amass_to_isaac(
        filename, outpath, force_neutral_body=True
):
    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="smpllib/data/smpl",
    )

    print(f"Processing {filename}")

    motion_data = np.load(filename, allow_pickle=True)

    betas = motion_data["shape"][0]
    gender = motion_data["gender"]
    pose = motion_data["pose"][0]
    trans = motion_data["trans"][0]

    # First fix height
    pose_aa = torch.tensor(pose)  # After sampling the bound
    trans = torch.tensor(trans)  # After sampling the bound
    betas = torch.from_numpy(betas)

    if force_neutral_body:
        betas[:] = 0
        gender = "neutral"

    motion_data = {
        "pose_aa": pose_aa.numpy(),
        "trans": trans.numpy(),
        "beta": betas.numpy(),
        "gender": gender,
    }

    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = motion_data["pose_aa"].shape[0]
    pose_aa = np.concatenate([motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
                             axis=1)  # TODO: need to extract correct handle rotations instead of zero

    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)[..., smpl_2_mujoco, :]

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

    # smpl_local_robot.load_from_skeleton(betas=betas[None,], gender=gender_number, objs_info=None)
    # smpl_local_robot.write_xml(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")
    # skeleton_tree = SkeletonTree.from_mjcf(f"{TMP_SMPL_DIR}/smpl_humanoid_1.xml")
    skeleton_tree = SkeletonTree.from_mjcf("parkour_anim/data/assets/mjcf/smpl_humanoid_no_limits.xml")

    root_trans_offset = torch.from_numpy(motion_data["trans"] + skeleton_tree.local_translation[0].numpy())

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
        torch.from_numpy(pose_quat),
        root_trans_offset,
        is_local=True
    )
    sk_motion = SkeletonMotion.from_skeleton_state(
        new_sk_state,
        fps=30
    )

    if robot_cfg["upright_start"]:
        B = pose_aa.shape[0]
        pose_quat_global = (
                    sRot.from_quat(sk_motion.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat(
                [0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)

        # new_sk_state = SkeletonState.from_rotation_and_root_translation(
        #     skeleton_tree,
        #     torch.from_numpy(pose_quat_global),
        #     root_trans_offset,
        #     is_local=False)

        ############################################################
        left_to_right_index = [
            0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14,
            15, 16, 17, 18
        ]
        # pose_quat_global = pose_quat_global[:, left_to_right_index]
        # pose_quat_global[..., 0] *= -1
        # pose_quat_global[..., 2] *= -1

        # root_trans_offset[..., 1] *= -1
        ###########################################################
    else:
        raise NotImplementedError

    global_translations = sk_motion.global_translation
    min_height = torch.min(
        global_translations[:, :, 0])  # take x axis (up axis) across all frames and joints

    trans = root_trans_offset
    trans[:, 0] -= min_height  # subtract offset from root x axis for all frames

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        torch.from_numpy(pose_quat_global),
        trans,
        is_local=False)

    sk_motion = SkeletonMotion.from_skeleton_state(
        sk_state,
        fps=30
    )

    # print(f"Saving to {outpath}")
    # sk_motion.to_file(str(outpath))
    return sk_motion

def main(path: str):
    smpl_model_dir = "smpllib/data/smpl"
    pyvista_visualizer = SMPLVisualizer(smpl_model_dir=smpl_model_dir, generator_func=None, distance=12, elevation=10, device="cpu", verbose=False, show_smpl=True, show_skeleton=False, enable_shadow=True)

    smpl = SMPL_Parser(model_path=smpl_model_dir, gender='neutral').to("cpu")

    #### Load and show data

    path = ".".join(path.split(".")[:-1])

    ## TODO: convert to isaac format
    fix_heights = True
    height_offset = 0
    start_offset = 0
    time_extra = 10
    outpath = "/tmp/mfm.npy"
    target_motion = convert_amass_to_isaac(path + ".pkl", outpath)
    # target_motion = SkeletonMotion.from_file(outpath)
    num_frames = target_motion.local_rotation.shape[0]

    longer_local_rotation = torch.zeros((num_frames + time_extra, target_motion.local_rotation.shape[1], target_motion.local_rotation.shape[2]))
    longer_local_rotation[:start_offset] = target_motion.local_rotation[0]
    longer_local_rotation[-(time_extra - start_offset):] = target_motion.local_rotation[-1]
    longer_local_rotation[start_offset:-(time_extra - start_offset)] = target_motion.local_rotation

    longer_root_translation = torch.zeros((num_frames + time_extra, target_motion.root_translation.shape[1]))
    longer_root_translation[:start_offset] = target_motion.root_translation[0]
    longer_root_translation[-(time_extra - start_offset):] = target_motion.root_translation[-1]
    longer_root_translation[start_offset:-(time_extra - start_offset)] = target_motion.root_translation

    if fix_heights:
        body_heights = target_motion.global_translation[..., 2]
        min_height = body_heights.min()
        new_height_offset = min_height + height_offset
    else:
        new_height_offset = 0

    target_motion = SkeletonMotion.from_skeleton_state(
        SkeletonState.from_rotation_and_root_translation(
            target_motion.skeleton_tree,
            longer_local_rotation,
            longer_root_translation - new_height_offset,
            is_local=True,
        ),
        fps=target_motion.fps,
    )

    min_initial_height = (target_motion.global_translation[0, :, 2] - new_height_offset).min()
    ref_respawn_offset = 0.1 - min_initial_height

    target_motion.to_file(outpath)

    recording_dir = Path(f"/tmp/recording")
    checkpoint = "results/mimic_llc/lightning_logs/a100_transformer/last.ckpt"
    extra_opts="mimic_llc_transformer,mimic_global_tracking,record_motion"
    backbone="isaacgym_smpl_no_limits"
    run_command = (
            f"python parkour_anim/eval_agent.py +exp=mimic +backbone={backbone} +opt=[mimic_llc,{extra_opts}]"
            + f" checkpoint={str(checkpoint)} +max_eval_steps={num_frames + 10}"
            + f" num_envs=1"
            + f" record_dir={str(recording_dir)}"
            + f" motion_file={str(outpath)}"
            + f" headless=True init_start_prob=1 ref_respawn_offset={ref_respawn_offset}"
    )

    assert_and_run(run_command)
    motion_recordings = [f.with_suffix(f".pkl") for f in list(recording_dir.rglob("*.pkl"))]

    motion_recordings[0].rename(f"{path}_physics.pkl")

    ## TODO: convert back to SMPL
    # smpl_seq = {}
    # from smpllib.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
    #
    # mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle',
    #                       'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder',
    #                       'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow',
    #                       'R_Wrist', 'R_Hand']
    # mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]
    #
    # pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
    #
    # skeleton_tree = target_motion.skeleton_tree
    #
    # N = target_motion.root_translation.shape[0]
    #
    # offset = target_motion.root_translation[0].clone()
    # offset[2] = 0
    # root_trans = target_motion.root_translation
    #
    # root_trans_offset = root_trans - offset
    # body_quat = target_motion.global_rotation
    #
    # pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot).as_quat().reshape(N, -1, 4)
    # new_sk_state = SkeletonState.from_rotation_and_root_translation(
    #     skeleton_tree,
    #     torch.from_numpy(pose_quat),
    #     root_trans_offset.cpu(), is_local=False
    # )
    # local_rot = new_sk_state.local_rotation
    # pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
    # pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(1, N, -1)

    ## TODO: visualize both side by side to make sure they are identical
    # datafile = np.load(path + ".pkl", allow_pickle=True)
    # smpl_pose_gt = torch.tensor(datafile["pose"])
    # smpl_trans_gt = torch.tensor(datafile["trans"])
    #
    # pose_tmp = smpl_pose_gt.view(-1, 72)
    # smpl_verts_gt, smpl_joints_gt = smpl.get_joints_verts(
    #     pose=pose_tmp,
    #     th_betas=torch.zeros_like(pose_tmp[..., :10]),
    #     th_trans=smpl_trans_gt.view(-1, 3),
    # )
    # smpl_verts_gt = smpl_verts_gt.view(*smpl_pose_gt.shape[:2], -1, 3)
    # smpl_joints_gt = smpl_joints_gt.view(*smpl_pose_gt.shape[:2], -1, 3)
    #
    # num_frames = smpl_pose_gt.shape[1] if smpl_pose_gt is not None else smpl_joints_gt.shape[1]
    #
    # smpl_seq_gt = {
    #     "shape": torch.zeros((1, num_frames, 10)).to(smpl_joints_gt.device),
    #     "joints_pos": smpl_joints_gt[0],
    #     "gender": "neutral",
    #     "orig_num_frames": num_frames,
    #     "pose": smpl_pose_gt[0].float(),
    #     "trans": smpl_trans_gt[0]
    # }
    #
    # smpl_seq["gt"] = smpl_seq_gt
    #
    # smpl_pose_pred = torch.tensor(pose_aa)
    # smpl_trans_pred = torch.tensor(root_trans_offset)
    #
    # smpl_trans_pred[:, :2] +=  smpl_trans_gt[0, 0, :2]
    #
    # pose_tmp = smpl_pose_pred.view(-1, 72)
    # smpl_verts_pred, smpl_joints_pred = smpl.get_joints_verts(
    #     pose=pose_tmp,
    #     th_betas=torch.zeros_like(pose_tmp[..., :10]),
    #     th_trans=smpl_trans_pred.view(-1, 3),
    # )
    # smpl_verts_pred = smpl_verts_pred.view(*smpl_pose_pred.shape[:2], -1, 3)
    # smpl_joints_pred = smpl_joints_pred.view(*smpl_pose_pred.shape[:2], -1, 3)
    #
    # num_frames = smpl_pose_pred.shape[1] if smpl_pose_pred is not None else smpl_joints_pred.shape[1]
    #
    # smpl_seq_pred = {
    #     "shape": torch.zeros((1, num_frames, 10)).to(smpl_joints_pred.device),
    #     "joints_pos": smpl_joints_pred[0],
    #     "gender": "neutral",
    #     "orig_num_frames": num_frames,
    #     "pose": smpl_pose_pred[0].float(),
    #     "trans": smpl_trans_pred
    # }
    #
    # smpl_seq["pred"] = smpl_seq_pred
    #
    # pyvista_visualizer.show_animation(init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1000, 1000))

    smpl_seq = {}

    for ending, name in [(".pkl", "gt"), ("_physics.pkl", "pred")]:
        datafile = np.load(path + ending, allow_pickle=True)

        start_frame = 0
        if name == "pred":
            start_frame = start_offset

        smpl_pose = torch.tensor(datafile["pose"][:, start_frame:start_frame+num_frames])
        smpl_trans = torch.tensor(datafile["trans"][:, start_frame:start_frame+num_frames])

        if name == "gt":
            root_trans_offset = smpl_trans[0, 0].clone()
            root_trans_offset[2] = 0
        else:
            smpl_trans += root_trans_offset

        pose_tmp = smpl_pose.view(-1, 72)
        smpl_verts, smpl_joints = smpl.get_joints_verts(
            pose=pose_tmp,
            th_betas=torch.zeros_like(pose_tmp[..., :10]),
            th_trans=smpl_trans.view(-1, 3),
        )
        smpl_verts = smpl_verts.view(*smpl_pose.shape[:2], -1, 3)
        smpl_joints = smpl_joints.view(*smpl_pose.shape[:2], -1, 3)

        num_frames = smpl_pose.shape[1] if smpl_pose is not None else smpl_joints.shape[1]

        smpl_seq_new = {
            "shape": torch.zeros((1, num_frames, 10)).to(smpl_joints.device),
            "joints_pos": smpl_joints[0],
            "gender": "neutral",
            "orig_num_frames": num_frames,
            "pose": smpl_pose[0].float(),
            "trans": smpl_trans[0]
        }

        smpl_seq[name] = smpl_seq_new

    pyvista_visualizer.show_animation(init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1000, 1000))
    # video_file = f'{path}_physics.mp4'
    # pyvista_visualizer.save_animation_as_video(video_file, init_args={'smpl_seq': smpl_seq, 'mode': 'gt'}, window_size=(1000, 1000))

if __name__ == "__main__":
    typer.run(main)
