import numpy as np
import smplx
import torch
import joblib
from scipy.spatial.transform import Rotation as sRot
import argparse
import os
def main(args):
    data_path = args.data_path
    parent_name = args.parent_name
    seq_name = args.scene
    types = args.types
    try:
      raw_data_path = f"{data_path}/{parent_name}/{seq_name}_{types}.npz"
      after_data_path = f"output/recordings/000/trajectory_pose_aa_0.pkl"
      raw_data = np.load(raw_data_path)
      after_data = joblib.load(after_data_path)
    except:
      return
    T_align = raw_data['T_align'].astype(np.float32)
    T_transform_back = np.linalg.inv(T_align).astype(np.float32)
    poses = torch.from_numpy(after_data['pose'][0]).float()
    transl = torch.from_numpy(after_data['trans'][0]).float()
    previous_shape = poses.shape[0]
    # 49:reference_motion: 707 RL out: 706
    poses = torch.cat([poses[0:1],poses], dim=0)
    transl = torch.cat([transl[0:1],transl], dim=0)
    betas = torch.from_numpy(raw_data['betas']).float()
    if len(betas) != len(poses):
      betas=betas[1:]
    print('reference motion:', betas.shape[0])
    # betas = torch.cat([betas[0:1], betas], dim=0)
    print('rendered motion:', transl.shape, poses.shape, betas.shape)
    # print(
    model = smplx.create("data/smpl/SMPL_NEUTRAL.pkl", model_type="smpl",
                            gender="neutral",
                            num_betas=10,
                            batch_size=1,
                            ext="pkl")
    output_after_rl = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:],
                    return_verts=True)
    vertices_after_rl = output_after_rl.vertices.detach().cpu()
    vertices = vertices_after_rl.numpy()
    transform = sRot.from_matrix(T_transform_back[:3,:3])
    new_root_rot = (transform * sRot.from_rotvec(poses[:, :3])).as_rotvec()
    poses[:, :3] = torch.from_numpy(new_root_rot)
    transl = transl @ transform.as_matrix().T + T_transform_back[:3,3]
    model = smplx.create("data/smpl/SMPL_NEUTRAL.pkl", model_type="smpl",
                            gender="neutral",
                            num_betas=10,
                            batch_size=1,
                            ext="pkl")
    output_temp = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:],
                    return_verts=True)
    vertices_temp = output_temp.vertices.detach().cpu()
    root_offset = vertices_after_rl @ T_transform_back[:3,:3].T + T_transform_back[:3,3] - vertices_temp
    root_offset = root_offset[:,0,:]
    transl = transl + root_offset
    output_final = model(betas=betas, transl = transl, global_orient = poses[:,:3], body_pose = poses[:,3:], return_verts=True)
    vertices = output_final.vertices.detach().cpu().numpy()
    afttter_shape = transl.shape[0]
    # assert previous_shape == afttter_shape
    # np.savez(f"{data_path}/{parent_name}/{seq_name}_afterrl_rotback.npz", trans=transl.numpy()[:], gender="neutral", betas=betas.numpy()[:], poses=poses.numpy()[:])
    org_path = '/data3/zihanwa3/_Robotics/humanoid/hybrid-imitation-parkour/post_results'
    np.savez(f"{org_path}/{seq_name}_afterrl_rotback.npz",
    trans=transl.numpy()[:],
    gender="neutral",
    betas=betas.numpy()[:],
    poses=poses.numpy()[:])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform SMPL sequence back to world coordinates.")
    parser.add_argument('--scene', required=True, help='Sequence name (e.g., 000)')
    parser.add_argument('--types', required=True, help='Sequence name (e.g., 000)')
    parser.add_argument('--parent_name', default='emdb_new', help='Parent folder name')
    parser.add_argument('--data_path', default='motion_data/', help='Root path to motion data')
    parser.add_argument('--after_data_path', default='output/recordings/000/trajectory_pose_aa_0.pkl', help='Path to optimized result')
    args = parser.parse_args()
    main(args)