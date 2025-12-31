from motion_tracking.agents.callbacks.base_callback import RL_EvalCallback
from motion_tracking.agents.ppo import PPO
from motion_tracking.envs.common.common_humanoid import BaseHumanoid

import torch
from isaac_utils import torch_utils
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import os.path as osp
import shutil

import numpy as np
import yaml
import pickle
from scipy.spatial.transform import Rotation as sRot


class ExportMotion(RL_EvalCallback):
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.record_dir = Path(config.record_dir)
        self.record_dir.mkdir(exist_ok=True, parents=True)

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: BaseHumanoid = self.training_loop.env
        self.env = env

    def on_post_evaluate_policy(self):
        self.write_recordings()

    def write_recordings(self):
        fps = np.round(1.0 / self.env.dt)
        for idx in range(self.env.num_envs):
            trajectory_data = self.env.motion_recording

            save_dir = self.record_dir / f"{(idx + self.config.index_offset):03d}"
            save_dir.mkdir(exist_ok=True, parents=True)

            if self.config.store_poselib:
                skeleton_tree = self.env.motion_lib.state.motions[0].skeleton_tree

                curr_root_pos = torch.stack([root_pos[idx] for root_pos in trajectory_data["root_pos"]])
                curr_root_pos[..., :2] = self.env.convert_to_global_coords(curr_root_pos[..., :2], self.env.env_offsets[idx, :2].view(1, 1, 2).cpu())
                curr_body_rot = torch.stack([global_rot[idx] for global_rot in trajectory_data["global_rot"]])

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, curr_body_rot, curr_root_pos, is_local=False
                )
                sk_motion = SkeletonMotion.from_skeleton_state(sk_state, fps=fps)

                sk_motion.to_file(str(save_dir / f"trajectory_poselib_{idx}.npy"))

                if "target_poses" in trajectory_data:
                    target_poses = torch.tensor(np.stack([target_pose[idx] for target_pose in trajectory_data["target_poses"]]))
                    if not hasattr(self.env, "export_motion_dont_convert_to_global"):
                        target_poses[..., :2] = self.env.convert_to_global_coords(target_poses[..., :2], self.env.env_offsets[idx, :2].view(1, 1, 2).cpu())
                    np.save(str(save_dir / f"target_poses_{idx}.npy"), target_poses.cpu().numpy())

                if hasattr(self.env, "object_ids") and self.env.object_ids[idx] >= 0:
                    object_id = self.env.object_ids[idx].item()
                    object_category, object_name = self.env.spawned_object_names[object_id].split("_")
                    object_offset = self.env.object_offsets[object_category]
                    object_pos = self.env.object_id_to_object_position[object_id].clone()
                    object_pos[0] += object_offset[0]
                    object_pos[1] += object_offset[1]

                    object_bbs = self.env.object_id_to_object_bounding_box[object_id].clone()

                    # Add the height offset for the bounding box to match in global coords
                    object_center_xy = self.env.object_root_states[object_id, :2].view(1, 2)
                    terrain_height_below_object = self.env.get_ground_heights_below_base(object_center_xy).view(1)
                    object_bbs[:, -1] += terrain_height_below_object

                    object_info = {
                        "object_pos": [object_pos[0].item(), object_pos[1].item(), object_pos[2].item()],
                        "object_name": object_name,
                        "object_bbox": object_bbs.cpu().tolist()
                    }
                    with open(str(save_dir / f"object_info_{idx}.yaml"), "w") as file:
                        yaml.dump(object_info, file)
                    category_root = osp.join(self.env.config.object_asset_root, object_category)
                    # copy urdf and obj files to new dir, using copy functions
                    shutil.copyfile(str(osp.join(category_root, f"{object_name}.urdf")), str(save_dir / f"{object_name}.urdf"))
                    shutil.copyfile(str(osp.join(category_root, f"{object_name}.obj")), str(save_dir / f"{object_name}.obj"))

            else:
                if "smpl" in self.env.config.asset.robot_type:
                    from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPLH_MUJOCO_NAMES
                    if self.env.config.asset.robot_type == "smpl":
                        mujoco_joint_names = SMPL_MUJOCO_NAMES
                        joint_names = SMPL_BONE_ORDER_NAMES
                    elif self.env.config.asset.robot_type == "smplx" or self.env.config.asset.robot_type == "smplh":
                        mujoco_joint_names = SMPLH_MUJOCO_NAMES
                        joint_names = SMPLH_BONE_ORDER_NAMES
                    else:
                        raise NotImplementedError

                    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]
                else:
                    raise NotImplementedError

                pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])

                body_quat = torch.stack(trajectory_data["rigid_body_rot"])[:, idx]
                root_trans = torch.stack(trajectory_data["rigid_body_pos"])[:, idx, 0, :]

                N = body_quat.shape[0]

                skeleton_tree = self.env.motion_lib.state.motions[0].skeleton_tree

                # offset = skeleton_tree.local_translation[0]
                # offset = root_trans[0].clone()
                # offset[2] = 0
                root_trans_offset = root_trans - skeleton_tree.local_translation[0]

                pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * pre_rot).as_quat().reshape(N, -1, 4)
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat),
                    root_trans.cpu(), is_local=False
                )
                local_rot = new_sk_state.local_rotation
                pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
                pose_aa = pose_aa[:, mujoco_2_smpl, :].reshape(1, N, -1)

                with open(save_dir / f"trajectory_pose_aa_{idx}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "pose": pose_aa,
                            "trans": root_trans_offset.unsqueeze(0).cpu().numpy(),
                            "shape": np.zeros((N, 10)),
                            "gender": "neutral"
                        },
                        f
                    )

            # Also save data needed for separate Viser robot visualization
            try:
                # Stack rigid body states over time for this env
                rb_pos = torch.stack(trajectory_data["rigid_body_pos"])[:, idx]   # (T, B, 3)
                rb_rot = torch.stack(trajectory_data["rigid_body_rot"])[:, idx]   # (T, B, 4) xyzw

                # Compute a world offset if available (e.g., triangle mesh tiling in MimicHumanoid)
                world_off = torch.zeros(3)
                if hasattr(self.env, "_tm_env_tile_offsets"):
                    try:
                        world_off = self.env._tm_env_tile_offsets[idx].cpu()
                    except Exception:
                        pass

                # Save positions in world coordinates for convenience
                rb_pos_world = rb_pos + world_off.view(1, 1, 3)

                np.savez(
                    str(save_dir / f"rigid_bodies_{idx}.npz"),
                    pos=rb_pos_world.numpy(),
                    rot=rb_rot.numpy(),  # xyzw
                    dt=float(self.env.dt),
                )

                # Save MJCF asset used and body name ordering if available
                asset_src = getattr(self.env, "asset_mjcf_path", None)
                if asset_src is not None:
                    dst_xml = save_dir / "robot_asset.xml"
                    try:
                        shutil.copyfile(asset_src, dst_xml)
                    except Exception:
                        pass

                body_names = getattr(self.env, "body_names", None)
                vis_info = {
                    "asset_xml": "robot_asset.xml" if (save_dir / "robot_asset.xml").exists() else None,
                    "body_names": body_names if body_names is not None else [],
                    "dt": float(self.env.dt),
                }
                with open(save_dir / "robot_vis_info.yaml", "w") as f:
                    yaml.safe_dump(vis_info, f)
            except Exception as e:
                print(f"[ExportMotion] Failed to export Viser robot data for env {idx}: {e}")

        for key in self.env.motion_recording.keys():
            self.env.motion_recording[key] = []
