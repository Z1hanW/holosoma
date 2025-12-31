from enum import Enum

import numpy as np
import torch
import socket
from torch import Tensor
import librosa
import joblib
from isaac_utils import torch_utils
from motion_tracking.utils.motion_lib import MotionLib
from motion_tracking.envs.utils.general import HistoryBuffer

from motion_tracking.envs.common.utils import (
    build_disc_observations
)
from motion_tracking.data.assets.skeleton_configs import get_num_jd_obs

from typing import List, Optional, Tuple, Union, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from motion_tracking.envs.isaacgym.disc_humanoid import DiscHumanoid
else:
    DiscHumanoid = object

from isaac_utils import torch_utils, rotations
import time

from motion_tracking.utils.urdfpoints import save_points
import os

class BaseDisc(DiscHumanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        self.save_count = 0
        

        if config.sync_motion:
            control_freq_inv = config.control_freq_inv
            self.sync_motion_dt = control_freq_inv / config.sim_fps
            print("HACK SLOW DOWN")
            config.pd_control = False

        self.state_init = self.StateInit[config.state_init]
        self.hybrid_init_prob = config.hybrid_init_prob
        self.num_disc_steps = config.num_disc_steps
        assert self.num_disc_steps >= 2

        self.reset_default_env_ids = []
        self.reset_ref_env_ids = []
        self.dual_task = False

        self.scene_in_disc = config.scene_in_disc
        self.start_id = 0
        self.statistics = False
        if self.statistics:
            
            self.ref_data = joblib.load("motion_data/demo_data_full.pkl")
            self.ref_data_mean = self.ref_data['mean']
            self.ref_data_std = self.ref_data['std']
            self.dtw_data = []
            '''
            (Pdb) obs.shape
            torch.Size([1, 232])
            (Pdb) root_h_obs.shape
            torch.Size([1, 1])
            (Pdb) root_rot_obs.shape
            torch.Size([1, 6])
            (Pdb) local_root_vel.shape
            torch.Size([1, 3])
            (Pdb) local_root_ang_vel.shape
            torch.Size([1, 3])
            (Pdb) dof_obs.shape
            torch.Size([1, 138])
            (Pdb) dof_vel.shape
            torch.Size([1, 69])
            (Pdb) flat_local_key_pos.shape
            torch.Size([1, 12])
            '''
            self.id_to_video = {
                0:'1533_1599', 1:'1962_2069', 2:'2793_2839', 3:'2840_2904', 4:'2905_3016', 5:'3017_3120', 6:'4118_4192', 7:'4193_4288',
                8:'4289_4340', 9:'4405_4462', 10:'5254_5299', 11:'5777_5816', 12:'5817_5918', 13:'6288_6345', 14:'6694_6750', 15:'6889_6943', 
                16:'7443_7488', 17:'8014_8070', 18:'8071_8116'
            }
            self.video_to_id = {v: k for k, v in self.id_to_video.items()}
            self.dtw_weights = np.concatenate([
                    np.full(1, 100),  # h
                    np.full(6, 10),  # root_rot
                    np.full(3, 10),  # root_vel
                    np.full(3, 5),   # root ang vel 
                    np.full(138, 1),  # dof obs
                    np.full(69, 1),  # dof vel
                    np.full(12, 10),  # end key pos
                    
                ])
            self.correct_or_not_list = []
            self.i_list = []
            self.D_list = []
        super().__init__(config, device)
      

        if self.scene_in_disc:
            pointcloud = save_points(os.path.join(self.asset_root, self.asset_file),"",save=False)
            self.pointcloud_disc = torch.from_numpy(pointcloud)[None].repeat(self.num_disc_steps * self.num_envs, 1, 1)
            self.pointcloud_disc = self.pointcloud_disc.to(device)

        
        self.add_noise = False
        self.noise_level = 0

        try:
            if config.noise_level!=0:
                self.noise_level = config.noise_level
                self.add_noise = True
        except:
            pass

        self.disc_hist_buf = HistoryBuffer(
            self.num_disc_steps,
            self.num_envs,
            shape=(self.num_disc_obs_per_step,),
            device=self.device,
        )
        self.motion_lib: MotionLib = self.instantiate_motion_lib(motion_lib)

        num_motions = self.motion_lib.num_sub_motions()
        motion_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        motion_ids = torch.remainder(motion_ids, num_motions)
        self.sync_motion_times = torch.zeros_like(self.motion_lib.state.motion_timings[motion_ids, 0])

    ###############################################################
    # Set up environment
    ###############################################################
    def small_random_quaternions(self, n: int, max_angle: float,) -> torch.Tensor:
        """
        Generate random quaternions representing small rotations.

        Args:
            n: Number of quaternions in a batch to return.
            max_angle: Maximum rotation angle in radians.
            dtype: Type to return.
            device: Desired device of returned tensor. Default: uses the current device for the default tensor type.

        Returns:
            Quaternions as tensor of shape (N, 4).
        """
        # Generate random rotation axis and small angle
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)  # Normalize axis
        angles = max_angle * torch.rand((n, 1), device=self.device)

        # Convert angle-axis to quaternion
        sin_half_angle = torch.sin(angles / 2)
        cos_half_angle = torch.cos(angles / 2)

        q = torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)  # Construct quaternion
        return q

    def setup_character_props(self):
        super().setup_character_props()

        self.num_disc_obs_per_step = get_num_jd_obs(self.config)
        if self.scene_in_disc:
            self.num_disc_obs_per_step += self.config.num_obs_num_point//1 * 3

    ###############################################################
    # Environment step logic
    ###############################################################
    def pre_physics_step(self, actions):
        if self.config.sync_motion:
            actions *= 0

        super().pre_physics_step(actions)

    def post_physics_step(self):
        super().post_physics_step()

        if self.config.sync_motion:
            self.sync_motion()

        if self.config.output_motion:
            self.output_motion()
        # import pdb;pdb.set_trace()
        if not self.config.disable_jd:
            self.disc_hist_buf.rotate()
            self.compute_disc_observations()

            disc_obs = self.make_disc_obs()
        else:
            disc_obs = None

        self.extras["disc_obs"] = disc_obs
        if self.statistics:
            

            self.dtw_data.append(disc_obs[0,:232].cpu().clone().numpy())
            root_states = self.get_humanoid_root_states()
            root_pos = root_states[:, :3]
            closest_id = torch.argmin(torch.norm(root_pos[:,None,:] - self.route,dim=-1),dim=-1).cpu().numpy()
            current_motion_index = np.searchsorted(self.sub_sub_route_id, closest_id, side='right') - 1
            # print(self.motion_index,current_motion_index)
            if current_motion_index!=self.motion_index:
                
                if len(self.dtw_data) > 30:
                    self.dtw_data = np.vstack(self.dtw_data)[:,:232]
                    
                    min_D=100000
                    min_D_i = 0
                    
                    X = self.dtw_data[:150] * self.dtw_weights
                    
                    video_name = "_".join(self.motion_lib.state.motion_files[self.motion_index[0]][:-4].split("_")[-2:])
                    gt_i = self.video_to_id[video_name]
                    X_length = int(len(X)*0.8)
                    # length = 
                    for i in range(19):
                        
                        
                        ref_data = self.ref_data[i]
                        Y = ref_data * self.dtw_weights
                        
                        
                        D_values = []
                        D_lengths = []
                        for j in range(3):
                            if j==0:
                                X_ = X[:X_length]
                            elif j==1:
                                X_ = X[-X_length:]
                            else:
                                X_ = X
                            # import pdb;pdb.set_trace()
                            
                            
                            # X = (X-self.ref_data_mean + 1e-6) / (self.ref_data_std+1e-6)
                            # Y = (Y-self.ref_data_mean + 1e-6) / (self.ref_data_std+1e-6)
                            # import pdb;pdb.set_trace()
                            D, wp = librosa.sequence.dtw(X=X_.T, Y=Y.T, metric='euclidean')
                            # import pdb;pdb.set_trace()
                            D_value = D[-1,-1]#/len(wp)
                            # print(i,D_value)
                            if D_value<min_D:
                                min_D = D_value
                                min_D_i = i 
                            D_values.append(D_value)
                            D_lengths.append(len(wp))
                        if i==gt_i:
                            D_gt_value = min(np.array(D_values)/np.array(len(wp)))

                    self.D_list.append(D_gt_value)
                    if min_D_i in [8,9] and gt_i in [8,9]:
                        self.correct_or_not_list.append(1)
                    elif min_D_i in [17,18] and gt_i in [17,18]:
                        self.correct_or_not_list.append(1)
                    elif min_D_i in [2,3] and gt_i in [2,3]:
                        self.correct_or_not_list.append(1)
                    else:
                        self.correct_or_not_list.append(min_D_i==gt_i)
                    
                    self.i_list.append(min_D_i)
                        
                    # print(f"closest motion id:{min_D_i}, current motion id:{gt_i}", self.correct_or_not_list[-1])
                    # import pdb;pdb.set_trace()
                # import pdb;pdb.set_trace()
                self.dtw_data = []
                self.motion_index = current_motion_index
                if len(self.correct_or_not_list)%10==0:
                    print("###",len(self.correct_or_not_list),np.mean(self.correct_or_not_list),np.mean(self.D_list))
                if len(self.correct_or_not_list)>=300:
                    import pdb;pdb.set_trace()
                # print(self.dtw_data.shape,self.dtw_weights.shape)
                

    ###############################################################
    # Getters
    ###############################################################
    def get_num_disc_obs(self):
        return self.num_disc_steps * self.num_disc_obs_per_step

    def get_task_obs_size(self):
        return 0

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_default(self, env_ids):
        super().reset_default(env_ids)
        self.reset_default_env_ids = env_ids

    def reset_actors(self, env_ids):
        if self.state_init == self.StateInit.Default:
            self.reset_default(env_ids)
        elif (
                self.state_init == self.StateInit.Start
                or self.state_init == self.StateInit.Random
        ):
            self.reset_ref_state_init(env_ids)
        elif self.state_init == self.StateInit.Hybrid:
            self.reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self.state_init)
            )

    def reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return

        self.reset_default_env_ids = []
        self.reset_ref_env_ids = []

        super().reset_envs(env_ids)

        if not self.config.disable_jd:
            self.reset_disc_hist_buf(env_ids)

    def reset_ref_state_init(
            self,
            env_ids,
            motion_ids: Optional[Tensor] = None,
            motion_times: Optional[Tensor] = None,
            object_ids: Optional[Tensor] = None,
            append_to_lists=False,
    ):
        # print(motion_times)
        num_envs = env_ids.shape[0]
        if motion_ids is None:
            motion_ids = self.motion_lib.sample_motions(num_envs)
            if self.state_init == self.StateInit.Start:
                motion_ids = (motion_ids // 5) * 0   + self.start_id  #
                # self.start_id += 5
                # if self.start_id == 240:
                #     self.start_id = 0
                # print(motion_ids)
        if object_ids is None:
            object_ids = self.sample_object_ids(motion_ids)

        if motion_times is None:
            if (
                    self.state_init == self.StateInit.Random
                    or self.state_init == self.StateInit.Hybrid
            ):
                max_steps = self.num_disc_steps
                
                motion_times = self.sample_time_without_negatives(
                    motion_ids, earliest_time=self.dt * max_steps
                )
                mask = torch.bernoulli(torch.full(motion_times.shape, 0.8, device=motion_times.device))
                motion_times = motion_times * mask
                # 

            elif self.state_init == self.StateInit.Start:
                motion_times = torch.zeros(num_envs, device=self.device)
            else:
                assert False, "Unsupported state initialization strategy: {:s}".format(
                    str(self.state_init)
                )

        # print(motion_ids, motion_times)
        # time.sleep(1)

        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel
        ) = self.motion_lib.get_motion_state(motion_ids, motion_times)
        
        if self.add_noise:
            
            # dof_pos += torch.randn_like(dof_pos) * 0.2
            start_id = motion_times == 0
            not_start_id = ~start_id

            if self.noise_level==1:
                root_rot_noise_start = self.small_random_quaternions(root_rot.shape[0], max_angle=15*3.14/180)
                root_rot_noise_not_start = self.small_random_quaternions(root_rot.shape[0], max_angle=10*3.14/180)

                root_transl_noise_start = torch.randn_like(root_pos) * 0.1
                root_transl_noise_not_start = torch.randn_like(root_pos) * 0.05

                dof_pos_noise_start = torch.randn_like(dof_pos) * 0.1
                dof_pos_noise_not_start = torch.randn_like(dof_pos) * 0.05

                dof_vel_noise_start = torch.randn_like(dof_vel) * 0.25
                dof_vel_noise_not_start = torch.randn_like(dof_vel) * 0.15

            if self.noise_level==2:
                root_rot_noise_start = self.small_random_quaternions(root_rot.shape[0], max_angle=18*3.14/180)
                root_rot_noise_not_start = self.small_random_quaternions(root_rot.shape[0], max_angle=12*3.14/180)

                root_transl_noise_start = torch.randn_like(root_pos) * 0.15
                root_transl_noise_start[:, 2] = torch.randn_like(root_pos)[:, 2] * 0.1
                root_transl_noise_not_start = torch.randn_like(root_pos) * 0.08

                dof_pos_noise_start = torch.randn_like(dof_pos) * 0.15
                dof_pos_noise_not_start = torch.randn_like(dof_pos) * 0.08

                dof_vel_noise_start = torch.randn_like(dof_vel) * 0.3
                dof_vel_noise_not_start = torch.randn_like(dof_vel) * 0.2
            
            root_pos[start_id] += root_transl_noise_start[start_id]
            dof_pos[start_id] += dof_pos_noise_start[start_id]
            dof_vel[start_id] += dof_vel_noise_start[start_id]

            root_pos[not_start_id] += root_transl_noise_not_start[not_start_id]
            dof_pos[not_start_id] += dof_pos_noise_not_start[not_start_id]
            dof_vel[not_start_id] += dof_vel_noise_not_start[not_start_id]


            root_rot[start_id] = rotations.quat_mul(root_rot_noise_start[start_id], root_rot[start_id], self.w_last)
            root_rot[not_start_id] = rotations.quat_mul(root_rot_noise_not_start[not_start_id], root_rot[not_start_id], self.w_last)
        
        


        root_offset = root_pos[:, :2].clone()
        # root_offset[:, 0] = 0
        # root_offset[:, 1] = -0.2

        root_pos[:, :2] = 0
        root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids, rb_pos=rb_pos, offset=root_offset, object_ids=object_ids
        )

        #js check
        # 
        # root_pos[:,0] += motion_ids
        # root_pos[:,1] += motion_ids

        
        root_pos = self.transfer_to_env_coordinates(root_pos, env_ids)
        dof_pos, dof_vel = self.convert_dof(dof_pos, dof_vel)

        rb_pos[:, :, :3] -= rb_pos[:, 0, :3].unsqueeze(1).clone()
        rb_pos[:, :, :3] += root_pos.unsqueeze(1)

        self.set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rb_pos=rb_pos,
            rb_rot=rb_rot,
            rb_vel=rb_vel,
            rb_ang_vel=rb_ang_vel,
        )

        if append_to_lists and len(self.reset_ref_env_ids) > 0:
            self.reset_ref_env_ids = torch.cat([env_ids, self.reset_ref_env_ids], dim=0)
            self.reset_ref_motion_ids = torch.cat(
                [motion_ids, self.reset_ref_motion_ids], dim=0
            )
            self.reset_ref_motion_times = torch.cat(
                [motion_times, self.reset_ref_motion_times], dim=0
            )
        else:
            self.reset_ref_env_ids = env_ids
            self.reset_ref_motion_ids = motion_ids
            self.reset_ref_motion_times = motion_times

            # self.dualenv_ref_motion_ids = motion_ids
            # self.dualenv_ref_motion_times = motion_times
            if self.dual_task:
                self.motion_ids[env_ids] = motion_ids
                self.motion_times[env_ids] = motion_times


    def reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = torch_utils.to_torch(
            np.array([self.hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self.reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self.reset_default(default_reset_ids)

    def reset_disc_hist_buf(self, env_ids):
        self.compute_disc_observations(env_ids)

        if len(self.reset_default_env_ids) > 0:
            self.reset_disc_hist_default(self.reset_default_env_ids)

        if len(self.reset_ref_env_ids) > 0:
            self.reset_disc_hist_ref(
                self.reset_ref_env_ids,
                self.reset_ref_motion_ids,
                self.reset_ref_motion_times,
            )

    def reset_disc_hist_default(self, env_ids):
        self.disc_hist_buf.set_hist(
            self.disc_hist_buf.get_current(env_ids), env_ids=env_ids
        )

    def reset_disc_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self.num_disc_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(0, self.num_disc_steps - 1, device=self.device) + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel
        ) = self.motion_lib.get_motion_state(motion_ids, motion_times)

        # Conversion
        dof_pos, dof_vel = self.convert_dof(dof_pos, dof_vel)

        disc_obs_demo = build_disc_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            torch.zeros(len(motion_ids), 1, device=self.device),
            self.local_root_obs,
            self.root_height_obs,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last
        )
        # 
        if self.scene_in_disc:
            disc_obs_demo = torch.cat([disc_obs_demo, self.build_disc_point_obs_demo(root_pos, root_rot, self.pointcloud[env_ids].repeat(self.num_disc_steps - 1, 1, 1))], dim=-1)

        self.disc_hist_buf.set_hist(
            disc_obs_demo.view(len(env_ids), self.num_disc_steps - 1, -1).permute(
                1, 0, 2
            ),
            env_ids,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def make_disc_obs(self):
        return self.disc_hist_buf.get_all_flattened()

    def sync_motion(self):
        num_motions = self.motion_lib.num_sub_motions()
        motion_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device) + self.config.fixed_motion_offset
        if self.config.get("fixed_motion_id", None):
            motion_ids *= 0
            motion_ids += self.config.fixed_motion_id

        if hasattr(self, "motion_ids"):
            self.motion_ids[:self.num_envs] = motion_ids
        if hasattr(self, "motion_times"):
            self.motion_times[:self.num_envs] = self.sync_motion_times
        
        motion_ids = torch.remainder(motion_ids, num_motions)

        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel
        ) = self.motion_lib.get_motion_state(
            motion_ids, self.sync_motion_times + self.motion_lib.state.motion_timings[motion_ids, 0]
        )
        object_ids = self.sample_object_ids(motion_ids, get_first_matching_object=True)

        root_vel *= 0
        root_ang_vel *= 0
        dof_vel *= 0
        rb_vel *= 0
        rb_ang_vel *= 0

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Transfer to the proper coordinates
        root_offset = root_pos[:, :2].clone()

        root_pos[:, :2] = 0
        root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids, rb_pos=rb_pos, offset=root_offset, object_ids=object_ids
        )

        root_pos = self.transfer_to_env_coordinates(root_pos, env_ids)
        dof_pos, dof_vel = self.convert_dof(dof_pos, dof_vel)

        rb_pos[:, :, :3] -= rb_pos[:, 0, :3].unsqueeze(1).clone()
        rb_pos[:, :, :3] += root_pos.unsqueeze(1)

        self.set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rb_pos=rb_pos,
            rb_rot=rb_rot,
            rb_vel=rb_vel,
            rb_ang_vel=rb_ang_vel,
        )

        self.reset_env_tensors(env_ids)

        motion_dur = self.motion_lib.state.motion_timings[motion_ids, 1] - self.motion_lib.state.motion_timings[motion_ids, 0]
        to_fmod = self.sync_motion_times + self.sync_motion_dt
        # if to_fmod >= motion_dur:
        #     self.hack_motion_done = True
        self.sync_motion_times = torch.fmod(to_fmod, motion_dur)

    def build_disc_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self.num_disc_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self.num_disc_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # motion ids above are "sub_motions" so first we map to motion file itself and then extract the length.
        lengths = self.motion_lib.state.motion_lengths[self.motion_lib.state.sub_motion_to_motion[motion_ids]]

        assert torch.all(motion_times >= 0)
        assert torch.all(motion_times <= lengths)

        
        
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel
        ) = self.motion_lib.get_motion_state(motion_ids, motion_times)
        
        # demo_data = []
        # for i in range(20):
        #     motion_ids_ = i
        #     lengths_ = self.motion_lib.state.motion_lengths[self.motion_lib.state.sub_motion_to_motion[motion_ids_]]
        #     num_steps = lengths_/self.dt
        #     query_motion_ids = torch.full((int(torch.round(num_steps+1)),),motion_ids_)
        #     query_motion_times = torch.arange(start=0, end=0 + int(torch.round(num_steps+1)) * self.dt, step=self.dt)
        #     # import pdb;pdb.set_trace()
        #     (
        #         root_pos,
        #         root_rot,
        #         dof_pos,
        #         root_vel,
        #         root_ang_vel,
        #         dof_vel,
        #         key_pos,
        #         rb_pos,
        #         rb_rot,
        #         rb_vel,
        #         rb_ang_vel
        #     ) = self.motion_lib.get_motion_state(query_motion_ids.to(self.device), query_motion_times.to(self.device))
        #     dof_pos, dof_vel = self.convert_dof(dof_pos, dof_vel)
        #     # import pdb;pdb.set_trace()
        #     disc_obs_demo = build_disc_observations(
        #         root_pos,
        #         root_rot,
        #         root_vel,
        #         root_ang_vel,
        #         dof_pos,
        #         dof_vel,
        #         key_pos,
        #         torch.zeros(len(query_motion_ids), 1, device=self.device),
        #         self.local_root_obs,
        #         self.root_height_obs,
        #         self.dof_obs_size,
        #         self.get_dof_offsets(),
        #         False,
        #         self.w_last
        #     )
        #     demo_data.append(disc_obs_demo.clone())
        # tensor_dict = {idx: tensor.cpu().numpy() for idx, tensor in enumerate(demo_data)}
        # joblib.dump(tensor_dict, "motion_data/demo_data.pkl")
        # import pdb;pdb.set_trace()
            
            
        dof_pos, dof_vel = self.convert_dof(dof_pos, dof_vel)

        disc_obs_demo = build_disc_observations(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            torch.zeros(len(motion_ids), 1, device=self.device),
            self.local_root_obs,
            self.root_height_obs,
            self.dof_obs_size,
            self.get_dof_offsets(),
            False,
            self.w_last
        )
        # 

        if self.scene_in_disc:
            # disc_obs_demo = torch.cat([disc_obs_demo, self.build_disc_point_obs_demo(root_pos[:65536//2], root_rot[:65536//2], self.pointcloud_disc[:65536//2])], dim=-1)
            # temp_point_disc_obs_demo1 = self.build_disc_point_obs_demo(root_pos[:65536//2], root_rot[:65536//2], self.pointcloud_disc[:65536//2])
            # temp_point_disc_obs_demo2 = self.build_disc_point_obs_demo(root_pos[65536//2:], root_rot[65536//2:], self.pointcloud_disc[65536//2:])
            mini_batch_size = root_pos.shape[0]  // 32
            for ii in range(32):
                if ii==0:
                    temp_point_disc_obs_demo = self.build_disc_point_obs_demo(root_pos[:mini_batch_size], root_rot[:mini_batch_size], self.pointcloud_disc[:])
                else:
                    temp_point_disc_obs_demo = torch.cat([temp_point_disc_obs_demo, self.build_disc_point_obs_demo(root_pos[mini_batch_size*ii:mini_batch_size*(ii+1)], root_rot[mini_batch_size*ii:mini_batch_size*(ii+1)], self.pointcloud_disc[:])])
            disc_obs_demo = torch.cat([disc_obs_demo, temp_point_disc_obs_demo], dim=-1)
        return disc_obs_demo

    def build_disc_point_obs_demo(self, root_pos, root_rot, points):

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)
        
        heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], points.shape[1], heading_rot.shape[1]))
        heading_rot_exp = torch.reshape(heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]))

        points_delta = points - root_pos.unsqueeze(-2)

        points_delta_flat = torch.reshape(points_delta, (points_delta.shape[0] * points_delta.shape[1], points_delta.shape[2]))

        local_points_pos = rotations.quat_rotate(heading_rot_exp, points_delta_flat, self.w_last)

        local_points_pos = torch.reshape(local_points_pos, points_delta.shape)

        dis = torch.norm(local_points_pos, dim=-1)
        
        _, indices = torch.topk(dis, self.num_closest_point//1, dim=1, largest=False)

        closest_local_points_pos = torch.gather(local_points_pos, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        closest_local_points_pos = closest_local_points_pos.reshape(closest_local_points_pos.shape[0],-1)

        return closest_local_points_pos
    
    def convert_dof(self, dof_pos, dof_vel):
        return dof_pos, dof_vel

    def sample_time_without_negatives(self, motion_ids: Tensor, earliest_time: float):
        """
        Samples time in the range [earliest_time, motion_lengths[motion_ids]].
        This is done so that calls to the motion lib get-state functions
        don't have any negative times in them, even after subtracting dts
        when calculating histories.
        """
        times = self.motion_lib.sample_time(motion_ids, truncate_time=earliest_time)
        return times + earliest_time

    def compute_disc_observations(self, env_ids=None):
        bodies_positions, bodies_rotations, bodies_velocities, bodies_ang_velocities = self.get_bodies_state()
        dof_pos, dof_vel = self.get_dof_state()
        key_body_pos = bodies_positions[:, self.key_body_ids, :]

        env_global_positions = self.convert_to_global_coords(bodies_positions[:, 0, :2], self.env_offsets[..., :2])


        if env_ids is None:
            disc_obs = build_disc_observations(
                bodies_positions[:, 0, :],
                bodies_rotations[:, 0, :],
                bodies_velocities[:, 0, :],
                bodies_ang_velocities[:, 0, :],
                dof_pos,
                dof_vel,
                key_body_pos,
                self.get_ground_heights_below_base(env_global_positions),
                self.local_root_obs,
                self.root_height_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last
            )
            if self.scene_in_disc:
                disc_obs = torch.cat([disc_obs, self.build_disc_point_obs_demo(bodies_positions[:, 0, :], bodies_rotations[:, 0, :], self.pointcloud)], dim=-1)
            self.disc_hist_buf.set_curr(disc_obs)
        else:
            disc_obs = build_disc_observations(
                bodies_positions[env_ids][:, 0, :],
                bodies_rotations[env_ids][:, 0, :],
                bodies_velocities[env_ids][:, 0, :],
                bodies_ang_velocities[env_ids][:, 0, :],
                dof_pos[env_ids],
                dof_vel[env_ids],
                key_body_pos[env_ids],
                self.get_ground_heights_below_base(env_global_positions)[env_ids],
                self.local_root_obs,
                self.root_height_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                False,
                self.w_last
            )
            if self.scene_in_disc:
                disc_obs = torch.cat([disc_obs, self.build_disc_point_obs_demo(bodies_positions[env_ids][:, 0, :], bodies_rotations[env_ids][:, 0, :], self.pointcloud[env_ids])], dim=-1)
            self.disc_hist_buf.set_curr(disc_obs, env_ids)

# def build_disc_point_obs_demo2(root_pos, root_rot, points, w_last=True, num_closest_point=60):

#         heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
        
#         heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], points.shape[1], heading_rot.shape[1]))
#         heading_rot_exp = torch.reshape(heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]))

#         points_delta = points - root_pos.unsqueeze(-2)

#         points_delta_flat = torch.reshape(points_delta, (points_delta.shape[0] * points_delta.shape[1], points_delta.shape[2]))

#         local_points_pos = rotations.quat_rotate(heading_rot_exp, points_delta_flat, w_last)

#         local_points_pos = torch.reshape(local_points_pos, points_delta.shape)

#         dis = torch.norm(local_points_pos, dim=-1)
        
#         _, indices = torch.topk(dis, num_closest_point//1, dim=1, largest=False)

#         closest_local_points_pos = torch.gather(local_points_pos, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
#         closest_local_points_pos = closest_local_points_pos.reshape(closest_local_points_pos.shape[0],-1)

#         return closest_local_points_pos
