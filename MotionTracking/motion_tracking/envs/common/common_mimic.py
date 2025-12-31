import torch
from torch import Tensor
import numpy as np
from motion_tracking.utils.motion_lib import MotionLib

import math
import time
from isaac_utils import torch_utils, rotations

from motion_tracking.envs.utils.general import StepTracker
from motion_tracking.envs.common.utils import build_disc_observations

from typing import Optional, List, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from motion_tracking.envs.isaacgym.mimic_humanoid import MimicHumanoid
else:
    MimicHumanoid = object


class BaseMimic(MimicHumanoid):
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        super().__init__(config, device, motion_lib=motion_lib)
            
        # Used by the tl in eval.
        self.disable_reset_track = False

        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_times = torch.zeros(
            self.num_envs, device=self.device
        )
        self.object_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        ) - 1

        if self.config.use_phase_obs:
            self.mimic_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        if self.config.provide_future_states:
            self.mimic_target_poses = torch.zeros(
                self.num_envs, self.config.num_future_steps * self.config.num_obs_per_target_pose,
                dtype=torch.float, device=self.device
            )

        self.seperate_point_goal = config.seperate_point_goal
        self.mimic_goal = None
        wants_point_cloud = self.seperate_point_goal or getattr(self, "force_point_cloud_obs", False)
        if wants_point_cloud:
            if not self.voxel:
                dim = self.config.num_obs_mimic_scene * self.config.num_obs_num_point
            else:
                dim = 1000
            self.mimic_scene = torch.zeros(
                self.num_envs, dim,
                dtype=torch.float, device=self.device
            )
            if self.seperate_point_goal and self.use_mimic_goal_obs and self.config.num_obs_mimic_goal > 0:
                self.mimic_goal = torch.zeros(
                        self.num_envs, self.config.num_obs_mimic_goal,
                        dtype=torch.float, device=self.device
                    )
        else:
            self.mimic_scene = torch.zeros(
                    self.num_envs, self.config.num_obs_mimic_scene,
                    dtype=torch.float, device=self.device
                )

        self.failed_due_bad_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        if self.config.dynamic_sample:
            self.setup_dynamic_sampling()

        # Sampling vectors
        self.init_start_probs = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        ) * self.config.init_start_prob

        self.reset_track_steps = StepTracker(
            self.num_envs,
            min_steps=self.config.reset_track_steps_min,
            max_steps=self.config.reset_track_steps_max,
            device=self.device,
        )

        self.reset_track(torch.arange(0, self.num_envs, device=self.device, dtype=torch.long))

        self.respawn_offsets = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        self.respawned_on_flat = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def setup_dynamic_sampling(self):
        num_buckets_list = []
        bucket_motion_ids_list = []
        bucket_starts_list = []
        bucket_lengths_list = []
        bucket_width = self.config.bucket_width


        start_times = self.motion_lib.state.motion_timings[:, 0].tolist()
        # End-time is evaluated AFTER the step, this means motions can reach 'end_time + self.dt'.
        # When mapping to bucket this can result in a shift to the next motion (or if it's the last motion, crash).
        # Fixed by adding self.dt to all motions for end_times when constructing the buckets.
        end_times = (self.motion_lib.state.motion_timings[:, 1] + self.dt).tolist()
        for motion_id, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            if self.config.fixed_motion_id is not None:
                # When training on a fixed motion, only add buckets for that motion.
                if motion_id != self.config.fixed_motion_id:
                    continue

            length = end_time - start_time
            buckets = math.ceil(length / bucket_width)
            num_buckets_list.append(buckets)
            for j in range(buckets):
                bucket_motion_ids_list.append(motion_id)
                start = j * bucket_width + start_time
                end = min(end_time, start + bucket_width)
                assert end - start > 0
                bucket_starts_list.append(start)
                bucket_lengths_list.append(end - start)

        num_buckets = torch.tensor(
            num_buckets_list, dtype=torch.long, device=self.device
        )

        rolled = num_buckets.roll(1)
        rolled[0] = 0
        self.bucket_offsets = rolled.cumsum(0)

        total_num_buckets = sum(num_buckets_list)

        self.bucket_scores = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.device
        )
        self.bucket_frames_spent = torch.zeros(
            total_num_buckets, dtype=torch.long, device=self.device
        )

        # Minimal score equivalent to 1/30000 frames failing (1 fail in 1000 seconds, ~30 mins).
        # Minimal score just serves as a non-zero value to ensure we re-sample all motions.
        self._min_bucket_weight = 1./30000
        self.bucket_weights = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.device
        )

        # The motion id each bucket corresponds to.
        self.bucket_motion_ids = torch.tensor(
            bucket_motion_ids_list, dtype=torch.long, device=self.device
        )

        # The start time (in seconds) of the bucket with
        # respect to its corresponding motion.
        self.bucket_starts = torch.tensor(
            bucket_starts_list, dtype=torch.float, device=self.device
        )

        # The length of each bucket, in seconds (some bucket
        # at the end of clips may be shorter than the set
        # bucket width).
        self.bucket_lengths = torch.tensor(
            bucket_lengths_list, dtype=torch.float, device=self.device
        )

    def dynamic_sample(self, n: int):
        weights = self.bucket_weights.clone().clamp(min=self._min_bucket_weight)
        weights[self.bucket_weights == 0] = 1.0

        min_weight = weights.min()
        norm_weight = weights / min_weight
        if False: # TODO: bring back self.config.dynamic_weight_max is not None:
            norm_weight = norm_weight.clamp(max=self.config.dynamic_weight_max)

        chosen_bucket_indices = torch.multinomial(
            norm_weight, num_samples=n, replacement=True
        )
        
        motion_ids = self.bucket_motion_ids[chosen_bucket_indices]
        bucket_starts = self.bucket_starts[chosen_bucket_indices]
        bucket_lengths = self.bucket_lengths[chosen_bucket_indices]

        motion_times = (
            torch.rand(n, device=self.device) * bucket_lengths + bucket_starts
        )

        return motion_ids, motion_times

    def update_dynamic_stats(self):
        # When training with terrains, make sure we only oversample if motion failed on "simple" setting.
        if torch.any(self.respawned_on_flat):
            on_flat_motion_ids = self.motion_ids[self.respawned_on_flat]
            on_flat_motion_times = self.motion_times[self.respawned_on_flat]
            on_flat_failed_due_bad_reward = self.failed_due_bad_reward[self.respawned_on_flat]
            on_flat_rew_buf = self.rew_buf[self.respawned_on_flat]

            if self.config.fixed_motion_id is not None:
                # If we're using a fixed motion, all buckets will correspond to that motion.
                bucket_indices = torch.zeros_like(on_flat_motion_ids)
            else:
                bucket_indices = on_flat_motion_ids

            base_offsets = self.bucket_offsets[bucket_indices]
            # A motion doesn't necessarily start at 0.
            sub_motion_delta = on_flat_motion_times - self.motion_lib.state.motion_timings[on_flat_motion_ids, 0]
            # A motion can span multiple buckets. Find which bucket the current portion of the motion corresponds to.
            extra_offsets = torch.floor(sub_motion_delta / self.config.bucket_width).long()
            bucket_indices = base_offsets + extra_offsets

            # NOTE These two lines
            # self.bucket_frames_spent[bucket_indices] += 1
            # self.bucket_scores[bucket_indices] += self.rew_buf
            # are NOT what we want, see
            # https://discuss.pytorch.org/t/how-to-do-atomic-add-on-slice-with-duplicate-indices/136193

            self.bucket_frames_spent.scatter_add_(
                0, bucket_indices, torch.ones_like(bucket_indices)
            )
            if self.config.dynamic_weight_criteria == "early_reward_term":
                self.bucket_scores.scatter_add_(0, bucket_indices, on_flat_failed_due_bad_reward)
            elif self.config.dynamic_weight_criteria == "reward":
                self.bucket_scores.scatter_add_(0, bucket_indices, on_flat_rew_buf)
            else:
                raise NotImplementedError("Dynamic weight criteria can be either 'early_reward_term' or 'reward")

    def refresh_dynamic_weights(self):
        visited = self.bucket_frames_spent > 0
        average_score = self.bucket_scores[visited] / self.bucket_frames_spent[visited]
        if self.config.dynamic_weight_criteria == "early_reward_term":
            weight = torch.pow(average_score, self.config.dynamic_weight_pow)
        elif self.config.dynamic_weight_criteria == "reward":
            weight = torch.pow(1 / average_score, self.config.dynamic_weight_pow)
        else:
            raise NotImplementedError("Dynamic weight criteria can be either 'early_reward_term' or 'reward")

        self.bucket_weights[visited] = torch.clamp(weight + self.bucket_weights[visited] * 0.7, min=self._min_bucket_weight)

        tensors_of_interest = {
            "bucket_frames_spent": self.bucket_frames_spent.float(),
            "bucket_average_score": average_score,
            "bucket_scores": self.bucket_scores,
            "bucket_weights": self.bucket_weights,
            "bucket_added_weights": weight
        }

        for k, v in tensors_of_interest.items():
            if v.shape[0] > 0:
                self.log_dict[f"{k}_min"] = v.min()
                self.log_dict[f"{k}_max"] = v.max()
                self.log_dict[f"{k}_mean"] = v.mean()

        self.bucket_frames_spent[:] = 0
        self.bucket_scores[:] = 0

    def reset_track(self, env_ids, new_motion_ids=None):
        if self.disable_reset_track:
            return

        if self.config.fixed_motion_per_env:
            new_motion_ids = torch.fmod(
                env_ids + self.config.fixed_motion_offset, self.motion_lib.num_sub_motions()
            )
            new_times = self.motion_lib.state.motion_timings[new_motion_ids, 0]
        elif self.config.dynamic_sample and new_motion_ids is None:
            new_motion_ids, new_times = self.dynamic_sample(len(env_ids))
            # find all indices shared in object_interaction_motion_ids and new_motion_ids
            # for object interaction ids, sample a random time since prioritization may lead to over-sampling imperfect
            # object-human positioning

            # if len(self.motion_lib.motion_to_object_ids) > 0 and self.total_num_objects > 0:
            #     object_interaction_mask = self.motion_lib.objects_per_motion[new_motion_ids] != -1
            #     if torch.any(object_interaction_mask):
            #         # # Shift back to 2 seconds to avoid spawning over an object.
            #         # shifted_time = new_times[object_interaction_mask] - 2
            #         # # Ensure at least self.dt time from the start
            #         # clamped_time = torch.clamp(shifted_time, min=self.motion_lib.state.motion_timings[new_motion_ids[object_interaction_mask], 0] + self.dt)
            #         # new_times[object_interaction_mask] = clamped_time
            #
            #         new_times[object_interaction_mask] = self.motion_lib.sample_time(
            #             new_motion_ids[object_interaction_mask], truncate_time=self.dt
            #         )
        else:
            if new_motion_ids is None:
                new_motion_ids = self.motion_lib.sample_motions(len(env_ids))
            if self.config.fixed_motion_id is not None:
                new_motion_ids = torch.zeros_like(new_motion_ids) + self.config.fixed_motion_id
            new_times = self.motion_lib.sample_time(
                new_motion_ids, truncate_time=self.dt
            )
        # 
        if self.config.init_start_prob > 0:
            init_start = torch.bernoulli(self.init_start_probs[:len(env_ids)])
            new_times = torch.where(
                init_start == 1, self.motion_lib.state.motion_timings[new_motion_ids, 0], new_times
            )
        # print(new_times)#
        # new_times[0] = 0.15
        # time.sleep(1)
        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times

        self.object_ids[env_ids] = self.sample_object_ids(self.motion_ids[env_ids])

        self.reset_track_steps.reset_steps(env_ids)

        if not self.config.headless:
            self.randomize_color(env_ids)

        return new_motion_ids, new_times

    def reset_actors(self, env_ids):
        if env_ids.shape[0] > 0:
            # On reset actor, shift the counter backwards to reset the grace period
            
            self.reset_track_steps.shift_counter(env_ids, self.reset_track_steps.steps[env_ids])

        if self.config.reset_track_on_reset:
            self.reset_track(env_ids)

        self.reset_ref_state_init(
            env_ids,
            motion_ids=self.motion_ids[env_ids],
            motion_times=self.motion_times[env_ids],
            object_ids=self.object_ids[env_ids],
        )

    def get_envs_respawn_position(self, env_ids, offset=0, rb_pos: torch.tensor=None, object_ids: torch.tensor=None):
        respawn_position = super().get_envs_respawn_position(
            env_ids, offset=offset, rb_pos=rb_pos, object_ids=object_ids
        )

        (target_cur_gt, _, _, _, _, _) = self.motion_lib.get_mimic_motion_state(
            self.motion_ids[env_ids], self.motion_times[env_ids]
        )
        target_cur_root_pos = target_cur_gt[:, 0, :]

        self.respawn_offsets[env_ids, :2] = respawn_position[:, :2] - target_cur_root_pos[:, :2]

        if self.terrain is not None:
            # Check if spawned on flat, for prioritized sampling
            global_respawn_position = self.convert_to_global_coords(respawn_position, self.env_offsets[env_ids])
            new_root_pos = global_respawn_position[..., :2].clone().reshape(env_ids.shape[0], 1, 2)
            new_root_pos -= self.terrain_offset
            new_root_pos = (new_root_pos / self.terrain.horizontal_scale).long()
            px = new_root_pos[:, :, 0].view(-1)
            py = new_root_pos[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            self.respawned_on_flat[env_ids] = self.terrain.flat_field_raw[px, py] == 0
            # if object motion -- also consider as "flat"
            if object_ids is not None:
                object_interaction_envs_mask = object_ids != -1
                self.respawned_on_flat[env_ids[object_interaction_envs_mask]] = True
        else:
            self.respawned_on_flat[env_ids] = True

        return respawn_position

    def compute_reset(self):
        super().compute_reset()

        if self.config.early_reward_term is not None:
            reward_too_bad = torch.zeros_like(self.reset_buf).bool()
            
            for entry in self.config.early_reward_term:
                
                if entry.get("from_other", False):
                    from_dict = self.last_other_rewards
                elif entry.use_scaled:
                    from_dict = self.last_scaled_rewards
                else:
                    from_dict = self.last_unscaled_rewards

                if entry.less_than:
                    entry_too_bad = (
                        from_dict[entry.early_reward_term_key]
                        < entry.early_reward_term_thresh
                    )
                    entry_on_flat_too_bad = (
                        from_dict[entry.early_reward_term_key]
                        < entry.early_reward_term_thresh_on_flat
                    )
                else:
                    entry_too_bad = (
                        from_dict[entry.early_reward_term_key]
                        > entry.early_reward_term_thresh
                    )
                    entry_on_flat_too_bad = (
                        from_dict[entry.early_reward_term_key]
                        > entry.early_reward_term_thresh_on_flat
                    )

                no_object_interaction = self.object_ids[:] < 0
                tight_tracking_threshold = torch.logical_and(no_object_interaction, self.respawned_on_flat)

                entry_too_bad[tight_tracking_threshold] = entry_on_flat_too_bad[tight_tracking_threshold]

                reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)

            has_reset_grace = self.reset_track_steps.steps <= self.config.reset_track_grace_period

            reward_too_bad = torch.logical_and(
                reward_too_bad,
                torch.logical_not(has_reset_grace)
            )
            self.failed_due_bad_reward[:] = 0
            self.failed_due_bad_reward[reward_too_bad] = 1

            self.reset_buf[reward_too_bad] = 1
            self.terminate_buf[reward_too_bad] = 1

            self.log_dict["reward_too_bad"] = reward_too_bad.float().mean()

        end_times = self.motion_lib.state.motion_timings[self.motion_ids, 1]
        done_clip = (self.motion_times + self.dt) > end_times
        
        if self.config.reset_on_reset_track:
            self.reset_buf[done_clip] = 1
        done_ids = torch.nonzero(done_clip == 1, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_track(done_ids)

    def handle_reset_track(self):
        self.reset_track_steps.advance()
        reset_track_ids = self.reset_track_steps.done_indices()

        if len(reset_track_ids) > 0:
            self.reset_track(reset_track_ids)

            if self.config.reset_on_reset_track:
                self.reset_buf[reset_track_ids] = 1

    def post_physics_step(self):
        self.motion_times += self.dt
        end_times = self.motion_lib.state.motion_timings[self.motion_ids, 1]
        start_times = self.motion_lib.state.motion_timings[self.motion_ids, 0]
        
        super().post_physics_step()

        # Don't update stats while in eval mode.
        if self.config.dynamic_sample and not self.disable_reset:
            self.update_dynamic_stats()

        # Remove start time before fmod and then add it back.
        self.motion_times = torch.fmod(
            self.motion_times - start_times,
            end_times - start_times
        ) + start_times
        # 
        if not self.disable_reset_track:
            self.handle_reset_track()

    def store_motion_data(self, skip=False):
        super().store_motion_data()
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        (target_pos, _, _, _, _, _) = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos += self.respawn_offsets.clone().view(self.num_envs, 1, 3)

        self.motion_recording["target_poses"].append(target_pos.cpu().numpy())


    def process_kb(self, gt: Tensor, gr: Tensor):
        kb = gt[:, self.key_body_ids]
        # 
        if self.config.relative_kb_pos:
            rt = gt[:, 0]
            rr = gr[:, 0]
            kb = kb - rt.unsqueeze(1)

            heading_rot = torch_utils.calc_heading_quat_inv(rr, self.w_last)
            rr_expand = heading_rot.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
            kb = rotations.quat_rotate(
                rr_expand.reshape(-1, 4), kb.view(-1, 3), self.w_last
            ).view(kb.shape)
        # else:
        #     rr_expand = rr.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
        #     kb = rotations.quat_rotate_inverse(rr_expand.reshape(-1, 4), kb.view(-1, 3), self.w_last).view(
        #         kb.shape
        #     )

        return kb

    def rotate_pos_to_local(self, pos: Tensor, heading: Optional[Tensor] = None):
        if heading is None:
            # root_rot = self.rigid_body_rot[:, 0]
            root_rot = self.get_bodies_state[1][:, 0]
            heading = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

        pos_num_dims = len(pos.shape)
        expanded_heading = heading.view(
            [heading.shape[0]] + [1] * (pos_num_dims - 2) + [heading.shape[1]]
        ).expand(pos.shape[:-1] + (4,))

        rotated = rotations.quat_rotate(
            expanded_heading.reshape(-1, 4), pos.reshape(-1, 3), self.w_last
        ).view(pos.shape)
        return rotated

    def compute_reward(self, actions):
        """
        Abbreviations:

        gt = global translation
        gr = global rotation
        rt = root translation
        rr = root rotation
        kb = key bodies
        dv = dof (degrees of freedom velocity)
        """
        # print(self.motion_ids, self.motion_times)
        (
            ref_gt,
            ref_gr,
            ref_dp,
            ref_gv,
            ref_gav,
            ref_dv
        ) = self.motion_lib.get_mimic_motion_state(self.motion_ids, self.motion_times)

        all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

        ref_gt = self.transfer_to_env_coordinates(ref_gt, all_env_ids)
        ref_dp, ref_dv = self.convert_dof(ref_dp, ref_dv)

        ref_lr = dof_to_local(ref_dp, self.get_dof_offsets(), self.w_last)

        ref_kb = self.process_kb(ref_gt, ref_gr)

        gt, gr, gv, gav = self.get_bodies_state()
        # first remove height based on current position
        env_global_positions = self.convert_to_global_coords(gt[:, 0, :2], self.env_offsets[..., :2])
        gt[:, :, -1:] -= self.get_ground_heights_below_base(env_global_positions).view(self.num_envs, 1, 1)
        # then remove offset to get back to the ground-truth data position
        gt[..., :2] -= self.respawn_offsets.clone()[..., :2].view(self.num_envs, 1, 2)

        kb = self.process_kb(gt, gr)

        rt = gt[:, 0]
        ref_rt = ref_gt[:, 0]

        if self.config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, self.w_last)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, self.w_last)

        rv = gv[:, 0]
        ref_rv = ref_gv[:, 0]

        rav = gav[:, 0]
        ref_rav = ref_gav[:, 0]

        dp, dv = self.get_dof_state()

        lr = dof_to_local(dp, self.get_dof_offsets(), self.w_last)

        if self.config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]

            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        rew_dict = self.exp_tracking_reward(
            gt=gt,
            rt=rt,
            kb=kb,
            gr=gr,
            lr=lr,
            rv=rv,
            rav=rav,
            gv=gv,
            gav=gav,
            dv=dv,
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
        )

        current_contact_forces = self.get_bodies_contact_buf()
        forces_delta = torch.clip(self.prev_contact_forces - current_contact_forces, min=0)[:, self.contact_body_ids, 2]  # get the Z axis
        kbf_rew = forces_delta.sum(-1).mul(self.config.kbf_rew_c).exp()

        rew_dict["kbf_rew"] = kbf_rew

        if self.config.backbone == "isaacgym":
            # TODO: support power reward for IsaacSim
            power = torch.abs(torch.multiply(self.dof_force_tensor, self.dof_vel)).sum(dim=-1)
            pow_rew = - power

            rew_dict["pow_rew"] = pow_rew

        self.last_scaled_rewards: Dict[str, Tensor] = {
            k: v * getattr(self.config, f"{k}_w") for k, v in rew_dict.items()
        }
        self.last_scaled_rewards['pow_rew'] = torch.clamp(self.last_scaled_rewards['pow_rew'],min=-0.5)
        tracking_rew = sum(self.last_scaled_rewards.values())

        self.rew_buf = tracking_rew + self.config.positive_constant

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        for rew_name, rew in self.last_scaled_rewards.items():
            self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(gt, inv_heading)
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean(-1)
        )

        translation_mask_coeff = 1. / self.num_bodies
        rotation_mask_coeff = 1. / self.num_bodies

        gt_err = (ref_gt - gt).pow(2).sum(-1).sqrt().sum(-1).mul(translation_mask_coeff)
        max_joint_err = (ref_gt - gt).pow(2).sum(-1).sqrt().max(-1)[0]

        gr_err = quat_diff_norm(gr, ref_gr, self.w_last).sum(-1).mul(rotation_mask_coeff)
        gr_err_degrees = gr_err * 180 / torch.pi

        other_log_terms = {
            "tracking_rew": tracking_rew,
            "total_rew": self.rew_buf,
            "cartesian_err": cartesian_err,
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "max_joint_err": max_joint_err
        }
        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"{rew_name}_mean"] = rew.mean()
            self.log_dict[f"{rew_name}_std"] = rew.std()

        self.last_unscaled_rewards: Dict[str, Tensor] = rew_dict
        self.last_scaled_rewards = self.last_scaled_rewards
        self.last_other_rewards = other_log_terms


    def exp_tracking_reward(
            self,
            gt: Tensor,
            rt: Tensor,
            rv: Tensor,
            rav: Tensor,
            gv: Tensor,
            gav: Tensor,
            kb: Tensor,
            gr: Tensor,
            lr: Tensor,
            dv: Tensor,
            ref_gt: Tensor,
            ref_rt: Tensor,
            ref_rv: Tensor,
            ref_rav: Tensor,
            ref_gv: Tensor,
            ref_gav: Tensor,
            ref_kb: Tensor,
            ref_gr: Tensor,
            ref_lr: Tensor,
            ref_dv: Tensor,
    ) -> Dict[str, Tensor]:
        def mul_exp_sum(x: Tensor, coef: float):
            if self.config.sum_before_exp:
                return x.sum(-1).sqrt().mul(coef).exp()
            else:
                return x.sqrt().mul(coef).exp().sum(-1)

        # # First sum across xyz
        # 
        gt_rew = mul_exp_sum((gt - ref_gt).pow(2).sum(-1), self.config.gt_rew_c)

        rh = gt[:, 0, 2]
        ref_rh = ref_gt[:, 0, 2]

        rh_rew = (rh - ref_rh).pow(2).sqrt().mul(self.config.rh_rew_c).exp()

        rt_rew = (rt - ref_rt).pow(2).sum(-1).sqrt().mul(self.config.rt_rew_c).exp()
        rv_rew = (rv - ref_rv).pow(2).sum(-1).sqrt().mul(self.config.rv_rew_c).exp()
        rav_rew = (rav - ref_rav).pow(2).sum(-1).sqrt().mul(self.config.rav_rew_c).exp()
        gv_rew = mul_exp_sum((gv - ref_gv).pow(2).sum(-1), self.config.gv_rew_c)
        gav_rew = mul_exp_sum((gav - ref_gav).pow(2).sum(-1), self.config.gav_rew_c)

        # First sum across xyz
        kb_rew = mul_exp_sum((kb - ref_kb).pow(2).sum(-1), self.config.kb_rew_c)

        if self.config.tan_norm_reward:
            gr = torch_utils.quat_to_tan_norm(gr.view(-1, 4), self.w_last).view(*gr.shape[:-1], 6)
            ref_gr = torch_utils.quat_to_tan_norm(ref_gr.view(-1, 4), self.w_last).view(*ref_gr.shape[:-1], 6)

            gr_rew = mul_exp_sum((gr - ref_gr).pow(2).sum(-1), self.config.gr_rew_c)

            lr = torch_utils.quat_to_tan_norm(lr.view(-1, 4), self.w_last).view(*lr.shape[:-1], 6)
            ref_lr = torch_utils.quat_to_tan_norm(ref_lr.view(-1, 4), self.w_last).view(*ref_lr.shape[:-1], 6)

            lr_rew = mul_exp_sum((lr - ref_lr).pow(2).sum(-1), self.config.lr_rew_c)

        else:
            gr_rew = mul_exp_sum(
                quat_diff_norm(gr, ref_gr, self.w_last).pow(2), self.config.gr_rew_c
            )

            lr_rew = mul_exp_sum(
                quat_diff_norm(lr, ref_lr, self.w_last).pow(2), self.config.lr_rew_c
            )

        dv_rew = mul_exp_sum((dv - ref_dv).pow(2), self.config.dv_rew_c)

        rew_dict = {
            "gt_rew": gt_rew,
            "rt_rew": rt_rew,
            "rh_rew": rh_rew,
            "rv_rew": rv_rew,
            "rav_rew": rav_rew,
            "gv_rew": gv_rew,
            "gav_rew": gav_rew,
            "kb_rew": kb_rew,
            "gr_rew": gr_rew,
            "lr_rew": lr_rew,
            "dv_rew": dv_rew
        }

        return rew_dict

    def get_phase_obs(self, motion_ids: Tensor, motion_times: Tensor):
        phase = (motion_times - self.motion_lib.state.motion_timings[motion_ids, 0]) / self.motion_lib.get_sub_motion_length(motion_ids)
        sin_phase = phase.sin().unsqueeze(-1)
        cos_phase = phase.cos().unsqueeze(-1)

        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
        return phase_obs

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        

        if self.config.use_phase_obs:
            self.mimic_phase[env_ids] = self.get_phase_obs(
                self.motion_ids[env_ids], self.motion_times[env_ids]
            )
        # 
        if self.config.provide_future_states:
            # TODO: take env_ids as input here
            if self.config.num_future_steps == 1:
                self.mimic_target_poses[:] = 0
            else:
                self.mimic_target_poses[:] = self.build_target_poses(self.config.num_future_steps)

    def on_epoch_end(self, current_epoch: int):
        super().on_epoch_end(current_epoch)
        if (
                self.config.dynamic_sample
                and current_epoch > 0
                and current_epoch % self.config.update_dynamic_weight_epochs == 0
        ):
            self.refresh_dynamic_weights()

    def build_target_poses(self, num_future_steps, target_pose_type=None):
        if target_pose_type is None:
            target_pose_type = self.config.target_pose_type
        if target_pose_type == "max-coords":
            return self.build_max_coords_target_poses(num_future_steps)
        elif target_pose_type == "max-coords-future-rel":
            return self.build_max_coords_target_poses_future_rel(num_future_steps)
        elif target_pose_type == "max-coords-future-rel-with-time":
            return self.build_max_coords_target_poses_future_rel_with_time(num_future_steps)
        else:
            raise ValueError(
                f"Unknown target pose type '{target_pose_type}'"
            )

    def build_max_coords_target_poses(self, num_future_steps):
        """
            This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        raw_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        (flat_target_pos, flat_target_rot, flat_target_dof_pos, _, _, _) = self.motion_lib.get_mimic_motion_state(
            flat_ids, flat_times
        )

        flat_target_dof_pos, _ = self.convert_dof(flat_target_dof_pos, None)

        all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        flat_target_pos = self.transfer_to_env_coordinates(flat_target_pos, all_env_ids)

        cur_gt, cur_gr, _, _ = self.get_bodies_state()
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        env_global_positions = self.convert_to_global_coords(cur_gt[:, 0, :2], self.env_offsets[..., :2])
        cur_gt[:, :, -1:] -= self.get_ground_heights_below_base(env_global_positions).view(self.num_envs, 1, 1)
        cur_gt[..., :2] -= self.respawn_offsets.clone()[..., :2].view(self.num_envs, 1, 2)

        expanded_body_pos = cur_gt.unsqueeze(1).expand(self.num_envs, num_future_steps, *cur_gt.shape[1:])
        expanded_body_rot = cur_gr.unsqueeze(1).expand(self.num_envs, num_future_steps, *cur_gr.shape[1:])

        flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
        flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

        if self.config.pos_rel_to_data:
            # When precise tracking isn't crucial.

            (target_cur_gt, target_cur_gr, _, _, _, _) = self.motion_lib.get_mimic_motion_state(
                self.motion_ids, self.motion_times
            )
            target_cur_gt = self.transfer_to_env_coordinates(target_cur_gt, all_env_ids)

            target_expanded_body_pos = target_cur_gt.unsqueeze(1).expand(self.num_envs, num_future_steps,
                                                                         *target_cur_gt.shape[1:])
            target_expanded_body_rot = target_cur_gr.unsqueeze(1).expand(self.num_envs, num_future_steps,
                                                                         *target_cur_gr.shape[1:])

            flat_target_cur_pos = target_expanded_body_pos.reshape(flat_target_pos.shape)
            flat_target_cur_rot = target_expanded_body_rot.reshape(flat_target_rot.shape)

            # Keep root height from current state
            root_pos = flat_cur_pos[:, 0, :].clone()
            root_pos[:, :2] = flat_target_cur_pos[:, 0, :2]

            # Convert rotation to hold yaw from target and roll and pitch from sim
            # first rotation sim_rot by the inverse of the sim_heading
            # then rotation that new rotation with target_heading
            sim_root_rot = flat_cur_rot[:, 0, :]
            target_root_rot = flat_target_cur_rot[:, 0, :]

            inv_sim_heading = torch_utils.calc_heading_quat_inv(sim_root_rot, self.w_last)
            self_rotated_sim_rot = rotations.quat_mul(inv_sim_heading, sim_root_rot, self.w_last)
            target_heading = torch_utils.calc_heading_quat(target_root_rot, self.w_last)

            root_rot = rotations.quat_mul(target_heading, self_rotated_sim_rot, self.w_last)
        else:
            root_pos = flat_cur_pos[:, 0, :]
            root_rot = flat_cur_rot[:, 0, :]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                      heading_rot_expand.shape[2])

        root_pos_expand = root_pos.unsqueeze(-2)

        """target"""
        # target body pos   [N, 3xB]
        target_rel_body_pos = flat_target_pos - flat_cur_pos
        flat_target_rel_body_pos = target_rel_body_pos.reshape(
            target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
            target_rel_body_pos.shape[2])
        flat_target_rel_body_pos = torch_utils.quat_rotate(flat_heading_rot, flat_target_rel_body_pos, self.w_last)
        target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0],
                                                               target_rel_body_pos.shape[1] * target_rel_body_pos.shape[
                                                                   2])

        # target body pos   [N, 3xB]
        flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1],
            flat_target_pos.shape[2])
        flat_target_body_pos = torch_utils.quat_rotate(flat_heading_rot, flat_target_body_pos, self.w_last)
        target_body_pos = flat_target_body_pos.reshape(flat_target_pos.shape[0],
                                                       flat_target_pos.shape[1] * flat_target_pos.shape[2])

        # target body rot   [N, 6xB]
        target_rel_body_rot = rotations.quat_mul(
            rotations.quat_conjugate(flat_cur_rot, self.w_last), flat_target_rot, self.w_last
        )
        target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), self.w_last).view(
            target_rel_body_rot.shape[0], -1)

        # target body rot   [N, 6xB]
        target_body_rot = rotations.quat_mul(heading_rot_expand, flat_target_rot, self.w_last)
        target_body_rot_obs = torch_utils.quat_to_tan_norm(target_body_rot.view(-1, 4), self.w_last).view(
            target_rel_body_rot.shape[0], -1)

        obs = torch.cat(
            (target_rel_body_pos, target_body_pos, target_rel_body_rot_obs, target_body_rot_obs), dim=-1
        ).view(self.num_envs, -1)

        return obs

    def build_max_coords_target_poses_future_rel(self, num_future_steps):
        """
            This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        raw_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        (flat_target_pos, flat_target_rot, flat_target_dof_pos, _, _, _) = self.motion_lib.get_mimic_motion_state(
            flat_ids, flat_times
        )

        flat_target_dof_pos, _ = self.convert_dof(flat_target_dof_pos, None)

        all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        flat_target_pos = self.transfer_to_env_coordinates(flat_target_pos, all_env_ids)

        cur_gt, cur_gr, _, _ = self.get_bodies_state()
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        env_global_positions = self.convert_to_global_coords(cur_gt[:, 0, :2], self.env_offsets[..., :2])
        cur_gt[:, :, -1:] -= self.get_ground_heights_below_base(env_global_positions).view(self.num_envs, 1, 1)
        cur_gt[..., :2] -= self.respawn_offsets.clone()[..., :2].view(self.num_envs, 1, 2)

        reference_pos = flat_target_pos.reshape(self.num_envs, num_future_steps, *cur_gt.shape[1:]).clone().roll(shifts=1, dims=1)
        reference_pos[:, 0] = cur_gt
        flat_reference_pos = reference_pos.reshape(flat_target_pos.shape)

        reference_rot = flat_target_rot.reshape(self.num_envs, num_future_steps, *flat_target_rot.shape[1:]).clone().roll(shifts=1, dims=1)
        reference_rot[:, 0] = cur_gr
        flat_reference_rot = reference_rot.reshape(flat_target_rot.shape)

        if self.config.pos_rel_to_data:
            # When precise tracking isn't crucial.

            (target_cur_gt, target_cur_gr, _, _, _, _) = self.motion_lib.get_mimic_motion_state(
                self.motion_ids, self.motion_times
            )
            target_cur_gt = self.transfer_to_env_coordinates(target_cur_gt, all_env_ids)

            # Keep root height from current state
            reference_root_pos = reference_pos[:, :, 0, :].clone()
            reference_root_pos[:, 0, :2] = target_cur_gt[:, 0, :2]
            reference_root_pos = reference_root_pos.reshape(self.num_envs * num_future_steps, -1)

            # Convert rotation to hold yaw from target and roll and pitch from sim
            # first rotation sim_rot by the inverse of the sim_heading
            # then rotation that new rotation with target_heading
            cur_sim_root_rot = cur_gr[:, 0, :]
            target_root_rot = target_cur_gr[:, 0, :]

            inv_sim_heading = torch_utils.calc_heading_quat_inv(cur_sim_root_rot, self.w_last)
            self_rotated_sim_rot = rotations.quat_mul(inv_sim_heading, cur_sim_root_rot, self.w_last)
            target_heading = torch_utils.calc_heading_quat(target_root_rot, self.w_last)

            cur_reference_root_rot = rotations.quat_mul(target_heading, self_rotated_sim_rot, self.w_last)
            reference_root_rot = reference_rot[:, :, 0, :].clone()
            reference_root_rot[:, 0] = cur_reference_root_rot
            reference_root_rot = reference_root_rot.reshape(self.num_envs * num_future_steps, -1)
        else:
            reference_root_pos = flat_reference_pos[:, 0, :]
            reference_root_rot = flat_reference_rot[:, 0, :]

        heading_rot = torch_utils.calc_heading_quat_inv(reference_root_rot, self.w_last)

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, flat_reference_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                                      heading_rot_expand.shape[2])

        reference_root_pos_expand = reference_root_pos.unsqueeze(-2)

        """target"""
        # target body pos   [N, 3xB]
        target_rel_body_pos = flat_target_pos - flat_reference_pos
        flat_target_rel_body_pos = target_rel_body_pos.reshape(target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
                                                               target_rel_body_pos.shape[2])
        flat_target_rel_body_pos = torch_utils.quat_rotate(flat_heading_rot, flat_target_rel_body_pos, self.w_last)

        # target body pos   [N, 3xB]
        flat_target_body_pos = (
                flat_target_pos - reference_root_pos_expand
        ).reshape(flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2])
        flat_target_body_pos = torch_utils.quat_rotate(flat_heading_rot, flat_target_body_pos, self.w_last)

        # target body rot   [N, 6xB]
        target_rel_body_rot = rotations.quat_mul(
            rotations.quat_conjugate(flat_reference_rot, self.w_last), flat_target_rot, self.w_last
        )
        target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), self.w_last).view(
            target_rel_body_rot.shape[0], -1)

        # target body rot   [N, 6xB]
        target_body_rot = rotations.quat_mul(heading_rot_expand, flat_target_rot, self.w_last)
        target_body_rot_obs = torch_utils.quat_to_tan_norm(target_body_rot.view(-1, 4), self.w_last).view(
            target_rel_body_rot.shape[0], -1)

        target_rel_body_pos = flat_target_rel_body_pos.reshape(target_rel_body_pos.shape[0],
                                                               target_rel_body_pos.shape[1] *
                                                               target_rel_body_pos.shape[2])
        target_body_pos = flat_target_body_pos.reshape(flat_target_pos.shape[0],
                                                       flat_target_pos.shape[1] * flat_target_pos.shape[2])

        obs = torch.cat(
            (
                target_rel_body_pos,
                target_body_pos,
                target_rel_body_rot_obs,
                target_body_rot_obs
            ), dim=-1
        ).view(self.num_envs, -1)

        return obs

    def build_max_coords_target_poses_future_rel_with_time(self, num_future_steps):
        target_pose_obs = self.build_max_coords_target_poses_future_rel(num_future_steps).view(self.num_envs, num_future_steps, -1)

        time_offsets = (
                torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
                * self.dt
        )

        raw_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        times = torch.minimum(raw_future_times.view(-1), lengths).view(self.num_envs, num_future_steps, 1) - self.motion_times.view(self.num_envs, 1, 1)

        obs = torch.cat([target_pose_obs, times], dim=-1).view(self.num_envs, -1)

        return obs

    def pre_physics_step(self, actions):
        if self.config.residual_control:
            actions = self.residual_actions_to_actual(actions)

        self.prev_contact_forces = self.get_bodies_contact_buf()

        return super().pre_physics_step(actions)

    def residual_actions_to_actual(
        self,
        residual_actions: Tensor,
        target_ids: Optional[Tensor] = None,
        target_times: Optional[Tensor] = None,
    ):
        if target_ids is None:
            target_ids = self.motion_ids

        if target_times is None:
            target_times = self.motion_times + self.dt

        target_states = self.motion_lib.get_motion_state(target_ids, target_times)
        target_dof_pos = target_states[2]

        target_dof_pos, _ = self.convert_dof(target_dof_pos, None)

        target_local_rot = dof_to_local(target_dof_pos, self.get_dof_offsets(), self.w_last)
        residual_actions_as_quats = dof_to_local(residual_actions, self.get_dof_offsets(), self.w_last)

        actions_as_quats = rotations.quat_mul(residual_actions_as_quats, target_local_rot, self.w_last)
        actions = torch_utils.quat_to_exp_map(actions_as_quats, self.w_last).view(self.num_envs, -1)

        return actions


def quat_diff_norm(quat1, quat2, w_last):
    if w_last:
        w = 3
    else:
        w = 0
    quat1inv = rotations.quat_conjugate(quat1, w_last)
    mul = rotations.quat_mul(quat1inv, quat2, w_last)
    norm = mul[..., w].clip(-1, 1).arccos() * 2
    # Trying both rotation directions
    norm = torch.min(norm, math.pi * 2 - norm)
    return norm


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_local(pose: Tensor, dof_offsets: List[int], w_last: bool) -> Tensor:
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    assert pose.shape[-1] == dof_offsets[-1]
    local_rot_shape = pose.shape[:-1] + (num_joints, 4)
    local_rot = torch.zeros(local_rot_shape, device=pose.device)

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # jp hack
        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose, w_last)
        elif dof_size == 1:
            axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            joint_pose_q = rotations.quat_from_angle_axis(joint_pose[..., 0], axis, w_last)
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        local_rot[:, j] = joint_pose_q

    return local_rot
