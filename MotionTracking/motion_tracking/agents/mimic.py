import random
from pathlib import Path

import tqdm

from motion_tracking.agents.ppo import PPO
from motion_tracking.envs.common.common_mimic import MimicHumanoid
import torch
import numpy as np
from copy import deepcopy


class Mimic(PPO):
    env: MimicHumanoid

    def initialize_env(self):
        super().initialize_env()
        self.motion_lib = self.env.motion_lib

    # def create_actor_state(self):
    #     state = super().create_actor_state()
    #     state["motion_ids"] = self.env.motion_ids.clone()
    #     return state
    #
    # def create_eval_actor_state(self):
    #     state = super().create_eval_actor_state()
    #     state["motion_ids"] = self.env.motion_ids.clone()
    #     return state
    #
    # def handle_eval_reset(self, actor_state):
    #     actor_state = super().handle_eval_reset(actor_state)
    #     return actor_state
    #
    # def eval_step_print(self, actor_state):
    #     print(
    #         self.env.motion_ids[0].item(),
    #         actor_state["rewards"][0].item(),
    #         self.env.last_other_rewards["cartesian_err"][0].item(),
    #     )
    #     for key, value in self.env.last_scaled_rewards.items():
    #         print(f"Reward {key}: {value}")

    @torch.no_grad()
    def calc_eval_metrics(self):
        was_training = self.training
        self.eval()

        num_motions = self.motion_lib.num_sub_motions()
        if self.env.config.fixed_motion_id is not None:
            num_motions = 1

        metrics = {
            "reward_too_bad": torch.zeros(num_motions, device=self.device),
            "max_average_deviation": torch.zeros(num_motions, device=self.device),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

        remaining_motions = num_motions
        motions_offset = 0

        while remaining_motions > 0:
            num_motions_this_iter = min(remaining_motions, self.num_envs)
            motion_ids = torch.arange(motions_offset, motions_offset + num_motions_this_iter, dtype=torch.long, device=self.device)
            if self.env.config.fixed_motion_id is not None:
                motion_ids = torch.zeros_like(motion_ids) + self.env.config.fixed_motion_id

            self.env.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(0, num_motions_this_iter, dtype=torch.long, device=self.device)

            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()

            max_len = motion_num_frames.max().item() if self.config.eval_length is None else self.config.eval_length

            for eval_episode in range(self.config.eval_num_episodes):
                steps = torch.zeros(num_motions_this_iter, dtype=torch.float, device=self.device)

                elapsed_time = torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0]) * dt
                self.env.motion_times[:num_motions_this_iter] = self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
                self.env.reset_track_steps.reset_steps(env_ids)
                self.env.disable_reset = True
                self.env.disable_reset_track = True
                self.env.object_ids[:num_motions_this_iter] = self.env.sample_object_ids(self.env.motion_ids[:num_motions_this_iter])
                obs = self.env.reset(torch.arange(0, num_motions_this_iter, dtype=torch.long, device=self.device))

                actor_state = self.create_eval_actor_state()
                actor_state["obs"] = obs
                actor_state = self.get_extra_obs_from_env(actor_state)

                for l in range(max_len):
                    actor_state = self.pre_eval_env_step(actor_state)

                    actor_state = self.eval_env_step(actor_state)

                    actor_state = self.post_eval_env_step(actor_state)
                    elapsed_time += dt
                    clip_done = (motion_lengths - dt) < elapsed_time
                    clip_not_done = torch.logical_not(clip_done)
                    metrics_dict = deepcopy(self.env.last_unscaled_rewards)
                    metrics_dict.update(self.env.last_other_rewards)
                    for k in self.config.eval_metric_keys:
                        metric = metrics_dict[k][:num_motions_this_iter]
                        metric *= (1 - clip_done.long())
                        metrics[k][motions_offset:motions_offset + num_motions_this_iter] += metric
                        metrics[f"{k}_max"][motions_offset:motions_offset + num_motions_this_iter] = torch.maximum(metrics[f"{k}_max"][motions_offset:motions_offset + num_motions_this_iter], metric)
                        metrics[f"{k}_min"][motions_offset:motions_offset + num_motions_this_iter] = torch.minimum(metrics[f"{k}_min"][motions_offset:motions_offset + num_motions_this_iter], metric)

                    metrics["max_average_deviation"][motions_offset:motions_offset + num_motions_this_iter][clip_not_done] = torch.maximum(metrics["max_average_deviation"][motions_offset:motions_offset + num_motions_this_iter][clip_not_done], metrics_dict["gt_err"][:num_motions_this_iter][clip_not_done])

                    if self.env.config.early_reward_term is not None:
                        reward_too_bad = torch.zeros(num_motions_this_iter, device=self.device, dtype=bool)
                        for entry in self.env.config.early_reward_term:
                            if entry.get("from_other", False):
                                from_dict = self.env.last_other_rewards
                            elif entry.use_scaled:
                                from_dict = self.env.last_scaled_rewards
                            else:
                                from_dict = self.env.last_unscaled_rewards

                            if entry.less_than:
                                entry_too_bad = (
                                        from_dict[entry.early_reward_term_key][:num_motions_this_iter]
                                        < entry.early_reward_term_thresh
                                )
                            else:
                                entry_too_bad = (
                                        from_dict[entry.early_reward_term_key][:num_motions_this_iter]
                                        > entry.early_reward_term_thresh
                                )

                            reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)
                            reward_too_bad = torch.logical_and(reward_too_bad, torch.logical_not(clip_done))

                        # Don't early track terminate if we very recently switched
                        # the tracking clip.
                        reward_too_bad = torch.logical_and(
                            reward_too_bad,
                            steps >= self.env.config.reset_track_grace_period,
                        )

                        steps += 1

                        metrics["reward_too_bad"][motions_offset:motions_offset + num_motions_this_iter][reward_too_bad] += 1

            remaining_motions -= self.num_envs
            motions_offset += self.num_envs

        if self.env.config.fixed_motion_id is None:
            motion_lengths = self.motion_lib.state.motion_timings[:, 1] - self.motion_lib.state.motion_timings[:, 0]
            motion_num_frames = (motion_lengths / dt).floor().long()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean()

        mean_reset_errors = (metrics["reward_too_bad"] > 0).float()
        to_log["eval/reward_too_bad"] = mean_reset_errors.detach().mean()
        to_log["eval/mean_reward_too_bad"] = (metrics["reward_too_bad"] / (motion_num_frames * self.config.eval_num_episodes)).detach().mean()

        tracking_failures = (metrics["max_average_deviation"] > 0.5).float()
        to_log["eval/tracking_success_rate"] = 1. - tracking_failures.detach().mean()

        # get indices of failed motions and save list to file
        failed_motions = torch.nonzero(tracking_failures).flatten().tolist()

        save_dir = Path(self.trainer.default_root_dir)
        print(f"Saving to: {save_dir / 'failed_motions.txt'}")
        with open(save_dir / "failed_motions.txt", "w") as f:
            for motion_id in failed_motions:
                f.write(f"{motion_id}\n")

        self.log_dict(
            to_log,
            on_epoch=True,
            on_step=False,
        )

        stop_early = (
                self.config.early_terminate_cart_err is not None
                or self.config.early_terminate_reward_too_bad_prob is not None
        )
        if self.config.early_terminate_cart_err is not None:
            cart_err = to_log["eval/cartesian_err"]
            stop_early = stop_early and (cart_err <= self.config.early_terminate_cart_err)
        if self.config.early_terminate_reward_too_bad_prob is not None:
            early_term_prob = to_log["eval/mean_reward_too_bad"]

            stop_early = stop_early and (early_term_prob <= self.config.early_terminate_reward_too_bad_prob)

        if stop_early:
            print(f"Stopping early! Target error reached, cart_err: {to_log['eval/cartesian_err'].item()}, early_term_prob: {to_log['eval/reward_too_bad'].item()}")
            self.save()
            self.terminate_early()

        self.env.disable_reset = False
        self.env.disable_reset_track = False
        self.env.force_respawn_on_flat = False

        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.reset_track(all_ids)

        self.force_full_restart = True
        self.train(was_training)

        schedule = self.config.sigma_schedule
        if schedule is not None:
            if self.config.sigma_schedule.get("mimic_success", False):
                total_failing_motions = torch.sum((metrics["reward_too_bad"] > 0).float())
                percent_failure = total_failing_motions / num_motions

                new_logstd = self.config.init_logstd - percent_failure * (self.config.init_logstd - self.config.sigma_schedule.end_logstd)
            else:
                new_logstd = schedule.init_logstd + min(
                    max(0, self.current_epoch - schedule.get("start_epoch", 0)) / schedule.end_epoch, 1
                ) * (schedule.end_logstd - schedule.init_logstd)

            self.actor.set_logstd(new_logstd)

    @torch.no_grad()
    def record_data(self):
        self.save_dir = Path(self.config.checkpoint).parent.resolve()

        self.eval()

        num_motions = self.motion_lib.num_sub_motions()
        if self.env.config.fixed_motion_id is not None:
            num_motions = 1
        if num_motions <= self.num_envs:
            motion_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)
        else:
            motion_ids = torch.randperm(num_motions)[:self.num_envs]
            num_motions = self.num_envs
        if self.env.config.fixed_motion_id is not None:
            motion_ids = torch.zeros_like(motion_ids) + self.env.config.fixed_motion_id

        self.env.motion_ids[:num_motions] = motion_ids

        env_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)

        dt: float = self.env.dt
        motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
        motion_num_frames = (motion_lengths / dt).floor().long()

        metrics = {"reward_too_bad": torch.zeros(num_motions, device=self.device)}
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)
        max_len = motion_num_frames.max().item()

        trajectories = [[{}] for _ in range(num_motions * self.config.eval_num_episodes)]

        for eval_episode in range(self.config.eval_num_episodes):
            consecutive_failures = torch.zeros(
                num_motions, dtype=torch.int, device=self.device
            )

            steps = torch.zeros(num_motions, dtype=torch.float, device=self.device)

            elapsed_time = torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0]) * dt
            self.env.motion_times[:num_motions] = self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
            self.env.reset_track_steps.reset_steps(env_ids)
            self.env.disable_reset = True
            self.env.disable_reset_track = True
            obs = self.env.reset(torch.arange(0, num_motions, dtype=torch.long, device=self.device))

            actor_state = self.create_eval_actor_state()
            actor_state["obs"] = obs
            actor_state = self.get_extra_obs_from_env(actor_state)

            if trajectories is not None:
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
                ) = self.motion_lib.get_motion_state(self.env.motion_ids[:num_motions], self.env.motion_times[:num_motions])

                root_pos = self.env.transfer_to_env_coordinates(root_pos, env_ids)
                dof_pos, dof_vel = self.env.convert_dof(dof_pos, dof_vel)

                root_pos[:, 2] += self.env.config.ref_respawn_offset
                rb_pos[:, :, :3] -= rb_pos[:, 0, :3].unsqueeze(1).clone()
                rb_pos[:, :, :3] += root_pos.unsqueeze(1)

                for i in range(num_motions):
                    initial_state = {
                        "root_pos": root_pos[i],
                        "root_rot": root_rot[i],
                        "dof_pos": dof_pos[i],
                        "root_vel": root_vel[i],
                        "root_ang_vel": root_ang_vel[i],
                        "dof_vel": dof_vel[i],
                        "rb_pos": rb_pos[i],
                        "rb_rot": rb_rot[i],
                        "rb_vel": rb_vel[i],
                        "rb_ang_vel": rb_ang_vel[i],
                    }

                    trajectories[i + eval_episode * num_motions].append(initial_state)

            for l in range(max_len):
                if trajectories is not None:
                    root_states = self.env.get_humanoid_root_states()
                    bodies_contact_buf = self.env.get_bodies_contact_buf()
                    body_pose, body_rotations, body_vel, body_ang_vel = self.env.get_bodies_state()

                    env_global_positions = self.env.convert_to_global_coords(body_pose[:, 0, :2], self.env.env_offsets[..., :2])
                    body_pose[:, :, -1:] -= self.env.get_ground_heights_below_base(env_global_positions).view(self.num_envs, 1, 1)
                    body_pose[..., :2] -= self.env.respawn_offsets.clone()[..., :2].view(self.num_envs, 1, 2)

                    dof_pose, dof_vel = self.env.get_dof_state()
                    for i in range(num_motions):
                        if elapsed_time[i].item() < motion_lengths[i].item():
                            current_state = {
                                "body_pose": body_pose[i].detach().cpu(),
                                "body_rotations": body_rotations[i].detach().cpu(),
                                "body_vel": body_vel[i].detach().cpu(),
                                "body_ang_vel": body_ang_vel[i].detach().cpu(),
                                "dof_pose": dof_pose[i].detach().cpu(),
                                "dof_vel": dof_vel[i].detach().cpu(),
                                "root_state": root_states[i].detach().cpu(),
                                "bodies_contact_buf": bodies_contact_buf[i].detach().cpu()
                            }
                            trajectories[i + eval_episode * num_motions].append(current_state)

                actor_state = self.pre_eval_env_step(actor_state)
                if trajectories is not None:
                    for i in range(num_motions):
                        if elapsed_time[i].item() < motion_lengths[i].item():
                            trajectories[i + eval_episode * num_motions][-1]["action"] = actor_state["actions"][i].detach().cpu()

                actor_state = self.eval_env_step(actor_state)

                actor_state = self.post_eval_env_step(actor_state)
                elapsed_time += dt
                clip_done = motion_lengths - dt < elapsed_time
                metrics_dict = deepcopy(self.env.last_unscaled_rewards)
                metrics_dict.update(self.env.last_other_rewards)
                for k in self.config.eval_metric_keys:
                    metric = metrics_dict[k][:num_motions]
                    metric *= (1 - clip_done.long())
                    metrics[k] += metric
                    metrics[f"{k}_max"] = torch.maximum(metrics[f"{k}_max"], metric)
                    metrics[f"{k}_min"] = torch.minimum(metrics[f"{k}_min"], metric)

                    if trajectories is not None:
                        for i in range(num_motions):
                            if elapsed_time[i].item() < motion_lengths[i].item():
                                if k not in trajectories[i + eval_episode * num_motions][0]:
                                    trajectories[i + eval_episode * num_motions][0][k] = 0
                                trajectories[i + eval_episode * num_motions][0][k] += metric[i]

                if self.env.config.early_reward_term is not None:
                    reward_too_bad = torch.zeros(num_motions, device=self.device, dtype=bool)
                    for entry in self.env.config.early_reward_term:
                        if entry.get("from_other", False):
                            from_dict = self.env.last_other_rewards
                        elif entry.use_scaled:
                            from_dict = self.env.last_scaled_rewards
                        else:
                            from_dict = self.env.last_unscaled_rewards

                        if entry.less_than:
                            entry_too_bad = (
                                    from_dict[entry.early_reward_term_key][:num_motions]
                                    < entry.early_reward_end_term_thresh
                            )
                        else:
                            entry_too_bad = (
                                    from_dict[entry.early_reward_term_key][:num_motions]
                                    > entry.early_reward_end_term_thresh
                            )

                        reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)
                        reward_too_bad = torch.logical_and(reward_too_bad, torch.logical_not(clip_done))

                    # Don't early track terminate if we very recently switched
                    # the tracking clip.
                    reward_too_bad = torch.logical_and(
                        reward_too_bad,
                        steps >= self.env.config.reset_track_grace_period,
                    )

                    reward_not_too_bad = torch.logical_not(reward_too_bad)

                    consecutive_failures[reward_too_bad] += 1
                    consecutive_failures[reward_not_too_bad] = 0

                    too_many_failures = consecutive_failures > self.env.config.max_consecutive_failures

                    steps += 1

                    metrics["reward_too_bad"][too_many_failures] += 1

                    if trajectories is not None:
                        for i in range(num_motions):
                            if elapsed_time[i].item() < motion_lengths[i].item():
                                if "reward_too_bad" not in trajectories[i + eval_episode * num_motions][0]:
                                    trajectories[i + eval_episode * num_motions][0]["reward_too_bad"] = 0
                                trajectories[i + eval_episode * num_motions][0]["reward_too_bad"] += reward_too_bad[i].detach().cpu().float()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean()

        mean_reset_errors = (metrics["reward_too_bad"] > 0).float()
        to_log["eval/reward_too_bad"] = mean_reset_errors.detach().mean()
        to_log["eval/mean_reward_too_bad"] = (metrics["reward_too_bad"] / (motion_num_frames * self.config.eval_num_episodes)).detach().mean()

        if trajectories is not None and mean_reset_errors.item() == self.config.early_terminate_reward_too_bad_prob:
            save_dir = Path(self.save_dir)
            print(f"Saving to: {save_dir / 'recordings.pt'}")
            torch.save(trajectories, save_dir / "recordings.pt")

    @torch.no_grad()
    def test_data(self):
        self.save_dir = Path(self.config.checkpoint).parent.resolve()

        self.eval()

        num_motions = self.motion_lib.num_sub_motions()
        if self.env.config.fixed_motion_id is not None:
            num_motions = 1
        if num_motions <= self.num_envs:
            motion_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)
        else:
            motion_ids = torch.randperm(num_motions)[:self.num_envs]
            num_motions = self.num_envs
        if self.env.config.fixed_motion_id is not None:
            motion_ids = torch.zeros_like(motion_ids) + self.env.config.fixed_motion_id

        self.env.motion_ids[:num_motions] = motion_ids

        env_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)

        save_dir = Path(self.save_dir)
        print(f"Loading from: {save_dir / 'recordings.pt'}")
        recordings = torch.load(save_dir / "recordings.pt")

        dt: float = self.env.dt
        motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
        motion_num_frames = (motion_lengths / dt).floor().long()

        for eval_episode in range(self.config.eval_num_episodes):
            elapsed_time = torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0]) * dt
            self.env.motion_times[:num_motions] = self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
            self.env.reset_track_steps.reset_steps(env_ids)
            self.env.disable_reset = True
            self.env.disable_reset_track = True
            # obs = self.env.reset(torch.arange(0, num_motions, dtype=torch.long, device=self.device))

            demo_init_state = recordings[eval_episode][1]
            init_state = {}
            for k in demo_init_state.keys():
                init_state[k] = torch.stack([recordings[env_id + eval_episode][1][k] for env_id in env_ids], dim=0).to(self.device)
            recorded_state = {}

            time_offset = 0
            for k in recordings[eval_episode][2 + time_offset].keys():
                recorded_state[k] = torch.stack([recordings[env_id + eval_episode][2 + time_offset][k] for env_id in env_ids], dim=0).to(self.device)

            init_state['root_pos'] = recorded_state["root_state"][..., :3].to(self.device)
            init_state['root_rot'] = recorded_state["root_state"][..., 3:7].to(self.device)
            init_state['dof_pos'] = recorded_state["dof_pose"].to(self.device)
            init_state['root_vel'] = recorded_state["body_vel"][..., 0, :].to(self.device)
            init_state['root_ang_vel'] = recorded_state["body_ang_vel"][..., 0, :].to(self.device)
            init_state['dof_vel'] = recorded_state["dof_vel"].to(self.device)
            init_state['rb_pos'] = recorded_state["body_pose"].to(self.device)
            init_state['rb_rot'] = recorded_state["body_rotations"].to(self.device)
            init_state['rb_ang_vel'] = recorded_state["body_ang_vel"].to(self.device)
            init_state['rb_vel'] = recorded_state["body_vel"].to(self.device)

            init_state["env_ids"] = env_ids

            self.env.set_env_state(**init_state)
            self.env.reset_happened = True
            self.env.reset_ref_env_ids = env_ids
            self.env.reset_env_tensors(env_ids)
            self.env.refresh_sim_tensors()
            self.env.compute_observations(env_ids)

            obs = self.env.obs_buf

            actor_state = self.create_eval_actor_state()
            actor_state["obs"] = obs
            actor_state = self.get_extra_obs_from_env(actor_state)

            for l in range(time_offset, len(recordings[eval_episode]) - 2):
                root_states = self.env.get_humanoid_root_states()
                bodies_contact_buf = self.env.get_bodies_contact_buf()
                body_pose, body_rotations, body_vel, body_ang_vel = self.env.get_bodies_state()
                dof_pose, dof_vel = self.env.get_dof_state()
                current_state = {
                    "body_pose": body_pose.detach(),
                    "body_rotations": body_rotations.detach(),
                    "body_vel": body_vel.detach(),
                    "body_ang_vel": body_ang_vel.detach(),
                    "dof_pose": dof_pose.detach(),
                    "dof_vel": dof_vel.detach(),
                    "root_state": root_states.detach(),
                    # "bodies_contact_buf": bodies_contact_buf[0].detach().cpu()
                }

                recorded_state = {}
                for k in recordings[eval_episode][l + 2].keys():
                    recorded_state[k] = torch.stack([recordings[env_id + eval_episode][l + 2][k] for env_id in env_ids], dim=0).to(self.device)

                for k, v in current_state.items():
                    for env_id in range(self.num_envs):
                        if (v[env_id] - recorded_state[k][env_id]).abs().mean() > 1e-5:
                            print(f"Error in {k} for env {env_id} at step {l} of size {(v[env_id] - recorded_state[k][env_id]).abs().mean()}")

                actor_state = self.pre_eval_env_step(actor_state)

                # recorded_action = recordings[eval_episode][l + 2]["action"]
                recorded_action = torch.stack([recordings[env_id + eval_episode][l + 2]["action"] for env_id in env_ids], dim=0).to(self.device)
                current_action = actor_state["actions"][0].detach().cpu()

                # actor_state["actions"][0] = recorded_action.to(self.device)
                actor_state["actions"] = recorded_action

                # if (recorded_action - current_action).abs().mean() > 1e-5:
                #     print(f"Error in action at step {l} of size {(recorded_action - current_action).abs().mean()}")

                actor_state = self.eval_env_step(actor_state)

                actor_state = self.post_eval_env_step(actor_state)
                elapsed_time += dt

    def evaluate_mimic(self):
        self.eval()
        self.create_eval_callbacks()
        self.pre_evaluate_policy(reset_env=False)

        num_motions = self.motion_lib.num_sub_motions() - self.env.config.fixed_motion_offset
        if num_motions > self.num_envs:
            num_motions = self.num_envs

        motion_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device) + self.env.config.fixed_motion_offset
        if self.env.config.fixed_motion_id is not None:
            motion_ids = torch.zeros_like(motion_ids) + self.env.config.fixed_motion_id

        if hasattr(self, "vae_noise"):
            self.reset_vae_noise(None)

        total_successes = torch.zeros(num_motions, device=self.device)
        tracked_successes = torch.zeros(num_motions, self.config.eval_num_episodes, device=self.device)

        self.env.motion_ids[:num_motions] = motion_ids

        env_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)

        dt: float = self.env.dt
        motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
        motion_num_frames = (motion_lengths / dt).floor().long()

        metrics = {
            "reward_too_bad": torch.zeros(num_motions, device=self.device),
            "max_average_deviation": torch.zeros(num_motions, device=self.device),
            "max_max_deviation": torch.zeros(num_motions, device=self.device),
            "min_object_distance": torch.ones(self.config.eval_num_episodes, num_motions, device=self.device) * 1000,
            "success_object": torch.zeros(self.config.eval_num_episodes, num_motions, device=self.device),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)
        max_len = motion_num_frames.max().item() if self.config.eval_length is None else self.config.eval_length

        for eval_episode in range(self.config.eval_num_episodes):
            consecutive_failures = torch.zeros(
                num_motions, dtype=torch.int, device=self.device
            )

            random_offset = random.randint(0, 8000)

            steps = torch.zeros(num_motions, dtype=torch.float, device=self.device)

            elapsed_time = torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0]) * dt
            self.env.motion_times[:num_motions] = self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
            self.env.reset_track_steps.reset_steps(env_ids)
            self.env.disable_reset = True
            self.env.disable_reset_track = True
            self.env.object_ids[:num_motions] = self.env.sample_object_ids(self.env.motion_ids[:num_motions])
            obs = self.env.reset(torch.arange(0, num_motions, dtype=torch.long, device=self.device))

            actor_state = self.create_eval_actor_state()
            actor_state["obs"] = obs
            actor_state = self.get_extra_obs_from_env(actor_state)

            if "motion_ids" in self.extra_obs_inputs:
                if self.actor.mu_model.extra_input_models["motion_ids"].config.random_embedding.use_random_embeddings:
                    actor_state["motion_ids"] = actor_state["motion_ids"].clone() + random_offset

            if hasattr(self, "vae_noise"):
                self.reset_vae_noise(actor_state["done_indices"])

            for l in tqdm.trange(max_len):
                actor_state = self.pre_eval_env_step(actor_state)

                actor_state = self.eval_env_step(actor_state)
                if "motion_ids" in self.extra_obs_inputs:
                    if self.actor.mu_model.extra_input_models["motion_ids"].config.random_embedding.use_random_embeddings:
                        actor_state["motion_ids"] = actor_state["motion_ids"].clone() + random_offset

                actor_state = self.post_eval_env_step(actor_state)
                elapsed_time += dt
                clip_done = (motion_lengths - dt) < elapsed_time
                clip_not_done = torch.logical_not(clip_done)
                metrics_dict = deepcopy(self.env.last_unscaled_rewards)
                metrics_dict.update(self.env.last_other_rewards)
                for k in self.config.eval_metric_keys:
                    metric = metrics_dict[k][:num_motions]
                    metric *= (1 - clip_done.long())
                    metrics[k] += metric
                    metrics[f"{k}_max"][clip_not_done] = torch.maximum(metrics[f"{k}_max"], metric)[clip_not_done]
                    metrics[f"{k}_min"][clip_not_done] = torch.minimum(metrics[f"{k}_min"], metric)[clip_not_done]

                metrics["max_average_deviation"][clip_not_done] = torch.maximum(metrics["max_average_deviation"][clip_not_done], metrics_dict["gt_err"][:num_motions][clip_not_done])
                metrics["max_max_deviation"][clip_not_done] = torch.maximum(metrics["max_max_deviation"][clip_not_done], metrics_dict["max_joint_err"][:num_motions][clip_not_done])

                if "success_object_position" in self.config.eval_metric_keys:
                    non_zero_distance = metrics_dict["distance_to_object_position"][:num_motions] > 0
                    metrics["min_object_distance"][eval_episode][non_zero_distance] = torch.minimum(metrics["min_object_distance"][eval_episode], metrics_dict["distance_to_object_position"][:num_motions])[non_zero_distance]
                    metrics["success_object"][eval_episode][non_zero_distance] = torch.torch.maximum(metrics["success_object"][eval_episode], metrics_dict["success_object_position"][:num_motions])[non_zero_distance]

                if self.env.config.early_reward_term is not None:
                    reward_too_bad = torch.zeros(num_motions, device=self.device, dtype=bool)
                    for entry in self.env.config.early_reward_term:
                        if entry.get("from_other", False):
                            from_dict = self.env.last_other_rewards
                        elif entry.use_scaled:
                            from_dict = self.env.last_scaled_rewards
                        else:
                            from_dict = self.env.last_unscaled_rewards

                        if entry.less_than:
                            entry_too_bad = (
                                    from_dict[entry.early_reward_term_key][:num_motions]
                                    < entry.early_reward_end_term_thresh
                            )
                        else:
                            entry_too_bad = (
                                    from_dict[entry.early_reward_term_key][:num_motions]
                                    > entry.early_reward_end_term_thresh
                            )

                        reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)
                        reward_too_bad = torch.logical_and(reward_too_bad, torch.logical_not(clip_done))

                    # Don't early track terminate if we very recently switched
                    # the tracking clip.
                    reward_too_bad = torch.logical_and(
                        reward_too_bad,
                        steps >= self.env.config.reset_track_grace_period,
                    )

                    reward_not_too_bad = torch.logical_not(reward_too_bad)

                    consecutive_failures[reward_too_bad] += 1
                    consecutive_failures[reward_not_too_bad] = 0

                    too_many_failures = consecutive_failures > self.env.config.max_consecutive_failures

                    steps += 1

                    metrics["reward_too_bad"][too_many_failures] += 1

            mean_tracking_errors = metrics["max_average_deviation"] < 0.5
            total_successes[mean_tracking_errors] += 1
            tracked_successes[mean_tracking_errors, eval_episode] += 1

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean()

        if "reach_success" in metrics_dict:
            to_log["eval/reach_success"] = torch.tensor(metrics_dict["reach_success"])
            to_log["eval/reach_distance"] = torch.tensor(metrics_dict["reach_distance"])

        mean_tracking_errors = metrics["max_average_deviation"]
        to_log["eval/max_average_deviation"] = mean_tracking_errors.detach().mean()
        mean_tracking_errors = metrics["max_max_deviation"]
        to_log["eval/max_max_deviation"] = mean_tracking_errors.detach().mean()

        mean_tracking_errors = (metrics["max_average_deviation"] > 0.5).float()
        to_log["eval/count_bound_max_average_deviation"] = mean_tracking_errors.detach().mean()
        mean_tracking_errors = (metrics["max_max_deviation"] > 0.5).float()
        to_log["eval/count_bound_max_max_deviation"] = mean_tracking_errors.detach().mean()

        successful_envs = (metrics["max_average_deviation"] < 0.5)
        gt_err = metrics["gt_err"] / (motion_num_frames * self.config.eval_num_episodes)
        to_log["eval/gt_err_success"] = gt_err[successful_envs].detach().mean()
        gr_err = metrics["gr_err"] / (motion_num_frames * self.config.eval_num_episodes)
        to_log["eval/gr_err_success"] = gr_err[successful_envs].detach().mean()

        mean_reset_errors = (metrics["reward_too_bad"] > 0).float()
        to_log["eval/reward_too_bad"] = mean_reset_errors.detach().mean()
        to_log["eval/mean_reward_too_bad"] = (metrics["reward_too_bad"] / (motion_num_frames * self.config.eval_num_episodes)).detach().mean()

        to_log["eval/success_rate"] = total_successes.detach().mean() / self.config.eval_num_episodes
        any_success = (total_successes > 0).float()
        to_log["eval/success_rate_top_k"] = any_success.detach().mean()

        average_object_success = metrics["success_object"].mean()
        to_log["eval/success_object"] = average_object_success.detach().mean()

        average_min_object_distance = metrics["min_object_distance"].mean()
        to_log["eval/min_object_distance"] = average_min_object_distance.detach().mean()

        indices = [idx for idx in range(self.config.eval_num_episodes)]
        for i in range(self.config.eval_num_episodes):
            successes = []
            for _ in range(50):
                np.random.shuffle(indices)
                count_undereq_i = (tracked_successes[:, indices[:i + 1]].detach().sum(dim=1) > 0).float()
                successes.append(count_undereq_i.cpu().detach().mean())

            # store quantiles for successes
            to_log[f"eval/success_rate_top_{i}"] = np.mean(successes)
            to_log[f"eval/success_rate_top_{i}_25q"] = np.quantile(successes, 0.25)
            to_log[f"eval/success_rate_top_{i}_75q"] = np.quantile(successes, 0.75)

        print("--- EVAL MIMIC RESULTS ---")
        for key, value in to_log.items():
            print(f"{key}: {value.item()}")

        print(f"Object motion results: {total_successes.int().detach().cpu().numpy()}")

        self.post_evaluate_policy()
