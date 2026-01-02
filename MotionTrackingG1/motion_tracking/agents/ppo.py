import torch

from torch import nn, Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_

import random
import numpy as np
import time
import math
from pathlib import Path
from typing import Optional, List

from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from hydra.utils import instantiate

from isaac_utils import torch_utils

from motion_tracking.utils.time_report import TimeReport
from motion_tracking.utils.average_meter import AverageMeter, TensorAverageMeterDict
from motion_tracking.agents.utils.data_utils import DictDataset, ExperienceBuffer
from motion_tracking.agents.models.actor import PPO_Actor
from motion_tracking.agents.models.common import NormObsBase
from motion_tracking.envs.common.common_disc import DiscHumanoid
from motion_tracking.utils.running_mean_std import RunningMeanStd
from motion_tracking.utils.device_dtype_mixin import DeviceDtypeModuleMixin
from motion_tracking.agents.callbacks.base_callback import RL_EvalCallback
from motion_tracking.data.assets.skeleton_configs import get_obs_and_act_sizes
from motion_tracking.agents.models.cvae import CVAE

def get_params(obj) -> List[nn.Parameter]:
    """
    Gets list of params from either a list of params
    (where nothing happens) or a list of param groups
    """
    as_list = list(obj)
    if isinstance(as_list[0], Tensor):
        return as_list
    else:
        params = []
        for group in as_list:
            params = params + list(group["params"])
        return params


class PPO(DeviceDtypeModuleMixin, LightningModule):
    current_rewards: Tensor
    current_lengths: Tensor
    rand_action_probs: Tensor
    rand_action_mask: Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_llc = self.config.use_llc
        self.num_envs: int = config.num_envs
        self.num_obs, self.num_act = get_obs_and_act_sizes(self.config.env.config)
        if self.use_llc:
            self.num_act = 64
        
        self.num_steps: int = config.num_steps
        self.gamma: float = config.gamma
        self.tau: float = config.tau
        self.e_clip: float = config.e_clip
        self.task_reward_w: float = config.task_reward_w
        self.num_mini_epochs: int = config.num_mini_epochs

        self.actor: PPO_Actor = instantiate(
            config.actor, num_in=self.num_obs, num_act=self.num_act
        )
        critic_in = self.num_obs
        self.shared_actor_critic_base = self.config.get("shared_actor_critic_base", False)
        if self.shared_actor_critic_base:
            critic_in = self.actor.get_features_size()
        if self.config.dual_env:
            self.critic: NormObsBase = instantiate(config.critic, num_in=critic_in+1, num_out=1)
        else:
            self.critic: NormObsBase = instantiate(config.critic, num_in=critic_in, num_out=1)
        
        self.dual_critic=False
        if self.dual_critic:
            self.critic2: NormObsBase = instantiate(config.critic, num_in=critic_in, num_out=1)

        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps)
        
        if self.config.normalize_values:
            self.running_val_norm = RunningMeanStd(
                shape=(1,), device="cpu", clamp_value=self.config.val_clamp_value
            )
        else:
            self.running_val_norm = None

        # timer
        self.time_report = TimeReport()

        if config.schedules is None:
            self.schedules = None
        else:
            self.schedules = [instantiate(s, obj=self) for s in config.schedules]

        self.experience_buffer.register_key("obs", shape=(self.num_obs,))
        if self.config.dual_env:
            self.experience_buffer.register_key("obs_add_env", shape=(self.num_obs+1,))
        self.experience_buffer.register_key("mus", shape=(self.num_act,))
        self.experience_buffer.register_key("sigmas", shape=(self.num_act,))
        self.experience_buffer.register_key("actions", shape=(self.num_act,))
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")

        self.extra_obs_inputs = self.config.extra_inputs
        if self.extra_obs_inputs is not None:
            keys = list(self.extra_obs_inputs.keys())
            for key in keys:
                val = self.extra_obs_inputs[key]
                if not val.get("retrieve_from_env", True):
                    del self.extra_obs_inputs[key]
                    continue
                dtype = getattr(torch, val.get("dtype", "float"))
                self.experience_buffer.register_key(key, shape=(val.size,), dtype=dtype)

        self.use_rand_action_masks = self.config.use_rand_action_masks
        if self.use_rand_action_masks:
            self.experience_buffer.register_key("rand_action_mask", dtype=torch.long)
            all_env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            # self._rand_action_probs = 1.0 - env_ids / (num_envs - 1.0)
            rand_action_probs = 1.0 - torch.exp(
                10 * (all_env_ids / (self.num_envs - 1.0) - 1.0)
            )
            rand_action_probs[0] = 1.0
            rand_action_probs[-1] = 0.0

            self.register_buffer(
                "rand_action_probs", rand_action_probs, persistent=False
            )
            self.register_buffer(
                "rand_action_mask",
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                persistent=False,
            )
        

        # Obs deliberately not on here, since its updated before env step
        self.actor_state_to_experience_buffer_list = [
            "mus",
            "sigmas",
            "actions",
            "neglogp",
            "rewards",
            "dones",
        ]

        self.register_buffer(
            "current_lengths",
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            persistent=False,
        )

        self.register_buffer(
            "current_rewards",
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            persistent=False,
        )

        self.episode_reward_meter = AverageMeter(1, 100)
        self.episode_length_meter = AverageMeter(1, 100)
        self.episode_env_tensors = TensorAverageMeterDict()
        self.step_count = 0

        self.automatic_optimization = False

        self.critic_bad_grads_count = 0
        self.actor_bad_grads_count = 0
        self.actor_grad_norm_before_clip = 0.0
        self.actor_grad_norm_after_clip = 0.0
        self.critic_grad_norm_before_clip = 0.0
        self.critic_grad_norm_after_clip = 0.0

        if self.dual_critic:
            self.critic2_bad_grads_count = 0
            self.critic2_grad_norm_before_clip = 0.0
            self.critic2_grad_norm_after_clip = 0.0

        self.force_full_restart = False

        self.eval_callbacks: list[RL_EvalCallback] = []

        # 
        self.save_data_for_cvae = False
        if self.config.save_data_for_cvae:
            self.save_data_for_cvae=True
            self.data_for_cvae=[]
        
        if self.use_llc:
            input_dim = 878 - 93 # 358 + 69 #878 # Example for flattened MNIST images
            condition_dim = 358  # Example for one-hot encoded digits
            action_dim = 69
            latent_dim = 64

            self.llc_model = CVAE(input_dim, condition_dim, action_dim, latent_dim).to(self.device)
            # self.llc_model = torch.load("results/cvae_model_new2.pth").to(self.device)
            self.llc_model.load_state_dict(torch.load('results/cvae_model_new10_noscene_increase_kl_5000.pth'))
            self.llc_model.eval()


    def training_step(self, batch, batch_idx: int):
        if self.config.dual_env:
            self.extra_env_obs = torch.zeros((self.num_envs, 1), device=self.device)
            self.extra_env_obs[:self.env.num_amp_envs]-=1
            self.extra_env_obs[self.env.num_amp_envs:]+=1
        
        optimizers = self.optimizers()
        actor_optimizer = optimizers[0]
        critic_optimizer = optimizers[1]

        if self.dual_critic:
            critic2_optimizer = optimizers[2]

        if batch_idx == 0:
            self.eval()
            self.play_steps()
            self.generate_datasets()
            self.train()

        if batch_idx < self.ac_max_num_batches():
            actor_loss = self.actor_step(batch_idx)
            actor_optimizer.zero_grad()
            self.manual_backward(actor_loss)
            self.handle_actor_grad_clipping()
            actor_optimizer.step()
            self.actor.logstd_tick(self.current_epoch)

            critic_loss = self.critic_step(batch_idx)
            critic_optimizer.zero_grad()

            if self.dual_critic:
                critic2_loss = self.critic2_step(batch_idx)
                critic2_optimizer.zero_grad()
                self.manual_backward(critic2_loss)
                critic2_optimizer.step()

            self.manual_backward(critic_loss)
            self.handle_critic_grad_clipping()
            critic_optimizer.step()

            self.extra_optimization_steps(batch_idx)

        if self.trainer.is_last_batch:
            lrs = self.lr_schedulers()
            if lrs is not None:
                if isinstance(lrs, list):
                    for lr in lrs:
                        lr.step()
                else:
                    lrs.step()

    def extra_optimization_steps(self, batch_idx: int):
        pass

    @torch.no_grad()
    def play_steps(self):
        self.pre_play_steps()
        actor_state = self.create_actor_state()

        for i in range(self.num_steps):
            actor_state["step"] = i

            actor_state = self.handle_reset(actor_state)

            # Invoke actor and critic, generate actions/values
            actor_state = self.pre_env_step(actor_state)

            # Step env
            actor_state = self.env_step(actor_state)

            all_done_indices = actor_state["dones"].nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            actor_state["done_indices"] = done_indices

            # Store things in experience buffer
            actor_state = self.post_env_step(actor_state)
            actor_state = self.compute_next_values(actor_state)

        self.post_play_steps(actor_state)

    def handle_reset(self, actor_state):
        done_indices = actor_state["done_indices"]
        if self.force_full_restart:
            done_indices = None
            self.force_full_restart = False

        obs = self.env_reset(done_indices)
        actor_state["obs"] = obs

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state

    def get_extra_obs_from_env(self, actor_state):
        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                env_obs_name = self.extra_obs_inputs[key].get("env_obs_name", key)
                val = getattr(self.env, env_obs_name, None)
                assert val is not None, f"Env does not have attribute {env_obs_name}"
                actor_state[key] = val.view(-1, self.extra_obs_inputs[key].size)
        return actor_state

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return obs

    def env_step(self, actor_state):
        if self.use_llc:
            obs, rewards, dones, extras = self.env.step(actor_state["llc_actions"])
        else:
            obs, rewards, dones, extras = self.env.step(actor_state["actions"])
        rewards = rewards * self.task_reward_w
        actor_state.update(
            {"obs": obs, "rewards": rewards, "dones": dones, "extras": extras}
        )

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state

    def create_actor_state(self):
        return {"done_indices": []}

    def pre_env_step(self, actor_state):
        self.experience_buffer.update_data(
            "obs", actor_state["step"], actor_state["obs"]
        )
        if self.config.dual_env:
            self.experience_buffer.update_data(
            "obs_add_env", actor_state["step"], torch.cat([actor_state["obs"],self.extra_env_obs], dim=-1))
        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                self.experience_buffer.update_data(key, actor_state["step"], actor_state[key])
        # 
        actor_inputs = self.create_actor_args(actor_state)
        actor_outs = self.actor.eval_forward(actor_inputs)
        # 
        if self.use_rand_action_masks:
            rand_action_mask = torch.bernoulli(self.rand_action_probs)
            deterministic_actions = rand_action_mask == 0
            actor_outs["actions"][deterministic_actions] = actor_outs["mus"][
                deterministic_actions
            ]
            self.experience_buffer.update_data(
                "rand_action_mask", actor_state["step"], rand_action_mask
            )

        critic_inputs = self.create_critic_args(actor_state)

        values = self.critic(critic_inputs)
        if self.dual_critic:
            values2 = self.critic2(critic_inputs)

        if self.config.normalize_values:
            values = self.running_val_norm.normalize(values, un_norm=True)
        # 
        if self.use_llc:
            with torch.no_grad():
                llc_actions = self.llc_model.decode(actor_outs['actions'].clone(), actor_state['obs'].clone())
            actor_outs['llc_actions'] = llc_actions.clone()
        actor_state.update(actor_outs)

        # We want unnormalized values here.
        self.experience_buffer.update_data("values", actor_state["step"], values.view(-1))

        return actor_state

    def post_env_step(self, actor_state):
        self.current_rewards += actor_state["rewards"]
        self.current_lengths += 1

        done_indices = actor_state["done_indices"]

        self.episode_reward_meter.update(self.current_rewards[done_indices])
        self.episode_length_meter.update(self.current_lengths[done_indices])

        not_dones = 1.0 - actor_state["dones"].float()

        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * not_dones

        for k in self.actor_state_to_experience_buffer_list:
            self.experience_buffer.update_data(k, actor_state["step"], actor_state[k])

        self.episode_env_tensors.add(actor_state["extras"]["to_log"])

        return actor_state

    def compute_next_values(self, actor_state):
        critic_inputs = self.create_critic_args(actor_state)
        values = self.critic(critic_inputs).view(-1)
        if self.dual_critic:
            values2 = self.critic2(critic_inputs).view(-1)

        if self.config.normalize_values:
            values = self.running_val_norm.normalize(values, un_norm=True)

        next_values = values * (1 - actor_state["extras"]["terminate"].float())

        self.experience_buffer.update_data(
            "next_values", actor_state["step"], next_values
        )
        return actor_state

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.num_steps)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def pre_play_steps(self):
        pass

    def post_play_steps(self, actor_state):
        
        self.step_count += self.get_step_count_increment()

        rewards = self.experience_buffer.rewards
        self.last_scaled_task_rewards_mean = rewards.detach().mean()

        extra_rewards = self.calculate_extra_reward()

        self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
        total_rewards = rewards + extra_rewards

        self.experience_buffer.batch_update_data("total_rewards", total_rewards)

        advantages = self.discount_values(
            self.experience_buffer.dones,
            self.experience_buffer.values,
            total_rewards,
            self.experience_buffer.next_values,
        )
        returns = advantages + self.experience_buffer.values

        self.experience_buffer.batch_update_data("returns", returns)

        if self.config.normalize_advantage:
            if not self.use_rand_action_masks:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                adv_mask = (self.experience_buffer.rand_action_mask != 0).float()
                advantages = normalization_with_masks(advantages, adv_mask)

        self.experience_buffer.batch_update_data("advantages", advantages)

    @torch.no_grad()
    def generate_datasets(self):
        actor_critic_data_dict = self.experience_buffer.make_dict()
        if self.config.normalize_values:
            self.running_val_norm.update(actor_critic_data_dict["values"])
            self.running_val_norm.update(actor_critic_data_dict["returns"])

            actor_critic_data_dict["values"] = self.running_val_norm.normalize(
                actor_critic_data_dict["values"]
            )
            actor_critic_data_dict["returns"] = self.running_val_norm.normalize(
                actor_critic_data_dict["returns"]
            )

        # Saves memory
        if hasattr(self, "actor_critic_dataset"):
            del self.actor_critic_dataset
        
        self.actor_critic_dataset = DictDataset(
            self.config.batch_size, actor_critic_data_dict, shuffle=True
        )

    def actor_train_forward(self, batch_dict):
        return self.actor.training_forward(batch_dict)

    def actor_step(self, batch_idx) -> Tensor:
        dataset_idx = batch_idx % len(self.actor_critic_dataset)
        # Reshuffling the data at the beginning of each mini epoch.
        # Only doing this in the actor and not the critic to
        # avoid extra reshuffles.
        if dataset_idx == 0 and batch_idx != 0 and self.actor_critic_dataset.do_shuffle:
            self.actor_critic_dataset.shuffle()
        batch_dict = self.actor_critic_dataset[dataset_idx]

        actor_outs = self.actor_train_forward(batch_dict)
        actor_info = self.actor_loss(
            batch_dict["neglogp"],
            actor_outs["neglogp"],
            batch_dict["advantages"],
            self.e_clip,
        )
        actor_ppo_loss: Tensor = actor_info["actor_loss"]
        actor_clipped: Tensor = actor_info["actor_clipped"].float()

        if self.config.bounds_loss_coef > 0:
            bounds_loss: Tensor = (
                self.bounds_loss(actor_outs["mus"]) * self.config.bounds_loss_coef
            )
        else:
            bounds_loss = torch.zeros(self.num_envs, device=self.device)

        if self.use_rand_action_masks:
            rand_action_mask = batch_dict["rand_action_mask"]
            action_loss_mask = (rand_action_mask != 0).float()
            action_mask_sum = torch.sum(action_loss_mask)

            actor_ppo_loss = (actor_ppo_loss * action_loss_mask).sum() / action_mask_sum
            actor_clipped = (actor_clipped * action_loss_mask).sum() / action_mask_sum
            bounds_loss = (bounds_loss * action_loss_mask).sum() / action_mask_sum
        else:
            actor_ppo_loss = actor_ppo_loss.mean()
            actor_clipped = actor_clipped.mean()
            bounds_loss = bounds_loss.mean()

        extra_loss = self.calculate_extra_actor_loss(batch_idx, batch_dict)
        actor_loss = actor_ppo_loss + bounds_loss + extra_loss

        self.log(
            "actor/ppo_loss", actor_ppo_loss.detach(), on_epoch=True, on_step=False
        )
        self.log(
            "actor/bounds_loss", bounds_loss.detach(), on_epoch=True, on_step=False
        )
        self.log(
            "actor/extra_loss", extra_loss.detach(), on_epoch=True, on_step=False
        )
        self.log(
            "actor/clip_frac", actor_clipped.detach(), on_epoch=True, on_step=False
        )
        self.log(
            "losses/actor_loss", actor_loss.detach(), on_epoch=True, on_step=False
        )
        return actor_loss

    def bounds_loss(self, mu: Tensor) -> Tensor:
        soft_bound = 1.0
        mu_loss_high = (
            torch.maximum(mu - soft_bound, torch.tensor(0, device=self.device)) ** 2
        )
        mu_loss_low = (
            torch.minimum(mu + soft_bound, torch.tensor(0, device=self.device)) ** 2
        )
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        return b_loss

    def calculate_extra_actor_loss(self, batch_idx, batch_dict) -> Tensor:
        return torch.tensor(0.0, device=self.device)

    def critic_train_forward(self, batch_dict):
        if self.shared_actor_critic_base:
            critic_inputs = {"obs": self.actor.get_extracted_features(batch_dict)}
            return self.critic(critic_inputs)
        else:
            return self.critic(batch_dict)

    def critic_step(self, batch_idx) -> Tensor:
        
        batch_dict = self.actor_critic_dataset[
            batch_idx % len(self.actor_critic_dataset)
        ]
        
        if self.config.dual_env:
            critic_batch_input = {'obs': batch_dict['obs_add_env'].clone()}
            if self.extra_obs_inputs is not None:
                keys = list(self.extra_obs_inputs.keys())
                for key in keys:
                    critic_batch_input[key] = batch_dict[key]
            values = self.critic_train_forward(critic_batch_input)
        else:
            values = self.critic_train_forward(batch_dict)

        if self.config.clip_critic_loss:
            critic_loss_unclipped = (values - batch_dict["returns"]).pow(2)
            v_clipped = batch_dict["values"] + torch.clamp(
                values - batch_dict["values"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            critic_loss_clipped = (v_clipped - batch_dict["returns"]).pow(2)
            critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
            critic_loss = 0.5 * critic_loss_max.mean()
        else:
            critic_loss = 0.5 * (batch_dict["returns"] - values).pow(2).mean()

        self.log(
            "losses/critic_loss", critic_loss.detach(), on_epoch=True, on_step=False
        )
        return critic_loss

    def actor_loss(
        self, old_action_neglogprobs, action_neglogprobs, advantage, curr_e_clip
    ):
        # = p(actions) / p_old(actions)
        ratio = torch.exp(old_action_neglogprobs - action_neglogprobs)

        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {"actor_loss": ppo_loss, "actor_clipped": clipped}
        return info

    def on_fit_start(self):
        if self.config.seed is None:
            seed = random.randint(0, 1000)
        else:
            seed = self.config.seed

        seed = seed + self.global_rank
        print(f"Seeding global rank {self.global_rank} with seed {seed}")
        seed_everything(seed)

        self.initialize_env()
        self.env_reset()
        self.fit_start_time = time.time()
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("epoch")
        self.time_report.start_timer("algorithm")

    def on_fit_end(self) -> None:
        
        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.save()

    def configure_optimizers(self):
        actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=self.actor_params_for_optimizer(),
            _convert_="all",
        )

        actor_conf = {"optimizer": actor_optimizer}

        critic_optimizer = instantiate(
            self.config.critic_optimizer, params=self.critic_params_for_optimizer()
        )
        if self.dual_critic:
            critic2_optimizer = instantiate(
                self.config.critic_optimizer, params=self.critic2_params_for_optimizer()
            )

        critic_conf = {"optimizer": critic_optimizer}

        if self.config.actor_lr_scheduler is not None:
            actor_lr_scheduler = instantiate(
                self.config.actor_lr_scheduler, optimizer=actor_optimizer
            )

            actor_conf["lr_scheduler"] = actor_lr_scheduler

        if self.config.critic_lr_scheduler is not None:
            critic_lr_scheduler = instantiate(
                self.config.critic_lr_scheduler, optimizer=critic_optimizer
            )

            critic_conf["lr_scheduler"] = critic_lr_scheduler

        return actor_conf, critic_conf

    @rank_zero_only
    def save(self):
        save_dir = Path(self.trainer.loggers[0].log_dir)
        self.trainer.save_checkpoint(save_dir / "last.ckpt")
        if self.current_epoch % self.config.save_freq == 0 and self.current_epoch>0:
            self.trainer.save_checkpoint(save_dir / f"Epoch_{(self.current_epoch):06d}.ckpt")
        for logger in self.loggers:
            if isinstance(logger, WandbLogger) and self.config.wandb_log_models:
                logger.experiment.save(save_dir / "last.ckpt")

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        self.time_report.start_timer("epoch")

    def on_train_epoch_end(self) -> None:
        
        self.time_report.end_timer("epoch")
        self.post_epoch_logging()
        if self.schedules is not None:
            for s in self.schedules:
                s.step()

        if (
            self.config.eval_metrics_every is not None
            and self.current_epoch > 0
            and self.current_epoch % self.config.eval_metrics_every == 0
        ):
            self.calc_eval_metrics()
            
        if self.current_epoch % self.config.manual_save_every == 0:
            self.save()

        self.env.on_epoch_end(self.current_epoch)

    def calc_eval_metrics(self):
        pass

    def post_epoch_logging(self):
        end_time = time.time()
        self.log(
            "info/episode_length",
            self.episode_length_meter.get_mean().item(),
        )
        self.log(
            "info/episode_reward",
            self.episode_reward_meter.get_mean().item(),
        )
        self.log(
            "info/frames",
            torch.tensor(self.step_count),
        )
        self.log(
            "info/gframes", torch.tensor(self.step_count / (10**9)), prog_bar=True
        )
        self.log(
            "times/fps_last_epoch",
            self.get_step_count_increment() / (end_time - self.epoch_start_time),
            prog_bar=True,
        )
        self.log("times/fps_total", self.step_count / (end_time - self.fit_start_time))
        self.log("times/training_hours", (end_time - self.fit_start_time) / 3600)
        self.log("times/training_minutes", (end_time - self.fit_start_time) / 60)
        self.log("times/last_epoch_seconds", (end_time - self.epoch_start_time))

        self.log("rewards/task_rewards", self.experience_buffer.rewards.mean())
        self.log("rewards/extra_rewards", self.experience_buffer.extra_rewards.mean())
        self.log("rewards/total_rewards", self.experience_buffer.total_rewards.mean())

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            self.log_dict(env_log_dict)

        self.log("actor/grad_norm_before_clip", self.actor_grad_norm_before_clip)
        self.log("actor/grad_norm_after_clip", self.actor_grad_norm_after_clip)
        self.log("critic/grad_norm_before_clip", self.critic_grad_norm_before_clip)
        self.log("critic/grad_norm_after_clip", self.critic_grad_norm_after_clip)
        self.log(
            "actor/bad_grads_count",
            self.actor_bad_grads_count,
            sync_dist=True,
            reduce_fx="sum",
        )
        self.log(
            "critic/bad_grads_count",
            self.critic_bad_grads_count,
            sync_dist=True,
            reduce_fx="sum",
        )

    def create_actor_args(self, actor_state):
        actor_args = {"obs": actor_state["obs"]}

        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                if key in actor_state:
                    actor_args[key] = actor_state[key]

        return actor_args

    def create_critic_args(self, actor_state):
        if self.shared_actor_critic_base:
            actor_args = self.create_actor_args(actor_state)
            critic_args = {"obs": self.actor.get_extracted_features(input_dict=actor_args)}
        else:
            if self.config.dual_env:
                critic_args = {"obs": torch.cat([actor_state["obs"], self.extra_env_obs],dim=-1)}
            else:
                critic_args = {"obs": actor_state["obs"]}

            if self.extra_obs_inputs is not None:
                for key in self.extra_obs_inputs.keys():
                    if key in actor_state:
                        critic_args[key] = actor_state[key]
        return critic_args

    def calculate_extra_reward(self):
        return torch.zeros(self.num_steps, self.num_envs, device=self.device)

    def get_step_count_increment(self):
        return self.num_steps * self.num_envs * self.config.ngpu

    def initialize_env(self):
        self.env: DiscHumanoid = instantiate(self.config.env, device=self.device)
        if self.config.backbone == 'isaacsim':
            task = instantiate(self.config.env.config.task, device=self.device)
            task.set_env(self.env)
            sim_config = task.sim_config.get_physics_params()
            self.env.set_task(task, sim_config)

    def ac_max_num_batches(self):
        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.num_mini_epochs
            / self.config.batch_size
        )

    def max_num_batches(self):
        return self.ac_max_num_batches()

    def actor_params_for_optimizer(self):
        return list(self.actor.parameters())

    def critic_params_for_optimizer(self):
        return list(self.critic.parameters())

    def critic2_params_for_optimizer(self):
        return list(self.critic2.parameters())

    @torch.no_grad()
    def evaluate_policy(self):
        self.eval()
        self.create_eval_callbacks()
        self.pre_evaluate_policy()

        actor_state = self.create_eval_actor_state()
        step = 0
        games_count = 0
        while (
            not actor_state["stop"]
            and (self.config.num_games is None or games_count < self.config.num_games)
            and (
                self.config.max_eval_steps is None or step < self.config.max_eval_steps
            )
        ):
            
            actor_state["step"] = step
            actor_state["games_count"] = games_count

            actor_state = self.handle_eval_reset(actor_state)

            # Invoke actor and critic, generate actions/values
            actor_state = self.pre_eval_env_step(actor_state)

            # Step env
            actor_state = self.eval_env_step(actor_state)

            all_done_indices = actor_state["dones"].nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            actor_state["done_indices"] = done_indices

            actor_state = self.post_eval_env_step(actor_state)

            games_count += len(done_indices)
            step += 1

        self.post_evaluate_policy()

    def create_eval_callbacks(self):
        if self.config.export_trajectory:
            self.eval_callbacks.append(
                instantiate(self.config.export_trajectory_cb, training_loop=self)
            )

        if self.config.export_video:
            self.eval_callbacks.append(
                instantiate(self.config.export_video_cb, training_loop=self)
            )

        if self.config.export_motion:
            self.eval_callbacks.append(
                instantiate(self.config.export_motion_cb, training_loop=self)
            )

    def pre_evaluate_policy(self, reset_env=True):
        self.eval()
        self.initialize_env()
        if reset_env:
            self.env_reset()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def create_eval_actor_state(self):
        return {"done_indices": [], "stop": False}

    def handle_eval_reset(self, actor_state):
        done_indices = actor_state["done_indices"]
        obs = self.env_reset(done_indices)
        actor_state["obs"] = obs

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state

    def pre_eval_env_step(self, actor_state: dict):
        actor_inputs = self.create_eval_actor_args(actor_state)

        # 

        actor_outs = self.actor.eval_forward(actor_inputs)

        # 
        if self.use_llc:
            with torch.no_grad():
                # llc_actions = self.llc_model.decode(torch.randn_like(actor_outs['mus']), actor_inputs['obs'].clone())
                llc_actions = self.llc_model.decode(actor_outs['mus'].clone(), actor_inputs['obs'].clone())
            actor_outs['llc_actions'] = llc_actions.clone()

        actor_state.update(actor_outs)
        actor_state["sampled_actions"] = actor_state["actions"]

        # By default, use deterministic policy in eval
        # (unless overriden in callbacks).
        actor_state["actions"] = actor_state["mus"]

        

        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        
        
        if self.save_data_for_cvae:
            
            self.data_for_cvae.append([actor_state['obs'].clone(),actor_state['mimic_scene'].clone(),actor_state['actions'].clone(),self.env.progress_buf.clone()])
            if len(self.data_for_cvae)>200:
                
                torch.save(self.data_for_cvae, "tempsupp.pt")
            



        return actor_state

    def eval_env_step(self, actor_state):
        if self.use_llc:
            obs, rewards, dones, extras = self.env.step(actor_state["llc_actions"])
        else:
            obs, rewards, dones, extras = self.env.step(actor_state["actions"])

        actor_state.update(
            {"obs": obs, "rewards": rewards, "dones": dones, "extras": extras}
        )

        actor_state = self.get_extra_obs_from_env(actor_state)

        if not self.env.headless:
            self.eval_step_print(actor_state)
        return actor_state

    def eval_step_print(self, actor_state):
        # print(actor_state["rewards"][0].item(), actor_state["dones"][0].item())
        pass

    def post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def create_eval_actor_args(self, actor_state):
        actor_args = {"obs": actor_state["obs"]}

        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                if key in actor_state:
                    actor_args[key] = actor_state[key]

        return actor_args

    def handle_actor_grad_clipping(self):
        actor_params = get_params(self.actor_params_for_optimizer())
        self.actor_grad_norm_before_clip = torch_utils.grad_norm(actor_params)

        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(self.actor_grad_norm_before_clip)
                or self.actor_grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(self.actor_grad_norm_before_clip)

        # sanity check
        if bad_grads:

            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in actor_params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {self.actor_grad_norm_before_clip.item()}"
                )
            else:
                self.actor_bad_grads_count += 1
                for p in actor_params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            clip_grad_norm_(actor_params, self.config.gradient_clip_val)
        self.actor_grad_norm_after_clip = torch_utils.grad_norm(actor_params)

    def handle_critic_grad_clipping(self):
        critic_params = get_params(self.critic_params_for_optimizer())
        self.critic_grad_norm_before_clip = torch_utils.grad_norm(critic_params)

        if self.config.check_grad_mag:
            bad_grads = (
                    torch.isnan(self.critic_grad_norm_before_clip)
                    or self.critic_grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(self.critic_grad_norm_before_clip)

        # sanity check
        if bad_grads:
            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in critic_params if p.grad is not None],
                    dim=0,
                )
                print(
                    "NaN gradient",
                    all_params.isfinite().logical_not().float().mean().item(),
                    all_params.abs().min().item(),
                    all_params.abs().max().item(),
                    self.critic_grad_norm_before_clip.item(),
                )
                raise ValueError
            else:
                self.critic_bad_grads_count += 1
                for p in critic_params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            clip_grad_norm_(critic_params, self.config.gradient_clip_val)
        self.critic_grad_norm_after_clip = torch_utils.grad_norm(critic_params)

    def log_image(self, key: str, image):
        for logger in self.loggers:
            experiment = logger.experiment

            if isinstance(logger, WandbLogger):
                logger.log_image(key=key, images=[image])
            elif isinstance(logger, TensorBoardLogger):
                image_array = np.array(image)
                experiment.add_image(
                    key, image_array, dataformats="HWC", global_step=self.global_step
                )
            else:
                raise ValueError(f"Unsupported logger type {type(logger)}")

    def terminate_early(self):
        self.trainer.should_stop = True


def normalization_with_masks(values: Tensor, masks: Optional[Tensor]):
    if masks is None:
        return (values - values.mean()) / (values.std() + 1e-8)

    values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values


def get_mean_var_with_masks(values: Tensor, masks: Tensor):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = (((values_mask) ** 2) / sum_mask).sum() - (
        (values_mask / sum_mask).sum()
    ) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)
    return values_mean, values_var
