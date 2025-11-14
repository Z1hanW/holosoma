from __future__ import annotations

import copy
import os
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from holosoma.utils.safe_torch_import import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element: list | dict
        val = getattr(obj, key)
        if isinstance(val, list):
            element = []
            element.extend(class_to_dict(item) for item in val)
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def pre_process_config(config) -> None:
    # compute observation_dim

    obs_dim_dict = {}
    _obs_key_list = config.env.config.obs.obs_dict

    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # Handle multiple formats for obs_dims, allows for local checkpoint loading vs. only from wandb URIs:
    if isinstance(config.env.config.obs.obs_dims, (dict, DictConfig)):
        # Loading from saved config, already in the correct format
        each_dict_obs_dims = dict(config.env.config.obs.obs_dims)
    else:
        # Loading directly from wandb://
        each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}

    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")
    auxiliary_obs_dims: dict[str, int] = {}
    if hasattr(config.env.config.obs, "obs_auxiliary"):
        _aux_obs_key_list = config.env.config.obs.obs_auxiliary
        auxiliary_obs_dims = {}
        for aux_obs_key, aux_config in _aux_obs_key_list.items():
            auxiliary_obs_dims[aux_obs_key] = 0
            for _key, _num in aux_config.items():
                assert _key in config.env.config.obs.obs_dims
                auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
        logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"):
                processed_key = key[:-4]
            else:
                processed_key = key
            if processed_key in config.env.config.obs.obs_dims:
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[processed_key]
                logger.info(f"{obs_key}: {processed_key} has dim: {config.env.config.obs.obs_dims[processed_key]}")
            elif processed_key in auxiliary_obs_dims:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[processed_key]
                logger.info(f"{obs_key}: {processed_key} has dim: {auxiliary_obs_dims[processed_key]}")
            else:
                logger.error(f"{obs_key}: {processed_key} not found in obs_dims")
                raise ValueError(f"{obs_key}: {processed_key} not found in obs_dims")

    OmegaConf.set_struct(config, False)
    if "robot" not in config:
        # `robot` may be missing in the config due to the hydra->tyro migration
        config.robot = {}
    config.robot.algo_obs_dim_dict = obs_dim_dict
    OmegaConf.set_struct(config, True)
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    # compute action_dim for ppo
    # for agent in config.algo.config.network_dict.keys():
    #     for network in config.algo.config.network_dict[agent].keys():
    #         output_dim = config.algo.config.network_dict[agent][network].output_dim
    #         if output_dim == "action_dim":
    #             config.algo.config.network_dict[agent][network].output_dim = config.env.config.robot.actions_dim

    # print the config
    if hasattr(config, "algo") and hasattr(config.algo, "config") and hasattr(config.algo.config, "module_dict"):
        logger.debug("PPO CONFIG")
        logger.debug(f"{config.algo.config.module_dict}")
    # logger.debug(f"{config.algo.config.network_dict}")


def parse_observation(
    cls: Any,
    obs_key: str,
    key_list: list[str],
    buf_dict: dict[str, torch.Tensor],
    obs_scales: dict[str, float],
    noise_scales: dict[str, float],
    noise_levels: dict[str, float],
    current_noise_curriculum_value: Any = 1.0,
) -> None:
    """Parse observations for the legged_robot_base class"""
    noise_level = noise_levels[obs_key]
    # print(f"current_noise_curriculum_value: {current_noise_curriculum_value}")
    # print(f"noise_level: {noise_level}")
    for key in key_list:
        obs_noise = noise_scales[key] * current_noise_curriculum_value * noise_level
        actor_obs = getattr(cls, f"_get_obs_{key}")().clone()
        obs_scale = obs_scales[key]
        # Yuanhang: use rand_like (uniform 0-1) instead of randn_like (N~[0,1])
        # buf_dict[key] = actor_obs * obs_scale + (torch.randn_like(actor_obs)* 2. - 1.) * obs_noise
        buf_dict[key] = (actor_obs + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise) * obs_scale


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, "memory_a"):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_1.pt")
        model = copy.deepcopy(actor_critic.actor).to("cpu")
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_lstm_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
