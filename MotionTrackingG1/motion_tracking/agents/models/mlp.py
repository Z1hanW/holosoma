import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn, Tensor
from copy import copy
from motion_tracking.agents.models.common import NormObsBase
from motion_tracking.utils import model_utils


def default_init(m: nn.Linear):
    m.bias.data.zero_()
    return m


INIT_DICT = {
    "orthogonal": lambda m: model_utils.init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
    ),
    "default": default_init,
}


def build_mlp(config, num_in: int, num_out: int):
    init = INIT_DICT[config.initializer]

    indim = num_in
    layers = []
    for outdim in config.units:
        layers.append(init(nn.Linear(indim, outdim)))
        layers.append(model_utils.get_activation_func(config.activation))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(outdim))
        indim = outdim

    layers.append(init(nn.Linear(outdim, num_out)))
    mlp = nn.Sequential(*layers)
    return mlp


class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in)
        self.num_out = num_out
        self.trunk = build_mlp(self.config, self.num_input_units(), num_out)

    def num_input_units(self):
        return self.num_in

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        outs: Tensor = self.trunk(obs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs


class MultiHeadedMLP(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int):
        self.extra_input_keys = []
        self.connected_keys = {}
        if config.extra_inputs is not None:
            self.extra_input_keys = sorted(config.extra_inputs.keys())

            for extra_input_key, extra_input in config.extra_inputs.items():
                if extra_input.config.get("connected_keys", None) is not None:
                    self.connected_keys[extra_input_key] = sorted(extra_input.config.connected_keys)
                    for connected_key in self.connected_keys[extra_input_key]:
                        self.extra_input_keys.remove(connected_key)
                else:
                    self.connected_keys[extra_input_key] = []

        self.feature_size = num_in
        extra_input_models = {}
        for key in self.extra_input_keys:
            model = instantiate(
                config.extra_inputs[key]
            )
            extra_input_models[key] = model
            self.feature_size += config.extra_inputs[key].num_out

        super().__init__(config, num_in, num_out)
        self.extra_input_models = nn.ModuleDict(extra_input_models)

    def num_input_units(self):
        return self.feature_size

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        cat_obs = obs
        for key in self.extra_input_keys:
            
            key_obs = {"obs": input_dict[key]}
            for connected_key in self.connected_keys[key]:
                key_obs[connected_key] = input_dict[connected_key]
            
            if self.config.extra_inputs[key].config.get("operations") is None:
                cat_obs = torch.cat([cat_obs, self.extra_input_models[key](key_obs)], dim=-1)
            else: 
                batch_size = input_dict["obs"].shape[0]
                key_obs = input_dict[key]
                for operation in self.config.extra_inputs[key].config.get("operations", []):
                    
                    if operation.type == "reshape":
                        new_shape = copy(operation.new_shape)
                        if new_shape[0] == "batch_size":
                            new_shape[0] = batch_size
                        key_obs = key_obs.reshape(*new_shape)
                    elif operation.type == "encode":
                        key_obs = {"obs": key_obs}
                        key_obs = self.extra_input_models[key](key_obs)                    
                    elif operation.type == "maxpooling":
                        key_obs = torch.max(key_obs, dim=1)[0]
                cat_obs = torch.cat([cat_obs, key_obs], dim=-1)
                    
        outs: Tensor = self.trunk(cat_obs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs
        
class MultiHeadedMLP2(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int):
        self.extra_input_keys = []
        self.connected_keys = {}
        if config.extra_inputs is not None:
            self.extra_input_keys = sorted(config.extra_inputs.keys())

            for extra_input_key, extra_input in config.extra_inputs.items():
                if extra_input.config.get("connected_keys", None) is not None:
                    self.connected_keys[extra_input_key] = sorted(extra_input.config.connected_keys)
                    for connected_key in self.connected_keys[extra_input_key]:
                        self.extra_input_keys.remove(connected_key)
                else:
                    self.connected_keys[extra_input_key] = []

        self.feature_size = num_in
        extra_input_models = {}
        for key in self.extra_input_keys:
            model = instantiate(
                config.extra_inputs[key]
            )
            extra_input_models[key] = model
            self.feature_size += config.extra_inputs[key].num_out

        super().__init__(config, num_in, num_out)
        self.extra_input_models = nn.ModuleDict(extra_input_models)

    def num_input_units(self):
        return self.feature_size

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        cat_obs = obs
        for key in self.extra_input_keys:
            
            key_obs = {"obs": input_dict[key]}
            for connected_key in self.connected_keys[key]:
                key_obs[connected_key] = input_dict[connected_key]
            
            if self.config.extra_inputs[key].config.get("operations") is None:
                cat_obs = torch.cat([cat_obs, self.extra_input_models[key](key_obs)], dim=-1)
            else: 
                batch_size = input_dict["obs"].shape[0]
                key_obs = input_dict[key]
                for operation in self.config.extra_inputs[key].config.get("operations", []):
                    
                    if operation.type == "reshape":
                        new_shape = copy(operation.new_shape)
                        if new_shape[0] == "batch_size":
                            new_shape[0] = batch_size
                        key_obs = key_obs.reshape(*new_shape)
                    elif operation.type == "encode":
                        key_obs = {"obs": key_obs}
                        key_obs = self.extra_input_models[key](key_obs)                    
                    elif operation.type == "maxpooling":
                        key_obs = torch.max(key_obs, dim=1)[0]
                cat_obs = torch.cat([cat_obs, key_obs], dim=-1)
                    
        outs: Tensor = self.trunk(cat_obs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs


class MultiOutputNetwork(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in)
        self.trunk = instantiate(self.config.trunk, num_in=self.num_input_units())

        self.output_keys = sorted(config.outputs.keys())

        output_models = {}
        for key in self.output_keys:
            model = instantiate(
                config.outputs[key]
            )
            output_models[key] = model
        self.output_models = nn.ModuleDict(output_models)

    def num_input_units(self):
        return self.num_in

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        outs: Tensor = self.trunk(input_dict, already_normalized, return_norm_obs)

        if return_norm_obs:
            outs, obs = outs["outs"], outs["norm_obs"]

        outputs = {}
        for output_model_key in self.output_models.keys():
            outputs[output_model_key] = self.output_models[output_model_key]({"obs": outs})

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outputs, "norm_obs": obs}
        else:
            return outputs
