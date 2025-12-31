from motion_tracking.agents.models.mlp import MultiHeadedMLP,MultiHeadedMLP2


class CriticMLP(MultiHeadedMLP):
    def __init__(self, config, num_in: int, num_out: int = 1):
        super().__init__(config, num_in, num_out)

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        # import pdb;pdb.set_trace()
        out = super().forward(
            input_dict,
            return_norm_obs=return_norm_obs,
            already_normalized=already_normalized,
        )
        if return_norm_obs:
            out["outs"] = out["outs"].squeeze(-1)
            return out
        else:
            return out.squeeze(-1)
        
class CriticMLP2(MultiHeadedMLP2):
    def __init__(self, config, num_in: int, num_out: int = 1):
        super().__init__(config, num_in, num_out)

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        import pdb;pdb.set_trace()
        out = super().forward(
            input_dict,
            return_norm_obs=return_norm_obs,
            already_normalized=already_normalized,
        )
        if return_norm_obs:
            out["outs"] = out["outs"].squeeze(-1)
            return out
        else:
            return out.squeeze(-1)
