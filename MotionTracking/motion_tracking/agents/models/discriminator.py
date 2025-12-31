from typing import List

import torch
from torch import nn

from motion_tracking.agents.models.critic import CriticMLP, CriticMLP2

DISC_LOGIT_INIT_SCALE = 1.0


class JointDiscMLP(CriticMLP):
    def __init__(self, config, num_in: int, num_out: int =  1):
        super().__init__(config, num_in, num_out)
        torch.nn.init.uniform_(
            self.trunk[-1].weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE
        )

    def all_jd_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        return [self.trunk[-1].weight]
