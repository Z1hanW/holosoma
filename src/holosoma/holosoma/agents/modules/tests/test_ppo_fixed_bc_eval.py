from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO


class _DummyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def act_inference(self, policy_state_dict):
        return policy_state_dict["actor_obs"] + 1.0


def _make_stub_ppo() -> PPO:
    ppo = object.__new__(PPO)
    ppo.is_main_process = True
    ppo.fixed_bc_eval_num_samples = 2
    ppo.fixed_bc_eval_log_interval = 1
    ppo.dagger_enabled = True
    ppo.dagger_ignore_zero_teacher_actions = True
    ppo.actor_perception_key = ""
    ppo._fixed_bc_eval_ready = False
    ppo._fixed_bc_eval_size = 0
    ppo._fixed_bc_eval_actor_obs_parts = []
    ppo._fixed_bc_eval_teacher_actions_parts = []
    ppo._fixed_bc_eval_actor_perception_parts = []
    ppo._fixed_bc_eval_dataset = {}
    ppo.device = "cpu"
    ppo.actor = _DummyActor()
    ppo.actor_obs_normalizers = {"actor_obs": nn.Identity()}
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update=False: obs, ppo)
    return ppo


def test_capture_fixed_bc_eval_samples_respects_mask_and_zero_teacher():
    ppo = _make_stub_ppo()
    actor_obs_raw = torch.tensor([[1.0], [2.0], [3.0]])
    teacher_actions = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    teacher_bc_mask = torch.tensor([[True], [True], [False]])

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=actor_obs_raw,
        actor_perception_obs=None,
        teacher_actions=teacher_actions,
        teacher_bc_mask=teacher_bc_mask,
    )

    assert ppo._fixed_bc_eval_ready is False
    assert ppo._fixed_bc_eval_size == 1

    ppo._maybe_capture_fixed_bc_eval_samples(
        actor_obs_raw=torch.tensor([[4.0], [5.0]]),
        actor_perception_obs=None,
        teacher_actions=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        teacher_bc_mask=torch.tensor([[True], [True]]),
    )

    assert ppo._fixed_bc_eval_ready is True
    assert ppo._fixed_bc_eval_dataset["actor_obs_raw"].shape == (2, 1)
    assert ppo._fixed_bc_eval_dataset["teacher_actions"].shape == (2, 2)
    assert torch.equal(ppo._fixed_bc_eval_dataset["actor_obs_raw"].squeeze(-1), torch.tensor([2.0, 4.0]))


def test_fixed_bc_eval_metrics_use_deterministic_actor_mean():
    ppo = _make_stub_ppo()
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "teacher_actions": torch.tensor([[2.0, 3.0], [5.0, 4.0]]),
    }

    metrics = ppo._get_fixed_bc_eval_metrics(current_iteration=4)

    assert metrics["fixed_bc_num_samples"] == 2.0
    assert metrics["fixed_bc_mu_mse"] == 0.5
