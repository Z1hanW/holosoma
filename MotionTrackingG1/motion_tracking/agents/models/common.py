import torch
from torch import nn
import torch.nn.functional as F

from motion_tracking.utils.running_mean_std import RunningMeanStd


class NormObsBase(nn.Module):
    def __init__(self, config, num_in, num_out=None):
        super().__init__()
        self.config = config
        self.num_in = num_in
        self.build_norm()

    def build_norm(self):
        if self.config.normalize_obs:
            self.running_obs_norm = RunningMeanStd(
                shape=self.get_norm_shape(),
                device="cpu",
                clamp_value=self.config.obs_clamp_value,
            )

    def get_norm_shape(self):
        return (self.num_in,)

    def maybe_normalize_obs(self, obs):
        if not self.config.normalize_obs:
            return obs

        # Only update obs during training
        if self.training:
            self.running_obs_norm.update(obs)
        return self.running_obs_norm.normalize(obs)


class Flatten(NormObsBase):
    def __init__(self, config=None, num_in=None, num_out=None):
        super().__init__(config, num_in)
        self.flatten = nn.Flatten()

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = input_dict["obs"]
        obs = self.flatten(obs)
        obs = (
            self.maybe_normalize_obs(obs)
            if not already_normalized
            else obs
        )
        if return_norm_obs:
            assert not already_normalized
            return {"outs": obs, "norm_obs": obs}
        else:
            return obs


class Embedding(NormObsBase):
    def __init__(self, config, num_in, num_out):
        super().__init__(config, num_in)
        self.embedding = nn.Embedding(num_in, num_out)

        if self.config.random_embedding.use_random_embeddings:
            self.random_embedding_vectors = torch.rand((self.config.random_embedding.num_random_embeddings, num_out))

    def _apply(self, fn):
        super()._apply(fn)
        if self.config.random_embedding.use_random_embeddings:
            self.random_embedding_vectors = fn(self.random_embedding_vectors)
        return self

    def forward(self, input_dict):
        obs = input_dict["obs"]

        if self.config.random_embedding.use_random_embeddings:
            embedding = self.embedding(obs * 0)  # CT hack: ensure indices within range
            embedding = self.random_embedding_vectors[torch.fmod(obs.view(-1), self.config.random_embedding.num_random_embeddings)].reshape(embedding.shape)
        else:
            embedding = self.embedding(obs)

        if self.config.normalized_embedding:
            embedding = F.normalize(embedding, dim=-1)
        else:
            embedding = embedding

        return self.maybe_normalize_obs(embedding)
