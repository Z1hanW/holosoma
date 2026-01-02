from typing import Tuple

import math
import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from vector_quantize_pytorch import FSQ

from motion_tracking.agents.models.mlp import MultiHeadedMLP, MLP_WithNorm, MultiOutputNetwork


class PPO_Actor(nn.Module):
    def __init__(self, config, num_in, num_act):
        super().__init__()
        self.config = config
        self.num_in = num_in
        self.num_act = num_act

    def training_forward(self, input_dict):
        assert self.training

        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        prev_neglogp = self.neglogp(input_dict["actions"], mu, sigma, logstd)
        return {"mus": mu, "sigmas": sigma, "neglogp": prev_neglogp}

    def eval_forward(self, input_dict):
        assert not self.training
        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        neglogp = self.neglogp(selected_action, mu, sigma, logstd)
        return {
            "mus": mu,
            "sigmas": sigma,
            "actions": selected_action,
            "neglogp": neglogp,
        }

    def get_mu_logstd(self, input_dict) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def get_extracted_features(self, input_dict):
        raise NotImplementedError()

    @staticmethod
    def neglogp(x, mean, std, logstd):
        return (
            0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
            + logstd.sum(dim=-1)
        )

    def logstd_tick(self, current_epoch: int):
        pass


class ActorFixedSigma(PPO_Actor):
    logstd: Tensor

    def __init__(self, config, num_in, num_act):
        super().__init__(config, num_in, num_act)
        self.mu_model: MultiHeadedMLP = instantiate(
            self.config.mu_model, num_in=num_in, num_out=num_act
        )
        self.logstd = nn.Parameter(
            torch.ones(num_act) * config.init_logstd,
            requires_grad=self.config.learnable_sigma,
        )

    def get_mu_logstd(self, input_dict):
        mu = self.mu_model(input_dict)
        return mu, self.logstd

    def get_features_size(self):
        if hasattr(self.mu_model, "get_extracted_features") and callable(self.mu_model.get_extracted_features):
            return self.mu_model.get_features_size()

    def get_extracted_features(self, input_dict):
        if hasattr(self.mu_model, "get_extracted_features") and callable(self.mu_model.get_extracted_features):
            return self.mu_model.get_extracted_features(input_dict)
        else:
            return input_dict["obs"]

    def logstd_tick(self, current_epoch: int):
        schedule = self.config.sigma_schedule
        if not self.config.learnable_sigma or schedule is None:
            return

        current_mean = self.logstd.mean()

        target_mean = self.config.init_logstd + min(
            current_epoch / schedule.end_epoch, 1
        ) * (schedule.end_logstd - self.config.init_logstd)

        adjust = target_mean - current_mean
        with torch.no_grad():
            self.logstd += adjust

    def set_logstd(self, value: float):
        if self.config.learnable_sigma:
            raise NotImplementedError("You should not manually update sigma when it is learnable.")
        with torch.no_grad():
            self.logstd[:] = value


class ActorFixedSigmaVAE(ActorFixedSigma):
    def __init__(self, config, num_in, num_act):
        super().__init__(config, num_in, num_act)

        self.vae_dim = self.config.vae_dim

        self.vae_encoder: MLP_WithNorm = instantiate(self.config.vae_encoder, num_in=num_in)
        self.vae_prior: MLP_WithNorm = instantiate(self.config.vae_prior, num_in=num_in)

        self.vae_random_latent = self.config.vae_random_latent

    def reparameterization(self, mean, std, vae_noise):
        z = mean + std * vae_noise  # reparameterization trick
        return z

    def get_vae_mu_logvar(self, input_dict):
        if not self.config.residual_encoder:
            if self.vae_random_latent:
                mu_logvar_prior = self.vae_prior(input_dict)

                assert isinstance(mu_logvar_prior, dict)
                mu = mu_logvar_prior["mu"]
                logvar = mu_logvar_prior["logvar"]
            else:
                mu_logvar_posterior = self.vae_encoder(input_dict)
                assert isinstance(mu_logvar_posterior, dict)
                mu = mu_logvar_posterior["mu"]
                logvar = mu_logvar_posterior["logvar"]
        else:
            mu_logvar_prior = self.vae_prior(input_dict)

            assert isinstance(mu_logvar_prior, dict)
            mu_prior = mu_logvar_prior["mu"]
            logvar_prior = mu_logvar_prior["logvar"]

            if self.vae_random_latent:
                mu_posterior = 0
                logvar_posterior = 0
            else:
                mu_logvar_posterior = self.vae_encoder(input_dict)
                assert isinstance(mu_logvar_posterior, dict)
                mu_posterior = mu_logvar_posterior["mu"]
                logvar_posterior = mu_logvar_posterior["logvar"] - logvar_prior
            mu = mu_prior + mu_posterior
            logvar = logvar_prior + logvar_posterior

        return mu, logvar

    def training_forward(self, input_dict):
        assert self.training

        vae_latent = self.get_latent(input_dict)
        input_dict["vae_latent"] = vae_latent

        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        prev_neglogp = self.neglogp(input_dict["actions"], mu, sigma, logstd)
        return {
            "mus": mu,
            "sigmas": sigma,
            "neglogp": prev_neglogp,
        }

    def eval_forward(self, input_dict):
        assert not self.training

        vae_latent = self.get_latent(input_dict)
        input_dict["vae_latent"] = vae_latent

        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        neglogp = self.neglogp(selected_action, mu, sigma, logstd)
        return {
            "mus": mu,
            "sigmas": sigma,
            "actions": selected_action,
            "neglogp": neglogp,
        }

    def kl_loss(self, input_dict):
        mu_logvar_prior = self.vae_prior(input_dict)
        assert isinstance(mu_logvar_prior, dict)
        mu_prior = mu_logvar_prior["mu"]
        logvar_prior = mu_logvar_prior["logvar"]

        mu_logvar_posterior = self.vae_encoder(input_dict)
        assert isinstance(mu_logvar_posterior, dict)
        mu_posterior = mu_logvar_posterior["mu"]
        logvar_posterior = mu_logvar_posterior["logvar"]

        if not self.config.residual_encoder:
            return 0.5 * (logvar_prior - logvar_posterior + torch.exp(logvar_posterior) / torch.exp(logvar_prior) + (mu_posterior - mu_prior).pow(2) / torch.exp(logvar_prior) - 1)
        else:
            return 0.5 * (logvar_prior - logvar_posterior + torch.exp(logvar_posterior) / torch.exp(logvar_prior) + mu_posterior ** 2 / torch.exp(logvar_prior) - 1)

    def get_latent(self, input_dict):
        vae_mu, vae_logvar = self.get_vae_mu_logvar(input_dict)

        z = self.reparameterization(vae_mu, torch.exp(0.5 * vae_logvar), input_dict["vae_noise"])

        return z


class ActorFixedSigmaVQVAE(ActorFixedSigma):
    def __init__(self, config, num_in, num_act):
        super().__init__(config, num_in, num_act)

        self.vae_encoder: MLP_WithNorm = instantiate(self.config.vae_encoder, num_in=num_in)

        self.vae_quantizer = FSQ(
            levels=self.config.vae_quantizer.levels,
        )

    def training_forward(self, input_dict):
        assert self.training

        vae_latent = self.get_latent(input_dict)
        input_dict["vae_latent"] = vae_latent

        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        prev_neglogp = self.neglogp(input_dict["actions"], mu, sigma, logstd)
        return {
            "mus": mu,
            "sigmas": sigma,
            "neglogp": prev_neglogp,
        }

    def eval_forward(self, input_dict):
        assert not self.training

        vae_latent = self.get_latent(input_dict)
        input_dict["vae_latent"] = vae_latent

        mu, logstd = self.get_mu_logstd(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        neglogp = self.neglogp(selected_action, mu, sigma, logstd)
        return {
            "mus": mu,
            "sigmas": sigma,
            "actions": selected_action,
            "neglogp": neglogp,
        }

    def get_latent(self, input_dict):
        latent = self.vae_encoder(input_dict)
        quantized_latent, indices = self.quantize_latent(latent)

        return quantized_latent

    def quantize_latent(self, latent):
        quantized_latent, indices = self.vae_quantizer(latent.view(latent.size(0), 1, -1))
        quantized_latent = quantized_latent.view(latent.size(0), -1)
        indices = indices.view(latent.size(0), 1)

        return quantized_latent, indices


class ActorFixedSigmaVQVAEWithPrior(ActorFixedSigmaVQVAE):
    def __init__(self, config, num_in, num_act):
        super().__init__(config, num_in, num_act)

        self.vae_dim = np.cumprod(self.config.vae_quantizer.levels)[-1]

        self.vae_prior: MLP_WithNorm = instantiate(self.config.vae_prior, num_in=num_in)

        # self.vae_encoder.requires_grad_(False)
        # self.vae_quantizer.requires_grad_(False)
        # self.mu_model.requires_grad_(False)

    def reparameterization(self, logits, vae_noise, temperature=1.0, eps=1e-20):
        sample_gumbel = -torch.log(-torch.log(vae_noise + eps) + eps)
        y = logits + sample_gumbel
        return torch.nn.functional.softmax(y / temperature, dim=-1)

    def get_latent(self, input_dict):
        logits = self.vae_prior(input_dict)
        _, indices = self.reparameterization(logits, input_dict["vae_noise"]).max(dim=-1)
        quantized_latent = self.vae_quantizer.indices_to_codes(indices.view(-1, 1)).view(logits.shape[0], -1)

        return quantized_latent

    def prior_prediction(self, input_dict):
        return self.vae_prior(input_dict)

    def posterior_prediction(self, input_dict):
        return self.vae_encoder(input_dict)
