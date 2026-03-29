from __future__ import annotations

import logging
from typing import Any

import torch
from box import Box
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["DDPM"]

logger = logging.getLogger(__name__)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model noise schedule (Ho et al., 2020).

    Pre-computes and registers the closed-form forward-process coefficients as
    non-trainable buffers so they move with the model to any device.

    Forward process (q):
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)

    When ``learn_variance=True`` (Improved DDPM, Nichol & Dhariwal 2021) the
    DiT head outputs 2 * in_channels: the first half is the noise prediction ε̂
    and the second half is an unconstrained scalar v used to interpolate the
    log-variance between the lower bound (posterior variance) and upper bound
    (β_t). The combined loss is MSE(ε̂, ε) + 0.001 * VLB.

    Buffers (all float32, shape [num_timesteps]):
        sqrt_alphas_cumprod         — sqrt(ᾱ_t)
        sqrt_one_minus_alphas_cumprod — sqrt(1 - ᾱ_t)
        alphas_cumprod              — ᾱ_t  (needed by DDIM sampler)
        posterior_log_variance_clipped — log variance lower bound
        log_betas                   — log variance upper bound
        posterior_mean_coef1/2      — coefficients for the posterior mean
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        learn_variance: bool = True,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.variance = learn_variance

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        one_minus_alphas_cumprod = 1.0 - alphas_cumprod
        one_minus_alphas_cumprod_safe = torch.clamp(one_minus_alphas_cumprod, min=1e-12)

        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(one_minus_alphas_cumprod).float())

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / one_minus_alphas_cumprod_safe
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.cat([posterior_variance[1:2], posterior_variance[1:]])).float())
        self.register_buffer('log_betas', torch.log(torch.cat([posterior_variance[1:2], betas[1:]])).float())
        self.register_buffer('posterior_mean_coef1',
                             (betas * torch.sqrt(alphas_cumprod_prev) / one_minus_alphas_cumprod_safe).float())
        self.register_buffer('posterior_mean_coef2',
                             ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus_alphas_cumprod_safe).float())

        logger.debug(
            "DDPM schedule: T=%d, β ∈ [%.4f, %.4f], learn_variance=%s",
            num_timesteps, beta_start, beta_end, learn_variance,
        )

    @classmethod
    def from_config(cls, config: Box) -> "DDPM":
        cfg = config.diffusion.methods[config.diffusion.method]
        return cls(
            num_timesteps=cfg.num_timesteps,
            learn_variance=cfg.learn_variance,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def expected_out_channels(self, in_channels: int) -> int:
        """Declares how many output channels the model must produce for this method."""
        return 2 * in_channels if self.variance else in_channels

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Standard Forward Process: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise"""
        # view_shape broadcasts over any ndim (4D images or 5D video tensors)
        view_shape = (-1,) + (1,) * (x_0.ndim - 1)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(view_shape)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(view_shape)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def calc_vlb_loss(
        self,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        eps_pred: Tensor,
        var_v: Tensor,
    ) -> Tensor:
        # 1. Get True Distribution (q)
        true_mean = self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_0 + \
                    self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        true_log_var = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)

        # 2. Get Predicted Distribution (p)
        min_log_var = true_log_var
        max_log_var = self.log_betas[t].view(-1, 1, 1, 1)
        # Model predicts var_v which interpolates between min and max log variance
        model_log_var = var_v * max_log_var + (1 - var_v) * min_log_var

        # 3. Estimate x_0 to find predicted mean
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        inv_sqrt_alpha_bar = (1.0 / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1))
        pred_x_0 = (x_t - sqrt_one_minus_alpha_bar * eps_pred) * inv_sqrt_alpha_bar

        model_mean = self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * pred_x_0 + \
                     self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t

        # 4. KL Divergence
        kl = 0.5 * (-1.0 + true_log_var - model_log_var + torch.exp(model_log_var - true_log_var) +
                    (true_mean - model_mean) ** 2 * torch.exp(-true_log_var))
        return kl.flatten(1).mean(1).mean()

    def loss(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        model_output: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Combines MSE and VLB if learn_variance is enabled."""
        if not self.variance:
            return F.mse_loss(model_output, noise)

        eps_pred, var_v = torch.split(model_output, x_0.shape[1], dim=1)
        loss_mse = F.mse_loss(eps_pred, noise)

        # Improved DDPM: Stop gradients for noise prediction during VLB calculation
        loss_vlb = self.calc_vlb_loss(x_0, x_t, t, eps_pred.detach(), var_v)
        return loss_mse + 0.001 * loss_vlb

    def __repr__(self) -> str:
        return (
            f"DDPM(num_timesteps={self.num_timesteps}, "
            f"learn_variance={self.variance})"
        )
