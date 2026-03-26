from __future__ import annotations

import torch
from box import Box
from torch import nn, Tensor
import torch.nn.functional as F


class DDPM(nn.Module):
    def __init__(self, num_timesteps=1000, learn_variance=True, beta_start=0.0001, beta_end=0.02):
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

        # FIX: Precompute the upper bound for variance interpolation
        self.register_buffer('log_betas', torch.log(torch.cat([posterior_variance[1:2], betas[1:]])).float())

        self.register_buffer('posterior_mean_coef1',
                             (betas * torch.sqrt(alphas_cumprod_prev) / one_minus_alphas_cumprod_safe).float())
        self.register_buffer('posterior_mean_coef2',
                             ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus_alphas_cumprod_safe).float())

    @classmethod
    def from_config(cls, config: Box) -> "DDPM":
        cfg = config.diffusion[config.diffusion.method]
        return cls(
            num_timesteps=cfg.num_timesteps,
            learn_variance=cfg.learn_variance,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )

    def expected_out_channels(self, in_channels: int) -> int:
        """Declares how many output channels the model must produce for this method."""
        return 2 * in_channels if self.variance else in_channels

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def q_sample(self, x_0, t, noise):
        """Standard Forward Process: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise"""
        # view_shape broadcasts over any ndim (4D images or 5D video tensors)
        view_shape = (-1,) + (1,) * (x_0.ndim - 1)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(view_shape)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(view_shape)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def calc_vlb_loss(self, x_0, x_t, t, eps_pred, var_v):
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

    def loss(self, model, x_0, x_t, t, model_output, noise):
        """Combines MSE and VLB if learn_sigma is enabled."""
        if not self.variance:
            return F.mse_loss(model_output, noise)

        eps_pred, var_v = torch.split(model_output, x_0.shape[1], dim=1)
        loss_mse = F.mse_loss(eps_pred, noise)

        # Improved DDPM: Stop gradients for noise prediction during VLB calculation
        loss_vlb = self.calc_vlb_loss(x_0, x_t, t, eps_pred.detach(), var_v)
        return loss_mse + 0.001 * loss_vlb


class DiffusionEngine(nn.Module):
    def __init__(self, method: DDPM, sampler) -> None:
        super().__init__()
        self.method = method
        self.sampler = sampler

    def compute_loss(self, model: nn.Module, x_0: Tensor, cond: dict) -> Tensor:
        t = self.method.sample_timesteps(x_0.shape[0], x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.method.q_sample(x_0, t, noise)
        model_output = model(x_t, t, cond)
        return self.method.loss(model, x_0, x_t, t, model_output, noise)

    def sample(self, model_fn: callable, shape: tuple, device: torch.device, **kwargs) -> Tensor:
        kwargs.pop("scheduler", None)  # sampler is chosen at build time
        return self.sampler.sample_loop(model_fn, shape, device, **kwargs)
