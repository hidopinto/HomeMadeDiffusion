from __future__ import annotations

import torch
from torch import Tensor

from diffusion_engine import DDPM


class DDPMSampler:
    """Classic ancestral sampling (Ho et al., 2020). 1000 stochastic steps."""

    def __init__(self, schedule: DDPM) -> None:
        self.schedule = schedule

    def _step(self, model_fn: callable, x_t: Tensor, t_idx: int,
              model_kwargs: dict | None = None) -> Tensor:
        t = torch.full((x_t.shape[0],), t_idx, device=x_t.device, dtype=torch.long)
        eps = model_fn(x_t, t, **(model_kwargs or {}))

        x_0_pred = (x_t - self.schedule.sqrt_one_minus_alphas_cumprod[t_idx] * eps) \
                   / self.schedule.sqrt_alphas_cumprod[t_idx]
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        mean = (self.schedule.posterior_mean_coef1[t_idx] * x_0_pred
                + self.schedule.posterior_mean_coef2[t_idx] * x_t)
        log_var = self.schedule.posterior_log_variance_clipped[t_idx]
        noise = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample_loop(self, model_fn: callable, shape: tuple, device: torch.device,
                    model_kwargs: dict | None = None) -> Tensor:
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.schedule.num_timesteps)):
            x = self._step(model_fn, x, t_idx, model_kwargs=model_kwargs)
        return x


class DDIMSampler:
    """Non-Markovian deterministic sampler (Song et al., 2021, arXiv 2010.02502).
    Uses the same trained model as DDPM; no retraining needed.
    eta=0 → fully deterministic; eta=1 → recovers DDPM stochastic sampling.
    """

    def __init__(self, schedule: DDPM) -> None:
        self.schedule = schedule

    def _step(self, model_fn: callable, x_t: Tensor, t_idx: int, t_prev_idx: int,
              eta: float = 0.0, model_kwargs: dict | None = None) -> Tensor:
        t = torch.full((x_t.shape[0],), t_idx, device=x_t.device, dtype=torch.long)
        eps = model_fn(x_t, t, **(model_kwargs or {}))

        alpha_t = self.schedule.alphas_cumprod[t_idx]
        alpha_prev = (self.schedule.alphas_cumprod[t_prev_idx] if t_prev_idx >= 0
                      else torch.ones(1, device=x_t.device))

        x_0_pred = (x_t - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        noise = torch.randn_like(x_t) if t_prev_idx >= 0 else torch.zeros_like(x_t)
        return alpha_prev.sqrt() * x_0_pred + (1 - alpha_prev - sigma ** 2).sqrt() * eps + sigma * noise

    @torch.no_grad()
    def sample_loop(self, model_fn: callable, shape: tuple, device: torch.device,
                    num_steps: int = 50, eta: float = 0.0,
                    model_kwargs: dict | None = None) -> Tensor:
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.schedule.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        for i, t_idx in enumerate(timesteps):
            t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
            x = self._step(model_fn, x, int(t_idx), t_prev, eta=eta, model_kwargs=model_kwargs)
        return x
