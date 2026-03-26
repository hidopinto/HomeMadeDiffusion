from __future__ import annotations

from typing import Callable

import torch
from box import Box
from torch import Tensor

from diffusion.methods.ddpm import DDPM
from diffusion.samplers.base import IntermediateCollector

__all__ = ["DDPMSampler"]


class DDPMSampler:
    """Classic ancestral sampling (Ho et al., 2020). 1000 stochastic steps."""

    def __init__(self, schedule: DDPM) -> None:
        self.schedule = schedule

    @classmethod
    def from_config(cls, config: Box, schedule: DDPM) -> "DDPMSampler":
        return cls(schedule)

    def _step(
        self,
        model_fn: Callable,
        x_t: Tensor,
        t_idx: int,
        model_kwargs: dict | None = None,
    ) -> Tensor:
        t = torch.full((x_t.shape[0],), t_idx, device=x_t.device, dtype=torch.long)
        eps = model_fn(x_t, t, **(model_kwargs or {}))

        x_0_pred = (x_t - self.schedule.sqrt_one_minus_alphas_cumprod[t_idx] * eps) \
                   / self.schedule.sqrt_alphas_cumprod[t_idx]

        mean = (self.schedule.posterior_mean_coef1[t_idx] * x_0_pred
                + self.schedule.posterior_mean_coef2[t_idx] * x_t)
        log_var = self.schedule.posterior_log_variance_clipped[t_idx]
        noise = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
        return mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def sample_loop(
        self,
        model_fn: Callable,
        shape: tuple,
        device: torch.device,
        model_kwargs: dict | None = None,
        collector: IntermediateCollector | None = None,
    ) -> Tensor:
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.schedule.num_timesteps)):
            x = self._step(model_fn, x, t_idx, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(t_idx, self.schedule.num_timesteps, x)
        return x
