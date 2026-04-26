from __future__ import annotations

from typing import Any, Callable

import torch
from box import Box
from torch import Tensor

from diffusion.methods.ddpm import DDPM
from diffusion.samplers.base import IntermediateCollector, ProgressFn

__all__ = ["DDIMSampler"]


class DDIMSampler:
    """Non-Markovian deterministic sampler (Song et al., 2021, arXiv 2010.02502).

    Uses the same trained model as DDPM; no retraining needed.
    eta=0 → fully deterministic; eta=1 → recovers DDPM stochastic sampling.
    """

    def __init__(self, schedule: DDPM, num_steps: int = 50, eta: float = 0.0) -> None:
        self.schedule = schedule
        self.num_steps = num_steps
        self.eta = eta

    @classmethod
    def from_config(cls, config: Box, schedule: DDPM) -> "DDIMSampler":
        cfg = config.diffusion.samplers.ddim
        return cls(schedule, num_steps=cfg.num_steps, eta=cfg.eta)

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _step(
        self,
        model_fn: Callable,
        x_t: Tensor,
        t_idx: int,
        t_prev_idx: int,
        eta: float = 0.0,
        model_kwargs: dict | None = None,
    ) -> Tensor:
        t = torch.full((x_t.shape[0],), t_idx, device=x_t.device, dtype=torch.long)
        eps = model_fn(x_t, t, **(model_kwargs or {}))

        alpha_t = self.schedule.alphas_cumprod[t_idx]
        alpha_prev = (self.schedule.alphas_cumprod[t_prev_idx] if t_prev_idx >= 0
                      else torch.ones(1, device=x_t.device))

        x_0_pred = (x_t - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()

        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        noise = torch.randn_like(x_t) if t_prev_idx >= 0 else torch.zeros_like(x_t)
        return alpha_prev.sqrt() * x_0_pred + (1 - alpha_prev - sigma ** 2).sqrt() * eps + sigma * noise

    @torch.no_grad()
    def sample_loop(
        self,
        model_fn: Callable,
        shape: tuple,
        device: torch.device,
        model_kwargs: dict | None = None,
        collector: IntermediateCollector | None = None,
        progress_fn: ProgressFn | None = None,
    ) -> Tensor:
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.schedule.num_timesteps - 1, 0, self.num_steps, dtype=torch.long).tolist()
        for i, t_idx in enumerate(timesteps):
            t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
            x = self._step(model_fn, x, int(t_idx), t_prev, eta=self.eta, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(i, self.num_steps, x)
            if progress_fn is not None:
                progress_fn(i, self.num_steps)
        return x
