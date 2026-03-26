from __future__ import annotations

import torch
from box import Box
from dataclasses import dataclass, field
from typing import Callable
from torch import Tensor

from diffusion_engine import DDPM, FlowMatching


@dataclass
class IntermediateCollector:
    capture_fn: Callable[[int, int, Tensor], bool]
    latents: list[Tensor] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)
    decoded_images: list[Tensor] = field(default_factory=list)

    def maybe_collect(self, step_idx: int, total_steps: int, x: Tensor) -> None:
        if self.capture_fn(step_idx, total_steps, x):
            self.latents.append(x.clone())
            self.step_indices.append(step_idx)


class DDPMSampler:
    """Classic ancestral sampling (Ho et al., 2020). 1000 stochastic steps."""

    def __init__(self, schedule: DDPM) -> None:
        self.schedule = schedule

    @classmethod
    def from_config(cls, config: Box, schedule: DDPM) -> "DDPMSampler":
        return cls(schedule)

    def _step(self, model_fn: callable, x_t: Tensor, t_idx: int,
              model_kwargs: dict | None = None) -> Tensor:
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
    def sample_loop(self, model_fn: callable, shape: tuple, device: torch.device,
                    model_kwargs: dict | None = None,
                    collector: IntermediateCollector | None = None) -> Tensor:
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.schedule.num_timesteps)):
            x = self._step(model_fn, x, t_idx, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(t_idx, self.schedule.num_timesteps, x)
        return x


class DDIMSampler:
    """Non-Markovian deterministic sampler (Song et al., 2021, arXiv 2010.02502).
    Uses the same trained model as DDPM; no retraining needed.
    eta=0 → fully deterministic; eta=1 → recovers DDPM stochastic sampling.
    """

    def __init__(self, schedule: DDPM) -> None:
        self.schedule = schedule

    @classmethod
    def from_config(cls, config: Box, schedule: DDPM) -> "DDIMSampler":
        return cls(schedule)

    def _step(self, model_fn: callable, x_t: Tensor, t_idx: int, t_prev_idx: int,
              eta: float = 0.0, model_kwargs: dict | None = None) -> Tensor:
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
    def sample_loop(self, model_fn: callable, shape: tuple, device: torch.device,
                    num_steps: int = 50, eta: float = 0.0,
                    model_kwargs: dict | None = None,
                    collector: IntermediateCollector | None = None) -> Tensor:
        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.schedule.num_timesteps - 1, 0, num_steps, dtype=torch.long).tolist()
        for i, t_idx in enumerate(timesteps):
            t_prev = int(timesteps[i + 1]) if i + 1 < len(timesteps) else -1
            x = self._step(model_fn, x, int(t_idx), t_prev, eta=eta, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(i, num_steps, x)
        return x


class FlowMatchingSampler:
    """Euler ODE sampler for Flow Matching (Lipman et al., 2022).

    Integrates the learned velocity field from t_cont=0 (noise) to t_cont=1 (data)
    using fixed-step Euler method. No stochastic terms — sampling is fully deterministic
    given the initial noise.
    """

    def __init__(self, schedule: FlowMatching) -> None:
        self.schedule = schedule

    @classmethod
    def from_config(cls, config: Box, schedule: FlowMatching) -> "FlowMatchingSampler":
        return cls(schedule)

    def _step(
        self,
        model_fn: Callable,
        x: Tensor,
        t_idx: int,
        dt: float,
        model_kwargs: dict | None = None,
    ) -> Tensor:
        t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
        velocity = model_fn(x, t, **(model_kwargs or {}))
        return x + dt * velocity

    @torch.no_grad()
    def sample_loop(
        self,
        model_fn: Callable,
        shape: tuple,
        device: torch.device,
        num_steps: int = 50,
        model_kwargs: dict | None = None,
        collector: IntermediateCollector | None = None,
    ) -> Tensor:
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        t_indices = torch.linspace(
            0, self.schedule.num_timesteps - 1, num_steps, dtype=torch.long
        ).tolist()
        for i, t_idx in enumerate(t_indices):
            x = self._step(model_fn, x, int(t_idx), dt, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(i, num_steps, x)
        return x
