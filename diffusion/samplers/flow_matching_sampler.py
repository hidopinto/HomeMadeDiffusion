from __future__ import annotations

from typing import Any, Callable

import torch
from box import Box
from torch import Tensor

from diffusion.methods.flow_matching import FlowMatching
from diffusion.samplers.base import IntermediateCollector, ProgressFn

__all__ = ["FlowMatchingSampler"]


class FlowMatchingSampler:
    """Euler ODE sampler for Flow Matching (Lipman et al., 2022).

    Integrates the learned velocity field from t_cont=0 (noise) to t_cont=1 (data)
    using fixed-step Euler method. No stochastic terms — sampling is fully deterministic
    given the initial noise.
    """

    def __init__(self, schedule: FlowMatching, num_steps: int = 50) -> None:
        self.schedule = schedule
        self.num_steps = num_steps

    @classmethod
    def from_config(cls, config: Box, schedule: FlowMatching) -> "FlowMatchingSampler":
        cfg = config.diffusion.samplers.flow_matching
        return cls(schedule, num_steps=cfg.num_steps)

    def update_settings(self, **kwargs: Any) -> None:
        # eta is intentionally absent — FM is always deterministic; unknown keys are dropped
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
        model_kwargs: dict | None = None,
        collector: IntermediateCollector | None = None,
        progress_fn: ProgressFn | None = None,
    ) -> Tensor:
        x = torch.randn(shape, device=device)
        dt = 1.0 / self.num_steps
        t_indices = torch.linspace(
            0, self.schedule.num_timesteps - 1, self.num_steps, dtype=torch.long
        ).tolist()
        for i, t_idx in enumerate(t_indices):
            x = self._step(model_fn, x, int(t_idx), dt, model_kwargs=model_kwargs)
            if collector is not None:
                collector.maybe_collect(i, self.num_steps, x)
            if progress_fn is not None:
                progress_fn(i, self.num_steps)
        return x
