from __future__ import annotations

from typing import Callable

import torch
from box import Box
from torch import Tensor

from diffusion.methods.flow_matching import FlowMatching
from diffusion.samplers.base import IntermediateCollector

__all__ = ["FlowMatchingSampler"]


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
