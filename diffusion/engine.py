from __future__ import annotations

from typing import Callable

import torch
from torch import nn, Tensor

from diffusion.methods.base import DiffusionMethod
from diffusion.samplers.base import SamplerProtocol
from models.condition_manager import ConditionOutput

__all__ = ["DiffusionEngine"]


class DiffusionEngine(nn.Module):
    """Stateful wrapper that pairs a DiffusionMethod with a Sampler.

    ``compute_loss`` delegates the full training step (sample t, corrupt x_0,
    run the model, compute method-specific loss) to the method. ``sample``
    delegates the full reverse-process loop to the sampler.

    The ``scheduler`` kwarg is silently popped from ``sample`` kwargs because
    legacy callers may pass it; the actual sampler is chosen at build time and
    cannot be swapped at inference time.
    """

    def __init__(self, method: DiffusionMethod, sampler: SamplerProtocol) -> None:
        super().__init__()
        self.method = method
        self.sampler = sampler

    def compute_loss(self, model: nn.Module, x_0: Tensor, cond: ConditionOutput) -> Tensor:
        t = self.method.sample_timesteps(x_0.shape[0], x_0.device)
        noise = self.method.prepare_noise(x_0, torch.randn_like(x_0))
        x_t = self.method.q_sample(x_0, t, noise)
        model_output = model(x_t, t, conditions=cond)
        return self.method.loss(model, x_0, x_t, t, model_output, noise)

    def sample(self, model_fn: Callable, shape: tuple, device: torch.device, **kwargs) -> Tensor:
        kwargs.pop("scheduler", None)  # sampler is chosen at build time
        model_kwargs = kwargs.pop("model_kwargs", None)
        collector = kwargs.pop("collector", None)
        progress_fn = kwargs.pop("progress_fn", None)
        # remaining kwargs are sampler settings — sampler decides what applies via hasattr
        self.sampler.update_settings(**kwargs)
        return self.sampler.sample_loop(
            model_fn, shape, device,
            model_kwargs=model_kwargs, collector=collector, progress_fn=progress_fn,
        )
