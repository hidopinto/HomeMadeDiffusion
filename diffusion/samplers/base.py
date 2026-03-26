from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = ["IntermediateCollector", "SamplerProtocol"]


@dataclass
class IntermediateCollector:
    """Collects latent tensors from intermediate denoising steps.

    capture_fn is called at every step; if it returns True the current latent
    is saved to ``latents``. Decoded images are populated externally by
    ``LatentDiffusion.generate()`` after the loop completes.
    """

    capture_fn: Callable[[int, int, Tensor], bool]
    latents: list[Tensor] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)
    decoded_images: list[Tensor] = field(default_factory=list)

    def maybe_collect(self, step_idx: int, total_steps: int, x: Tensor) -> None:
        if self.capture_fn(step_idx, total_steps, x):
            self.latents.append(x.clone())
            self.step_indices.append(step_idx)


@runtime_checkable
class SamplerProtocol(Protocol):
    """Structural protocol satisfied by DDPMSampler, DDIMSampler, and FlowMatchingSampler.

    Used as the type for the ``sampler`` argument of DiffusionEngine to avoid
    a circular import (engine.py only imports this protocol, not the concrete classes).
    """

    def sample_loop(
        self,
        model_fn: Callable,
        shape: tuple,
        device: torch.device,
        **kwargs,
    ) -> Tensor: ...
