from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn, Tensor

__all__ = ["DiffusionMethod"]


@runtime_checkable
class DiffusionMethod(Protocol):
    """Structural protocol satisfied by DDPM and FlowMatching.

    Any object that implements these five methods is a valid diffusion method
    and can be passed to DiffusionEngine — no inheritance required. This makes
    the duck-typed interface explicit and type-checkable, and means adding a
    third method (e.g. score matching) requires zero changes to DiffusionEngine.
    """

    num_timesteps: int

    def update_settings(self, **kwargs: Any) -> None: ...

    def expected_out_channels(self, in_channels: int) -> int: ...

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor: ...

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor: ...

    def loss(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        model_output: Tensor,
        noise: Tensor,
    ) -> Tensor: ...
