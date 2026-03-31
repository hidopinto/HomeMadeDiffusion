from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from box import Box
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
import torch.nn.functional as F

__all__ = ["FlowMatching", "_ot_reorder_noise"]

logger = logging.getLogger(__name__)


def _ot_reorder_noise(x_0: Tensor, noise: Tensor) -> Tensor:
    """Reorder noise to minimise intra-batch L2 transport cost.

    Cost matrix is computed on GPU via torch.cdist; only the small (B×B)
    result is transferred to CPU for the Hungarian solver.
    Complexity: O(B²·D) on GPU + O(B³) solver on the (B×B) CPU matrix.
    """
    x_flat = rearrange(x_0.detach().float(), "b ... -> b (...)")
    n_flat = rearrange(noise.detach().float(), "b ... -> b (...)")
    cost = torch.cdist(x_flat, n_flat, compute_mode="use_mm_for_euclid_dist").pow(2).cpu().numpy()
    _, col_ind = linear_sum_assignment(cost)
    return noise[col_ind]


class FlowMatching(nn.Module):
    """Optimal-Transport Flow Matching (Lipman et al., 2022 / Albergo & Vanden-Eijnden 2022).

    Uses straight-line OT conditional paths:
        x_t = (1 - t_cont) * noise + t_cont * x_0,  t_cont ∈ [0, 1]
    Target velocity field:
        u_t = x_0 - noise  (constant along the path)
    Loss:
        MSE(model_output, u_t)

    When ``use_minibatch_ot=True``, noise samples are reordered within each
    mini-batch via the Hungarian algorithm to minimise the total L2 transport
    cost between x_0 and noise pairs before computing the conditional path.
    """

    def __init__(self, num_timesteps: int = 1000, use_minibatch_ot: bool = False) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.use_minibatch_ot = use_minibatch_ot
        logger.debug(
            "FlowMatching: T=%d, use_minibatch_ot=%s", num_timesteps, use_minibatch_ot
        )

    @classmethod
    def from_config(cls, config: Box) -> "FlowMatching":
        cfg = config.diffusion.methods[config.diffusion.method]
        return cls(num_timesteps=cfg.num_timesteps, use_minibatch_ot=cfg.use_minibatch_ot)

    def update_settings(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def expected_out_channels(self, in_channels: int) -> int:
        """FM predicts the velocity field — same channel count as input, no variance head."""
        return in_channels

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def prepare_noise(self, x_0: Tensor, noise: Tensor) -> Tensor:
        if self.use_minibatch_ot:
            return _ot_reorder_noise(x_0, noise)
        return noise

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """OT conditional path: x_t = (1 - t_cont) * noise + t_cont * x_0."""
        t_cont = t.float() / (self.num_timesteps - 1)
        view_shape = (-1,) + (1,) * (x_0.ndim - 1)
        t_bc = t_cont.view(view_shape)
        return (1.0 - t_bc) * noise + t_bc * x_0

    def loss(
        self,
        model: nn.Module,
        x_0: Tensor,
        x_t: Tensor,
        t: Tensor,
        model_output: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """MSE against the constant OT velocity target u_t = x_0 - noise."""
        target = x_0 - noise
        return F.mse_loss(model_output, target)

    def __repr__(self) -> str:
        return (
            f"FlowMatching(num_timesteps={self.num_timesteps}, "
            f"use_minibatch_ot={self.use_minibatch_ot})"
        )
