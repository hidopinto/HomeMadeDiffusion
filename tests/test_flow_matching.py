"""Flow Matching noise schedule, interpolation, and loss tests.

All tensors on CPU. GPU path is exercised indirectly via test_overfit.py
when configured with flow_matching method.
"""

from __future__ import annotations

import numpy as np
import torch
from unittest.mock import MagicMock

from diffusion_engine import FlowMatching, _ot_reorder_noise

_B = 2
_C = 4
_H = _W = 16
_CPU = "cpu"
_T = 100  # small num_timesteps for fast tests


def _fm() -> FlowMatching:
    return FlowMatching(num_timesteps=_T, use_minibatch_ot=False)


def _latents() -> torch.Tensor:
    return torch.randn(_B, _C, _H, _W)


# ---------------------------------------------------------------------------
# sample_timesteps
# ---------------------------------------------------------------------------

def test_sample_timesteps_range():
    fm = _fm()
    t = fm.sample_timesteps(_B, _CPU)
    assert t.shape == (_B,)
    assert t.min() >= 0
    assert t.max() < fm.num_timesteps


def test_sample_timesteps_dtype():
    fm = _fm()
    t = fm.sample_timesteps(_B, _CPU)
    assert t.dtype == torch.long


# ---------------------------------------------------------------------------
# q_sample (OT conditional path)
# ---------------------------------------------------------------------------

def test_q_sample_shape():
    fm = _fm()
    x_0 = _latents()
    t = fm.sample_timesteps(_B, _CPU)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    assert x_t.shape == x_0.shape


def test_q_sample_at_t_zero_is_noise():
    """At t=0, t_cont=0.0 → x_t = noise."""
    fm = _fm()
    x_0 = _latents()
    t = torch.zeros(_B, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    assert torch.allclose(x_t, noise, atol=1e-5)


def test_q_sample_at_t_max_is_x0():
    """At t=T-1, t_cont=1.0 → x_t = x_0."""
    fm = _fm()
    x_0 = _latents()
    t = torch.full((_B,), _T - 1, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    assert torch.allclose(x_t, x_0, atol=1e-5)


def test_q_sample_interpolation_midpoint():
    """At t = (T-1)/2, x_t should match the expected linear blend."""
    fm = _fm()
    x_0 = _latents()
    mid = (_T - 1) // 2
    t = torch.full((_B,), mid, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    t_cont_val = mid / (_T - 1)
    expected = (1.0 - t_cont_val) * noise + t_cont_val * x_0
    assert torch.allclose(x_t, expected, atol=1e-5)


def test_q_sample_shape_5d_video():
    """q_sample must handle 5D video tensors (B, C, F, H, W)."""
    fm = _fm()
    x_0 = torch.randn(_B, _C, 4, _H, _W)
    t = fm.sample_timesteps(_B, _CPU)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    assert x_t.shape == x_0.shape


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def test_loss_scalar():
    fm = _fm()
    x_0 = _latents()
    t = fm.sample_timesteps(_B, _CPU)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    model_output = torch.randn_like(x_0)
    loss = fm.loss(model=None, x_0=x_0, x_t=x_t, t=t, model_output=model_output, noise=noise)
    assert loss.shape == ()
    assert loss.item() > 0


def test_loss_zero_when_perfect():
    """MSE = 0 when model_output exactly equals the target velocity x_0 - noise."""
    fm = _fm()
    x_0 = _latents()
    t = fm.sample_timesteps(_B, _CPU)
    noise = torch.randn_like(x_0)
    x_t = fm.q_sample(x_0, t, noise)
    perfect_output = x_0 - noise
    loss = fm.loss(model=None, x_0=x_0, x_t=x_t, t=t, model_output=perfect_output, noise=noise)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_expected_out_channels():
    """FM always returns in_channels — no variance prediction head."""
    fm = _fm()
    assert fm.expected_out_channels(4) == 4
    assert fm.expected_out_channels(8) == 8


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------

def test_from_config():
    config = MagicMock()
    config.diffusion.method = "flow_matching"
    fm_cfg = MagicMock()
    fm_cfg.num_timesteps = 500
    fm_cfg.use_minibatch_ot = False
    config.diffusion.__getitem__.return_value = fm_cfg
    fm = FlowMatching.from_config(config)
    assert fm.num_timesteps == 500
    assert fm.use_minibatch_ot is False


# ---------------------------------------------------------------------------
# minibatch OT reordering
# ---------------------------------------------------------------------------

def test_ot_reorder_noise_shape():
    """Reordered noise must have the same shape as the input noise."""
    x_0 = _latents()
    noise = torch.randn_like(x_0)
    reordered = _ot_reorder_noise(x_0, noise)
    assert reordered.shape == noise.shape


def test_ot_reorder_reduces_cost():
    """Total L2 transport cost after OT reordering must be ≤ identity pairing cost."""
    torch.manual_seed(0)
    x_0 = _latents()
    noise = torch.randn_like(x_0)
    reordered = _ot_reorder_noise(x_0, noise)

    def total_cost(n: torch.Tensor) -> float:
        diff = (x_0 - n).reshape(_B, -1)
        return float((diff ** 2).sum().item())

    assert total_cost(reordered) <= total_cost(noise) + 1e-6
