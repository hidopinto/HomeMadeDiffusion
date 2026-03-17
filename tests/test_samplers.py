"""DDPM and DDIM sampler step and loop shape tests."""

import torch

from diffusion_engine import DDPM
from samplers import DDPMSampler, DDIMSampler

_B = 2
_C = 4
_H = _W = 8    # small spatial dim keeps loops fast


def _shape():
    return (_B, _C, _H, _W)


def _model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Minimal noise predictor: returns random noise matching input shape."""
    return torch.randn_like(x)


def _fast_ddpm() -> DDPM:
    """10-step schedule — sample_loop iterates only 10 denoising steps."""
    return DDPM(num_timesteps=10, learn_variance=False)


# ---------------------------------------------------------------------------
# DDPMSampler
# ---------------------------------------------------------------------------

def test_ddpm_step_shape(device):
    schedule = _fast_ddpm()
    sampler = DDPMSampler(schedule)
    x_t = torch.randn(_shape(), device=device)
    out = sampler._step(_model_fn, x_t, t_idx=5)
    assert out.shape == x_t.shape


def test_ddpm_loop_output_shape(device):
    schedule = _fast_ddpm()
    sampler = DDPMSampler(schedule)
    out = sampler.sample_loop(_model_fn, _shape(), device)
    assert out.shape == _shape()


# ---------------------------------------------------------------------------
# DDIMSampler
# ---------------------------------------------------------------------------

def test_ddim_step_shape(device):
    schedule = _fast_ddpm()
    sampler = DDIMSampler(schedule)
    x_t = torch.randn(_shape(), device=device)
    out = sampler._step(_model_fn, x_t, t_idx=5, t_prev_idx=4, eta=0.0)
    assert out.shape == x_t.shape


def test_ddim_loop_output_shape(device):
    schedule = _fast_ddpm()
    sampler = DDIMSampler(schedule)
    out = sampler.sample_loop(_model_fn, _shape(), device, num_steps=3, eta=0.0)
    assert out.shape == _shape()


def test_ddim_deterministic(device):
    """eta=0 → no stochastic noise term → identical outputs for the same seed."""
    schedule = _fast_ddpm()
    sampler = DDIMSampler(schedule)

    torch.manual_seed(42)
    out1 = sampler.sample_loop(_model_fn, _shape(), device, num_steps=3, eta=0.0)

    torch.manual_seed(42)
    out2 = sampler.sample_loop(_model_fn, _shape(), device, num_steps=3, eta=0.0)

    assert torch.allclose(out1, out2)
