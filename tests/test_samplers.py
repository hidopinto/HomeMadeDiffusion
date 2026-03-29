"""DDPM and DDIM sampler step and loop shape tests."""

import torch

from diffusion.methods.ddpm import DDPM
from diffusion.methods.flow_matching import FlowMatching
from diffusion.samplers import DDPMSampler, DDIMSampler, FlowMatchingSampler
from diffusion.samplers.base import SamplerProtocol

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
    sampler = DDIMSampler(schedule, num_steps=3, eta=0.0)
    out = sampler.sample_loop(_model_fn, _shape(), device)
    assert out.shape == _shape()


def test_ddim_deterministic(device):
    """eta=0 → no stochastic noise term → identical outputs for the same seed."""
    schedule = _fast_ddpm()
    sampler = DDIMSampler(schedule, num_steps=3, eta=0.0)

    torch.manual_seed(42)
    out1 = sampler.sample_loop(_model_fn, _shape(), device)

    torch.manual_seed(42)
    out2 = sampler.sample_loop(_model_fn, _shape(), device)

    assert torch.allclose(out1, out2)


def test_ddim_update_settings(device):
    """update_settings must mutate known attributes and silently ignore unknown ones."""
    schedule = _fast_ddpm()
    sampler = DDIMSampler(schedule, num_steps=10, eta=0.0)
    sampler.update_settings(num_steps=3, eta=0.5, unknown_key="ignored")
    assert sampler.num_steps == 3
    assert sampler.eta == 0.5
    assert not hasattr(sampler, "unknown_key")


# ---------------------------------------------------------------------------
# FlowMatchingSampler
# ---------------------------------------------------------------------------

def _fast_fm() -> FlowMatching:
    """10-step FM schedule — sample_loop iterates 10 Euler steps."""
    return FlowMatching(num_timesteps=10, use_minibatch_ot=False)


def test_fm_step_shape(device):
    """Single Euler step must preserve spatial shape."""
    fm = _fast_fm()
    sampler = FlowMatchingSampler(fm)
    x = torch.randn(_shape(), device=device)
    out = sampler._step(_model_fn, x, t_idx=3, dt=0.1)
    assert out.shape == x.shape


def test_fm_loop_output_shape(device):
    """Full Euler loop must produce tensor with correct shape."""
    fm = _fast_fm()
    sampler = FlowMatchingSampler(fm, num_steps=5)
    out = sampler.sample_loop(_model_fn, _shape(), device)
    assert out.shape == _shape()


def test_fm_loop_deterministic(device):
    """FM ODE has no stochastic terms — identical outputs for the same seed."""
    fm = _fast_fm()
    sampler = FlowMatchingSampler(fm, num_steps=5)

    torch.manual_seed(42)
    out1 = sampler.sample_loop(_model_fn, _shape(), device)

    torch.manual_seed(42)
    out2 = sampler.sample_loop(_model_fn, _shape(), device)

    assert torch.allclose(out1, out2)


def test_fm_loop_nonzero_output(device):
    """sample_loop output must not be all-zeros (basic sanity check)."""
    torch.manual_seed(0)
    fm = _fast_fm()
    sampler = FlowMatchingSampler(fm, num_steps=3)
    out = sampler.sample_loop(_model_fn, _shape(), device)
    assert not torch.allclose(out, torch.zeros(_shape(), device=device))


def test_fm_update_settings_ignores_eta(device):
    """FlowMatchingSampler must silently drop eta — FM is always deterministic."""
    fm = _fast_fm()
    sampler = FlowMatchingSampler(fm, num_steps=5)
    sampler.update_settings(num_steps=3, eta=0.9)  # eta should be ignored
    assert sampler.num_steps == 3
    assert not hasattr(sampler, "eta")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_all_samplers_satisfy_sampler_protocol():
    """All three samplers must satisfy SamplerProtocol at runtime — catches interface drift."""
    ddpm = DDPM(num_timesteps=10)
    fm = FlowMatching(num_timesteps=10)
    assert isinstance(DDPMSampler(ddpm), SamplerProtocol)
    assert isinstance(DDIMSampler(ddpm), SamplerProtocol)
    assert isinstance(FlowMatchingSampler(fm), SamplerProtocol)
