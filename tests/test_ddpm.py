"""Noise schedule and loss tests for DDPM.

DDPM buffers live on CPU by default; keep all tensors on CPU here so there is
no device mismatch.  GPU-path is exercised indirectly via the overfit tests.
"""

import torch


_B = 2
_C = 4
_H = _W = 16
_CPU = "cpu"


def _latents() -> torch.Tensor:
    return torch.randn(_B, _C, _H, _W)


# ---------------------------------------------------------------------------
# q_sample
# ---------------------------------------------------------------------------

def test_q_sample_shape(ddpm):
    x0 = _latents()
    t = torch.randint(0, ddpm.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    assert xt.shape == x0.shape


def test_q_sample_identity_at_t0(ddpm):
    """At t=0 the forward diffusion adds almost no noise."""
    x0 = _latents()
    t = torch.zeros(_B, dtype=torch.long)
    noise = torch.zeros_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    # sqrt_alphas_cumprod[0] ≈ 1; output should be very close to x0
    assert torch.allclose(xt, x0, atol=0.05)


# ---------------------------------------------------------------------------
# sample_timesteps
# ---------------------------------------------------------------------------

def test_sample_timesteps_range(ddpm):
    t = ddpm.sample_timesteps(_B, _CPU)
    assert t.shape == (_B,)
    assert t.min() >= 0
    assert t.max() < ddpm.num_timesteps


# ---------------------------------------------------------------------------
# loss (no variance)
# ---------------------------------------------------------------------------

def test_loss_no_variance(ddpm):
    x0 = _latents()
    t = torch.randint(0, ddpm.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    model_output = torch.randn_like(noise)
    loss = ddpm.loss(model=None, x_0=x0, x_t=xt, t=t, model_output=model_output, noise=noise)
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# loss (with variance)
# ---------------------------------------------------------------------------

def test_loss_with_variance(ddpm_with_variance):
    x0 = _latents()
    t = torch.randint(0, ddpm_with_variance.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm_with_variance.q_sample(x0, t, noise)
    # model_output has 2× channels: noise + var_v
    model_output = torch.randn(_B, 2 * _C, _H, _W)
    loss = ddpm_with_variance.loss(
        model=None, x_0=x0, x_t=xt, t=t, model_output=model_output, noise=noise
    )
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# calc_vlb_loss
# ---------------------------------------------------------------------------

def test_vlb_loss_shape(ddpm_with_variance):
    x0 = _latents()
    t = torch.randint(0, ddpm_with_variance.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm_with_variance.q_sample(x0, t, noise)
    eps_pred = torch.randn_like(x0)
    var_v = torch.sigmoid(torch.randn_like(x0))   # interpolation in [0, 1]
    vlb = ddpm_with_variance.calc_vlb_loss(x0, xt, t, eps_pred, var_v)
    assert vlb.shape == ()
