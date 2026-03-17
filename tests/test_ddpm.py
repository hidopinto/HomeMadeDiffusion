"""Noise schedule and loss tests for DDPM."""

import torch


_B = 2
_C = 4
_H = _W = 16


def _latents(device: str) -> torch.Tensor:
    return torch.randn(_B, _C, _H, _W, device=device)


# ---------------------------------------------------------------------------
# q_sample
# ---------------------------------------------------------------------------

def test_q_sample_shape(ddpm, device):
    x0 = _latents(device)
    t = torch.randint(0, ddpm.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    assert xt.shape == x0.shape


def test_q_sample_identity_at_t0(ddpm, device):
    """At t=0 the forward diffusion adds almost no noise."""
    x0 = _latents(device)
    t = torch.zeros(_B, dtype=torch.long)
    noise = torch.zeros_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    # sqrt_alphas_cumprod[0] ≈ 1; output should be very close to x0
    assert torch.allclose(xt, x0, atol=0.05)


# ---------------------------------------------------------------------------
# sample_timesteps
# ---------------------------------------------------------------------------

def test_sample_timesteps_range(ddpm, device):
    t = ddpm.sample_timesteps(_B, device)
    assert t.shape == (_B,)
    assert t.min() >= 0
    assert t.max() < ddpm.num_timesteps


# ---------------------------------------------------------------------------
# loss (no variance)
# ---------------------------------------------------------------------------

def test_loss_no_variance(ddpm, device):
    x0 = _latents(device)
    t = torch.randint(0, ddpm.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    # model_output has the same shape as noise (no variance doubling)
    model_output = torch.randn_like(noise)
    loss = ddpm.loss(model=None, x_0=x0, x_t=xt, t=t, model_output=model_output, noise=noise)
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# loss (with variance)
# ---------------------------------------------------------------------------

def test_loss_with_variance(ddpm_with_variance, device):
    x0 = _latents(device)
    t = torch.randint(0, ddpm_with_variance.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm_with_variance.q_sample(x0, t, noise)
    # model_output has 2× channels: noise + var_v
    model_output = torch.randn(_B, 2 * _C, _H, _W, device=device)
    loss = ddpm_with_variance.loss(
        model=None, x_0=x0, x_t=xt, t=t, model_output=model_output, noise=noise
    )
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# calc_vlb_loss
# ---------------------------------------------------------------------------

def test_vlb_loss_shape(ddpm_with_variance, device):
    x0 = _latents(device)
    t = torch.randint(0, ddpm_with_variance.num_timesteps, (_B,))
    noise = torch.randn_like(x0)
    xt = ddpm_with_variance.q_sample(x0, t, noise)
    eps_pred = torch.randn_like(x0)
    var_v = torch.sigmoid(torch.randn_like(x0))   # interpolation in [0, 1]
    vlb = ddpm_with_variance.calc_vlb_loss(x0, xt, t, eps_pred, var_v)
    assert vlb.shape == ()
