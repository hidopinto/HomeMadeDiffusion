"""Architecture shape tests for DiT and its sub-components."""

import numpy as np
import torch
from timm.models.vision_transformer import Attention

from models.conditioning import TimestepEmbedder, SinCosPosEmbed2D, SinCosPosEmbed3D
from models.layers import PatchEmbed, FinalLayer, AdaLNZeroStrategy, AdaLNTextProjector

# Re-use constants from conftest without importing the module directly
_B = 2
_HIDDEN = 128
_PATCH = [2, 2]
_IN_CH = 4
_INPUT = 16   # latent grid H = W = 16


# ---------------------------------------------------------------------------
# DiT forward pass
# ---------------------------------------------------------------------------

def test_forward_no_variance(dit_model, batch_latents, batch_text_embeds, device):
    dit_model.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model(batch_latents, t, batch_text_embeds)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_forward_with_variance(dit_model_with_variance, batch_latents, batch_text_embeds, device):
    dit_model_with_variance.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model_with_variance(batch_latents, t, batch_text_embeds)
    # learn_variance=True doubles channel count
    assert out.shape == (_B, 2 * _IN_CH, _INPUT, _INPUT)


def test_forward_no_cond(dit_model, batch_latents, device):
    """y=None should not crash and produce the right shape."""
    dit_model.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model(batch_latents, t, y=None)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

def test_patch_embed_shape(device):
    ph, pw = _PATCH
    embed = PatchEmbed(patch_size=_PATCH, in_chans=_IN_CH, embed_dim=_HIDDEN).to(device)
    x = torch.randn(_B, _IN_CH, _INPUT, _INPUT, device=device)
    out = embed(x)
    expected_n = (_INPUT // ph) * (_INPUT // pw)   # 8 * 8 = 64
    assert out.shape == (_B, expected_n, _HIDDEN)


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

def test_final_layer_shape(device):
    """FinalLayer un-projects from hidden_size to patch pixels (before spatial fold)."""
    import numpy as np
    ph, pw = _PATCH
    layer = FinalLayer(
        hidden_size=_HIDDEN, patch_size=_PATCH,
        out_channels=_IN_CH, learn_variance=False
    ).to(device)
    N = (_INPUT // ph) * (_INPUT // pw)   # 64 tokens
    x = torch.randn(_B, N, _HIDDEN, device=device)
    cond = torch.randn(_B, _HIDDEN, device=device)
    out = layer(x, cond)
    # v=1, out_channels=4, patch area=4 → each token predicts 16 values
    assert out.shape == (_B, N, 1 * _IN_CH * int(np.prod(_PATCH)))


# ---------------------------------------------------------------------------
# AdaLNZeroStrategy
# ---------------------------------------------------------------------------

def test_adaLN_zero_modulation(device):
    N = 64
    strategy = AdaLNZeroStrategy(hidden_size=_HIDDEN, c=_HIDDEN).to(device)
    x = torch.randn(_B, N, _HIDDEN, device=device)
    cond = torch.randn(_B, _HIDDEN, device=device)
    res_msa, gate_msa, res_mlp, gate_mlp = strategy(x, cond)
    assert res_msa.shape == (_B, N, _HIDDEN)
    assert gate_msa.shape == (_B, _HIDDEN)
    assert res_mlp.shape == (_B, N, _HIDDEN)
    assert gate_mlp.shape == (_B, _HIDDEN)


# ---------------------------------------------------------------------------
# SinCosPosEmbed2D
# ---------------------------------------------------------------------------

def test_sincos_2d_shape(device):
    grid_size = _INPUT // _PATCH[0]   # 8
    embedder = SinCosPosEmbed2D(_HIDDEN, grid_size=grid_size).to(device)
    N = grid_size * grid_size  # 64
    x = torch.randn(_B, N, _HIDDEN, device=device)
    out = embedder(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# SinCosPosEmbed3D
# ---------------------------------------------------------------------------

def test_sincos_3d_shape(device):
    """3D embedder output should match the 3D token sequence shape."""
    grid_size = 4       # 4×4 spatial grid → 16 spatial patches
    max_frames = 8
    num_frames = 4      # actual temporal patches in this batch
    N = num_frames * grid_size * grid_size   # 64 total tokens
    embedder = SinCosPosEmbed3D(_HIDDEN, grid_size=grid_size, max_frames=max_frames).to(device)
    x = torch.randn(_B, N, _HIDDEN, device=device)
    out = embedder(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------

def test_timestep_embedder_shape(device):
    embedder = TimestepEmbedder(
        hidden_size=_HIDDEN, frequency_embedding_size=64, max_period=10000
    ).to(device)
    t = torch.randint(0, 1000, (_B,), device=device)
    out = embedder(t)
    assert out.shape == (_B, _HIDDEN)
