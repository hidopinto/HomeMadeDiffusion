"""Architecture shape tests for DiT and its sub-components."""

import numpy as np
import torch
from timm.models.vision_transformer import Attention

from models.conditioning import TimestepEmbedder, SinCosPosEmbed2D, SinCosPosEmbed3D
from models.layers import PatchEmbed, FinalLayer, AdaLNZeroStrategy
from models.projectors import AdaLNTextProjector, CrossAttnTextProjector
from models.cross_attention import CrossAttention
from models.condition_manager import ConditionOutput, ConditionManager

# Re-use constants from conftest without importing the module directly
_B = 2
_HIDDEN = 128
_PATCH = [2, 2]
_IN_CH = 4
_INPUT = 16   # latent grid H = W = 16
_COND_DIM = 128


# ---------------------------------------------------------------------------
# DiT forward pass
# ---------------------------------------------------------------------------

def test_forward_no_variance(dit_model, batch_latents, condition_output, device):
    dit_model.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model(batch_latents, t, conditions=condition_output)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_forward_with_variance(dit_model_with_variance, batch_latents, condition_output, device):
    dit_model_with_variance.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model_with_variance(batch_latents, t, conditions=condition_output)
    assert out.shape == (_B, 2 * _IN_CH, _INPUT, _INPUT)


def test_forward_no_cond(dit_model, batch_latents, device):
    """conditions=None should not crash and produce the right shape."""
    dit_model.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model(batch_latents, t, conditions=None)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


# ---------------------------------------------------------------------------
# Cross-attention DiT tests
# ---------------------------------------------------------------------------

def test_cross_attn_forward_shape(dit_model_cross_attn, batch_latents, condition_output, device):
    """Cross-attn DiT output shape must match non-cross-attn version."""
    dit_model_cross_attn.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model_cross_attn(batch_latents, t, conditions=condition_output)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_cross_attn_no_conditions(dit_model_cross_attn, batch_latents, device):
    """conditions=None must not crash even with cross-attn blocks."""
    dit_model_cross_attn.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model_cross_attn(batch_latents, t, conditions=None)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_cross_attn_adaLN_only(dit_model_cross_attn, batch_latents, condition_output_no_seq, device):
    """ConditionOutput with adaLN but no sequences: cross-attn is skipped cleanly."""
    dit_model_cross_attn.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    out = dit_model_cross_attn(batch_latents, t, conditions=condition_output_no_seq)
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_cross_attn_mask_partial(dit_model_cross_attn, batch_latents, device):
    """Partial mask (10/77 real tokens) must not produce NaNs."""
    dit_model_cross_attn.eval()
    t = torch.randint(0, 100, (_B,), device=device)
    mask = torch.zeros(2, 77, dtype=torch.long, device=device)
    mask[:, :10] = 1  # only first 10 tokens are real
    cond = ConditionOutput(
        adaLN=torch.randn(2, _HIDDEN, device=device),
        sequences=[(torch.randn(2, 77, _HIDDEN, device=device), mask)],
    )
    out = dit_model_cross_attn(batch_latents, t, conditions=cond)
    assert not torch.isnan(out).any(), "NaN detected with partial mask"
    assert out.shape == (_B, _IN_CH, _INPUT, _INPUT)


def test_cross_attn_gradient_checkpointing(batch_latents, condition_output, device):
    """Gradients must flow through blocks when gradient_checkpointing=True."""
    from models.conditioning import SinCosPosEmbed2D
    grid_size = _INPUT // _PATCH[0]
    pos_emb = SinCosPosEmbed2D(_HIDDEN, grid_size=grid_size).to(device)
    from models.models import DiT
    model = DiT(
        is_video=False,
        input_size=_INPUT,
        patch_size=_PATCH,
        in_channels=_IN_CH,
        out_channels=_IN_CH,
        hidden_size=_HIDDEN,
        frequency_embedding_size=64,
        max_period=10000,
        depth=2,
        num_heads=4,
        pos_embedder=pos_emb,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        cross_attn_class=CrossAttention,
        gradient_checkpointing=True,
    ).to(device)
    model.train()
    t = torch.randint(0, 100, (_B,), device=device)
    out = model(batch_latents, t, conditions=condition_output)
    out.sum().backward()
    grad = next(p.grad for p in model.parameters() if p.grad is not None)
    assert grad is not None


# ---------------------------------------------------------------------------
# ConditionManager routing
# ---------------------------------------------------------------------------

def test_condition_manager_routing(batch_text_embeds, device):
    """Global projector → adaLN; sequence projector → sequences."""
    ada_proj = AdaLNTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    seq_proj = CrossAttnTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    manager = ConditionManager([("text", ada_proj), ("text", seq_proj)]).to(device)

    out = manager({"text": batch_text_embeds})
    assert out.adaLN is not None
    assert out.adaLN.shape == (_B, _HIDDEN)
    assert len(out.sequences) == 1
    ctx, mask = out.sequences[0]
    assert ctx.shape == (_B, 77, _HIDDEN)
    assert mask.shape == (_B, 77)


def test_condition_manager_global_only(batch_text_embeds, device):
    """Only global projectors → sequences is empty list."""
    ada_proj = AdaLNTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    manager = ConditionManager([("text", ada_proj)]).to(device)
    out = manager({"text": batch_text_embeds})
    assert out.adaLN is not None
    assert out.sequences == []


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
    ph, pw = _PATCH
    layer = FinalLayer(
        hidden_size=_HIDDEN, patch_size=_PATCH,
        out_channels=_IN_CH,
    ).to(device)
    N = (_INPUT // ph) * (_INPUT // pw)   # 64 tokens
    x = torch.randn(_B, N, _HIDDEN, device=device)
    cond = torch.randn(_B, _HIDDEN, device=device)
    out = layer(x, cond)
    assert out.shape == (_B, N, _IN_CH * int(np.prod(_PATCH)))


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
