"""LatentDiffusion tests with mock VAE and CLIP."""

import pytest
import torch
from unittest.mock import MagicMock

from timm.models.vision_transformer import Attention

from models.conditioning import SinCosPosEmbed2D
from models.layers import AdaLNZeroStrategy, AdaLNTextProjector
from models.models import DiT, LatentDiffusion
from diffusion.methods.ddpm import DDPM
from diffusion.engine import DiffusionEngine
from diffusion.samplers.ddim_sampler import DDIMSampler

_B = 2
_HIDDEN = 128
_COND_DIM = 128
_IN_CH = 4
_INPUT = 16
_PATCH = [2, 2]


# ---------------------------------------------------------------------------
# Helpers to build fakes
# ---------------------------------------------------------------------------

def _make_fake_tokenizer(B: int = _B, seq_len: int = 77) -> MagicMock:
    enc_out = MagicMock()
    enc_out.input_ids = torch.zeros(B, seq_len, dtype=torch.long)
    enc_out.attention_mask = torch.ones(B, seq_len, dtype=torch.long)
    enc_out.to.return_value = enc_out   # .to(device) returns itself

    tokenizer = MagicMock()
    tokenizer.model_max_length = seq_len
    tokenizer.return_value = enc_out
    return tokenizer


def _make_fake_text_encoder(B: int = _B, seq_len: int = 77, hidden_dim: int = 768) -> MagicMock:
    encoder = MagicMock()
    encoder.return_value = (torch.randn(B, seq_len, hidden_dim),)
    encoder.eval.return_value = encoder
    encoder.requires_grad_.return_value = encoder
    return encoder


def _make_fake_vae() -> MagicMock:
    vae = MagicMock()
    vae.eval.return_value = vae
    vae.requires_grad_.return_value = vae
    return vae


def _make_tiny_dit(device: str) -> DiT:
    grid_size = _INPUT // _PATCH[0]
    pos_emb = SinCosPosEmbed2D(_HIDDEN, grid_size=grid_size).to(device)
    txt_proj = AdaLNTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    return DiT(
        is_video=False,
        input_size=_INPUT,
        patch_size=_PATCH,
        in_channels=_IN_CH,
        out_channels=_IN_CH,
        hidden_size=_HIDDEN,
        text_projector=txt_proj,
        frequency_embedding_size=64,
        max_period=10000,
        depth=2,
        num_heads=4,
        pos_embedder=pos_emb,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
    ).to(device)


def _make_latent_diffusion(device: str) -> LatentDiffusion:
    dit = _make_tiny_dit(device)
    ddpm = DDPM(num_timesteps=100, learn_variance=False)
    engine = DiffusionEngine(method=ddpm, sampler=DDIMSampler(ddpm))
    config = MagicMock()
    config.dit.in_channels = _IN_CH
    config.dit.vae_scale_factor = 0.18215
    config.training.cfg_dropout_prob = 0.0
    return LatentDiffusion(
        config=config,
        dit_model=dit,
        vae=_make_fake_vae(),
        text_encoder=_make_fake_text_encoder(),
        tokenizer=_make_fake_tokenizer(),
        engine=engine,
    ).to(device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_encode_text_shapes(device):
    model = _make_latent_diffusion(device)
    text_embeds = model.encode_text(["a cat", "a dog"], device=device)
    assert text_embeds["hidden_states"].shape == (_B, 77, 768)
    assert text_embeds["attention_mask"].shape == (_B, 77)


def test_forward_returns_scalar_loss(device):
    model = _make_latent_diffusion(device)
    latents = torch.randn(_B, _IN_CH, _INPUT, _INPUT, device=device)
    text_embeds = {
        "hidden_states": torch.randn(_B, 77, _COND_DIM, device=device),
        "attention_mask": torch.ones(_B, 77, dtype=torch.long, device=device),
    }
    loss = model(latents, text_embeds)
    assert loss.shape == ()


def test_loss_has_grad(device):
    model = _make_latent_diffusion(device)
    latents = torch.randn(_B, _IN_CH, _INPUT, _INPUT, device=device)
    text_embeds = {
        "hidden_states": torch.randn(_B, 77, _COND_DIM, device=device),
        "attention_mask": torch.ones(_B, 77, dtype=torch.long, device=device),
    }
    loss = model(latents, text_embeds)
    assert loss.requires_grad


# ---------------------------------------------------------------------------
# Generate helpers and tests
# ---------------------------------------------------------------------------

# _INPUT=16, _PATCH=[2,2] → h_lat=w_lat=16 → height=width=128
_H_IMG = _INPUT * 8  # = 128


def _make_latent_diffusion_for_generate(device: str) -> LatentDiffusion:
    """Same as _make_latent_diffusion but with text encoder dim matching DiT's cond_dim."""
    dit = _make_tiny_dit(device)
    ddpm = DDPM(num_timesteps=10, learn_variance=False)
    engine = DiffusionEngine(method=ddpm, sampler=DDIMSampler(ddpm))
    config = MagicMock()
    config.dit.in_channels = _IN_CH
    config.dit.vae_scale_factor = 0.18215
    config.training.cfg_dropout_prob = 0.0

    vae = _make_fake_vae()
    vae.dtype = torch.float32
    vae.decode.return_value.sample = torch.zeros(1, 3, _H_IMG, _H_IMG, device=device)

    # Text encoder must return tensors on the same device as its input
    text_encoder = MagicMock()
    text_encoder.side_effect = lambda ids: (torch.randn(ids.shape[0], 77, _COND_DIM, device=ids.device),)
    text_encoder.eval.return_value = text_encoder
    text_encoder.requires_grad_.return_value = text_encoder

    # Tokenizer must also return tensors on device so attention_mask ends up on device
    enc_out = MagicMock()
    enc_out.input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
    enc_out.attention_mask = torch.ones(1, 77, dtype=torch.long, device=device)
    enc_out.to.return_value = enc_out
    tokenizer = MagicMock()
    tokenizer.model_max_length = 77
    tokenizer.return_value = enc_out

    return LatentDiffusion(
        config=config,
        dit_model=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        engine=engine,
    ).to(device)


def test_generate_output_shape(device):
    """generate() returns (B, 3, H, W) with correct spatial dims."""
    model = _make_latent_diffusion_for_generate(device)
    out = model.generate(["a cat"], height=_H_IMG, width=_H_IMG, num_steps=2)
    assert out.shape == (1, 3, _H_IMG, _H_IMG)


def test_generate_wrong_size_raises(device):
    """Passing height/width mismatched to pos_embedder raises RuntimeError."""
    model = _make_latent_diffusion_for_generate(device)
    with pytest.raises(RuntimeError):
        # height=64 → h_lat=8 → 4×4=16 patches, but pos_embed has 64
        model.generate(["a cat"], height=64, width=64, num_steps=2)
