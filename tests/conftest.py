import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from timm.models.vision_transformer import Attention

from models.conditioning import SinCosPosEmbed2D
from models.layers import AdaLNZeroStrategy, AdaLNTextProjector
from models.models import DiT
from diffusion.methods.ddpm import DDPM

# ---------------------------------------------------------------------------
# Tiny config constants — keeps all tests fast (< 1 s on CPU)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
DEPTH = 2
NUM_HEADS = 4
INPUT_SIZE = 16        # 16×16 latent grid
PATCH_SIZE = [2, 2]
IN_CHANNELS = 4
COND_DIM = 128         # tiny text embedding dim
FREQ_EMBED_SIZE = 64
MAX_PERIOD = 10000


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Batch tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_latents(device: str) -> torch.Tensor:
    """(B, C, H, W) = (2, 4, 16, 16) — INPUT_SIZE=16, patch 2×2 → 8×8=64 patches."""
    return torch.randn(2, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)


@pytest.fixture
def batch_text_embeds(device: str) -> dict:
    return {
        "hidden_states": torch.randn(2, 77, COND_DIM, device=device),
        "attention_mask": torch.ones(2, 77, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Isolated component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pos_embedder(device: str) -> SinCosPosEmbed2D:
    grid_size = INPUT_SIZE // PATCH_SIZE[0]  # 8
    return SinCosPosEmbed2D(HIDDEN_SIZE, grid_size=grid_size).to(device)


@pytest.fixture
def text_projector(device: str) -> AdaLNTextProjector:
    return AdaLNTextProjector(cond_dim=COND_DIM, hidden_size=HIDDEN_SIZE).to(device)


# ---------------------------------------------------------------------------
# DiT model helpers
# ---------------------------------------------------------------------------

def _make_dit(device: str, out_channels: int) -> DiT:
    grid_size = INPUT_SIZE // PATCH_SIZE[0]
    pos_emb = SinCosPosEmbed2D(HIDDEN_SIZE, grid_size=grid_size).to(device)
    txt_proj = AdaLNTextProjector(cond_dim=COND_DIM, hidden_size=HIDDEN_SIZE).to(device)
    return DiT(
        is_video=False,
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=out_channels,
        hidden_size=HIDDEN_SIZE,
        text_projector=txt_proj,
        frequency_embedding_size=FREQ_EMBED_SIZE,
        max_period=MAX_PERIOD,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        pos_embedder=pos_emb,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
    ).to(device)


@pytest.fixture
def dit_model(device: str) -> DiT:
    return _make_dit(device, out_channels=IN_CHANNELS)


@pytest.fixture
def dit_model_with_variance(device: str) -> DiT:
    return _make_dit(device, out_channels=2 * IN_CHANNELS)


# ---------------------------------------------------------------------------
# DDPM fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ddpm() -> DDPM:
    return DDPM(num_timesteps=100, learn_variance=False)


@pytest.fixture
def ddpm_with_variance() -> DDPM:
    return DDPM(num_timesteps=100, learn_variance=True)
