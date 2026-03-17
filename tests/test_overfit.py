"""Gradient-flow smoke test: tiny DiT + DiffusionEngine should overfit a single batch.

Designed for extensibility:
- Tests operate against DiffusionEngine.compute_loss(model, x_0, cond) — swapping in a
  rectified-flow engine (future work) requires no changes here.
- Parametrized over is_video so both image and video paths are covered.
"""

import pytest
import torch
from torch.optim import AdamW
from timm.models.vision_transformer import Attention

from models.conditioning import SinCosPosEmbed2D, SinCosPosEmbed3D
from models.layers import AdaLNZeroStrategy, AdaLNTextProjector
from models.models import DiT
from diffusion_engine import DDPM, DiffusionEngine
from samplers import DDIMSampler

_HIDDEN = 128
_IN_CH = 4
_COND_DIM = 128
_PATCH_2D = [2, 2]
_PATCH_3D = [2, 2, 2]
_NUM_STEPS = 10
_GRAD_STEPS = 10


# ---------------------------------------------------------------------------
# Model builders (no nested functions — use module-level helpers)
# ---------------------------------------------------------------------------

def _make_2d_dit(device: str) -> DiT:
    grid_size = 8   # 16 // 2
    pos_emb = SinCosPosEmbed2D(_HIDDEN, grid_size=grid_size).to(device)
    txt_proj = AdaLNTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    return DiT(
        is_video=False,
        input_size=16,
        patch_size=_PATCH_2D,
        in_channels=_IN_CH,
        hidden_size=_HIDDEN,
        text_projector=txt_proj,
        frequency_embedding_size=64,
        max_period=10000,
        depth=2,
        num_heads=4,
        pos_embedder=pos_emb,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        learn_variance=False,
    ).to(device)


def _make_3d_dit(device: str) -> DiT:
    # Spatial: 16×16 with patch 2×2 → 8×8 = 64 spatial patches
    # Temporal: 4 frames with patch 2 → 2 temporal patches
    grid_size = 8
    max_frames = 4
    pos_emb = SinCosPosEmbed3D(_HIDDEN, grid_size=grid_size, max_frames=max_frames).to(device)
    txt_proj = AdaLNTextProjector(cond_dim=_COND_DIM, hidden_size=_HIDDEN).to(device)
    return DiT(
        is_video=True,
        input_size=16,
        patch_size=_PATCH_3D,
        in_channels=_IN_CH,
        hidden_size=_HIDDEN,
        text_projector=txt_proj,
        frequency_embedding_size=64,
        max_period=10000,
        depth=2,
        num_heads=4,
        pos_embedder=pos_emb,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        learn_variance=False,
    ).to(device)


def _make_engine() -> DiffusionEngine:
    ddpm = DDPM(num_timesteps=_NUM_STEPS, learn_variance=False)
    return DiffusionEngine(method=ddpm, sampler=DDIMSampler(ddpm))


def _make_cond(B: int, device: str) -> dict:
    return {
        "hidden_states": torch.randn(B, 77, _COND_DIM, device=device),
        "attention_mask": torch.ones(B, 77, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Overfit helper — shared logic for 2D and 3D
# ---------------------------------------------------------------------------

def _run_overfit(model: DiT, x_0: torch.Tensor, cond: dict, device: str) -> None:
    engine = _make_engine()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    model.train()
    initial_loss: float | None = None

    for step in range(_GRAD_STEPS):
        optimizer.zero_grad()
        loss = engine.compute_loss(model, x_0, cond)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.95, (
        f"Loss did not decrease by ≥5%: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_overfit_2d(device):
    torch.manual_seed(0)
    model = _make_2d_dit(device)
    x_0 = torch.randn(2, _IN_CH, 16, 16, device=device)
    cond = _make_cond(2, device)
    _run_overfit(model, x_0, cond, device)


def test_overfit_3d_video(device):
    torch.manual_seed(0)
    model = _make_3d_dit(device)
    # (B, C, F, H, W) — 4 frames, 16×16 spatial
    x_0 = torch.randn(2, _IN_CH, 4, 16, 16, device=device)
    cond = _make_cond(2, device)
    _run_overfit(model, x_0, cond, device)
