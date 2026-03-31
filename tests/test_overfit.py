"""Gradient-flow smoke test: tiny DiT + DiffusionEngine should overfit a single batch.

Designed for extensibility:
- Tests operate against DiffusionEngine.compute_loss(model, x_0, cond) — swapping in a
  rectified-flow engine (future work) requires no changes here.
- Parametrized over is_video so both image and video paths are covered.
- Parametrized over diffusion method (DDPM, FlowMatching) so both training objectives
  are validated end-to-end at the gradient-flow level.
"""

import pytest
import torch
from torch.optim import AdamW
from timm.models.vision_transformer import Attention

from models.conditioning import SinCosPosEmbed2D, SinCosPosEmbed3D
from models.layers import AdaLNZeroStrategy
from models.condition_manager import ConditionOutput
from models.models import DiT
from diffusion.methods.ddpm import DDPM
from diffusion.methods.flow_matching import FlowMatching
from diffusion.engine import DiffusionEngine
from diffusion.samplers.ddim_sampler import DDIMSampler
from diffusion.samplers.flow_matching_sampler import FlowMatchingSampler

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
    return DiT(
        is_video=False,
        input_size=16,
        patch_size=_PATCH_2D,
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
    ).to(device)


def _make_3d_dit(device: str) -> DiT:
    # Spatial: 16×16 with patch 2×2 → 8×8 = 64 spatial patches
    # Temporal: 4 frames with patch 2 → 2 temporal patches
    grid_size = 8
    max_frames = 4
    pos_emb = SinCosPosEmbed3D(_HIDDEN, grid_size=grid_size, max_frames=max_frames).to(device)
    return DiT(
        is_video=True,
        input_size=16,
        patch_size=_PATCH_3D,
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
    ).to(device)


def _make_ddpm_engine(device: str) -> DiffusionEngine:
    ddpm = DDPM(num_timesteps=_NUM_STEPS, learn_variance=False).to(device)
    return DiffusionEngine(method=ddpm, sampler=DDIMSampler(ddpm))


def _make_fm_engine(device: str) -> DiffusionEngine:
    fm = FlowMatching(num_timesteps=_NUM_STEPS).to(device)
    return DiffusionEngine(method=fm, sampler=FlowMatchingSampler(fm))


def _make_cond(B: int, device: str) -> ConditionOutput:
    # Plain tensors (no grad graph) — avoids "backward through graph twice" across loop steps.
    # The overfit test exercises DiT gradients only; projector weights are not under test here.
    return ConditionOutput(
        adaLN=torch.randn(B, _HIDDEN, device=device),
    )


# ---------------------------------------------------------------------------
# Overfit helper — shared logic for 2D and 3D
# ---------------------------------------------------------------------------

def _run_overfit(model: DiT, x_0: torch.Tensor, cond: ConditionOutput, device: str,
                 make_engine) -> None:
    engine = make_engine(device)
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

@pytest.mark.parametrize("make_engine", [_make_ddpm_engine, _make_fm_engine],
                         ids=["ddpm", "flow_matching"])
def test_overfit_2d(make_engine, device):
    torch.manual_seed(0)
    model = _make_2d_dit(device)
    x_0 = torch.randn(2, _IN_CH, 16, 16, device=device)
    cond = _make_cond(2, device)
    _run_overfit(model, x_0, cond, device, make_engine)


@pytest.mark.parametrize("make_engine", [_make_ddpm_engine, _make_fm_engine],
                         ids=["ddpm", "flow_matching"])
def test_overfit_3d_video(make_engine, device):
    torch.manual_seed(0)
    model = _make_3d_dit(device)
    # (B, C, F, H, W) — 4 frames, 16×16 spatial
    x_0 = torch.randn(2, _IN_CH, 4, 16, 16, device=device)
    cond = _make_cond(2, device)
    _run_overfit(model, x_0, cond, device, make_engine)
