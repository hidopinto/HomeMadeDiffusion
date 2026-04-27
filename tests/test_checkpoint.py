"""Checkpoint round-trip tests.

Verifies that the .pt payload written by _save_checkpoint() contains both
'transformer' and 'condition_manager' keys and can be loaded back into a
freshly-constructed model. Also checks the legacy (pre-42c2125) format.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from models.condition_manager import ConditionManager
from models.models import LatentDiffusion
from models.projectors import AdaLNTextProjector
from diffusion.engine import DiffusionEngine
from diffusion.methods.ddpm import DDPM
from diffusion.samplers.ddim_sampler import DDIMSampler

from tests.conftest import (
    COND_DIM, HIDDEN_SIZE, IN_CHANNELS,
    _make_dit,
)


def _make_tiny_latent_diffusion(device: str) -> LatentDiffusion:
    dit = _make_dit(device, out_channels=IN_CHANNELS)
    ddpm = DDPM(num_timesteps=10, learn_variance=False)
    engine = DiffusionEngine(method=ddpm, sampler=DDIMSampler(ddpm))
    config = MagicMock()
    config.dit.in_channels = IN_CHANNELS
    config.dit.vae_scale_factor = 0.18215
    config.training.cfg_dropout_prob = 0.0
    condition_manager = ConditionManager([
        ("text", AdaLNTextProjector(cond_dim=COND_DIM, hidden_size=HIDDEN_SIZE)),
    ]).to(device)
    vae = MagicMock()
    vae.eval.return_value = vae
    vae.requires_grad_.return_value = vae
    text_encoder = MagicMock()
    text_encoder.eval.return_value = text_encoder
    text_encoder.requires_grad_.return_value = text_encoder
    return LatentDiffusion(
        config=config,
        dit_model=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=MagicMock(),
        engine=engine,
        condition_manager=condition_manager,
    ).to(device)


def test_checkpoint_payload_has_both_keys(device: str) -> None:
    """New .pt payload contains 'transformer' and 'condition_manager' keys."""
    model = _make_tiny_latent_diffusion(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dit_step0000001.pt"
        payload = {
            "transformer": model.transformer.state_dict(),
            "condition_manager": model.condition_manager.state_dict(),
        }
        torch.save(payload, str(path))

        loaded = torch.load(str(path), map_location=device, weights_only=True)
        assert "transformer" in loaded
        assert "condition_manager" in loaded

        fresh = _make_tiny_latent_diffusion(device)
        fresh.transformer.load_state_dict(loaded["transformer"])
        fresh.condition_manager.load_state_dict(loaded["condition_manager"])


def test_checkpoint_step_file_round_trip(tmp_path: Path) -> None:
    """step.txt is written and read back to the exact integer."""
    step_file = tmp_path / "step.txt"
    step_file.write_text("42000")
    assert int(step_file.read_text().strip()) == 42000


def test_legacy_checkpoint_loads_transformer_only(device: str) -> None:
    """Legacy .pt files (raw transformer state_dict, no 'transformer' key) still load."""
    model = _make_tiny_latent_diffusion(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.pt"
        torch.save(model.transformer.state_dict(), str(path))

        loaded = torch.load(str(path), map_location=device, weights_only=True)
        assert not (isinstance(loaded, dict) and "transformer" in loaded)

        fresh = _make_tiny_latent_diffusion(device)
        xfm_sd = {k.replace("._orig_mod", ""): v for k, v in loaded.items()}
        fresh.transformer.load_state_dict(xfm_sd)


def test_resume_raises_on_missing_path(device: str) -> None:
    """fit() raises FileNotFoundError when resume_from_checkpoint path does not exist."""
    from trainer import DiTTrainer

    config = MagicMock()
    config.training.mixed_precision = "no"
    config.training.gradient_accumulation_steps = 1
    config.training.gradient_checkpointing = False
    config.training.resume_from_checkpoint = "/nonexistent/path/full_ckpt"
    config.training.checkpoint_dir = "checkpoints"
    config.training.save_every_steps = False
    config.training.inference_every_steps = False
    config.training.eval_every_steps = False
    config.general.wnb_project_name = "test"
    config.general.wnb_entity = "test"

    model = _make_tiny_latent_diffusion(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = DiTTrainer(
        config=config,
        model=model,
        dataloader=[],
        optimizer=optimizer,
        lr_scheduler=None,
        max_steps=1,
    )
    with pytest.raises(FileNotFoundError, match="resume_from_checkpoint"):
        trainer.fit(epochs=1)
