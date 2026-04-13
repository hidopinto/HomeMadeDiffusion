import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vae_cache import VaeCachedDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cache(tmp_path: Path, num_samples: int) -> Path:
    cache_dir = tmp_path / "cache"
    latent_dir = cache_dir / "latents"
    latent_dir.mkdir(parents=True)
    for i in range(num_samples):
        torch.save(torch.randn(4, 8, 8), latent_dir / f"{i:06d}.pt")
    with (cache_dir / "captions.jsonl").open("w") as f:
        for i in range(num_samples):
            f.write(json.dumps({"id": i, "caption": f"caption {i}"}) + "\n")
    return cache_dir


def _config() -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(encoding_batch_size=4))


def _dummy_encoder_args() -> tuple:
    return MagicMock(), {"text_embed": MagicMock()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaeCachedDataset:
    def test_cached_dataset_init_and_len(self, tmp_path: Path) -> None:
        """Valid cache → correct __len__ and num_samples."""
        cache_dir = _make_fake_cache(tmp_path, num_samples=5)
        tokenizer, text_encoders = _dummy_encoder_args()

        dataset = VaeCachedDataset(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            text_encoders=text_encoders,
            config=_config(),
            device="cpu",
        )

        assert len(dataset) == 5
        assert dataset.num_samples == 5

    def test_cached_dataset_missing_captions_raises(self, tmp_path: Path) -> None:
        """Missing captions.jsonl → FileNotFoundError at init time."""
        cache_dir = tmp_path / "cache"
        (cache_dir / "latents").mkdir(parents=True)
        tokenizer, text_encoders = _dummy_encoder_args()

        with pytest.raises(FileNotFoundError):
            VaeCachedDataset(
                cache_dir=cache_dir,
                tokenizer=tokenizer,
                text_encoders=text_encoders,
                config=_config(),
                device="cpu",
            )

    def test_cached_dataset_empty_captions_len_zero(self, tmp_path: Path) -> None:
        """Empty captions.jsonl → len 0 (documents the silent-zero-step-training scenario)."""
        cache_dir = _make_fake_cache(tmp_path, num_samples=0)
        tokenizer, text_encoders = _dummy_encoder_args()

        dataset = VaeCachedDataset(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            text_encoders=text_encoders,
            config=_config(),
            device="cpu",
        )

        assert len(dataset) == 0
