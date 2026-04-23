"""Unit tests for _CLIPModelWrapper — no GPU or real CLIP weights required."""
from types import SimpleNamespace

import torch

from evaluation.metrics import _CLIPModelWrapper


class _FakeBaseOutput:
    """Stand-in for transformers.BaseModelOutputWithPooling."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.pooler_output = tensor


class _FakeCLIPModel(torch.nn.Module):
    def get_image_features(self, *args, **kwargs) -> _FakeBaseOutput:
        return _FakeBaseOutput(torch.ones(2, 768))

    def get_text_features(self, *args, **kwargs) -> _FakeBaseOutput:
        return _FakeBaseOutput(torch.ones(2, 512))

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace()


def test_wrapper_extracts_image_tensor():
    wrapper = _CLIPModelWrapper(_FakeCLIPModel())
    out = wrapper.get_image_features()
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 768)


def test_wrapper_extracts_text_tensor():
    wrapper = _CLIPModelWrapper(_FakeCLIPModel())
    out = wrapper.get_text_features()
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 512)


def test_wrapper_supports_norm():
    """Reproduces the exact call that crashed in torchmetrics clip_score.py:193."""
    wrapper = _CLIPModelWrapper(_FakeCLIPModel())
    feats = wrapper.get_image_features()
    normed = feats / feats.norm(p=2, dim=-1, keepdim=True)
    assert normed.shape == feats.shape


def test_wrapper_to_device():
    wrapper = _CLIPModelWrapper(_FakeCLIPModel())
    wrapper = wrapper.cpu()
    out = wrapper.get_image_features()
    assert isinstance(out, torch.Tensor)


def test_wrapper_config_forwarded():
    wrapper = _CLIPModelWrapper(_FakeCLIPModel())
    _ = wrapper.config  # must not raise


def test_wrapper_passthrough_when_no_pooler_output():
    """If the model already returns a plain Tensor, return it unchanged."""

    class _PlainModel(torch.nn.Module):
        def get_image_features(self, *a, **kw) -> torch.Tensor:
            return torch.zeros(3, 512)

        @property
        def config(self) -> SimpleNamespace:
            return SimpleNamespace()

    wrapper = _CLIPModelWrapper(_PlainModel())
    out = wrapper.get_image_features()
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 512)
