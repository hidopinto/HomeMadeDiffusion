#!/usr/bin/env python
"""Smoke-test all three evaluation metrics (FID, IS, CLIPScore) end-to-end.

Does NOT require a running training loop or checkpoint.  Uses small batches of
random tensors to verify nothing crashes before committing to a >15-hour run.

Run:  python check_eval.py
"""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore

from evaluation.metrics import _CLIPModelWrapper

_N = 8       # samples per split — exercises code paths, not statistically reliable
_H, _W = 299, 299  # InceptionV3 native resolution; avoids internal resize warnings


def _rand_float(n: int) -> torch.Tensor:
    return torch.rand(n, 3, _H, _W)


def _rand_uint8(n: int) -> torch.Tensor:
    return torch.randint(0, 256, (n, 3, 224, 224), dtype=torch.uint8)


def check_fid() -> None:
    print("\n[FID] loading InceptionV3...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    fid.update(_rand_float(_N), real=True)
    fid.update(_rand_float(_N), real=False)
    val = fid.compute().item()
    print(f"[FID] value={val:.2f}  (random vs random, large value expected)")
    assert val >= 0.0, f"FID should be non-negative, got {val}"
    print("[FID] PASSED")


def check_is() -> None:
    print("\n[IS] loading InceptionV3...")
    isc = InceptionScore(normalize=True)
    isc.update(_rand_float(_N))
    mean, std = isc.compute()
    print(f"[IS]  mean={mean.item():.3f}  std={std.item():.3f}")
    assert mean.item() >= 0.0
    print("[IS] PASSED")


def check_clip() -> None:
    print("\n[CLIPScore] loading openai/clip-vit-large-patch14...")
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
    clip_score.model = _CLIPModelWrapper(clip_score.model)
    clip_score.eval()
    prompts = [
        "a photo of a cat", "a photo of a dog", "mountains at sunset",
        "a red car on a street", "a bowl of fruit on a table",
        "children playing in a park", "a snowy winter landscape",
        "an astronaut on the moon",
    ]
    clip_score.update(_rand_uint8(_N), prompts)
    score = clip_score.compute().item()
    print(f"[CLIPScore] score={score:.3f}  (random images, expect ~20–35)")
    assert 0.0 <= score <= 100.0, f"Score out of valid range: {score}"
    print("[CLIPScore] PASSED")


def main() -> None:
    check_fid()
    check_is()
    check_clip()
    print("\nAll evaluation metrics PASSED — safe to run training.")


if __name__ == "__main__":
    main()
