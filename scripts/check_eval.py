#!/usr/bin/env python
"""Smoke-test the CLIP evaluation wrapper end-to-end.

Verifies that the _CLIPModelWrapper correctly bridges transformers 5.x
BaseModelOutputWithPooling back to the plain-Tensor interface that torchmetrics
CLIPScore expects.

Run:  python check_eval.py
"""
import torch
from torchmetrics.multimodal.clip_score import CLIPScore

from evaluation.metrics import _CLIPModelWrapper


def main() -> None:
    print("Loading CLIPScore (openai/clip-vit-large-patch14)...")
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
    clip_score.model = _CLIPModelWrapper(clip_score.model)
    clip_score.eval()

    imgs = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8)
    prompts = [
        "a photo of a cat",
        "a photo of a dog",
        "mountains at sunset",
        "a red car on a street",
    ]

    print("Running CLIPScore.update()...")
    clip_score.update(imgs, prompts)
    score = clip_score.compute().item()
    print(f"CLIP Score: {score:.3f}  (random-noise images, expect ~20–35)")
    assert 0.0 <= score <= 100.0, f"Score out of valid range: {score}"
    print("check_eval.py PASSED")


if __name__ == "__main__":
    main()
