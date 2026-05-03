#!/usr/bin/env python3
"""
Portfolio evaluation: generate high-quality images for curated prompts across
multiple checkpoints, and score each image with CLIP.

CLIP score measures text-image alignment (did the model follow the prompt?).
It does NOT measure aesthetics or sharpness. Use it to:
  - Filter out failures (score < 0.22 → model diverged from the prompt)
  - Compare checkpoints per prompt (a 0.02+ gap is meaningful)
  - Identify which images are worth eyeballing for portfolio selection

The score is embedded in each filename so file browsers sort by quality.
A summary table and CSV are printed/saved at the end.

Edit PORTFOLIO_PROMPTS below before running.

Usage:
    python scripts/portfolio_eval.py \\
        --checkpoints \\
            "/media/hido-pinto/מחסן/checkpoints/pickapic_checkpoints/continued from cc12m/dit_step0030000.pt" \\
            "/media/hido-pinto/מחסן/checkpoints/pickapic_checkpoints/continued from cc12m/dit_step0040000.pt" \\
            "/media/hido-pinto/מחסן/checkpoints/pickapic_checkpoints/continued from cc12m/dit_step0050000.pt" \\
        --config configs/train_pickapic_cc12m_continued.yaml \\
        --out-dir /media/hido-pinto/מחסן/outputs/portfolio_eval \\
        --num-images 4 \\
        --num-steps 100 \\
        --guidance-scale 6.5
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_builder import build_model
from utils import load_config
from evaluation.metrics import _CLIPModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit this list — prompts that looked promising in probe_categories.py.
# ---------------------------------------------------------------------------
PORTFOLIO_PROMPTS: list[str] = [
    # step 50k strengths
    "An astronaut in a white spacesuit walking on a salt flat",
    "A bowl of ramen with vegetables and soft boiled egg",
    "A pencil sketch portrait of an elderly man with wrinkles",
    "An oil painting of a stormy sea with dramatic waves",
    "A professional cinematic photo of a solitary astronaut walking on a white salt flat, clear blue sky, highly detailed suit.",
    # step 40k strengths
    "A cherry blossom tree in full bloom",
    "A cozy living room with a fireplace and bookshelves",
    "A bright kitchen with stainless steel appliances",
    "A narrow cobblestone street in an old European town",
    # step 30k strengths
    "A sunflower field under a blue sky",
    "A tall green cactus in a sandy desert",
    # others worth testing
    "A golden retriever dog sitting on grass",
    "A mountain lake reflecting the surrounding peaks at sunset",
    "A sandy beach with turquoise ocean water and palm trees",
    "A red sports car parked on a mountain road",
    "A slice of pepperoni pizza on a plate",
]

CURATED_PORTFOLIO_PROMPTS = [
     "A professional cinematic photo of a solitary astronaut walking on a white salt flat, clear blue sky, highly detailed suit.",
     "An oil painting of a stormy sea with dramatic waves",
     "A cherry blossom tree in full bloom",
     "A sunflower field under a blue sky",
     "A narrow cobblestone street in an old European town",
     "A golden retriever dog sitting on grass",
     "A sandy beach with turquoise ocean water and palm trees",
     "A mountain lake reflecting the surrounding peaks at sunset",
     "An astronaut in a white spacesuit walking on a salt flat",
     "A tall green cactus in a sandy desert",
 ]


def _ckpt_label(path: Path) -> str:
    m = re.search(r"step(\d+)", path.stem)
    if not m:
        return path.stem[:12]
    n = int(m.group(1))
    return f"step{n // 1000}k" if n % 1000 == 0 else f"step{n}"


def _prompt_slug(prompt: str, max_len: int = 55) -> str:
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", prompt)
    return "_".join(slug.split())[:max_len]


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(str(path), map_location=device, weights_only=True)
    xfm_sd = {k.replace("._orig_mod", ""): v for k, v in state["transformer"].items()}
    cm_sd  = {k.replace("._orig_mod", ""): v for k, v in state["condition_manager"].items()}
    model.transformer.load_state_dict(xfm_sd)
    model.condition_manager.load_state_dict(cm_sd)


def _build_clip_scorer(clip_model_name: str, device: str):
    """Returns score(images, prompts) → list[float], using full CLIP (text + vision).

    images: (B, 3, H, W) float tensor in [0, 1]
    prompts: list of B strings
    Returns per-image cosine similarity scores, typically in [0.20, 0.35].
    """
    from transformers import CLIPModel, CLIPProcessor

    logger.info("Loading CLIP scorer (%s) ...", clip_model_name)
    clip_raw = CLIPModel.from_pretrained(clip_model_name, torch_dtype=torch.float32).to(device)
    clip = _CLIPModelWrapper(clip_raw)
    clip.eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    @torch.no_grad()
    def score(images_float: torch.Tensor, prompts: list[str]) -> list[float]:
        # images_float: (B, 3, H, W) in [0, 1] — convert to uint8 PIL for processor
        pil_images = [to_pil_image(img.cpu().float().clamp(0, 1)) for img in images_float]
        inputs = processor(
            text=prompts, images=pil_images,
            return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        img_feats = clip.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feats = clip.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        sims = (img_feats * txt_feats).sum(dim=-1)
        return sims.cpu().tolist()

    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio eval: generate + CLIP-score images across checkpoints"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", default="configs/train_pickapic_cc12m_continued.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    scheduler = config.diffusion.sampler

    ckpt_paths = [Path(p) for p in args.checkpoints]
    labels = [_ckpt_label(p) for p in ckpt_paths]

    logger.info("Building generation model on %s ...", device)
    model = build_model(config, device, gradient_checkpointing=False, compile_blocks=False)
    model.cache_null_embed(torch.device(device))
    model.vae.enable_slicing()

    clip_score_fn = _build_clip_scorer(config.evaluation.clip_model_name, device)

    # scores[prompt_slug][ckpt_label] = [score_per_image, ...]
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    total = len(ckpt_paths) * len(CURATED_PORTFOLIO_PROMPTS)
    done = 0

    for ckpt_path, label in zip(ckpt_paths, labels):
        logger.info("Loading checkpoint: %s (%s)", ckpt_path.name, label)
        _load_checkpoint(model, ckpt_path, device)
        model.transformer.eval()

        for prompt in CURATED_PORTFOLIO_PROMPTS:
            slug = _prompt_slug(prompt)
            prompt_dir = out_dir / slug
            prompt_dir.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                images = model.generate(
                    [prompt] * args.num_images,
                    height=args.height,
                    width=args.width,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    scheduler=scheduler,
                )  # (N, 3, H, W) float [0, 1]

            img_scores = clip_score_fn(images, [prompt] * args.num_images)
            scores[slug][label] = img_scores

            for i, (img, sc) in enumerate(zip(images, img_scores)):
                # Score in filename → file browsers sort by quality automatically
                out_path = prompt_dir / f"{label}_{i:02d}_clip{sc:.3f}.png"
                to_pil_image(img.cpu().float()).save(out_path)

            mean_sc = sum(img_scores) / len(img_scores)
            max_sc  = max(img_scores)
            torch.cuda.empty_cache()
            done += 1
            logger.info(
                "[%d/%d] %-10s | clip mean=%.3f max=%.3f | %s",
                done, total, label, mean_sc, max_sc, prompt[:65],
            )

    # ── Summary table ────────────────────────────────────────────────────────
    col_w = 13
    header_w = 46
    sep = "=" * (header_w + col_w * len(labels))
    print(f"\n{sep}")
    print("CLIP SCORE SUMMARY  (mean over images; higher = better prompt alignment)")
    print(sep)
    header = f"{'Prompt':<{header_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for prompt in CURATED_PORTFOLIO_PROMPTS:
        slug = _prompt_slug(prompt)
        row = f"{prompt[:header_w - 1]:<{header_w}}"
        for label in labels:
            sc_list = scores[slug].get(label, [])
            row += f"{sum(sc_list)/len(sc_list):>{col_w}.3f}" if sc_list else f"{'—':>{col_w}}"
        print(row)
    print(sep)
    print("Per-checkpoint overall means:")
    for label in labels:
        all_sc = [s for slug in scores for s in scores[slug].get(label, [])]
        if all_sc:
            print(f"  {label:<10}  mean={sum(all_sc)/len(all_sc):.3f}  "
                  f"max={max(all_sc):.3f}  min={min(all_sc):.3f}")
    print(sep)

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "clip_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "checkpoint", "image_idx", "clip_score"])
        for prompt in CURATED_PORTFOLIO_PROMPTS:
            slug = _prompt_slug(prompt)
            for label in labels:
                for i, sc in enumerate(scores[slug].get(label, [])):
                    writer.writerow([prompt, label, i, f"{sc:.4f}"])
    logger.info("Scores saved to %s", csv_path)
    logger.info("Images saved to %s", out_dir)


if __name__ == "__main__":
    main()
