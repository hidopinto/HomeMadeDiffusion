#!/usr/bin/env python3
"""
Portfolio evaluation: generate high-quality images for curated prompts across
multiple checkpoints. Use this to decide which checkpoint is portfolio-ready
and whether T5 training is needed.

Edit PORTFOLIO_PROMPTS below before running — pick the prompts that looked
promising in the probe_categories.py run.

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
import logging
import re
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_builder import build_model
from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edit this list — prompts that looked promising in probe_categories.py.
# These are the probe prompts that showed best results across checkpoints.
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


def _ckpt_label(path: Path) -> str:
    """Extract a short label from checkpoint filename, e.g. 'dit_step0050000.pt' → 'step50k'."""
    m = re.search(r"step(\d+)", path.stem)
    if not m:
        return path.stem[:12]
    n = int(m.group(1))
    return f"step{n // 1000}k" if n % 1000 == 0 else f"step{n}"


def _prompt_slug(prompt: str, max_len: int = 55) -> str:
    slug = re.sub(r"[^a-zA-Z0-9 ]", "", prompt)
    slug = "_".join(slug.split())
    return slug[:max_len]


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(str(path), map_location=device, weights_only=True)
    xfm_sd = {k.replace("._orig_mod", ""): v for k, v in state["transformer"].items()}
    cm_sd  = {k.replace("._orig_mod", ""): v for k, v in state["condition_manager"].items()}
    model.transformer.load_state_dict(xfm_sd)
    model.condition_manager.load_state_dict(cm_sd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio eval: generate polished images for curated prompts across checkpoints"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="One or more .pt checkpoint paths")
    parser.add_argument("--config", default="configs/train_pickapic_cc12m_continued.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-images", type=int, default=4,
                        help="Images per prompt per checkpoint")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Inference steps (100 = high quality)")
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

    logger.info("Building model on %s ...", device)
    model = build_model(config, device, gradient_checkpointing=False, compile_blocks=False)
    model.cache_null_embed(torch.device(device))
    model.vae.enable_slicing()

    total = len(ckpt_paths) * len(PORTFOLIO_PROMPTS)
    done = 0

    for ckpt_path, label in zip(ckpt_paths, labels):
        logger.info("Loading checkpoint: %s (%s)", ckpt_path.name, label)
        _load_checkpoint(model, ckpt_path, device)
        model.transformer.eval()

        for prompt in PORTFOLIO_PROMPTS:
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
                )

            for i, img in enumerate(images):
                out_path = prompt_dir / f"{label}_{i:02d}.png"
                to_pil_image(img.cpu().float()).save(out_path)

            torch.cuda.empty_cache()
            done += 1
            logger.info("[%d/%d] %s | %s", done, total, label, prompt[:70])

    logger.info("Done. Images saved to %s", out_dir)
    logger.info("Browse by prompt directory to compare checkpoints side-by-side.")


if __name__ == "__main__":
    main()
