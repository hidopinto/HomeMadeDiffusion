#!/usr/bin/env python3
"""
Category diagnostic: generate images for a diverse set of category prompts and save
them organized by category. Use this to identify which categories the current checkpoint
handles well before committing to T5 training.

Usage:
    python scripts/probe_categories.py \\
        --checkpoint "/media/hido-pinto/מחסן/checkpoints/pickapic_checkpoints/continued from cc12m/dit_step0050000.pt" \\
        --config configs/train_pickapic_cc12m_continued.yaml \\
        --out_dir /media/hido-pinto/מחסן/outputs/probe_categories/step50k \\
        [--num-steps 50] [--guidance-scale 6.5] [--images-per-prompt 2]
"""
from __future__ import annotations

import argparse
import logging
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
# Category probe prompts — grouped by category for easy triage.
# Adjust freely; the goal is to identify which categories the model handles well.
# ---------------------------------------------------------------------------
CATEGORY_PROMPTS: dict[str, list[str]] = {
    "animals_common": [
        "A golden retriever dog sitting on grass",
        "A tabby cat sleeping on a couch",
        "A horse galloping in a field",
    ],
    "animals_rare": [
        "A colorful parrot perched on a branch",
        "A large brown bear catching a fish in a river",
        "A bright red and orange tropical fish swimming",
    ],
    "plants_and_nature": [
        "A tall green cactus in a sandy desert",
        "A sunflower field under a blue sky",
        "A cherry blossom tree in full bloom",
    ],
    "people": [
        "A baby crawling on a wooden floor",
        "A woman in a red dress dancing",
        "A runner wearing a green shirt in a marathon",
        "An astronaut in a white spacesuit walking on a salt flat",
    ],
    "food": [
        "A slice of pepperoni pizza on a plate",
        "A bowl of ramen with vegetables and soft boiled egg",
        "A chocolate cake with strawberries on top",
    ],
    "scenes_outdoor": [
        "A mountain lake reflecting the surrounding peaks at sunset",
        "A narrow cobblestone street in an old European town",
        "A sandy beach with turquoise ocean water and palm trees",
    ],
    "scenes_indoor": [
        "A cozy living room with a fireplace and bookshelves",
        "A bright kitchen with stainless steel appliances",
    ],
    "vehicles": [
        "A red sports car parked on a mountain road",
        "A steam locomotive crossing a bridge over a gorge",
        "A sailboat on the ocean at sunrise",
    ],
    "objects": [
        "A wooden chair with a red cushion",
        "A vintage leather suitcase with travel stickers",
        "An open book on a wooden table with a cup of coffee",
    ],
    "art_and_style": [
        "An oil painting of a stormy sea with dramatic waves",
        "A pencil sketch portrait of an elderly man with wrinkles",
    ],
}


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(str(path), map_location=device, weights_only=True)
    xfm_sd = {k.replace("._orig_mod", ""): v for k, v in state["transformer"].items()}
    cm_sd  = {k.replace("._orig_mod", ""): v for k, v in state["condition_manager"].items()}
    model.transformer.load_state_dict(xfm_sd)
    model.condition_manager.load_state_dict(cm_sd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Category probe: generate images per category")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", default="configs/train_pickapic_cc12m_continued.yaml")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Inference steps (FM: 50 is high quality)")
    parser.add_argument("--guidance-scale", type=float, default=6.5)
    parser.add_argument("--images-per-prompt", type=int, default=2,
                        help="Number of images per prompt (2 = easy to spot failures)")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    ckpt_path = Path(args.checkpoint)

    logger.info("Building model on %s ...", device)
    model = build_model(config, device, gradient_checkpointing=False, compile_blocks=False)
    model.cache_null_embed(torch.device(device))
    model.vae.enable_slicing()

    logger.info("Loading checkpoint: %s", ckpt_path)
    _load_checkpoint(model, ckpt_path, device)
    model.transformer.eval()

    scheduler = config.diffusion.sampler
    total_prompts = sum(len(v) for v in CATEGORY_PROMPTS.values())
    done = 0

    for category, prompts in CATEGORY_PROMPTS.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for prompt in prompts:
            prompt_slug = prompt[:60].replace(" ", "_").replace("/", "-")
            img_dir = cat_dir / prompt_slug
            img_dir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                images = model.generate(
                    [prompt] * args.images_per_prompt,
                    height=args.height,
                    width=args.width,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    scheduler=scheduler,
                )
            for i, img in enumerate(images):
                to_pil_image(img.cpu().float()).save(img_dir / f"sample_{i:02d}.png")
            torch.cuda.empty_cache()
            done += 1
            logger.info("[%d/%d] %s | %s", done, total_prompts, category, prompt[:60])

    logger.info("Done. Images saved to %s", out_dir)
    logger.info("Browse by category to find which prompts produce good results for demo selection.")


if __name__ == "__main__":
    main()
