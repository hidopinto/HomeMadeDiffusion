"""Sweep every dit_stepNNNNNNN.pt in a directory: compute FID + CLIP Score per checkpoint.

Usage:
    python scripts/eval_checkpoints.py \
        --checkpoint_dir "/media/hido-pinto/מחסן/checkpoints/pickapic_checkpoints/continued from cc12m" \
        --config configs/train_pickapic_v1.yaml \
        [--out_dir /media/hido-pinto/מחסן/outputs/sweep]

Outputs:
  {out_dir}/sweep_results.csv          — step, fid, clip_score
  {out_dir}/step{N:07d}/sample_*.png   — 2 inference images per checkpoint
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

# Scripts run from the repo root; add it to sys.path so imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import build_dataloader
from evaluation.metrics import EvaluationEngine, _fid_stats_path
from model_builder import build_model
from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"dit_step(\d+)\.pt$")


def _parse_step(path: Path) -> int:
    m = _STEP_RE.match(path.name)
    if m is None:
        raise ValueError(f"Cannot parse step number from filename: {path.name}")
    return int(m.group(1))


def _load_checkpoint(model, path: Path, device: str) -> None:
    state = torch.load(str(path), map_location=device, weights_only=True)
    xfm_sd = {k.replace("._orig_mod", ""): v for k, v in state["transformer"].items()}
    cm_sd  = {k.replace("._orig_mod", ""): v for k, v in state["condition_manager"].items()}
    model.transformer.load_state_dict(xfm_sd)
    model.condition_manager.load_state_dict(cm_sd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FID + CLIP Score for every checkpoint in a directory"
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing dit_stepNNNNNNN.pt files")
    parser.add_argument("--config", default="config.yaml",
                        help="Config YAML path (default: config.yaml)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for images and CSV "
                             "(default: <checkpoint_dir>/eval_sweep)")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Images per generation batch during eval; overrides config if set")
    parser.add_argument("--eval_num_steps", type=int, default=None,
                        help="Diffusion steps for FID generation; overrides config if set")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint_dir) / "eval_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(
        [p for p in ckpt_dir.glob("*.pt") if _STEP_RE.match(p.name)],
        key=_parse_step,
    )
    if not checkpoints:
        logger.error("No dit_stepNNNNNNN.pt files found in %s", ckpt_dir)
        sys.exit(1)
    logger.info("Found %d checkpoints: %s", len(checkpoints), [p.name for p in checkpoints])

    # Build model — keep all sub-models on GPU (inference-only, no offloading needed).
    logger.info("Building model on %s...", device)
    model = build_model(config, device, gradient_checkpointing=False, compile_blocks=False)
    model.cache_null_embed(torch.device(device))
    model.vae.enable_slicing()

    # Build EvaluationEngine.
    # If the FID real-image stats cache already exists (written during training), the engine
    # loads it from disk and never touches val_dataloader — so we pass None in that case.
    if args.eval_batch_size is not None:
        config.training.eval_batch_size = args.eval_batch_size
    if args.eval_num_steps is not None:
        config.training.eval_num_steps = args.eval_num_steps

    eval_num_samples: int = getattr(config.training, "eval_num_samples", 2048)
    max_real = max(2048, min(eval_num_samples * 4, 10_000))
    fid_cache_path = _fid_stats_path(config, max_real)

    if fid_cache_path.exists():
        logger.info("FID stats cache found at %s — no val_dataloader needed.", fid_cache_path)
        val_dataloader = None
    else:
        logger.info("FID stats cache not found — building val_dataloader to compute real stats.")
        val_split = getattr(config.data, "val_split", None)
        if not val_split:
            logger.error(
                "FID cache missing and config.data.val_split is not set. "
                "Cannot compute real FID statistics without a validation set."
            )
            sys.exit(1)
        val_dataloader = build_dataloader(
            config, model.vae, model.tokenizer, model.text_encoder, device,
            split=val_split,
            shuffle=False,
        )

    eval_engine = EvaluationEngine(config, val_dataloader, model, device)  # type: ignore[arg-type]

    # Inference settings — use the same prompt as in-training inference for visual consistency.
    inference_prompt: str = getattr(config.training, "inference_prompt", "A photo.")
    sampler_cfg = config.diffusion.samplers[config.diffusion.sampler]
    inference_steps: int = getattr(sampler_cfg, "num_steps", 50)
    inference_scheduler: str = config.diffusion.sampler
    guidance_scale: float = getattr(config.training, "guidance_scale", 7.5)
    inference_height: int = getattr(config.training, "inference_height", 512)
    inference_width: int = getattr(config.training, "inference_width", 512)

    results: list[dict] = []

    for ckpt_path in checkpoints:
        step = _parse_step(ckpt_path)
        logger.info("=== Checkpoint step %07d — %s ===", step, ckpt_path.name)

        _load_checkpoint(model, ckpt_path, device)

        # FID + CLIP Score — compute() handles .eval() / .train() on transformer internally.
        metrics = eval_engine.compute(model, step)
        fid = metrics.get("eval/fid", float("nan"))
        clip = metrics.get("eval/clip_score", float("nan"))
        logger.info("  FID=%.2f  CLIP=%.3f", fid, clip)

        # Inference — two images for side-by-side visual comparison.
        img_dir = out_dir / f"step{step:07d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        model.transformer.eval()
        with torch.no_grad():
            images = model.generate(
                [inference_prompt, inference_prompt],
                height=inference_height,
                width=inference_width,
                num_steps=inference_steps,
                guidance_scale=guidance_scale,
                scheduler=inference_scheduler,
            )
        for i, img in enumerate(images):
            to_pil_image(img.cpu().float()).save(img_dir / f"sample_{i:04d}.png")
        torch.cuda.empty_cache()

        results.append({"step": step, "fid": fid, "clip_score": clip})

    # Print summary table.
    print("\n" + "=" * 52)
    print(f"{'Step':>10}  {'FID':>10}  {'CLIP Score':>12}")
    print("-" * 52)
    for r in results:
        print(f"{r['step']:>10}  {r['fid']:>10.2f}  {r['clip_score']:>12.3f}")
    print("=" * 52 + "\n")

    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "fid", "clip_score"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", csv_path)
    logger.info("Inference images saved under %s/step*/", out_dir)


if __name__ == "__main__":
    main()
