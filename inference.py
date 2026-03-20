import argparse
from functools import partial
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from model_builder import build_model
from samplers import IntermediateCollector
from utils import load_config


def _every_n_steps(freq: int, step_idx: int, total_steps: int, x: torch.Tensor) -> bool:
    return step_idx % freq == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiT text-to-image inference")
    parser.add_argument("--checkpoint", required=True, help="Path to DiT state_dict .pt file")
    parser.add_argument("--prompt", required=True, nargs="+", help="Text prompt(s)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--scheduler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--eta", type=float, default=0.0,
                        help="Stochasticity: 0=DDIM deterministic, 1=full DDPM noise")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--format", choices=["png", "jpg"], default="png")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--intermediate_freq", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="Home-Made-Diffusion")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    model = build_model(config, device, gradient_checkpointing=False)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.transformer.load_state_dict(state_dict)
    model.transformer.eval()

    collector = None
    if args.save_intermediates:
        capture_fn = partial(_every_n_steps, args.intermediate_freq)
        collector = IntermediateCollector(capture_fn=capture_fn)

    images = model.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        scheduler=args.scheduler,
        eta=args.eta,
        collector=collector,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for i, img_tensor in enumerate(images):
        out_path = Path(args.out_dir) / f"sample_{i:04d}.{args.format}"
        to_pil_image(img_tensor.cpu().float()).save(out_path)
        print(f"Saved {out_path}")

    if collector is not None and collector.decoded_images:
        inter_dir = Path(args.out_dir) / "intermediates"
        inter_dir.mkdir(parents=True, exist_ok=True)
        for step_idx, img_batch in zip(collector.step_indices, collector.decoded_images):
            for prompt_i, img_tensor in enumerate(img_batch):
                out_path = inter_dir / f"sample_{prompt_i:04d}_step{step_idx:04d}.{args.format}"
                to_pil_image(img_tensor.cpu().float()).save(out_path)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
        log_dict = {}
        for i, img_tensor in enumerate(images):
            log_dict[f"inference/final_{i:04d}"] = wandb.Image(img_tensor.cpu())
        if collector is not None:
            for step_idx, img_batch in zip(collector.step_indices, collector.decoded_images):
                for prompt_i, img_tensor in enumerate(img_batch):
                    log_dict[f"inference/prompt_{prompt_i:04d}/step_{step_idx:04d}"] = wandb.Image(img_tensor.cpu())
        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
