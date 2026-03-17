import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from model_builder import build_model
from utils import load_config


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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config)
    model = build_model(config, device, gradient_checkpointing=False)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.transformer.load_state_dict(state_dict)
    model.transformer.eval()

    images = model.generate(
        args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        guidance_scale=args.guidance,
        scheduler=args.scheduler,
        eta=args.eta,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for i, img_tensor in enumerate(images):
        out_path = Path(args.out_dir) / f"sample_{i:04d}.png"
        to_pil_image(img_tensor.cpu()).save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
