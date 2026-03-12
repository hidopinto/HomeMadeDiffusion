import torch
import weave
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW

from trainer import DiTTrainer
from models import DiT, LatentDiffusion, SinCosPosEmbed2D, Attention, AdaLNZeroStrategy
from diffusion_engine import DiffusionEngine, DDPM


def load_frozen_models(device):
    # SDXL VAE is generally preferred for its improved latent space
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    # CLIP Text Encoder (Standard for most DiT/Stable Diffusion research)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)

    return vae.to(device), text_encoder.to(device), tokenizer


def main():
    weave.init("video-diffusion-research")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Frozen Giants
    vae, text_encoder, tokenizer = load_frozen_models(device)

    # 2. Setup the "Math"
    method = DDPM(learn_sigma=False)
    engine = DiffusionEngine(method=method)

    # 3. Setup 2D Positional Strategy
    # Using 32 for a 256x256 image (8x VAE downscale)
    pos_embedder = SinCosPosEmbed2D(hidden_size=1152, grid_size=32)

    model_core = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        pos_embedder=pos_embedder,
        processor_class=Attention,  # Fixed from None
        conditioner_class=AdaLNZeroStrategy  # Fixed from None
    )

    # Wrap in LatentDiffusion
    model = LatentDiffusion(model_core, vae=vae, text_encoder=text_encoder, engine=engine)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.transformer.parameters(), lr=1e-4, weight_decay=0)

    # 5. Execute
    # TODO: add dataloader
    trainer = DiTTrainer(model, dataloader=None, optimizer=optimizer, lr_scheduler=None)
    trainer.fit(epochs=100)


if __name__ == "__main__":
    main()
