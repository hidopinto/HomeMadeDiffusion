import torch
import weave
import yaml
from box import Box
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.optim import AdamW

from trainer import DiTTrainer
from models import DiT, LatentDiffusion, SinCosPosEmbed2D, Attention, AdaLNZeroStrategy
from diffusion_engine import DiffusionEngine, DDPM


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return Box(yaml.safe_load(f))


def load_frozen_models(config, device):
    # SDXL VAE is generally preferred for its improved latent space
    vae = AutoencoderKL.from_pretrained(config.external_models.vae, torch_dtype=torch.float16)

    # CLIP Text Encoder (Standard for most DiT/Stable Diffusion research)
    tokenizer = CLIPTokenizer.from_pretrained(config.external_models.tokenizer)
    text_encoder = CLIPTextModel.from_pretrained(config.external_models.text_encoder, torch_dtype=torch.float16)

    return vae.to(device), text_encoder.to(device), tokenizer


def main():
    config = load_config(config_path="config.yaml")

    weave.init(config.general.wnb_project_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Frozen Giants
    vae, text_encoder, tokenizer = load_frozen_models(config, device)

    # 2. Setup the "Math"
    method = DDPM(learn_variance=config.dit.learn_variance)
    engine = DiffusionEngine(method=method)

    # 3. Setup 2D Positional Strategy
    pos_embedder = SinCosPosEmbed2D(hidden_size=config.dit.hidden_size, grid_size=config.dit.grid_size)

    model_core = DiT(
        input_size=config.dit.input_size,
        patch_size=config.dit.patch_size,
        in_channels=config.dit.in_channels,
        hidden_size=config.dit.hidden_size,
        cond_dim=config.dit.cond_dim,
        frequency_embedding_size=config.dit.frequency_embedding_size,
        max_period=config.dit.max_period,
        depth=config.dit.depth,
        num_heads=config.dit.num_heads,
        pos_embedder=pos_embedder,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        learn_variance=config.dit.learn_variance,
        gradient_checkpointing=config.training.gradient_checkpointing,
        use_reentrant=config.training.use_reentrant,
    )

    # Wrap in LatentDiffusion
    model = LatentDiffusion(config=config, dit_model=model_core, vae=vae, text_encoder=text_encoder, engine=engine)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.transformer.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)

    # 5. Execute
    # TODO: add dataloader
    trainer = DiTTrainer(config=config, model=model, dataloader=None, optimizer=optimizer, lr_scheduler=None)
    trainer.fit(epochs=config.training.epochs)


if __name__ == "__main__":
    main()
