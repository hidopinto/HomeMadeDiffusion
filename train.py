import weave
from trainer import DiTTrainer
from models import DiT, LatentDiffusion, SinCosPosEmbed3D
from diffusion_engine import DiffusionEngine, DDPM
from torch.optim import AdamW


def main():
    # 1. Start Weave tracking
    weave.init("video-diffusion-research")

    # 2. Setup the "Math"
    method = DDPM(learn_sigma=False)
    engine = DiffusionEngine(method=method)

    # 3. Setup the DiT with your 3D Positional Strategy
    pos_embedder = SinCosPosEmbed3D(hidden_size=1152, grid_size=32)

    model_core = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        pos_embedder=pos_embedder,
        processor_class=None,  # e.g. Attention
        conditioner_class=None  # e.g. AdaLNZeroStrategy
    )

    # Wrap in LatentDiffusion
    # Note: You'll need to pass your frozen VAE and Text Encoder here
    # TODO: load vae
    # TODO: load rext-encoder
    model = LatentDiffusion(model_core, vae=None, text_encoder=None, engine=engine)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.transformer.parameters(), lr=1e-4, weight_decay=0)

    # 5. Execute
    # TODO: add dataloader
    trainer = DiTTrainer(model, dataloader=None, optimizer=optimizer, lr_scheduler=None)
    trainer.fit(epochs=100)


if __name__ == "__main__":
    main()
