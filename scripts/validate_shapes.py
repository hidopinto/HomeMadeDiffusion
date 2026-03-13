import torch
from box import Box
from models import DiT, SinCosPosEmbed2D, Attention, AdaLNZeroStrategy


def validate_shapes():
    # 1. Mock Config (Matching your 'Simple' 3090 setup)
    config_dict = {
        "dit": {
            "input_size": 32,
            "patch_size": [2, 2],
            "in_channels": 4,
            "hidden_size": 768,
            "cond_dim": 768,
            "frequency_embedding_size": 256,
            "max_period": 10000,
            "depth": 4,
            "num_heads": 8,
            "learn_variance": True  # Testing the "Comment 3" bug fix
        },
        "general": {"is_video": False},
        "training": {"gradient_checkpointing": False, "use_reentrant": False}
    }
    config = Box(config_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_size = config.dit.input_size // config.dit.patch_size[-1]

    # 2. Initialize Model Parts
    pos_embedder = SinCosPosEmbed2D(config.dit.hidden_size, grid_size).to(device)

    model = DiT(
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
        is_video=config.general.is_video
    ).to(device)

    # 3. Create Dummy Inputs
    batch_size = 2
    # Simulated SDXL Latents: (B, C, H, W)
    x = torch.randn(batch_size, 4, 32, 32).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    y = torch.randn(batch_size, 768).to(device)  # Pooled CLIP cond

    print(f"Input Shape: {x.shape}")

    # 4. Forward Pass
    with torch.no_grad():
        output = model(x, t, y)

    # 5. Verification
    expected_channels = 8 if config.dit.learn_variance else 4
    expected_shape = (batch_size, expected_channels, 32, 32)

    print(f"Output Shape: {output.shape}")

    if output.shape == expected_shape:
        print("✅ SUCCESS: Output shape matches expectations.")
    else:
        print(f"❌ ERROR: Expected {expected_shape}, got {output.shape}")


if __name__ == "__main__":
    validate_shapes()
