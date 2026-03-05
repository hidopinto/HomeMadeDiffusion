from torch import nn
import torch.nn.functional as F

from layers import Attention, Mlp, PatchEmbed, FinalLayer


class DiTBlock(nn.Module):
    # TODO: Implement Spatio-Temporal Attention. For video,
    #  we'll need to decide between 'Joint Space-Time' or 'Factorized' (Spatial then Temporal) attention.
    pass


class DiT(nn.Module):
    """
    TODO:
    - Support 3D Attention: For video, we will need to swap 2D spatial attention
      for (Spatial + Temporal) attention blocks.
    - Implement 'adaLN-Zero' for timestep/label conditioning.
    """
    def __init__(self, input_size, patch_size, in_channels, hidden_size, depth, num_heads):
        super().__init__()
        # TODO: Initialize PatchEmbed, TimestepEmbedder, and nn.ModuleList of DiTBlocks
        pass

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W) latents
        t: (B,) diffusion timesteps
        y: (B,) conditional info (CLIP embeddings or class labels)
        """
        pass


class LatentDiffusion(nn.Module):
    def __init__(self, dit_model, vae, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.transformer = dit_model  # Your DiT
        self.vae = vae  # Frozen VAE
        self.text_encoder = text_encoder  # Frozen CLIP/T5
        self.scheduler = scheduler  # Noise schedule (DDPM/DDIM/Flow)

        # Freeze the giants
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    def process_input(self, images, prompt):
        # 1. Image -> Latent
        # 2. Text -> Embedding
        # 3. Add Noise to Latents
        pass

    def forward(self, x, t, cond):
        # The actual training step
        return self.transformer(x, t, cond)
