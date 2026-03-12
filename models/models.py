import torch
from einops import rearrange
from torch import nn

from layers import PatchEmbed, FinalLayer, AdaLNZeroStrategy
from models.conditioning import TimestepEmbedder


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, processor, conditioner):
        super().__init__()
        self.conditioner = conditioner  # e.g., AdaLNZeroStrategy
        self.processor = processor  # e.g., SelfAttention or CrossAttention
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, c):
        # 1. Ask the conditioner to prepare the inputs
        x_msa, gate_msa, x_mlp, gate_mlp = self.conditioner(self.norm(x), c)

        # 2. Apply Attention and MLP with their respective gates
        x = x + gate_msa.unsqueeze(1) * self.processor(x_msa)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class DiT(nn.Module):
    def __init__(
            self,
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            pos_embedder,
            processor_class,  # e.g., Attention from timm
            conditioner_class  # e.g., AdaLNZeroStrategy
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.input_size = input_size  # e.g., 32 for 256px images with VAE (8x downscale)

        # 1. Embedders
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embedder = pos_embedder  # <--- New Strategy Class
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Linear(hidden_size, hidden_size)

        # 2. Transformer Blocks (Stays the same)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                processor=processor_class(hidden_size, num_heads=num_heads, qkv_bias=True),
                conditioner=conditioner_class(hidden_size, hidden_size)
            ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W) or (B, C, F, H, W)
        t: (B,) timesteps
        y: (B, D) condition (already projected or class-embedded)
        """
        is_video = x.dim() == 5
        b, c = x.shape[0], x.shape[1]

        if is_video:
            f, h, w = x.shape[2:]
        else:
            h, w = x.shape[2:]

        # 1. Patchify & Position
        x = self.patch_embed(x)

        # 2. Add Positional Embeddings via Strategy
        # The embedder now handles the logic of 2D vs 3D repeat/interpolation
        x = x + self.pos_embedder(x)

        # 3. Conditioning
        c = self.t_embedder(t)
        if y is not None:
            c = c + self.y_embedder(y)

        # 4. Blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Final Projection & Unpatchify
        x = self.final_layer(x, c)

        p = self.patch_size
        if is_video:
            x = rearrange(x, 'b (f h w) (c p1 p2) -> b c f (h p1) (w p2)',
                          p1=p, p2=p, f=f, h=h // p, w=w // p)
        else:
            x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                          p1=p, p2=p, h=h // p, w=w // p)
        return x


class LatentDiffusion(nn.Module):
    def __init__(self, dit_model, vae, text_encoder, engine):
        super().__init__()
        self.transformer = dit_model
        self.vae = vae
        self.text_encoder = text_encoder
        self.engine = engine  # This replaces 'scheduler' for training logic

        # Freeze the giants
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def process_input(self, pixel_values, text_input):
        # 1. Image -> Latent (scaled by VAE factor, usually 0.18215)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215

        # 2. Text -> Embedding
        # Assuming text_input is already tokenized or handled by text_encoder
        encoder_hidden_states = self.text_encoder(text_input)[0]

        return latents, encoder_hidden_states

    def forward(self, pixel_values, text_input):
        # This is what your Trainer calls every step
        latents, cond = self.process_input(pixel_values, text_input)

        # The engine handles the noise math (DDPM or Flow)
        # and returns the MSE loss
        loss = self.engine.compute_loss(self.transformer, latents, cond)
        return loss
