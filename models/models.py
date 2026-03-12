from einops import rearrange
from torch import nn
import torch.nn.functional as F

from layers import Attention, Mlp, PatchEmbed, FinalLayer, AdaLNZeroStrategy
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
    def __init__(self, input_size, patch_size, in_channels, hidden_size, depth, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # In a real scenario, y might be class labels (nn.Embedding) or
        # text features (already projected to hidden_size).
        # We'll assume y is already an embedding of 'hidden_size' or None.

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                # Processor: Standard Attention (can be swapped for 3D/Temporal later)
                processor=Attention(hidden_size, num_heads=num_heads, qkv_bias=True),
                # Conditioner: Our modular AdaLNZero strategy
                conditioner=AdaLNZeroStrategy(hidden_size, hidden_size)
            ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W) or (B, C, F, H, W)
        t: (B,) timesteps
        y: (B, D) optional conditioning
        """
        is_video = x.dim() == 5
        if is_video:
            original_f = x.shape[2]
            original_h = x.shape[3]
            original_w = x.shape[4]
        else:
            original_h = x.shape[2]
            original_w = x.shape[3]

        # 1. Patchify
        # x goes from (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)

        # 2. Add Positional Embeddings (Not implemented here, but necessary!)
        x = x + self.pos_embed

        # 3. Create conditioning vector 'c'
        # DiT combines timestep and class information by summing them
        c = self.t_embedder(t)
        if y is not None:
            c = c + y

        # 4. Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 5. Unpatchify
        x = self.final_layer(x, c)  # (B, N, patch_size**2 * out_channels)

        # Inside DiT.forward()
        x = self.final_layer(x, c)  # Result: (B, N, patch_size**2 * out_channels)

        # To compare this to your original pixels/latents, you must do:
        if is_video:
            # (B, (F*H*W), C*P*P) -> (B, C, F, H*P, W*P)
            x = rearrange(x, 'b (f h w) (c p1 p2) -> b c f (h p1) (w p2)',
                          p1=self.patch_size, p2=self.patch_size, f=original_f, h=original_h, w=original_w)
        else:
            # (B, (H*W), C*P*P) -> (B, C, H*P, W*P)
            x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                          p1=self.patch_size, p2=self.patch_size, h=original_h, w=original_w)

        return x


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
