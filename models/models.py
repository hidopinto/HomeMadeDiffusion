__all__ = ['DiT', 'LatentDiffusion', 'Attention']

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Attention

from layers import PatchEmbed, FinalLayer
from models.conditioning import TimestepEmbedder


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, processor, conditioner):
        super().__init__()
        self.conditioner = conditioner  # e.g., AdaLNZeroStrategy
        self.processor = processor  # e.g., SelfAttention or CrossAttention
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x, condition):
        # 1. Ask the conditioner to prepare the inputs
        x_msa, gate_msa, x_mlp, gate_mlp = self.conditioner(self.norm(x), condition)

        # 2. Apply Attention and MLP with their respective gates
        x = x + gate_msa.unsqueeze(1) * self.processor(x_msa)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class DiT(nn.Module):
    def __init__(
            self,
            is_video,
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            cond_dim,
            frequency_embedding_size,
            max_period,
            depth,
            num_heads,
            pos_embedder,
            processor_class,  # e.g., Attention from timm
            conditioner_class,  # e.g., AdaLNZeroStrategy
            learn_variance,
            gradient_checkpointing=False,
            use_reentrant=False
    ):
        super().__init__()
        self.is_video = is_video
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.input_size = input_size  # e.g., 32 for 256px images with VAE (8x downscale)
        self.learn_variance = learn_variance
        self.gradient_checkpointing = gradient_checkpointing
        self.use_reentrant = use_reentrant

        # 1. Embedders
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embedder = pos_embedder
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size, max_period)
        self.y_embedder = nn.Linear(cond_dim, hidden_size)

        # 2. Transformer Blocks (Stays the same)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                processor=processor_class(hidden_size, num_heads=num_heads, qkv_bias=True),
                conditioner=conditioner_class(hidden_size, hidden_size)
            ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels, learn_variance=learn_variance)

    def forward(self, x, t, y=None):
        """
        x: (B, C, H, W) or (B, C, F, H, W)
        t: (B,) timesteps
        y: (B, D) condition (already projected or class-embedded)
        """
        # Capture original dimensions before patch_embed flattens them
        if x.ndim == 5:
            orig_f, orig_h, orig_w = x.shape[2:]
        else:
            orig_f = None
            orig_h, orig_w = x.shape[2:]

        # 1. Patchify & Position
        x = self.patch_embed(x)
        x = x + self.pos_embedder(x)

        # 2. Conditioning
        condition = self.t_embedder(t)
        if y is not None:
            condition = condition + self.y_embedder(y)

        # 3. Blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, condition, self.use_reentrant)
            else:
                x = block(x, condition)

        # 4. Final Projection
        x = self.final_layer(x, condition)

        # 5. Unpatchify for Mean + Variance
        v = 2 if self.learn_variance else 1
        c = self.in_channels

        if self.is_video:
            p_t, p_h, p_w = self.patch_size
            f_p, h_p, w_p = orig_f // p_t, orig_h // p_h, orig_w // p_w

            x = rearrange(x, 'b (f_p h_p w_p) (v c p_t p_h p_w) -> b (v c) (f_p p_t) (h_p p_h) (w_p p_w)',
                          v=v, c=c, f_p=f_p, h_p=h_p, w_p=w_p, p_t=p_t, p_h=p_h, p_w=p_w)
        else:
            p_h, p_w = self.patch_size
            h_p, w_p = orig_h // p_h, orig_w // p_w

            x = rearrange(x, 'b (h_p w_p) (v c p_h p_w) -> b (v c) (h_p p_h) (w_p p_w)',
                          v=v, c=c, h_p=h_p, w_p=w_p, p_h=p_h, p_w=p_w)

        return x


class LatentDiffusion(nn.Module):
    def __init__(self, config, dit_model, vae, text_encoder, tokenizer, engine):
        super().__init__()
        self.config = config

        self.transformer = dit_model
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.engine = engine  # This replaces 'scheduler' for training logic

        # Freeze the giants
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    # models.py - LatentDiffusion.process_input
    @torch.no_grad()
    def process_input(self, pixel_values, text_prompts):
        # 1. Handle Tokenization inside the model to keep train.py clean
        text_inputs = self.tokenizer(
            text_prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(pixel_values.device)

        # 2. Match VAE/Encoder dtypes (bf16 for RTX 3090)
        pixel_values = pixel_values.to(self.vae.dtype)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.config.dit.vae_scale_factor

        encoder_hidden_states = self.text_encoder(text_inputs.input_ids)[0]
        pooled_cond = encoder_hidden_states.mean(dim=1)

        # 3. Ensure latents and cond are float32 for the DiT core if not using
        # full bf16 training, or let Accelerator handle it.
        return latents.float(), pooled_cond.float()

    def forward(self, pixel_values, text_input):
        # This is what your Trainer calls every step
        latents, cond = self.process_input(pixel_values, text_input)

        # The engine handles the noise math (DDPM or Flow)
        loss = self.engine.compute_loss(self.transformer, latents, cond)
        return loss
