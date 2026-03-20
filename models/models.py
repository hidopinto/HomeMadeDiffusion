__all__ = ['DiT', 'LatentDiffusion', 'Attention']

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Attention

from models.conditioning import TimestepEmbedder
from models.layers import PatchEmbed, FinalLayer, AdaLNTextProjector


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
            text_projector: AdaLNTextProjector,
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
        self.input_size = input_size  # e.g., 32 for 256px images with VAE (8x downscale)
        self.learn_variance = learn_variance
        self.gradient_checkpointing = gradient_checkpointing
        self.use_reentrant = use_reentrant

        # 1. Embedders
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embedder = pos_embedder
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size, max_period)
        self.text_projector = text_projector

        # 2. Transformer Blocks (Stays the same)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                processor=processor_class(hidden_size, num_heads=num_heads, qkv_bias=True),
                conditioner=conditioner_class(hidden_size, hidden_size)
            ) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels, learn_variance=learn_variance)

    def forward(self, x, t, y: dict | None = None):
        """
        x: (B, C, H, W) or (B, C, F, H, W)
        t: (B,) timesteps
        y: dict with "hidden_states" (B, T, D) and "attention_mask" (B, T)
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
            condition = condition + self.text_projector(y)

        # 3. Blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, condition, use_reentrant=self.use_reentrant)
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

        self._null_hidden_states: torch.Tensor | None = None
        self._null_attention_mask: torch.Tensor | None = None

        # Freeze the giants
        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_inputs(self, pixel_values: Tensor, text_prompts) -> tuple[Tensor, dict]:
        # 1. Handle Tokenization inside the model to keep train.py clean
        text_inputs = self.tokenizer(
            text_prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(pixel_values.device)

        # 2. Match VAE/Encoder dtypes (bf16 for RTX 3090)
        pixel_values = pixel_values.to(self.vae.dtype)

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.config.dit.vae_scale_factor

        hidden_states = self.text_encoder(text_inputs.input_ids)[0]
        text_embeds = {
            "hidden_states": hidden_states.float(),
            "attention_mask": text_inputs.attention_mask,
        }

        return latents.float(), text_embeds

    @torch.no_grad()
    def cache_null_embed(self, device: torch.device) -> None:
        null = self.encode_text([""], device)
        self._null_hidden_states = null["hidden_states"].detach()
        self._null_attention_mask = null["attention_mask"].detach()

    def _get_null_embed(self, batch_size: int, device: torch.device) -> dict:
        assert self._null_hidden_states is not None, "Call cache_null_embed() before training"
        return {
            "hidden_states": self._null_hidden_states.expand(batch_size, -1, -1).to(device),
            "attention_mask": self._null_attention_mask.expand(batch_size, -1).to(device),
        }

    def forward(self, latents: Tensor, text_embeds: dict) -> Tensor:
        if self.training:
            cfg_p = getattr(self.config.training, 'cfg_dropout_prob', 0.0)
            if cfg_p > 0.0:
                null = self._get_null_embed(latents.shape[0], latents.device)
                mask = torch.rand(latents.shape[0], device=latents.device) < cfg_p
                text_embeds = {
                    "hidden_states": torch.where(
                        mask[:, None, None],
                        null["hidden_states"],
                        text_embeds["hidden_states"]
                    ),
                    "attention_mask": torch.where(
                        mask[:, None],
                        null["attention_mask"],
                        text_embeds["attention_mask"]
                    ),
                }
        loss = self.engine.compute_loss(self.transformer, latents, text_embeds)
        return loss

    @torch.no_grad()
    def encode_text(self, prompts: list[str], device: torch.device) -> dict:
        tokens = self.tokenizer(
            prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        hidden_states = self.text_encoder(tokens.input_ids.to(device))[0]
        return {"hidden_states": hidden_states.float(), "attention_mask": tokens.attention_mask}

    def _cfg_model_fn(self, x_t: Tensor, t: Tensor, cond_embeds: dict,
                      null_embeds: dict, guidance_scale: float) -> Tensor:
        eps_u = self.transformer(x_t, t, null_embeds)
        eps_c = self.transformer(x_t, t, cond_embeds)
        if self.transformer.learn_variance:
            eps_u, _ = torch.split(eps_u, self.config.dit.in_channels, dim=1)
            eps_c, _ = torch.split(eps_c, self.config.dit.in_channels, dim=1)
        return eps_u + guidance_scale * (eps_c - eps_u)

    def _decode_latents(self, latents: Tensor) -> Tensor:
        scaled = latents / self.config.dit.vae_scale_factor
        images = self.vae.decode(scaled.to(self.vae.dtype)).sample
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        height: int = 256,
        width: int = 256,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        scheduler: str = "ddim",
        eta: float = 0.0,
        collector: "IntermediateCollector | None" = None,
    ) -> Tensor:
        device = next(self.transformer.parameters()).device
        cond_embeds = self.encode_text(prompts, device)
        null_embeds = self.encode_text([""] * len(prompts), device)

        model_kwargs = {
            "cond_embeds": cond_embeds,
            "null_embeds": null_embeds,
            "guidance_scale": guidance_scale,
        }

        h_lat, w_lat = height // 8, width // 8
        shape = (len(prompts), self.config.dit.in_channels, h_lat, w_lat)
        latents = self.engine.sample(
            self._cfg_model_fn, shape, device,
            num_steps=num_steps, scheduler=scheduler, eta=eta,
            model_kwargs=model_kwargs,
            collector=collector,
        )

        if collector is not None:
            collector.decoded_images = [self._decode_latents(lat) for lat in collector.latents]
        return self._decode_latents(latents)
