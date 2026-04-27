from __future__ import annotations

__all__ = ['DiT', 'LatentDiffusion', 'Attention', 'CrossAttnDiTBlock']

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Attention

from models.conditioning import TimestepEmbedder
from models.condition_manager import ConditionOutput, ConditionManager
from models.layers import PatchEmbed, FinalLayer


class DiTBlock(nn.Module):
    """Single transformer block with AdaLN-Zero conditioning.

    A shared MLP (the ``conditioner``) projects the combined
    (timestep + text) embedding into 6 per-channel parameters:
    ``shift_msa, scale_msa, gate_msa`` for self-attention and
    ``shift_mlp, scale_mlp, gate_mlp`` for the MLP sub-layer.

    Zero-initialization of the conditioner linear ensures every block
    acts as an identity mapping at the start of training, giving a
    stable gradient signal regardless of depth.

    ``context`` and ``context_mask`` are accepted but ignored in the base
    class — present only so ``CrossAttnDiTBlock`` can override without
    changing the call signature used in ``DiT.forward``.
    """

    def __init__(self, hidden_size: int, processor: nn.Module, conditioner: nn.Module) -> None:
        super().__init__()
        self.conditioner = conditioner  # e.g., AdaLNZeroStrategy
        self.processor = processor      # e.g., timm Attention
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

    def forward(self, x: Tensor, condition: Tensor,
                context: Tensor | None = None,
                context_mask: Tensor | None = None) -> Tensor:
        x_msa, gate_msa, x_mlp, gate_mlp = self.conditioner(self.norm(x), condition)
        x = x + gate_msa.unsqueeze(1) * self.processor(x_msa)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class CrossAttnDiTBlock(DiTBlock):
    """DiTBlock extended with a cross-attention sub-layer.

    Applies cross-attention between patch tokens (Q) and the pre-projected
    context sequence (K/V) after self-attention and before the MLP residual.
    No AdaLN gate on the cross-attn residual — zero-init would silence
    cross-attention at the start of training, defeating its purpose.

    Falls back to base behaviour when ``context`` is None.
    """

    def __init__(self, hidden_size: int, processor: nn.Module,
                 conditioner: nn.Module, cross_attn: nn.Module) -> None:
        super().__init__(hidden_size, processor, conditioner)
        self.cross_attn = cross_attn
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x: Tensor, condition: Tensor,
                context: Tensor | None = None,
                context_mask: Tensor | None = None) -> Tensor:
        x_msa, gate_msa, x_mlp, gate_mlp = self.conditioner(self.norm(x), condition)
        x = x + gate_msa.unsqueeze(1) * self.processor(x_msa)
        if context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context, context_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (Peebles & Xie, 2022) with hybrid AdaLN + Cross-Attention.

    Architecture: patch embedding → sinusoidal positional embedding →
    N × DiTBlock or CrossAttnDiTBlock (AdaLN-Zero) → FinalLayer → unpatchify.

    Conditioning is decoupled from the model: ``DiT`` receives a ``ConditionOutput``
    (pre-projected tensors from ``ConditionManager``) and knows nothing about text
    encoders or projectors. ``adaLN`` is summed into the timestep embedding;
    ``sequences`` are concatenated into a single (context, context_mask) pair
    before the block loop.

    Supports both 2-D images (B, C, H, W) and 3-D videos (B, C, F, H, W)
    controlled by the ``is_video`` flag.

    Optional gradient checkpointing trades compute for VRAM: when enabled,
    activations are recomputed on the backward pass rather than stored.
    ``ConditionOutput`` is consumed before the block loop so ``checkpoint``
    always receives plain tensors.
    """

    def __init__(
            self,
            is_video: bool,
            input_size: int,
            patch_size,
            in_channels: int,
            hidden_size: int,
            frequency_embedding_size: int,
            max_period: int,
            depth: int,
            num_heads: int,
            pos_embedder: nn.Module,
            processor_class: type,
            conditioner_class: type,
            out_channels: int,
            cross_attn_class: type | None = None,
            gradient_checkpointing: bool = False,
            use_reentrant: bool = False,
            compile_blocks: bool = False,
    ) -> None:
        super().__init__()
        self.is_video = is_video
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gradient_checkpointing = gradient_checkpointing
        self.use_reentrant = use_reentrant

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.pos_embedder = pos_embedder
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size, max_period)

        blocks = [
            DiT._build_block(hidden_size, num_heads, processor_class,
                             conditioner_class, cross_attn_class)
            for _ in range(depth)
        ]
        if compile_blocks:
            blocks = [torch.compile(b, mode="default") for b in blocks]
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    @staticmethod
    def _build_block(hidden_size: int, num_heads: int, processor_class: type,
                     conditioner_class: type,
                     cross_attn_class: type | None) -> DiTBlock:
        proc = processor_class(hidden_size, num_heads=num_heads, qkv_bias=True)
        cond = conditioner_class(hidden_size, hidden_size)
        if cross_attn_class is not None:
            return CrossAttnDiTBlock(hidden_size, proc, cond,
                                     cross_attn_class(hidden_size, num_heads))
        return DiTBlock(hidden_size, proc, cond)

    def forward(self, x: Tensor, t: Tensor,
                conditions: ConditionOutput | None = None) -> Tensor:
        """
        x:          (B, C, H, W) or (B, C, F, H, W)
        t:          (B,) timesteps
        conditions: pre-projected ConditionOutput from ConditionManager, or None
        """
        if x.ndim == 5:
            orig_f, orig_h, orig_w = x.shape[2:]
        else:
            orig_f = None
            orig_h, orig_w = x.shape[2:]

        x = self.patch_embed(x)
        x = x + self.pos_embedder(x)

        condition = self.t_embedder(t)
        context: Tensor | None = None
        context_mask: Tensor | None = None

        if conditions is not None:
            if conditions.adaLN is not None:
                condition = condition + conditions.adaLN
            if conditions.sequences:
                context = torch.cat([c for c, _ in conditions.sequences], dim=1)
                context_mask = torch.cat([m for _, m in conditions.sequences], dim=1)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, condition, context, context_mask,
                               use_reentrant=self.use_reentrant)
            else:
                x = block(x, condition, context, context_mask)

        x = self.final_layer(x, condition)

        c = self.out_channels
        if self.is_video:
            p_t, p_h, p_w = self.patch_size
            f_p, h_p, w_p = orig_f // p_t, orig_h // p_h, orig_w // p_w
            x = rearrange(x, 'b (f_p h_p w_p) (c p_t p_h p_w) -> b c (f_p p_t) (h_p p_h) (w_p p_w)',
                          c=c, f_p=f_p, h_p=h_p, w_p=w_p, p_t=p_t, p_h=p_h, p_w=p_w)
        else:
            p_h, p_w = self.patch_size
            h_p, w_p = orig_h // p_h, orig_w // p_w
            x = rearrange(x, 'b (h_p w_p) (c p_h p_w) -> b c (h_p p_h) (w_p p_w)',
                          c=c, h_p=h_p, w_p=w_p, p_h=p_h, p_w=p_w)

        return x


class LatentDiffusion(nn.Module):
    """Full latent diffusion model: frozen encoders + trainable DiT + DiffusionEngine.

    Training path (``forward``):
        Takes pre-encoded latents and pre-encoded text embeddings (both produced
        by ``LatentCachingEngine``). Applies CFG dropout on raw text dicts, then
        projects via ``ConditionManager`` and delegates to ``DiffusionEngine.compute_loss``.
        Only DiT and ConditionManager weights receive gradients.

    Inference path (``generate``):
        Encodes text on-the-fly, runs the reverse-process sampler via
        ``DiffusionEngine.sample``, and decodes latents with the frozen VAE.
    """

    def __init__(self, config, dit_model: DiT, vae, text_encoder, tokenizer,
                 engine, condition_manager: ConditionManager) -> None:
        super().__init__()
        self.config = config

        self.transformer = dit_model
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.engine = engine
        self.condition_manager = condition_manager  # trainable — do NOT freeze

        self._null_hidden_states: torch.Tensor | None = None
        self._null_attention_mask: torch.Tensor | None = None

        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_inputs(self, pixel_values: Tensor, text_prompts) -> tuple[Tensor, dict]:
        text_inputs = self.tokenizer(
            text_prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(pixel_values.device)

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

    def _project(self, text_embeds: dict) -> ConditionOutput:
        return self.condition_manager({"text": text_embeds})

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
        projected = self._project(text_embeds)
        return self.engine.compute_loss(self.transformer, latents, projected)

    @torch.no_grad()
    def encode_text(self, prompts: list[str], device: torch.device) -> dict:
        tokens = self.tokenizer(
            prompts, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).to(device)
        hidden_states = self.text_encoder(tokens.input_ids.to(device))[0]
        return {"hidden_states": hidden_states.float(), "attention_mask": tokens.attention_mask}

    def _cfg_model_fn(self, x_t: Tensor, t: Tensor,
                      cond_proj: ConditionOutput, null_proj: ConditionOutput,
                      guidance_scale: float) -> Tensor:
        eps_u = self.transformer(x_t, t, conditions=null_proj)
        eps_c = self.transformer(x_t, t, conditions=cond_proj)
        in_c = self.config.dit.in_channels
        if self.transformer.out_channels > in_c:
            eps_u, _ = torch.split(eps_u, in_c, dim=1)
            eps_c, _ = torch.split(eps_c, in_c, dim=1)
        return eps_u + guidance_scale * (eps_c - eps_u)

    def _decode_latents(self, latents: Tensor, vae_device: str | None = None) -> Tensor:
        scaled = latents / self.config.dit.vae_scale_factor
        if vae_device:
            self.vae.to(vae_device)
            images = self.vae.decode(scaled.to(device=vae_device, dtype=self.vae.dtype)).sample
            return (images.to(latents.device).clamp(-1.0, 1.0) + 1.0) / 2.0
        images = self.vae.decode(scaled.to(self.vae.dtype)).sample
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        height: int = 512,
        width: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        scheduler: str = "ddim",
        eta: float = 0.0,
        collector: "IntermediateCollector | None" = None,
        progress_fn: "ProgressFn | None" = None,
        vae_device: str | None = None,
    ) -> Tensor:
        device = next(self.transformer.parameters()).device
        cond_proj = self._project(self.encode_text(prompts, device))
        null_proj = self._project(self.encode_text([""] * len(prompts), device))

        model_kwargs = {
            "cond_proj": cond_proj,
            "null_proj": null_proj,
            "guidance_scale": guidance_scale,
        }

        h_lat, w_lat = height // 8, width // 8
        shape = (len(prompts), self.config.dit.in_channels, h_lat, w_lat)
        latents = self.engine.sample(
            self._cfg_model_fn, shape, device,
            num_steps=num_steps, scheduler=scheduler, eta=eta,
            model_kwargs=model_kwargs,
            collector=collector,
            progress_fn=progress_fn,
        )

        if collector is not None:
            collector.decoded_images = [self._decode_latents(lat, vae_device) for lat in collector.latents]
        return self._decode_latents(latents, vae_device)
