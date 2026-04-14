from __future__ import annotations

import logging

import torch
from box import Box
from diffusers import AutoencoderKL
from timm.models.vision_transformer import Attention
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer

__all__ = ["load_frozen_models", "build_model", "METHOD_REGISTRY", "SAMPLER_REGISTRY"]

from diffusion import DDPM, FlowMatching, DiffusionEngine
from diffusion.samplers import DDIMSampler, DDPMSampler, FlowMatchingSampler
from models import (AdaLNTextProjector, CrossAttnTextProjector, CrossAttention,
                    AdaLNZeroStrategy, ConditionManager,
                    DiT, LatentDiffusion, SinCosPosEmbed2D, SinCosPosEmbed3D)

logger = logging.getLogger(__name__)

METHOD_REGISTRY: dict[str, type] = {
    "ddpm": DDPM,
    "flow_matching": FlowMatching,
}

SAMPLER_REGISTRY: dict[str, type] = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "flow_matching": FlowMatchingSampler,
}


def load_frozen_models(config: Box, device: str) -> tuple[AutoencoderKL, CLIPTextModel, CLIPTokenizer]:
    logger.info("Loading frozen models (VAE + CLIP text encoder)...")
    logger.info("  Loading VAE (%s)...", config.external_models.vae)
    vae = AutoencoderKL.from_pretrained(config.external_models.vae, torch_dtype=torch.bfloat16)
    logger.info("  VAE loaded.")
    logger.info("  Loading CLIP tokenizer (%s)...", config.external_models.tokenizer)
    tokenizer = CLIPTokenizer.from_pretrained(config.external_models.tokenizer)
    logger.info("  CLIP tokenizer loaded.")
    logger.info("  Loading CLIP text encoder (%s)...", config.external_models.text_encoder)
    _clip = CLIPModel.from_pretrained(config.external_models.text_encoder, torch_dtype=torch.bfloat16)
    text_encoder = _clip.text_model
    del _clip.vision_model
    logger.info("  CLIP text encoder loaded.")
    logger.info("  Moving frozen models to %s...", device)
    vae_on_device = vae.to(device)
    text_encoder_on_device = text_encoder.to(device)
    logger.info("  Frozen models on device.")
    return vae_on_device, text_encoder_on_device, tokenizer


def build_model(config: Box, device: str, gradient_checkpointing: bool = False) -> LatentDiffusion:
    vae, text_encoder, tokenizer = load_frozen_models(config, device)

    method  = METHOD_REGISTRY[config.diffusion.method].from_config(config)
    logger.info(
        "Building DiT: depth=%d, hidden_size=%d, method=%s, sampler=%s",
        config.dit.depth, config.dit.hidden_size,
        config.diffusion.method, config.diffusion.sampler,
    )
    sampler = SAMPLER_REGISTRY[config.diffusion.sampler].from_config(config, method)
    engine  = DiffusionEngine(method=method, sampler=sampler)

    in_channels  = config.dit.in_channels
    out_channels = method.expected_out_channels(in_channels)

    patch_size = config.dit.patch_size[-1]
    grid_size  = config.dit.input_size // patch_size

    if config.general.is_video:
        pos_embedder = SinCosPosEmbed3D(config.dit.hidden_size, grid_size, config.dit.max_frames)
    else:
        pos_embedder = SinCosPosEmbed2D(config.dit.hidden_size, grid_size)

    use_cross_attn = getattr(config.dit, "cross_attention", False)

    projector_pairs = [
        ("text", AdaLNTextProjector(config.dit.cond_dim, config.dit.hidden_size)),
    ]
    if use_cross_attn:
        projector_pairs.append(
            ("text", CrossAttnTextProjector(config.dit.cond_dim, config.dit.hidden_size))
        )
    # source_keys are a structural config, not saved in state_dict.
    # Checkpoint reconstruction requires the same projector_pairs order here.
    condition_manager = ConditionManager(projector_pairs).to(device)

    model_core = DiT(
        is_video=config.general.is_video,
        input_size=config.dit.input_size,
        patch_size=config.dit.patch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_size=config.dit.hidden_size,
        frequency_embedding_size=config.dit.frequency_embedding_size,
        max_period=config.dit.max_period,
        depth=config.dit.depth,
        num_heads=config.dit.num_heads,
        pos_embedder=pos_embedder,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        cross_attn_class=CrossAttention if use_cross_attn else None,
        gradient_checkpointing=gradient_checkpointing,
        use_reentrant=config.training.use_reentrant,
    )
    num_params = sum(p.numel() for p in model_core.parameters())
    logger.info("DiT built: %.2fM trainable parameters.", num_params / 1e6)
    logger.info("  Moving DiT to %s...", device)
    model = LatentDiffusion(config, model_core.to(device), vae, text_encoder, tokenizer,
                            engine, condition_manager=condition_manager)
    logger.info("  DiT on device.")
    return model
