from __future__ import annotations

import torch
from box import Box
from diffusers import AutoencoderKL
from timm.models.vision_transformer import Attention
from transformers import CLIPModel, CLIPTokenizer

from diffusion_engine import DDPM, DiffusionEngine
from models import AdaLNTextProjector, AdaLNZeroStrategy, DiT, LatentDiffusion, SinCosPosEmbed2D, SinCosPosEmbed3D
from samplers import DDIMSampler, DDPMSampler

METHOD_REGISTRY: dict[str, type] = {
    "ddpm": DDPM,
}

SAMPLER_REGISTRY: dict[str, type] = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
}


def load_frozen_models(config: Box, device: str) -> tuple:
    vae = AutoencoderKL.from_pretrained(config.external_models.vae, torch_dtype=torch.bfloat16)
    tokenizer = CLIPTokenizer.from_pretrained(config.external_models.tokenizer)
    clip = CLIPModel.from_pretrained(config.external_models.text_encoder, torch_dtype=torch.bfloat16)
    return vae.to(device), clip.text_model.to(device), tokenizer


def build_model(config: Box, device: str, gradient_checkpointing: bool = False) -> LatentDiffusion:
    vae, text_encoder, tokenizer = load_frozen_models(config, device)
    method_name = config.diffusion.method
    method_cfg = config.diffusion[method_name]
    method = METHOD_REGISTRY[method_name](
        num_timesteps=method_cfg.num_timesteps,
        learn_variance=config.dit.learn_variance,
        beta_start=method_cfg.beta_start,
        beta_end=method_cfg.beta_end,
    )

    sampler = SAMPLER_REGISTRY[config.diffusion.sampler](method)
    engine = DiffusionEngine(method=method, sampler=sampler)

    patch_size = config.dit.patch_size[-1]
    grid_size = config.dit.input_size // patch_size

    if config.general.is_video:
        pos_embedder = SinCosPosEmbed3D(config.dit.hidden_size, grid_size, config.dit.max_frames)
    else:
        pos_embedder = SinCosPosEmbed2D(config.dit.hidden_size, grid_size)

    text_projector = AdaLNTextProjector(config.dit.cond_dim, config.dit.hidden_size)

    model_core = DiT(
        is_video=config.general.is_video,
        input_size=config.dit.input_size,
        patch_size=config.dit.patch_size,
        in_channels=config.dit.in_channels,
        hidden_size=config.dit.hidden_size,
        text_projector=text_projector,
        frequency_embedding_size=config.dit.frequency_embedding_size,
        max_period=config.dit.max_period,
        depth=config.dit.depth,
        num_heads=config.dit.num_heads,
        pos_embedder=pos_embedder,
        processor_class=Attention,
        conditioner_class=AdaLNZeroStrategy,
        learn_variance=config.dit.learn_variance,
        gradient_checkpointing=gradient_checkpointing,
        use_reentrant=config.training.use_reentrant,
    )
    return LatentDiffusion(config, model_core, vae, text_encoder, tokenizer, engine)
