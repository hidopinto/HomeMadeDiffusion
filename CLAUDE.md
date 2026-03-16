# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Latent Diffusion model** implementation using a **Diffusion Transformer (DiT)** architecture for text-conditional image generation (with optional video support). 
It targets RTX 3090 GPUs and uses SDXL-compatible latent space.

## Commands

```bash
# Run training
python train.py

# Validate tensor shapes (smoke test for architecture)
python scripts/validate_shapes.py

# Check GPU  (Skill)
python /vram-check

# Check ViT shapes  (Skill)
python /check-shapes
```

No formal test suite exists — `scripts/validate_shapes.py` is the primary correctness check.

## Hardware Constraints
- **VRAM Limit**: 24GB. 
- **Precision**: Use `bf16` or `fp16` mixed precision for all training.
- **Memory**: Use Gradient Checkpointing; limit video batch size to 1-2.
- **Safety**: Do not exceed 90% GPU utility to prevent system instability.

## Style & Workflow
- **Rules**: Use `einops` for tensor manipulation; avoid `view`/`reshape`.
- **Types**: Mandatory Type Hints for all PyTorch modules.
- **Planning**: For any multi-file change, ask for a `/plan` first.

## Architecture

### Data Flow

```
text + pixel values
  → frozen CLIP Tokenizer → frozen CLIP Text Encoder → encoder_hidden_states (B, seq, 768)
  → frozen SDXL VAE.encode() → latents (B, 4, H, W) × 0.18215

latents + timestep t + encoder_hidden_states
  → DiT → predicted noise ε (or ε + variance)
  → DDPM.loss() → MSE + 0.001 × VLB
  → only DiT weights updated
```

### Key Components

| File | Responsibility |
|------|---------------|
| `train.py` | Entry point: loads config, frozen models, wires everything together |
| `trainer.py` | `DiTTrainer` — training loop using HuggingFace `Accelerator` (bf16, gradient checkpointing, W&B logging) |
| `diffusion_engine.py` | `DDPM` — noise schedule, `q_sample()`, `calc_vlb_loss()`; `DiffusionEngine` wraps DDPM with DiT |
| `models/models.py` | `DiT` — full transformer; `LatentDiffusion` — training wrapper; `DiTBlock` — AdaLN-Zero block |
| `models/layers.py` | `PatchEmbed`, `FinalLayer`, `AdaLNZeroStrategy` |
| `models/conditioning.py` | `TimestepEmbedder`, `SinCosPosEmbed2D`, `SinCosPosEmbed3D` |
| `data.py` | `LatentDataset` skeleton — **not implemented yet** |
| `config.yaml` | All hyperparameters (model, training, external model IDs) |

### DiT Block (AdaLN-Zero)

Each `DiTBlock` uses AdaLN-Zero conditioning: a shared MLP projects `(timestep_emb + text_emb)` into scale/shift/gate factors applied to both self-attention and MLP sub-layers. No cross-attention — text is fused via AdaLN.

### 2D vs 3D

`config.general.is_video` toggles between `SinCosPosEmbed2D` (images) and `SinCosPosEmbed3D` (videos with separable temporal + spatial embeddings).

## Configuration

`config.yaml` is loaded via `python-box` for dot-notation access. Key sections:

- `external_models`: HuggingFace model IDs for VAE, tokenizer, text encoder
- `dit`: architecture hyperparameters (patch size, hidden size, depth, heads, etc.)
- `training`: lr, epochs, mixed precision, gradient checkpointing
- `general`: `is_video`, W&B project name

## Known TODOs / Incomplete Areas

- Inference/sampling pipeline — not implemented
- `hydra-core`, `deepspeed`, `optimum` — declared as dependencies but unused
