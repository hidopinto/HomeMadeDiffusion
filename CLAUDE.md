# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Latent Diffusion model** implementation using a **Diffusion Transformer (DiT)** architecture for text-conditional image generation (with optional video support). 
It targets RTX 3090 GPUs and uses SDXL-compatible latent space.

## Package Manager

This project uses **`uv`** for dependency management. Use `uv add <pkg>` (not `pip install`).

## Commands

```bash
# Run training
python train.py

# Run full test suite
pytest tests/ -v

# Run only shape tests (fast architecture check)
pytest tests/ -v -k "shape"

# Run overfit smoke test
pytest tests/test_overfit.py -v

# Smoke-test all eval metrics before a training run
python scripts/check_eval.py

# Check GPU  (Skill)
python /vram-check

# Check ViT shapes  (Skill)
python /check-shapes
```

## Testing

Tests live in `tests/` and use a tiny config (hidden_size=128, depth=2, 16×16 latent grid) so the full suite runs in < 30 s on CPU.

| File | Coverage |
|------|----------|
| `tests/test_dit.py` | DiT forward shapes, PatchEmbed, FinalLayer, AdaLN, positional embedders, timestep embedder |
| `tests/test_ddpm.py` | Noise schedule, q_sample, loss (MSE + VLB) |
| `tests/test_samplers.py` | DDPM and DDIM step shapes, loop output, DDIM determinism; FM step/loop/determinism |
| `tests/test_flow_matching.py` | FlowMatching q_sample boundaries, interpolation, loss, OT reordering correctness |
| `tests/test_latent_diffusion.py` | encode_text shapes, forward loss scalar, gradient presence |
| `tests/test_overfit.py` | Gradient-flow smoke test: loss drops ≥5% in 10 steps for both 2D and 3D |
| `tests/test_eval_metrics.py` | `_CLIPModelWrapper` isolation: tensor extraction, `.norm()` compat, device movement, passthrough |

## Hardware Constraints
- **VRAM Limit**: 24GB. 
- **Precision**: Use `bf16` or `fp16` mixed precision for all training.
- **Memory**: Use Gradient Checkpointing; limit video batch size to 1-2.
- **Safety**: Do not exceed 90% GPU utility to prevent system instability.

## Style & Workflow
- **Rules**: Use `einops` for tensor manipulation; avoid `view`/`reshape`.
- **Types**: Mandatory Type Hints for all PyTorch modules.
- **Planning**: For any multi-file change, ask for a `/plan` first.
- **No nested functions**: Never define a function inside another function. Use class methods, `functools.partial`, or `model_kwargs` dicts to pass context instead.

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
| `diffusion_engine.py` | `DDPM` — noise schedule, `q_sample()`, `calc_vlb_loss()`; `FlowMatching` — OT conditional flow, MSE velocity loss, optional minibatch OT; `DiffusionEngine` wraps either method with DiT |
| `samplers.py` | `DDPMSampler`, `DDIMSampler`, `FlowMatchingSampler` — Euler ODE from noise→data |
| `models/models.py` | `DiT` — pure transformer; `LatentDiffusion` — training wrapper; `DiTBlock` — AdaLN-Zero block; `CrossAttnDiTBlock` — adds cross-attention |
| `models/condition_manager.py` | `ConditionOutput` — pre-projected conditions dataclass; `ConditionManager` — routes encoder outputs to AdaLN / cross-attn paths |
| `models/projectors.py` | `AdaLNTextProjector` (role=global, pools → AdaLN); `CrossAttnTextProjector` (role=sequence, preserves sequence) |
| `models/cross_attention.py` | `CrossAttention` — SDPA-based multi-head cross-attention (Q from patches, K/V from context) |
| `models/layers.py` | `masked_mean_pool`, `PatchEmbed`, `FinalLayer`, `AdaLNZeroStrategy` |
| `models/conditioning.py` | `TimestepEmbedder`, `SinCosPosEmbed2D`, `SinCosPosEmbed3D` |
| `data/` | `LatentDataset`, `LatentCachingEngine`, `build_dataloader` — fully implemented |
| `evaluation/metrics.py` | `EvaluationEngine` — FID / IS / CLIPScore every `eval_every_steps`; real FID stats cached to disk; `_CLIPModelWrapper` bridges transformers 5.x API |
| `config.yaml` | All hyperparameters (model, training, external model IDs) |

### DiT Conditioning (Hybrid AdaLN + Cross-Attention)

`ConditionManager` projects raw encoder outputs and routes them to `ConditionOutput`:
- `adaLN` (global): pooled text → added to timestep embedding, modulates all blocks via AdaLN-Zero scale/shift/gate.
- `sequences` (cross-attn): full text token sequence → concatenated and attended by `CrossAttnDiTBlock`.

`DiT` is a pure transformer — it accepts `ConditionOutput` and knows nothing about encoders or projectors. `LatentDiffusion` owns the `ConditionManager` and calls `_project()` before passing to the engine. CFG dropout operates on raw text dicts before projection.

`CrossAttnDiTBlock` extends `DiTBlock` with a cross-attention sub-layer (after self-attn, before MLP). Enable with `dit.cross_attention: True` in config.

### 2D vs 3D

`config.general.is_video` toggles between `SinCosPosEmbed2D` (images) and `SinCosPosEmbed3D` (videos with separable temporal + spatial embeddings).

## Configuration

`config.yaml` is loaded via `python-box` for dot-notation access. Key sections:

- `external_models`: HuggingFace model IDs for VAE, tokenizer, text encoder
- `dit`: architecture hyperparameters (patch size, hidden size, depth, heads, etc.)
- `training`: lr, epochs, mixed precision, gradient checkpointing
- `general`: `is_video`, W&B project name
- `evaluation`: `clip_model_name` — HuggingFace model ID used by `CLIPScore`

## Known TODOs / Incomplete Areas

- `hydra-core`, `deepspeed`, `optimum` — declared as dependencies but unused

## Maintenance Rule

After any interface change to a module (new argument, renamed class, changed data flow), update the Architecture table above and any relevant skill `SKILL.md` files in `.claude/skills/`.
