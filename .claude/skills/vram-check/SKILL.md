---
name: vram-check
description: Checks NVIDIA GPU status and VRAM availability to prevent OOM before training.
---

# VRAM Check Skill

When this skill is invoked via `/vram-check`, Claude will:

1. Run `nvidia-smi` to check current VRAM usage.
2. Report the available memory on the RTX 3090.
3. If usage is > 20%, identify which processes (e.g., Xorg, Chrome) are consuming it.
4. Provide a recommendation on whether the current project's `config.yaml` batch size will fit.

## Implementation Script
! nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits

## Guidelines
- If free VRAM is < 16GB, warn the user that Video Diffusion training might fail.
- Always suggest closing browser tabs (like the Claude PWA) if VRAM is tight.