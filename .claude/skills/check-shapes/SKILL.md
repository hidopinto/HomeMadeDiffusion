---
name: check-shapes
description: Runs architecture validation and provides a deep-dive analysis of tensor dimension mismatches.
---

# Check Shapes Skill

Use this skill when the user asks to "check the model architecture" or when a `shape mismatch` error occurs.

## Execution
1. Run the validation script:
   ! python scripts/validate_shapes.py
2. If it fails, capture the traceback.
3. Analyze the mismatch specifically for:
   - **Latent Space**: Are we at `(B, 4, H/8, W/8)` for SDXL?
   - **Temporal Dim**: Is the frame dimension being correctly squeezed?
   - **AdaLN Fusion**: Are the scale/shift parameters matching the DiT hidden size?

## Guidelines
- If a mismatch is found, do not just report it. Propose a `/plan` to fix the specific layer in `models/layers.py` or `models/models.py`.
- Check `config.yaml` to see if `is_video` is True—if so, ensure the `SinCosPosEmbed3D` is receiving 5D tensors `(B, C, T, H, W)`.