import numpy as np
from torch import nn, Tensor
from einops import rearrange  # Essential for "patchifying" and video tensor reshapes

__all__ = [
    "masked_mean_pool",
    "PatchEmbed",
    "AdaLNZeroStrategy",
    "FinalLayer",
]


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Masked mean pool over token dim. hidden_states: (B, T, D), mask: (B, T)"""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class PatchEmbed(nn.Module):
    """Patchify spatial (or spatio-temporal) input and project to embedding space.

    Handles both 4-D image tensors (B, C, H, W) and 5-D video tensors
    (B, C, F, H, W) via ``einops.rearrange``. The rearrange is equivalent
    to a strided convolution but avoids ``view``/``reshape`` and makes the
    patch layout explicit. The final projection is a single ``nn.Linear``
    rather than ``Conv2d`` since the patch flattening is already done by
    ``rearrange``.
    """

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        # patch_size should be (t, h, w)
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size, patch_size)
        self.proj = nn.Linear(in_chans * np.prod(self.patch_size), embed_dim)

    def forward(self, x):
        if x.ndim == 5: # Video (B, C, F, H, W)
            pt, ph, pw = self.patch_size
            x = rearrange(x, 'b c (f pt) (h ph) (w pw) -> b (f h w) (c pt ph pw)',
                          pt=pt, ph=ph, pw=pw)
        else: # Image (B, C, H, W)
            ph, pw = self.patch_size
            x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (c ph pw)',
                          ph=ph, pw=pw)
        return self.proj(x)


class AdaLNZeroStrategy(nn.Module):
    """AdaLN-Zero conditioning strategy for a single DiT block.

    Projects the block condition (timestep + text embedding, shape (B, hidden_size))
    into 6 per-channel modulation parameters via a single SiLU→Linear layer:
    ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp``.

    The zero-initialization of the linear layer means all parameters start at
    zero, so every block begins as an identity function. This provides a
    well-conditioned starting point for gradient descent regardless of depth.
    """

    def __init__(self, hidden_size, c):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(c, 6 * hidden_size)

        # Initialize to zero for identity behavior at start
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, condition):
        # Generate the 6 parameters
        params = self.linear(self.silu(condition)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params

        # This is where 'modulate' lived. Now it's internal logic.
        # We return the two modulated versions of x (for Attn and MLP) and the gates
        res_msa = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        res_mlp = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        return res_msa, gate_msa, res_mlp, gate_mlp


class FinalLayer(nn.Module):
    """AdaLN-modulated final projection layer.

    Unlike DiTBlock (6 modulation parameters), FinalLayer uses only 2
    (shift + scale) since there is no gate or MLP residual here — it is a
    pure projection head. The AdaLN linear is zero-initialized for the same
    stability reason as AdaLNZeroStrategy.
    """

    def __init__(self, hidden_size: int, patch_size, out_channels: int):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.out_channels = out_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels * np.prod(patch_size), bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.norm_final(x)
        x = self.linear(x)
        return x
