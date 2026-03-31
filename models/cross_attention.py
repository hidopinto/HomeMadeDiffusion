from __future__ import annotations

import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

__all__ = ["CrossAttention"]


class CrossAttention(nn.Module):
    """Multi-head cross-attention via F.scaled_dot_product_attention (flash-attn compatible).

    Q from patch tokens (B, N, D); K/V from pre-projected context (B, T, D).
    context_mask: (B, T) int64 — 1=real token, 0=pad.
    SDPA uses True=ignore, so mask is inverted internally.

    Note: CLIP always produces BOS/EOS tokens so all-pad sequences cannot occur
    in the current text-only setup, but the mask inversion is correct regardless.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, context: Tensor, context_mask: Tensor) -> Tensor:
        attn_bias = rearrange(context_mask == 0, 'b t -> b 1 1 t')  # True = ignore
        q = rearrange(self.q_proj(x),       'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(context), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.v_proj(context), 'b t (h d) -> b h t d', h=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return self.out_proj(rearrange(out, 'b h n d -> b n (h d)'))
