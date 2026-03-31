from __future__ import annotations

from torch import nn, Tensor

from models.layers import masked_mean_pool

__all__ = ["AdaLNTextProjector", "CrossAttnTextProjector"]


class AdaLNTextProjector(nn.Module):
    """Masked-mean-pools text sequence → (B, hidden_size) for AdaLN global conditioning."""
    role: str = "global"

    def __init__(self, cond_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_size)

    def forward(self, text: dict) -> Tensor:
        pooled = masked_mean_pool(text["hidden_states"], text["attention_mask"])
        return self.linear(pooled)


class CrossAttnTextProjector(nn.Module):
    """Projects text token sequence → (B, T, hidden_size) for cross-attention."""
    role: str = "sequence"

    def __init__(self, cond_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_size)

    def forward(self, text: dict) -> tuple[Tensor, Tensor]:
        return self.linear(text["hidden_states"]), text["attention_mask"]
