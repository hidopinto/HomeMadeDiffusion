from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn, Tensor

__all__ = ["ConditionOutput", "ConditionManager"]


@dataclass
class ConditionOutput:
    """Pre-projected conditions consumed by DiT.forward.

    adaLN:     (B, hidden_size) — sum of all global projections; added to t_embedder output.
    sequences: [(ctx, mask), ...] — concatenated in DiT.forward before cross-attention.
               Each ctx: (B, T_i, hidden_size), mask: (B, T_i) int64.
    """
    adaLN: Tensor | None = None
    sequences: list[tuple[Tensor, Tensor]] = field(default_factory=list)


class ConditionManager(nn.Module):
    """Projects raw encoder outputs and routes to ConditionOutput.

    Projectors are registered as (source_key, projector) pairs.
    source_key indexes into the raw_conditions dict passed to forward.
    A source_key may appear multiple times (one encoder → multiple projectors).

    Projector.role determines routing:
      "global"   → output (B, D) summed into ConditionOutput.adaLN
      "sequence" → output (B, T, D), (B, T) appended to ConditionOutput.sequences

    Note: source_keys is a structural config, not a learned weight. It must match
    the construction order in model_builder.py when loading a checkpoint.
    """

    def __init__(self, projectors: list[tuple[str, nn.Module]]) -> None:
        super().__init__()
        self.projector_modules = nn.ModuleList([p for _, p in projectors])
        self.source_keys: list[str] = [k for k, _ in projectors]

    def forward(self, raw_conditions: dict[str, dict]) -> ConditionOutput:
        adaLN_parts: list[Tensor] = []
        sequences: list[tuple[Tensor, Tensor]] = []
        for source_key, projector in zip(self.source_keys, self.projector_modules):
            raw = raw_conditions[source_key]
            if projector.role == "global":
                adaLN_parts.append(projector(raw))
            else:
                sequences.append(projector(raw))
        adaLN = sum(adaLN_parts) if adaLN_parts else None  # type: ignore[arg-type]
        return ConditionOutput(adaLN=adaLN, sequences=sequences)
