from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class LatentEncoderProtocol(Protocol):
    dtype: torch.dtype

    def encode(self, pixel_values: Tensor): ...


@runtime_checkable
class TextEncoderProtocol(Protocol):
    dtype: torch.dtype

    def __call__(self, input_ids: Tensor) -> tuple: ...
