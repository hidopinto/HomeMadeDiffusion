import io

__all__ = ["encode_batch"]

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor

from data.protocols import LatentEncoderProtocol, TextEncoderProtocol


@torch.no_grad()
def encode_batch(
    images: list,
    captions: list[str],
    vae: LatentEncoderProtocol,
    tokenizer,
    text_encoders: dict[str, TextEncoderProtocol],
    image_size: int,
    vae_scale_factor: float,
    device: str,
) -> tuple[Tensor, dict[str, dict[str, Tensor]]]:
    pixel_arrays = []
    for img in images:
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((image_size, image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        pixel_arrays.append(arr)

    pixel_tensor = torch.from_numpy(np.stack(pixel_arrays))  # (B, H, W, 3)
    pixel_tensor = rearrange(pixel_tensor, 'b h w c -> b c h w')
    pixel_tensor = pixel_tensor.to(device).to(vae.dtype)

    latents = vae.encode(pixel_tensor).latent_dist.sample()
    latents = latents * vae_scale_factor
    latents = latents.float()

    text_inputs = tokenizer(
        captions,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask  # (B, 77)

    text_embeds: dict[str, dict[str, Tensor]] = {}
    for key, encoder in text_encoders.items():
        hidden_states = encoder(input_ids)[0]  # (B, 77, 768)
        text_embeds[key] = {
            "hidden_states": hidden_states.float(),
            "attention_mask": attention_mask,
        }

    return latents, text_embeds
