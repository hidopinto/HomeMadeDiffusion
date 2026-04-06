from collections.abc import Iterator

__all__ = ["StreamingLatentDataset"]

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from data.encoding import encode_batch
from data.protocols import LatentEncoderProtocol, TextEncoderProtocol


class StreamingLatentDataset(IterableDataset):
    """
    Yields encoded latents and text embeddings on-the-fly from a HuggingFace streaming dataset.

    Raw images and captions are accumulated into micro-batches of `encoding_batch_size`, encoded
    on GPU using the provided frozen VAE and text encoders, then yielded as individual CPU tensors.
    The batch format is identical to LatentDataset: {"latent": Tensor, "text_embed": {...}}.

    num_workers=0 is mandatory in the DataLoader: GPU tensors cannot cross subprocess fork
    boundaries. Worker splitting via get_worker_info() is intentionally omitted for this reason.

    Extension note: to add native WebDataset support (e.g. via the `webdataset` library), replace
    the HF IterableDataset passed at construction with a wds.WebDataset pipeline — the
    encode-and-yield loop in __iter__ requires no changes.
    """

    def __init__(
        self,
        hf_dataset,
        vae: LatentEncoderProtocol,
        tokenizer,
        text_encoders: dict[str, TextEncoderProtocol],
        image_key: str,
        caption_key: str,
        image_size: int,
        vae_scale_factor: float,
        encoding_batch_size: int,
        device: str,
    ) -> None:
        self.dataset = hf_dataset
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoders = text_encoders
        self.image_key = image_key
        self.caption_key = caption_key
        self.image_size = image_size
        self.vae_scale_factor = vae_scale_factor
        self.encoding_batch_size = encoding_batch_size
        self.device = device

    def _yield_encoded_micro_batch(
        self, images: list, captions: list[str]
    ) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        latents, text_embeds = encode_batch(
            images,
            captions,
            self.vae,
            self.tokenizer,
            self.text_encoders,
            self.image_size,
            self.vae_scale_factor,
            self.device,
        )
        B = latents.shape[0]
        for i in range(B):
            sample: dict[str, Tensor | dict[str, Tensor]] = {"latent": latents[i].cpu()}
            for key, embeds in text_embeds.items():
                sample[key] = {
                    "hidden_states": embeds["hidden_states"][i].cpu(),
                    "attention_mask": embeds["attention_mask"][i].cpu(),
                }
            yield sample

    def __iter__(self) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        images: list = []
        captions: list[str] = []
        for raw in self.dataset:
            images.append(raw[self.image_key])
            captions.append(raw[self.caption_key])
            if len(images) == self.encoding_batch_size:
                yield from self._yield_encoded_micro_batch(images, captions)
                images = []
                captions = []
        if images:
            yield from self._yield_encoded_micro_batch(images, captions)
