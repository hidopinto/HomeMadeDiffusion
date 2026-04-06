import io
import json
import random
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["VaeCacheManifest", "VaeCachingEngine", "VaeCachedDataset"]

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from data.protocols import LatentEncoderProtocol, TextEncoderProtocol


@dataclass
class VaeCacheManifest:
    dataset_name: str
    split: str
    image_size: int
    vae_model_id: str
    num_samples: int

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "VaeCacheManifest":
        return cls(**json.loads(path.read_text()))

    def matches(self, config) -> bool:
        return (
            self.dataset_name == config.data.dataset_name
            and self.image_size == config.data.image_size
            and self.vae_model_id == config.external_models.vae
        )


class VaeCachingEngine:
    """
    Encodes images with a frozen VAE and saves latents + captions to disk.
    Text encoding is deliberately skipped — CLIP runs per-step during training instead.

    Output layout:
      {cache_dir}/latents/{index:06d}.pt  — one latent tensor per sample
      {cache_dir}/captions.jsonl          — one JSON-encoded caption string per line
      {cache_dir}/manifest.json           — VaeCacheManifest

    Resumable: counts existing latent files and skips that many samples in the HF streaming
    dataset at startup. Latent saves are atomic (tmp → rename). Captions are written after
    each micro-batch's saves are confirmed, so latent count and caption line count stay in sync.
    """

    def __init__(self, vae: LatentEncoderProtocol, config, device: str) -> None:
        self.vae = vae
        self.dataset_name = config.data.dataset_name
        self.image_size = config.data.image_size
        self.vae_scale_factor = config.dit.vae_scale_factor
        self.encoding_batch_size = config.data.encoding_batch_size
        self.image_key = config.data.image_key
        self.caption_key = config.data.caption_key
        self.device = device
        self.vae_model_id = config.external_models.vae

    @staticmethod
    def _save_latent(idx: int, latent: Tensor, latent_dir: Path) -> None:
        tmp = latent_dir / f"tmp_{idx:06d}.pt"
        torch.save(latent.cpu(), tmp)
        tmp.rename(latent_dir / f"{idx:06d}.pt")

    @torch.no_grad()
    def _encode_batch(self, images: list) -> Tensor:
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
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            pixel_arrays.append(arr)
        pixel_tensor = torch.from_numpy(np.stack(pixel_arrays))
        pixel_tensor = rearrange(pixel_tensor, "b h w c -> b c h w")
        pixel_tensor = pixel_tensor.to(self.device).to(self.vae.dtype)
        latents = self.vae.encode(pixel_tensor).latent_dist.sample()
        latents = latents * self.vae_scale_factor
        return latents.float()

    def run(self, hf_dataset, cache_root: Path, split: str) -> Path:
        cache_dir = cache_root / self.dataset_name.replace("/", "--") / split
        latent_dir = cache_dir / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        captions_path = cache_dir / "captions.jsonl"

        existing_count = len(list(latent_dir.glob("*.pt")))
        if existing_count > 0:
            print(f"[VaeCachingEngine] Resuming from sample {existing_count} ...")
            hf_dataset = hf_dataset.skip(existing_count)

        executor = ThreadPoolExecutor(max_workers=16)
        global_idx = existing_count
        images: list = []
        captions: list[str] = []

        with captions_path.open("a", encoding="utf-8") as caption_file:
            for raw in hf_dataset:
                images.append(raw[self.image_key])
                captions.append(raw[self.caption_key])

                if len(images) == self.encoding_batch_size:
                    latents = self._encode_batch(images)
                    batch_futures: list[Future] = [
                        executor.submit(self._save_latent, global_idx + j, latents[j], latent_dir)
                        for j in range(len(images))
                    ]
                    for f in batch_futures:
                        f.result()
                    for caption in captions:
                        caption_file.write(json.dumps(caption, ensure_ascii=False) + "\n")
                    caption_file.flush()
                    global_idx += len(images)
                    images = []
                    captions = []

                    if global_idx % 10_000 == 0:
                        print(f"[VaeCachingEngine] Cached {global_idx:,} samples ...")

            if images:
                latents = self._encode_batch(images)
                batch_futures = [
                    executor.submit(self._save_latent, global_idx + j, latents[j], latent_dir)
                    for j in range(len(images))
                ]
                for f in batch_futures:
                    f.result()
                for caption in captions:
                    caption_file.write(json.dumps(caption, ensure_ascii=False) + "\n")
                global_idx += len(images)

        executor.shutdown(wait=False)

        manifest = VaeCacheManifest(
            dataset_name=self.dataset_name,
            split=split,
            image_size=self.image_size,
            vae_model_id=self.vae_model_id,
            num_samples=global_idx,
        )
        manifest.save(cache_dir / "manifest.json")
        print(f"[VaeCachingEngine] Done. {global_idx:,} samples cached to {cache_dir}")
        return cache_dir


class VaeCachedDataset(IterableDataset):
    """
    Loads pre-cached VAE latents from disk and encodes captions with CLIP per micro-batch.

    Latent tensors are loaded from {cache_dir}/latents/{index:06d}.pt; captions are read from
    {cache_dir}/captions.jsonl. CLIP text encoding runs on GPU in micro-batches of
    encoding_batch_size, identical to StreamingLatentDataset's pattern.

    Output format per sample matches StreamingLatentDataset and LatentDataset exactly:
      {"latent": Tensor, "text_embed": {"hidden_states": Tensor, "attention_mask": Tensor}}

    Implements __len__ so DataLoader.len() works (used for LR schedule step counting).
    Shuffles sample order on each __iter__ call (i.e., each epoch).
    num_workers=0 is required in the DataLoader: CLIP runs on GPU and cannot cross fork boundaries.
    """

    def __init__(
        self,
        cache_dir: Path,
        tokenizer,
        text_encoders: dict[str, TextEncoderProtocol],
        config,
        device: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.latent_dir = self.cache_dir / "latents"
        self.tokenizer = tokenizer
        self.text_encoders = text_encoders
        self.encoding_batch_size = config.data.encoding_batch_size
        self.device = device

        captions_path = self.cache_dir / "captions.jsonl"
        with captions_path.open("r", encoding="utf-8") as f:
            self.captions: list[str] = [json.loads(line) for line in f if line.strip()]
        self.num_samples = len(self.captions)

    def __len__(self) -> int:
        return self.num_samples

    def _yield_encoded_micro_batch(
        self, indices: list[int], captions: list[str]
    ) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        latents = torch.stack([
            torch.load(self.latent_dir / f"{i:06d}.pt", map_location="cpu", weights_only=True)
            for i in indices
        ]).to(self.device)

        text_inputs = self.tokenizer(
            captions,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask  # (B, 77), stays on CPU

        text_embeds: dict[str, dict[str, Tensor]] = {}
        with torch.no_grad():
            for key, encoder in self.text_encoders.items():
                hidden_states = encoder(input_ids)[0]  # (B, 77, 768)
                text_embeds[key] = {
                    "hidden_states": hidden_states.float(),
                    "attention_mask": attention_mask,
                }

        for i in range(latents.shape[0]):
            sample: dict[str, Tensor | dict[str, Tensor]] = {"latent": latents[i].cpu()}
            for key, embeds in text_embeds.items():
                sample[key] = {
                    "hidden_states": embeds["hidden_states"][i].cpu(),
                    "attention_mask": embeds["attention_mask"][i].cpu(),
                }
            yield sample

    def __iter__(self) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        batch_indices: list[int] = []
        batch_captions: list[str] = []
        for idx in indices:
            batch_indices.append(idx)
            batch_captions.append(self.captions[idx])
            if len(batch_indices) == self.encoding_batch_size:
                yield from self._yield_encoded_micro_batch(batch_indices, batch_captions)
                batch_indices = []
                batch_captions = []
        if batch_indices:
            yield from self._yield_encoded_micro_batch(batch_indices, batch_captions)
