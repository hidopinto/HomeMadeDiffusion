import json
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["CacheManifest", "LatentCachingEngine"]

import torch
from torch import Tensor

from data.encoding import encode_batch
from data.protocols import LatentEncoderProtocol, TextEncoderProtocol


@dataclass
class CacheManifest:
    dataset_name: str
    split: str
    image_size: int
    vae_model_id: str
    encoder_keys: list[str]
    encoder_model_ids: dict[str, str]
    num_samples: int
    is_video: bool

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CacheManifest":
        return cls(**json.loads(path.read_text()))

    def matches(self, other: "CacheManifest") -> bool:
        return (
            self.dataset_name == other.dataset_name
            and self.split == other.split
            and self.image_size == other.image_size
            and self.vae_model_id == other.vae_model_id
            and sorted(self.encoder_keys) == sorted(other.encoder_keys)
            and self.encoder_model_ids == other.encoder_model_ids
            and self.is_video == other.is_video
        )


class LatentCachingEngine:
    def __init__(
        self,
        vae: LatentEncoderProtocol,
        tokenizer,
        text_encoders: dict[str, TextEncoderProtocol],
        config,
        device: str,
        encoder_model_ids: dict[str, str] | None = None,
    ) -> None:
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoders = text_encoders
        self.config = config
        self.device = device
        self.encoder_model_ids = encoder_model_ids or {}

    @staticmethod
    def _save_sample(
        idx: int,
        latent: Tensor,
        text_embeds: dict[str, dict[str, Tensor]],
        latent_dir: Path,
        cache_dir: Path,
    ) -> None:
        tmp = latent_dir / f"tmp_{idx:06d}.pt"
        torch.save(latent.cpu(), tmp)
        tmp.rename(latent_dir / f"{idx:06d}.pt")
        for key, embeds in text_embeds.items():
            tmp = cache_dir / key / f"tmp_{idx:06d}.pt"
            sample = {
                "hidden_states": embeds["hidden_states"].cpu(),
                "attention_mask": embeds["attention_mask"].cpu(),
            }
            torch.save(sample, tmp)
            tmp.rename(cache_dir / key / f"{idx:06d}.pt")

    def run(self, dataset, cache_root: Path, split: str | None = None) -> Path:
        dataset_name = self.config.data.dataset_name
        split = split if split is not None else self.config.data.split
        cache_dir = cache_root / dataset_name.replace("/", "--") / split

        latent_dir = cache_dir / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        for key in self.text_encoders:
            (cache_dir / key).mkdir(parents=True, exist_ok=True)

        n = len(dataset)
        batch_size = self.config.data.encoding_batch_size
        image_key = self.config.data.image_key
        caption_key = self.config.data.caption_key

        print(f"Caching {n} samples to {cache_dir} ...")
        executor = ThreadPoolExecutor(max_workers=16)
        pending: list[Future] = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = list(range(start, end))

            # Resumable: skip batches where all outputs already exist
            all_exist = all(
                (latent_dir / f"{i:06d}.pt").exists()
                and all((cache_dir / k / f"{i:06d}.pt").exists() for k in self.text_encoders)
                for i in batch_indices
            )
            if all_exist:
                continue

            batch_data = dataset[start:end]
            images = batch_data[image_key]
            captions = batch_data[caption_key]
            latents, text_embeds = self._encode_batch(images, captions)

            for j, i in enumerate(batch_indices):
                per_sample_embeds = {
                    key: {
                        "hidden_states": embeds["hidden_states"][j],
                        "attention_mask": embeds["attention_mask"][j],
                    }
                    for key, embeds in text_embeds.items()
                }
                pending.append(
                    executor.submit(
                        self._save_sample,
                        i, latents[j], per_sample_embeds, latent_dir, cache_dir,
                    )
                )

        for f in pending:
            f.result()
        executor.shutdown(wait=False)

        manifest = CacheManifest(
            dataset_name=dataset_name,
            split=split,
            image_size=self.config.data.image_size,
            vae_model_id=self.config.external_models.vae,
            encoder_keys=list(self.text_encoders.keys()),
            encoder_model_ids=self.encoder_model_ids,
            num_samples=n,
            is_video=self.config.general.is_video,
        )
        manifest.save(cache_dir / "manifest.json")
        return cache_dir

    def _encode_batch(
        self, images: list, captions: list[str]
    ) -> tuple[Tensor, dict[str, dict[str, Tensor]]]:
        return encode_batch(
            images,
            captions,
            self.vae,
            self.tokenizer,
            self.text_encoders,
            self.config.data.image_size,
            self.config.dit.vae_scale_factor,
            self.device,
        )
