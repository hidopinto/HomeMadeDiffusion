import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, cache_root: Path, dataset_name: str, split: str) -> None:
        cache_dir = cache_root / dataset_name.replace("/", "--") / split
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        self.num_samples: int = manifest["num_samples"]
        self.encoder_keys: list[str] = manifest["encoder_keys"]
        self.latent_dir = cache_dir / "latents"
        self.embed_dirs: dict[str, Path] = {k: cache_dir / k for k in self.encoder_keys}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: dict[str, Tensor] = {
            "latent": torch.load(self.latent_dir / f"{index:06d}.pt", weights_only=True),
        }
        for key, embed_dir in self.embed_dirs.items():
            sample[key] = torch.load(embed_dir / f"{index:06d}.pt", weights_only=True)
        return sample
