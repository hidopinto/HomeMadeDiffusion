import time
from pathlib import Path

__all__ = ["build_dataloader"]

from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from data.cache import CacheManifest, LatentCachingEngine
from data.dataset import LatentDataset


def build_dataloader(config, vae, tokenizer, text_encoder, device: str) -> DataLoader:
    cache_root = Path(config.data.cache_dir)
    dataset_name = config.data.dataset_name
    split = config.data.split
    cache_dir = cache_root / dataset_name.replace("/", "--") / split
    manifest_path = cache_dir / "manifest.json"

    expected = CacheManifest(
        dataset_name=dataset_name,
        split=split,
        image_size=config.data.image_size,
        vae_model_id=config.external_models.vae,
        encoder_keys=["text_embed"],
        encoder_model_ids={"text_embed": config.external_models.text_encoder},
        num_samples=-1,
        is_video=config.general.is_video,
    )

    cache_valid = False
    if manifest_path.exists():
        try:
            stored = CacheManifest.load(manifest_path)
            cache_valid = stored.matches(expected)
        except Exception:
            cache_valid = False

    if cache_valid:
        print("Cache valid, loading from disk...")
    else:
        num_proc = config.data.dataset_num_proc
        max_retries = config.data.dataset_max_retries
        download_config = DownloadConfig(max_retries=max_retries)
        raw_dataset = None
        for attempt in range(1, max_retries + 1):
            try:
                raw_dataset = load_dataset(
                    dataset_name,
                    split=split,
                    num_proc=num_proc,
                    download_config=download_config,
                )
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                print(f"Download attempt {attempt}/{max_retries} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
        engine = LatentCachingEngine(
            vae=vae,
            tokenizer=tokenizer,
            text_encoders={"text_embed": text_encoder},
            config=config,
            device=device,
            encoder_model_ids={"text_embed": config.external_models.text_encoder},
        )
        engine.run(raw_dataset, cache_root)

    dataset = LatentDataset(cache_root, dataset_name, split)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
