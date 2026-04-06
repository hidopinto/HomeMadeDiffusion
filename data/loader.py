import os
import time
from pathlib import Path

__all__ = ["build_dataloader"]

from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from data.cache import CacheManifest, LatentCachingEngine
from data.dataset import LatentDataset
from data.streaming import StreamingLatentDataset


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False


def build_dataloader(
    config,
    vae,
    tokenizer,
    text_encoder,
    device: str,
    split: str | None = None,
    shuffle: bool = True,
) -> DataLoader:
    mode = getattr(config.data, "mode", "cache")
    if mode == "streaming":
        return _build_streaming_dataloader(config, vae, tokenizer, text_encoder, device, split)
    return _build_cached_dataloader(config, vae, tokenizer, text_encoder, device, split, shuffle)


def _build_cached_dataloader(
    config,
    vae,
    tokenizer,
    text_encoder,
    device: str,
    split: str | None,
    shuffle: bool,
) -> DataLoader:
    cache_root = Path(config.data.cache_dir)
    dataset_name = config.data.dataset_name
    split = split if split is not None else config.data.split
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
        local_data_dir = getattr(config.data, "local_data_dir", None)
        if local_data_dir:
            resolved = str(Path(local_data_dir).expanduser())
            print(f"Loading dataset from local Parquet files: {resolved}")
            raw_dataset = load_dataset(
                "parquet",
                data_dir=resolved,
                split=split,
                num_proc=num_proc,
            )
        else:
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
        engine.run(raw_dataset, cache_root, split=split)

    dataset = LatentDataset(cache_root, dataset_name, split)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


def _build_streaming_dataloader(
    config,
    vae,
    tokenizer,
    text_encoder,
    device: str,
    split: str | None,
) -> DataLoader:
    split = split if split is not None else config.data.split
    # streaming=True: no download, no disk I/O — samples arrive on-the-fly from HF Hub shards
    # NOTE: num_proc and DownloadConfig are not valid kwargs for streaming datasets
    # Resolve a writable HF metadata cache dir; even streaming mode writes a few KB of builder
    # metadata, so we must avoid paths that may be unmounted or read-only.
    _default_hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    hf_cache = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("HF_HOME")
        or _default_hf_cache
    )
    if not _is_writable(hf_cache):
        hf_cache = _default_hf_cache
        hf_cache.mkdir(parents=True, exist_ok=True)
    raw_dataset = load_dataset(
        config.data.dataset_name, split=split, streaming=True, cache_dir=str(hf_cache)
    )
    # Optional shuffle buffer: add config.data.shuffle_buffer_size and call
    # raw_dataset = raw_dataset.shuffle(buffer_size=..., seed=42) here when needed
    dataset = StreamingLatentDataset(
        hf_dataset=raw_dataset,
        vae=vae,
        tokenizer=tokenizer,
        text_encoders={"text_embed": text_encoder},
        image_key=config.data.image_key,
        caption_key=config.data.caption_key,
        image_size=config.data.image_size,
        vae_scale_factor=config.dit.vae_scale_factor,
        encoding_batch_size=config.data.encoding_batch_size,
        device=device,
    )
    # num_workers=0: GPU encoding cannot be forked into subprocess workers
    # pin_memory=False: ineffective with num_workers=0
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=0,
        pin_memory=False,
    )
