import logging
import os
import time
from pathlib import Path

__all__ = ["build_dataloader"]

logger = logging.getLogger(__name__)

from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader

from data.cache import CacheManifest, LatentCachingEngine
from data.dataset import LatentDataset
from data.streaming import StreamingLatentDataset
from data.vae_cache import VaeCacheManifest, VaeCachingEngine, VaeCachedDataset


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
    if mode == "cache_then_train":
        return _build_cache_then_train_dataloader(config, vae, tokenizer, text_encoder, device, split, shuffle)
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
        logger.info("Cache valid, loading from disk...")
    else:
        num_proc = config.data.dataset_num_proc
        local_data_dir = getattr(config.data, "local_data_dir", None)
        if local_data_dir:
            resolved = str(Path(local_data_dir).expanduser())
            logger.info("Loading dataset from local Parquet files: %s", resolved)
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
                    logger.warning("Download attempt %d/%d failed (%s). Retrying in %ds...", attempt, max_retries, e, wait)
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
    num_workers = config.data.num_workers
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=num_workers > 0,
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
    # Streaming datasets do not support percent-slice notation (e.g. "train[99%:]").
    # Strip any [...] suffix and fall back to .take() on a fixed sample count instead.
    _take_n: int | None = None
    if "[" in split:
        base_split = split[:split.index("[")]
        _take_n = getattr(config.data, "val_streaming_samples", 2048)
        logger.info(
            "[loader] Streaming mode does not support split slice '%s'. "
            "Loading base split '%s' and capping at %d samples via .take().",
            split, base_split, _take_n,
        )
        split = base_split
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
    if _take_n is not None:
        raw_dataset = raw_dataset.take(_take_n)
    else:
        # Skip the samples reserved for validation so train/val sets are non-overlapping.
        val_split_str = getattr(config.data, "val_split", None)
        if val_split_str and "[" in val_split_str:
            skip_n = getattr(config.data, "val_streaming_samples", 2048)
            raw_dataset = raw_dataset.skip(skip_n)
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


def _build_cache_then_train_dataloader(
    config,
    vae,
    tokenizer,
    text_encoder,
    device: str,
    split: str | None,
    shuffle: bool,
) -> DataLoader:
    """
    Runs a one-time VAE caching pass on first launch, then serves training batches from disk.

    On first run: streams the HF dataset, encodes every image with the frozen VAE, and saves
    latents + captions to vae_cache_dir. Subsequent runs skip straight to loading from cache.
    CLIP text encoding still runs per-step (fast ~0.3-0.5s vs VAE's ~3s).
    """
    split = split if split is not None else config.data.split

    # Streaming datasets do not support percent-slice notation — strip and use .take() instead.
    take_n: int | None = None
    if "[" in split:
        base_split = split[:split.index("[")]
        take_n = getattr(config.data, "val_streaming_samples", 2048)
        logger.info(
            "[loader] cache_then_train: split slice '%s' not supported in streaming. "
            "Loading '%s' and capping at %d samples.",
            split, base_split, take_n,
        )
        split = base_split

    vae_cache_root = Path(config.data.vae_cache_dir)
    cache_dir = vae_cache_root / config.data.dataset_name.replace("/", "--") / split
    manifest_path = cache_dir / "manifest.json"

    cache_valid = False
    if manifest_path.exists():
        try:
            stored = VaeCacheManifest.load(manifest_path)
            cache_valid = stored.matches(config)
        except Exception:
            cache_valid = False

    if not cache_valid:
        logger.info("[loader] VAE cache not found or config mismatch — running caching pass ...")
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
        if take_n is not None:
            raw_dataset = raw_dataset.take(take_n)
        engine = VaeCachingEngine(vae=vae, config=config, device=device)
        engine.run(raw_dataset, vae_cache_root, split=split, hf_cache=str(hf_cache))
        logger.info("[loader] Caching complete. Starting training from cache ...")
    else:
        logger.info("[loader] VAE cache valid (%d samples). Loading from disk ...", VaeCacheManifest.load(manifest_path).num_samples)

    dataset = VaeCachedDataset(
        cache_dir=cache_dir,
        tokenizer=tokenizer,
        text_encoders={"text_embed": text_encoder},
        config=config,
        device=device,
    )
    # num_workers=0: CLIP runs on GPU and cannot cross subprocess fork boundaries
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    logger.info("[loader] DataLoader ready: %d batches.", len(dataloader))
    return dataloader
