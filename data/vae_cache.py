import errno
import io
import json
import os
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets as hf_datasets
from huggingface_hub import list_repo_files

__all__ = ["VaeCacheManifest", "VaeCachingEngine", "VaeCachedDataset"]

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from data.protocols import LatentEncoderProtocol, TextEncoderProtocol


def _load_shard_state(cache_dir: Path) -> dict | None:
    path = cache_dir / "shard_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_shard_state(cache_dir: Path, state: dict) -> None:
    tmp = cache_dir / "shard_state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.rename(cache_dir / "shard_state.json")


def _sync_captions_to_latents(captions_path: Path, existing_count: int) -> None:
    """Remove orphaned (id >= existing_count) and duplicate entries from captions.jsonl.

    Handles two crash-recovery cases:
    - Orphaned captions: written after the crash point where latents were not saved.
    - Accumulated duplicates: same ID appended across multiple crash/resume cycles.
    First occurrence of each ID wins; rewrite is skipped if the file is already clean.
    """
    if not captions_path.exists():
        return
    lines = captions_path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[int] = set()
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj["id"] not in seen and obj["id"] < existing_count:
            seen.add(obj["id"])
            kept.append(line)
    if len(kept) == len(lines):
        return  # already clean, skip rewrite
    tmp = captions_path.with_suffix(".tmp")
    tmp.write_text("".join(kept), encoding="utf-8")
    tmp.rename(captions_path)


def _sync_latents_to_captions(captions_path: Path, latent_dir: Path, shard_size: int) -> int:
    """Validate shard files against captions. Returns valid latent count.

    For shard-based storage: counts from captions.jsonl and removes any shard files
    that extend beyond the valid range. Raises if old per-file format is detected.
    Aligns down to shard boundary to prevent overwrite on resume.
    """
    old_files = next(latent_dir.glob("[0-9]*.pt"), None)
    if old_files is not None:
        raise RuntimeError(
            f"[VaeCachingEngine] Old per-file latent format detected in {latent_dir}. "
            "Run 'python scripts/migrate_latent_shards.py' to convert to shard format."
        )
    if not captions_path.exists():
        return 0
    with captions_path.open(encoding="utf-8") as f:
        existing_count = sum(1 for line in f if line.strip())

    partial = existing_count % shard_size
    if partial != 0:
        aligned_count = existing_count - partial
        partial_shard_id = aligned_count // shard_size
        partial_shard = latent_dir / f"shard_{partial_shard_id:06d}.pt"
        if partial_shard.exists():
            partial_shard.unlink()
            print(
                f"[VaeCachingEngine] Deleted partial shard {partial_shard.name} "
                f"({partial} samples will be re-encoded on resume)"
            )
        existing_count = aligned_count

    max_valid_shard = (existing_count - 1) // shard_size if existing_count > 0 else -1
    removed = 0
    for pt in latent_dir.glob("shard_*.pt"):
        if pt.name.endswith(".tmp"):
            pt.unlink(missing_ok=True)
            removed += 1
            continue
        shard_id = int(pt.stem.split("_")[1])
        if shard_id > max_valid_shard:
            pt.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"[VaeCachingEngine] Removed {removed} out-of-range shard(s).")
    return existing_count


def _build_shard_resume_dataset(
    dataset_name: str,
    split: str,
    existing_count: int,
    shard_state: dict,
    hf_cache: str,
) -> tuple[Any, int]:
    """
    Returns (streaming_dataset_starting_from_target_shard, within_shard_skip_count).
    Falls back to (full_streaming_dataset, existing_count) on any lookup failure.
    """
    within_skip = existing_count - shard_state["start"]
    target_filename = shard_state["url"].rstrip("/").split("/")[-1]

    try:
        all_files = sorted([
            f for f in list_repo_files(dataset_name, repo_type="dataset")
            if f.endswith(".tar") and split in f
        ])
        start_pos = next(
            (i for i, f in enumerate(all_files) if f.endswith(target_filename)),
            None,
        )
        if start_pos is None:
            raise ValueError(f"Shard '{target_filename}' not found in repo file listing")
        remaining = [f"hf://datasets/{dataset_name}/{f}" for f in all_files[start_pos:]]
        dataset = hf_datasets.load_dataset(
            dataset_name,
            data_files={"train": remaining},
            streaming=True,
            cache_dir=hf_cache,
        )["train"]
        print(
            f"[VaeCachingEngine] Shard-aware resume: '{target_filename}' "
            f"({start_pos + 1}/{len(all_files)} shards), within-shard skip: {within_skip}"
        )
        return dataset, within_skip
    except Exception as e:
        print(f"[VaeCachingEngine] Shard-aware resume failed, will retry: {e}")
        raise


@dataclass
class VaeCacheManifest:
    dataset_name: str
    split: str
    image_size: int
    vae_model_id: str
    num_samples: int
    complete: bool = False  # True only after the full dataset stream has been exhausted

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "VaeCacheManifest":
        data = json.loads(path.read_text())
        data.setdefault("complete", False)  # backward-compat: old manifests land as incomplete
        return cls(**data)

    def matches(self, config) -> bool:
        return (
            self.complete  # incomplete caches always trigger a caching resume
            and self.dataset_name == config.data.dataset_name
            and self.image_size == config.data.image_size
            and self.vae_model_id == config.external_models.vae
        )


class VaeCachingEngine:
    """
    Encodes images with a frozen VAE and saves latents + captions to disk.
    Text encoding is deliberately skipped — CLIP runs per-step during training instead.

    Output layout:
      {cache_dir}/latents/shard_{id:06d}.pt  — stacked Tensor[N, C, H, W] per shard
      {cache_dir}/captions.jsonl             — one JSON-encoded caption string per line
      {cache_dir}/manifest.json              — VaeCacheManifest
      {cache_dir}/shard_state.json           — current HF shard URL + its global start index

    Resumable: on restart, shard_state.json is used to reload the HF streaming dataset
    starting from the right tar shard, then skip only the within-shard offset.
    Shard writes are atomic (tmp → rename). Captions are written per-shard so that
    existing_count is always shard-aligned on crash recovery.
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
        self.max_retries = config.data.dataset_max_retries
        self.shard_size = config.data.latent_shard_size

    @staticmethod
    def _write_latent_shard(shard_id: int, latents: Tensor, latent_dir: Path) -> None:
        """Atomically write a stacked latent shard via tmp → rename."""
        tmp = latent_dir / f"shard_{shard_id:06d}.pt.tmp"
        dst = latent_dir / f"shard_{shard_id:06d}.pt"
        torch.save(latents, tmp)
        tmp.rename(dst)

    @staticmethod
    def _commit_shard(
        shard_id: int,
        global_idx: int,
        buf: list[Tensor],
        captions: list[str],
        latent_dir: Path,
        caption_file,
    ) -> tuple[int, int]:
        """Write one shard and its captions atomically. Returns (new_shard_id, new_global_idx)."""
        VaeCachingEngine._write_latent_shard(shard_id, torch.stack(buf), latent_dir)
        for j, cap in enumerate(captions):
            caption_file.write(
                json.dumps({"id": global_idx + j, "caption": cap}, ensure_ascii=False) + "\n"
            )
        caption_file.flush()
        os.fsync(caption_file.fileno())
        return shard_id + 1, global_idx + len(buf)

    @torch.no_grad()
    def _encode_batch(self, images: list, captions: list[str]) -> tuple[Tensor | None, list[str]]:
        pixel_arrays = []
        valid_captions = []
        for img, caption in zip(images, captions):
            try:
                if isinstance(img, dict):
                    img = img.get("bytes") or open(img["path"], "rb").read()
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
                valid_captions.append(caption)
            except Exception as e:
                print(f"[VaeCachingEngine] Skipping corrupt image: {e}")
        if not pixel_arrays:
            return None, []
        pixel_tensor = torch.from_numpy(np.stack(pixel_arrays))
        pixel_tensor = rearrange(pixel_tensor, "b h w c -> b c h w")
        pixel_tensor = pixel_tensor.to(self.device).to(self.vae.dtype)
        latents = self.vae.encode(pixel_tensor).latent_dist.sample()
        latents = latents * self.vae_scale_factor
        return latents.float(), valid_captions

    def _flush_batch(
        self,
        images: list,
        captions: list[str],
    ) -> tuple[Tensor | None, list[str]]:
        """Encode a batch of images. Returns (latents_cpu | None, valid_captions)."""
        if not images:
            return None, []
        latents, valid_captions = self._encode_batch(images, captions)
        if latents is not None:
            return latents.cpu(), valid_captions
        return None, []

    def run(self, hf_dataset, cache_root: Path, split: str, hf_cache: str | None = None) -> Path:
        cache_dir = cache_root / self.dataset_name.replace("/", "--") / split
        latent_dir = cache_dir / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        captions_path = cache_dir / "captions.jsonl"

        base_dataset = hf_dataset.cast_column(self.image_key, hf_datasets.Image(decode=False))

        _NON_RETRYABLE_ERRNOS = {errno.EIO, errno.EROFS, errno.EACCES}

        attempt = 0
        while True:
            existing_count = _sync_latents_to_captions(captions_path, latent_dir, self.shard_size)
            _sync_captions_to_latents(captions_path, existing_count)
            shard_state = _load_shard_state(cache_dir)
            if shard_state and shard_state["start"] > existing_count:
                shard_state = None
            _last_recorded_url: str | None = shard_state["url"] if shard_state else None

            if existing_count > 0 and shard_state and hf_cache is not None:
                resume_ds, within_skip = _build_shard_resume_dataset(
                    self.dataset_name, split, existing_count, shard_state, hf_cache
                )
                dataset_iter = resume_ds.cast_column(self.image_key, hf_datasets.Image(decode=False))
                if within_skip > 0:
                    dataset_iter = dataset_iter.skip(within_skip)
            elif existing_count > 0:
                dataset_iter = base_dataset.skip(existing_count)
                print(f"[VaeCachingEngine] Resuming from sample {existing_count} (attempt {attempt + 1}) ...")
            else:
                dataset_iter = base_dataset

            global_idx = existing_count
            shard_id = existing_count // self.shard_size
            images: list = []
            captions: list[str] = []
            shard_buf: list[Tensor] = []
            shard_captions: list[str] = []

            try:
                with captions_path.open("a", encoding="utf-8") as caption_file:
                    for raw in dataset_iter:
                        url = raw.get("__url__", "")
                        if url and url != _last_recorded_url:
                            latents, valid_caps = self._flush_batch(images, captions)
                            images, captions = [], []
                            if latents is not None:
                                for j in range(len(valid_caps)):
                                    shard_buf.append(latents[j])
                                    shard_captions.append(valid_caps[j])
                            while len(shard_buf) >= self.shard_size:
                                shard_id, global_idx = self._commit_shard(
                                    shard_id, global_idx,
                                    shard_buf[:self.shard_size],
                                    shard_captions[:self.shard_size],
                                    latent_dir, caption_file,
                                )
                                shard_buf = shard_buf[self.shard_size:]
                                shard_captions = shard_captions[self.shard_size:]
                            _save_shard_state(cache_dir, {"url": url, "start": global_idx})
                            _last_recorded_url = url

                        images.append(raw[self.image_key])
                        captions.append(raw[self.caption_key])

                        if len(images) == self.encoding_batch_size:
                            latents, valid_caps = self._flush_batch(images, captions)
                            images, captions = [], []
                            if latents is not None:
                                for j in range(len(valid_caps)):
                                    shard_buf.append(latents[j])
                                    shard_captions.append(valid_caps[j])
                            while len(shard_buf) >= self.shard_size:
                                shard_id, global_idx = self._commit_shard(
                                    shard_id, global_idx,
                                    shard_buf[:self.shard_size],
                                    shard_captions[:self.shard_size],
                                    latent_dir, caption_file,
                                )
                                shard_buf = shard_buf[self.shard_size:]
                                shard_captions = shard_captions[self.shard_size:]
                                if global_idx % 10_000 == 0:
                                    print(f"[VaeCachingEngine] Cached {global_idx:,} samples ...")

                    # Flush remaining images
                    if images:
                        latents, valid_caps = self._flush_batch(images, captions)
                        if latents is not None:
                            for j in range(len(valid_caps)):
                                shard_buf.append(latents[j])
                                shard_captions.append(valid_caps[j])

                    # Flush remaining shard buffer (final partial shard)
                    while len(shard_buf) >= self.shard_size:
                        shard_id, global_idx = self._commit_shard(
                            shard_id, global_idx,
                            shard_buf[:self.shard_size],
                            shard_captions[:self.shard_size],
                            latent_dir, caption_file,
                        )
                        shard_buf = shard_buf[self.shard_size:]
                        shard_captions = shard_captions[self.shard_size:]
                    if shard_buf:
                        shard_id, global_idx = self._commit_shard(
                            shard_id, global_idx,
                            shard_buf, shard_captions,
                            latent_dir, caption_file,
                        )
                break  # success

            except (OSError, TimeoutError, ConnectionError, RuntimeError) as e:
                if isinstance(e, OSError) and e.errno in _NON_RETRYABLE_ERRNOS:
                    raise
                attempt += 1
                wait = min(2 ** attempt * 30, 300)
                print(f"[VaeCachingEngine] Streaming error (attempt {attempt}): {e}. Retrying in {wait}s ...")
                time.sleep(wait)

        manifest = VaeCacheManifest(
            dataset_name=self.dataset_name,
            split=split,
            image_size=self.image_size,
            vae_model_id=self.vae_model_id,
            num_samples=global_idx,
            complete=True,
        )
        manifest.save(cache_dir / "manifest.json")
        print(f"[VaeCachingEngine] Done. {global_idx:,} samples cached to {cache_dir}")
        return cache_dir


class VaeCachedDataset(IterableDataset):
    """
    Loads pre-cached VAE latents from shard files and encodes captions with CLIP per micro-batch.

    Latent shards are loaded from {cache_dir}/latents/shard_{id:06d}.pt; captions from
    {cache_dir}/captions.jsonl. CLIP text encoding runs on GPU in micro-batches.

    Output format per sample:
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
        self.shard_size = config.data.latent_shard_size
        self.device = device
        self._shard_cache: dict[int, Tensor] = {}

        captions_path = self.cache_dir / "captions.jsonl"
        with captions_path.open("r", encoding="utf-8") as f:
            self.captions: dict[int, str] = {
                obj["id"]: obj["caption"]
                for line in f if line.strip()
                for obj in (json.loads(line),)
            }
        self.num_samples = len(self.captions)

        # Precompute shard → sample-id mapping so __iter__ can shuffle at shard level
        # rather than globally. Eliminates per-micro-batch random shard thrashing.
        self._shard_to_indices: dict[int, list[int]] = {}
        for idx in self.captions:
            sid = idx // self.shard_size
            self._shard_to_indices.setdefault(sid, []).append(idx)

    def __len__(self) -> int:
        return self.num_samples

    def _load_latent(self, idx: int) -> Tensor:
        shard_id = idx // self.shard_size
        local_idx = idx % self.shard_size
        if shard_id not in self._shard_cache:
            if len(self._shard_cache) >= 8:
                self._shard_cache.pop(next(iter(self._shard_cache)))
            self._shard_cache[shard_id] = torch.load(
                self.latent_dir / f"shard_{shard_id:06d}.pt",
                map_location="cpu",
                weights_only=True,
            )
        return self._shard_cache[shard_id][local_idx]

    def iter_latents(self, batch_size: int) -> Iterator[Tensor]:
        """Yield batches of latents from disk without any text encoding.

        Intended for use-cases that only need latent tensors (e.g. pre-populating
        FID real-image statistics). Order is not shuffled.
        """
        indices = list(self.captions.keys())
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start : start + batch_size]
            yield torch.stack([self._load_latent(i) for i in batch_ids])

    def _yield_encoded_micro_batch(
        self, indices: list[int], captions: list[str]
    ) -> Iterator[dict[str, Tensor | dict[str, Tensor]]]:
        latents = torch.stack([self._load_latent(i) for i in indices]).to(self.device)

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
        # Shard-level shuffle: randomise shard visitation order each epoch, then
        # shuffle samples within each shard. A micro-batch of encoding_batch_size
        # samples therefore touches at most 2 consecutive shards (at a boundary),
        # vs the ~580 shards that a full global shuffle would require. Disk I/O
        # per micro-batch drops from ~36 GB (random) to ~63 MB (sequential shard).
        shard_order = list(self._shard_to_indices.keys())
        random.shuffle(shard_order)
        batch_indices: list[int] = []
        batch_captions: list[str] = []
        for shard_id in shard_order:
            shard_idxs = list(self._shard_to_indices[shard_id])
            random.shuffle(shard_idxs)
            for idx in shard_idxs:
                batch_indices.append(idx)
                batch_captions.append(self.captions[idx])
                if len(batch_indices) == self.encoding_batch_size:
                    yield from self._yield_encoded_micro_batch(batch_indices, batch_captions)
                    batch_indices = []
                    batch_captions = []
        if batch_indices:
            yield from self._yield_encoded_micro_batch(batch_indices, batch_captions)
