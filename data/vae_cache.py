import errno
import io
import json
import os
import random
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
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


def _sync_latents_to_captions(captions_path: Path, latent_dir: Path) -> int:
    """Delete latent files with no matching caption entry. Returns valid latent count.

    Called at resume time so that any latents saved by background executor threads
    after a crash (with no corresponding caption) are cleaned up automatically.
    """
    if not captions_path.exists():
        return len(list(latent_dir.glob("[0-9]*.pt")))
    with captions_path.open(encoding="utf-8") as f:
        caption_ids: set[int] = {json.loads(line)["id"] for line in f if line.strip()}
    removed = 0
    for pt in latent_dir.glob("[0-9]*.pt"):
        if int(pt.stem) not in caption_ids:
            pt.unlink(missing_ok=True)
            removed += 1
    if removed:
        print(f"[VaeCachingEngine] Removed {removed} orphaned latent(s) with no caption.")
    return len(caption_ids)


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
      {cache_dir}/latents/{index:06d}.pt  — one latent tensor per sample
      {cache_dir}/captions.jsonl          — one JSON-encoded caption string per line
      {cache_dir}/manifest.json           — VaeCacheManifest
      {cache_dir}/shard_state.json        — current shard URL + its global start index

    Resumable: on restart, shard_state.json is used to reload the HF streaming dataset
    starting from the right tar shard, then skip only the within-shard offset. This avoids
    re-downloading all preceding shards. Falls back to .skip(N) if shard_state.json is absent.
    Latent saves are atomic (tmp → rename). Captions are written after each batch's saves are
    confirmed, so latent count and caption line count stay in sync.
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

    @staticmethod
    def _save_latent(idx: int, latent: Tensor, latent_dir: Path) -> None:
        dest = latent_dir / f"{idx:06d}.pt"
        for attempt in range(3):
            try:
                with open(dest, "wb") as f:
                    torch.save(latent.cpu(), f)
                return
            except OSError as e:
                if e.errno == errno.ENOSPC and attempt < 2:
                    wait = 2 ** attempt  # 1s, 2s
                    print(
                        f"[VaeCachingEngine] ENOSPC saving latent {idx} "
                        f"(attempt {attempt + 1}/3), retrying in {wait}s ..."
                    )
                    time.sleep(wait)
                    continue
                print(
                    f"[VaeCachingEngine] Failed to save latent {idx} "
                    f"after {attempt + 1} attempt(s): {e}"
                )
                raise

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
        global_idx: int,
        latent_dir: Path,
        caption_file,
        executor: ThreadPoolExecutor,
    ) -> tuple[list, list[str], int]:
        """Encode and save a batch to disk. Returns ([], [], updated_global_idx)."""
        if not images:
            return images, captions, global_idx
        latents, valid_captions = self._encode_batch(images, captions)
        if latents is not None:
            futures: list[Future] = [
                executor.submit(self._save_latent, global_idx + j, latents[j], latent_dir)
                for j in range(len(valid_captions))
            ]
            for f in futures:
                f.result()
            for j, cap in enumerate(valid_captions):
                caption_file.write(json.dumps({"id": global_idx + j, "caption": cap}, ensure_ascii=False) + "\n")
            caption_file.flush()
            os.fsync(caption_file.fileno())
            global_idx += len(valid_captions)
        return [], [], global_idx

    def run(self, hf_dataset, cache_root: Path, split: str, hf_cache: str | None = None) -> Path:
        cache_dir = cache_root / self.dataset_name.replace("/", "--") / split
        latent_dir = cache_dir / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        captions_path = cache_dir / "captions.jsonl"

        # Cast once; keep base_dataset so we can recreate the iterator on retry
        base_dataset = hf_dataset.cast_column(self.image_key, hf_datasets.Image(decode=False))

        _NON_RETRYABLE_ERRNOS = {errno.EIO, errno.EROFS, errno.EACCES}

        attempt = 0
        while True:
            # Sync captions down to the raw file count, then drop latents with no caption.
            # Together these handle all crash-recovery cases: orphaned captions (written
            # after latent saves failed) and orphaned latents (saved by background executor
            # threads after the caption-write was interrupted).
            _sync_captions_to_latents(captions_path, len(list(latent_dir.glob("[0-9]*.pt"))))
            existing_count = _sync_latents_to_captions(captions_path, latent_dir)
            shard_state = _load_shard_state(cache_dir)
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
            images: list = []
            captions: list[str] = []

            # Create a fresh executor each attempt so stale futures from a previous
            # failed attempt cannot interfere with the new iteration.
            executor = ThreadPoolExecutor(max_workers=16)
            try:
                with captions_path.open("a", encoding="utf-8") as caption_file:
                    for raw in dataset_iter:
                        url = raw.get("__url__", "")
                        if url and url != _last_recorded_url:
                            # Flush buffered samples from the previous shard before recording
                            # the new shard boundary so that shard_state["start"] is accurate.
                            images, captions, global_idx = self._flush_batch(
                                images, captions, global_idx, latent_dir, caption_file, executor
                            )
                            _save_shard_state(cache_dir, {"url": url, "start": global_idx})
                            _last_recorded_url = url

                        images.append(raw[self.image_key])
                        captions.append(raw[self.caption_key])

                        if len(images) == self.encoding_batch_size:
                            images, captions, global_idx = self._flush_batch(
                                images, captions, global_idx, latent_dir, caption_file, executor
                            )
                            if global_idx % 10_000 == 0:
                                print(f"[VaeCachingEngine] Cached {global_idx:,} samples ...")

                    if images:
                        _, _, global_idx = self._flush_batch(
                            images, captions, global_idx, latent_dir, caption_file, executor
                        )
                executor.shutdown(wait=True)
                break  # success

            except (OSError, TimeoutError, ConnectionError, RuntimeError) as e:
                executor.shutdown(wait=False)
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
            self.captions: dict[int, str] = {
                obj["id"]: obj["caption"]
                for line in f if line.strip()
                for obj in (json.loads(line),)
            }
        self.num_samples = len(self.captions)

    def __len__(self) -> int:
        return self.num_samples

    def iter_latents(self, batch_size: int) -> Iterator[Tensor]:
        """Yield batches of latents from disk without any text encoding.

        Intended for use-cases that only need latent tensors (e.g. pre-populating
        FID real-image statistics). Order is not shuffled.
        """
        indices = list(self.captions.keys())
        for start in range(0, len(indices), batch_size):
            batch_ids = indices[start : start + batch_size]
            yield torch.stack([
                torch.load(
                    self.latent_dir / f"{i:06d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for i in batch_ids
            ])

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
        indices = list(self.captions.keys())
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
