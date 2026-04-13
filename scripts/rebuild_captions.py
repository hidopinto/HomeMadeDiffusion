"""Rebuild captions.jsonl from the HF streaming dataset after a disk incident.

Streams through the first N samples (N = number of .pt files in latents/) and
reconstructs captions.jsonl, shard_state.json, and manifest.json without
re-encoding any images through the VAE.

Usage:
    python scripts/rebuild_captions.py

Crash-resilient: progress is saved to captions.jsonl.rebuild every 10k lines.
On re-run, the script picks up where it left off. The existing captions.jsonl
is left untouched until the full rebuild is complete (atomic rename at the end).
"""

import json
import sys
from pathlib import Path

import datasets as hf_datasets
import yaml
from box import Box

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
config = Box(yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text()))

DATASET_NAME: str = config.data.dataset_name
SPLIT: str = config.data.split
IMAGE_KEY: str = config.data.image_key
CAPTION_KEY: str = config.data.caption_key
HF_CACHE: str = config.data.cache_dir

CACHE_DIR = Path(config.data.vae_cache_dir) / DATASET_NAME.replace("/", "--") / SPLIT
LATENT_DIR = CACHE_DIR / "latents"

# ---------------------------------------------------------------------------
# Reuse atomic helpers from data.vae_cache
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
from data.vae_cache import VaeCacheManifest, _save_shard_state  # noqa: E402


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a text file; returns 0 if the file does not exist."""
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Determine target sample count from latent files on disk
    # ------------------------------------------------------------------
    target_count = len(list(LATENT_DIR.glob("[0-9]*.pt")))
    if target_count == 0:
        print("[rebuild_captions] No latent files found — nothing to do.")
        return
    print(f"[rebuild_captions] Target: {target_count:,} samples (from latent file count)")

    # ------------------------------------------------------------------
    # 2. Check for an in-progress rebuild to determine resume point
    # ------------------------------------------------------------------
    rebuild_path = CACHE_DIR / "captions.jsonl.rebuild"
    final_path = CACHE_DIR / "captions.jsonl"

    existing_rebuild = _count_lines(rebuild_path)
    if existing_rebuild > 0:
        print(f"[rebuild_captions] Resuming rebuild from sample {existing_rebuild:,}")
    else:
        print("[rebuild_captions] Starting fresh caption rebuild")

    global_idx: int = existing_rebuild
    shard_state: dict = {"url": None, "start": 0}
    last_url: str | None = None

    # ------------------------------------------------------------------
    # 3. Stream the dataset (images kept as bytes — never decoded)
    # ------------------------------------------------------------------
    dataset = hf_datasets.load_dataset(
        DATASET_NAME,
        streaming=True,
        cache_dir=HF_CACHE,
    )["train"]
    dataset = dataset.cast_column(IMAGE_KEY, hf_datasets.Image(decode=False))

    if existing_rebuild > 0:
        dataset = dataset.skip(existing_rebuild)

    # ------------------------------------------------------------------
    # 4. Write captions progressively to the rebuild scratch file
    # ------------------------------------------------------------------
    with rebuild_path.open("a", encoding="utf-8") as out:
        for raw in dataset:
            if global_idx >= target_count:
                break

            url: str = raw.get("__url__", "")
            if url and url != last_url:
                shard_state = {"url": url, "start": global_idx}
                last_url = url

            caption: str = raw[CAPTION_KEY]
            out.write(json.dumps({"id": global_idx, "caption": caption}, ensure_ascii=False) + "\n")
            global_idx += 1

            if global_idx % 10_000 == 0:
                out.flush()
                print(f"[rebuild_captions] Rebuilt {global_idx:,} / {target_count:,} captions ...")

    # ------------------------------------------------------------------
    # 5. Atomic rename + update state files
    # ------------------------------------------------------------------
    rebuild_path.rename(final_path)
    print(f"[rebuild_captions] captions.jsonl written ({global_idx:,} lines)")

    _save_shard_state(CACHE_DIR, shard_state)
    print(f"[rebuild_captions] shard_state.json → {json.dumps(shard_state)}")

    manifest = VaeCacheManifest(
        dataset_name=DATASET_NAME,
        split=SPLIT,
        image_size=config.data.image_size,
        vae_model_id=config.external_models.vae,
        num_samples=global_idx,
    )
    manifest.save(CACHE_DIR / "manifest.json")
    print(f"[rebuild_captions] manifest.json written (num_samples={global_idx:,})")
    print(f"[rebuild_captions] Done. Ready to resume caching from sample {global_idx:,}")


if __name__ == "__main__":
    main()
