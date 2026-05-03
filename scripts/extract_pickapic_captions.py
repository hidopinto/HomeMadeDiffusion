#!/usr/bin/env python3
"""
Extract captions from PickaPic HF dataset and write captions.jsonl alongside the local cache.

PickaPic stores one caption per image pair (two generated images, one shared prompt).
The local LatentDataset cache (per-file format) was built by iterating the HF dataset in
order — so row index i in the HF stream maps to cached file i/000000i.pt.

Usage:
    python scripts/extract_pickapic_captions.py [--cache-dir PATH] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_CACHE_DIR = "/media/hido-pinto/מחסן/cache/pickapic-anonymous--pickapic_v1/train"
_DATASET = "pickapic-anonymous/pickapic_v1"
_SPLIT = "train"
_CAPTION_KEY = "caption"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR,
                        help="PickaPic LatentDataset cache directory (train split)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing captions.jsonl, skipping already-written IDs")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_path = cache_dir / "captions.jsonl"

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[extract_pickapic_captions] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    num_cached = json.loads(manifest_path.read_text())["num_samples"]
    print(f"[extract_pickapic_captions] Cache has {num_cached:,} samples")

    already_done: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["id"])
        print(f"[extract_pickapic_captions] Resuming — {len(already_done):,} already written")

    try:
        import datasets as hf_datasets
    except ImportError:
        print("[extract_pickapic_captions] Install `datasets` first.", file=sys.stderr)
        sys.exit(1)

    print(f"[extract_pickapic_captions] Streaming {_DATASET} ({_SPLIT}) ...")
    ds = hf_datasets.load_dataset(_DATASET, split=_SPLIT, streaming=True)

    write_mode = "a" if args.resume else "w"
    written = 0
    skipped = 0
    with out_path.open(write_mode, encoding="utf-8") as fout:
        for idx, row in enumerate(ds):
            if idx >= num_cached:
                break
            if idx in already_done:
                continue

            caption = row.get(_CAPTION_KEY, "")
            if not isinstance(caption, str):
                caption = str(caption)
            caption = caption.strip()

            fout.write(json.dumps({"id": idx, "caption": caption}, ensure_ascii=False) + "\n")
            written += 1

            if written % 50_000 == 0:
                fout.flush()
                print(f"[extract_pickapic_captions] {written + len(already_done):,} written ...")

    total = written + len(already_done)
    print(f"\n[extract_pickapic_captions] Done. {total:,} captions written to {out_path}")
    if total < num_cached:
        print(
            f"  WARNING: only {total:,} captions written but cache has {num_cached:,} samples. "
            "The HF dataset may have fewer rows than cached samples (possible if both images "
            "per pair were cached). Diversity sampling will cover only the {total:,} with captions.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
