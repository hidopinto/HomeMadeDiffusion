#!/usr/bin/env python3
"""
Extract captions from the local PickaPic Parquet files and write captions.jsonl
alongside the LatentDataset cache.

PickaPic has one caption per comparison pair (two images share the same prompt).
The LatentDataset cache was built by iterating the Parquet shards in sorted filename
order — so Parquet row index i maps to cached file {i:06d}.pt.

Usage:
    python scripts/extract_pickapic_captions.py \\
        [--parquet-dir /media/hido-pinto/מחסן/datasets/pickapic-anonymous--pickapic_v1/data] \\
        [--cache-dir  /media/hido-pinto/מחסן/cache/pickapic-anonymous--pickapic_v1/train] \\
        [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_PARQUET_DIR = "/media/hido-pinto/מחסן/datasets/pickapic-anonymous--pickapic_v1/data"
_DEFAULT_CACHE_DIR   = "/media/hido-pinto/מחסן/cache/pickapic-anonymous--pickapic_v1/train"
_CAPTION_KEY = "caption"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write captions.jsonl for PickaPic from local Parquet files"
    )
    parser.add_argument("--parquet-dir", default=_DEFAULT_PARQUET_DIR,
                        help="Directory containing train-*.parquet files")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR,
                        help="LatentDataset cache directory (where captions.jsonl will be written)")
    parser.add_argument("--resume", action="store_true",
                        help="Append to existing captions.jsonl, skipping already-written IDs")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("[extract_pickapic_captions] pandas is required. Run: uv add pandas pyarrow",
              file=sys.stderr)
        sys.exit(1)

    parquet_dir = Path(args.parquet_dir)
    cache_dir   = Path(args.cache_dir)
    out_path    = cache_dir / "captions.jsonl"

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[extract_pickapic_captions] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    num_cached = json.loads(manifest_path.read_text())["num_samples"]
    print(f"[extract_pickapic_captions] Cache has {num_cached:,} samples")

    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        print(f"[extract_pickapic_captions] No train-*.parquet files found in {parquet_dir}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[extract_pickapic_captions] Found {len(shards)} train shards in {parquet_dir}")

    already_done: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["id"])
        print(f"[extract_pickapic_captions] Resuming — {len(already_done):,} already written")

    write_mode = "a" if args.resume else "w"
    written = 0
    global_idx = 0
    total_parquet_rows = 0

    with out_path.open(write_mode, encoding="utf-8") as fout:
        for shard_path in shards:
            df = pd.read_parquet(shard_path, columns=[_CAPTION_KEY])
            total_parquet_rows += len(df)
            for caption in df[_CAPTION_KEY]:
                if global_idx >= num_cached:
                    break
                if global_idx not in already_done:
                    if not isinstance(caption, str):
                        caption = str(caption) if caption is not None else ""
                    fout.write(
                        json.dumps({"id": global_idx, "caption": caption.strip()},
                                   ensure_ascii=False) + "\n"
                    )
                    written += 1
                global_idx += 1

            if written % 50_000 == 0 and written > 0:
                fout.flush()
                print(f"[extract_pickapic_captions] {written + len(already_done):,} written ...")

            if global_idx >= num_cached:
                break

    total = written + len(already_done)
    print(f"\n[extract_pickapic_captions] Done. {total:,} captions written to {out_path}")

    if abs(total_parquet_rows - num_cached) > 500:
        print(
            f"  WARNING: Parquet row count ({total_parquet_rows:,}) differs from cache size "
            f"({num_cached:,}) by more than 500. The sequential ID mapping may be off for "
            "some samples.",
            file=sys.stderr,
        )
    else:
        print(f"  Parquet rows processed: {total_parquet_rows:,} — matches cache within tolerance. ✓")


if __name__ == "__main__":
    main()
