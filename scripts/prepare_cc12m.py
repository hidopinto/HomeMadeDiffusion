#!/usr/bin/env python3
"""
Build a filtered global-index file for CC12M Phase 2 training.

Reads cc12m_text_stats.jsonl (and optionally cc12m_clip_stats.jsonl) from the
CC12M VAE cache directory, applies quality thresholds, and writes one global_idx
per line to the output file. Pass this file via `data.filtered_indices_file` in
config.yaml to train VaeCachedDataset on the quality-gated subset.

Usage:
    python scripts/prepare_cc12m.py \\
        [--cache-dir PATH] \\
        [--min-tokens 5] \\
        [--min-words 3] \\
        [--exclude-noisy] \\
        [--min-clip-score 0.22] \\
        [--out filtered_indices.txt]
"""

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_CACHE_DIR = "/media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--min-tokens", type=int, default=5,
                        help="Minimum CLIP BPE token count")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Minimum word count")
    parser.add_argument("--exclude-noisy", action="store_true", default=True,
                        help="Exclude samples flagged as noisy (URL/filename patterns)")
    parser.add_argument("--min-clip-score", type=float, default=None,
                        help="Minimum CLIP image-text cosine similarity (requires clip stats file)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: {cache-dir}/filtered_indices.txt)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    text_stats_path = cache_dir / "cc12m_text_stats.jsonl"
    clip_stats_path = cache_dir / "cc12m_clip_stats.jsonl"
    out_path = Path(args.out) if args.out else cache_dir / "filtered_indices.txt"

    if not text_stats_path.exists():
        print(f"[prepare_cc12m] Text stats not found: {text_stats_path}", file=sys.stderr)
        print("  Run: python scripts/score_cc12m.py first", file=sys.stderr)
        sys.exit(1)

    clip_scores: dict[int, float] = {}
    if args.min_clip_score is not None:
        if not clip_stats_path.exists():
            print(f"[prepare_cc12m] --min-clip-score set but clip stats not found: {clip_stats_path}",
                  file=sys.stderr)
            sys.exit(1)
        with clip_stats_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    clip_scores[obj["id"]] = obj["clip_score"]
        print(f"[prepare_cc12m] Loaded CLIP scores for {len(clip_scores):,} samples")

    kept: list[int] = []
    total = 0
    rejected_text = 0
    rejected_clip = 0

    with text_stats_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1
            gid = obj["id"]

            if obj["token_count"] < args.min_tokens:
                rejected_text += 1
                continue
            if obj["word_count"] < args.min_words:
                rejected_text += 1
                continue
            if args.exclude_noisy and obj.get("has_noise", False):
                rejected_text += 1
                continue

            if args.min_clip_score is not None:
                score = clip_scores.get(gid)
                if score is None or score < args.min_clip_score:
                    rejected_clip += 1
                    continue

            kept.append(gid)

    kept.sort()
    out_path.write_text("\n".join(str(i) for i in kept) + "\n", encoding="utf-8")

    kept_pct = len(kept) / max(total, 1) * 100
    print(f"[prepare_cc12m] {total:,} total samples")
    print(f"  Rejected by text quality: {rejected_text:,}")
    print(f"  Rejected by CLIP score:   {rejected_clip:,}")
    print(f"  Kept: {len(kept):,} ({kept_pct:.1f}%)")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
