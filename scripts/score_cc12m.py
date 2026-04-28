#!/usr/bin/env python3
"""
Score CC12M cached samples by text quality (CPU-only, runs during PickaPic training).

Reads captions.jsonl from the CC12M VAE cache and writes per-sample text statistics
to cc12m_text_stats.jsonl in the same directory. No GPU or image access required.

Output fields per line:
  id, token_count, word_count, avg_word_len, has_noise

Usage:
    python scripts/score_cc12m.py [--cache-dir PATH] [--resume]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import CLIPTokenizer


_NOISE_RE = re.compile(
    r"([\w.-]+\.(jpg|jpeg|png|gif|bmp|webp|tiff|svg)\b|https?://\S+|www\.\S+)",
    re.IGNORECASE,
)


def _score(caption: str, tokenizer) -> dict:
    caption = caption.strip()
    ids = tokenizer(caption, truncation=False)["input_ids"]
    token_count = max(len(ids) - 2, 0)  # strip BOS / EOS
    words = caption.split()
    word_count = len(words)
    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 2)
    has_noise = bool(
        _NOISE_RE.search(caption)
        or token_count < 5
        or word_count <= 1
    )
    return {
        "token_count": token_count,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
        "has_noise": has_noise,
    }


def _percentiles(values: list, qs=(10, 25, 50, 75, 90, 95, 99)) -> dict:
    values = sorted(values)
    n = len(values)
    return {f"p{q}": values[int(q / 100 * n)] for q in qs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="/media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train",
        help="CC12M VAE cache train split directory",
    )
    parser.add_argument(
        "--tokenizer",
        default="openai/clip-vit-large-patch14",
        help="HF tokenizer model ID (CLIP BPE for consistent token count)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output file, skipping already-scored IDs",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    captions_path = cache_dir / "captions.jsonl"
    out_path = cache_dir / "cc12m_text_stats.jsonl"

    if not captions_path.exists():
        print(f"[score_cc12m] captions.jsonl not found: {captions_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[score_cc12m] Loading tokenizer: {args.tokenizer}")
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer)

    already_done: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["id"])
        print(f"[score_cc12m] Resuming — {len(already_done):,} samples already scored")

    token_counts: list[int] = []
    word_counts: list[int] = []
    noise_count = 0
    processed = 0

    write_mode = "a" if args.resume else "w"
    with captions_path.open("r", encoding="utf-8") as fin, \
         out_path.open(write_mode, encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            gid = obj["id"]
            if gid in already_done:
                continue
            stats = _score(obj["caption"], tokenizer)
            fout.write(json.dumps({"id": gid, **stats}, ensure_ascii=False) + "\n")
            token_counts.append(stats["token_count"])
            word_counts.append(stats["word_count"])
            if stats["has_noise"]:
                noise_count += 1
            processed += 1
            if processed % 500_000 == 0:
                fout.flush()
                print(f"[score_cc12m] {processed + len(already_done):,} samples scored ...")

    total = processed + len(already_done)
    noise_pct = noise_count / max(processed, 1) * 100
    print(f"\n[score_cc12m] Done. {total:,} total samples scored.")
    print(f"  Noisy (this run): {noise_count:,} / {processed:,} = {noise_pct:.1f}%")
    if token_counts:
        print(f"  Token count: {_percentiles(token_counts)}")
        print(f"  Word count:  {_percentiles(word_counts)}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
