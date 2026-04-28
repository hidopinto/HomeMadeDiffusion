#!/usr/bin/env python3
"""
Score CC12M cached samples by CLIP image-text similarity (GPU, optional, ~25-30 hrs).

Streams pixparse/cc12m-wds from HF Hub in shard order (same order as the VAE cache),
computes CLIP cosine similarity between each image and its caption, and writes results
to cc12m_clip_stats.jsonl. Resume-safe: skips already-scored IDs on restart.

Only run this if the text-only stats from score_cc12m.py are insufficient for Phase 2
quality filtering.

Usage:
    python scripts/score_cc12m_clip.py [--cache-dir PATH] [--batch-size 1024] [--resume]
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


_DEFAULT_CACHE_DIR = "/media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train"
_DEFAULT_CLIP = "openai/clip-vit-large-patch14"
_DATASET = "pixparse/cc12m-wds"


def _load_model(model_id: str, device: str):
    print(f"[score_cc12m_clip] Loading CLIP: {model_id}")
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = model.to(device).eval()
    return model, processor


@torch.no_grad()
def _score_batch(
    images: list[Image.Image],
    captions: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> list[float]:
    inputs = processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    img_emb = F.normalize(outputs.image_embeds, dim=-1)
    txt_emb = F.normalize(outputs.text_embeds, dim=-1)
    sims = (img_emb * txt_emb).sum(dim=-1)
    return sims.float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--clip-model", default=_DEFAULT_CLIP)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    captions_path = cache_dir / "captions.jsonl"
    out_path = cache_dir / "cc12m_clip_stats.jsonl"

    if not captions_path.exists():
        print(f"[score_cc12m_clip] captions.jsonl not found: {captions_path}", file=sys.stderr)
        sys.exit(1)

    already_done: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already_done.add(json.loads(line)["id"])
        print(f"[score_cc12m_clip] Resuming — {len(already_done):,} already scored")

    captions: dict[int, str] = {}
    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if obj["id"] not in already_done:
                    captions[obj["id"]] = obj["caption"]
    print(f"[score_cc12m_clip] {len(captions):,} samples to score")

    if not captions:
        print("[score_cc12m_clip] Nothing to score.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_model(args.clip_model, device)

    try:
        import datasets as hf_datasets
    except ImportError:
        print("[score_cc12m_clip] Install `datasets` to stream from HF Hub.", file=sys.stderr)
        sys.exit(1)

    print(f"[score_cc12m_clip] Streaming {_DATASET} ...")
    ds = hf_datasets.load_dataset(_DATASET, split="train", streaming=True)
    ds = ds.cast_column("jpg", hf_datasets.Image(decode=False))

    batch_ids: list[int] = []
    batch_imgs: list[Image.Image] = []
    batch_caps: list[str] = []
    processed = 0
    skipped = 0

    write_mode = "a" if args.resume else "w"
    with out_path.open(write_mode, encoding="utf-8") as fout:
        for row in ds:
            gid = row.get("__key__")
            if gid is None:
                skipped += 1
                continue
            try:
                gid = int(gid)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if gid not in captions:
                continue

            img_bytes = row["jpg"]
            if isinstance(img_bytes, dict):
                img_bytes = img_bytes.get("bytes") or open(img_bytes["path"], "rb").read()
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                skipped += 1
                continue

            batch_ids.append(gid)
            batch_imgs.append(img)
            batch_caps.append(captions[gid])

            if len(batch_ids) == args.batch_size:
                scores = _score_batch(batch_imgs, batch_caps, model, processor, device)
                for sid, score in zip(batch_ids, scores):
                    fout.write(json.dumps({"id": sid, "clip_score": round(score, 4)}) + "\n")
                processed += len(batch_ids)
                batch_ids, batch_imgs, batch_caps = [], [], []
                if processed % 100_000 == 0:
                    fout.flush()
                    print(f"[score_cc12m_clip] {processed + len(already_done):,} scored ...")

        if batch_ids:
            scores = _score_batch(batch_imgs, batch_caps, model, processor, device)
            for sid, score in zip(batch_ids, scores):
                fout.write(json.dumps({"id": sid, "clip_score": round(score, 4)}) + "\n")
            processed += len(batch_ids)

    total = processed + len(already_done)
    print(f"\n[score_cc12m_clip] Done. {total:,} scored, {skipped:,} skipped.")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
