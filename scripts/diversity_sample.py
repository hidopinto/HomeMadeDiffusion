#!/usr/bin/env python3
"""
Diversity-balanced sampling from a captioned dataset.

Embeds captions with a sentence-transformer, clusters with mini-batch K-means,
then caps each cluster to at most --target-per-cluster samples. This upsamples
rare categories (small clusters stay intact) and downsamples dominant ones (large
clusters are trimmed), producing a semantically diverse subset.

Optionally generates a UMAP scatter plot of before/after distributions — useful
for README visualisations.

Usage:
    # CC12M (apply after prepare_cc12m.py produces filtered_indices.txt)
    python scripts/diversity_sample.py \\
        --captions /media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train/captions.jsonl \\
        --filtered-indices /media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train/filtered_indices.txt \\
        --k-clusters 500 --target-per-cluster 800 \\
        --out /media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train/diverse_indices.txt \\
        --plot diverse_cc12m.png

    # PickaPic (after extract_pickapic_captions.py; no filter file needed)
    python scripts/diversity_sample.py \\
        --captions /media/hido-pinto/מחסן/cache/pickapic-anonymous--pickapic_v1/train/captions.jsonl \\
        --k-clusters 400 --target-per-cluster 500 \\
        --out /media/hido-pinto/מחסן/cache/pickapic-anonymous--pickapic_v1/train/diverse_indices.txt \\
        --plot diverse_pickapic.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _load_captions(captions_path: Path, filtered_ids: set[int] | None) -> dict[int, str]:
    captions: dict[int, str] = {}
    with captions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gid = int(obj["id"])
            if filtered_ids is None or gid in filtered_ids:
                captions[gid] = obj["caption"]
    return captions


def _embed(texts: list[str], model_name: str, batch_size: int) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"[diversity_sample] Embedding {len(texts):,} captions with {model_name} ...")
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def _cluster(embeddings: "np.ndarray", k: int, seed: int) -> "np.ndarray":
    from sklearn.cluster import MiniBatchKMeans
    print(f"[diversity_sample] Clustering into K={k} clusters (mini-batch K-means) ...")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3, max_iter=100)
    return km.fit_predict(embeddings)


def _balance(ids: list[int], labels: "np.ndarray", target_per_cluster: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    clusters: dict[int, list[int]] = {}
    for sample_id, label in zip(ids, labels.tolist()):
        clusters.setdefault(label, []).append(sample_id)
    kept: list[int] = []
    for cluster_ids in clusters.values():
        if len(cluster_ids) <= target_per_cluster:
            kept.extend(cluster_ids)
        else:
            kept.extend(rng.sample(cluster_ids, target_per_cluster))
    kept.sort()
    return kept


def _umap_plot(
    embeddings: "np.ndarray",
    labels: "np.ndarray",
    kept_set: set[int],
    all_ids: list[int],
    out_path: Path,
    subsample: int,
    seed: int,
) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap as umap_module

    n = len(all_ids)
    rng = np.random.default_rng(seed)
    if n > subsample:
        idx = rng.choice(n, subsample, replace=False)
    else:
        idx = np.arange(n)

    sub_emb = embeddings[idx]
    sub_labels = labels[idx]
    sub_in_kept = np.array([all_ids[i] in kept_set for i in idx])

    print(f"[diversity_sample] Running UMAP on {len(idx):,} samples ...")
    reducer = umap_module.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.05)
    coords = reducer.fit_transform(sub_emb)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    axes[0].scatter(coords[:, 0], coords[:, 1],
                    c=sub_labels, cmap="tab20", s=1.5, alpha=0.4, linewidths=0)
    axes[0].set_title(f"Full dataset (subsample)\n({len(idx):,} samples)", fontsize=11)

    axes[1].scatter(coords[~sub_in_kept, 0], coords[~sub_in_kept, 1],
                    c="lightgray", s=1.0, alpha=0.25, linewidths=0, zorder=0, label="removed")
    axes[1].scatter(coords[sub_in_kept, 0], coords[sub_in_kept, 1],
                    c=sub_labels[sub_in_kept], cmap="tab20", s=2, alpha=0.6,
                    linewidths=0, zorder=1, label="kept")
    axes[1].set_title(f"Diversity-sampled subset\n({sub_in_kept.sum():,} shown)", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Caption semantic space (UMAP, colored by K-means cluster)", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[diversity_sample] UMAP plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diversity-balanced dataset sampling via caption K-means clustering"
    )
    parser.add_argument("--captions", required=True,
                        help="captions.jsonl: one {id, caption} JSON per line")
    parser.add_argument("--filtered-indices", default=None,
                        help="Optional filtered_indices.txt to subset before clustering")
    parser.add_argument("--k-clusters", type=int, default=500)
    parser.add_argument("--target-per-cluster", type=int, default=800,
                        help="Max samples kept per cluster")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name")
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--out", required=True,
                        help="Output path for balanced indices (one ID per line)")
    parser.add_argument("--plot", default=None,
                        help="Optional path for UMAP scatter plot PNG")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip sampling; load existing --out file and generate --plot only. "
                             "Embeds --umap-subsample captions instead of the full set (~1-2 min).")
    parser.add_argument("--umap-subsample", type=int, default=100_000,
                        help="Max samples used for UMAP visualisation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    captions_path = Path(args.captions)
    if not captions_path.exists():
        print(f"[diversity_sample] captions.jsonl not found: {captions_path}", file=sys.stderr)
        sys.exit(1)

    filtered_ids: set[int] | None = None
    if args.filtered_indices:
        fpath = Path(args.filtered_indices)
        filtered_ids = {int(l.strip()) for l in fpath.read_text().splitlines() if l.strip()}
        print(f"[diversity_sample] Loaded {len(filtered_ids):,} filtered indices")

    # ── Fast plot-only path ──────────────────────────────────────────────────
    if args.plot_only:
        if not args.plot:
            print("[diversity_sample] --plot-only requires --plot <path>", file=sys.stderr)
            sys.exit(1)
        import numpy as np

        captions = _load_captions(captions_path, filtered_ids)
        all_ids = sorted(captions.keys())
        print(f"[diversity_sample] plot-only: {len(all_ids):,} IDs available")

        kept_set: set[int] = set()
        if args.out and Path(args.out).exists():
            kept_set = {int(l.strip()) for l in Path(args.out).read_text().splitlines() if l.strip()}
            print(f"[diversity_sample] Loaded {len(kept_set):,} kept indices from {args.out}")

        # Subsample for UMAP — embed only this many captions instead of the full set
        rng = np.random.default_rng(args.seed)
        n = min(args.umap_subsample, len(all_ids))
        sub_ids = [all_ids[i] for i in rng.choice(len(all_ids), n, replace=False)]
        sub_texts = [captions[i] for i in sub_ids]

        print(f"[diversity_sample] Embedding {n:,} captions for UMAP ...")
        sub_emb = _embed(sub_texts, args.model, args.embed_batch_size)
        sub_labels = _cluster(sub_emb, min(args.k_clusters, n // 10), args.seed)
        _umap_plot(sub_emb, sub_labels, kept_set, sub_ids, Path(args.plot),
                   n, args.seed)
        return

    # ── Full sampling path ───────────────────────────────────────────────────
    captions = _load_captions(captions_path, filtered_ids)
    print(f"[diversity_sample] {len(captions):,} captions loaded")

    if not captions:
        print("[diversity_sample] No captions to process.", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    ids = sorted(captions.keys())
    texts = [captions[i] for i in ids]

    embeddings = _embed(texts, args.model, args.embed_batch_size)
    labels = _cluster(embeddings, args.k_clusters, args.seed)

    cluster_sizes = np.bincount(labels, minlength=args.k_clusters)
    print(
        f"[diversity_sample] Cluster sizes — "
        f"min={cluster_sizes.min()}, "
        f"p25={int(np.percentile(cluster_sizes, 25))}, "
        f"median={int(np.median(cluster_sizes))}, "
        f"p75={int(np.percentile(cluster_sizes, 75))}, "
        f"max={cluster_sizes.max()}"
    )
    large = (cluster_sizes > args.target_per_cluster).sum()
    print(f"[diversity_sample] {large} / {args.k_clusters} clusters exceed target_per_cluster "
          f"({args.target_per_cluster}) and will be trimmed")

    kept = _balance(ids, labels, args.target_per_cluster, args.seed)
    print(f"[diversity_sample] Kept {len(kept):,} / {len(ids):,} samples "
          f"({len(kept) / len(ids) * 100:.1f}%)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(i) for i in kept) + "\n", encoding="utf-8")
    print(f"[diversity_sample] Indices written to {out_path}")

    if args.plot:
        _umap_plot(embeddings, labels, set(kept), ids, Path(args.plot),
                   args.umap_subsample, args.seed)


if __name__ == "__main__":
    main()
