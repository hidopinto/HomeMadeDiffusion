"""In-place migration: individual .pt latent files → stacked shard files.

Converts millions of individual latent files into shard files of 1000 latents each,
reducing NTFS MFT pressure and preventing ENOSPC.

Safe to interrupt and resume — uses NVMe staging directory as crash-recovery buffer.

Usage:
    # Manual bootstrap (first iteration — NTFS has 0 free slots):
    python scripts/migrate_latent_shards.py --bootstrap

    # Full migration (after bootstrap freed slots):
    python scripts/migrate_latent_shards.py
"""
import argparse
import shutil
from pathlib import Path

import torch

SHARD_SIZE = 1000
STAGING_DIR = Path("/tmp/migration_staging")
LATENT_DIR = Path(
    "/media/hido-pinto/מחסן/vae_cache/pixparse--cc12m-wds/train/latents"
)


def _move_staged_shard(
    staging_path: Path, dst: Path, latent_dir: Path, batch_indices: list[int]
) -> None:
    """Copy staged shard to NTFS and clean up."""
    for idx in batch_indices:
        (latent_dir / f"{idx:06d}.pt").unlink(missing_ok=True)
    tmp = dst.with_suffix(".pt.tmp")
    shutil.copy2(staging_path, tmp)
    tmp.rename(dst)
    staging_path.unlink()


def bootstrap(latent_dir: Path) -> None:
    """Manual first iteration: stage shard on NVMe, print instructions for user."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = STAGING_DIR / "shard_000000.pt"
    dst = latent_dir / "shard_000000.pt"

    if dst.exists():
        print("shard_000000.pt already exists on NTFS — bootstrap not needed.")
        print("Run without --bootstrap to continue migration.")
        return

    if staging_path.exists():
        print(f"Staged shard found at {staging_path} from previous interrupted run.")
        print(f"Size: {staging_path.stat().st_size / 1e6:.1f} MB")
    else:
        files = sorted(
            [p for p in latent_dir.iterdir() if p.suffix == ".pt" and p.stem.isdigit()],
            key=lambda p: int(p.stem),
        )[:SHARD_SIZE]
        if not files:
            print("No individual .pt files found — nothing to migrate.")
            return
        print(f"Reading {len(files)} latents into memory...")
        tensors = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
        stacked = torch.stack(tensors)
        torch.save(stacked, staging_path)
        print(f"Shard staged at {staging_path} ({staging_path.stat().st_size / 1e6:.1f} MB)")

    first_idx = int(
        sorted(
            [p for p in latent_dir.iterdir() if p.suffix == ".pt" and p.stem.isdigit()],
            key=lambda p: int(p.stem),
        )[0].stem
    )
    last_idx = first_idx + SHARD_SIZE - 1

    print(f"\n{'='*60}")
    print("MANUAL STEPS (run these in your terminal):")
    print(f"{'='*60}")
    print()
    print(f"1. Delete the first {SHARD_SIZE} individual files from NTFS:")
    print(f'   cd "{latent_dir}"')
    print(f"   ls [0-9]*.pt | head -{SHARD_SIZE} | xargs rm")
    print()
    print("2. Verify writes work:")
    print(f'   touch "{latent_dir}/write_test" && rm "{latent_dir}/write_test"')
    print()
    print("3. Copy the shard into place:")
    print(f'   cp "{staging_path}" "{dst}"')
    print(f'   rm "{staging_path}"')
    print()
    print("4. Run the full automated migration:")
    print("   python scripts/migrate_latent_shards.py")
    print(f"{'='*60}")


def migrate(latent_dir: Path) -> None:
    """Migrate all remaining individual .pt files into shard files."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    individual_files = sorted(
        [p for p in latent_dir.iterdir() if p.suffix == ".pt" and p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if not individual_files:
        print("No individual .pt files found — migration already complete.")
        return

    all_indices = [int(p.stem) for p in individual_files]
    total = len(all_indices)
    print(f"Migrating {total:,} remaining latents into shards of {SHARD_SIZE}...")

    existing_shards = {
        int(p.stem.split("_")[1])
        for p in latent_dir.glob("shard_*.pt")
        if not p.name.endswith(".tmp")
    }

    processed = 0
    for batch_start in range(0, total, SHARD_SIZE):
        batch_indices = all_indices[batch_start : batch_start + SHARD_SIZE]
        batch_files = individual_files[batch_start : batch_start + SHARD_SIZE]
        shard_id = batch_indices[0] // SHARD_SIZE

        dst = latent_dir / f"shard_{shard_id:06d}.pt"

        if shard_id in existing_shards:
            for f in batch_files:
                f.unlink(missing_ok=True)
            processed += len(batch_indices)
            continue

        staging_path = STAGING_DIR / f"shard_{shard_id:06d}.pt"

        if staging_path.exists():
            _move_staged_shard(staging_path, dst, latent_dir, batch_indices)
            processed += len(batch_indices)
            continue

        tensors = [
            torch.load(f, map_location="cpu", weights_only=True) for f in batch_files
        ]
        stacked = torch.stack(tensors)
        torch.save(stacked, staging_path)

        for f in batch_files:
            f.unlink(missing_ok=True)

        tmp = dst.with_suffix(".pt.tmp")
        shutil.copy2(staging_path, tmp)
        tmp.rename(dst)
        staging_path.unlink()

        processed += len(batch_indices)
        if (processed // SHARD_SIZE) % 100 == 0:
            print(f"  {processed:,} / {total:,} migrated ({processed * 100 // total}%)")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    shard_count = len(list(latent_dir.glob("shard_*.pt")))
    print(f"Migration complete. {shard_count:,} shard files, 0 individual files remaining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate per-file latents to sharded format"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run manual first-iteration bootstrap (when NTFS has 0 free slots)",
    )
    parser.add_argument("--latent-dir", type=Path, default=LATENT_DIR)
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap(args.latent_dir)
    else:
        migrate(args.latent_dir)
