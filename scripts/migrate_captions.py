"""Migrate captions.jsonl from plain strings to {"id": N, "caption": "..."} format.

Usage:
    python scripts/migrate_captions.py /path/to/captions.jsonl

Safe to run on already-migrated files (passes through new-format lines unchanged).
Uses atomic rename so the file is never in a partial state.
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tmp = path.with_suffix(".jsonl.tmp")

count = 0
with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
    for i, line in enumerate(src):
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, str):
            dst.write(json.dumps({"id": i, "caption": value}, ensure_ascii=False) + "\n")
        else:
            dst.write(line + "\n")
        count += 1
        if count % 500_000 == 0:
            print(f"  {count:,} lines processed...")

tmp.rename(path)
print(f"Done. {count:,} lines migrated → {path}")
