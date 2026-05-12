#!/usr/bin/env python3
r"""Compute pairwise role similarity from saved embeddings.

Pairing policy:
- Never mix track or level.
- Compare only within the same role_key (track + level).
- By default, compare only across different run tags.

run commands:
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-roles/role-similarity.py --overwrite


"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


FILE_RE = re.compile(
    r"^final-(generalist|sme|normative)-(midlevel|senior)(?:-([A-Za-z0-9._-]+))?\.md$",
    re.IGNORECASE,
)


@dataclass
class EmbeddingItem:
    file_name: str
    file_path: str
    track: str
    level: str
    run_tag: str | None
    role_key: str
    embedding: List[float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir.parent / "final_mds" / "llm-generated-roles"
    default_embeddings = default_dir / "role-embeddings.json"
    default_json = default_dir / "role-similarity.json"
    default_csv = default_dir / "role-similarity.csv"

    parser = argparse.ArgumentParser(
        description="Compute strict role pair similarities from embeddings."
    )
    parser.add_argument("--embeddings-file", type=Path, default=default_embeddings)
    parser.add_argument("--output-json", type=Path, default=default_json)
    parser.add_argument("--output-csv", type=Path, default=default_csv)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-untagged",
        action="store_true",
        help="Include files without run tag. Default excludes them.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=-1.0,
        help="Optional cosine threshold filter.",
    )
    args = parser.parse_args()
    if not (-1.0 <= args.min_similarity <= 1.0):
        parser.error("--min-similarity must be between -1.0 and 1.0")
    return args


def parse_identity(file_name: str) -> Dict[str, str | None]:
    match = FILE_RE.match(file_name)
    if not match:
        raise ValueError(
            f"Filename does not match expected role pattern: {file_name}"
        )
    track, level, run_tag = match.groups()
    track = track.lower()
    level = level.lower()
    return {
        "track": track,
        "level": level,
        "run_tag": run_tag,
        "role_key": f"{track}-{level}",
    }


def load_embedding_items(path: Path) -> List[EmbeddingItem]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        records = raw.get("records", [])
    elif isinstance(raw, list):
        records = raw
    else:
        raise RuntimeError("Unsupported embeddings JSON format.")

    items: List[EmbeddingItem] = []
    for rec in records:
        file_name = str(rec.get("file_name", "")).strip()
        if not file_name:
            raise RuntimeError("Embeddings record missing file_name.")

        identity = {
            "track": rec.get("track"),
            "level": rec.get("level"),
            "run_tag": rec.get("run_tag"),
            "role_key": rec.get("role_key"),
        }
        if not identity["track"] or not identity["level"] or not identity["role_key"]:
            identity = parse_identity(file_name)

        embedding = rec.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(f"Record for {file_name} has invalid embedding.")

        items.append(
            EmbeddingItem(
                file_name=file_name,
                file_path=str(rec.get("file_path", "")),
                track=str(identity["track"]).lower(),
                level=str(identity["level"]).lower(),
                run_tag=identity["run_tag"],
                role_key=str(identity["role_key"]).lower(),
                embedding=[float(v) for v in embedding],
            )
        )

    return items


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Cannot compute cosine similarity for vectors of different lengths.")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


def main() -> int:
    args = parse_args()
    embeddings_file = args.embeddings_file.resolve()
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()

    if not embeddings_file.exists() or not embeddings_file.is_file():
        print(f"ERROR: embeddings file not found: {embeddings_file}", file=sys.stderr)
        return 2

    for path in [output_json, output_csv]:
        if path.exists() and not args.overwrite:
            print(
                f"ERROR: output exists: {path} (use --overwrite to replace).",
                file=sys.stderr,
            )
            return 2

    try:
        items = load_embedding_items(embeddings_file)
    except Exception as exc:
        print(f"ERROR: failed to load embeddings: {exc}", file=sys.stderr)
        return 2

    if not args.include_untagged:
        items = [item for item in items if item.run_tag]

    if not items:
        print("ERROR: no embedding items available after filters.", file=sys.stderr)
        return 2

    groups: Dict[str, List[EmbeddingItem]] = defaultdict(list)
    for item in items:
        groups[item.role_key].append(item)

    role_keys = sorted(groups.keys())
    print(f"Embeddings file: {embeddings_file}")
    print(f"Role groups:     {len(role_keys)}")
    for role_key in role_keys:
        print(f"  - {role_key}: {len(groups[role_key])} item(s)")

    pair_rows: List[Dict[str, Any]] = []
    for role_key in role_keys:
        role_items = sorted(groups[role_key], key=lambda x: (str(x.run_tag), x.file_name))
        for idx in range(len(role_items)):
            for jdx in range(idx + 1, len(role_items)):
                left = role_items[idx]
                right = role_items[jdx]

                # Strict rule: if both have run tags, only compare cross-run.
                if left.run_tag and right.run_tag and left.run_tag == right.run_tag:
                    continue

                score = cosine_similarity(left.embedding, right.embedding)
                if score < args.min_similarity:
                    continue

                pair_rows.append(
                    {
                        "role_key": role_key,
                        "track": left.track,
                        "level": left.level,
                        "file_a": left.file_name,
                        "run_a": left.run_tag or "",
                        "file_b": right.file_name,
                        "run_b": right.run_tag or "",
                        "cosine_similarity": score,
                    }
                )

    pair_rows.sort(
        key=lambda row: (
            row["role_key"],
            row["run_a"],
            row["run_b"],
            row["file_a"],
            row["file_b"],
        )
    )

    print(f"Pairs formed:    {len(pair_rows)}")
    if args.dry_run:
        print("Dry run complete. No files written.")
        return 0

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": utc_now_iso(),
        "embeddings_file": str(embeddings_file),
        "include_untagged": args.include_untagged,
        "min_similarity": args.min_similarity,
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role_key",
                "track",
                "level",
                "file_a",
                "run_a",
                "file_b",
                "run_b",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        for row in pair_rows:
            writer.writerow(
                {
                    **row,
                    "cosine_similarity": f"{row['cosine_similarity']:.8f}",
                }
            )

    print(f"Similarity JSON: {output_json}")
    print(f"Similarity CSV:  {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
