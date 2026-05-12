#!/usr/bin/env python3
r"""Build embeddings for generated role markdown files.

This script reads role markdown files (for example final-*-run1.md), computes
one embedding per file, and writes a JSON artifact consumed by role-similarity.py.

run commands: 
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-roles/role-embeddings.py --overwrite

# source run files location: /Users/HP/Documents/scrapping/working_dir/final_mds/llm-generated-roles


"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


FILE_RE = re.compile(
    r"^final-(generalist|sme|normative)-(midlevel|senior)(?:-([A-Za-z0-9._-]+))?\.md$",
    re.IGNORECASE,
)


@dataclass
class EmbeddingRecord:
    file_name: str
    file_path: str
    track: str
    level: str
    run_tag: str | None
    role_key: str
    embedding_model: str
    embedding_dimensions: int
    char_count: int
    embedding: List[float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "final_mds" / "llm-generated-roles"
    default_output = default_input / "role-embeddings.json"

    parser = argparse.ArgumentParser(
        description="Create embeddings for role markdown files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--glob",
        type=str,
        default="final-*.md",
        help="Glob pattern for role markdown files in input dir.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence Transformers embedding model.",
    )
    parser.add_argument("--output-file", type=Path, default=default_output)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of filenames to embed.",
    )
    return parser.parse_args()


def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers not available. Install with: pip install sentence-transformers"
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}': {exc}"
        ) from exc


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


def resolve_targets(input_dir: Path, pattern: str, only_arg: str) -> List[Path]:
    all_candidates = sorted(input_dir.glob(pattern))
    valid_candidates = []
    for path in all_candidates:
        if not path.is_file():
            continue
        if FILE_RE.match(path.name):
            valid_candidates.append(path)

    if not only_arg.strip():
        return valid_candidates

    requested = [name.strip() for name in only_arg.split(",") if name.strip()]
    requested_set = set(requested)
    present = {path.name for path in valid_candidates}
    missing = sorted(requested_set - present)
    if missing:
        raise ValueError(
            "Unknown file(s) in --only (or not matching expected name pattern): "
            + ", ".join(missing)
        )
    return [path for path in valid_candidates if path.name in requested_set]


def create_embedding(model, text: str) -> List[float]:
    vector = model.encode(text, convert_to_numpy=True)
    if vector is None or len(vector) == 0:
        raise RuntimeError("Embedding vector missing from model output.")
    return vector.tolist()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_file = args.output_file.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    try:
        targets = resolve_targets(input_dir=input_dir, pattern=args.glob, only_arg=args.only)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("ERROR: no matching role files found to embed.", file=sys.stderr)
        return 2

    if output_file.exists() and not args.overwrite:
        print(
            f"ERROR: output file exists: {output_file} (use --overwrite to replace).",
            file=sys.stderr,
        )
        return 2

    print(f"Input dir:    {input_dir}")
    print(f"Output file:  {output_file}")
    print(f"Model:        {args.model}")
    print(f"Files:        {len(targets)}")
    for path in targets:
        print(f"  - {path.name}")

    if args.dry_run:
        print("Dry run complete. No embeddings requested.")
        return 0

    try:
        embedding_model = load_embedding_model(args.model)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    records: List[EmbeddingRecord] = []
    failures: List[Dict[str, str]] = []

    for path in targets:
        try:
            identity = parse_identity(path.name)
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                raise RuntimeError("File is empty.")

            vector = create_embedding(model=embedding_model, text=content)
            record = EmbeddingRecord(
                file_name=path.name,
                file_path=str(path.resolve()),
                track=str(identity["track"]),
                level=str(identity["level"]),
                run_tag=identity["run_tag"],
                role_key=str(identity["role_key"]),
                embedding_model=args.model,
                embedding_dimensions=len(vector),
                char_count=len(content),
                embedding=vector,
            )
            records.append(record)
            print(f"[OK]   {path.name}: embedded ({len(vector)} dims)")
        except Exception as exc:
            failures.append({"file_name": path.name, "error": str(exc)})
            print(f"[FAIL] {path.name}: {exc}", file=sys.stderr)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "embedding_model": args.model,
        "file_count_requested": len(targets),
        "file_count_embedded": len(records),
        "file_count_failed": len(failures),
        "records": [asdict(record) for record in records],
        "failures": failures,
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Embeddings written: {output_file}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
