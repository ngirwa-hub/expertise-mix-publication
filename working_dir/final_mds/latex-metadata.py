#!/usr/bin/env python3
"""Generate a LaTeX table from intermediate role markdown files.

Columns:
- s/no
- file-name
- target-role
- job-title
- source
- link
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

EXPECTED_INPUT_FILES = [
    "generalist-midlevel.md",
    "generalist-senior.md",
    "sme-midlevel.md",
    "sme-senior.md",
    "normative-midlevel.md",
    "normative-senior.md",
]

JOB_HEADING_RE = re.compile(r"^##\s*job(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
LINK_RE = re.compile(
    r"^\s*link_job(\d+)\s*\([^)]*\)\s*:\s*(https?://\S+)\s*$",
    re.IGNORECASE,
)


@dataclass
class Row:
    file_name: str
    target_role: str
    job_title: str
    source: str
    link: str


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "intermediate-mds"
    default_output = script_dir / "intermediate_jobs_table.tex"

    parser = argparse.ArgumentParser(description="Generate LaTeX table for intermediate job inputs.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of markdown filenames to process.",
    )
    return parser.parse_args()


def resolve_targets(only_arg: str) -> List[str]:
    if not only_arg.strip():
        return sorted(EXPECTED_INPUT_FILES)

    requested = [p.strip() for p in only_arg.split(",") if p.strip()]
    unknown = sorted(set(requested) - set(EXPECTED_INPUT_FILES))
    if unknown:
        raise ValueError(f"Unknown file(s) in --only: {', '.join(unknown)}")
    return sorted(dict.fromkeys(requested))


def latex_escape(value: str) -> str:
    repl = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def target_role_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    track, level = stem.split("-", 1)
    track_label = {
        "generalist": "Generalist",
        "sme": "SME",
        "normative": "Normative",
    }.get(track.lower(), track.title())
    level_label = {
        "midlevel": "Mid-Level",
        "senior": "Senior",
    }.get(level.lower(), level.title())
    return f"{track_label} - {level_label}"


def source_from_link(link: str) -> str:
    netloc = urlparse(link).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def parse_rows_for_file(path: Path) -> List[Row]:
    lines = path.read_text(encoding="utf-8").splitlines()
    titles: Dict[int, str] = {}
    links: Dict[int, str] = {}

    for line in lines:
        m = JOB_HEADING_RE.match(line.strip())
        if m:
            titles[int(m.group(1))] = m.group(2).strip()
            continue

        m = LINK_RE.match(line.strip())
        if m:
            links[int(m.group(1))] = m.group(2).strip()

    rows: List[Row] = []
    file_name = path.name
    target_role = target_role_from_filename(file_name)
    for job_num in sorted(titles):
        job_title = titles[job_num]
        link = links.get(job_num, "")
        source = source_from_link(link) if link else ""
        rows.append(
            Row(
                file_name=file_name,
                target_role=target_role,
                job_title=job_title,
                source=source,
                link=link,
            )
        )
    return rows


def render_table(rows: List[Row]) -> str:
    header = [
        r"\\begin{tabular}{r l l l l l}",
        r"\\hline",
        r"S/No & File-Name & Target-Role & Job-Title & Source & Link \\",
        r"\\hline",
    ]

    body = []
    for i, row in enumerate(rows, start=1):
        body.append(
            "{} & {} & {} & {} & {} & \\url{{{}}} \\\\".format(
                i,
                latex_escape(row.file_name),
                latex_escape(row.target_role),
                latex_escape(row.job_title),
                latex_escape(row.source),
                row.link,
            )
        )

    footer = [r"\\hline", r"\\end{tabular}"]
    return "\n".join(header + body + footer) + "\n"


def main() -> int:
    args = parse_args()
    targets = resolve_targets(args.only)

    rows: List[Row] = []
    for file_name in targets:
        path = args.input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        rows.extend(parse_rows_for_file(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    preamble = "\n".join(
        [
            "% Auto-generated job table from intermediate markdown files",
            "% Requires: \\usepackage{url}",
            "",
        ]
    )
    content = preamble + render_table(rows)
    args.output.write_text(content, encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
