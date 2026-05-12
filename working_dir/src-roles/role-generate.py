#!/usr/bin/env python3
r"""Generate finalized role markdown files from raw role inputs via OpenAI API.

## Commands per run:
set -a
source /Users/HP/Documents/scrapping/openai-api.env
set +a
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src/role-generate.py --overwrite --temperature 0.2 --run-tag run1
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src/role-generate.py --overwrite --temperature 0.2 --run-tag run2
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src/role-generate.py --overwrite --temperature 0.2 --run-tag run3

Output stats:


"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


EXPECTED_INPUT_FILES = [
    "generalist-midlevel.md",
    "generalist-senior.md",
    "sme-midlevel.md",
    "sme-senior.md",
    "normative-midlevel.md",
    "normative-senior.md",
]

SYSTEM_PROMPT = """You are an expert role-synthesis assistant.

Your task is to synthesize one consolidated role from a single raw markdown input file.
The raw file contains multiple job postings under headings in this format: ## jobN: <Job Title>.

Hard requirements:
1) Use only the provided raw file content. Do not use external facts.
2) Synthesize a clear role name and output it in the **Name** field.
3) Do not output placeholder names such as "Role Name", "TBD", or "Untitled".
4) The role name and all sections must be grounded in the provided jobs.
5) Do not mention or compare to other role files.
6) Job headings (## jobN: <Job Title>) are authoritative for role synthesis.
7) Ignore any source/reference link lines (e.g., link_jobN and raw URLs). They are metadata only.
8) Output valid markdown only, with no preface.
9) Respect the output token budget provided in the user instruction for this file.
10) The generated profile must align with the input file identity from filename metadata.
11) Treat filename identity as the primary anchor for track/level interpretation when source signals are mixed.
12) Keep synthesis flexible in content detail, but do not drift to a different role family than the input file identity.



Output structure (in this exact section order):
# SOUL.md
**Name:** <Synthesized Role Name>
**Role:** <Track> — <Level> (<Short role focus phrase>)

## Personality
## What You're Good At
## What You Care About

Style:
- concise, specific, and professional
- synthesize repeated themes into coherent bullets
- avoid copying long text verbatim from inputs
- avoid generic filler (e.g., "hardworking", "team player") unless grounded in repeated source evidence
- make the profile level-appropriate (Mid-Level vs Senior) and track-appropriate (Generalist / SME / Normative)

Section expectations:
## Personality
- 1-3 short paragraphs that describe working style, decision-making approach, and operating context.
- Explain how this role balances technical depth, stakeholder interaction, and execution ownership.
- Reflect the dominant environment from source jobs (for example: project delivery, regulatory environments, field/service operations, or analytical/reporting settings).

## What You're Good At
- Use grouped bullets (optionally with mini sub-headings) covering core capability clusters.
- Prioritize concrete capabilities repeatedly observed across job entries (methods, systems, workflows, tooling, governance, delivery).
- Include practical examples in parentheses when useful (e.g., standards, analyses, reporting artifacts, engineering deliverables).
- Senior profiles should emphasize leadership, ownership, and cross-functional influence; mid-level profiles should emphasize applied execution and reliable delivery.

## What You Care About
- Focus on professional priorities and quality bars implied by source jobs.
- Cover outcomes and principles such as safety, compliance, reliability, performance, stakeholder trust, data quality, and continuous improvement.
- Keep this section specific to the role domain; avoid motivational fluff or personal-life values.
"""

USER_PROMPT_TEMPLATE = """Generate the final consolidated role markdown from this source file.

Source filename: {filename}

Expected identity from filename:
- Track: {expected_track}
- Level: {expected_level}

Instruction:
- Synthesize the role from source content while staying aligned to this file identity.
- Keep the profile centered on the expected track/level above.

Output token budget: at most {token_cap} tokens.

Raw source content:

{content}
"""

JOB_HEADING_RE = re.compile(r"^##\s*job(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
JOB_PREFIX_RE = re.compile(r"^##\s*job\d+\b", re.IGNORECASE)
SOUL_TITLE_RE = re.compile(r"^#\s+SOUL\.md\s*$", re.IGNORECASE)
NAME_LINE_RE = re.compile(r"^\*\*Name:\*\*\s*(.+?)\s*$", re.IGNORECASE)
ROLE_LINE_RE = re.compile(r"^\*\*Role:\*\*\s*(.+?)\s*$", re.IGNORECASE)
PLACEHOLDER_NAME_RE = re.compile(r"^(role\s*name|tbd|untitled)\b", re.IGNORECASE)
LINK_JOB_LINE_RE = re.compile(r"^\s*link_job\d*\b", re.IGNORECASE)
URL_ONLY_LINE_RE = re.compile(r"^\s*https?://\S+\s*$", re.IGNORECASE)
LINK_DESCRIPTOR_RE = re.compile(
    r"^\s*link_job(\d+)\s*\(([^)]*?)\)\s*:\s*https?://\S+\s*$",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    jobs_found: int


@dataclass
class ProcessResult:
    input_file: str
    output_file: str
    status: str
    error: str | None = None
    jobs_found: int | None = None
    warnings: List[str] | None = None
    generated_tokens: int | None = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "intermediate-mds"
    default_output = script_dir.parent / "final_mds" / "llm-generated-roles"

    parser = argparse.ArgumentParser(
        description="Generate finalized roles from intermediate markdown files using OpenAI API."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model to use for generation.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of input filenames to process.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=500,
        help="Max tokens requested from the model for each generated role.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional tag appended to output filenames (e.g., run1, run2).",
    )

    parser.add_argument(
        "--temperature", 
        type=float,
        default=0.2,
        help="Sampling temperature for generation (lower is more deterministic)."
    )
    args = parser.parse_args()
    if not (0.0 <= args.temperature <= 0.2):
        parser.error("--temperature must be between 0.0 and 0.2")

    return args


def resolve_targets(only_arg: str) -> List[str]:
    if not only_arg.strip():
        return sorted(EXPECTED_INPUT_FILES)

    requested = [p.strip() for p in only_arg.split(",") if p.strip()]
    unknown = sorted(set(requested) - set(EXPECTED_INPUT_FILES))
    if unknown:
        raise ValueError(f"Unknown file(s) in --only: {', '.join(unknown)}")
    return sorted(dict.fromkeys(requested))


def normalize_title(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def validate_input_file(path: Path) -> ValidationResult:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    errors: List[str] = []
    warnings: List[str] = []
    expected_next = 1
    jobs_found = 0
    job_titles: Dict[int, str] = {}
    link_descriptors: List[Tuple[int, int, str]] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not JOB_PREFIX_RE.match(stripped):
            continue

        m = JOB_HEADING_RE.match(stripped)
        if not m:
            errors.append(
                f"Line {idx}: malformed job heading. Expected '## jobN: <Job Title>', got '{stripped}'."
            )
            continue

        jobs_found += 1
        job_num = int(m.group(1))
        job_title = m.group(2).strip()
        job_titles[job_num] = job_title

        if job_num != expected_next:
            errors.append(
                f"Line {idx}: expected job{expected_next} but found job{job_num}."
            )
            expected_next = job_num + 1
        else:
            expected_next += 1

        if not job_title:
            errors.append(
                f"Line {idx}: empty job title after ':'. Expected '## jobN: <Job Title>'."
            )

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        m = LINK_DESCRIPTOR_RE.match(stripped)
        if not m:
            continue
        link_job_num = int(m.group(1))
        link_title = m.group(2).strip()
        link_descriptors.append((idx, link_job_num, link_title))

    for idx, link_job_num, link_title in link_descriptors:
        canonical = job_titles.get(link_job_num)
        if not canonical:
            warnings.append(
                f"Line {idx}: link_job{link_job_num} has no matching '## job{link_job_num}: ...' heading."
            )
            continue
        if normalize_title(canonical) != normalize_title(link_title):
            warnings.append(
                f"Line {idx}: link_job{link_job_num} descriptor '{link_title}' differs from authoritative heading title '{canonical}'."
            )

    if jobs_found == 0:
        errors.append("No valid job headings found. Expected at least one '## job1: <Job Title>'.")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, jobs_found=jobs_found)

def build_output_filename(input_name: str, run_tag: str = "") -> str:
    stem = f"final-{Path(input_name).stem}"
    return f"{stem}-{run_tag}.md" if run_tag else f"{stem}.md"


def expected_identity_for_filename(filename: str) -> Tuple[str, str]:
    stem = filename.rsplit(".", 1)[0].lower()
    if stem.startswith("generalist-"):
        track = "Generalist"
    elif stem.startswith("sme-"):
        track = "SME"
    elif stem.startswith("normative-"):
        track = "Normative"
    else:
        raise ValueError(f"Cannot derive track from filename: {filename}")

    if stem.endswith("-midlevel"):
        level = "Mid-Level"
    elif stem.endswith("-senior"):
        level = "Senior"
    else:
        raise ValueError(f"Cannot derive level from filename: {filename}")

    return track, level


def sanitize_raw_content(raw_content: str) -> str:
    """Remove reference-only link lines before sending text to the model."""
    kept_lines: List[str] = []
    for line in raw_content.splitlines():
        stripped = line.strip()
        if LINK_JOB_LINE_RE.match(stripped):
            continue
        if URL_ONLY_LINE_RE.match(stripped):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip() + "\n"


def count_tokens(text: str) -> int | None:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return None


def ensure_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenAI SDK not available. Install with: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Load /Users/HP/Documents/scrapping/openai-api.env first."
        )

    return OpenAI(api_key=api_key)


def extract_text_from_response(response) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)

    merged = "\n".join(parts).strip()
    if not merged:
        raise RuntimeError("Model response was empty.")
    return merged


def generate_role_markdown(
    client,
    model: str,
    filename: str,
    expected_track: str,
    expected_level: str,
    raw_content: str,
    max_output_tokens: int,
    token_cap: int,
    temperature: float,
) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=filename,
        expected_track=expected_track,
        expected_level=expected_level,
        content=raw_content,
        token_cap=token_cap,
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return extract_text_from_response(response)


def validate_generated_markdown(markdown: str) -> Tuple[bool, str | None]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    if not SOUL_TITLE_RE.match(first_line):
        return False, "Generated markdown must start with '# SOUL.md'."

    name_line = next((line for line in lines if NAME_LINE_RE.match(line)), "")
    if not name_line:
        return False, "Generated markdown is missing '**Name:** <...>'."

    name_value = NAME_LINE_RE.match(name_line).group(1).strip()  # type: ignore[union-attr]
    if not name_value or PLACEHOLDER_NAME_RE.match(name_value):
        return False, f"Generated placeholder Name detected: '{name_value}'."

    role_line = next((line for line in lines if ROLE_LINE_RE.match(line)), "")
    if not role_line:
        return False, "Generated markdown is missing '**Role:** <...>'."

    required_sections = [
        "## Personality",
        "## What You're Good At",
        "## What You Care About",
    ]
    for section in required_sections:
        if section not in markdown:
            return False, f"Generated markdown is missing required section: '{section}'."
    return True, None


def check_role_identity(markdown: str, expected_track: str, expected_level: str) -> str | None:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    role_line = next((line for line in lines if ROLE_LINE_RE.match(line)), "")
    if not role_line:
        return "Missing '**Role:** line."
    role_value = ROLE_LINE_RE.match(role_line).group(1).strip()  # type: ignore[union-attr]
    role_lower = role_value.lower()
    if expected_track.lower() not in role_lower or expected_level.lower() not in role_lower:
        return (
            f"Role identity mismatch: got '{role_value}', expected to include "
            f"'{expected_track}' and '{expected_level}'."
        )
    return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    try:
        targets = resolve_targets(args.only)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 2

    missing = [name for name in targets if not (input_dir / name).exists()]
    if missing:
        print("ERROR: missing required input file(s):", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        return 2

    if args.dry_run:
        print("Dry run: validated directory and target file list.")
        print(f"Input dir:  {input_dir}")
        print(f"Output dir: {output_dir}")
        print(f"Model:      {args.model}")
        print("Targets:")
        for name in targets:
            print(f"  - {name} -> {build_output_filename(name)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            client = ensure_openai_client()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    results: List[ProcessResult] = []

    for name in targets:
        input_path = input_dir / name
        output_name = build_output_filename(name, args.run_tag)
        output_path = output_dir / output_name
        expected_track, expected_level = expected_identity_for_filename(name)

        validation = validate_input_file(input_path)
        if not validation.ok:
            results.append(
                ProcessResult(
                    input_file=name,
                    output_file=output_name,
                    status="failed_validation",
                    error=" | ".join(validation.errors),
                    jobs_found=validation.jobs_found,
                    warnings=validation.warnings or None,
                )
            )
            print(f"[FAIL] {name}: validation failed.", file=sys.stderr)
            for err in validation.errors:
                print(f"       - {err}", file=sys.stderr)
            for warn in validation.warnings:
                print(f"       [warn] {warn}", file=sys.stderr)
            continue

        if validation.warnings:
            print(f"[WARN] {name}: {len(validation.warnings)} link/title mismatch warning(s).")
            for warn in validation.warnings:
                print(f"       - {warn}")

        if output_path.exists() and not args.overwrite:
            results.append(
                ProcessResult(
                    input_file=name,
                    output_file=output_name,
                    status="skipped_exists",
                    jobs_found=validation.jobs_found,
                    warnings=validation.warnings or None,
                )
            )
            print(f"[SKIP] {name}: output exists (use --overwrite to replace).")
            continue

        if args.dry_run:
            results.append(
                ProcessResult(
                    input_file=name,
                    output_file=output_name,
                    status="dry_run_ok",
                    jobs_found=validation.jobs_found,
                    warnings=validation.warnings or None,
                )
            )
            print(f"[OK]   {name}: validation passed ({validation.jobs_found} jobs).")
            continue

        try:
            raw_content = input_path.read_text(encoding="utf-8")
            sanitized_content = sanitize_raw_content(raw_content)
            markdown = generate_role_markdown(
                client=client,
                model=args.model,
                filename=name,
                expected_track=expected_track,
                expected_level=expected_level,
                raw_content=sanitized_content,
                max_output_tokens=args.max_output_tokens,
                token_cap=args.max_output_tokens,
                temperature=args.temperature,
            )

            md_ok, md_error = validate_generated_markdown(markdown)
            if not md_ok:
                raise RuntimeError(md_error)

            token_count = count_tokens(markdown)
            generation_warnings: List[str] = list(validation.warnings or [])
            if token_count is not None and token_count > args.max_output_tokens:
                generation_warnings.append(
                    f"Token count {token_count} exceeds configured cap {args.max_output_tokens} by local estimator."
                )
            identity_warning = check_role_identity(markdown, expected_track, expected_level)
            if identity_warning:
                generation_warnings.append(identity_warning)
                print(f"[WARN] {name}: {identity_warning}")

            output_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            results.append(
                ProcessResult(
                    input_file=name,
                    output_file=output_name,
                    status="generated",
                    jobs_found=validation.jobs_found,
                    warnings=generation_warnings or None,
                    generated_tokens=token_count,
                )
            )
            print(f"[OK]   {name}: generated -> {output_name}")
        except Exception as exc:
            results.append(
                ProcessResult(
                    input_file=name,
                    output_file=output_name,
                    status="failed_generation",
                    error=str(exc),
                    jobs_found=validation.jobs_found,
                    warnings=validation.warnings or None,
                    generated_tokens=None,
                )
            )
            print(f"[FAIL] {name}: generation error: {exc}", file=sys.stderr)

    summary = {
        "timestamp_utc": utc_now_iso(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "targets": targets,
        "results": [r.__dict__ for r in results],
        "temperature": args.temperature,
        "run_tag": args.run_tag,
    }

    summary_path = output_dir / "run-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written: {summary_path}")

    failed = [r for r in results if r.status.startswith("failed")]
    if failed:
        print(f"Completed with failures: {len(failed)} file(s).", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
