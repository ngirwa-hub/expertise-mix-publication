r"""
.\.venv\Scripts\python.exe rag_pipeline\roles-enhanced\analysis\clean_barriers.py

"""
import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "expert_responses" / "context_barrier_mention_all_recovered.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV file.")
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for analysis outputs.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_text(text):
    text = text or ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\ufffd", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_list_prefix(text):
    text = clean_text(text)
    text = re.sub(r"^[\-\*\u2022]+\s*", "", text)
    text = re.sub(r"^\s*\d+[\).:-]?\s*", "", text)
    return text.strip()


def strip_title_prefix(text):
    return re.sub(r"^(?:title\s*:\s*)+", "", text, flags=re.IGNORECASE).strip()


def strip_title_suffix(text):
    return re.sub(r"\s*\(title\)\s*$", "", text, flags=re.IGNORECASE).strip()


def strip_explanation_prefix(text):
    return re.sub(r"^(?:explanation\s*:\s*)+", "", text, flags=re.IGNORECASE).strip()


def extract_parenthetical_title(title_text):
    match = re.match(r"^(.*?)\s*\(title:\s*(.+?)\)\s*$", title_text, flags=re.IGNORECASE)
    if not match:
        return title_text, ""

    explanation_part = match.group(1).strip(" .;:-")
    title_part = match.group(2).strip(" .;:-")
    return title_part, explanation_part


def maybe_split_inline_title_explanation(title_text, explanation_text):
    if explanation_text:
        return title_text, explanation_text

    parts = re.split(r"\s*:\s*", title_text, maxsplit=1)
    if len(parts) != 2:
        return title_text, explanation_text

    candidate_title, candidate_explanation = parts
    candidate_title = candidate_title.strip(" .;:-")
    candidate_explanation = candidate_explanation.strip()

    # Recover rows where the model merged title and explanation into one field.
    if (
        candidate_title
        and candidate_explanation
        and len(candidate_title.split()) <= 8
        and len(candidate_explanation.split()) >= 5
    ):
        return candidate_title, candidate_explanation

    return title_text, explanation_text


def clean_title(text):
    text = strip_list_prefix(text)
    text = strip_title_prefix(text)
    text = strip_title_suffix(text)
    text = text.strip(" .;:-")
    return text


def clean_explanation(text):
    text = strip_list_prefix(text)
    text = strip_explanation_prefix(text)
    return text.strip()


def normalize_title(text):
    text = clean_title(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_rows(rows):
    cleaned = []
    for row in rows:
        clean_row = dict(row)
        raw_title = clean_text(row.get("barrier_title", ""))
        raw_explanation = clean_text(row.get("explanation", ""))

        raw_title, extracted_explanation = extract_parenthetical_title(raw_title)
        if extracted_explanation and not raw_explanation:
            raw_explanation = extracted_explanation

        raw_title, raw_explanation = maybe_split_inline_title_explanation(
            raw_title,
            raw_explanation,
        )

        clean_row["barrier_title"] = clean_title(raw_title)
        clean_row["explanation"] = clean_explanation(raw_explanation)
        clean_row["_title_norm"] = normalize_title(clean_row["barrier_title"])
        cleaned.append(clean_row)
    return cleaned


def deduplicate_rows(rows):
    seen = set()
    kept_rows = []
    removed_rows = []

    for row in rows:
        group_key = (row.get("iteration", ""), row.get("variant_id", ""))
        dedup_key = group_key + (row["_title_norm"],)
        if dedup_key in seen:
            removed_rows.append(row)
        else:
            seen.add(dedup_key)
            kept_rows.append(row)

    return kept_rows, removed_rows


def build_repetition_report(cleaned_rows, deduped_rows):
    original_counts = defaultdict(int)
    unique_title_counts = defaultdict(set)
    deduped_counts = defaultdict(int)

    for row in cleaned_rows:
        key = (row.get("iteration", ""), row.get("variant_id", ""))
        original_counts[key] += 1
        unique_title_counts[key].add(row["_title_norm"])

    for row in deduped_rows:
        key = (row.get("iteration", ""), row.get("variant_id", ""))
        deduped_counts[key] += 1

    report_rows = []
    for key in sorted(original_counts.keys(), key=lambda item: (str(item[0]), str(item[1]))):
        iteration, variant_id = key
        original_count = original_counts[key]
        unique_count = len(unique_title_counts[key])
        deduped_count = deduped_counts.get(key, 0)
        duplicates_removed = original_count - deduped_count
        report_rows.append(
            {
                "iteration": iteration,
                "variant_id": variant_id,
                "original_rows": original_count,
                "unique_titles_in_group": unique_count,
                "rows_after_dedup": deduped_count,
                "duplicates_removed": duplicates_removed,
            }
        )
    return report_rows


def strip_internal_columns(rows):
    stripped = []
    for row in rows:
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        stripped.append(out)
    return stripped


def print_stats(raw_rows, deduped_rows, report_rows, outdir):
    duplicates_removed = len(raw_rows) - len(deduped_rows)
    groups_with_duplicates = sum(
        1 for row in report_rows if int(row["duplicates_removed"]) > 0
    )
    print(f"Input rows: {len(raw_rows)}")
    print(f"Output rows: {len(deduped_rows)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Groups with duplicates: {groups_with_duplicates}")
    print(f"Clean CSV: {outdir / 'context_barrier_mention_clean.csv'}")
    print(f"Report CSV: {outdir / 'context_barrier_mention_repetition_report.csv'}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(input_path)
    cleaned_rows = clean_rows(raw_rows)
    deduped_rows, _removed_rows = deduplicate_rows(cleaned_rows)
    report_rows = build_repetition_report(cleaned_rows, deduped_rows)

    if raw_rows:
        clean_fieldnames = [key for key in raw_rows[0].keys()]
    else:
        clean_fieldnames = []

    report_fieldnames = [
        "iteration",
        "variant_id",
        "original_rows",
        "unique_titles_in_group",
        "rows_after_dedup",
        "duplicates_removed",
    ]

    write_csv(
        outdir / "context_barrier_mention_clean.csv",
        clean_fieldnames,
        strip_internal_columns(deduped_rows),
    )
    write_csv(
        outdir / "context_barrier_mention_repetition_report.csv",
        report_fieldnames,
        report_rows,
    )
    print_stats(raw_rows, deduped_rows, report_rows, outdir)


if __name__ == "__main__":
    main()
