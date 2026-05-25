r"""
.\.venv\Scripts\python.exe rag_pipeline\roles-enhanced\analysis\cluster-naming.py --temperature 0.2 --timeout-sec 120
"""
import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "context_barrier_mention_hdbscan.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "cluster-names-gpt5.csv"
DEFAULT_NAMED_OUTPUT = SCRIPT_DIR / "context_barrier_mention_hdbscan_gpt5named.csv"
DEFAULT_ENV = SCRIPT_DIR.parents[1] / "configs" / "openai-api.env"
DEFAULT_LOG = SCRIPT_DIR / "cluster-naming-gpt5.log"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV with cluster assignments.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Cluster-level naming output CSV.")
    parser.add_argument(
        "--named-output",
        default=str(DEFAULT_NAMED_OUTPUT),
        help="Row-level CSV with cluster names merged back in.",
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV), help="Path to env file with OPENAI_API_KEY.")
    parser.add_argument("--log-file", default=str(DEFAULT_LOG), help="Path to log file.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--timeout-sec", type=float, default=120.0, help="Per-call timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per cluster on API or parse failure.")
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional sleep between successful calls.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=16000,
        help="Approximate character cap for grouped cluster_text sent to GPT.",
    )
    return parser.parse_args()


def setup_logging(log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def load_env_file(env_path):
    env_path = Path(env_path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def append_csv_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_processed_cluster_ids(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        processed = set()
        for row in csv.DictReader(handle):
            cluster_value = str(row.get("cluster", "")).strip()
            if cluster_value:
                processed.add(cluster_value)
        return processed


def load_cluster_names_df(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_input_df(df):
    required = ["cluster", "cluster_text", "row_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def build_cluster_input(df):
    cluster_df = df.loc[pd.to_numeric(df["cluster"], errors="coerce").fillna(-1) >= 0].copy()
    cluster_df["cluster"] = pd.to_numeric(cluster_df["cluster"], errors="coerce").astype(int)
    grouped = (
        cluster_df.groupby("cluster")
        .agg(
            cluster_size=("row_id", "count"),
            texts=("cluster_text", lambda s: [str(x).strip() for x in s.dropna().astype(str) if str(x).strip()]),
        )
        .reset_index()
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    return cluster_df, grouped


def build_cluster_prompt(row, max_chars):
    cluster_id = int(row["cluster"])
    cluster_size = int(row["cluster_size"])

    header = f"""
You are naming one semantic cluster of barriers to DC adoption.

Your task:
- Read the grouped barrier texts in this cluster.
- Propose one concise cluster name that captures the shared meaning of the texts.
- Use only the content provided in the cluster texts.
- Do not invent content not supported by the texts.
- Return strict JSON with exactly these keys:
  - cluster_id
  - cluster_name

Rules for cluster_name:
- lowercase
- 4 to 6 words
- noun phrase
- specific enough to distinguish this cluster from other clusters
- based on the common meaning of the cluster texts

Cluster ID: {cluster_id}
Cluster size: {cluster_size}

Cluster texts:
""".strip()

    parts = [header]
    used = len(header)
    for index, text in enumerate(row["texts"], start=1):
        block = f"\n\nItem {index}:\n{text}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    parts.append("\n\nReturn only JSON.")
    return "".join(parts)


def extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def call_openai_chat(api_key, model, temperature, timeout_sec, prompt):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You assign one concise name to a semantic cluster and return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        },
        timeout=timeout_sec,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def name_cluster_with_retries(api_key, row, args):
    total_attempts = args.max_retries + 1
    last_error = None
    prompt = build_cluster_prompt(row, args.max_chars)
    cluster_id = int(row["cluster"])

    for attempt in range(1, total_attempts + 1):
        try:
            logging.info(
                "Naming cluster %s attempt %s/%s (size=%s)",
                cluster_id,
                attempt,
                total_attempts,
                int(row["cluster_size"]),
            )
            content = call_openai_chat(
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout_sec=args.timeout_sec,
                prompt=prompt,
            )
            payload = extract_json_object(content)
            cluster_name = str(payload.get("cluster_name", "")).strip()
            returned_cluster_id = payload.get("cluster_id", cluster_id)
            if not cluster_name:
                raise ValueError("Model returned empty cluster_name.")
            return {
                "cluster": cluster_id,
                "cluster_id_returned": returned_cluster_id,
                "cluster_size": int(row["cluster_size"]),
                "cluster_name": cluster_name,
                "name_attempts": attempt,
                "name_error": "",
                "name_model": args.model,
                "name_temperature": args.temperature,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logging.warning(
                "Cluster naming failed for %s on attempt %s/%s: %s",
                cluster_id,
                attempt,
                total_attempts,
                last_error,
            )

    return {
        "cluster": cluster_id,
        "cluster_id_returned": cluster_id,
        "cluster_size": int(row["cluster_size"]),
        "cluster_name": "",
        "name_attempts": total_attempts,
        "name_error": last_error or "Unknown error",
        "name_model": args.model,
        "name_temperature": args.temperature,
    }


def write_named_output(input_df, cluster_names_df, named_output_path):
    named_df = input_df.copy()
    if "cluster" in named_df.columns:
        named_df["cluster"] = pd.to_numeric(named_df["cluster"], errors="coerce")
    merge_df = cluster_names_df.copy()
    if not merge_df.empty:
        merge_df["cluster"] = pd.to_numeric(merge_df["cluster"], errors="coerce")
    named_df = named_df.merge(
        merge_df[["cluster", "cluster_name"]] if not merge_df.empty else pd.DataFrame(columns=["cluster", "cluster_name"]),
        on="cluster",
        how="left",
    )
    named_output_path = Path(named_output_path)
    named_output_path.parent.mkdir(parents=True, exist_ok=True)
    named_df.to_csv(named_output_path, index=False)


def main():
    args = parse_args()
    log_path = setup_logging(args.log_file)
    load_env_file(args.env_file)

    logging.info("Starting cluster naming")
    logging.info("Input CSV: %s", args.input)
    logging.info("Cluster-name output CSV: %s", args.output)
    logging.info("Named row-level output CSV: %s", args.named_output)
    logging.info("Env file: %s", args.env_file)
    logging.info(
        "Model: %s | temperature=%s | timeout_sec=%s | max_retries=%s | max_chars=%s",
        args.model,
        args.temperature,
        args.timeout_sec,
        args.max_retries,
        args.max_chars,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if args.temperature < 0 or args.temperature > 2:
        raise ValueError("--temperature must be between 0 and 2.")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be greater than 0.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be at least 0.")
    if args.max_chars < 1000:
        raise ValueError("--max-chars is too low to be useful.")

    input_df = pd.read_csv(args.input)
    validate_input_df(input_df)
    cluster_df, cluster_input_df = build_cluster_input(input_df)
    logging.info("Loaded %s input rows", len(input_df))
    logging.info("Non-noise clustered rows: %s", len(cluster_df))
    logging.info("Clusters to consider: %s", len(cluster_input_df))

    output_path = Path(args.output)
    processed_cluster_ids = load_processed_cluster_ids(output_path)
    logging.info("Found %s already processed clusters", len(processed_cluster_ids))

    output_fieldnames = [
        "cluster",
        "cluster_id_returned",
        "cluster_size",
        "cluster_name",
        "name_attempts",
        "name_error",
        "name_model",
        "name_temperature",
    ]

    newly_processed = 0
    skipped_existing = 0
    error_clusters = 0

    for _, row in cluster_input_df.iterrows():
        cluster_id = str(int(row["cluster"]))
        if cluster_id in processed_cluster_ids:
            skipped_existing += 1
            logging.info("Skipping existing cluster %s", cluster_id)
            continue

        result = name_cluster_with_retries(api_key, row, args)
        append_csv_row(output_path, output_fieldnames, result)
        processed_cluster_ids.add(cluster_id)
        newly_processed += 1

        if result["name_error"]:
            error_clusters += 1
            logging.warning("Saved cluster %s with naming error", cluster_id)
        else:
            logging.info("Saved cluster %s | name=%s", cluster_id, result["cluster_name"])

        if args.sleep_sec > 0:
            logging.info("Sleeping for %ss", args.sleep_sec)
            time.sleep(args.sleep_sec)

    cluster_names_df = load_cluster_names_df(output_path)
    write_named_output(input_df, cluster_names_df, args.named_output)

    logging.info("Clusters newly processed: %s", newly_processed)
    logging.info("Clusters skipped from existing output: %s", skipped_existing)
    logging.info("Clusters with naming errors: %s", error_clusters)
    logging.info("Cluster-name CSV: %s", output_path)
    logging.info("Named row-level CSV: %s", args.named_output)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
