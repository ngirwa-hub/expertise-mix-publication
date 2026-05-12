r"""
# creating variants: ollama create <new-model-name> -f <Modelfile-path>

# running the script: 
.\.venv\Scripts\python.exe rag_pipeline\roles-enhanced\barrier_mention.py --runs 10 --ollama-timeout-sec 400 --transport-retries 3 --retry-backoff-sec 5


"""
import argparse
import csv
import datetime
import json
import re
import time
from pathlib import Path

import requests
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
VARIANTS_PATH = SCRIPT_DIR / "variants-few-enh.yaml"
OUTPUT_FOLDER = SCRIPT_DIR / "expert_responses"
RAW_LOG_DIR = OUTPUT_FOLDER / "raw_logs"
MASTER_CSV = OUTPUT_FOLDER / "context_barrier_mention_all.csv"
COUNTER_FILE = SCRIPT_DIR / "context_barrierMention_counter.txt"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MIN_BARRIERS = 5
MAX_REPROMPTS = 3

PERSONA_ACTIVATION = (
    "Respond according to your assigned expertise and internal role persona."
)


def load_project():
    return (
        "- The Shift to Direct Current (Shift2DC) project is one of two initiatives selected under a recent call focused on advancing direct current (DC) technologies.\n"
        "- The objective of this call is to establish guidelines for the widespread application of low and medium voltage DC systems.\n"
        "- The project will deliver 30 DC-related solutions, including software tools, simulation platforms, and hardware components such as cables and converters.\n"
        "- Several demonstrators are planned to test and showcase these solutions in real-world settings.\n"
        "- The project adopts a comprehensive approach, addressing technical barriers, regulatory frameworks, stakeholder engagement, and user perspectives.\n"
    )


def load_demonstration():
    return (
        "- The Shift2DC project includes four key demonstration areas: ports, industry, data centers, and buildings.\n"
        "- Two of these areas-data centers and industry-feature physical demonstrators where technologies will be implemented and tested on-site.\n"
        "- The data center demonstration is located in Germany and focuses on edge data centers. It explores how DC can be integrated to support renewable energy use, heat reuse, and powering not only the computing infrastructure but also office spaces.\n"
        "- The industry demonstration involves a functioning factory environment where DC technologies will be piloted.\n"
        "- Live demonstrations will also take place in buildings, while the port demonstration includes a small-scale testbed supported by a digital twin to explore DC scalability in port operations.\n"
        "- In the port use case, one focus is to assess DC as a viable alternative for onshore power supply, especially in light of varying vessel frequency standards (50 Hz vs. 60 Hz).\n"
        "- The port demonstration also considers powering port operations-such as forklifts and electric vehicles-through a DC microgrid using hardware-in-the-loop simulations.\n"
        "- Finally, the project will gather perspectives not only from experts but also from end-user observers, such as tourists, to better understand public awareness and acceptance of DC technologies.\n"
    )


def load_elicitation():
    return (
        "- This expert elicitation aims to collect expert insights on the feasibility, importance, challenges, and opportunities associated with proposed DC solutions.\n"
        "- Expert elicitation is a structured technique that draws on the knowledge and judgment of experts to inform complex decision-making.\n"
        "- The process covers a series of predefined topics. Experts are asked to respond to targeted questions, and their responses will be analyzed to identify areas of agreement, divergence, and uncertainty.\n"
    )


def load_instructions():
    return (
        "You are participating in an expert elicitation exercise.\n"
        "Propose new barriers to DC adoption.\n"
        "Be concise. Start with a short title, then give a brief explanation."
    )


def load_question():
    return (
        "Provide up to 5 new barriers to DC adoption that are NOT in the list.\n"
        "Consider numbering each item.\n"
        "Each item must include a short title and a brief explanation."
    )


def ensure_output_dirs():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_variants():
    with VARIANTS_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    variants = []
    for variant_id, spec in (config.get("variants") or {}).items():
        if not spec.get("enabled", False):
            continue
        variants.append(
            {
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "role": spec.get("role", ""),
                "model": spec["model"],
            }
        )

    if not variants:
        raise RuntimeError(f"No enabled variants found in {VARIANTS_PATH}")

    return variants


def load_iteration_counter():
    if COUNTER_FILE.exists():
        try:
            return int(COUNTER_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            return 0
    return 0


def save_iteration_counter(value):
    COUNTER_FILE.write_text(str(value), encoding="utf-8")


def ensure_counter_file():
    if not COUNTER_FILE.exists():
        save_iteration_counter(0)


def ensure_csv_headers():
    if MASTER_CSV.exists():
        return

    with MASTER_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "base_model",
                "variant_id",
                "role",
                "model",
                "barrier_title",
                "explanation",
                "iteration",
                "timestamp",
                "truncated",
                "is_duplicate",
                "attempts",
                "transport_attempts",
                "transport_error",
            ],
        )
        writer.writeheader()


def build_prompt(question):
    context = "\n\n".join(
        [load_project(), load_demonstration(), load_elicitation(), load_instructions()]
    )
    return (
        f"{context}\n\n"
        f"Expertise guidance:\n{PERSONA_ACTIVATION}\n\n"
        f"Question:\n{question}"
    )


def build_retry_prompt(question, extracted_count, attempt_number):
    base_prompt = build_prompt(question)
    return (
        f"{base_prompt}\n\n"
        "Retry instruction:\n"
        f"The previous response produced only {extracted_count} valid numbered barriers after parsing. "
        f"This is retry {attempt_number - 1} of {MAX_REPROMPTS}.\n"
        f"Please return exactly {MIN_BARRIERS} numbered items.\n"
        "Format each item as:\n"
        "1. Short title\n"
        "Brief explanation\n"
        "Do not add any introduction or conclusion."
    )


def query_expert_once(model, prompt, ollama_timeout_sec):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True,
        timeout=ollama_timeout_sec,
    )
    response.raise_for_status()

    full_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        result = json.loads(line.decode("utf-8"))
        full_text += result.get("response", "")
    return full_text


def query_expert_with_transport_retries(
    model,
    prompt,
    ollama_timeout_sec,
    transport_retries,
    retry_backoff_sec,
):
    total_transport_attempts = transport_retries + 1
    last_error = None

    for transport_attempt in range(1, total_transport_attempts + 1):
        try:
            response_text = query_expert_once(model, prompt, ollama_timeout_sec)
            return {
                "response_text": response_text,
                "transport_attempts": transport_attempt,
                "transport_error": None,
            }
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if transport_attempt < total_transport_attempts:
                wait_sec = retry_backoff_sec * transport_attempt
                print(
                    f"Transport error for {model} on attempt {transport_attempt}/"
                    f"{total_transport_attempts}: {last_error}. Retrying in {wait_sec}s."
                )
                time.sleep(wait_sec)
            else:
                print(
                    f"Transport error for {model} after {total_transport_attempts} attempts: "
                    f"{last_error}"
                )

    return {
        "response_text": "",
        "transport_attempts": total_transport_attempts,
        "transport_error": last_error,
    }


def extract_custom_barriers(response_text):
    pattern = re.findall(r"(?m)^\s*(\d+)[\).:-]?\s*(.*?)\n\s*(.+)", response_text.strip())
    results = []
    seen_titles = set()
    for _, title, explanation in pattern:
        normalized_title = re.sub(r"\W+", "", title.strip().lower())
        is_duplicate = normalized_title in seen_titles
        results.append(
            {
                "title": title.strip(),
                "explanation": explanation.strip(),
                "is_duplicate": is_duplicate,
            }
        )
        seen_titles.add(normalized_title)
    return results


def query_variant_with_retries(
    variant,
    question,
    ollama_timeout_sec,
    transport_retries,
    retry_backoff_sec,
    iteration_counter,
):
    attempt_logs = []
    extracted = []
    response_text = ""
    total_attempts = MAX_REPROMPTS + 1

    for attempt_number in range(1, total_attempts + 1):
        if attempt_number == 1:
            prompt = build_prompt(question)
        else:
            prompt = build_retry_prompt(question, len(extracted), attempt_number)

        print(f"Querying {variant['variant_id']} (attempt {attempt_number}/{total_attempts})...")
        transport_result = query_expert_with_transport_retries(
            model=variant["model"],
            prompt=prompt,
            ollama_timeout_sec=ollama_timeout_sec,
            transport_retries=transport_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        response_text = transport_result["response_text"]
        extracted = extract_custom_barriers(response_text)
        attempt_logs.append(
            {
                "attempt": attempt_number,
                "barrier_count": len(extracted),
                "response_text": response_text,
                "transport_attempts": transport_result["transport_attempts"],
                "transport_error": transport_result["transport_error"],
            }
        )
        save_raw_log_for_variant(
            {
                **variant,
                "attempt_logs": attempt_logs,
            },
            iteration_counter,
        )

        if transport_result["transport_error"] and not response_text:
            if attempt_number < total_attempts:
                print(
                    f"{variant['variant_id']} had no usable response due to transport failure; "
                    "retrying."
                )
                continue
            print(
                f"{variant['variant_id']} had no usable response after content and transport "
                "retries; saving partial result."
            )
            break

        if len(extracted) >= MIN_BARRIERS:
            break

        if attempt_number < total_attempts:
            print(
                f"{variant['variant_id']} returned {len(extracted)} custom barriers; "
                "retrying."
            )
        else:
            print(
                f"{variant['variant_id']} returned {len(extracted)} custom barriers after "
                f"{total_attempts} attempts; saving partial result."
            )

    return {
        **variant,
        "barriers": extracted,
        "attempts": len(attempt_logs),
        "response_text": response_text,
        "attempt_logs": attempt_logs,
    }


def append_results(structured_responses, iteration_counter):
    with MASTER_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "base_model",
                "variant_id",
                "role",
                "model",
                "barrier_title",
                "explanation",
                "iteration",
                "timestamp",
                "truncated",
                "is_duplicate",
                "attempts",
                "transport_attempts",
                "transport_error",
            ],
        )
        for variant in structured_responses:
            full_count = len(variant["barriers"])
            truncated_flag = "yes" if full_count > MIN_BARRIERS else "no"
            for index, barrier in enumerate(variant["barriers"][:MIN_BARRIERS], start=1):
                row_id = f"{variant['variant_id']}_{iteration_counter:02d}_b{index}"
                writer.writerow(
                    {
                        "row_id": row_id,
                        "base_model": variant["family"],
                        "variant_id": variant["variant_id"],
                        "role": variant["role"],
                        "model": variant["model"],
                        "barrier_title": barrier["title"],
                        "explanation": barrier["explanation"],
                        "iteration": iteration_counter,
                        "timestamp": TIMESTAMP,
                        "truncated": truncated_flag,
                        "is_duplicate": barrier["is_duplicate"],
                        "attempts": variant["attempts"],
                        "transport_attempts": sum(
                            log_entry["transport_attempts"]
                            for log_entry in variant["attempt_logs"]
                        ),
                        "transport_error": variant["attempt_logs"][-1]["transport_error"]
                        if variant["attempt_logs"]
                        else None,
                    }
                )


def raw_log_path(variant_id, iteration_counter):
    return RAW_LOG_DIR / f"context_{variant_id}_{iteration_counter:02d}_{TIMESTAMP}.txt"


def save_raw_log_for_variant(variant_result, iteration_counter):
    txt_path = raw_log_path(variant_result["variant_id"], iteration_counter)
    parts = []
    for log_entry in variant_result["attempt_logs"]:
        parts.append(
            f"=== Attempt {log_entry['attempt']} | parsed barriers: {log_entry['barrier_count']} | "
            f"transport attempts: {log_entry['transport_attempts']} | "
            f"transport error: {log_entry['transport_error']} ===\n"
            f"{log_entry['response_text']}\n"
        )
    txt_path.write_text("\n".join(parts), encoding="utf-8")


def save_raw_logs(raw_responses, iteration_counter):
    for variant in raw_responses:
        save_raw_log_for_variant(variant, iteration_counter)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of full runs to execute.")
    parser.add_argument(
        "--ollama-timeout-sec",
        type=int,
        default=600,
        help="Per-request timeout for Ollama calls in seconds.",
    )
    parser.add_argument(
        "--transport-retries",
        type=int,
        default=3,
        help="Number of transport retries for timeout or connection failures.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=int,
        default=5,
        help="Base backoff in seconds between transport retries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.ollama_timeout_sec < 1:
        raise ValueError("--ollama-timeout-sec must be at least 1")
    if args.transport_retries < 0:
        raise ValueError("--transport-retries must be at least 0")
    if args.retry_backoff_sec < 0:
        raise ValueError("--retry-backoff-sec must be at least 0")

    ensure_output_dirs()
    ensure_csv_headers()
    ensure_counter_file()
    question = load_question()
    variants = load_variants()

    for run_index in range(args.runs):
        iteration_counter = load_iteration_counter()
        print(f"Starting run {run_index + 1} of {args.runs} (iteration {iteration_counter})...")

        structured_responses = []
        raw_responses = []

        for variant in variants:
            variant_result = query_variant_with_retries(
                variant=variant,
                question=question,
                ollama_timeout_sec=args.ollama_timeout_sec,
                transport_retries=args.transport_retries,
                retry_backoff_sec=args.retry_backoff_sec,
                iteration_counter=iteration_counter,
            )
            structured_responses.append(variant_result)
            raw_responses.append(variant_result)

        append_results(structured_responses, iteration_counter)
        save_raw_logs(raw_responses, iteration_counter)
        save_iteration_counter(iteration_counter + 1)
        print(f"Results appended to: {MASTER_CSV}")


if __name__ == "__main__":
    main()
