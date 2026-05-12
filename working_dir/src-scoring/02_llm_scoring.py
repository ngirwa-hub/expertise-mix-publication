r""""
dry-run for this script:
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-scoring/02_llm_scoring.py --overwrite


real command for this script: 
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-scoring/02_llm_scoring.py --overwrite

"""

import os
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
PROFILES_FOLDER = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/final-profiles"
VARIABLES_JSON = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/generated_dimensions.json"
OUTPUT_CSV = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/profile_scores.csv"
OUTPUT_JSON = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/profile_scores.json"
ERROR_LOG = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/scoring_errors.json"
MODEL_NAME = "gpt-5.4-mini"
ENV_FILE = "/Users/HP/Documents/scrapping/openai-api.env"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score profiles on generated variables using OpenAI."
    )
    parser.add_argument("--profiles-folder", type=Path, default=Path(PROFILES_FOLDER))
    parser.add_argument("--variables-json", type=Path, default=Path(VARIABLES_JSON))
    parser.add_argument("--output-csv", type=Path, default=Path(OUTPUT_CSV))
    parser.add_argument("--output-json", type=Path, default=Path(OUTPUT_JSON))
    parser.add_argument("--error-log", type=Path, default=Path(ERROR_LOG))
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subset of profile filenames to score.",
    )
    return parser.parse_args()


def load_api_key_from_env_file(env_file: str) -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key

    path = Path(env_file)
    if not path.exists() or not path.is_file():
        return ""

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != "OPENAI_API_KEY":
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ["OPENAI_API_KEY"] = value
            return value
    return ""


def ensure_openai_client():
    api_key = load_api_key_from_env_file(ENV_FILE)
    if not api_key:
        raise EnvironmentError(
            f"OPENAI_API_KEY is not set. Load {ENV_FILE} or export the variable."
        )
    return OpenAI(api_key=api_key)


def read_profiles(folder_path: Path, only_arg: str) -> List[Dict[str, str]]:
    profiles: List[Dict[str, str]] = []
    only_set = {name.strip() for name in only_arg.split(",") if name.strip()}

    for file_path in sorted(folder_path.glob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in [".md", ".txt"]:
            continue
        if only_set and file_path.name not in only_set:
            continue
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            profiles.append({"file_name": file_path.name, "content": content})

    if only_set:
        found = {p["file_name"] for p in profiles}
        missing = sorted(only_set - found)
        if missing:
            raise FileNotFoundError(f"Requested files not found or empty: {', '.join(missing)}")

    return profiles


def read_variables(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variables = data.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("Variables JSON missing non-empty 'variables' list.")

    cleaned: List[Dict[str, Any]] = []
    for item in variables:
        if not isinstance(item, dict):
            continue
        code = str(item.get("variable_code", "")).strip()
        name = str(item.get("variable_name", "")).strip()
        definition = str(item.get("short_definition", "")).strip()
        matters = str(item.get("why_it_matters", "")).strip()
        guidance = item.get("scoring_guidance_1_to_3")
        if not isinstance(guidance, dict):
            # Backward-compatible fallback for older dimensions files.
            guidance = item.get("scoring_guidance_1_to_5", {})
        if not code or not name or not isinstance(guidance, dict):
            continue
        cleaned.append(
            {
                "variable_code": code,
                "variable_name": name,
                "short_definition": definition,
                "why_it_matters": matters,
                "scoring_guidance_1_to_3": guidance,
            }
        )

    if not cleaned:
        raise ValueError("No valid variable definitions found.")
    return cleaned


def build_scoring_prompt(profile_name: str, profile_text: str, variables: List[Dict[str, Any]]) -> str:
    variable_lines: List[str] = []
    schema_lines: List[str] = []
    for var in variables:
        code = var["variable_code"]
        name = var["variable_name"]
        definition = var["short_definition"]
        matters = var["why_it_matters"]
        guide = var["scoring_guidance_1_to_3"]
        guide_text = ", ".join([f"{k}: {guide.get(k, '')}" for k in ["1", "2", "3"]])
        variable_lines.append(
            f"- {code} ({name})\n"
            f"  definition: {definition}\n"
            f"  why_it_matters: {matters}\n"
            f"  scale_guidance: {guide_text}"
        )
        schema_lines.append(f'  "{code}": 3,')

    variables_block = "\n".join(variable_lines)
    scores_schema_block = "\n".join(schema_lines)

    prompt = f"""
You are scoring one expert profile.

Task:
1. Read the profile text carefully.
2. Infer:
   - role_type (Generalist, Normative, SME, or closest clear label from the profile)
   - seniority_level (Mid-Level, Senior, or closest clear label from the profile)
3. Score the profile on each variable below using integers only from 1 to 3.

Variables:
{variables_block}

Rules:
- Output integer scores only (no decimals).
- Use only values 1, 2, 3.
- Interpret labels as: 1 = Low, 2 = Moderate, 3 = High.
- Return JSON only. No markdown, no explanation.

Required JSON format:
{{
  "role_type": "Generalist",
  "seniority_level": "Mid-Level",
{scores_schema_block}
}}

Profile filename: {profile_name}
Profile content:
{profile_text}
""".strip()
    return prompt


def call_model(client: OpenAI, model_name: str, prompt: str) -> str:
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.2,
    )
    text = getattr(response, "output_text", "") or ""
    return text.strip()


def parse_json_strict(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Small salvage attempt for fenced or extra text responses.
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw_text[start : end + 1])
        raise


def validate_scoring_payload(data: Dict[str, Any], variables: List[Dict[str, Any]]) -> bool:
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("role_type"), str) or not data["role_type"].strip():
        return False
    if not isinstance(data.get("seniority_level"), str) or not data["seniority_level"].strip():
        return False

    for var in variables:
        code = var["variable_code"]
        value = data.get(code)
        if not isinstance(value, int):
            return False
        if value < 1 or value > 3:
            return False
    return True


def save_csv(rows: List[Dict[str, Any]], path: Path, variable_codes: List[str]) -> None:
    fieldnames = ["profile_name", "role_type", "seniority_level"] + variable_codes
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()

    profiles_folder = args.profiles_folder.resolve()
    variables_json = args.variables_json.resolve()
    output_csv = args.output_csv.resolve()
    output_json = args.output_json.resolve()
    error_log = args.error_log.resolve()

    if not profiles_folder.exists() or not profiles_folder.is_dir():
        raise FileNotFoundError(f"Profiles folder not found: {profiles_folder}")
    if not variables_json.exists() or not variables_json.is_file():
        raise FileNotFoundError(f"Variables JSON not found: {variables_json}")

    for out_file in [output_csv, output_json, error_log]:
        if out_file.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Output exists: {out_file}. Use --overwrite to replace.")

    variables = read_variables(variables_json)
    variable_codes = [v["variable_code"] for v in variables]
    profiles = read_profiles(profiles_folder, args.only)
    if not profiles:
        raise FileNotFoundError(f"No profile files found in {profiles_folder}")

    if args.dry_run:
        print("Dry run: configuration and input validation successful.")
        print(f"Profiles folder:  {profiles_folder}")
        print(f"Profiles loaded:  {len(profiles)}")
        print(f"Variables loaded: {len(variables)} ({', '.join(variable_codes)})")
        print(f"Model:            {args.model}")
        print(f"Output CSV:       {output_csv}")
        print(f"Output JSON:      {output_json}")
        print(f"Error log:        {error_log}")
        return

    client = ensure_openai_client()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for profile in profiles:
        file_name = profile["file_name"]
        prompt = build_scoring_prompt(file_name, profile["content"], variables)
        try:
            raw_output = call_model(client, args.model, prompt)
            parsed = parse_json_strict(raw_output)
            if not validate_scoring_payload(parsed, variables):
                raise ValueError("Invalid scoring schema or score range in model output.")

            row = {
                "profile_name": file_name,
                "role_type": parsed["role_type"].strip(),
                "seniority_level": parsed["seniority_level"].strip(),
            }
            for code in variable_codes:
                row[code] = int(parsed[code])
            results.append(row)
            print(f"[OK]   {file_name}")
        except Exception as exc:
            errors.append(
                {
                    "profile_name": file_name,
                    "error": str(exc),
                    "raw_output": raw_output if "raw_output" in locals() else "",
                }
            )
            print(f"[FAIL] {file_name}: {exc}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    error_log.parent.mkdir(parents=True, exist_ok=True)

    save_csv(results, output_csv, variable_codes)
    output_json.write_text(
        json.dumps(
            {
                "timestamp_utc": utc_now_iso(),
                "model": args.model,
                "variables_json": str(variables_json),
                "profiles_folder": str(profiles_folder),
                "results_count": len(results),
                "errors_count": len(errors),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    error_log.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. Scored {len(results)} profile(s).")
    print(f"CSV saved:   {output_csv}")
    print(f"JSON saved:  {output_json}")
    print(f"Error log:   {error_log}")


if __name__ == "__main__":
    main()
