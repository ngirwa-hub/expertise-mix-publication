r""""
# dry-run for this script:
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-scoring/01_llm_variables.py --dry-run


# actual command:
/Users/HP/Documents/.venv312/bin/python /Users/HP/Documents/scrapping/working_dir/src-scoring/01_llm_variables.py

"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
PROFILES_FOLDER = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/final-profiles"
OUTPUT_JSON = "/Users/HP/Documents/scrapping/working_dir/final_mds/weighting/generated_dimensions.json"
MODEL_NAME = "gpt-5.4-mini"
TEMPERATURE = 0.2
ENV_FILE = "/Users/HP/Documents/scrapping/openai-api.env"

# Add your DC context here or load it from a file if preferred
DC_DESCRIPTION = """
In the broader energy domain, current initiatives on medium and low voltage direct current (DC) systems aim to accelerate their adoption by developing integrated technological, regulatory, and socio-technical frameworks. These efforts typically focus on delivering a portfolio of DC solutions that combine hardware (e.g., converters, cables, protection systems) and software (e.g., control algorithms, simulation tools, and digital platforms), alongside validation through real-world and virtual demonstrators. Demonstration activities commonly span multiple application domains—such as industrial facilities, commercial and residential buildings, data centers, and transport or logistics hubs—where DC systems are tested for their ability to improve energy efficiency, integrate renewable energy sources, enable flexible load management, and support electrification. Both physical testbeds and digital twins are employed to evaluate system performance, scalability, and interoperability under realistic conditions. In parallel, these initiatives address key barriers including standardization, regulatory alignment, and market readiness, while incorporating stakeholder and end-user perspectives to assess usability, acceptance, and broader societal implications of transitioning toward DC-based energy systems.
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def read_profiles(folder_path: str) -> List[Dict[str, str]]:
    """Read all .md profile files from a folder."""
    folder = Path(folder_path)
    profiles = []

    for file_path in sorted(folder.glob("*")):
        if file_path.suffix.lower() not in [".md"]:
            continue

        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            profiles.append({
                "file_name": file_path.name,
                "content": content
            })

    return profiles


def build_prompt(profiles: List[Dict[str, str]], dc_description: str) -> str:
    """Build the prompt for generating 3 shared dimensions."""
    profiles_text = "\n\n".join(
        [f"PROFILE: {p['file_name']}\n{p['content']}" for p in profiles]
    )

    prompt = f"""
You are analyzing a set of expert-role profiles that will later be used for LLM-based expert elicitation in the energy domain.

Task:
Read all profiles together and identify exactly 3 cross-profile dimensional variables.
These variables must:
1. Work across all profiles
2. Differentiate role type and seniority level
3. Be useful later for scoring LLM roles
4. Be relevant to the following elicitation context on medium and low voltage DC systems

DC elicitation context:
{dc_description}

Important:
- This is dimension generation, not profile scoring
- Do not assign scores to profiles
- Do not create more than 3 variables
- Keep the variables generalizable across Generalist, Normative, and SME roles
- The variables should reflect latent dimensions of expertise or orientation, not superficial keywords
- Prefer variables that can later be scored on a 1 to 3 scale by another script

For each variable, provide:
- variable_code: short code such as AD
- variable_name
- short_definition
- why_it_matters
- scoring_guidance_1_to_3: brief guidance for what low to high means

Return only valid JSON using this exact structure:

{{
  "variables": [
    {{
      "variable_code": "",
      "variable_name": "",
      "short_definition": "",
      "why_it_matters": "",
      "scoring_guidance_1_to_3": {{
        "1": "",
        "2": "",
        "3": ""
      }}
    }}
  ]
}}


There must be exactly 3 variables in the "variables" list.

Profiles:
{profiles_text}
""".strip()

    return prompt


def call_model(client: OpenAI, model_name: str, prompt: str) -> str:
    """Call the OpenAI model and return the raw text output."""
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=TEMPERATURE,
    )

    return response.output_text.strip()


def validate_output(data: Dict[str, Any]) -> bool:
    """Validate the returned JSON structure."""
    if not isinstance(data, dict):
        return False

    variables = data.get("variables")
    if not isinstance(variables, list) or len(variables) != 3:
        return False

    required_keys = {
        "variable_code",
        "variable_name",
        "short_definition",
        "why_it_matters",
        "scoring_guidance_1_to_3"
    }

    for var in variables:
        if not isinstance(var, dict):
            return False

        if not required_keys.issubset(var.keys()):
            return False

        guide = var.get("scoring_guidance_1_to_3")
        if not isinstance(guide, dict):
            return False

        for score in ["1", "2", "3"]:
            if score not in guide:
                return False

    return True


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save JSON to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shared scoring variables from profile markdown files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and build prompt without calling OpenAI or writing output.",
    )
    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    api_key = load_api_key_from_env_file(ENV_FILE)
    if not api_key and not args.dry_run:
        raise EnvironmentError(
            f"OPENAI_API_KEY is not set. Load {ENV_FILE} or export the variable."
        )

    profiles = read_profiles(PROFILES_FOLDER)
    if not profiles:
        raise FileNotFoundError(f"No .md profiles found in '{PROFILES_FOLDER}'.")

    prompt = build_prompt(profiles, DC_DESCRIPTION)
    if args.dry_run:
        print("Dry run: configuration and prompt preparation successful.")
        print(f"Profiles folder: {PROFILES_FOLDER}")
        print(f"Profiles loaded: {len(profiles)}")
        print(f"Prompt chars:    {len(prompt)}")
        print(f"Model:           {MODEL_NAME}")
        print(f"Temperature:     {TEMPERATURE}")
        print(f"Output JSON:     {OUTPUT_JSON}")
        return

    client = OpenAI(api_key=api_key)
    raw_output = call_model(client, MODEL_NAME, prompt)

    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        error_path = "dimension_generation_error.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(raw_output)
        raise ValueError(f"Model returned invalid JSON. Raw output saved to {error_path}")

    if not validate_output(data):
        error_path = "dimension_generation_invalid_structure.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        raise ValueError(f"Model returned JSON with invalid structure. Output saved to {error_path}")

    save_json(data, OUTPUT_JSON)
    print(f"Done. 3 variables saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
