import os
from openai import OpenAI

# Initialize client (make sure your API key is set in environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_soul(structure_path, content_path, role_name, level_name, output_path):
    # Load files
    structure = read_file(structure_path)
    content = read_file(content_path)

    # SYSTEM PROMPT (STRICT CONTROL)
    system_prompt = """
You are an expert analyst generating structured personas from job descriptions.

STRICT RULES:
- Use ONLY the provided content.
- DO NOT invent, assume, or add external knowledge.
- If information is insufficient for a section, explicitly state:
  "Content not sufficient."
- Preserve fidelity to source material.
- Maintain clear, structured output.

Your task:
Transform raw job role descriptions into a structured persona using the provided template.
"""

    # USER PROMPT
    user_prompt = f"""
TARGET ROLE:
{role_name} — {level_name}

STRUCTURE TEMPLATE (SOUL.md):
----------------------------
{structure}

SOURCE CONTENT:
---------------
{content}

INSTRUCTIONS:
- Follow the structure EXACTLY.
- Populate each section using ONLY the source content.
- Synthesize across all jobs (do not summarize individually).
- Maintain domain-accurate language (engineering, energy, etc.).
- If energy relevance is weak, do NOT exaggerate it.
"""

    # API CALL
    response = client.chat.completions.create(
        model="gpt-5",  # or "gpt-4o" if preferred
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2  # low randomness for consistency
    )

    result = response.choices[0].message.content

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"SOUL.md generated → {output_path}")


# =========================
# 🔧 USAGE EXAMPLE
# =========================

if __name__ == "__main__":
    generate_soul(
        structure_path="SOUL.md",
        content_path="sme_senior.md",
        role_name="Subject-Matter Expert",
        level_name="Senior",
        output_path="SOUL_SME_SENIOR.md"
    )