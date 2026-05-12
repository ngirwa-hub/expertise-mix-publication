Create a Python script that uses an LLM API to read expert profile text files and assign scores on 4 variables for each profile.

Goal:
For each profile, the script should output 4 scores from 1 to 5:
1. Analytical Depth (AD)
2. Systems Integration Orientation (SIO)
3. Process & Data Governance Sensitivity (PDG)
4. Practical Implementation Orientation (PIO)

Context:
The profiles represent different expert roles and levels in energy-related expert elicitation. The LLM must read the profile content and score the profile based on the described role, responsibilities, strengths, and level.

Scoring scale:
Use only integers from 1 to 5.

Interpretation:
- 1 = very low
- 2 = low
- 3 = moderate
- 4 = high
- 5 = very high

Variable definitions:
- AD: Technical rigor, analytical reasoning, quantitative or method-based problem solving
- SIO: Ability to connect technical, regulatory, operational, and stakeholder dimensions
- PDG: Attention to documentation, traceability, quality control, auditability, and process discipline
- PIO: Focus on real-world feasibility, deployment, execution, and practical constraints

What the script should do:
1. Read multiple profile text files from a folder
2. Send each profile to the LLM with a structured scoring prompt
3. Ask the LLM to return only valid JSON
4. Parse the JSON safely
5. Save results to a CSV file

Required CSV columns:
- profile_name
- role_type
- seniority_level
- AD
- SIO
- PDG
- PIO

Important extraction rules:
- role_type should be inferred from the profile text, such as Generalist, Normative, or SME
- seniority_level should be inferred from the profile text, such as Mid-Level or Senior
- scores must be integers only
- if parsing fails, save the raw response in an error log file

Prompt requirements for the LLM:
The script must instruct the LLM to:
- read the full profile carefully
- score the profile on the 4 variables
- use the role and level described in the profile
- return only this JSON format:

{
  "role_type": "Generalist",
  "seniority_level": "Mid-Level",
  "AD": 3,
  "SIO": 4,
  "PDG": 4,
  "PIO": 3
}

Technical requirements:
- Use Python
- Keep the code simple and readable
- Use environment variables for the API key
- Add comments
- Include basic error handling
- Make it easy to change the model name
- Assume profiles are stored as .md or .txt files in one folder

Also include:
- a reusable function to build the scoring prompt
- a reusable function to call the API
- a reusable function to validate that returned scores are integers from 1 to 5

At the end, provide the full script only.