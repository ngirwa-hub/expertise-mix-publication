# Role Generation Pipeline (API-Based, Reproducible)

## Understanding of the task

We need to re-run role generation from raw markdown inputs using an OpenAI API call, instead of chat/manual prompting.

- Raw inputs (6 files) are in: `scrapping/working_dir/intermediate-mds/`
- The new pipeline must be reproducible and executable from code.
- API credentials come from: `scrapping/openai-api.env`
- Implementation file: `scrapping/working_dir/src/role-generate.py`
- Generation must follow the prompt-defined output skeleton only (no external reference folders).

## Goal

Create a script that:

1. Reads the 6 role raw markdown files.
2. Sends each file to OpenAI via API with a deterministic instruction template.
3. Produces finalized role markdown outputs.
4. Saves outputs to a dedicated generated-output directory.
5. Logs progress and failures clearly.
6. Can be re-run reliably with the same inputs and prompt settings.
7. Forces the model to synthesize and include a clear role name for each role file.

## Input files

Expected raw files in `scrapping/working_dir/intermediate-mds/`:

- `generalist-midlevel.md`
- `generalist-senior.md`
- `sme-midlevel.md`
- `sme-senior.md`
- `normative-midlevel.md`
- `normative-senior.md`

## Input format contract (raw files)

Each raw file is expected to structure job entries using this heading pattern:

- `## jobN: <Job Title>`

Examples:

- `## job1: sustainability analyst`
- `## job2: Market risk manager`

Rules:

1. `jobN` must be sequential per file (`job1`, `job2`, ...).
2. A non-empty job title must exist after `:`.
3. Job content must be placed under its corresponding heading.
4. The generator should treat these headings as the canonical units for synthesis.
5. The `jobN` title is authoritative if any metadata conflicts exist elsewhere in the file.

## Output files

Planned generated output directory:

- `scrapping/working_dir/final_mds/llm-generated-roles/`

Planned output naming:

- `final-generalist-midlevel.md`
- `final-generalist-senior.md`
- `final-sme-midlevel.md`
- `final-sme-senior.md`
- `final-normative-midlevel.md`
- `final-normative-senior.md`

## Reproducibility design

To improve reproducibility:

- Use a single fixed system instruction template.
- Use a fixed model selection (configurable via CLI/env).
- Use fixed generation parameters where supported.
- Keep one-pass deterministic processing order (sorted filenames).
- Save per-file metadata logs (timestamp, model, input file, output file).

Note: API models can still evolve over time; reproducibility is best-effort unless a pinned dated/versioned model snapshot is used.

## Prompt/output contract (critical)

For each input file, the generation instruction must explicitly require:

1. A synthesized role name derived from that specific file's content.
2. The output structure must follow this skeleton:
   - `# SOUL.md`
   - `**Name:** <Synthesized Role Name>`
   - `**Role:** <Track> — <Level> (<Short role focus phrase>)`
   - `## Personality`
   - `## What You're Good At`
   - `## What You Care About`
3. The role definition must be synthesized from the same file only.
4. No blending of content from other role files.
5. No placeholder names (for example: \"Role Name\", \"TBD\", \"Untitled Role\").
6. Ignore source links/references (for example: `link_jobN (...) : https://...`), which are for traceability only.
7. If `link_jobN (...)` title text differs from `## jobN: ...`, treat `## jobN: ...` as canonical.

This ensures each finalized role output is both named and grounded in its respective raw source.

## Environment setup

The script will read API key from env var `OPENAI_API_KEY`.

Example run setup:

```bash
set -a
source /Users/HP/Documents/scrapping/openai-api.env
set +a
python /Users/HP/Documents/scrapping/working_dir/src/role-generate.py
```

## Planned script behavior

`role-generate.py` will:

1. Validate required directories/files exist.
2. Validate `OPENAI_API_KEY` exists.
3. Validate raw heading format (`## jobN: <Job Title>`) and report malformed entries.
4. Cross-check `link_jobN (...)` descriptors against `## jobN: ...` titles and warn on mismatches.
5. Build prompt per input file using:
   - fixed system instruction
   - sanitized raw role markdown as source context (reference links removed)
   - explicit requirement to synthesize and output in `SOUL.md` schema
6. Call OpenAI API for each role file.
7. Write generated markdown output.
8. Write machine-readable run summary (JSON) under output directory.
9. Continue processing other files if one file fails, then report failures at end.

## Planned CLI options

Initial planned arguments:

- `--input-dir` (default: `../intermediate-mds` from script location)
- `--output-dir` (default: `../final_mds/llm-generated-roles`)
- `--model` (default set in script)
- `--overwrite` (allow replacing existing outputs)
- `--dry-run` (validate and print plan without API calls)
- `--only` (optional comma-separated subset of role filenames)

## Error handling expectations

- Missing input files: fail fast with clear message.
- Missing env key: fail fast with setup instructions.
- Malformed raw job headings (missing `: <title>`): fail for that file with clear diagnostics.
- `link_jobN (...)` mismatch with `## jobN: ...`: warn only; `## jobN: ...` remains authoritative.
- API error on one file: capture error and continue next file.
- Non-empty output exists without `--overwrite`: skip with warning.

## Quality checks

After generation, we should be able to verify:

1. Exactly 6 output files are created (unless filtered by `--only`).
2. Output file names map 1:1 with input files.
3. Each output begins with `# SOUL.md`.
4. Each output includes `**Name:**` and `**Role:**` with non-placeholder values.
5. Output includes sections: `Personality`, `What You're Good At`, `What You Care About`.
6. Role name and content are grounded in that file's raw source only.
7. Raw inputs passed heading validation (`## jobN: <Job Title>`).
8. Run summary JSON includes success/failure per file.
9. Output structure follows the prompt-defined skeleton consistently.

## Status

`role-generate.py` is implemented according to this guide.
