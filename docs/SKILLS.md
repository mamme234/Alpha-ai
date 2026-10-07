# AlphaAI Skills

A **skill** is a higher-level capability built on AlphaAI Core: real
computation, tools, files, network access, or a live model engine. Skills are
the operations AlphaAI offers beyond raw text generation.

```shell
alphaai skills list          # every skill, availability and schema
alphaai skills run skill.calculator '{"expression": "2 * (3 + 4)"}'
```

Every skill declares an input schema, an output schema, the tools it uses, the
permissions those tools imply, its engine requirements and a timeout. The skill
manager validates input, checks availability, runs the handler under a timeout,
and validates the output. A skill either performs the operation or returns a
structured error — AlphaAI never fabricates a skill result.

## The twelve skills

| Skill | Category | Needs engine | What it really does |
| ----- | -------- | ------------ | ------------------- |
| `skill.calculator` | mathematics | no | exact arithmetic with `int`/`float`/`Fraction` and math functions |
| `skill.reasoning` | reasoning | no | Gaussian elimination, brute-force SAT, sequence extrapolation, audited calculation chains, weighted option comparison |
| `skill.data_analysis` | data | no | describe, quantiles, value counts, group-by aggregation, Pearson correlation, least-squares trend and filtering over real CSV/JSON/JSONL |
| `skill.text_transform` | text | no | deterministic pipelines of text operations (replace, regex, case, wrap, template, line ops) |
| `skill.summarization` | text | extractive mode: no | TF-IDF sentence ranking with position/length priors; `engine` mode delegates to a live model |
| `skill.translation` | language | **yes** | translation through a real local engine — no offline dictionary pretends to translate |
| `skill.file_analysis` | files | no | hashes, sizes, language guess and line counts for files in the sandbox |
| `skill.document_processing` | files | no | reads and chunks real documents (JSON/CSV/text) into structured segments |
| `skill.code_analysis` | code | no | real static metrics: lines, functions/classes, imports, comments |
| `skill.tool_calling` | tools | no | parse tool calls out of model text (`parse`) or execute them (`execute`) |
| `skill.web_research` | network | no | fetches and extracts a web document — requires the **network** permission (denied by default) |
| `skill.agent_workflow` | agent | no | runs a **declared** plan of skill/tool steps with dependency ordering and a real completion report |

## Availability

A skill reports `available: true/false` plus a reason. Two skills are typically
unavailable in a fresh checkout:

* `skill.translation` — needs a usable model engine with multilingual capability.
  With no weights loaded it reports "No usable AlphaAI model engine is
  registered, and this skill needs real inference."
* `skill.web_research` — needs the `network` permission, which defaults to off.

Bringing a model up (see [`ENGINES.md`](ENGINES.md)) or enabling the network
permission in `configs/alphaai.toml` flips those to available — the code path
does not change.

## Errors

| Code | Meaning |
| ---- | ------- |
| `skill_not_found` | no skill with that id |
| `skill_invalid_input` | input failed the skill's schema |
| `skill_permission_denied` | a required tool/permission was denied |
| `skill_timeout` | the handler exceeded its timeout |
| `skill_execution_failed` | the handler raised a real error |

Each error carries a message and, where AlphaAI can compute one, a `remediation`.

## Calling a skill over the API

```shell
curl -s localhost:8000/api/skills/execute \
  -H 'content-type: application/json' \
  -d '{"skill_id": "skill.calculator", "inputs": {"expression": "6*7"}}'
```

The response is `{"ok": <bool>, "result": {...}}`; a failed result carries a
structured `error` with a `code`.
