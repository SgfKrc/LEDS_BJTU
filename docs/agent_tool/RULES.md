# Docagent Rules v1

`rules.yaml` is the machine-readable source for the document-maintenance scanner contract. It is intentionally JSON-compatible YAML so the core loader remains usable with Python's standard library; normal YAML files are also accepted when PyYAML is installed.

## Contract

- `schema_version` must equal `qlh.docagent.rules.v1`.
- `ruleset_version` is a positive integer and changes whenever rule semantics change.
- `defaults` defines scanner windows and accepted severity levels.
- `exemptions.status_contains` preserves the existing historical/frozen-document exceptions.
- `rules` must contain exactly the initial rule IDs `R1` through `R5`, with unique IDs, a valid severity, an `enabled` boolean, and a `parameters` object.
- R3 `topic_stop_words` and R5 `lifecycle_pattern` are part of the data contract so scanner matching does not depend on duplicated source constants.

The loader rejects an unknown schema, missing rule IDs, duplicate IDs, invalid levels, and malformed parameter shapes. It returns a normalized mapping and a stable SHA-256 fingerprint for later baseline and gate binding.

## Initial Rules

| ID | Level | Meaning |
|---|---|---|
| `R1` | `warn` | A stale status line conflicts with completion markers in the document body. |
| `R2` | `warn` | The current document has uncommitted workspace changes. |
| `R3` | `info` | The status date predates a conservatively related `src/` commit. |
| `R4` | `warn` | A repository-relative Markdown link has no file target. |
| `R5` | `info` | The first 12 lines have no status line. |

`DOCAGENT-P1B` makes the scanner consume these parameters and records the ruleset fingerprint in each audit report. The R1-R5 finding shape, messages, and ordering remain compatible with the pre-data scanner. The scanner also accepts a validated rules payload in tests and future gate tooling, while the default CLI loads this file.
