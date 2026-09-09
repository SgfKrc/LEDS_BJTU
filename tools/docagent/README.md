# docagent

`docagent` is a standalone, standard-library-first, read-only documentation maintenance scanner.

```text
python tools/docagent/run.py rules
python tools/docagent/run.py scan --root G:/path/to/project --fail-on none
python tools/docagent/run.py scan --root G:/path/to/project --profile minimal --json --fail-on none
python tools/docagent/run.py audit --root G:/path/to/project --output build/docagent/audit.json
python tools/docagent/run.py init --root G:/path/to/project
```

The P3B core scans a profile-defined documentation tree recursively using the versioned R1-R5 rules contract. Bundled profiles are `qlh` (the main project's Git-aware layout) and `minimal` (Git checks disabled). A project can run `init` to copy a profile into `.docagent/profile.yaml`, or pass `--profile <name-or-path>` explicitly. `scan` never writes the target repository. `audit --output` may write a report outside the configured documentation directory.

The main-project compatibility entrypoint is `python scripts/doc_maintenance_audit.py`. Its historical M1 flags delegate to this package and keep writing `build/doc-audit/audit.json` plus `audit.md`; M2/M3 extension flags remain on the legacy implementation until their dedicated adapters are migrated.

For CI conventions, stable exit codes, report formats, and a GitHub Actions example, see [`CI.md`](CI.md).

The dry-run baseline delta is available with `--baseline --dry-run`; it reports `new`, `gone`, `changed`, and `affected_docs`, with optional `--max-new N` and `--max-gone N` gates.

Reports include a `report_fingerprint` covering their complete path-free JSON
content. Generate a CI gate artifact and optionally append an M3 event in one
audit invocation:

```text
python tools/docagent/run.py audit --root . --profile qlh --json --fail-on error \
  --output build/docagent-gates/report.json \
  --gate build/docagent-gates/gate.json \
  --events build/docagent-gates/events.sqlite
```

Verify the report against the rules file and all supplied bindings before
accepting the artifact. A changed report or rules file returns code 1; a
malformed artifact or configuration returns code 2:

```text
python tools/docagent/run.py gate verify \
  --report build/docagent-gates/report.json \
  --rules .docagent/rules.yaml --profile qlh \
  --gate build/docagent-gates/gate.json
```

Rules are reverted through Git, then rescanned against the unchanged baseline.
The command is an explicit shorthand for the baseline dry-run and must have a
baseline, so a successful rollback is demonstrated by a zero delta:

```text
git revert <rules-change-commit>
python tools/docagent/run.py gate rescan --root . \
  --baseline build/doc-audit/baseline.json --json --fail-on none
```

Rule evolution is gated through `rules evolve`. An agent may prepare a candidate
rules file, but the candidate is only data: the scanner and validator code are
not part of this entry point. Every evolution record requires a non-empty
`change_note`, stores old/new fingerprints and a stable evolution fingerprint,
and follows `proposed -> preflight -> approved -> released` (or `rejected`).

```text
python tools/docagent/run.py rules evolve \
  --old .docagent/rules.yaml --new rules-next.yaml \
  --change-note "why this candidate is needed" \
  --state approved --output build/docagent-gates/evolution.json
```

Low-risk vocabulary/parameter changes are auto-approved. New or removed
`warn`/`error` rules and severity changes stop at `preflight` with exit code 1.
After review, provide an approval JSON whose
`evolution_fingerprint` matches the preflight record, then transition the
record to `approved` and finally to `released`. See [`CI.md`](CI.md) for the
approval record shape and gate behavior.
