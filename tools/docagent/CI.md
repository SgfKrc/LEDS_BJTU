# docagent in CI

`docagent` is read-only. It scans the configured documentation tree and returns
stable exit codes without modifying documents or Git state.

## Recommended command

Run from the repository root:

```text
python tools/docagent/run.py audit --root . --profile qlh --json \
  --output build/doc-audit/audit.json --fail-on error
```

Use `--profile minimal` for a small project that has no Git source-history
contract. A checked-in `.docagent/profile.yaml` is selected automatically when
`--profile` is omitted.

## Reports

`--json` writes machine-readable JSON. `--markdown` writes a stable Markdown
report. They are mutually exclusive. With `audit --output`, the selected
format is written to the requested path; JSON and Markdown are both generated
by the historical main-project compatibility entrypoint.

Keep report files under `build/` or another CI artifact directory, outside the
configured documentation directory.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | No finding matched `--fail-on` |
| 1 | At least one finding matched the selected severity or rule |
| 2 | Invalid profile/rules, missing documentation directory, invalid argument, or report write failure |

Typical CI policies are `--fail-on error` for blocking errors or
`--fail-on warn` for a stricter maintenance gate. Use `--fail-on none` when a
report is informational only.

## Baseline and rules changes

Create a locked, path-free baseline explicitly when the expected report is
known:

```text
python tools/docagent/run.py audit --root . --profile qlh \
  --lock build/doc-audit/baseline.json --fail-on none
```

Compare against it with `--baseline`. The comparison is rejected with code 2
if the baseline is missing, malformed, or was created with a different
ruleset/profile fingerprint:

```text
python tools/docagent/run.py scan --root . --profile qlh \
  --baseline build/doc-audit/baseline.json --dry-run \
  --json --fail-on error
```

Review a candidate rules file before using it:

```text
python tools/docagent/run.py rules diff \
  --old .docagent/rules.yaml --new rules-next.yaml
```

The diff identifies changed fields by structural path and includes old/new
ruleset fingerprints.

The dry-run JSON envelope contains `new`, `gone`, `changed`, and
`affected_docs`. `new` and `gone` count finding events, while
`summary.affected_docs` also includes documents whose content hash changed or
whose file was added/removed. `--max-new N` and `--max-gone N` return code 1
when the corresponding finding count exceeds its threshold. A dry-run without
`--baseline`, or with a mismatched rules/profile fingerprint, returns code 2.

## Integrity gate and rollback

Every scanner report includes a `report_fingerprint` over its complete
path-free JSON content. `--gate` writes a path-free gate artifact containing
the report, rules, profile, optional baseline, and optional released evolution
fingerprints. `--events` is an explicit opt-in to the standalone M3 SQLite
adapter; the normal scanner does not require SQLite state.

```text
python tools/docagent/run.py audit --root . --profile qlh --json --fail-on error \
  --output build/docagent-gates/report.json \
  --gate build/docagent-gates/gate.json \
  --events build/docagent-gates/events.sqlite
```

Verify the artifact against the actual rules file in CI. A modified report,
rules file, or binding fails with code 1; malformed JSON or invalid
configuration fails closed with code 2:

```text
python tools/docagent/run.py gate verify \
  --report build/docagent-gates/report.json \
  --rules .docagent/rules.yaml --profile qlh \
  --gate build/docagent-gates/gate.json
```

After reverting a rules commit, keep the baseline and rescan the reverted
checkout. `gate rescan` requires `--baseline` and performs the same dry-run
comparison without rewriting that baseline:

```text
git revert <rules-change-commit>
python tools/docagent/run.py gate rescan --root . \
  --baseline build/doc-audit/baseline.json --json --fail-on none
```

The rollback gate passes only when the rules/profile bindings match and the
resulting delta is within the configured thresholds.

## Rule evolution gate

An agent-produced rules file is reviewed as data through `rules evolve`. The
command requires `--change-note` and writes a path-free evolution record with
the old/new ruleset fingerprints, structural diff, risk classification, and an
`evolution_fingerprint` binding the approval to that exact candidate.

Create a preflight record:

```text
python tools/docagent/run.py rules evolve \
  --old .docagent/rules.yaml --new rules-next.yaml \
  --change-note "explain the intended rules change" \
  --state preflight --output build/docagent-gates/evolution.json
```

The state machine is `proposed -> preflight -> approved -> released`; a
reviewer may move `preflight` to `rejected` with a rejection approval record.
Vocabulary/parameter-only changes are low risk and can be auto-approved. New
or removed `warn`/`error` rules, severity changes, and activation changes on a
`warn`/`error` rule require a human approval record:

```json
{
 "schema_version": "qlh.docagent.approval.v1",
 "decision": "approved",
 "actor": "reviewer",
 "timestamp": "2026-09-09T00:00:00+00:00",
 "note": "reviewed the preflight impact",
 "evolution_fingerprint": "<64 lowercase hex characters>"
}
```

Apply the approval and release only after the approved record is persisted:

```text
python tools/docagent/run.py rules evolve \
  --record build/docagent-gates/evolution.json --state approved \
  --approval build/docagent-gates/approval.json \
  --output build/docagent-gates/evolution-approved.json
python tools/docagent/run.py rules evolve \
  --record build/docagent-gates/evolution-approved.json --state released \
  --output build/docagent-gates/evolution-released.json
```

Missing approval returns code 1 for the policy gate; malformed or fingerprint-
mismatched rules, records, or approval JSON returns code 2. The current scanner continues
to require the complete runtime `R1-R5` ruleset, so a candidate is not active
until it is separately released and installed by the owning project.

## GitHub Actions example

```yaml
- name: Audit documentation
  run: >-
    python tools/docagent/run.py audit --root . --profile qlh --json
    --output build/doc-audit/audit.json --fail-on error

- name: Upload documentation audit
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: docagent-audit
    path: build/doc-audit/
```

Configuration files are fail-closed. An unknown profile schema, an absolute or
parent-traversing path, an unknown field, or an invalid vocabulary/Git value
returns code 2 and identifies the affected configuration file and field on
stderr.
