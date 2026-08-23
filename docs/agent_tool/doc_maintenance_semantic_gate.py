"""M2 semantic-baseline validation for the documentation maintenance agent.

The baseline is an explicitly reviewed local fixture.  It is never a source of
truth for documentation state: it only measures whether a provider agrees with
the four frozen human labels before the provider is used for prioritization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_BASELINE_SCHEMA = "qlh.docagent.semantic_baseline.v1"
VALID_JUDGEMENTS = {"stale", "accurate", "needs_review"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAMPLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class SemanticBaselineError(ValueError):
    """Raised when a reviewed baseline cannot safely be applied."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticBaselineError(f"{label} must be an object")
    return value


def _sample(value: Any, index: int) -> dict[str, Any]:
    item = _mapping(value, f"samples[{index}]")
    required = {"id", "doc", "sha256", "rules", "expected_judgement", "human_rationale"}
    if set(item) != required:
        raise SemanticBaselineError(f"samples[{index}] has unsupported fields")
    sample_id = item["id"]
    doc = item["doc"]
    digest = item["sha256"]
    rules = item["rules"]
    expected = item["expected_judgement"]
    rationale = item["human_rationale"]
    if not isinstance(sample_id, str) or not _SAMPLE_ID_RE.fullmatch(sample_id):
        raise SemanticBaselineError(f"samples[{index}].id is invalid")
    if (
        not isinstance(doc, str)
        or not doc.startswith("docs/")
        or not doc.endswith(".md")
        or ".." in Path(doc).parts
    ):
        raise SemanticBaselineError(f"samples[{index}].doc is invalid")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise SemanticBaselineError(f"samples[{index}].sha256 is invalid")
    if (
        not isinstance(rules, list)
        or not rules
        or not all(isinstance(rule, str) for rule in rules)
        or len(rules) != len(set(rules))
        or any(rule not in {"R1", "R2", "R3", "R4", "R5"} for rule in rules)
    ):
        raise SemanticBaselineError(f"samples[{index}].rules is invalid")
    if expected not in VALID_JUDGEMENTS:
        raise SemanticBaselineError(f"samples[{index}].expected_judgement is invalid")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 512:
        raise SemanticBaselineError(f"samples[{index}].human_rationale is invalid")
    return {
        "id": sample_id,
        "doc": doc,
        "sha256": digest,
        "rules": sorted(rules),
        "expected_judgement": expected,
        "human_rationale": rationale,
    }


def load_semantic_baseline(path: str | Path) -> dict[str, Any]:
    """Load the exact four-sample, human-reviewed M2 acceptance fixture."""
    source = Path(path).expanduser()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticBaselineError("semantic baseline is unreadable") from exc
    value = _mapping(raw, "semantic baseline")
    required = {"schema_version", "baseline_id", "minimum_matches", "samples"}
    if set(value) != required or value.get("schema_version") != SEMANTIC_BASELINE_SCHEMA:
        raise SemanticBaselineError("semantic baseline schema is invalid")
    baseline_id = value["baseline_id"]
    if not isinstance(baseline_id, str) or not _SAMPLE_ID_RE.fullmatch(baseline_id):
        raise SemanticBaselineError("semantic baseline id is invalid")
    if value["minimum_matches"] != 3:
        raise SemanticBaselineError("semantic baseline minimum_matches must be 3")
    samples_raw = value["samples"]
    if not isinstance(samples_raw, list) or len(samples_raw) != 4:
        raise SemanticBaselineError("semantic baseline must contain exactly four samples")
    samples = [_sample(item, index) for index, item in enumerate(samples_raw)]
    if len({sample["id"] for sample in samples}) != 4:
        raise SemanticBaselineError("semantic baseline sample ids must be unique")
    if len({sample["doc"] for sample in samples}) != 4:
        raise SemanticBaselineError("semantic baseline documents must be unique")
    return {
        "schema_version": SEMANTIC_BASELINE_SCHEMA,
        "baseline_id": baseline_id,
        "minimum_matches": 3,
        "samples": samples,
    }


def prepare_semantic_baseline_audit(baseline: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Select only frozen samples and reject changed content or M1 signals."""
    records = audit.get("docs")
    if not isinstance(records, list):
        raise SemanticBaselineError("audit has no document records")
    by_doc: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping) and isinstance(record.get("doc"), str):
            by_doc[record["doc"]] = record

    selected: list[dict[str, Any]] = []
    for sample in baseline["samples"]:
        record = by_doc.get(sample["doc"])
        if record is None:
            raise SemanticBaselineError(f"sample_not_found:{sample['id']}")
        if record.get("sha256") != sample["sha256"]:
            raise SemanticBaselineError(f"sample_hash_mismatch:{sample['id']}")
        findings = record.get("findings")
        if not isinstance(findings, list):
            raise SemanticBaselineError(f"sample_findings_invalid:{sample['id']}")
        actual_rules = sorted({
            finding.get("rule") for finding in findings
            if isinstance(finding, Mapping) and isinstance(finding.get("rule"), str)
        })
        if actual_rules != sample["rules"]:
            raise SemanticBaselineError(f"sample_rules_mismatch:{sample['id']}")
        selected.append(dict(record))
    return {"docs": selected}


def evaluate_semantic_baseline(baseline: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Measure agreement without treating provider fallback as a valid judgement."""
    items = report.get("judgements")
    if not isinstance(items, list):
        return {
            "schema_version": "qlh.docagent.semantic_gate.v1",
            "baseline_id": baseline["baseline_id"],
            "status": "invalid",
            "matches": 0,
            "required_matches": baseline["minimum_matches"],
            "samples": [],
            "reasons": ["judgements_missing"],
        }
    by_doc: dict[str, Mapping[str, Any]] = {}
    duplicate = False
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("doc"), str):
            duplicate = True
            continue
        if item["doc"] in by_doc:
            duplicate = True
        by_doc[item["doc"]] = item

    outcomes: list[dict[str, Any]] = []
    reasons: list[str] = ["duplicate_or_invalid_judgement"] if duplicate else []
    matches = 0
    for sample in baseline["samples"]:
        item = by_doc.get(sample["doc"])
        observed = item.get("judgement") if isinstance(item, Mapping) else None
        source = item.get("source") if isinstance(item, Mapping) else None
        valid = observed in VALID_JUDGEMENTS and source == "llm"
        matched = valid and observed == sample["expected_judgement"]
        if matched:
            matches += 1
        if item is None:
            reasons.append(f"judgement_missing:{sample['id']}")
        elif not valid:
            reasons.append(f"judgement_not_provider_backed:{sample['id']}")
        outcomes.append({
            "sample_id": sample["id"],
            "expected": sample["expected_judgement"],
            "observed": observed if observed in VALID_JUDGEMENTS else None,
            "provider_backed": valid,
            "matched": matched,
        })

    known_docs = {sample["doc"] for sample in baseline["samples"]}
    if any(doc not in known_docs for doc in by_doc):
        reasons.append("unexpected_judgement")
    status = "invalid" if reasons else (
        "passed" if matches >= baseline["minimum_matches"] else "failed"
    )
    return {
        "schema_version": "qlh.docagent.semantic_gate.v1",
        "baseline_id": baseline["baseline_id"],
        "status": status,
        "matches": matches,
        "required_matches": baseline["minimum_matches"],
        "samples": outcomes,
        "reasons": reasons,
    }
