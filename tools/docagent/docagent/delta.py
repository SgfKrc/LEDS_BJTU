"""Baseline-to-report delta matrix for dry-run policy gates."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .baseline import ensure_matches


def _doc_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(doc["doc"]): doc for doc in report["docs"]}


def _finding_map(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for doc in report["docs"]:
        for finding in doc["findings"]:
            key = (str(doc["doc"]), str(finding["rule"]))
            result[key] = {
                "doc": str(doc["doc"]),
                "rule": str(finding["rule"]),
                "level": str(finding["level"]),
                "message": str(finding["message"]),
            }
    return result


def build_delta(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Build a deterministic finding/document delta after baseline validation."""
    ensure_matches(report, baseline)
    old_docs = _doc_map({"docs": baseline["docs"]})
    new_docs = _doc_map(report)
    old_findings = _finding_map({"docs": baseline["docs"]})
    new_findings = _finding_map(report)

    added_findings = [
        copy.deepcopy(new_findings[key])
        for key in sorted(set(new_findings) - set(old_findings))
    ]
    removed_findings = [
        copy.deepcopy(old_findings[key])
        for key in sorted(set(old_findings) - set(new_findings))
    ]
    changed_findings: list[dict[str, Any]] = []
    for key in sorted(set(old_findings) & set(new_findings)):
        old_finding = old_findings[key]
        new_finding = new_findings[key]
        if old_finding != new_finding:
            changed_findings.append({
                "doc": key[0],
                "rule": key[1],
                "old": {"level": old_finding["level"], "message": old_finding["message"]},
                "new": {"level": new_finding["level"], "message": new_finding["message"]},
            })

    new_documents = sorted(set(new_docs) - set(old_docs))
    gone_documents = sorted(set(old_docs) - set(new_docs))
    changed_documents = [
        {
            "doc": doc,
            "old_sha256": str(old_docs[doc]["sha256"]),
            "new_sha256": str(new_docs[doc]["sha256"]),
        }
        for doc in sorted(set(old_docs) & set(new_docs))
        if old_docs[doc]["sha256"] != new_docs[doc]["sha256"]
    ]
    affected_docs = sorted({
        *new_documents,
        *gone_documents,
        *(item["doc"] for item in changed_documents),
        *(item["doc"] for item in added_findings),
        *(item["doc"] for item in removed_findings),
        *(item["doc"] for item in changed_findings),
    })
    return {
        "schema_version": "qlh.docagent.delta.v1",
        "baseline": {
            "created_at": baseline["created_at"],
            "ruleset_version": baseline["ruleset_version"],
            "rules_fingerprint": baseline["rules_fingerprint"],
            "profile": copy.deepcopy(baseline["profile"]),
        },
        "current": {
            "run_ts": report["run_ts"],
            "ruleset_version": report["ruleset_version"],
            "rules_fingerprint": report["rules_fingerprint"],
            "report_fingerprint": report.get("report_fingerprint"),
            "profile": copy.deepcopy(report["profile"]),
        },
        "new": added_findings,
        "gone": removed_findings,
        "changed": changed_findings,
        "affected_docs": affected_docs,
        "documents": {
            "new": new_documents,
            "gone": gone_documents,
            "changed": changed_documents,
        },
        "summary": {
            "new": len(added_findings),
            "gone": len(removed_findings),
            "changed": len(changed_findings),
            "affected_docs": len(affected_docs),
            "new_documents": len(new_documents),
            "gone_documents": len(gone_documents),
            "changed_documents": len(changed_documents),
        },
    }


__all__ = ["build_delta"]
