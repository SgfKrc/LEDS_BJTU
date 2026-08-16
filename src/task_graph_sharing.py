"""Semantic Stage fingerprints for shadow-only TaskGraph sharing analysis.

This module classifies possible common-subtask sharing without rewriting a
graph.  The logical DAG remains the only execution source of truth.  Callers
must supply explicit, digest-only semantics contracts; payload bodies and raw
operation configuration are never accepted or returned.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from task_graph_optimization import require_graph_kind


STAGE_SEMANTICS_SCHEMA_VERSION = "qlh.task_graph_stage_semantics.v1"
SHARE_ANALYSIS_SCHEMA_VERSION = "qlh.task_graph_share_analysis.v1"
SHARE_ANALYZER_VERSION = "task-stage-share-v1"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_KEYS = frozenset({
    "schema_version",
    "stage_id",
    "input_signature_sha256",
    "input_schema_version",
    "output_schema_version",
    "operation_config_sha256",
    "data_scope",
    "side_effect_class",
    "determinism",
    "share_policy",
    "contract_digest",
})
_INPUT_REFERENCE_KEYS = frozenset({
    "ref_id",
    "schema_version",
    "content_sha256",
})
_ANALYSIS_KEYS = frozenset({
    "schema_version",
    "analyzer_version",
    "mode",
    "graph_digest",
    "stage_fingerprints",
    "pair_decisions",
    "summary",
    "digest",
})
_STAGE_RESULT_KEYS = frozenset({
    "stage_id",
    "eligible",
    "reason_code",
    "fingerprint_sha256",
    "contract_digest",
})
_PAIR_RESULT_KEYS = frozenset({
    "left_stage_id",
    "right_stage_id",
    "shareable",
    "reason_code",
    "fingerprint_sha256",
})
_SUMMARY_KEYS = frozenset({
    "stage_count",
    "eligible_stage_count",
    "pair_count",
    "shareable_pair_count",
})
_SIDE_EFFECT_CLASSES = frozenset({
    "none",
    "read_only_external",
    "external_mutation",
})
_DETERMINISM_CLASSES = frozenset({
    "deterministic",
    "seeded",
    "nondeterministic",
    "external_state",
})
_SHARE_POLICIES = frozenset({"allow", "deny", "independent"})
_ADMITTED_STAGE_TYPES = frozenset({"transform"})
_MODEL_IDENTITY_KEYS = frozenset({
    "engine",
    "format",
    "model_id",
    "revision",
    "sha256",
})
_STAGE_REASON_CODES = frozenset({
    "share_policy_admitted",
    "stage_type_not_admitted",
    "stage_not_pure",
    "model_identity_incomplete",
    "side_effect_not_none",
    "determinism_not_deterministic",
    "share_policy_denied",
    "independent_result_required",
})
_PAIR_REASON_CODES = frozenset({
    "identical_stage_fingerprint",
    "stage_ineligible",
    "stage_type_mismatch",
    "input_signature_mismatch",
    "schema_version_mismatch",
    "model_identity_mismatch",
    "provider_requirements_mismatch",
    "operation_config_mismatch",
    "data_scope_mismatch",
    "fingerprint_mismatch",
})
_FORBIDDEN_KEYS = frozenset({
    "body",
    "config",
    "content",
    "error",
    "output",
    "path",
    "prompt",
    "raw",
    "root_input",
    "secret",
    "token",
    "url",
})


class TaskGraphSharingError(ValueError):
    """Raised when a Stage sharing contract or report is malformed."""


def build_stage_semantics_contract(
    stage_id: str,
    *,
    input_signature_sha256: str,
    input_schema_version: str,
    output_schema_version: str,
    operation_config_sha256: str,
    data_scope: str,
    side_effect_class: str,
    determinism: str,
    share_policy: str,
) -> dict[str, Any]:
    """Build one digest-only semantics contract for a projected Stage."""

    contract = {
        "schema_version": STAGE_SEMANTICS_SCHEMA_VERSION,
        "stage_id": _identifier(stage_id, "stage_id"),
        "input_signature_sha256": _sha256(
            input_signature_sha256, "input_signature_sha256",
        ),
        "input_schema_version": _identifier(
            input_schema_version, "input_schema_version",
        ),
        "output_schema_version": _identifier(
            output_schema_version, "output_schema_version",
        ),
        "operation_config_sha256": _sha256(
            operation_config_sha256, "operation_config_sha256",
        ),
        "data_scope": _identifier(data_scope, "data_scope"),
        "side_effect_class": _enum(
            side_effect_class, _SIDE_EFFECT_CLASSES, "side_effect_class",
        ),
        "determinism": _enum(
            determinism, _DETERMINISM_CLASSES, "determinism",
        ),
        "share_policy": _enum(
            share_policy, _SHARE_POLICIES, "share_policy",
        ),
    }
    contract["contract_digest"] = _digest(contract)
    return validate_stage_semantics_contract(contract)


def validate_stage_semantics_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one versioned Stage semantics contract."""

    if not isinstance(contract, Mapping) or set(contract) != _CONTRACT_KEYS:
        raise TaskGraphSharingError("stage semantics contract fields are invalid")
    _assert_no_forbidden_fields(contract)
    if contract.get("schema_version") != STAGE_SEMANTICS_SCHEMA_VERSION:
        raise TaskGraphSharingError("unsupported stage semantics schema")
    _identifier(contract.get("stage_id"), "stage_id")
    _sha256(contract.get("input_signature_sha256"), "input_signature_sha256")
    _identifier(contract.get("input_schema_version"), "input_schema_version")
    _identifier(contract.get("output_schema_version"), "output_schema_version")
    _sha256(contract.get("operation_config_sha256"), "operation_config_sha256")
    _identifier(contract.get("data_scope"), "data_scope")
    _enum(
        contract.get("side_effect_class"),
        _SIDE_EFFECT_CLASSES,
        "side_effect_class",
    )
    _enum(contract.get("determinism"), _DETERMINISM_CLASSES, "determinism")
    _enum(contract.get("share_policy"), _SHARE_POLICIES, "share_policy")
    supplied_digest = _sha256(contract.get("contract_digest"), "contract_digest")
    unsigned = {
        key: value for key, value in contract.items() if key != "contract_digest"
    }
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphSharingError("stage semantics contract digest mismatch")
    return _detached(contract)


def digest_stage_input_references(
    references: Sequence[Mapping[str, Any]],
) -> str:
    """Digest normalized input references without retaining payload content."""

    if isinstance(references, (str, bytes)) or not isinstance(
        references, Sequence,
    ):
        raise TaskGraphSharingError("input references must be a sequence")
    normalized: list[dict[str, str]] = []
    ref_ids: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping) or set(reference) != _INPUT_REFERENCE_KEYS:
            raise TaskGraphSharingError("input reference fields are invalid")
        _assert_no_forbidden_fields(reference)
        ref_id = _identifier(reference.get("ref_id"), "input ref_id")
        if ref_id in ref_ids:
            raise TaskGraphSharingError("input ref_id values must be unique")
        ref_ids.add(ref_id)
        normalized.append({
            "ref_id": ref_id,
            "schema_version": _identifier(
                reference.get("schema_version"), "input schema_version",
            ),
            "content_sha256": _sha256(
                reference.get("content_sha256"), "input content_sha256",
            ),
        })
    normalized.sort(key=lambda item: item["ref_id"])
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def analyze_stage_sharing(
    projection: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify common-subtask candidates without constructing a new graph."""

    logical = require_graph_kind(projection, "logical_dag")
    checked_contracts = _contracts_by_stage(contracts)
    nodes = {
        node["stage_id"]: node
        for node in logical["nodes"]
        if node.get("node_kind") == "stage"
    }
    if set(checked_contracts) != set(nodes):
        raise TaskGraphSharingError(
            "semantics contracts must match projected Stage IDs exactly",
        )

    bases: dict[str, dict[str, Any]] = {}
    stage_results: list[dict[str, Any]] = []
    for stage_id in sorted(nodes):
        node = nodes[stage_id]
        contract = checked_contracts[stage_id]
        base = _fingerprint_base(node, contract)
        bases[stage_id] = base
        eligible, reason_code = _stage_eligibility(node, contract)
        stage_results.append({
            "stage_id": stage_id,
            "eligible": eligible,
            "reason_code": reason_code,
            "fingerprint_sha256": _digest(base),
            "contract_digest": contract["contract_digest"],
        })

    results_by_stage = {
        result["stage_id"]: result for result in stage_results
    }
    pair_decisions: list[dict[str, Any]] = []
    for left_stage_id, right_stage_id in combinations(sorted(nodes), 2):
        left_result = results_by_stage[left_stage_id]
        right_result = results_by_stage[right_stage_id]
        shareable, reason_code = _pair_decision(
            left_result,
            right_result,
            bases[left_stage_id],
            bases[right_stage_id],
        )
        pair_decisions.append({
            "left_stage_id": left_stage_id,
            "right_stage_id": right_stage_id,
            "shareable": shareable,
            "reason_code": reason_code,
            "fingerprint_sha256": (
                left_result["fingerprint_sha256"] if shareable else ""
            ),
        })

    analysis = {
        "schema_version": SHARE_ANALYSIS_SCHEMA_VERSION,
        "analyzer_version": SHARE_ANALYZER_VERSION,
        "mode": "shadow",
        "graph_digest": logical["digest"],
        "stage_fingerprints": stage_results,
        "pair_decisions": pair_decisions,
        "summary": {
            "stage_count": len(stage_results),
            "eligible_stage_count": sum(
                result["eligible"] for result in stage_results
            ),
            "pair_count": len(pair_decisions),
            "shareable_pair_count": sum(
                decision["shareable"] for decision in pair_decisions
            ),
        },
    }
    analysis["digest"] = _digest(analysis)
    return validate_share_analysis(analysis)


def validate_share_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted or operator-visible Stage sharing analysis."""

    if not isinstance(analysis, Mapping) or set(analysis) != _ANALYSIS_KEYS:
        raise TaskGraphSharingError("share analysis fields are invalid")
    _assert_no_forbidden_fields(analysis)
    if analysis.get("schema_version") != SHARE_ANALYSIS_SCHEMA_VERSION:
        raise TaskGraphSharingError("unsupported share analysis schema")
    if analysis.get("analyzer_version") != SHARE_ANALYZER_VERSION:
        raise TaskGraphSharingError("unsupported share analyzer version")
    if analysis.get("mode") != "shadow":
        raise TaskGraphSharingError("share analysis must remain shadow-only")
    _sha256(analysis.get("graph_digest"), "graph_digest")

    stage_results = analysis.get("stage_fingerprints")
    decisions = analysis.get("pair_decisions")
    summary = analysis.get("summary")
    if not isinstance(stage_results, list) or not isinstance(decisions, list):
        raise TaskGraphSharingError("share analysis result lists are invalid")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_KEYS:
        raise TaskGraphSharingError("share analysis summary is invalid")

    stage_ids: list[str] = []
    by_stage: dict[str, Mapping[str, Any]] = {}
    for result in stage_results:
        if not isinstance(result, Mapping) or set(result) != _STAGE_RESULT_KEYS:
            raise TaskGraphSharingError("stage fingerprint result is invalid")
        stage_id = _identifier(result.get("stage_id"), "stage result stage_id")
        eligible = result.get("eligible")
        if not isinstance(eligible, bool):
            raise TaskGraphSharingError("stage eligibility must be boolean")
        reason_code = _enum(
            result.get("reason_code"), _STAGE_REASON_CODES, "stage reason_code",
        )
        if eligible != (reason_code == "share_policy_admitted"):
            raise TaskGraphSharingError("stage eligibility and reason do not match")
        _sha256(result.get("fingerprint_sha256"), "stage fingerprint_sha256")
        _sha256(result.get("contract_digest"), "stage contract_digest")
        if stage_id in by_stage:
            raise TaskGraphSharingError("stage fingerprint IDs must be unique")
        stage_ids.append(stage_id)
        by_stage[stage_id] = result
    if stage_ids != sorted(stage_ids):
        raise TaskGraphSharingError("stage fingerprints must be sorted")

    expected_pairs = list(combinations(stage_ids, 2))
    supplied_pairs: list[tuple[str, str]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != _PAIR_RESULT_KEYS:
            raise TaskGraphSharingError("share pair decision is invalid")
        left = _identifier(decision.get("left_stage_id"), "left_stage_id")
        right = _identifier(decision.get("right_stage_id"), "right_stage_id")
        if left not in by_stage or right not in by_stage or left >= right:
            raise TaskGraphSharingError("share pair Stage IDs are invalid")
        supplied_pairs.append((left, right))
        shareable = decision.get("shareable")
        if not isinstance(shareable, bool):
            raise TaskGraphSharingError("pair shareable must be boolean")
        reason_code = _enum(
            decision.get("reason_code"), _PAIR_REASON_CODES, "pair reason_code",
        )
        fingerprint = decision.get("fingerprint_sha256")
        expected_shareable = (
            by_stage[left]["eligible"]
            and by_stage[right]["eligible"]
            and by_stage[left]["fingerprint_sha256"]
            == by_stage[right]["fingerprint_sha256"]
        )
        if shareable != expected_shareable:
            raise TaskGraphSharingError("pair decision contradicts fingerprints")
        if shareable:
            if reason_code != "identical_stage_fingerprint":
                raise TaskGraphSharingError("shareable pair reason is invalid")
            if fingerprint != by_stage[left]["fingerprint_sha256"]:
                raise TaskGraphSharingError("shared pair fingerprint is invalid")
        elif fingerprint != "" or reason_code == "identical_stage_fingerprint":
            raise TaskGraphSharingError("rejected pair must not expose a fingerprint")
    if supplied_pairs != expected_pairs:
        raise TaskGraphSharingError("share pair matrix is incomplete or unsorted")

    expected_summary = {
        "stage_count": len(stage_results),
        "eligible_stage_count": sum(
            result["eligible"] for result in stage_results
        ),
        "pair_count": len(decisions),
        "shareable_pair_count": sum(
            decision["shareable"] for decision in decisions
        ),
    }
    for key, expected in expected_summary.items():
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TaskGraphSharingError(f"summary {key} must be non-negative")
        if value != expected:
            raise TaskGraphSharingError(f"summary {key} does not match results")

    supplied_digest = _sha256(analysis.get("digest"), "analysis digest")
    unsigned = {key: value for key, value in analysis.items() if key != "digest"}
    if _digest(unsigned) != supplied_digest:
        raise TaskGraphSharingError("share analysis digest mismatch")
    return _detached(analysis)


def _contracts_by_stage(
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Sequence):
        raise TaskGraphSharingError("semantics contracts must be a sequence")
    by_stage: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        checked = validate_stage_semantics_contract(contract)
        stage_id = checked["stage_id"]
        if stage_id in by_stage:
            raise TaskGraphSharingError("semantics contract Stage IDs must be unique")
        by_stage[stage_id] = checked
    return by_stage


def _fingerprint_base(
    node: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    provider_constraints = node.get("provider_constraints", {})
    execution_constraints = node.get("execution_constraints", {})
    return {
        "fingerprint_schema_version": 1,
        "stage_type": node["stage_type"],
        "dependency_stage_ids": sorted(node.get("depends_on", [])),
        "input_bindings": sorted(
            node.get("input_bindings", []),
            key=lambda item: (
                item["target_key"],
                item["dependency_stage_id"],
                item["output_key"],
            ),
        ),
        "input_signature_sha256": contract["input_signature_sha256"],
        "input_schema_version": contract["input_schema_version"],
        "output_schema_version": contract["output_schema_version"],
        "model_identity": node.get("model_identity"),
        "provider_requirements": {
            "requested_provider": provider_constraints.get("requested_provider"),
            "fallback_providers": sorted(
                provider_constraints.get("fallback_providers", []),
            ),
            "pure": provider_constraints.get("pure"),
        },
        "minimum_successful_dependencies": execution_constraints.get(
            "minimum_successful_dependencies",
        ),
        "operation_config_sha256": contract["operation_config_sha256"],
        "data_scope": contract["data_scope"],
        "side_effect_class": contract["side_effect_class"],
        "determinism": contract["determinism"],
        "share_policy": contract["share_policy"],
    }


def _stage_eligibility(
    node: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[bool, str]:
    if node.get("stage_type") not in _ADMITTED_STAGE_TYPES:
        return False, "stage_type_not_admitted"
    if not node.get("provider_constraints", {}).get("pure", False):
        return False, "stage_not_pure"
    model_identity = node.get("model_identity")
    if model_identity is not None and (
        not isinstance(model_identity, Mapping)
        or set(model_identity) != _MODEL_IDENTITY_KEYS
        or not _SHA256_RE.fullmatch(str(model_identity.get("sha256", "")))
    ):
        return False, "model_identity_incomplete"
    if contract["side_effect_class"] != "none":
        return False, "side_effect_not_none"
    if contract["determinism"] != "deterministic":
        return False, "determinism_not_deterministic"
    if contract["share_policy"] == "independent":
        return False, "independent_result_required"
    if contract["share_policy"] != "allow":
        return False, "share_policy_denied"
    return True, "share_policy_admitted"


def _pair_decision(
    left_result: Mapping[str, Any],
    right_result: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[bool, str]:
    if not left_result["eligible"] or not right_result["eligible"]:
        return False, "stage_ineligible"
    comparisons = (
        ("stage_type", "stage_type_mismatch"),
        ("dependency_stage_ids", "input_signature_mismatch"),
        ("input_bindings", "input_signature_mismatch"),
        ("input_signature_sha256", "input_signature_mismatch"),
        ("input_schema_version", "schema_version_mismatch"),
        ("output_schema_version", "schema_version_mismatch"),
        ("model_identity", "model_identity_mismatch"),
        ("provider_requirements", "provider_requirements_mismatch"),
        ("minimum_successful_dependencies", "input_signature_mismatch"),
        ("operation_config_sha256", "operation_config_mismatch"),
        ("data_scope", "data_scope_mismatch"),
    )
    for key, reason_code in comparisons:
        if left[key] != right[key]:
            return False, reason_code
    if left_result["fingerprint_sha256"] != right_result["fingerprint_sha256"]:
        return False, "fingerprint_mismatch"
    return True, "identical_stage_fingerprint"


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TaskGraphSharingError("sharing contract key must be text")
            if key.lower() in _FORBIDDEN_KEYS:
                raise TaskGraphSharingError(
                    f"sharing contract contains forbidden field {key!r}",
                )
            _assert_no_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_forbidden_fields(nested)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TaskGraphSharingError(f"{field_name} must be a safe identifier")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise TaskGraphSharingError(f"{field_name} must be a SHA-256 digest")
    return value


def _enum(value: Any, allowed: frozenset[str], field_name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise TaskGraphSharingError(f"{field_name} is unsupported")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _detached(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True))


__all__ = [
    "SHARE_ANALYSIS_SCHEMA_VERSION",
    "SHARE_ANALYZER_VERSION",
    "STAGE_SEMANTICS_SCHEMA_VERSION",
    "TaskGraphSharingError",
    "analyze_stage_sharing",
    "build_stage_semantics_contract",
    "digest_stage_input_references",
    "validate_share_analysis",
    "validate_stage_semantics_contract",
]
