"""Model-free evidence contract for the CACHE-05 role asymmetry study.

The DeepSeek comparison is an architectural reference, not a claim that QLH
reproduces the same model.  This module records the smaller, testable system
roles that QLH can actually compare: local draft, strong verify, and the
specialized jobs assigned to the sub-1B candidates.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .ceiling import PublicEvidence


SCHEMA = "qlh.harness.role_asymmetry.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _validate_identifier(value: str, name: str) -> None:
    if not value or len(value) > 128 or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} is invalid")


def _validate_texts(values: tuple[str, ...], name: str) -> None:
    if not values or any(not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty text")


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """One bounded responsibility assigned to one or more model candidates."""

    id: str
    model_ids: tuple[str, ...]
    role: str
    responsibilities: tuple[str, ...]
    excluded_claims: tuple[str, ...]
    gate: str
    status: str = "candidate"

    def __post_init__(self) -> None:
        _validate_identifier(self.id, "role assignment id")
        if not self.model_ids or any(not model_id.strip() for model_id in self.model_ids):
            raise ValueError("role assignment needs model ids")
        _validate_identifier(self.role, "role")
        _validate_texts(self.responsibilities, "responsibilities")
        _validate_texts(self.excluded_claims, "excluded_claims")
        if not self.gate.strip() or self.status not in {"candidate", "reference_only", "experimental"}:
            raise ValueError("role assignment gate or status is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_ids": list(self.model_ids),
            "role": self.role,
            "responsibilities": list(self.responsibilities),
            "excluded_claims": list(self.excluded_claims),
            "gate": self.gate,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class RoleComparison:
    """A comparison with an explicit non-equivalence boundary."""

    id: str
    reference: str
    qlh_pattern: str
    mapping: Mapping[str, str]
    invariants: tuple[str, ...]
    non_equivalences: tuple[str, ...]
    status: str

    def __post_init__(self) -> None:
        _validate_identifier(self.id, "comparison id")
        if not self.reference.strip() or not self.qlh_pattern.strip() or not self.mapping:
            raise ValueError("comparison reference, pattern, and mapping are required")
        if any(not str(key).strip() or not str(value).strip() for key, value in self.mapping.items()):
            raise ValueError("comparison mapping must contain non-empty text")
        _validate_texts(self.invariants, "invariants")
        _validate_texts(self.non_equivalences, "non_equivalences")
        if self.status not in {"reference_only", "experimental", "planned"}:
            raise ValueError("comparison status is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reference": self.reference,
            "qlh_pattern": self.qlh_pattern,
            "mapping": dict(sorted(self.mapping.items())),
            "invariants": list(self.invariants),
            "non_equivalences": list(self.non_equivalences),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class RoleHypothesis:
    """A hypothesis with a measurable control and an explicit falsifier."""

    id: str
    statement: str
    control: str
    primary_metrics: tuple[str, ...]
    falsifier: str
    required_evidence: tuple[str, ...]
    priority: str = "P1"

    def __post_init__(self) -> None:
        _validate_identifier(self.id, "hypothesis id")
        if not self.statement.strip() or not self.control.strip() or not self.falsifier.strip():
            raise ValueError("hypothesis statement, control, and falsifier are required")
        _validate_texts(self.primary_metrics, "primary_metrics")
        _validate_texts(self.required_evidence, "required_evidence")
        if self.priority not in {"P0", "P1", "P2"}:
            raise ValueError("hypothesis priority is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "control": self.control,
            "primary_metrics": list(self.primary_metrics),
            "falsifier": self.falsifier,
            "required_evidence": list(self.required_evidence),
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class RoleAsymmetryReport:
    """Deterministic, non-production report for the CACHE-05 decision."""

    comparisons: tuple[RoleComparison, ...]
    assignments: tuple[RoleAssignment, ...]
    hypotheses: tuple[RoleHypothesis, ...]
    public_evidence: tuple[PublicEvidence, ...]
    evidence_scope: tuple[str, ...]
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported role asymmetry schema")
        if not self.comparisons or not self.assignments or not self.hypotheses:
            raise ValueError("role asymmetry report needs comparisons, assignments, and hypotheses")
        if len({item.id for item in self.comparisons}) != len(self.comparisons):
            raise ValueError("comparison ids must be unique")
        if len({item.id for item in self.assignments}) != len(self.assignments):
            raise ValueError("assignment ids must be unique")
        if len({item.id for item in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("hypothesis ids must be unique")
        _validate_texts(self.evidence_scope, "evidence_scope")

    @property
    def digest(self) -> str:
        return _digest(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "comparisons": [item.as_dict() for item in self.comparisons],
            "assignments": [item.as_dict() for item in self.assignments],
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "public_evidence": [item.as_dict() for item in self.public_evidence],
            "evidence_scope": list(self.evidence_scope),
            "runner_kind": "fixture",
            "weights_loaded": False,
            "network_used": False,
        }
        if include_digest:
            value["report_digest"] = self.digest
        return value

    def to_markdown(self) -> str:
        lines = [
            "# CACHE-05 非对称分工证据报告",
            "",
            f"- Schema: `{self.schema}`; digest: `{self.digest}`",
            "- Scope: architecture reference + QLH system-role hypotheses; no model loading, network, or production routing.",
            "",
            "## 对照",
            "",
            "| reference | QLH mapping | status | boundary |",
            "| --- | --- | --- | --- |",
        ]
        for item in self.comparisons:
            mapping = "; ".join(f"{key}={value}" for key, value in sorted(item.mapping.items()))
            lines.append(f"| {item.reference} | {mapping} | `{item.status}` | {'; '.join(item.non_equivalences)} |")
        lines.extend(("", "## 岗位", "", "| model scope | role | responsibility | gate |", "| --- | --- | --- | --- |"))
        for item in self.assignments:
            lines.append(f"| {', '.join(item.model_ids)} | `{item.role}` | {'; '.join(item.responsibilities)} | {item.gate} |")
        lines.extend(("", "## 可验证假设", "", "| id | control | metrics | falsifier |", "| --- | --- | --- | --- |"))
        for item in self.hypotheses:
            lines.append(f"| `{item.id}` | {item.control} | {', '.join(item.primary_metrics)} | {item.falsifier} |")
        lines.extend(("", "## 证据边界", "", *[f"- {item}" for item in self.evidence_scope], ""))
        return "\n".join(lines)


def build_role_asymmetry_report() -> RoleAsymmetryReport:
    """Build the CACHE-05 report without touching model artifacts or services."""

    comparisons = (
        RoleComparison(
            "deepseek-reference",
            "DeepSeek-V4.1-Flash: input side 8B active / output side 16B active",
            "QLH system-level input preparation plus stronger output verification",
            {"input": "context/cache and small-model preparation roles", "output": "verify/generation authority"},
            ("the output authority must define the accepted answer", "the comparison separates quality from coordination cost"),
            ("QLH does not reproduce the 552B MoE encoder-decoder architecture", "the parameter split is public reference data, not a local measurement"),
            "reference_only",
        ),
        RoleComparison(
            "draft-verify",
            "QLH speculative.py draft-verify contract",
            "local small draft proposes tokens; strong or external verify controls correction",
            {"draft": "Qwen2.5-0.5B / Qwen3-0.6B candidates", "verify": "DS3-0324-7B or an explicitly permitted external endpoint", "wire": "token ids plus per-token probabilities/logprobs"},
            ("the verifier remains the output distribution authority", "shared tokenizer and state/prefix contract are prerequisites"),
            ("the production decoding loop is not wired", "HTTP top-k reconstruction can be approximate and is not exact distribution evidence"),
            "experimental",
        ),
        RoleComparison(
            "sub1b-specialization",
            "QLH three sub-1B model experiment",
            "assign each small model a narrow link, thinking, or architecture-probe contract",
            {"Qwen2.5-0.5B": "link/context baseline", "Qwen3-0.6B": "thinking-switch and profile probe", "MiniCPM4-0.5B": "architecture and low-end resource probe"},
            ("each role has a measurable contract", "failure falls back to the host or a stronger model"),
            ("none of the three is promoted to the main answer model by this report", "role success is not general correctness evidence"),
            "planned",
        ),
    )
    assignments = (
        RoleAssignment(
            "sub1b-link-baseline",
            ("Qwen2.5-0.5B",),
            "link_baseline",
            ("measure registration, short-link, and context handoff cost", "provide a stable non-thinking small-model baseline"),
            ("do not claim objective correctness", "do not emit autonomous tool actions without a host gate"),
            "link completion and format contract pass on holdout",
        ),
        RoleAssignment(
            "sub1b-thinking-probe",
            ("Qwen3-0.6B",),
            "thinking_probe",
            ("measure enable_thinking=False behavior", "compare template and resource-budget adaptation"),
            ("do not restore a judging quality gate", "do not treat JSON output as verified tool calling"),
            "thinking-switch and profile capability evidence pass",
        ),
        RoleAssignment(
            "sub1b-architecture-probe",
            ("MiniCPM4-0.5B",),
            "architecture_probe",
            ("probe llama.cpp/GGUF loading and template/stop behavior", "measure low-end resource envelope"),
            ("do not force admission when compatibility is unknown", "do not convert a smoke result into quality evidence"),
            "architecture, template, and resource gates pass",
        ),
        RoleAssignment(
            "strong-verifier-candidate",
            ("DS3-0324-7B",),
            "verify_judge_candidate",
            ("serve as a quality-authority comparison candidate", "keep output acceptance separate from draft proposal quality"),
            ("do not imply production route switch", "do not hide the outstanding human/production gate"),
            "paired holdout and judging-policy evidence pass",
        ),
    )
    hypotheses = (
        RoleHypothesis(
            "ASYM-01",
            "With the same verifier, a draft can change coordination cost without changing the verifier-governed output distribution.",
            "paired single vs draft_verify runs with identical prompt, tokenizer, verifier, seed, and max output",
            ("quality_rate", "format_rate", "distribution_agreement", "acceptance_rate"),
            "output distribution or holdout quality diverges under exact verification, or the draft adds cost without exposing an independently visible verifier error",
            ("per-token verifier probabilities", "shared tokenizer identity", "paired holdout outputs", "no network or approximate top-k claim for exactness"),
        ),
        RoleHypothesis(
            "ASYM-02",
            "Role splitting pays only when accepted tokens amortize draft and round-trip coordination cost.",
            "single verifier baseline versus the same verifier with fixed draft gamma and measured RTT/local token rate",
            ("tokens_per_round", "acceptance_rate", "latency_p95_ms", "fallback_rate", "quality_rate"),
            "tokens_per_round <= 1.5, or RTT >= tokens_per_round / local_tok_per_second, or quality regresses on holdout",
            ("round-level timings", "accepted/rejected token counts", "fallback reason", "same prompt and sampling policy"),
        ),
        RoleHypothesis(
            "ASYM-03",
            "Sub-1B specialization reduces resource cost only when the assigned contract passes without pretending to be general model quality.",
            "each sub-1B role versus QW1.8B on the same link, template, or probe fixture",
            ("contract_pass_rate", "latency_p95_ms", "rss_peak_bytes", "format_rate", "unsupported_architecture_rate"),
            "the role has no resource benefit, or its contract failure rate exceeds the host fallback budget",
            ("model-specific profile digest", "holdout fixture", "RSS/VRAM sample", "explicit fallback record"),
        ),
    )
    evidence = (
        PublicEvidence(
            "PUB-DEEPSEEK-KV",
            "DeepSeek Context Caching",
            "https://api-docs.deepseek.com/guides/kv_cache",
            "cache-prefix and matching behavior",
            "architecture and cache reference only; not a local performance claim",
        ),
        PublicEvidence(
            "PUB-DEEPSEEK-V41",
            "DeepSeek V4.1-Flash release note",
            "https://api-docs.deepseek.com/news/news260910",
            "V4.1-Flash architecture and model claims",
            "record the public asymmetry reference without treating it as reproduced",
        ),
    )
    scope = (
        "The report is a fixture-only research contract: runner_kind=fixture, weights_loaded=false, network_used=false.",
        "speculative.py is an experimental measurement entry point; it is disabled by default and not connected to production decoding.",
        "Exact distribution claims require full verifier probabilities and a shared tokenizer; top-k HTTP reconstruction stays approximate.",
        "A role assignment can be promoted only by its own holdout, resource, and production-boundary gates.",
    )
    return RoleAsymmetryReport(comparisons, assignments, hypotheses, evidence, scope)


ROLE_ASYMMETRY_SCHEMA = SCHEMA


__all__ = [
    "ROLE_ASYMMETRY_SCHEMA",
    "RoleAssignment",
    "RoleAsymmetryReport",
    "RoleComparison",
    "RoleHypothesis",
    "build_role_asymmetry_report",
]
