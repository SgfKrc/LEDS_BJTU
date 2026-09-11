"""Reproducible, model-free design for small-model ceiling research.

This module describes experiments and evaluates supplied evidence. It never
loads a model, downloads an artifact, or treats a fake runner result as a
production capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^//|^/|[\\/]\.\.[\\/])")
DEFAULT_MODEL_IDS = (
    "Qwen2.5-0.5B",
    "Qwen3-0.6B",
    "MiniCPM4-0.5B",
    "QW1.8B",
    "DS3-0324-7B",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _path_free(value: Any) -> None:
    if isinstance(value, str) and _ABSOLUTE_PATH.search(value):
        raise ValueError("research records cannot contain absolute or parent-traversal paths")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _path_free(key)
            _path_free(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _path_free(child)


def _identifier(value: str, name: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError(f"{name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    id: str
    hypothesis: str
    factor: str
    primary_metrics: tuple[str, ...]
    priority: str = "P1"
    falsifier: str = ""

    def __post_init__(self) -> None:
        _identifier(self.id, "question id")
        _identifier(self.factor, "question factor")
        if self.priority not in {"P0", "P1", "P2"}:
            raise ValueError("priority must be P0, P1, or P2")
        if not self.hypothesis.strip() or not isinstance(self.primary_metrics, (tuple, list)) or not self.primary_metrics:
            raise ValueError("question hypothesis and metrics are required")
        if any(not str(item).strip() for item in self.primary_metrics):
            raise ValueError("question metrics must be non-empty")
        _path_free(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "factor": self.factor,
            "primary_metrics": list(self.primary_metrics),
            "priority": self.priority,
            "falsifier": self.falsifier,
        }


@dataclass(frozen=True, slots=True)
class ResearchFactor:
    id: str
    levels: tuple[str, ...]
    metric: str
    control_level: str
    rationale: str

    def __post_init__(self) -> None:
        _identifier(self.id, "factor id")
        if not self.levels or self.control_level not in self.levels:
            raise ValueError("factor levels must include the control level")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("factor levels must be unique")
        if not self.metric.strip() or not self.rationale.strip():
            raise ValueError("factor metric and rationale are required")
        _path_free(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "levels": list(self.levels),
            "metric": self.metric,
            "control_level": self.control_level,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    id: str
    stage: str
    model_id: str
    factor_values: Mapping[str, str]
    holdout_required: bool = True
    seed: int = 17
    comparison_group: str = ""

    def __post_init__(self) -> None:
        _identifier(self.id, "cell id")
        _identifier(self.model_id, "model id")
        if self.stage not in {"baseline", "ablation", "roles", "policy", "evidence"}:
            raise ValueError("unsupported experiment stage")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.factor_values:
            raise ValueError("experiment cell needs factor values")
        if not isinstance(self.holdout_required, bool):
            raise ValueError("holdout_required must be boolean")
        _path_free(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "model_id": self.model_id,
            "factor_values": dict(sorted(self.factor_values.items())),
            "holdout_required": self.holdout_required,
            "seed": self.seed,
            "comparison_group": self.comparison_group,
        }


@dataclass(frozen=True, slots=True)
class PublicEvidence:
    id: str
    title: str
    url: str
    claim_scope: str
    use: str

    def __post_init__(self) -> None:
        _identifier(self.id, "evidence id")
        if not self.url.startswith(("https://", "http://")) or any(char.isspace() for char in self.url):
            raise ValueError("public evidence URL must be an absolute HTTP(S) URL")
        if not self.title.strip() or not self.claim_scope.strip() or not self.use.strip():
            raise ValueError("public evidence fields are required")
        _path_free(self.as_dict())

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "url": self.url, "claim_scope": self.claim_scope, "use": self.use}


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    cell_id: str
    model_profile_digest: str
    artifact_digest: str
    runtime: str
    fixture_digest: str
    seed: int
    metrics: Mapping[str, float]
    runner_kind: str = "real"
    weights_loaded: bool = False
    network_used: bool = False
    status: str = "planned"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.cell_id, "cell id")
        for name, value in (("model_profile_digest", self.model_profile_digest), ("artifact_digest", self.artifact_digest), ("fixture_digest", self.fixture_digest)):
            if not _SHA256.fullmatch(str(value)):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.runtime.strip() or self.seed < 0:
            raise ValueError("runtime and seed are required")
        if self.status not in {"planned", "collected", "rejected"}:
            raise ValueError("unsupported evidence status")
        if not isinstance(self.weights_loaded, bool) or not isinstance(self.network_used, bool):
            raise ValueError("weights_loaded and network_used must be boolean")
        if not isinstance(self.metrics, Mapping) or any(not isinstance(key, str) or not key.strip() or not math.isfinite(float(value)) for key, value in self.metrics.items()):
            raise ValueError("metrics must contain finite numeric values")
        if self.runner_kind not in {"real", "injected", "fixture", "unknown"}:
            raise ValueError("unsupported runner kind")
        _path_free(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "model_profile_digest": self.model_profile_digest,
            "artifact_digest": self.artifact_digest,
            "runtime": self.runtime,
            "fixture_digest": self.fixture_digest,
            "seed": self.seed,
            "metrics": dict(sorted(self.metrics.items())),
            "runner_kind": self.runner_kind,
            "weights_loaded": self.weights_loaded,
            "network_used": self.network_used,
            "status": self.status,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CeilingStudyPlan:
    schema: str
    models: tuple[str, ...]
    questions: tuple[ResearchQuestion, ...]
    factors: tuple[ResearchFactor, ...]
    cells: tuple[ExperimentCell, ...]
    public_evidence: tuple[PublicEvidence, ...]
    fixture_set_digest: str
    v2_policy: Mapping[str, Any]
    seed: int = 17

    def __post_init__(self) -> None:
        if self.schema != "qlh.harness.ceiling_study.v1":
            raise ValueError("unsupported ceiling study schema")
        if not self.models or not self.questions or not self.factors or not self.cells:
            raise ValueError("ceiling study must contain models, questions, factors, and cells")
        if not _SHA256.fullmatch(self.fixture_set_digest):
            raise ValueError("fixture_set_digest must be a SHA-256 digest")
        if self.v2_policy.get("max_new_tokens") != 512 or self.v2_policy.get("match") != "loose_contains":
            raise ValueError("v2 policy must fix loose_contains and a 512-token budget")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        model_set = set(self.models)
        factor_set = {factor.id for factor in self.factors}
        if any(cell.model_id not in model_set for cell in self.cells):
            raise ValueError("experiment cell references an unknown model")
        if any(set(cell.factor_values) - factor_set for cell in self.cells):
            raise ValueError("experiment cell references an unknown factor")
        _path_free(self.as_dict(include_digest=False))

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict(include_digest=False))).hexdigest()

    def as_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "models": list(self.models),
            "questions": [item.as_dict() for item in self.questions],
            "factors": [item.as_dict() for item in self.factors],
            "cells": [item.as_dict() for item in self.cells],
            "public_evidence": [item.as_dict() for item in self.public_evidence],
            "fixture_set_digest": self.fixture_set_digest,
            "v2_policy": dict(self.v2_policy),
            "seed": self.seed,
        }
        if include_digest:
            value["study_digest"] = self.digest
        return value


def build_ceiling_study(
    *,
    fixture_set_digest: str,
    models: Sequence[str] = DEFAULT_MODEL_IDS,
    seed: int = 17,
) -> CeilingStudyPlan:
    """Build the first bounded study plan; no execution or model access occurs."""

    model_values = tuple(_identifier(model, "model id") for model in models)
    if not model_values or len(set(model_values)) != len(model_values):
        raise ValueError("models must be a non-empty unique sequence")
    if not _SHA256.fullmatch(fixture_set_digest):
        raise ValueError("fixture_set_digest must be a SHA-256 digest")
    factors = (
        ResearchFactor("context", ("n_ctx_512", "n_ctx_2048", "n_ctx_4096"), "early_recall_rate", "n_ctx_2048", "test memory before changing prompt or role"),
        ResearchFactor("template", ("native", "profile_minimal", "profile_structured"), "format_rate", "profile_minimal", "template and stop behavior precede sampling changes"),
        ResearchFactor("memory", ("off", "state", "retrieval"), "early_fact_recall", "off", "measure memory benefit without hiding omissions"),
        ResearchFactor("role", ("single", "draft_verify"), "quality_per_cost", "single", "compare coordination cost against quality gain"),
        ResearchFactor("quantization", ("q4_k_m", "q8_0"), "quality_delta", "q8_0", "measure quantization loss only when both artifacts exist"),
        ResearchFactor("policy", ("v1_192_normalized_contains", "v2_512_loose_contains"), "judgable_rate", "v2_512_loose_contains", "separate judging policy from model capability"),
    )
    questions = (
        ResearchQuestion("CEIL-COMP-01", "Longer context helps only while early-fact recall does not collapse.", "context", ("early_recall_rate", "input_tokens", "latency_p95_ms"), "P0", "flat or falling recall at a larger budget"),
        ResearchQuestion("CEIL-TPL-01", "Template and stop alignment dominate short-answer format reliability.", "template", ("format_rate", "truncation_rate", "output_tokens"), "P0", "native template is no worse on holdout"),
        ResearchQuestion("CEIL-MEM-01", "Structured memory improves early-fact recall at a bounded token cost.", "memory", ("early_fact_recall", "input_tokens", "summary_loss_rate"), "P1", "no recall gain after cost normalization"),
        ResearchQuestion("CEIL-ROLE-01", "Draft/verify raises quality per cost only when verifier errors are independently visible.", "role", ("quality_rate", "latency_p95_ms", "fallback_rate"), "P1", "coordination cost dominates quality gain"),
        ResearchQuestion("CEIL-QUANT-01", "Quantization loss is workload-specific and must not be inferred from file size.", "quantization", ("quality_rate", "format_rate", "rss_peak_bytes"), "P1", "no paired artifact or no stable holdout"),
        ResearchQuestion("CEIL-POLICY-01", "The v2 loose+512 policy removes a judging ceiling without claiming model improvement.", "policy", ("judgable_rate", "correctness_rate", "output_tokens"), "P0", "v2 does not improve judgability on the same outputs"),
    )
    cells: list[ExperimentCell] = []
    for model in model_values:
        cells.append(ExperimentCell(f"baseline:{model}", "baseline", model, {factor.id: factor.control_level for factor in factors}, True, seed, f"baseline:{model}"))
        for factor in factors:
            for level in factor.levels:
                if level == factor.control_level:
                    continue
                cells.append(ExperimentCell(f"ablation:{model}:{factor.id}:{level}", "ablation", model, {item.id: (level if item.id == factor.id else item.control_level) for item in factors}, True, seed, f"ablation:{model}:{factor.id}"))
        cells.append(ExperimentCell(f"policy:{model}:v1", "policy", model, {item.id: ("v1_192_normalized_contains" if item.id == "policy" else item.control_level) for item in factors}, True, seed, "policy:model-v1-v2"))
        cells.append(ExperimentCell(f"policy:{model}:v2", "policy", model, {item.id: ("v2_512_loose_contains" if item.id == "policy" else item.control_level) for item in factors}, True, seed, "policy:model-v1-v2"))
    for role in ("single", "draft_verify"):
        cells.append(ExperimentCell(f"roles:{role}", "roles", model_values[-1], {item.id: (role if item.id == "role" else item.control_level) for item in factors}, True, seed, "roles:single-vs-draft-verify"))
    evidence = (
        PublicEvidence("PUB-QWEN25", "Qwen2.5 Technical Report", "https://arxiv.org/abs/2412.15115", "training/post-training and model-scale claims", "context for public comparison; never substitute for local runs"),
        PublicEvidence("PUB-QWEN3", "Qwen3 Technical Report", "https://arxiv.org/abs/2505.09388", "0.6B-to-MoE family and reasoning-mode claims", "record architecture/mode claims before testing"),
        PublicEvidence("PUB-MINICPM4", "MiniCPM4 technical report entry", "https://github.com/OpenBMB/MiniCPM", "end-device efficiency claims", "compare reported efficiency mechanisms, not local quality"),
        PublicEvidence("PUB-LLAMA-QUANT", "llama.cpp quantization documentation", "https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md", "quantization mechanics and quality caveat", "define paired quantization evidence requirements"),
    )
    return CeilingStudyPlan("qlh.harness.ceiling_study.v1", model_values, questions, factors, tuple(cells), evidence, fixture_set_digest, {"match": "loose_contains", "max_new_tokens": 512, "historical_policy": {"match": "normalized_contains", "max_new_tokens": 192}}, seed)


def compare_factor(
    baseline: Mapping[str, float],
    variant: Mapping[str, float],
    *,
    higher_is_better: Iterable[str] = ("quality_rate", "format_rate", "early_recall_rate", "early_fact_recall", "judgable_rate", "correctness_rate", "quality_per_cost"),
) -> dict[str, Any]:
    """Return bounded deltas; it does not infer statistical significance."""

    higher = set(higher_is_better)
    keys = sorted(set(baseline) & set(variant))
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    for key in keys:
        left, right = float(baseline[key]), float(variant[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("factor metrics must be finite")
        delta = right - left
        deltas[key] = delta
        signed = delta if key in higher else -delta
        if signed > 0:
            improvements.append(key)
        elif signed < 0:
            regressions.append(key)
    return {"deltas": deltas, "improvements": tuple(improvements), "regressions": tuple(regressions), "evidence_only": True}


def pareto_points(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Mark quality/cost points without selecting a production winner."""

    required = {"id", "quality_rate", "latency_p95_ms", "rss_peak_bytes"}
    values = [dict(row) for row in rows]
    if any(required - set(row) for row in values):
        raise ValueError("pareto rows require id, quality_rate, latency_p95_ms, and rss_peak_bytes")
    output: list[dict[str, Any]] = []
    for current in values:
        dominated = False
        for other in values:
            if other["id"] == current["id"]:
                continue
            quality_ok = float(other["quality_rate"]) >= float(current["quality_rate"])
            latency_ok = float(other["latency_p95_ms"]) <= float(current["latency_p95_ms"])
            rss_ok = int(other["rss_peak_bytes"]) <= int(current["rss_peak_bytes"])
            strict = quality_ok and (float(other["quality_rate"]) > float(current["quality_rate"]) or latency_ok and float(other["latency_p95_ms"]) < float(current["latency_p95_ms"]) or rss_ok and int(other["rss_peak_bytes"]) < int(current["rss_peak_bytes"]))
            if quality_ok and latency_ok and rss_ok and strict:
                dominated = True
                break
        current["dominated"] = dominated
        output.append(current)
    return tuple(output)


def evaluate_evidence_gate(record: EvidenceRecord, *, require_real_weights: bool = True) -> dict[str, Any]:
    reasons: list[str] = []
    if record.status != "collected":
        reasons.append("evidence_not_collected")
    if record.network_used:
        reasons.append("network_used")
    if require_real_weights and not record.weights_loaded:
        reasons.append("weights_not_loaded")
    if record.runner_kind in {"fixture", "injected", "unknown"}:
        reasons.append("non_production_runner")
    if not record.metrics:
        reasons.append("metrics_missing")
    return {"status": "candidate" if reasons else "measured", "promotable": False, "reasons": tuple(dict.fromkeys(reasons))}


__all__ = [
    "DEFAULT_MODEL_IDS",
    "CeilingStudyPlan",
    "EvidenceRecord",
    "ExperimentCell",
    "PublicEvidence",
    "ResearchFactor",
    "ResearchQuestion",
    "build_ceiling_study",
    "compare_factor",
    "evaluate_evidence_gate",
    "pareto_points",
]
