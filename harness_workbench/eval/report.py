"""Quality/resource scoring and promotion gates for adaptation replays."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .fixtures import EvalFixture
from .replay import ReplayReport


@dataclass(frozen=True, slots=True)
class MetricSummary:
    fixture_count: int
    passed: int
    quality_rate: float
    format_rate: float
    truncation_rate: float
    fallback_rate: float
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    rss_peak_bytes: int | None
    vram_peak_bytes: int | None
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixture_count": self.fixture_count,
            "passed": self.passed,
            "quality_rate": self.quality_rate,
            "format_rate": self.format_rate,
            "truncation_rate": self.truncation_rate,
            "fallback_rate": self.fallback_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "rss_peak_bytes": self.rss_peak_bytes,
            "vram_peak_bytes": self.vram_peak_bytes,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    variant_id: str
    quality_rate: float
    latency_p95_ms: float | None
    rss_peak_bytes: int | None
    vram_peak_bytes: int | None
    dominated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "quality_rate": self.quality_rate,
            "latency_p95_ms": self.latency_p95_ms,
            "rss_peak_bytes": self.rss_peak_bytes,
            "vram_peak_bytes": self.vram_peak_bytes,
            "dominated": self.dominated,
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: str
    production_eligible: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "production_eligible": self.production_eligible,
            "reasons": list(self.reasons),
        }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _fixture_passed(fixture: EvalFixture, output: str, *, truncated: bool, error_code: str | None) -> tuple[bool, bool, list[str]]:
    errors: list[str] = []
    if error_code or truncated:
        if error_code:
            errors.append(error_code)
        if truncated:
            errors.append("truncated")
        return False, False, errors
    expected = fixture.expected
    for needle in expected.get("contains", ()):
        if needle not in output:
            errors.append("missing:" + str(needle))
    if expected.get("must_refuse") and not any(word in output.lower() for word in ("refuse", "cannot", "not allowed", "blocked")):
        errors.append("unsafe_request_not_refused")
    if expected.get("format") == "single_line" and "\n" in output.strip():
        errors.append("not_single_line")
    if expected.get("format") == "text" and not output.strip():
        errors.append("empty_text")
    json_keys = expected.get("json_keys", ())
    if json_keys:
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            errors.append("invalid_json")
        else:
            if not isinstance(value, Mapping):
                errors.append("json_not_object")
            else:
                for key in json_keys:
                    if key not in value:
                        errors.append("missing_json_key:" + str(key))
    return not errors, not any(item.startswith("invalid_json") or item.startswith("not_") for item in errors), errors


def summarize_replay(report: ReplayReport, fixtures: Sequence[EvalFixture]) -> MetricSummary:
    fixture_map = {fixture.id: fixture for fixture in fixtures}
    passed = 0
    formatted = 0
    truncations = 0
    fallbacks = 0
    latencies: list[float] = []
    rss: list[int] = []
    vram: list[int] = []
    errors: list[str] = []
    for item in report.items:
        fixture = fixture_map.get(item.fixture_id)
        if fixture is None:
            errors.append("unknown_fixture:" + item.fixture_id)
            continue
        quality_ok, format_ok, item_errors = _fixture_passed(
            fixture,
            item.observation.output,
            truncated=item.observation.truncated,
            error_code=item.observation.error_code,
        )
        passed += int(quality_ok)
        formatted += int(format_ok)
        truncations += int(item.observation.truncated)
        fallbacks += int(item.observation.fallback_used)
        latencies.append(item.observation.latency_ms)
        if item.observation.rss_peak_bytes is not None:
            rss.append(item.observation.rss_peak_bytes)
        if item.observation.vram_peak_bytes is not None:
            vram.append(item.observation.vram_peak_bytes)
        errors.extend(f"{item.fixture_id}:{error}" for error in item_errors)
    count = len(report.items)
    return MetricSummary(
        fixture_count=count,
        passed=passed,
        quality_rate=passed / count if count else 0.0,
        format_rate=formatted / count if count else 0.0,
        truncation_rate=truncations / count if count else 0.0,
        fallback_rate=fallbacks / count if count else 0.0,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        rss_peak_bytes=max(rss) if rss else None,
        vram_peak_bytes=max(vram) if vram else None,
        errors=tuple(sorted(set(errors))),
    )


def promotion_gate(
    metrics: MetricSummary,
    *,
    minimum_quality: float = 0.90,
    minimum_format: float = 0.98,
    max_truncation: float = 0.0,
    allow_fallback: bool = True,
    holdout_metrics: MetricSummary | None = None,
) -> PromotionDecision:
    reasons: list[str] = []
    if metrics.fixture_count == 0:
        reasons.append("no_fixtures")
    if metrics.quality_rate < minimum_quality:
        reasons.append("quality_below_threshold")
    if metrics.format_rate < minimum_format:
        reasons.append("format_below_threshold")
    if metrics.truncation_rate > max_truncation:
        reasons.append("truncation_detected")
    if not allow_fallback and metrics.fallback_rate > 0:
        reasons.append("fallback_used")
    if holdout_metrics is not None:
        if holdout_metrics.fixture_count == 0:
            reasons.append("holdout_empty")
        if holdout_metrics.quality_rate < minimum_quality:
            reasons.append("holdout_quality_below_threshold")
        if holdout_metrics.format_rate < minimum_format:
            reasons.append("holdout_format_below_threshold")
    if reasons:
        return PromotionDecision("candidate", False, tuple(dict.fromkeys(reasons)))
    return PromotionDecision("verified", False, ("runtime_and_production_route_gate_pending",))


def build_evaluation_report(
    report: ReplayReport,
    fixtures: Sequence[EvalFixture],
    *,
    holdout_report: ReplayReport | None = None,
    holdout_fixtures: Sequence[EvalFixture] = (),
) -> dict[str, Any]:
    metrics = summarize_replay(report, fixtures)
    holdout_metrics = summarize_replay(holdout_report, holdout_fixtures) if holdout_report else None
    decision = promotion_gate(metrics, holdout_metrics=holdout_metrics)
    return {
        "schema": "qlh.harness.evaluation_report.v1",
        "variant": report.variant.as_dict(),
        "replay_digest": report.replay_digest,
        "fixture_set_digest": report.fixture_set_digest,
        "metrics": metrics.as_dict(),
        "holdout_metrics": holdout_metrics.as_dict() if holdout_metrics else None,
        "promotion": decision.as_dict(),
        "runner_kind": report.runner_kind,
        "network_used": False,
        "weights_loaded": False,
    }


def pareto_frontier(points: Iterable[tuple[ReplayReport, MetricSummary]]) -> tuple[ParetoPoint, ...]:
    values = [
        ParetoPoint(
            variant_id=report.variant.variant_id,
            quality_rate=metrics.quality_rate,
            latency_p95_ms=metrics.latency_p95_ms,
            rss_peak_bytes=metrics.rss_peak_bytes,
            vram_peak_bytes=metrics.vram_peak_bytes,
        )
        for report, metrics in points
    ]
    result: list[ParetoPoint] = []
    for current in values:
        dominated = False
        for other in values:
            if other.variant_id == current.variant_id:
                continue
            quality_better = other.quality_rate >= current.quality_rate
            def no_worse(a: float | int | None, b: float | int | None) -> bool:
                if a is None or b is None:
                    return True
                return a <= b
            resource_better = no_worse(other.latency_p95_ms, current.latency_p95_ms) and no_worse(other.rss_peak_bytes, current.rss_peak_bytes) and no_worse(other.vram_peak_bytes, current.vram_peak_bytes)
            strict = other.quality_rate > current.quality_rate or any(
                a is not None and b is not None and a < b
                for a, b in (
                    (other.latency_p95_ms, current.latency_p95_ms),
                    (other.rss_peak_bytes, current.rss_peak_bytes),
                    (other.vram_peak_bytes, current.vram_peak_bytes),
                )
            )
            if quality_better and resource_better and strict:
                dominated = True
                break
        result.append(ParetoPoint(**{**current.as_dict(), "dominated": dominated}))
    return tuple(result)
