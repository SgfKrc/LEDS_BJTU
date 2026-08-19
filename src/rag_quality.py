"""Deterministic, redacted retrieval-quality gate for local RAG.

The evaluator consumes an operator-owned labelled query set and a search
callable. It never persists query text; reports contain only digests, ranks,
and citation-safe identifiers. The default gate is intentionally a benchmark
contract, not an assertion that a real embedding model already passes it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


QUALITY_SCHEMA = "qlh.rag_quality.v1"
_MAX_QUERY_CHARS = 512
_MAX_CASES = 10_000


class RagQualityError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True)
class RagQualityCase:
    case_id: str
    query: str
    relevant_source_ids: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RagQualityCase":
        if not isinstance(value, Mapping):
            raise RagQualityError("case_invalid", "quality case must be an object")
        case_id = value.get("case_id")
        query = value.get("query")
        source_ids = value.get("relevant_source_ids")
        chunk_ids = value.get("relevant_chunk_ids", ())
        if not isinstance(case_id, str) or not case_id.strip() or len(case_id) > 128:
            raise RagQualityError("case_invalid", "quality case_id is invalid")
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARS or "\x00" in query:
            raise RagQualityError("case_invalid", "quality query is invalid")
        if not isinstance(source_ids, (list, tuple)) or not source_ids:
            raise RagQualityError("case_invalid", "quality case needs relevant source ids")
        if not isinstance(chunk_ids, (list, tuple)):
            raise RagQualityError("case_invalid", "relevant chunk ids must be a list")
        normalized_sources = tuple(str(item).strip() for item in source_ids)
        normalized_chunks = tuple(str(item).strip() for item in chunk_ids)
        if any(not item or len(item) > 128 for item in (*normalized_sources, *normalized_chunks)):
            raise RagQualityError("case_invalid", "quality labels contain an invalid identifier")
        return cls(case_id.strip(), query.strip(), normalized_sources, normalized_chunks)


def load_quality_cases(path: str | Path, *, expected_count: int | None = 30) -> list[RagQualityCase]:
    """Load a local labelled query set; default S5B contract requires 30 cases."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RagQualityError("cases_unreadable", "quality case file is unreadable") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("cases")
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_CASES:
        raise RagQualityError("cases_invalid", "quality case file must contain a bounded non-empty list")
    if expected_count is not None and len(raw) != int(expected_count):
        raise RagQualityError("case_count_mismatch", f"quality benchmark requires exactly {expected_count} cases")
    cases = [RagQualityCase.from_mapping(item) for item in raw]
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise RagQualityError("case_duplicate", "quality case ids must be unique")
    return cases


def _digest_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def evaluate_rag_quality(
    cases: Sequence[RagQualityCase],
    search: Callable[[str, int], Sequence[Mapping[str, Any]]],
    *,
    top_k: int = 5,
    expected_count: int | None = 30,
    min_hit_rate: float = 0.8,
    min_citation_rate: float = 0.95,
) -> dict[str, Any]:
    """Evaluate source hit rate and citation completeness with redacted output."""
    if isinstance(top_k, bool) or not 1 <= int(top_k) <= 100:
        raise RagQualityError("config_invalid", "top_k is outside the allowed range")
    if expected_count is not None and len(cases) != int(expected_count):
        raise RagQualityError("case_count_mismatch", f"quality benchmark requires exactly {expected_count} cases")
    for threshold in (min_hit_rate, min_citation_rate):
        if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
            raise RagQualityError("config_invalid", "quality thresholds must be within 0..1")
    details: list[dict[str, Any]] = []
    hit_count = 0
    cited_hit_count = 0
    reciprocal_total = 0.0
    for case in cases:
        try:
            rows = list(search(case.query, int(top_k)))
        except Exception as exc:
            raise RagQualityError("search_failed", "quality search failed") from exc
        rows = rows[: int(top_k)]
        expected_sources = set(case.relevant_source_ids)
        expected_chunks = set(case.relevant_chunk_ids)
        hit_rank: int | None = None
        citation_valid = False
        for index, row in enumerate(rows, start=1):
            source_hit = str(row.get("source_id", "")) in expected_sources
            chunk_hit = bool(expected_chunks) and str(row.get("chunk_id", "")) in expected_chunks
            if source_hit or chunk_hit:
                hit_rank = index
                citation_valid = all(row.get(key) not in (None, "") for key in ("chunk_id", "source_id", "relative_ref"))
                break
        hit = hit_rank is not None
        if hit:
            hit_count += 1
            reciprocal_total += 1.0 / float(hit_rank)
            if citation_valid:
                cited_hit_count += 1
        details.append({
            "case_id": case.case_id,
            "query_sha256": _digest_query(case.query),
            "hit": hit,
            "rank": hit_rank,
            "citation_valid": citation_valid,
            "returned_count": len(rows),
        })
    count = len(cases)
    hit_rate = hit_count / count if count else 0.0
    citation_rate = cited_hit_count / hit_count if hit_count else 0.0
    mrr = reciprocal_total / count if count else 0.0
    status = "passed" if hit_rate >= float(min_hit_rate) and citation_rate >= float(min_citation_rate) else "failed"
    return {
        "schema": QUALITY_SCHEMA,
        "status": status,
        "case_count": count,
        "top_k": int(top_k),
        "hit_at_k": round(hit_rate, 6),
        "citation_rate": round(citation_rate, 6),
        "mean_reciprocal_rank": round(mrr, 6),
        "thresholds": {
            "min_hit_at_k": float(min_hit_rate),
            "min_citation_rate": float(min_citation_rate),
        },
        "details": details,
    }

