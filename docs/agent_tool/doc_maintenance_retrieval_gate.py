"""Frozen, document-level retrieval quality gate for DOCAGENT M3.

The gate evaluates a reviewed local fixture only.  It does not change source
documents, use RAG output as repository truth, or print benchmark queries.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RETRIEVAL_BASELINE_SCHEMA = "qlh.docagent.retrieval_baseline.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
_EXPECTED_CASE_COUNT = 30
_EXPECTED_TOP_K = 5
_MIN_HIT_AT_5 = 0.6


class RetrievalBaselineError(ValueError):
    """Raised when an M3 benchmark cannot safely be reused."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetrievalBaselineError(f"{label} must be an object")
    return value


def _validate_doc_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("docs/")
        or not value.endswith(".md")
        or ".." in Path(value).parts
    ):
        raise RetrievalBaselineError(f"{label} is invalid")
    return value


def _sample(value: Any, index: int) -> dict[str, Any]:
    item = _mapping(value, f"samples[{index}]")
    required = {"id", "query", "target_docs", "target_sha256", "human_rationale"}
    if set(item) != required:
        raise RetrievalBaselineError(f"samples[{index}] has unsupported fields")
    case_id = item["id"]
    query = item["query"]
    target_docs = item["target_docs"]
    target_sha256 = _mapping(item["target_sha256"], f"samples[{index}].target_sha256")
    rationale = item["human_rationale"]
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise RetrievalBaselineError(f"samples[{index}].id is invalid")
    if not isinstance(query, str) or not query.strip() or len(query) > 512 or "\x00" in query:
        raise RetrievalBaselineError(f"samples[{index}].query is invalid")
    if not isinstance(target_docs, list) or not target_docs:
        raise RetrievalBaselineError(f"samples[{index}].target_docs is invalid")
    docs = [_validate_doc_id(doc, f"samples[{index}].target_docs") for doc in target_docs]
    if len(set(docs)) != len(docs) or set(target_sha256) != set(docs):
        raise RetrievalBaselineError(f"samples[{index}].target_docs is inconsistent")
    for doc, digest in target_sha256.items():
        _validate_doc_id(doc, f"samples[{index}].target_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise RetrievalBaselineError(f"samples[{index}].target_sha256 is invalid")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 512:
        raise RetrievalBaselineError(f"samples[{index}].human_rationale is invalid")
    return {
        "id": case_id,
        "query": query.strip(),
        "target_docs": sorted(docs),
        "target_sha256": {doc: target_sha256[doc] for doc in sorted(docs)},
        "human_rationale": rationale,
    }


def load_retrieval_baseline(path: str | Path) -> dict[str, Any]:
    """Load exactly thirty reviewed queries and their target document hashes."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalBaselineError("retrieval baseline is unreadable") from exc
    value = _mapping(raw, "retrieval baseline")
    required = {"schema_version", "baseline_id", "embedding_model", "minimum_hit_at_5", "samples"}
    if set(value) != required or value.get("schema_version") != RETRIEVAL_BASELINE_SCHEMA:
        raise RetrievalBaselineError("retrieval baseline schema is invalid")
    baseline_id = value["baseline_id"]
    model = value["embedding_model"]
    if not isinstance(baseline_id, str) or not _CASE_ID_RE.fullmatch(baseline_id):
        raise RetrievalBaselineError("retrieval baseline id is invalid")
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise RetrievalBaselineError("retrieval baseline embedding_model is invalid")
    if value["minimum_hit_at_5"] != _MIN_HIT_AT_5:
        raise RetrievalBaselineError("retrieval baseline minimum_hit_at_5 must be 0.6")
    samples_raw = value["samples"]
    if not isinstance(samples_raw, list) or len(samples_raw) != _EXPECTED_CASE_COUNT:
        raise RetrievalBaselineError("retrieval baseline must contain exactly thirty samples")
    samples = [_sample(item, index) for index, item in enumerate(samples_raw)]
    if len({sample["id"] for sample in samples}) != _EXPECTED_CASE_COUNT:
        raise RetrievalBaselineError("retrieval baseline sample ids must be unique")
    return {
        "schema_version": RETRIEVAL_BASELINE_SCHEMA,
        "baseline_id": baseline_id,
        "embedding_model": model.strip(),
        "minimum_hit_at_5": _MIN_HIT_AT_5,
        "samples": samples,
    }


def prepare_retrieval_baseline(baseline: Mapping[str, Any], audit: Mapping[str, Any]) -> None:
    """Reject a result when any labelled document has drifted from review."""
    records = audit.get("docs")
    if not isinstance(records, list):
        raise RetrievalBaselineError("audit has no document records")
    by_doc: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping) and isinstance(record.get("doc"), str):
            if record["doc"] in by_doc:
                raise RetrievalBaselineError("audit has duplicate document records")
            by_doc[record["doc"]] = record
    for sample in baseline["samples"]:
        for doc, expected_sha256 in sample["target_sha256"].items():
            record = by_doc.get(doc)
            if record is None:
                raise RetrievalBaselineError(f"target_not_found:{sample['id']}")
            if record.get("sha256") != expected_sha256:
                raise RetrievalBaselineError(f"target_hash_mismatch:{sample['id']}")


def _digest_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def evaluate_retrieval_baseline(
    baseline: Mapping[str, Any],
    search: Callable[[str, int], Sequence[Mapping[str, Any]]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Measure document hit@5 without including query text in the result."""
    if mode not in {"fts", "semantic"}:
        raise RetrievalBaselineError("retrieval mode is invalid")
    details: list[dict[str, Any]] = []
    hits = 0
    reciprocal_rank = 0.0
    for sample in baseline["samples"]:
        try:
            rows = list(search(sample["query"], _EXPECTED_TOP_K))
        except Exception as exc:
            raise RetrievalBaselineError("retrieval search failed") from exc
        if len(rows) > _EXPECTED_TOP_K:
            raise RetrievalBaselineError("retrieval search exceeded top_k")
        rank: int | None = None
        targets = set(sample["target_docs"])
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping) or not isinstance(row.get("doc_id"), str):
                raise RetrievalBaselineError("retrieval search returned invalid document reference")
            if row["doc_id"] in targets:
                rank = index
                break
        hit = rank is not None
        if hit:
            hits += 1
            reciprocal_rank += 1.0 / rank
        details.append({
            "case_id": sample["id"],
            "query_sha256": _digest_query(sample["query"]),
            "hit": hit,
            "rank": rank,
            "returned_count": len(rows),
        })
    count = len(baseline["samples"])
    hit_rate = hits / count if count else 0.0
    result = {
        "schema_version": "qlh.docagent.retrieval_quality.v1",
        "baseline_id": baseline["baseline_id"],
        "mode": mode,
        "status": "passed" if hit_rate >= baseline["minimum_hit_at_5"] else "failed",
        "case_count": count,
        "top_k": _EXPECTED_TOP_K,
        "hit_at_5": round(hit_rate, 6),
        "mean_reciprocal_rank": round(reciprocal_rank / count if count else 0.0, 6),
        "thresholds": {"minimum_hit_at_5": baseline["minimum_hit_at_5"]},
        "details": details,
    }
    return result
