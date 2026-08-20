"""Local ANN adoption gate for RAG-S5D.

This is a decision helper, not an ANN implementation.  The current SQLite
FTS5 plus bounded cosine scan remains the default until a user explicitly
installs and validates ``sqlite-vec`` against a corpus larger than the local
scan budget.  Keeping the decision deterministic prevents an optional native
extension from silently changing retrieval semantics.
"""
from __future__ import annotations

import importlib.util
from typing import Any


SQLITE_VEC_MODULE = "sqlite_vec"


def sqlite_vec_available() -> bool:
    """Return whether the optional extension is installed in this interpreter."""
    return importlib.util.find_spec(SQLITE_VEC_MODULE) is not None


def evaluate_ann_decision(
    *,
    corpus_chunks: int,
    scan_budget: int,
    extension_available: bool | None = None,
) -> dict[str, Any]:
    """Produce a path-free, conservative Go/No-Go decision.

    ``GO`` is intentionally not a production approval.  It only means the
    corpus is beyond the bounded scan budget and the optional extension is
    available for a separately controlled benchmark.  ``NO_GO`` preserves the
    existing deterministic scan path.
    """
    if isinstance(corpus_chunks, bool) or not 0 <= int(corpus_chunks) <= 10_000_000:
        raise ValueError("corpus_chunks is outside the allowed range")
    if isinstance(scan_budget, bool) or not 1 <= int(scan_budget) <= 10_000:
        raise ValueError("scan_budget is outside the allowed range")
    available = sqlite_vec_available() if extension_available is None else bool(extension_available)
    chunks = int(corpus_chunks)
    budget = int(scan_budget)
    if chunks <= budget:
        decision = "NO_GO"
        reason = "bounded_cosine_within_scan_budget"
    elif not available:
        decision = "NO_GO"
        reason = "sqlite_vec_not_installed"
    else:
        decision = "GO"
        reason = "benchmark_gate_only"
    return {
        "decision": decision,
        "reason": reason,
        "corpus_chunks": chunks,
        "scan_budget": budget,
        "extension": SQLITE_VEC_MODULE,
        "extension_available": available,
        "production_approved": False,
    }
