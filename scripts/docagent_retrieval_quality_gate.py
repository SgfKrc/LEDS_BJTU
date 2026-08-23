#!/usr/bin/env python3
"""Run the local DOCAGENT M3 retrieval-quality benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "docs" / "agent_tool"
for path in (ROOT, TOOL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from doc_maintenance_audit import scan_all  # noqa: E402
from doc_maintenance_embeddings import EmbeddingUnavailable, OllamaEmbeddingProvider  # noqa: E402
from doc_maintenance_events import DocEventStore  # noqa: E402
from doc_maintenance_llm import load_docagent_config  # noqa: E402
from doc_maintenance_retrieval_gate import (  # noqa: E402
    RetrievalBaselineError,
    evaluate_retrieval_baseline,
    load_retrieval_baseline,
    prepare_retrieval_baseline,
)


def _invalid(error: str) -> dict:
    return {"schema_version": "qlh.docagent.retrieval_quality.v1", "status": "invalid", "error": error}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate DOCAGENT document retrieval quality")
    parser.add_argument(
        "--baseline", default="fixtures/docagent/m3-retrieval-baseline-v1.json",
        help="reviewed 30-query baseline relative to repository root",
    )
    parser.add_argument("--mode", choices=("fts", "semantic", "all"), default="all")
    parser.add_argument("--embedding-model", help="override only when it matches the frozen baseline model")
    args = parser.parse_args(argv)
    try:
        baseline = load_retrieval_baseline(ROOT / args.baseline)
        requested_model = args.embedding_model or baseline["embedding_model"]
        if requested_model != baseline["embedding_model"]:
            raise RetrievalBaselineError("embedding_model_mismatch")
        audit = scan_all(None, "none")
        prepare_retrieval_baseline(baseline, audit)
    except (RetrievalBaselineError, ValueError):
        print(json.dumps(_invalid("baseline_or_audit_invalid"), ensure_ascii=True))
        return 2

    result: dict = {
        "schema_version": "qlh.docagent.retrieval_quality_run.v1",
        "baseline_id": baseline["baseline_id"],
        "mode": args.mode,
        "results": {},
    }
    try:
        with DocEventStore(ROOT / "build" / "docagent-events.sqlite") as store:
            result["chunk_index"] = store.index_chunks(audit, ROOT)
            if args.mode in {"fts", "all"}:
                result["results"]["fts"] = evaluate_retrieval_baseline(
                    baseline, store.search_documents, mode="fts",
                )
            if args.mode in {"semantic", "all"}:
                config = load_docagent_config(ROOT / ".env.docagent")
                provider = OllamaEmbeddingProvider(
                    base_url=config.ollama_base_url, model=baseline["embedding_model"],
                )
                result["embedding"] = store.index_embeddings(audit, provider, ROOT)
                result["results"]["semantic"] = evaluate_retrieval_baseline(
                    baseline,
                    lambda query, limit: store.semantic_search_documents(query, provider, limit),
                    mode="semantic",
                )
    except (EmbeddingUnavailable, RetrievalBaselineError, ValueError):
        print(json.dumps(_invalid("local_retrieval_unavailable_or_invalid"), ensure_ascii=True))
        return 2

    statuses = [item["status"] for item in result["results"].values()]
    result["status"] = "passed" if statuses and all(status == "passed" for status in statuses) else "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
