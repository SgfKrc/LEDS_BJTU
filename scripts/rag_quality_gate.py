#!/usr/bin/env python3
"""Run the local RAG 30-case quality gate without exposing query text."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_quality import RagQualityError, evaluate_rag_quality, load_quality_cases
from src.rag_embedding import DEFAULT_OLLAMA_EMBEDDING_MODEL, EmbeddingProviderError, OllamaEmbeddingProvider
from src.rag_store import RagStore, RagStoreError


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local SQLite RAG retrieval quality")
    parser.add_argument("--db", required=True, help="user-owned RAG SQLite path")
    parser.add_argument(
        "--cases", default=str(ROOT / "fixtures" / "rag_quality" / "rag_s5b_queries.json"),
        help="redacted labelled query set",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("fts", "hybrid"), default="fts")
    parser.add_argument("--model-id", default=DEFAULT_OLLAMA_EMBEDDING_MODEL)
    parser.add_argument("--model-sha256")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-citation-rate", type=float, default=0.95)
    parser.add_argument("--output", help="optional redacted JSON result path")
    args = parser.parse_args()
    try:
        cases = load_quality_cases(args.cases, expected_count=30)
        store = RagStore(args.db)
        if args.mode == "hybrid" and not args.model_sha256:
            raise RagQualityError("config_invalid", "hybrid quality mode requires --model-sha256")
        embedder = OllamaEmbeddingProvider(
            model_id=args.model_id,
            base_url=args.ollama_base_url,
            expected_dimensions=args.dimensions,
        ) if args.mode == "hybrid" else None

        def search(query, limit):
            if embedder is None:
                return store.search(query, access_scope="owner", limit=limit)
            vector = embedder.embed([query]).vectors[0]
            return store.hybrid_search(
                query, vector, provider="ollama", model_id=args.model_id,
                model_sha256=args.model_sha256, dimensions=args.dimensions,
                access_scope="owner", limit=limit,
            )

        result = evaluate_rag_quality(
            cases, search,
            top_k=args.top_k,
            expected_count=30,
            min_hit_rate=args.min_hit_rate,
            min_citation_rate=args.min_citation_rate,
        )
    except (RagQualityError, RagStoreError, EmbeddingProviderError) as exc:
        print(json.dumps({"schema": "qlh.rag_quality.v1", "status": "invalid", "error_code": exc.code}, ensure_ascii=True))
        return 2
    encoded = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
