"""Offline hybrid RAG retrieval with bounded routes, fusion, and reuse."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .providers import EmbeddingProvider
from .query import QueryPlan, rewrite_query
from .store import RagHit, RagStore

_RERANK_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{1,63}|[\u3400-\u9fff]{2,32}")


def _rerank_score(query: str, hit: RagHit) -> float:
    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    normalized_text = unicodedata.normalize("NFKC", f"{hit.title} {hit.text}").casefold()
    query_tokens = set(_RERANK_TOKEN.findall(normalized_query))
    text_tokens = set(_RERANK_TOKEN.findall(normalized_text))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    phrase_bonus = 0.2 if len(query_tokens) > 1 and normalized_query in normalized_text else 0.0
    title_bonus = 0.1 if normalized_query and normalized_query in unicodedata.normalize("NFKC", hit.title).casefold() else 0.0
    return min(1.0, overlap + phrase_bonus + title_bonus)


@dataclass(frozen=True, slots=True)
class RagSearchConfig:
    top_k: int = 8
    per_route_k: int = 8
    rewrite_limit: int = 4
    rrf_k: int = 60
    fts_weight: float = 1.0
    embedding_weight: float = 1.0
    cache: bool = True
    max_embedding_candidates: int = 1000
    rerank_candidate_k: int = 32
    rerank_weight: float = 0.35
    rerank_mode: str = "lexical"

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 50 or not 1 <= self.per_route_k <= 50:
            raise ValueError("top_k and per_route_k must be between 1 and 50")
        if not 1 <= self.rewrite_limit <= 8 or self.rrf_k <= 0:
            raise ValueError("rewrite_limit or rrf_k is invalid")
        if (
            not math.isfinite(float(self.fts_weight))
            or not math.isfinite(float(self.embedding_weight))
            or self.fts_weight < 0
            or self.embedding_weight < 0
            or self.fts_weight + self.embedding_weight <= 0
        ):
            raise ValueError("at least one retrieval route must have a positive weight")
        if not 1 <= self.max_embedding_candidates <= 10_000:
            raise ValueError("max_embedding_candidates must be between 1 and 10000")
        if not 1 <= self.rerank_candidate_k <= 1_000:
            raise ValueError("rerank_candidate_k must be between 1 and 1000")
        if not math.isfinite(float(self.rerank_weight)) or not 0 <= float(self.rerank_weight) <= 1:
            raise ValueError("rerank_weight must be between 0 and 1")
        if self.rerank_mode not in {"lexical", "none"}:
            raise ValueError("rerank_mode must be lexical or none")

    def as_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "per_route_k": self.per_route_k,
            "rewrite_limit": self.rewrite_limit,
            "rrf_k": self.rrf_k,
            "fts_weight": self.fts_weight,
            "embedding_weight": self.embedding_weight,
            "cache": self.cache,
            "max_embedding_candidates": self.max_embedding_candidates,
            "rerank_candidate_k": self.rerank_candidate_k,
            "rerank_weight": self.rerank_weight,
            "rerank_mode": self.rerank_mode,
        }


@dataclass(frozen=True, slots=True)
class RagSearchResult:
    query: str
    rewritten_queries: tuple[str, ...]
    hits: tuple[RagHit, ...]
    route_counts: Mapping[str, int]
    cache_hit: bool = False
    candidate_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "rewritten_queries": list(self.rewritten_queries),
            "hits": [hit.as_dict() for hit in self.hits],
            "route_counts": dict(self.route_counts),
            "cache_hit": self.cache_hit,
            "candidate_count": self.candidate_count,
        }


class HybridRagRetriever:
    def __init__(
        self,
        store: RagStore,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        config: RagSearchConfig | None = None,
        expansions: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.config = config or RagSearchConfig()
        self.expansions = expansions

    def search(
        self,
        query: str,
        *,
        owner_scope: str = "local",
        source_ids: Sequence[str] = (),
        title_prefix: str | None = None,
        metadata_filters: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        limit: int | None = None,
        rewrite_limit: int | None = None,
        per_route_k: int | None = None,
        fts_weight: float | None = None,
        embedding_weight: float | None = None,
        rerank_candidate_k: int | None = None,
        rerank_weight: float | None = None,
        rerank_mode: str | None = None,
    ) -> RagSearchResult:
        config = self.config
        if any(value is not None for value in (rewrite_limit, per_route_k, fts_weight, embedding_weight, rerank_candidate_k, rerank_weight, rerank_mode)):
            config = replace(
                config,
                rewrite_limit=config.rewrite_limit if rewrite_limit is None else int(rewrite_limit),
                per_route_k=config.per_route_k if per_route_k is None else int(per_route_k),
                fts_weight=config.fts_weight if fts_weight is None else float(fts_weight),
                embedding_weight=config.embedding_weight if embedding_weight is None else float(embedding_weight),
                rerank_candidate_k=config.rerank_candidate_k if rerank_candidate_k is None else int(rerank_candidate_k),
                rerank_weight=config.rerank_weight if rerank_weight is None else float(rerank_weight),
                rerank_mode=config.rerank_mode if rerank_mode is None else str(rerank_mode),
            )
        effective_top_k = config.top_k if limit is None else int(limit)
        if not 1 <= effective_top_k <= 50:
            raise ValueError("limit must be between 1 and 50")
        plan = rewrite_query(query, expansions=self.expansions, max_variants=config.rewrite_limit)
        if not plan.variants:
            return RagSearchResult(query, (), (), {"fts": 0, "embedding": 0}, False, 0)
        cache_key = self._cache_key(plan, owner_scope, source_ids, title_prefix, metadata_filters, effective_top_k, config)
        if config.cache:
            cached = self.store.cache_get(cache_key, owner_scope=owner_scope)
            if cached:
                hits = tuple(_hit_from_dict(item) for item in cached.get("hits", ()) if isinstance(item, Mapping))
                return RagSearchResult(
                    query,
                    tuple(str(item) for item in cached.get("rewritten_queries", plan.variants)),
                    hits[:effective_top_k],
                    dict(cached.get("route_counts", {})),
                    True,
                    int(cached.get("candidate_count", len(hits))),
                )

        route_counts: dict[str, int] = {"fts": 0, "embedding": 0}
        ranked: dict[str, tuple[RagHit, float, set[str]]] = {}
        for route_index, variant in enumerate(plan.variants):
            hits = self.store.search(
                variant,
                owner_scope=owner_scope,
                limit=config.per_route_k,
                source_ids=tuple(source_ids),
                title_prefix=title_prefix,
                metadata_filters=metadata_filters,
            )
            route_counts["fts"] += len(hits)
            self._merge_ranked(ranked, hits, "fts", config.fts_weight, route_index, config)
        route_counts["fts_variants"] = len(plan.variants)

        if self.embedding_provider is not None and config.embedding_weight > 0:
            candidates = self.store.list_chunks(
                owner_scope=owner_scope,
                limit=config.max_embedding_candidates,
                source_ids=tuple(source_ids),
                title_prefix=title_prefix,
                metadata_filters=metadata_filters,
            )
            try:
                result = self.embedding_provider.embed([*plan.variants, *[item.text for item in candidates]])
                if len(result.vectors) != len(candidates) + len(plan.variants):
                    raise ValueError("embedding provider returned an unexpected vector count")
                embedding_hits: dict[str, RagHit] = {}
                for variant_index, query_vector_value in enumerate(result.vectors[:len(plan.variants)]):
                    query_vector = _vector(query_vector_value)
                    scored = sorted(
                        ((self._cosine(query_vector, _vector(vector)), hit) for hit, vector in zip(candidates, result.vectors[len(plan.variants):])),
                        key=lambda item: (-item[0], item[1].ordinal, item[1].chunk_id),
                    )[: config.per_route_k]
                    variant_hits = [RagHit(hit.source_id, hit.chunk_id, hit.title, hit.ordinal, hit.text, score, hit.granularity) for score, hit in scored if score > 0]
                    self._merge_ranked(ranked, variant_hits, "embedding", config.embedding_weight, variant_index, config)
                    for hit in variant_hits:
                        embedding_hits.setdefault(hit.chunk_id, hit)
                route_counts["embedding"] = len(embedding_hits)
                route_counts["embedding_variants"] = len(plan.variants)
            except Exception:
                route_counts["embedding_error"] = 1

        fused = sorted(ranked.values(), key=lambda item: (-item[1], item[0].ordinal, item[0].chunk_id))
        candidates = fused[:config.rerank_candidate_k]
        route_counts["rerank_candidate_count"] = len(candidates)
        route_counts["rerank_applied"] = int(config.rerank_mode == "lexical" and bool(candidates))
        max_fusion = max((float(score) for _, score, _ in candidates), default=0.0)
        final_rows: list[tuple[RagHit, float, set[str], float, float]] = []
        for hit, fusion_score, routes in candidates:
            normalized_fusion = float(fusion_score) / max_fusion if max_fusion > 0 else 0.0
            lexical_score = _rerank_score(plan.original, hit)
            final_score = (
                (1.0 - config.rerank_weight) * normalized_fusion + config.rerank_weight * lexical_score
                if config.rerank_mode == "lexical" else normalized_fusion
            )
            final_rows.append((hit, final_score, routes, float(fusion_score), lexical_score))
        final_rows.sort(key=lambda item: (-item[1], item[0].ordinal, item[0].chunk_id))
        final_hits = tuple(
            RagHit(
                hit.source_id, hit.chunk_id, hit.title, hit.ordinal, hit.text, float(score), hit.granularity,
                tuple(sorted(routes)), float(fusion_score), float(lexical_score), config.rerank_mode,
            )
            for hit, score, routes, fusion_score, lexical_score in final_rows[:effective_top_k]
        )
        response = RagSearchResult(plan.original, plan.variants, final_hits, route_counts, False, len(ranked))
        if config.cache:
            self.store.cache_put(
                cache_key,
                {
                    "rewritten_queries": list(plan.variants),
                    "hits": [hit.as_dict() for hit in final_hits],
                    "route_counts": route_counts,
                    "candidate_count": len(ranked),
                },
                owner_scope=owner_scope,
            )
        return response

    def _cache_key(
        self,
        plan: QueryPlan,
        owner_scope: str,
        source_ids: Sequence[str],
        title_prefix: str | None,
        metadata_filters: Mapping[str, Any] | None,
        effective_top_k: int,
        config: RagSearchConfig,
    ) -> str:
        material = json.dumps(
            {
                "query": plan.normalized,
                "owner_scope": owner_scope,
                "source_ids": sorted(str(item) for item in source_ids),
                "title_prefix": title_prefix or "",
                "metadata_filters": metadata_filters or {},
                "limit": effective_top_k,
                "config": config.as_dict(),
                "provider": type(self.embedding_provider).__name__ if self.embedding_provider is not None else "none",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return "rag_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _merge_ranked(
        self,
        ranked: dict[str, tuple[RagHit, float, set[str]]],
        hits: Sequence[RagHit],
        route: str,
        weight: float,
        route_index: int,
        config: RagSearchConfig,
    ) -> None:
        if weight <= 0:
            return
        for rank, hit in enumerate(hits):
            contribution = weight / (config.rrf_k + rank + 1)
            previous = ranked.get(hit.chunk_id)
            if previous is None:
                ranked[hit.chunk_id] = (hit, contribution, {route})
            else:
                ranked[hit.chunk_id] = (previous[0], previous[1] + contribution, previous[2] | {route})

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left or any(not math.isfinite(float(item)) for item in (*left, *right)):
            return 0.0
        left_norm = math.sqrt(sum(float(item) ** 2 for item in left))
        right_norm = math.sqrt(sum(float(item) ** 2 for item in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return max(0.0, sum(float(a) * float(b) for a, b in zip(left, right)) / (left_norm * right_norm))


def _vector(value: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def _hit_from_dict(value: Mapping[str, Any]) -> RagHit:
    return RagHit(
        str(value.get("source_id", "")),
        str(value.get("chunk_id", "")),
        str(value.get("title", "")),
        int(value.get("ordinal", 0)),
        str(value.get("text", "")),
        float(value.get("score", 0.0)),
        str(value.get("granularity", "fixed")),
        tuple(str(item) for item in value.get("routes", ()) if isinstance(item, str)),
        float(value["fusion_score"]) if value.get("fusion_score") is not None else None,
        float(value["rerank_score"]) if value.get("rerank_score") is not None else None,
        str(value["rerank_mode"]) if value.get("rerank_mode") is not None else None,
        float(value["keyword_score"]) if value.get("keyword_score") is not None else None,
        float(value["graph_score"]) if value.get("graph_score") is not None else None,
        tuple(str(item) for item in value.get("graph_entities", ()) if isinstance(item, str)),
    )


__all__ = ["HybridRagRetriever", "RagSearchConfig", "RagSearchResult"]
