"""Deterministic query normalization and bounded rewrite routes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original: str
    normalized: str
    variants: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"original": self.original, "normalized": self.normalized, "variants": list(self.variants)}


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "上下文": ("context",),
    "context": ("上下文",),
    "检索": ("retrieval", "search"),
    "retrieval": ("检索",),
    "搜索": ("search", "检索"),
    "数据库": ("database", "sqlite"),
    "database": ("数据库",),
    "sqlite": ("数据库",),
    "SQLite": ("数据库",),
    "分块": ("chunking", "chunk"),
    "chunking": ("分块",),
    "重排": ("rerank", "ranking"),
    "rerank": ("重排",),
    "怎么回事": ("原因", "说明"),
    "咋回事": ("原因", "说明"),
    "为啥": ("原因",),
    "咋办": ("解决", "处理"),
    "能不能": ("是否",),
}


def normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    value = unicodedata.normalize("NFKC", query).replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > 512:
        raise ValueError("query must be at most 512 characters")
    return value


def rewrite_query(
    query: str,
    *,
    expansions: Mapping[str, tuple[str, ...]] | None = None,
    max_variants: int = 4,
) -> QueryPlan:
    if not 1 <= max_variants <= 8:
        raise ValueError("max_variants must be between 1 and 8")
    normalized = normalize_query(query)
    if not normalized:
        return QueryPlan(query, normalized, ())
    aliases = dict(_ALIASES)
    if expansions:
        for key, values in expansions.items():
            clean_key = normalize_query(key)
            clean_values = tuple(normalize_query(item) for item in values if normalize_query(item))
            if clean_key and clean_values:
                aliases[clean_key] = clean_values[:4]
    variants: list[str] = [normalized]
    subqueries = tuple(part.strip() for part in re.split(r"\s+(?:and|or)\s+|[、；;]|(?:以及|并且)", normalized) if part.strip())
    for candidate in subqueries:
        if len(subqueries) > 1 and candidate not in variants:
            variants.append(candidate)
            if len(variants) >= max_variants:
                return QueryPlan(query, normalized, tuple(variants))
    for source, replacements in aliases.items():
        if source not in normalized:
            continue
        for replacement in replacements:
            candidate = re.sub(re.escape(source), replacement, normalized, count=1)
            candidate = normalize_query(candidate)
            if candidate and candidate not in variants:
                variants.append(candidate)
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break
    return QueryPlan(query, normalized, tuple(variants[:max_variants]))


__all__ = ["QueryPlan", "normalize_query", "rewrite_query"]
