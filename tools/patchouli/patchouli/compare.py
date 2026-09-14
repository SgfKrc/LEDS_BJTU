"""PATCH-07b RAG 参数对照：改写（rewrite）/ 重排（rerank）前后的检索对照（纯软件、不加载模型）。

- rewrite：复用主仓 `src/rag_store.rewrite_query`（规则层，可注入中文术语别名表）；
  fail-soft：导入失败时退回内置最小实现。
- 两路：baseline（单查询子串检索）vs enhanced（多变体融合 + lexical 重排）。
- 指标：MRR@k 公式对齐 `src/rag_quality.evaluate_rag_quality`（1/rank 平均）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from .search import MAX_RESULTS, search_docs

ALIASES: dict[str, tuple[str, ...]] = {
    "缓存": ("cache", "KV cache", "前缀缓存"),
    "票号": ("ticket",),
    "画像": ("profiler", "device_profiler"),
    "投机": ("speculative",),
    "量化": ("quantization", "quant"),
    "分布式": ("distributed",),
    "前缀": ("prefix",),
}


def _load_rewrite() -> Any:
    """优先复用主仓 rag_store.rewrite_query；失败退回内置最小实现。"""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from src.rag_store import rewrite_query  # type: ignore

        return rewrite_query
    except Exception:  # noqa: BLE001
        def _fallback(query: str, *, max_variants: int = 4, expansions: Any = None) -> tuple[str, ...]:
            normalized = " ".join(query.split())
            variants = [normalized] if normalized else []
            for source, replacements in {**ALIASES, **(expansions or {})}.items():
                if source in normalized:
                    for replacement in replacements:
                        candidate = normalized.replace(source, replacement, 1)
                        if candidate not in variants:
                            variants.append(candidate)
                        if len(variants) >= max_variants:
                            return tuple(variants)
            return tuple(variants)

        return _fallback


_REWRITE = _load_rewrite()


def rewrite_variants(query: str, max_variants: int = 4) -> list[str]:
    try:
        variants = _REWRITE(query, max_variants=max_variants, expansions=ALIASES)
    except Exception:  # noqa: BLE001
        variants = (query,)
    return list(dict.fromkeys(variants))[:max_variants]


def _rerank_score(row: dict[str, Any], query_terms: list[str]) -> float:
    score = 0.0
    name_lower = row.get("name", "").lower()
    line_lower = row.get("line", "").lower()
    for term in query_terms:
        if term and term in name_lower:
            score += 2.0  # 文件名命中权重最高
        if line_lower.startswith("- ") or line_lower.startswith("* "):
            score += 0.2  # 列表项（要点行）略加分
    if row.get("line_no") == 0:
        score += 0.5  # 元数据/文件名级命中
    return score


def compare_search(root: Path, catalog: dict[str, Any], query: str, top_k: int = 5) -> dict[str, Any]:
    """两路对照：返回 {"query","variants","baseline","enhanced","promoted","demoted","mrr_baseline","mrr_enhanced"}。"""
    variants = rewrite_variants(query)
    baseline = search_docs(Path(root), catalog, query, limit=MAX_RESULTS)

    # enhanced：多变体融合（按 (path,line) 去重，记录来源变体数）
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in variants:
        for row in search_docs(Path(root), catalog, variant, limit=MAX_RESULTS):
            key = (row["path"], row["line_no"])
            if key not in merged:
                merged[key] = dict(row, variants_hit=1)
            else:
                merged[key]["variants_hit"] += 1
    terms = [t.lower() for t in query.split() if t]
    enhanced = sorted(
        merged.values(),
        key=lambda r: (-(r["variants_hit"] + _rerank_score(r, terms)), r["name"], r["line_no"]),
    )

    def _rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for index, row in enumerate(rows[:top_k], start=1):
            ranks.setdefault(row["path"], index)
        return ranks

    base_docs = _rank_map(baseline)
    enh_docs = _rank_map(enhanced)
    # MRR@k（对齐 rag_quality：1/rank 平均；相关性=文档出现在两路任一 top-k）
    relevant = set(base_docs) | set(enh_docs)

    def _mrr(ranks: dict[str, int]) -> float:
        if not relevant:
            return 0.0
        total = sum(1.0 / rank for path, rank in ranks.items() if path in relevant)
        return total / len(relevant)

    promoted = [p for p in enh_docs if (p not in base_docs) or (enh_docs[p] < base_docs[p])]
    demoted = [p for p in base_docs if p not in enh_docs]
    return {
        "query": query,
        "variants": variants,
        "baseline": baseline[:top_k],
        "enhanced": enhanced[:top_k],
        "promoted": promoted,
        "demoted": demoted,
        "mrr_baseline": round(_mrr(base_docs), 4),
        "mrr_enhanced": round(_mrr(enh_docs), 4),
        "baseline_total": len(baseline),
        "enhanced_total": len(enhanced),
    }


def render_compare(result: dict[str, Any]) -> str:
    def _rows(rows: list[dict[str, Any]]) -> list[str]:
        out = []
        for index, row in enumerate(rows, start=1):
            loc = f"L{row['line_no']}" if row["line_no"] else "文件名"
            out.append(f"  {index}. [{loc}] {row['name']} — {row['line'][:60]}")
        return out or ["  （空）"]

    lines = [
        "== RAG 参数对照（rewrite / rerank 前后）==",
        f"查询: {result['query']}",
        f"改写变体({len(result['variants'])}): {result['variants']}",
        "",
        f"─ 基线（单查询）MRR@5={result['mrr_baseline']} · 总命中 {result['baseline_total']}",
        *_rows(result["baseline"]),
        "",
        f"─ 增强（改写+重排）MRR@5={result['mrr_enhanced']} · 总命中 {result['enhanced_total']}",
        *_rows(result["enhanced"]),
        "",
        f"提升进 top5: {len(result['promoted'])} 份 · 掉出 top5: {len(result['demoted'])} 份",
    ]
    if result["promoted"]:
        lines.append("  提升: " + " · ".join(Path(p).stem for p in result["promoted"]))
    if result["demoted"]:
        lines.append("  掉出: " + " · ".join(Path(p).stem for p in result["demoted"]))
    return "\n".join(lines)
