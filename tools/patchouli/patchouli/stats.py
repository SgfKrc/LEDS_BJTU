"""PATCH-06 馆藏统计：catalog 聚合视图（零依赖）。"""
from __future__ import annotations

from collections import Counter
from typing import Any


def compute_stats(catalog: dict[str, Any]) -> dict[str, Any]:
    docs = catalog.get("documents", [])
    total = len(docs)
    archived = sum(1 for d in docs if d.get("archived"))
    by_kind: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    status_counts = {"with": 0, "missing": 0}
    ticket_docs: Counter[str] = Counter()
    link_total = 0
    no_links = 0
    for doc in docs:
        by_kind[doc.get("kind", "other")] += 1
        month = (doc.get("updated") or "")[:7] or "未知"
        by_month[month] += 1
        if doc.get("status"):
            status_counts["with"] += 1
        else:
            status_counts["missing"] += 1
        for ticket in doc.get("tickets", []):
            ticket_docs[ticket] += 1
        link_total += doc.get("link_count", 0)
        if doc.get("link_count", 0) == 0:
            no_links += 1
    return {
        "total": total,
        "archived": archived,
        "active": total - archived,
        "by_kind": dict(by_kind.most_common()),
        "by_month": dict(sorted(by_month.items(), reverse=True)[:8]),
        "status_with": status_counts["with"],
        "status_missing": status_counts["missing"],
        "ticket_count": len(ticket_docs),
        "top_tickets": ticket_docs.most_common(8),
        "link_total": link_total,
        "no_links": no_links,
    }


def render_stats(stats: dict[str, Any]) -> str:
    total = stats["total"] or 1
    coverage = 100.0 * stats["status_with"] / total
    kinds = " · ".join(f"{k} {v}" for k, v in stats["by_kind"].items())
    months = " · ".join(f"{m} {v}" for m, v in stats["by_month"].items())
    tickets = " · ".join(f"{t}({n})" for t, n in stats["top_tickets"]) or "-"
    return "\n".join(
        [
            "== 馆藏统计 ==",
            f"总量: {stats['total']}（现行 {stats['active']} · 归档 {stats['archived']}）",
            f"状态行覆盖: {stats['status_with']}/{stats['total']}（{coverage:.1f}%）· 缺 {stats['status_missing']}",
            "",
            f"分类: {kinds}",
            f"更新时间（近 8 月）: {months}",
            "",
            f"票号: {stats['ticket_count']} 个 · Top: {tickets}",
            f"互链: 总 {stats['link_total']} · 无互链文档 {stats['no_links']}",
        ]
    )
