"""PATCH-03 检索台：文档库元数据 + 全文检索（本地、零依赖）。

查询语法（空格分词，可混用）：
    纯文本          全文子串匹配（行级，大小写不敏感）
    t:PATCH-01      票号过滤（文档命中所含票号）
    k:report        类型过滤（catalog 分类）
    s:缺            仅"缺状态行"文档
    a:              包含归档（默认不含归档）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_RESULTS = 200


def parse_query(query: str) -> dict[str, Any]:
    spec: dict[str, Any] = {"terms": [], "tickets": [], "kinds": [], "missing_status": False, "archived": False}
    for token in query.split():
        if token.startswith("t:") and len(token) > 2:
            spec["tickets"].append(token[2:].upper())
        elif token.startswith("k:") and len(token) > 2:
            spec["kinds"].append(token[2:])
        elif token.startswith("s:"):
            spec["missing_status"] = True
        elif token.startswith("a:"):
            spec["archived"] = True
        else:
            spec["terms"].append(token.lower())
    return spec


def _entry_matches_meta(entry: dict[str, Any], spec: dict[str, Any]) -> bool:
    if entry.get("archived") and not spec["archived"]:
        return False
    if spec["missing_status"] and entry.get("status") is not None:
        return False
    if spec["kinds"] and entry.get("kind") not in spec["kinds"]:
        return False
    if spec["tickets"]:
        doc_tickets = {t.upper() for t in entry.get("tickets", [])}
        if not set(spec["tickets"]) & doc_tickets:
            return False
    return True


def search_docs(root: Path, catalog: dict[str, Any], query: str, limit: int = MAX_RESULTS) -> list[dict[str, Any]]:
    """返回行级结果：[{path, name, kind, archived, line_no, line}]；文件名命中记 line_no=0。"""
    spec = parse_query(query)
    results: list[dict[str, Any]] = []
    root = Path(root)
    for entry in catalog.get("documents", []):
        if not _entry_matches_meta(entry, spec):
            continue
        name_lower = entry["name"].lower()
        name_hit = bool(spec["terms"]) and all(term in name_lower for term in spec["terms"])
        if name_hit:
            results.append(
                {
                    "path": entry["path"],
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "archived": entry["archived"],
                    "line_no": 0,
                    "line": "（文件名命中）",
                }
            )
        if not spec["terms"]:
            # 纯元数据查询（t:/k:/s:/a:）：元数据命中的文档直接进结果（文档级）
            results.append(
                {
                    "path": entry["path"],
                    "name": entry["name"],
                    "kind": entry["kind"],
                    "archived": entry["archived"],
                    "line_no": 0,
                    "line": "（元数据命中）",
                }
            )
            if len(results) >= limit:
                return results
            continue
        try:
            text = (root / entry["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if all(term in line.lower() for term in spec["terms"]):
                results.append(
                    {
                        "path": entry["path"],
                        "name": entry["name"],
                        "kind": entry["kind"],
                        "archived": entry["archived"],
                        "line_no": line_no,
                        "line": line.strip()[:160],
                    }
                )
                if len(results) >= limit:
                    return results
    return results[:limit]


def context_snippet(root: Path, path: str, line_no: int, radius: int = 5) -> str:
    """取匹配行上下文（±radius 行，带行号）。line_no=0 → 文件头。"""
    try:
        lines = (Path(root) / path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # noqa: BLE001
        return f"读取失败: {exc!r}"
    if line_no <= 0:
        start, end = 0, min(len(lines), radius * 4)
    else:
        start = max(0, line_no - 1 - radius)
        end = min(len(lines), line_no + radius)
    rows = []
    for idx in range(start, end):
        marker = ">>" if idx == line_no - 1 else "  "
        rows.append(f"{marker}{idx + 1:>5} | {lines[idx][:150]}")
    return "\n".join(rows)
