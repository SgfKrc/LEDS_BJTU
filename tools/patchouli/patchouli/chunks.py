"""PATCH-07a 父文档上下文：chunk 切分（标题分节）与 chunk→parent 映射（零依赖）。

"看一份 chunk 知全貌"：检索命中（行级）→ 映射回父文档 + 章节名 + 全貌入口（书架预览）。
chunk = 标题锚定的连续行区间；无标题文档整篇为一个 chunk。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_chunks(text: str) -> list[dict[str, Any]]:
    """按 markdown 标题切分：[{heading, level, start_line, end_line}]（行号 1-based、闭区间）。"""
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_no, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            if current is not None:
                current["end_line"] = line_no - 1
                chunks.append(current)
            current = {
                "heading": match.group(2).strip(),
                "level": len(match.group(1)),
                "start_line": line_no,
                "end_line": len(lines),
            }
    if current is not None:
        chunks.append(current)
    elif lines:
        chunks.append({"heading": "（无标题）", "level": 0, "start_line": 1, "end_line": len(lines)})
    return chunks


def chunk_for_line(chunks: list[dict[str, Any]], line_no: int) -> dict[str, Any] | None:
    """定位某行所属 chunk（行号落在哪个区间）。"""
    for chunk in chunks:
        if chunk["start_line"] <= line_no <= chunk["end_line"]:
            return chunk
    return chunks[0] if chunks else None


def load_chunks(root: Path, doc_path: str) -> list[dict[str, Any]]:
    try:
        text = (Path(root) / doc_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return split_chunks(text)


def map_hits_to_parents(root: Path, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """行级检索结果 → 父文档聚合：[{path,name,kind,archived,hits,hittest_line,best_chunk,best_heading}]。"""
    by_doc: dict[str, dict[str, Any]] = {}
    for result in results:
        path = result["path"]
        entry = by_doc.setdefault(
            path,
            {
                "path": path,
                "name": result.get("name", Path(path).stem),
                "kind": result.get("kind", "other"),
                "archived": bool(result.get("archived")),
                "hits": 0,
                "lines": [],
                "best_chunk": None,
                "best_heading": None,
            },
        )
        entry["hits"] += 1
        if result.get("line_no"):
            entry["lines"].append(result["line_no"])
    for entry in by_doc.values():
        first_line = entry["lines"][0] if entry["lines"] else 0
        chunks = load_chunks(root, entry["path"])
        chunk = chunk_for_line(chunks, first_line) if first_line else (chunks[0] if chunks else None)
        if chunk:
            entry["best_chunk"] = [chunk["start_line"], chunk["end_line"]]
            entry["best_heading"] = chunk["heading"]
    return sorted(by_doc.values(), key=lambda e: (-e["hits"], e["name"]))


def render_parents(parents: list[dict[str, Any]], max_rows: int = 12) -> str:
    if not parents:
        return "（无父文档命中）"
    rows = ["== 父文档映射（chunk→parent）=="]
    for entry in parents[:max_rows]:
        span = f"L{entry['best_chunk'][0]}-{entry['best_chunk'][1]}" if entry["best_chunk"] else "?"
        head = entry["best_heading"] or "?"
        rows.append(f"[{entry['hits']} 命中] {entry['name']}  §{head}  {span}  ({entry['path']})")
    if len(parents) > max_rows:
        rows.append(f"… 其余 {len(parents) - max_rows} 份省略")
    return "\n".join(rows)
