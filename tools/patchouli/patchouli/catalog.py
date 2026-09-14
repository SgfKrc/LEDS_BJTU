"""Patchouli 只读数据层（PATCH-01）。

扫描 docs/ 文档树，解析元数据：状态行 / 更新日期 / 票号 / 互链 / 分类。
只读、零第三方依赖、fail-soft（未知格式归"其他"，不崩溃）。

用法：
    python -m patchouli.catalog --root . --summary
    python -m patchouli.catalog --root . --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

_KIND_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("decision", re.compile(r"(决策|定位|基调|论证)")),
    ("ticket-plan", re.compile(r"(票计划|开发票|待完成|推进顺序|排期)")),
    ("special-plan", re.compile(r"(专项计划|支持计划|实施计划|计划|方案|展望|规划|基调)")),
    ("report", re.compile(r"(报告|审计|验收|实验|调查|调研|评估)")),
    ("guide", re.compile(r"(指南|说明|清单|手册|问答|清单)")),
    ("reference", re.compile(r"(架构|原理|接口|标准|速览|At-a-Glance|README)")),
)
_TICKET_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]+){1,5})\b")
_TICKET_BLOCKLIST = {"SHA", "UTF", "MD", "RFC", "ISO", "HTTP", "HTTPS", "AES", "RSA", "CRC", "UUID"}
_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")
_STATUS_RE = re.compile(r"^>\s*(?:文档)?状态[：:]\s*(.+?)\s*$", re.MULTILINE)
_UPDATED_RE = re.compile(r"^>\s*更新日期[：:]\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def classify(name: str) -> str:
    for kind, pattern in _KIND_RULES:
        if pattern.search(name):
            return kind
    return "other"


def parse_document(path: Path, root: Path) -> dict[str, Any]:
    """fail-soft parse of one markdown document (never raises on content)."""
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "name": path.stem,
        "kind": classify(path.stem),
        "archived": "archive" in path.parts,
        "status": None,
        "updated": None,
        "title": None,
        "tickets": [],
        "link_count": 0,
        "size_bytes": 0,
        "errors": [],
    }
    try:
        data = path.read_bytes()
        entry["size_bytes"] = len(data)
        text = data.decode("utf-8", errors="replace")
    except OSError as exc:  # noqa: BLE001
        entry["errors"].append(f"read: {exc!r}")
        return entry

    title = _TITLE_RE.search(text)
    if title:
        entry["title"] = title.group(1).strip()
    status = _STATUS_RE.search(text)
    if status:
        entry["status"] = status.group(1).strip()
    updated = _UPDATED_RE.search(text)
    if updated:
        entry["updated"] = updated.group(1)
    entry["tickets"] = sorted(
        value for value in set(_TICKET_RE.findall(text)) if value.split("-", 1)[0] not in _TICKET_BLOCKLIST
    )
    entry["link_count"] = len(_LINK_RE.findall(text))
    if entry["title"] is None:
        entry["errors"].append("no-h1-title")
    if entry["status"] is None:
        entry["errors"].append("no-status-line")
    return entry


def scan(root: Path, *, include_archive: bool = True) -> dict[str, Any]:
    root = Path(root)
    docs = root / "docs"
    if not docs.is_dir():
        raise SystemExit(f"docs/ not found under {root}")
    paths: list[Path] = sorted(docs.glob("*.md"))
    if include_archive:
        paths += sorted((docs / "archive").glob("*.md"))
    entries = [parse_document(p, root) for p in paths]
    by_kind: dict[str, int] = {}
    for entry in entries:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
    no_status = [e["path"] for e in entries if e["status"] is None]
    return {
        "schema": "qlh.patchouli.catalog.v1",
        "root": str(root),
        "doc_count": len(entries),
        "archived_count": sum(1 for e in entries if e["archived"]),
        "by_kind": dict(sorted(by_kind.items())),
        "no_status_count": len(no_status),
        "no_status_paths": no_status,
        "ticket_count": len({t for e in entries for t in e["tickets"]}),
        "documents": entries,
    }


def summarize(catalog: dict[str, Any]) -> str:
    lines = [
        f"docs: {catalog['doc_count']}（archive {catalog['archived_count']}） 票号 {catalog['ticket_count']}",
        f"分类: {catalog['by_kind']}",
        f"缺状态行: {catalog['no_status_count']}",
    ]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patchouli catalog (read-only docs scan)")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    catalog = scan(args.root.expanduser().resolve(), include_archive=not args.no_archive)
    if args.json:
        print(json.dumps(catalog, ensure_ascii=False, indent=2))
    else:
        print(summarize(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
