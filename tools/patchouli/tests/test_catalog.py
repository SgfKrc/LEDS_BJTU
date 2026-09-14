"""PATCH-01 数据层测试：合成样本解析 + 真实仓库冒烟。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchouli.catalog import classify, main, parse_document, scan, summarize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_classify_kinds() -> None:
    assert classify("QLH定位与模块归属终局决策-2026-09-14") == "decision"
    assert classify("缓存机制专项计划-2026-09-13") == "special-plan"
    assert classify("reasonix-codex-bridge审计报告-2026-09-12") == "report"
    assert classify("TUI使用指南") == "guide"
    assert classify("随便一个名字") == "other"


def test_parse_document_metadata(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "docs/示例计划-2026-01-01.md",
        "# 示例计划\n\n> 状态：**生效**\n> 更新日期：2026-01-02\n\n"
        "引用 [总计划](总计划.md) 与 [归档](archive/旧计划.md)。\n涉及票 `PATCH-01`、`AUD-07`。\n",
    )
    entry = parse_document(path, tmp_path)
    assert entry["kind"] == "special-plan"
    assert entry["title"] == "示例计划"
    assert entry["status"] == "**生效**"
    assert entry["updated"] == "2026-01-02"
    assert entry["link_count"] == 2
    assert "PATCH-01" in entry["tickets"] and "AUD-07" in entry["tickets"]
    assert entry["archived"] is False
    assert entry["errors"] == []


def test_parse_document_fail_soft(tmp_path: Path) -> None:
    path = _write(tmp_path, "docs/无题.md", "没有标题与状态行的内容。")
    entry = parse_document(path, tmp_path)
    assert entry["title"] is None and entry["status"] is None
    assert "no-h1-title" in entry["errors"] and "no-status-line" in entry["errors"]
    assert entry["kind"] == "other"


def test_scan_counts_and_archive(tmp_path: Path) -> None:
    _write(tmp_path, "docs/A计划.md", "# A\n> 状态：现行\n")
    _write(tmp_path, "docs/archive/B报告.md", "# B\n> 状态：历史参考\n> 更新日期：2025-01-01\n")
    catalog = scan(tmp_path)
    assert catalog["doc_count"] == 2
    assert catalog["archived_count"] == 1
    assert catalog["no_status_count"] == 0
    assert "special-plan" in catalog["by_kind"]
    summary = summarize(catalog)
    assert "docs: 2" in summary


def test_cli_json_smoke(tmp_path: Path, capsys) -> None:
    _write(tmp_path, "docs/A.md", "# A\n> 状态：现行\n")
    assert main(["--root", str(tmp_path), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"qlh.patchouli.catalog.v1"' in out


def test_real_repo_catalog_smoke() -> None:
    """真实仓库冒烟：主仓 docs/ 全量可解析、计数稳定。"""
    catalog = scan(REPO_ROOT)
    assert catalog["doc_count"] >= 70  # 2026-09-14: 75 份 + archive
    assert catalog["archived_count"] >= 8
    assert catalog["ticket_count"] >= 20
    paths = {e["path"] for e in catalog["documents"]}
    assert any(p.startswith("docs/archive/") for p in paths)
    parsed = [e for e in catalog["documents"] if "read:" not in " ".join(e["errors"])]
    assert len(parsed) == catalog["doc_count"]  # 全部可读
