"""文档维护 Agent M2.4：建议 patch 生成必须保持源文档不变。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TOOL_DIR = Path(__file__).resolve().parents[1] / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_suggestions import generate_suggestion_diff  # noqa: E402


def _repo(tmp_path: Path, content: str):
    repo = tmp_path / "repo"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    doc = docs / "x.md"
    doc.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "docs/x.md"], check=True)
    return repo, doc


def _report(suggestion: str, *, judgement="stale", confidence=0.9, doc="docs/x.md"):
    return {"judgements": [{
        "doc": doc, "judgement": judgement, "confidence": confidence,
        "suggestion": suggestion,
    }]}


def test_replacement_patch_is_valid_and_source_document_is_unchanged(tmp_path):
    repo, doc = _repo(tmp_path, "# X\n\n> 状态：规划\n\n正文\n")
    before = doc.read_bytes()
    manifest = generate_suggestion_diff(
        _report("> 状态：已完成"), repo, repo / "build" / "doc-audit"
    )
    assert doc.read_bytes() == before
    patch = repo / manifest["patch"]
    assert "> 状态：已完成" in patch.read_text(encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)


def test_missing_status_line_gets_reviewable_insertion(tmp_path):
    repo, doc = _repo(tmp_path, "# X\n\n正文\n")
    manifest = generate_suggestion_diff(
        _report("> 状态：需人工复核"), repo, repo / "build" / "doc-audit"
    )
    assert manifest["generated"] == ["docs/x.md"]
    assert doc.read_text(encoding="utf-8") == "# X\n\n正文\n"


@pytest.mark.parametrize("suggestion", ["建议改成完成", "> 状态：完成\n额外解释"])
def test_ambiguous_suggestion_is_skipped(tmp_path, suggestion):
    repo, _ = _repo(tmp_path, "# X\n\n> 状态：规划\n")
    manifest = generate_suggestion_diff(
        _report(suggestion), repo, repo / "build" / "doc-audit"
    )
    assert manifest["generated"] == []
    assert manifest["skipped"][0]["reason"] == "suggestion_is_not_a_single_status_line"


def test_accurate_or_low_confidence_result_does_not_generate_patch(tmp_path):
    repo, _ = _repo(tmp_path, "# X\n\n> 状态：规划\n")
    accurate = generate_suggestion_diff(
        _report("> 状态：已完成", judgement="accurate"),
        repo, repo / "build" / "accurate",
    )
    low = generate_suggestion_diff(
        _report("> 状态：已完成", confidence=0.5),
        repo, repo / "build" / "low",
    )
    assert accurate["generated"] == []
    assert low["generated"] == []


def test_document_escape_is_rejected(tmp_path):
    repo, _ = _repo(tmp_path, "# X\n")
    with pytest.raises(ValueError, match="escapes docs root"):
        generate_suggestion_diff(
            _report("> 状态：完成", doc="../outside.md"),
            repo, repo / "build" / "doc-audit",
        )
