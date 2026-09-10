from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import defense_story as story


def load_storyline() -> dict:
    return json.loads(story.DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def test_current_storyline_is_exactly_five_minutes_and_covers_p1_to_p4():
    summary = story.validate_file()

    assert summary["duration_seconds"] == 300
    assert summary["segment_count"] == 6
    assert summary["covered_tickets"] == ["DEF-P1", "DEF-P2", "DEF-P3", "DEF-P4"]
    assert summary["required_evidence_count"] == 13
    assert summary["claim_guard"]["real_model_demonstrated"] is False


@pytest.mark.parametrize(
    ("segment_index", "field", "value"),
    [
        (2, "start_second", 81),
        (5, "end_second", 301),
    ],
)
def test_storyline_rejects_time_gaps_and_overrun(segment_index, field, value):
    payload = load_storyline()
    payload["segments"][segment_index][field] = value

    with pytest.raises(story.StorylineError, match="空洞、重叠或越界"):
        story.validate_storyline(payload)


def test_storyline_rejects_real_model_claims():
    payload = load_storyline()
    payload["claim_guard"]["real_model_demonstrated"] = True

    with pytest.raises(story.StorylineError, match="声明边界"):
        story.validate_storyline(payload)


def test_storyline_rejects_a_segment_without_fallback():
    payload = load_storyline()
    del payload["segments"][3]["fallback"]

    with pytest.raises(story.StorylineError, match="翻车兜底"):
        story.validate_storyline(payload)


def test_storyline_rejects_live_or_abandoned_frontend_commands():
    payload = load_storyline()
    payload["rehearsal_commands"]["windows"][2] = "scripts\\demo\\demo.bat --mode live frontend\\index.html"

    with pytest.raises(story.StorylineError, match="真实模型、live 或旧前端"):
        story.validate_storyline(payload)


def test_validation_report_contains_only_repo_relative_evidence(tmp_path):
    summary = story.validate_file()
    report = tmp_path / "story-report.json"

    story.write_report(summary, report)

    text = report.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["status"] == "passed"
    assert payload["document"] == "docs/答辩演示-5分钟故事线-2026-09-10.md"
    assert str(ROOT) not in text


def test_storyline_rejects_unknown_segment_evidence():
    payload = copy.deepcopy(load_storyline())
    payload["segments"][0]["evidence_ids"].append("missing-evidence")

    with pytest.raises(story.StorylineError, match="未知证据"):
        story.validate_storyline(payload)


def test_storyline_rejects_unindexed_document_links(tmp_path):
    payload = load_storyline()
    for evidence in payload["evidence"]:
        if evidence["required"]:
            evidence_path = tmp_path / evidence["path"]
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text("fixture evidence", encoding="utf-8")
    document = tmp_path / payload["document"]
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(
        (ROOT / payload["document"]).read_text(encoding="utf-8")
        + "\n[unindexed](../README.md)\n",
        encoding="utf-8",
    )

    with pytest.raises(story.StorylineError, match="未登记为证据"):
        story.validate_storyline(payload, root=tmp_path)
