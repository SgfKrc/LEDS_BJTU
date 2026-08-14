import sys
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

pytestmark = pytest.mark.quality_gate


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_gate_sd15_instruction import (  # noqa: E402
    DEFAULT_CASES,
    _apply_manual_reviews,
    _image_metrics,
    main,
)


def test_instruction_gate_has_ten_distinct_commands_and_seeds():
    assert len(DEFAULT_CASES) == 10
    assert len({item[0] for item in DEFAULT_CASES}) == 10
    assert len({item[1] for item in DEFAULT_CASES}) == 10
    assert len({item[2] for item in DEFAULT_CASES}) == 10


def test_image_metrics_reject_unchanged_and_accept_structured_edit():
    source = Image.new("RGB", (64, 64), (20, 20, 20))
    ImageDraw.Draw(source).rectangle((12, 12, 51, 51), outline=(220, 220, 220), width=4)
    unchanged = _image_metrics(source, source.copy())
    edited = source.copy()
    ImageDraw.Draw(edited).rectangle((16, 16, 47, 47), fill=(80, 30, 30))
    changed = _image_metrics(source, edited)

    assert unchanged["automatic_pass"] is False
    assert changed["mae"] > unchanged["mae"]
    assert changed["edge_correlation"] > 0


def test_manual_gate_requires_two_distinct_passes():
    report = {
        "full_matrix": True,
        "automatic_gate": {"passed": True},
        "manual_gate": {"reviews": []},
    }

    assert _apply_manual_reviews(report, [{"name": "one", "decision": "pass"}]) == "pending_manual_review"
    assert report["manual_gate"]["reviews"][0]["reviewed_at"] > 0
    assert _apply_manual_reviews(report, [{"name": " ONE ", "decision": "pass"}]) == "pending_manual_review"
    assert len(report["manual_gate"]["reviews"]) == 1
    assert _apply_manual_reviews(report, [{"name": "two", "decision": "pass"}]) == "passed"


def test_reviewer_cannot_sign_before_generation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quality-gate", "--reviewer", "one=pass"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
