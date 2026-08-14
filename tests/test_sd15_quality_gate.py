"""
SD 1.5 十种子质量门测试 — 人工审核登记（--review-report）与审核者解析。
====================================================================
覆盖 quality_gate_sd15.py 的 _apply_manual_reviews / _parse_reviewer /
--review-report CLI 登记路径（不重跑 GPU 推理）。
"""

import argparse
import json
import os
import sys

import pytest

pytestmark = pytest.mark.quality_gate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quality_gate_sd15 import (  # noqa: E402
    _apply_manual_reviews,
    _image_metrics,
    _parse_reviewer,
)


class TestImageMetricsThresholds:
    """T-2：automatic gate 阈值参数化（--min-entropy/--min-stddev）。

    合成高熵噪声图在默认阈值下通过；调高阈值后同一图必须失败，
    证明阈值确实参与判定（而非仅记录）。
    """

    @staticmethod
    def _noise_image():
        from PIL import Image
        import random
        rng = random.Random(42)
        img = Image.new("RGB", (64, 64))
        img.putdata(
            [(rng.randrange(256), rng.randrange(256), rng.randrange(256))
             for _ in range(64 * 64)]
        )
        return img

    def test_default_thresholds_pass_noise(self, tmp_path):
        metrics = _image_metrics(self._noise_image(), tmp_path / "a.png")
        assert metrics["automatic_pass"] is True

    def test_raised_entropy_threshold_fails_same_image(self, tmp_path):
        metrics = _image_metrics(
            self._noise_image(), tmp_path / "b.png",
            min_entropy=100.0,  # 高熵噪声图也到不了
        )
        assert metrics["automatic_pass"] is False

    def test_raised_stddev_threshold_fails_same_image(self, tmp_path):
        metrics = _image_metrics(
            self._noise_image(), tmp_path / "c.png",
            min_stddev=100.0,
        )
        assert metrics["automatic_pass"] is False


def _report(automatic_pass=True, reviews=None, status=None):
    return {
        "schema_version": 1,
        "asset_id": "sd15_90s_retrovers_v1",
        "automatic_gate": {"passed": automatic_pass},
        "manual_gate": {
            "passed": False,
            "required_reviewers": 2,
            "reviews": reviews or [],
        },
        "status": status or (
            "pending_manual_review" if automatic_pass else "failed"
        ),
    }


class TestParseReviewer:
    def test_valid_name_pass(self):
        assert _parse_reviewer("Alice=pass") == {"name": "Alice", "decision": "pass"}

    def test_valid_name_fail(self):
        assert _parse_reviewer("Bob=fail") == {"name": "Bob", "decision": "fail"}

    def test_name_stripped_and_decision_lowercased(self):
        assert _parse_reviewer("  浅草爱音 = PASS  ") == {
            "name": "浅草爱音", "decision": "pass",
        }

    @pytest.mark.parametrize("value", ["noseparator", "=pass", "Alice=maybe", "=fail"])
    def test_invalid_rejected(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_reviewer(value)


class TestApplyManualReviews:
    def test_two_passes_promote_to_passed(self):
        report = _report(reviews=[{"name": "Alice", "decision": "pass"}])
        status = _apply_manual_reviews(
            report, [{"name": "Bob", "decision": "pass"}],
        )
        assert status == "passed"
        assert report["manual_gate"]["passed"] is True
        assert report["status"] == "passed"
        names = {r["name"] for r in report["manual_gate"]["reviews"]}
        assert names == {"Alice", "Bob"}

    def test_same_reviewer_override_not_counted_twice(self):
        report = _report(reviews=[{"name": "Alice", "decision": "pass"}])
        status = _apply_manual_reviews(
            report, [{"name": "Alice", "decision": "pass"}],
        )
        assert status == "pending_manual_review"  # 只有一人（去重）

    def test_fail_blocks_promotion(self):
        report = _report(
            reviews=[
                {"name": "Alice", "decision": "pass"},
                {"name": "Bob", "decision": "pass"},
            ],
        )
        status = _apply_manual_reviews(
            report, [{"name": "Carol", "decision": "fail"}],
        )
        assert status == "failed"
        assert report["manual_gate"]["passed"] is False

    def test_single_pass_keeps_pending(self):
        report = _report(reviews=[])
        status = _apply_manual_reviews(
            report, [{"name": "Alice", "decision": "pass"}],
        )
        assert status == "pending_manual_review"

    def test_failed_automatic_gate_cannot_be_promoted(self):
        report = _report(automatic_pass=False)
        status = _apply_manual_reviews(
            report,
            [
                {"name": "Alice", "decision": "pass"},
                {"name": "Bob", "decision": "pass"},
            ],
        )
        assert status == "failed"

    def test_fail_override_replaces_previous_pass(self):
        report = _report(reviews=[{"name": "Alice", "decision": "pass"}])
        status = _apply_manual_reviews(
            report, [{"name": "Alice", "decision": "fail"}],
        )
        assert status == "failed"
        assert report["manual_gate"]["reviews"][0]["decision"] == "fail"


class TestReviewReportCli:
    """--review-report CLI 集成：不重跑推理，直接登记并写回报告。"""

    def _run_main(self, monkeypatch, report_path, reviewers):
        import quality_gate_sd15 as gate

        monkeypatch.setattr(
            sys, "argv",
            ["quality_gate_sd15", "--review-report", str(report_path), *reviewers],
        )
        return gate.main()

    def test_registers_reviews_and_writes_report(self, tmp_path, monkeypatch):
        report_path = tmp_path / "quality-report.json"
        report_path.write_text(
            json.dumps(_report(), ensure_ascii=False), encoding="utf-8",
        )
        code = self._run_main(
            monkeypatch, report_path,
            ["--reviewer", "Siegfried Kkm.=pass", "--reviewer", "浅草爱音=pass"],
        )
        assert code == 0  # passed
        updated = json.loads(report_path.read_text(encoding="utf-8"))
        assert updated["status"] == "passed"
        assert updated["manual_gate"]["passed"] is True
        names = {r["name"] for r in updated["manual_gate"]["reviews"]}
        assert names == {"Siegfried Kkm.", "浅草爱音"}

    def test_requires_reviewer_flag(self, tmp_path, monkeypatch, capsys):
        report_path = tmp_path / "quality-report.json"
        report_path.write_text(
            json.dumps(_report(), ensure_ascii=False), encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            self._run_main(monkeypatch, report_path, [])
        assert "requires at least one --reviewer" in str(exc.value)

    def test_rejects_non_quality_report(self, tmp_path, monkeypatch):
        report_path = tmp_path / "other.json"
        report_path.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            self._run_main(
                monkeypatch, report_path,
                ["--reviewer", "Alice=pass", "--reviewer", "Bob=pass"],
            )
        assert "must point to an SD 1.5 quality report" in str(exc.value)

    def test_fail_reviewer_returns_nonzero(self, tmp_path, monkeypatch):
        report_path = tmp_path / "quality-report.json"
        report_path.write_text(
            json.dumps(_report(), ensure_ascii=False), encoding="utf-8",
        )
        code = self._run_main(
            monkeypatch, report_path,
            ["--reviewer", "Alice=pass", "--reviewer", "Bob=fail"],
        )
        assert code == 1
        updated = json.loads(report_path.read_text(encoding="utf-8"))
        assert updated["status"] == "failed"
