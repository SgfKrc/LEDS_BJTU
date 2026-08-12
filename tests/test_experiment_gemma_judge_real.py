"""EX-N3 真实 Gemma 判题 runner 单元测试（mock Ollama，不依赖真实模型）。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def prompts_file(tmp_path):
    images = []
    items = []
    for i, (text, elements) in enumerate([
        ("a red apple on a wooden table", ["red apple", "wooden table"]),
        ("a blue car driving on a highway", ["blue car", "highway"]),
    ]):
        img = tmp_path / f"img-{i}.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        images.append(img)
        items.append({"image": str(img), "prompt": f"describe {text}", "key_elements": elements})
    return tmp_path, items, images


class TestNormalizeAndMatch:
    def test_normalize_strips_case_whitespace_punctuation(self):
        from scripts.experiment_gemma_judge_real import _normalize
        assert _normalize("Red  Apple!!") == "redapple"

    def test_match_counts_all_hits(self):
        from scripts.experiment_gemma_judge_real import _match_counts
        hits, total, topic = _match_counts(
            "A red apple sits on a wooden table.", ["red apple", "wooden table"])
        assert (hits, total, topic) == (2, 2, 1)

    def test_match_counts_partial_no_topic(self):
        from scripts.experiment_gemma_judge_real import _match_counts
        hits, total, topic = _match_counts(
            "A red apple on the floor.", ["red apple", "wooden table", "kitchen"])
        assert (hits, total, topic) == (1, 3, 0)


class TestJudgeRunner:
    def _run(self, tmp_path, items, responses):
        import scripts.experiment_gemma_judge_real as runner

        class FakePost:
            def __init__(self, payload, timeout, counter):
                self._payload = payload
                self._counter = counter

            def raise_for_status(self):
                return None

            def json(self):
                idx = self._counter[0]
                self._counter[0] += 1
                return {"choices": [{"message": {"content": responses[idx]}}]}

        class FakeClient:
            def __init__(self):
                self.posted = []
                self._counter = [0]

            def post(self, url, json, timeout):
                self.posted.append(json)
                return FakePost(json, timeout, self._counter)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        prompts = tmp_path / "prompts.json"
        prompts.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        result = tmp_path / "evidence.json"
        report = tmp_path / "report.json"

        fake = FakeClient()
        orig = runner.httpx.Client
        runner.httpx.Client = lambda: fake
        try:
            rc = runner.main([
                "--prompts", str(prompts),
                "--ollama-url", "http://fake:11434",
                "--result-file", str(result),
                "--report-file", str(report),
            ])
        finally:
            runner.httpx.Client = orig
        return rc, result, report, fake

    def test_real_evidence_contract_and_counts(self, tmp_path, prompts_file, monkeypatch):
        _, items, _ = prompts_file
        rc, result, report, fake = self._run(
            tmp_path, items, ["a red apple on a wooden table", "a blue car on a highway"])
        assert rc == 0
        evidence = json.loads(result.read_text(encoding="utf-8"))
        # 白名单字段
        assert set(evidence) == {
            "model", "judge_contract_id", "judge_contract_sha256",
            "topic_hit", "key_element_coverage",
        }
        assert evidence["model"] == "gemma4:12b"
        assert evidence["judge_contract_id"] == "gemma-judge-counts-v1"
        assert evidence["topic_hit"] == {"evaluated_count": 2, "passed_count": 2}
        assert evidence["key_element_coverage"] == {
            "evaluated_count": 4, "passed_count": 4}
        # 报告不含 prompt/图像路径/描述
        raw_report = report.read_text(encoding="utf-8")
        assert "prompt" not in raw_report
        assert "content" not in raw_report
        assert "img-" not in raw_report

    def test_failure_fails_closed(self, tmp_path, prompts_file):
        _, items, _ = prompts_file
        rc, result, report, _ = self._run(
            tmp_path, items, ["a red apple on a wooden table", ""])
        assert rc == 0
        evidence = json.loads(result.read_text(encoding="utf-8"))
        # 第一张命中，第二张"空输出"失败 → evaluated=2, passed=1
        assert evidence["topic_hit"] == {"evaluated_count": 2, "passed_count": 1}
        summary = json.loads(report.read_text(encoding="utf-8"))
        assert summary["summary"]["failures"] == 1
        assert summary["failures"][0]["error"] == "ValueError"

    def test_missing_image_fails_closed(self, tmp_path, prompts_file):
        _, items, _ = prompts_file
        items[1]["image"] = str(tmp_path / "does-not-exist.png")
        rc, result, report, _ = self._run(
            tmp_path, items, ["a red apple on a wooden table", "unused"])
        assert rc == 0
        evidence = json.loads(result.read_text(encoding="utf-8"))
        assert evidence["topic_hit"] == {"evaluated_count": 2, "passed_count": 1}
        summary = json.loads(report.read_text(encoding="utf-8"))
        assert summary["failures"][0]["error"] == "missing_file"
