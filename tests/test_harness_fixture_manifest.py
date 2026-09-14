"""Deterministic fixture manifest tests (Koakumix transitional preparation)."""
from __future__ import annotations

import json
from pathlib import Path

from harness_workbench.tools.fixture_manifest import (
    FIXTURE_MANIFEST_SCHEMA,
    MANIFEST_NAME,
    build_manifest,
    main,
    verify_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_FIXTURES = REPO_ROOT / "fixtures"
PS_QWEN3_NOTHINK_SHA256 = "c55caa2a1a5348924624c6aadec912eb4808aff7468b6b21c0f13c0a14d554a9"
RUBRIC_V2_SHA256 = "14d13d9d0eed02e29c62cf625653d7547dc6b5fe00c55cb60517642a93a60525"


def _make_fixture_tree(root: Path) -> None:
    (root / "prompt_sets" / "ps-x").mkdir(parents=True)
    (root / "prompt_sets" / "ps-x" / "prompts.jsonl").write_text('{"prompt_id": "a"}\n', encoding="utf-8")
    (root / "quality_rubrics").mkdir(parents=True)
    (root / "quality_rubrics" / "r1.json").write_text("{}\n", encoding="utf-8")
    (root / "experiment-plans").mkdir(parents=True)
    (root / "experiment-plans" / "p1.json").write_text("{}\n", encoding="utf-8")


def test_build_manifest_is_sorted_and_typed(tmp_path: Path) -> None:
    _make_fixture_tree(tmp_path)
    manifest = build_manifest(tmp_path)
    assert manifest["schema_version"] == FIXTURE_MANIFEST_SCHEMA
    assert manifest["entry_count"] == 3
    paths = [entry["path"] for entry in manifest["entries"]]
    assert paths == sorted(paths)
    assert paths == ["experiment-plans/p1.json", "prompt_sets/ps-x/prompts.jsonl", "quality_rubrics/r1.json"]
    for entry in manifest["entries"]:
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] >= 0


def test_manifest_matches_runner_pinned_fingerprints() -> None:
    """The manifest hashes the same files the EX-N3 runner pins in plans."""
    manifest = build_manifest(REAL_FIXTURES)
    by_path = {entry["path"]: entry["sha256"] for entry in manifest["entries"]}
    assert by_path["prompt_sets/ps-qwen3-nothink/prompts.jsonl"] == PS_QWEN3_NOTHINK_SHA256
    assert by_path["quality_rubrics/llm-objective-ps-v1-v2.json"] == RUBRIC_V2_SHA256


def test_verify_detects_drift_and_roundtrip(tmp_path: Path) -> None:
    _make_fixture_tree(tmp_path)
    manifest = build_manifest(tmp_path)
    target = tmp_path / MANIFEST_NAME
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok, message = verify_manifest(tmp_path)
    assert ok, message
    (tmp_path / "quality_rubrics" / "r1.json").write_text('{"changed": true}\n', encoding="utf-8")
    ok, _ = verify_manifest(tmp_path)
    assert not ok


def test_cli_check_reports_missing_manifest(tmp_path: Path) -> None:
    _make_fixture_tree(tmp_path)
    assert main(["--root", str(tmp_path), "--check"]) == 1
    assert main(["--root", str(tmp_path)]) == 0
    assert main(["--root", str(tmp_path), "--check"]) == 0
