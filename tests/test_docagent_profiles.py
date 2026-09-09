from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _profile(**overrides: object) -> dict[str, object]:
    profile: dict[str, object] = {
        "schema_version": "qlh.docagent.profile.v1",
        "profile_version": 1,
        "name": "fixture",
        "docs_dir": "manual",
        "exclude": ["private/**"],
        "vocabulary": {
            "status_exemptions": ["archived"],
            "stale_status_hints": ["planned"],
            "done_markers": ["done"],
            "topic_stop_words": ["document"],
        },
        "git": {"enabled": False, "status_scope": "manual", "source_root": "src"},
    }
    profile.update(overrides)
    return profile


def test_qlh_profile_excludes_nested_agent_tool_docs():
    from tools.docagent.docagent.scanner import scan_repository

    report = scan_repository(ROOT)

    assert report["profile"]["name"] == "qlh"
    assert all(not record["doc"].startswith("docs/agent_tool/") for record in report["docs"])
    assert all(str(ROOT) not in json.dumps(record, ensure_ascii=False) for record in report["docs"])


def test_minimal_profile_scans_nested_docs_without_git_findings(tmp_path: Path):
    repo = tmp_path / "project"
    (repo / "docs" / "nested").mkdir(parents=True)
    (repo / "docs" / "guide.md").write_text("> Status: current\n", encoding="utf-8")
    (repo / "docs" / "nested" / "guide.md").write_text("> Status: current\n", encoding="utf-8")

    result = _run("scan", "--root", str(repo), "--profile", "minimal", "--json", "--fail-on", "none", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert {record["doc"] for record in report["docs"]} == {"docs/guide.md", "docs/nested/guide.md"}
    assert all(finding["rule"] not in {"R2", "R3"} for record in report["docs"] for finding in record["findings"])
    assert str(repo) not in result.stdout


def test_custom_profile_applies_docs_dir_and_exclude(tmp_path: Path):
    repo = tmp_path / "project"
    (repo / "manual" / "private").mkdir(parents=True)
    (repo / "manual" / "guide.md").write_text("> Status: current\n", encoding="utf-8")
    (repo / "manual" / "private" / "secret.md").write_text("> Status: current\n", encoding="utf-8")
    profile = tmp_path / "profile.yaml"
    profile.write_text(json.dumps(_profile(), ensure_ascii=False), encoding="utf-8")

    result = _run("scan", "--root", str(repo), "--profile", str(profile), "--json", "--fail-on", "none", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["profile"]["name"] == "fixture"
    assert [record["doc"] for record in report["docs"]] == ["manual/guide.md"]
    assert str(repo) not in result.stdout

