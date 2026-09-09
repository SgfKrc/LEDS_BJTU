from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "docagent"))

from docagent.compat import run_compat  # noqa: E402
from docagent.scanner import scan_repository  # noqa: E402


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "project"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "docagent@test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "docagent"], check=True)
    (docs / "status.md").write_text("> 状态：现行\n\n[broken](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    return repo, docs


def test_compat_writes_historical_outputs_and_preserves_fail_on(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    repo, _ = _repo(tmp_path)

    code = run_compat(["--json", "--fail-on", "R4"], repo)

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    expected = scan_repository(repo)
    assert report["rules"] == expected["rules"]
    assert report["docs"] == expected["docs"]
    assert (repo / "build" / "doc-audit" / "audit.json").is_file()
    assert (repo / "build" / "doc-audit" / "audit.md").is_file()


def test_main_script_delegates_standard_flags_to_compat_layer(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "doc_maintenance_audit.py"),
            "--json",
            "--fail-on",
            "none",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ruleset_version"] == 1
    assert len(report["rules_fingerprint"]) == 64
    assert (ROOT / "build" / "doc-audit" / "audit.json").is_file()


def test_compat_markdown_is_a_single_stable_stdout_document(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    repo, _ = _repo(tmp_path)

    code = run_compat(["--markdown", "--fail-on", "none"], repo)

    assert code == 0
    output = capsys.readouterr().out
    assert output.startswith("# Documentation maintenance audit\n")
    assert "| docs/status.md | R4 | warn |" in output
