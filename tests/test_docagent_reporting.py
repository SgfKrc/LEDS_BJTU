from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "project"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "docagent@test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "docagent"], check=True)
    (docs / "status.md").write_text("> Status: current\n\n[broken](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    return repo, docs


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_markdown_stdout_and_output_file_are_stable(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    output = tmp_path / "artifacts" / "audit.md"

    result = _run(
        "audit", "--root", str(repo), "--markdown", "--output", str(output),
        "--fail-on", "none", cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("# Documentation maintenance audit\n")
    assert "| docs/status.md | R4 | warn |" in result.stdout
    assert output.read_text(encoding="utf-8") == result.stdout
    assert str(repo) not in result.stdout
    assert "report:" in result.stderr


def test_json_stdout_stays_parseable_when_writing_output(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    output = tmp_path / "artifacts" / "audit.json"

    result = _run(
        "audit", "--root", str(repo), "--json", "--output", str(output),
        "--fail-on", "none", cwd=tmp_path,
    )

    assert result.returncode == 0
    stdout_report = json.loads(result.stdout)
    file_report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert "report:" in result.stderr


def test_fail_on_warn_and_error_return_distinct_gate_codes(tmp_path: Path):
    repo, _ = _repo(tmp_path)

    warn = _run("scan", "--root", str(repo), "--json", "--fail-on", "warn", cwd=tmp_path)
    error = _run("scan", "--root", str(repo), "--json", "--fail-on", "error", cwd=tmp_path)

    assert warn.returncode == 1
    assert error.returncode == 0
    assert json.loads(warn.stdout)["docs"]
    assert json.loads(error.stdout)["docs"]


def test_invalid_profile_reports_configuration_file_and_field(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    profile = tmp_path / "broken-profile.yaml"
    profile.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    result = _run(
        "scan", "--root", str(repo), "--profile", str(profile), "--json", "--fail-on", "none",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "configuration error" in result.stderr
    assert "broken-profile.yaml" in result.stderr
    assert "missing fields" in result.stderr
    assert result.stdout == ""


def test_invalid_rules_reports_configuration_error(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    rules = tmp_path / "broken-rules.yaml"
    rules.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    result = _run(
        "scan", "--root", str(repo), "--rules", str(rules), "--json", "--fail-on", "none",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "configuration error" in result.stderr
    assert "broken-rules.yaml" in result.stderr
    assert "unsupported rules schema" in result.stderr
