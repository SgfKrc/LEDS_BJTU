from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "project"
    docs = repo / "docs"
    docs.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "docagent@test")
    _git(repo, "config", "user.name", "docagent")
    return repo, docs


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_scan_runs_from_an_unrelated_cwd_without_writing_target_docs(tmp_path: Path):
    repo, docs = _repo(tmp_path)
    document = docs / "status.md"
    document.write_text("> 状态：现行\n\n[broken](missing.md)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    before_text = document.read_text(encoding="utf-8")
    before_status = _git(repo, "status", "--short", "--", "docs/")

    result = _run("scan", "--root", str(repo), "--json", "--fail-on", "none", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ruleset_version"] == 1
    assert len(report["rules_fingerprint"]) == 64
    assert report["docs"][0]["doc"] == "docs/status.md"
    assert report["docs"][0]["findings"] == [{
        "rule": "R4",
        "level": "warn",
        "message": "失效链接 [broken](missing.md)",
    }]
    assert document.read_text(encoding="utf-8") == before_text
    assert _git(repo, "status", "--short", "--", "docs/") == before_status


def test_rules_command_validates_bundled_contract(tmp_path: Path):
    result = _run("rules", "--json", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "qlh.docagent.rules.v1"
    assert {rule["id"] for rule in payload["rules"]} == {"R1", "R2", "R3", "R4", "R5"}
    assert len(payload["rules_fingerprint"]) == 64


def test_init_only_creates_docagent_configuration(tmp_path: Path):
    repo, docs = _repo(tmp_path)
    document = docs / "existing.md"
    document.write_text("# Existing\n", encoding="utf-8")
    before = document.read_bytes()

    result = _run("init", "--root", str(repo), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert (repo / ".docagent" / "rules.yaml").is_file()
    config = json.loads((repo / ".docagent" / "config.json").read_text(encoding="utf-8"))
    assert config["docs_dir"] == "docs"
    assert document.read_bytes() == before
    assert not (repo / ".docagent" / "rules.yaml").samefile(document)


def test_scan_prefers_initialized_project_rules(tmp_path: Path):
    repo, docs = _repo(tmp_path)
    (docs / "status.md").write_text("# Title\n", encoding="utf-8")
    assert _run("init", "--root", str(repo), cwd=tmp_path).returncode == 0
    project_rules = repo / ".docagent" / "rules.yaml"
    payload = json.loads(project_rules.read_text(encoding="utf-8"))
    next(rule for rule in payload["rules"] if rule["id"] == "R5")["enabled"] = False
    project_rules.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = _run("scan", "--root", str(repo), "--json", "--fail-on", "none", cwd=tmp_path)

    assert result.returncode == 0
    findings = json.loads(result.stdout)["docs"][0]["findings"]
    assert all(finding["rule"] != "R5" for finding in findings)
    assert any(finding["rule"] == "R2" for finding in findings)


def test_audit_rejects_report_output_inside_target_docs(tmp_path: Path):
    repo, docs = _repo(tmp_path)
    (docs / "status.md").write_text("> 状态：现行\n", encoding="utf-8")

    result = _run(
        "audit", "--root", str(repo), "--output", str(docs / "audit.json"),
        "--fail-on", "none", cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "inside target docs" in result.stderr
    assert result.stdout == ""
    assert not (docs / "audit.json").exists()
