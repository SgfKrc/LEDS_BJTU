from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"
RULES = ROOT / "tools" / "docagent" / "docagent" / "data" / "rules.yaml"


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


def _rules_copy(path: Path, *, r4_level: str | None = None) -> None:
    payload = json.loads(RULES.read_text(encoding="utf-8"))
    if r4_level is not None:
        next(rule for rule in payload["rules"] if rule["id"] == "R4")["level"] = r4_level
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_audit_lock_and_dry_run_baseline_comparison(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    baseline = tmp_path / "artifacts" / "baseline.json"

    lock = _run(
        "audit", "--root", str(repo), "--json", "--lock", str(baseline),
        "--fail-on", "none", cwd=tmp_path,
    )
    assert lock.returncode == 0, lock.stderr
    locked = json.loads(baseline.read_text(encoding="utf-8"))
    assert locked["schema_version"] == "qlh.docagent.baseline.v1"
    assert len(locked["rules_fingerprint"]) == 64
    assert str(repo) not in baseline.read_text(encoding="utf-8")

    before = baseline.read_bytes()
    compare = _run(
        "scan", "--root", str(repo), "--baseline", str(baseline), "--dry-run",
        "--json", "--fail-on", "none", cwd=tmp_path,
    )
    assert compare.returncode == 0, compare.stderr
    assert json.loads(compare.stdout)["summary"] == {
        "new": 0,
        "gone": 0,
        "changed": 0,
        "affected_docs": 0,
        "new_documents": 0,
        "gone_documents": 0,
        "changed_documents": 0,
    }
    assert baseline.read_bytes() == before


def test_missing_baseline_is_rejected_before_report_output(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    missing = tmp_path / "missing-baseline.json"

    result = _run(
        "scan", "--root", str(repo), "--baseline", str(missing), "--json", "--fail-on", "none",
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "baseline" in result.stderr
    assert result.stdout == ""


def test_rules_fingerprint_mismatch_is_rejected(tmp_path: Path):
    repo, _ = _repo(tmp_path)
    baseline = tmp_path / "baseline.json"
    changed_rules = tmp_path / "rules-next.json"
    assert _run("audit", "--root", str(repo), "--lock", str(baseline), "--fail-on", "none", cwd=tmp_path).returncode == 0
    _rules_copy(changed_rules, r4_level="error")

    result = _run(
        "scan", "--root", str(repo), "--rules", str(changed_rules), "--baseline", str(baseline),
        "--json", "--fail-on", "none", cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "rules fingerprint mismatch" in result.stderr
    assert result.stdout == ""


def test_dry_run_delta_and_max_new_max_gone_gates(tmp_path: Path):
    repo, docs = _repo(tmp_path)
    baseline = tmp_path / "baseline.json"
    assert _run("audit", "--root", str(repo), "--lock", str(baseline), "--fail-on", "none", cwd=tmp_path).returncode == 0
    (docs / "status.md").unlink()
    (docs / "new.md").write_text("# New document\n", encoding="utf-8")

    result = _run(
        "scan", "--root", str(repo), "--baseline", str(baseline), "--dry-run",
        "--json", "--max-new", "10", "--max-gone", "10", "--fail-on", "none", cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    delta = json.loads(result.stdout)
    assert delta["schema_version"] == "qlh.docagent.delta.v1"
    assert delta["summary"]["new"] == 2
    assert delta["summary"]["gone"] == 2
    assert delta["summary"]["changed"] == 0
    assert delta["summary"]["affected_docs"] == 2
    assert delta["affected_docs"] == ["docs/new.md", "docs/status.md"]

    max_new = _run(
        "scan", "--root", str(repo), "--baseline", str(baseline), "--dry-run",
        "--json", "--max-new", "1", "--fail-on", "none", cwd=tmp_path,
    )
    max_gone = _run(
        "scan", "--root", str(repo), "--baseline", str(baseline), "--dry-run",
        "--json", "--max-gone", "1", "--fail-on", "none", cwd=tmp_path,
    )
    assert max_new.returncode == 1
    assert max_gone.returncode == 1


def test_rules_diff_returns_structural_paths_and_fingerprints(tmp_path: Path):
    old = tmp_path / "rules-old.json"
    new = tmp_path / "rules-new.json"
    _rules_copy(old)
    _rules_copy(new, r4_level="error")

    result = _run("rules", "diff", "--old", str(old), "--new", str(new), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    diff = json.loads(result.stdout)
    assert diff["schema_version"] == "qlh.docagent.rules-diff.v1"
    assert diff["summary"]["identical"] is False
    assert any(change["path"] == "rules.R4.level" for change in diff["changed"])
    assert len(diff["old"]["fingerprint"]) == 64
    assert str(tmp_path) not in result.stdout


def test_rules_diff_rejects_invalid_rules_config(tmp_path: Path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    result = _run("rules", "diff", "--old", str(invalid), "--new", str(invalid), cwd=tmp_path)

    assert result.returncode == 2
    assert "configuration error" in result.stderr
    assert "invalid.json" in result.stderr
    assert result.stdout == ""
