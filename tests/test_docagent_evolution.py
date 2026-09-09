from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "docagent" / "run.py"
RULES = ROOT / "tools" / "docagent" / "docagent" / "data" / "rules.yaml"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _rules_payload() -> dict:
    return json.loads(RULES.read_text(encoding="utf-8"))


def _write_rules(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_rules_diff_accepts_candidate_rule_addition_and_removal(tmp_path: Path):
    old = _rules_payload()
    new = json.loads(json.dumps(old))
    new["rules"].append({
        "id": "R6",
        "name": "candidate rule",
        "level": "error",
        "enabled": True,
        "description": "candidate",
        "parameters": {},
    })
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    _write_rules(old_path, old)
    _write_rules(new_path, new)

    result = _run("rules", "diff", "--old", str(old_path), "--new", str(new_path), cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    diff = json.loads(result.stdout)
    assert any(item["path"] == "rules.R6" for item in diff["added"])
    assert len(diff["new"]["fingerprint"]) == 64


def test_low_risk_change_auto_approves_and_releases(tmp_path: Path):
    old = _rules_payload()
    new = json.loads(json.dumps(old))
    r3 = next(rule for rule in new["rules"] if rule["id"] == "R3")
    r3["parameters"]["topic_stop_words"].append("new-token")
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    preflight_path = tmp_path / "preflight.json"
    approved_path = tmp_path / "approved.json"
    released_path = tmp_path / "released.json"
    _write_rules(old_path, old)
    _write_rules(new_path, new)

    preflight = _run(
        "rules", "evolve", "--old", str(old_path), "--new", str(new_path),
        "--change-note", "extend the maintenance vocabulary", "--output", str(preflight_path),
        cwd=tmp_path,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert json.loads(preflight.stdout)["state"] == "preflight"

    approved = _run(
        "rules", "evolve", "--record", str(preflight_path), "--state", "approved",
        "--output", str(approved_path), cwd=tmp_path,
    )
    assert approved.returncode == 0, approved.stderr
    approved_record = json.loads(approved.stdout)
    assert approved_record["risk"] == "low"
    assert approved_record["approval"]["actor"] == "docagent:auto"
    assert approved_record["gate"] == "passed"

    released = _run(
        "rules", "evolve", "--record", str(approved_path), "--state", "released",
        "--output", str(released_path), cwd=tmp_path,
    )
    assert released.returncode == 0, released.stderr
    assert json.loads(released.stdout)["state"] == "released"


def test_warn_rule_removal_requires_bound_human_approval(tmp_path: Path):
    old = _rules_payload()
    new = json.loads(json.dumps(old))
    new["rules"] = [rule for rule in new["rules"] if rule["id"] != "R4"]
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    preflight_path = tmp_path / "preflight.json"
    approved_path = tmp_path / "approved.json"
    approval_path = tmp_path / "approval.json"
    _write_rules(old_path, old)
    _write_rules(new_path, new)

    blocked = _run(
        "rules", "evolve", "--old", str(old_path), "--new", str(new_path),
        "--change-note", "retire the obsolete link check", "--state", "approved",
        "--output", str(preflight_path), cwd=tmp_path,
    )
    assert blocked.returncode == 1
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["state"] == "preflight"
    assert preflight["risk"] == "high"
    assert preflight["gate"] == "pending_approval"
    assert any("removed_warn_rule:R4" == reason for reason in preflight["risk_reasons"])

    approval_path.write_text(json.dumps({
        "schema_version": "qlh.docagent.approval.v1",
        "decision": "approved",
        "actor": "reviewer",
        "timestamp": "2026-09-09T00:00:00+00:00",
        "note": "approved after reviewing the dry-run impact",
        "evolution_fingerprint": preflight["evolution_fingerprint"],
    }), encoding="utf-8")
    approved = _run(
        "rules", "evolve", "--record", str(preflight_path), "--state", "approved",
        "--approval", str(approval_path), "--output", str(approved_path), cwd=tmp_path,
    )
    assert approved.returncode == 0, approved.stderr
    record = json.loads(approved.stdout)
    assert record["state"] == "approved"
    assert record["approval"]["actor"] == "reviewer"


def test_evolution_requires_change_note_and_rejects_unbound_approval(tmp_path: Path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    approval_path = tmp_path / "approval.json"
    _write_rules(old_path, _rules_payload())
    _write_rules(new_path, _rules_payload())
    approval_path.write_text(json.dumps({
        "schema_version": "qlh.docagent.approval.v1",
        "decision": "approved",
        "actor": "reviewer",
        "timestamp": "2026-09-09T00:00:00+00:00",
        "note": "wrong binding",
        "evolution_fingerprint": "0" * 64,
    }), encoding="utf-8")

    missing_note = _run(
        "rules", "evolve", "--old", str(old_path), "--new", str(new_path), cwd=tmp_path,
    )
    assert missing_note.returncode == 2
    assert "change-note" in missing_note.stderr

    record_path = tmp_path / "record.json"
    created = _run(
        "rules", "evolve", "--old", str(old_path), "--new", str(new_path),
        "--change-note", "record an unchanged candidate", "--state", "preflight",
        "--output", str(record_path), cwd=tmp_path,
    )
    assert created.returncode == 0
    invalid = _run(
        "rules", "evolve", "--record", str(record_path), "--state", "approved",
        "--approval", str(approval_path), cwd=tmp_path,
    )
    assert invalid.returncode == 2
    assert "evolution_fingerprint" in invalid.stderr
