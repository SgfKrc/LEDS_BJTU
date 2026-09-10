from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "demo"))

import defense_preflight as preflight


def test_current_checklist_covers_all_categories_and_document_markers():
    manifest = preflight.load_manifest()

    assert set(manifest["categories"]) == preflight.EXPECTED_CATEGORIES
    assert {item["handler"] for item in manifest["automated_requirements"]} == preflight.KNOWN_HANDLERS
    assert len(manifest["automated_requirements"]) == 12
    assert len(manifest["manual_requirements"]) == 7
    assert len(manifest["recovery_playbook"]) == 7
    assert manifest["claim_guard"]["model_load_required"] is False


def test_checklist_rejects_an_automated_item_missing_from_document(tmp_path):
    payload = json.loads(preflight.MANIFEST.read_text(encoding="utf-8"))
    payload["automated_requirements"][0]["id"] = "device-missing-marker"
    manifest_path = tmp_path / "checklist.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(preflight.PreflightError, match="文档缺少"):
        preflight.load_manifest(manifest_path)


def test_model_scope_is_explicitly_deferred_without_enumerating_models():
    requirement = {
        "id": "model-scope",
        "category": "model",
        "handler": "model_scope",
        "blocking": False,
    }

    result = preflight._check_model_scope(requirement, preflight.PreflightConfig())

    assert result["state"] == "deferred"
    assert "model_load=not-required" in result["detail"]
    assert "real_model_claim=false" in result["detail"]


def test_required_frontend_port_failure_blocks_preflight(monkeypatch):
    manifest = preflight.load_manifest()

    def passing(requirement, _config):
        state = "deferred" if requirement["handler"] == "model_scope" else "passed"
        return preflight._result(requirement, state, "test")

    handlers = {name: passing for name in preflight.KNOWN_HANDLERS}
    handlers["frontend_port"] = lambda requirement, _config: preflight._result(
        requirement, "failed", "port=5174; state=in-use; owner=not-inspected"
    )
    monkeypatch.setattr(preflight, "HANDLERS", handlers)

    report = preflight.run_preflight(preflight.PreflightConfig(), manifest)

    assert report["status"] == "blocked"
    assert report["blocking_failures"] == ["port-frontend"]
    assert report["claim_guard"]["external_network_probed"] is False
    assert report["claim_guard"]["model_files_enumerated"] is False
    assert report["claim_guard"]["model_loaded"] is False


def test_optional_port_and_power_warnings_do_not_become_blocking(monkeypatch):
    manifest = preflight.load_manifest()

    def handler(requirement, _config):
        if requirement["id"] in {"power-supply", "port-api"}:
            return preflight._result(requirement, "warning", "manual-check")
        state = "deferred" if requirement["handler"] == "model_scope" else "passed"
        return preflight._result(requirement, state, "test")

    monkeypatch.setattr(preflight, "HANDLERS", {name: handler for name in preflight.KNOWN_HANDLERS})

    report = preflight.run_preflight(preflight.PreflightConfig(), manifest)

    assert report["status"] == "warning"
    assert report["blocking_failures"] == []
    assert report["warnings"] == ["power-supply", "port-api"]


def test_port_probe_releases_the_loopback_port():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    holder.close()

    assert preflight._port_available(port) is True

    repeat = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        repeat.bind(("127.0.0.1", port))
    finally:
        repeat.close()


def test_preflight_rejects_duplicate_service_ports():
    with pytest.raises(preflight.PreflightError, match="互不相同"):
        preflight.run_preflight(preflight.PreflightConfig(frontend_port=8000, api_port=8000))


def test_report_contains_no_absolute_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    report_path = tmp_path / "build" / "preflight.json"
    report = {
        "schema": "qlh.defense_preflight.v1",
        "status": "ready_for_manual_checks",
        "document": "docs/checklist.md",
        "checks": [],
    }

    preflight.write_report(report, report_path)

    encoded = report_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in encoded
    assert json.loads(encoded)["status"] == "ready_for_manual_checks"


def test_strict_warnings_returns_nonzero_without_turning_warning_into_block(monkeypatch):
    report = {
        "status": "warning",
        "checks": [],
        "manual_pending": ["manual-power-adapter"],
        "warnings": ["port-api"],
    }
    monkeypatch.setattr(preflight, "run_preflight", lambda _config: report)
    monkeypatch.setattr(preflight, "write_report", lambda _report, _path: None)

    exit_code = preflight.main(["--strict-warnings"])

    assert exit_code == 1
    assert report["status"] == "warning"
