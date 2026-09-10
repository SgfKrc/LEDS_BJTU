from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "demo"))

import defense_demo as demo


def test_fixture_mode_preflight_does_not_require_python_server(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "frontend_cybergothic")
    frontend = demo.FRONTEND_ROOT
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    (frontend / "node_modules" / "vite").mkdir(parents=True)
    (frontend / "node_modules" / "vite" / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(demo.shutil, "which", lambda command: "runtime")

    errors = demo._validate_layout(demo.DemoConfig(mode="fixtures"))

    assert errors == []


def test_live_frontend_command_uses_loopback_and_separate_ports(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "frontend_cybergothic")
    demo.FRONTEND_ROOT.mkdir()
    (demo.FRONTEND_ROOT / "package.json").write_text("{}", encoding="utf-8")
    (demo.FRONTEND_ROOT / "node_modules" / "vite").mkdir(parents=True)
    (demo.FRONTEND_ROOT / "node_modules" / "vite" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api_server.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(demo.shutil, "which", lambda command: "runtime")

    config = demo.DemoConfig(mode="live", api_port=18000, frontend_port=15174)
    assert demo._validate_layout(config) == []
    assert demo._loopback_url(config.api_port, "/api/health") == "http://127.0.0.1:18000/api/health"


def test_report_contains_no_absolute_paths(tmp_path):
    report = tmp_path / "report.json"
    run = demo.DemoRun(demo.DemoConfig(report_path=report))
    run.record("preflight", True, "mode=fixtures")
    demo._write_report(run)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema"] == "qlh.defense_demo.v1"
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
    assert payload["steps"][0]["ok"] is True
    assert demo.DEFAULT_REPORT.is_file()


def test_cluster_port_skips_frontend_port():
    config = demo.DemoConfig(mode="live", api_port=18000, frontend_port=18001)

    assert demo._cluster_port(config) == 18002


def test_fixture_scenarios_validate_current_cybergothic_sources():
    run = demo.DemoRun(demo.DemoConfig())

    demo.verify_fixture_scenarios(run)

    assert [step["name"] for step in run.steps] == [
        "fixture-dialog",
        "fixture-image",
        "fixture-topology",
    ]
    assert "claim=fixture/redacted/not-live" in run.steps[-1]["detail"]


def test_topology_scenario_report_uses_direct_defense_route(monkeypatch, tmp_path):
    report = tmp_path / "topology-report.json"
    monkeypatch.setattr(demo, "BUILD_ROOT", tmp_path / "logs")
    monkeypatch.setattr(demo, "DEFAULT_REPORT", tmp_path / "latest.json")
    monkeypatch.setattr(demo, "wait_for_ready", lambda _url, _timeout: None)
    monkeypatch.setattr(demo, "_start_process", lambda *_args, **_kwargs: None)

    result = demo.run_demo(demo.DemoConfig(
        mode="fixtures",
        scenario="topology",
        frontend_port=15185,
        duration=0,
        report_path=report,
    ))

    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["frontend_url"] == "http://127.0.0.1:15185/#/cluster?fixtures=1&defense=topology"


def test_scenario_is_rejected_outside_fixture_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "frontend_cybergothic")
    demo.FRONTEND_ROOT.mkdir()
    (demo.FRONTEND_ROOT / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api_server.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(demo.shutil, "which", lambda _command: "runtime")

    errors = demo._validate_layout(demo.DemoConfig(mode="live", scenario="topology"))

    assert "预置场景只能用于 fixtures 模式" in errors


def test_fixture_scenarios_reject_unknown_source(monkeypatch, tmp_path):
    manifest = tmp_path / "scenarios.json"
    manifest.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "kind": "dialog",
                        "route": "#/workbench?fixtures=1",
                        "markers": [{"source": "outside", "value": "marker"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(demo, "SCENARIO_MANIFEST", manifest)

    try:
        demo._load_fixture_scenarios()
    except RuntimeError as exc:
        assert "来源无效" in str(exc)
    else:
        raise AssertionError("unknown scenario source must be rejected")


def test_scenario_sources_follow_the_current_frontend_root(monkeypatch, tmp_path):
    frontend = tmp_path / "frontend_cybergothic"
    monkeypatch.setattr(demo, "FRONTEND_ROOT", frontend)

    sources = demo._scenario_sources()

    assert sources["fixture-data"] == frontend / "src" / "data" / "fixtures.ts"
    assert sources["topology-snapshot"] == frontend / "src" / "data" / "defense-topology.json"


def test_topology_snapshot_rejects_sensitive_node_fields():
    snapshot = json.loads((
        Path(__file__).parents[1]
        / "frontend_cybergothic"
        / "src"
        / "data"
        / "defense-topology.json"
    ).read_text(encoding="utf-8"))
    snapshot["nodes"][0]["address"] = "10.0.0.1"

    try:
        demo._validate_topology_snapshot(snapshot)
    except RuntimeError as exc:
        assert "IPv4" in str(exc) or "脱敏白名单" in str(exc)
    else:
        raise AssertionError("sensitive topology node fields must be rejected")


def test_topology_snapshot_rejects_layer_gaps():
    snapshot = json.loads((
        Path(__file__).parents[1]
        / "frontend_cybergothic"
        / "src"
        / "data"
        / "defense-topology.json"
    ).read_text(encoding="utf-8"))
    snapshot["layer_plan"]["assignments"][1]["start_layer"] = 9

    try:
        demo._validate_topology_snapshot(snapshot)
    except RuntimeError as exc:
        assert "连续" in str(exc)
    else:
        raise AssertionError("gapped topology layer plans must be rejected")


def test_startup_timeout_is_capped_by_two_minute_budget(monkeypatch):
    monkeypatch.setattr(demo.time, "monotonic", lambda: 110.0)

    timeout = demo._startup_timeout(demo.DemoConfig(startup_timeout=30.0), 0.0)

    assert timeout == 10.0


def test_fixture_mode_requires_only_current_frontend_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "frontend_cybergothic")
    demo.FRONTEND_ROOT.mkdir()
    (demo.FRONTEND_ROOT / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(demo.shutil, "which", lambda command: "runtime")

    errors = demo._validate_layout(demo.DemoConfig(mode="fixtures"))

    assert errors == ["现行前端依赖未安装：请在 frontend_cybergothic 目录执行 npm ci"]


def test_failure_mode_does_not_require_frontend_or_npm(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "missing-frontend")
    monkeypatch.setattr(demo.shutil, "which", lambda _command: None)

    errors = demo._validate_layout(demo.DemoConfig(mode="failure"))

    assert errors == []


def test_failure_evidence_is_redacted_and_enforces_reassignment_order():
    snapshot = {
        "state": "completed",
        "stage_count": 1,
        "attempt_count": 2,
        "retry_count": 1,
        "stages": [{
            "state": "completed",
            "requested_provider": "remote_demo-worker-primary",
            "selected_provider": "demo-local-recovery",
            "last_retry_error_code": "remote_worker_disconnected",
            "output_available": True,
            "output": {"content": "must-not-be-retained"},
            "attempts": [
                {
                    "provider": "remote_demo-worker-primary",
                    "provider_kind": "remote_full_worker",
                    "provider_node_id": "demo-worker-primary",
                    "state": "expired",
                    "lease_epoch": 1,
                },
                {
                    "provider": "demo-local-recovery",
                    "provider_kind": "deterministic_fake",
                    "provider_node_id": "demo-master",
                    "state": "completed",
                    "lease_epoch": 2,
                },
            ],
        }],
    }

    evidence = demo._summarize_failure_workflow(snapshot, demo.FAILURE_WORKER_EXIT_CODE)
    demo._validate_failure_evidence(evidence)

    assert "must-not-be-retained" not in json.dumps(evidence)
    assert evidence["workflow"]["stage"]["last_retry_error_code"] == "remote_worker_disconnected"


def test_failure_mode_runs_real_worker_exit_and_recovers(monkeypatch, tmp_path):
    report = tmp_path / "failure-report.json"
    monkeypatch.setattr(demo, "BUILD_ROOT", tmp_path / "logs")
    monkeypatch.setattr(demo, "DEFAULT_REPORT", tmp_path / "latest.json")

    result = demo.run_demo(demo.DemoConfig(
        mode="failure",
        startup_timeout=30.0,
        report_path=report,
    ))

    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    failure = payload["failure_injection"]
    assert failure["execution_environment"]["process_count"] == 2
    assert failure["execution_environment"]["real_model_loaded"] is False
    assert failure["workflow"]["state"] == "completed"
    assert failure["workflow"]["retry_count"] == 1
    assert [item["state"] for item in failure["workflow"]["stage"]["attempts"]] == [
        "expired",
        "completed",
    ]
    assert "fixture-recovered" not in report.read_text(encoding="utf-8")
