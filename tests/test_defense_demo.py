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
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "api_server.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(demo.shutil, "which", lambda command: "runtime")

    errors = demo._validate_layout(demo.DemoConfig(mode="fixtures"))

    assert errors == []


def test_live_frontend_command_uses_loopback_and_separate_ports(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "FRONTEND_ROOT", tmp_path / "frontend_cybergothic")
    demo.FRONTEND_ROOT.mkdir()
    (demo.FRONTEND_ROOT / "package.json").write_text("{}", encoding="utf-8")
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


def test_cluster_port_skips_frontend_port():
    config = demo.DemoConfig(mode="live", api_port=18000, frontend_port=18001)

    assert demo._cluster_port(config) == 18002
