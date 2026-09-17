import os

import qlh


def test_help_is_local_and_does_not_spawn(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(qlh.subprocess, "run", lambda *args, **kwargs: called.append(args))

    assert qlh.main(["--help"]) == 0
    assert "qlh chat" in capsys.readouterr().out
    assert called == []


def test_chat_fixture_uses_zero_dependency_smoke(monkeypatch):
    seen = {}

    def fake_smoke(path):
        seen["path"] = path
        return 0

    monkeypatch.setattr(qlh, "_load_fixture_smoke", fake_smoke)
    assert qlh.main(["chat", "--fixture", "fixtures/chat.sse"]) == 0
    assert seen["path"] == "fixtures/chat.sse"


def test_models_is_a_plain_tui_command(monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return type("Result", (), {"returncode": 3})()

    monkeypatch.setattr(qlh.subprocess, "run", fake_run)
    assert qlh.main(["models"]) == 3
    assert os.path.normpath(seen[0][1]).endswith(os.path.normpath("src/tui_admin.py"))
    assert seen[0][2] == "models"


def test_no_args_enters_unified_tui_and_auto_starts(monkeypatch):
    seen = {}

    class FakeTui:
        @staticmethod
        def main(args):
            seen["args"] = args
            return 0

    monkeypatch.setattr(qlh, "_load_tui_admin", lambda: FakeTui)
    assert qlh.main([]) == 0
    assert seen["args"] == ["--auto-start", "--screen", "chat"]


def test_chat_url_enters_unified_tui_without_remote_autostart(monkeypatch):
    seen = {}

    class FakeTui:
        @staticmethod
        def main(args):
            seen["args"] = args
            return 0

    monkeypatch.setattr(qlh, "_load_tui_admin", lambda: FakeTui)
    assert qlh.main(["chat", "--host", "http://100.100.52.106:8000",
                     "--route", "distributed_preferred"]) == 0
    assert seen["args"] == [
        "--screen", "chat", "--host", "100.100.52.106", "--port", "8000",
        "--route", "distributed_preferred",
    ]


def test_chat_help_is_local(capsys):
    assert qlh.main(["chat", "--help"]) == 0
    output = capsys.readouterr().out
    assert "qlh chat" in output
    assert "--fixture" in output
