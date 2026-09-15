import os

import qlh


def test_help_is_local_and_does_not_spawn(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(qlh.subprocess, "run", lambda *args, **kwargs: called.append(args))

    assert qlh.main(["--help"]) == 0
    assert "qlh chat" in capsys.readouterr().out
    assert called == []


def test_chat_dispatches_to_core_chat(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(qlh.subprocess, "run", fake_run)
    assert qlh.main(["chat", "--fixture", "fixtures/chat.sse"]) == 0
    assert os.path.normpath(seen["command"][1]).endswith(os.path.normpath("src/tui_chat.py"))
    assert seen["command"][2:] == ["--fixture", "fixtures/chat.sse"]
    assert seen["kwargs"]["cwd"] == str(qlh.ROOT)


def test_models_is_a_plain_tui_command(monkeypatch):
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return type("Result", (), {"returncode": 3})()

    monkeypatch.setattr(qlh.subprocess, "run", fake_run)
    assert qlh.main(["models"]) == 3
    assert os.path.normpath(seen[0][1]).endswith(os.path.normpath("src/tui_admin.py"))
    assert seen[0][2] == "models"
