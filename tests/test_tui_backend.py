"""进程内 TUI 后端监督器测试。"""

import sys
import logging
import threading
import types

import pytest

sys.path.insert(0, "src")

import tui_backend


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_existing_backend_is_reused(monkeypatch):
    monkeypatch.setattr(tui_backend.urllib.request, "urlopen",
                        lambda *args, **kwargs: _Response())
    supervisor = tui_backend.BackendSupervisor(startup_timeout=1)

    assert supervisor.ensure_ready() is False
    assert supervisor.started_here is False


def test_remote_backend_is_never_started(monkeypatch):
    def refused(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(tui_backend.urllib.request, "urlopen", refused)
    supervisor = tui_backend.BackendSupervisor(host="100.100.52.106")

    with pytest.raises(tui_backend.BackendStartupError, match="远程后端"):
        supervisor.ensure_ready()


def test_missing_local_backend_runs_uvicorn_in_daemon_thread(monkeypatch):
    probes = iter([False, True])
    monkeypatch.setattr(tui_backend.BackendSupervisor, "probe",
                        lambda self: next(probes))
    servers = []

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False
            self.config = config
            self.ran = threading.Event()
            servers.append(self)

        def run(self):
            self.ran.set()

    fake_api = types.SimpleNamespace(
        app=object(),
        register_uvicorn_server=lambda server: None,
    )
    fake_uvicorn = types.SimpleNamespace(
        Config=lambda *args, **kwargs: (args, kwargs),
        Server=FakeServer,
    )
    monkeypatch.setitem(sys.modules, "api_server", fake_api)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    supervisor = tui_backend.BackendSupervisor(startup_timeout=1)
    assert supervisor.ensure_ready() is True
    assert supervisor.started_here is True
    assert supervisor.thread is not None
    assert supervisor.thread.daemon is True
    assert servers and servers[0].ran.is_set()
    assert servers[0].config[1]["log_level"] == "warning"
    assert servers[0].config[1]["access_log"] is False
    supervisor.stop()


def test_backend_console_handlers_are_silenced_but_file_handler_survives(
    monkeypatch, tmp_path,
):
    console = logging.StreamHandler()
    file_handler = logging.FileHandler(tmp_path / "backend.log", encoding="utf-8")
    root = types.SimpleNamespace(handlers=[console, file_handler])
    named = types.SimpleNamespace(handlers=[console])

    def fake_get_logger(name=None):
        return root if name is None else named

    fake_logging = types.SimpleNamespace(
        CRITICAL=logging.CRITICAL,
        StreamHandler=logging.StreamHandler,
        FileHandler=logging.FileHandler,
        getLogger=fake_get_logger,
    )
    monkeypatch.setattr(tui_backend, "logging", fake_logging)
    original_console_level = console.level
    tui_backend.BackendSupervisor._silence_console_logging()
    try:
        assert console.level > logging.CRITICAL
        assert file_handler.level == logging.NOTSET
    finally:
        console.setLevel(original_console_level)
        file_handler.close()
