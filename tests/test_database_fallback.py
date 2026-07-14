"""Database fallback must stay non-blocking after the first failure."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_failed_database_connection_is_cooled_down(monkeypatch):
    import db
    import scheduler

    calls = []

    def fail_pool():
        calls.append(time.monotonic())
        raise TimeoutError("database unavailable")

    monkeypatch.setattr(db, "get_pool", fail_pool)
    monkeypatch.setattr(scheduler, "_db", None)
    monkeypatch.setattr(scheduler, "_db_available", False)
    monkeypatch.setattr(scheduler, "_db_attempted", False)
    monkeypatch.setattr(scheduler, "_db_retry_after", 0.0)
    monkeypatch.setattr(scheduler, "_db_last_error", "")

    started = time.monotonic()
    assert scheduler._get_db() is None
    assert scheduler._get_db() is None
    elapsed = time.monotonic() - started

    assert len(calls) == 1
    assert elapsed < 0.5
    status = scheduler.get_database_status()
    assert status["attempted"] is True
    assert status["available"] is False
    assert status["retry_in_seconds"] > 0
    assert "database unavailable" in status["last_error"]


def test_unconfigured_install_uses_local_store_without_retry(monkeypatch):
    import db
    import scheduler

    calls = []
    monkeypatch.setattr(db, "DB_ENABLED", False)
    monkeypatch.setattr(db, "get_pool", lambda: calls.append(True))
    monkeypatch.setattr(scheduler, "_db", None)
    monkeypatch.setattr(scheduler, "_db_available", False)
    monkeypatch.setattr(scheduler, "_db_attempted", False)
    monkeypatch.setattr(scheduler, "_db_disabled", False)
    monkeypatch.setattr(scheduler, "_db_retry_after", 0.0)
    monkeypatch.setattr(scheduler, "_db_last_error", "")

    assert scheduler._get_db() is None
    assert scheduler._get_db(force_retry=True) is None

    assert calls == []
    status = scheduler.get_database_status()
    assert status["configured"] is False
    assert status["retry_in_seconds"] == 0
    assert "本地文件" in status["last_error"]
