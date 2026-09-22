"""Static gates for the retired remote PostgreSQL runtime."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _scheduler_sources() -> list[Path]:
    sources = [ROOT / "src" / "scheduler.py"]
    sources.extend(sorted((ROOT / "src").glob("scheduler_*.py")))
    return sources


def test_python_runtime_has_no_postgresql_entrypoint():
    assert not (ROOT / "src" / "db.py").exists()
    assert not (ROOT / "scripts" / "setup_test_db.py").exists()
    assert not (ROOT / "scripts" / "verify_fixes.py").exists()

    forbidden = {
        "psycopg": [],
        "QLH_DB_": [],
    }
    import_pattern = re.compile(r"(?m)^\s*(?:from\s+db\s+import|import\s+db\b)")
    imports = []
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        for token in forbidden:
            if token in source:
                forbidden[token].append(relative)
        if import_pattern.search(source):
            imports.append(relative)

    assert forbidden == {"psycopg": [], "QLH_DB_": []}
    assert imports == []


def test_scheduler_and_model_host_expose_no_database_compatibility_flags():
    scheduler_sources = _scheduler_sources()
    assert scheduler_sources
    model_host_source = (ROOT / "src" / "model_host.py").read_text(encoding="utf-8")
    for token in (
        "_get_db",
        "_db_available",
        "_db_disabled",
        "_register_master_in_db",
        "_start_master_db_heartbeat",
        "_start_database_reconnect_monitor",
    ):
        assert all(token not in path.read_text(encoding="utf-8") for path in scheduler_sources)
        assert token not in model_host_source


def test_retired_control_runtime_is_not_present():
    """The single-node QLH runtime no longer ships the retired control service."""
    assert not (ROOT / "control").exists()
    assert not (ROOT / "gateway").exists()
