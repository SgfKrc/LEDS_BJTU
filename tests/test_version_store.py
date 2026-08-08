import json
import sys
import zipfile
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

from version_store import VersionStore, VersionStoreError


def _bundle(path: Path, marker: str = "ok") -> Path:
    path.mkdir()
    (path / "health.ok").write_text(marker, encoding="utf-8")
    (path / "payload.txt").write_text(marker, encoding="utf-8")
    return path


def test_stage_activate_and_rollback_keeps_previous_payload(tmp_path):
    store = VersionStore(tmp_path / "store")
    first = _bundle(tmp_path / "first", "one")
    second = _bundle(tmp_path / "second", "two")

    store.stage_directory(first, "0.1.8.1", "cpu")
    store.activate("0.1.8.1", "cpu", health_check=lambda path: (path / "health.ok").is_file())
    store.stage_directory(second, "0.1.9", "cpu")
    store.activate("0.1.9", "cpu", health_check=lambda path: (path / "health.ok").is_file())

    assert store.current().version == "0.1.9"
    assert store.previous().version == "0.1.8.1"
    assert store.active_path().joinpath("payload.txt").read_text(encoding="utf-8") == "two"

    restored = store.rollback(health_check=lambda path: (path / "health.ok").is_file())
    assert restored.version == "0.1.8.1"
    assert store.active_path().joinpath("payload.txt").read_text(encoding="utf-8") == "one"


def test_failed_health_check_does_not_change_current(tmp_path):
    store = VersionStore(tmp_path / "store")
    first = _bundle(tmp_path / "first")
    bad = _bundle(tmp_path / "bad")
    store.stage_directory(first, "0.1.8.1", "cpu")
    store.activate("0.1.8.1", "cpu")
    store.stage_directory(bad, "0.1.9", "cpu")

    with pytest.raises(VersionStoreError, match="health check failed"):
        store.activate("0.1.9", "cpu", health_check=lambda _path: False)

    assert store.current().version == "0.1.8.1"
    assert store.previous() is None


def test_recover_restores_previous_when_current_pointer_is_corrupt(tmp_path):
    store = VersionStore(tmp_path / "store")
    first = _bundle(tmp_path / "first")
    second = _bundle(tmp_path / "second")
    store.stage_directory(first, "0.1.8.1", "cpu")
    store.activate("0.1.8.1", "cpu")
    store.stage_directory(second, "0.1.9", "cpu")
    store.activate("0.1.9", "cpu")
    store.current_file.write_text("{broken", encoding="utf-8")

    recovered = store.recover()
    assert recovered.version == "0.1.8.1"
    assert store.current().version == "0.1.8.1"


def test_stage_archive_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")

    with pytest.raises(VersionStoreError, match="escapes"):
        VersionStore(tmp_path / "store").stage_archive(archive, "0.1.9", "cpu")


def test_stage_archive_and_status_are_deterministic(tmp_path):
    source = _bundle(tmp_path / "source", "archive")
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in source.iterdir():
            bundle.write(path, path.name)

    store = VersionStore(tmp_path / "store")
    target = store.stage_archive(archive, "0.1.9", "lite")
    assert target.joinpath("payload.txt").read_text(encoding="utf-8") == "archive"
    status = store.status()
    assert status["current"] is None
    assert status["versions"] == ["0.1.9-lite"]


def test_pointer_validation_rejects_path_escape(tmp_path):
    store = VersionStore(tmp_path / "store")
    store.root.mkdir(parents=True)
    store.current_file.write_text(json.dumps({
        "schema_version": 1,
        "version": "0.1.9",
        "variant": "cpu",
        "directory": "../outside",
    }), encoding="utf-8")
    assert store.current() is None

