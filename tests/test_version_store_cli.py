import sys
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import updater


def _bundle(path: Path, marker: str) -> Path:
    path.mkdir()
    (path / "health.ok").write_text(marker, encoding="utf-8")
    (path / "payload.txt").write_text(marker, encoding="utf-8")
    return path


def test_version_cli_stage_activate_status_and_rollback(tmp_path, capsys):
    store = tmp_path / "store"
    first = _bundle(tmp_path / "first", "one")
    second = _bundle(tmp_path / "second", "two")

    assert updater.main(["version-stage", "--version-store", str(store), "--bundle", str(first), "--version", "0.1.8.1", "--variant", "cpu", "--json"]) == 0
    assert updater.main(["version-activate", "--version-store", str(store), "--version", "0.1.8.1", "--variant", "cpu", "--json"]) == 0
    assert updater.main(["version-stage", "--version-store", str(store), "--bundle", str(second), "--version", "0.1.9", "--variant", "cpu", "--json"]) == 0
    assert updater.main(["version-activate", "--version-store", str(store), "--version", "0.1.9", "--variant", "cpu", "--json"]) == 0
    assert updater.main(["version-rollback", "--version-store", str(store), "--json"]) == 0

    output = capsys.readouterr().out
    assert '"rolled_back"' in output
    assert '0.1.8.1' in output


def test_version_cli_requires_health_marker(tmp_path, capsys):
    store = tmp_path / "store"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "payload.txt").write_text("bad", encoding="utf-8")

    assert updater.main(["version-stage", "--version-store", str(store), "--bundle", str(bad), "--version", "0.1.9", "--variant", "cpu"]) == 0
    assert updater.main(["version-activate", "--version-store", str(store), "--version", "0.1.9", "--variant", "cpu"]) == 2
    assert updater.VersionStore(store).current() is None
    assert "health check failed" in capsys.readouterr().out

