import json
import sys
import zipfile
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

from launcher_slots import LauncherSlotError, LauncherSlotStore


def _bundle(root: Path, name: str = "QLH-Launcher.exe", marker: str = "ok") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(b"launcher payload")
    (root / "health.ok").write_text(marker, encoding="utf-8")
    return root


def test_launcher_slots_activate_rollback_and_recover(tmp_path):
    store = LauncherSlotStore(tmp_path / "slots")
    first = store.stage_directory(_bundle(tmp_path / "first"), "0.1.8.1")
    active_first = store.activate(first.version, health_check=lambda path: (path / "health.ok").is_file())
    assert active_first.slot == "a"

    second = store.stage_directory(_bundle(tmp_path / "second"), "0.1.9")
    active_second = store.activate(second.version, health_check=lambda path: (path / "health.ok").is_file())
    assert active_second.slot == "b"
    assert store.previous().version == "0.1.8.1"

    rolled = store.rollback(health_check=lambda path: (path / "health.ok").is_file())
    assert rolled.version == "0.1.8.1"
    store.current_file.write_text("not json", encoding="utf-8")
    recovered = store.recover()
    assert recovered is not None
    assert recovered.version == "0.1.9"


def test_launcher_activation_health_failure_preserves_pointer(tmp_path):
    store = LauncherSlotStore(tmp_path / "slots")
    first = store.stage_directory(_bundle(tmp_path / "first"), "0.1.8.1")
    store.activate(first.version)
    second = store.stage_directory(_bundle(tmp_path / "second"), "0.1.9")
    with pytest.raises(LauncherSlotError, match="health"):
        store.activate(second.version, health_check=lambda _path: False)
    assert store.current().version == "0.1.8.1"


def test_launcher_archive_rejects_traversal_and_supports_staging(tmp_path):
    archive = tmp_path / "launcher.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("QLH-Launcher.exe", b"launcher payload")
        bundle.writestr("health.ok", "ok")
    store = LauncherSlotStore(tmp_path / "slots")
    staged = store.stage_archive(archive, "0.1.9")
    assert staged.entrypoint == "QLH-Launcher.exe"
    assert staged.slot == "a"

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(LauncherSlotError, match="escapes"):
        store.stage_archive(unsafe, "0.2.0")
    assert not (tmp_path / "escape.txt").exists()


def test_launcher_diagnostics_redacts_secrets(tmp_path, monkeypatch):
    state = tmp_path / "slots"
    store = LauncherSlotStore(state)
    diagnostic_state = tmp_path / "diagnostic-state"
    monkeypatch.setenv("QLH_LAUNCHER_STATE_DIR", str(diagnostic_state))
    (diagnostic_state / "launcher.json").parent.mkdir(parents=True)
    (diagnostic_state / "launcher.json").write_text(
        json.dumps({"token": "do-not-export", "node": "n1"}), encoding="utf-8",
    )
    output = store.diagnostics(tmp_path / "diagnostics.zip")
    with zipfile.ZipFile(output) as bundle:
        text = bundle.read("state/launcher.json").decode("utf-8")
    assert "do-not-export" not in text
    assert "<redacted>" in text


def test_launcher_diagnostics_keeps_diagnosis_json_in_its_own_redacted_entry(tmp_path, monkeypatch):
    store = LauncherSlotStore(tmp_path / "slots")
    state = tmp_path / "state"
    monkeypatch.setenv("QLH_LAUNCHER_STATE_DIR", str(state))
    report = state / "diagnostics" / "qlh-diagnose-test.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"token": "do-not-export", "issue": "disk"}), encoding="utf-8")

    output = store.diagnostics(tmp_path / "diagnostics.zip", extra_paths=[report])
    with zipfile.ZipFile(output) as bundle:
        text = bundle.read("diagnosis/qlh-diagnose-test.json").decode("utf-8")
    assert "do-not-export" not in text
    assert "<redacted>" in text
