import json
import os
import sys
from pathlib import Path

import pytest

PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import data_retention as retention  # noqa: E402
from data_retention import (  # noqa: E402
    DataRetentionError,
    RETENTION_JOURNAL,
    RETENTION_MARKER,
    auto_reassociate_user_data,
    reassociate_user_data,
    retain_user_data,
    retention_status,
)


USER_DIRS = ("models", "chat_history", "logs", "config", "local_docs")


def _make_install(tmp_path: Path) -> tuple[Path, Path]:
    install = tmp_path / "install"
    data = tmp_path / "user-data"
    install.mkdir()
    for name in USER_DIRS:
        (install / name).mkdir()
    (install / "models" / "model.gguf").write_bytes(b"model")
    (install / "chat_history" / "session.json").write_text('{"ok":true}\n', encoding="utf-8")
    (install / "config" / "settings.json").write_text('{"port":8000}\n', encoding="utf-8")
    return install, data


def test_retention_moves_only_declared_user_directories(tmp_path):
    install, data = _make_install(tmp_path)
    program = install / "QLH-Edge-Inference.exe"
    program.write_bytes(b"program")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")

    report = retain_user_data(install, data, confirm=True)

    assert report["action"] == "retained"
    assert not (install / "models").exists()
    assert (data / "models" / "model.gguf").read_bytes() == b"model"
    assert program.read_bytes() == b"program"
    assert outside.read_bytes() == b"untouched"
    marker = json.loads((data / RETENTION_MARKER).read_text(encoding="utf-8"))
    assert marker["state"] == "retained"
    assert set(report["directories"]) == set(USER_DIRS)


def test_reassociation_restores_data_and_is_idempotent(tmp_path):
    install, data = _make_install(tmp_path)
    retain_user_data(install, data, confirm=True)
    install.mkdir(exist_ok=True)
    for name in USER_DIRS:
        (install / name).mkdir()

    report = auto_reassociate_user_data(install, data)

    assert report["action"] == "reassociated"
    assert (install / "models" / "model.gguf").read_bytes() == b"model"
    assert not (data / "models").exists()
    assert reassociate_user_data(install, data, confirm=True)["action"] == "already-reassociated"
    assert retention_status(install, data)["state"] == "reassociated"


def test_retention_requires_confirmation_and_refuses_merge(tmp_path):
    install, data = _make_install(tmp_path)
    with pytest.raises(DataRetentionError, match="explicit confirmation"):
        retain_user_data(install, data)

    data.mkdir()
    (data / "models").mkdir()
    with pytest.raises(DataRetentionError, match="already exists"):
        retain_user_data(install, data, confirm=True)
    assert (install / "models" / "model.gguf").exists()


def test_reassociation_refuses_nonempty_fresh_install_directory(tmp_path):
    install, data = _make_install(tmp_path)
    retain_user_data(install, data, confirm=True)
    install.mkdir(exist_ok=True)
    (install / "models").mkdir()
    (install / "models" / "new-model").write_bytes(b"new")
    for name in USER_DIRS[1:]:
        (install / name).mkdir()

    with pytest.raises(DataRetentionError, match="non-empty"):
        reassociate_user_data(install, data, confirm=True)
    assert (data / "models" / "model.gguf").exists()
    assert (install / "models" / "new-model").exists()


def test_reassociation_without_a_retention_marker_is_a_safe_noop(tmp_path):
    install, data = _make_install(tmp_path)
    data.mkdir()

    report = reassociate_user_data(install, data, confirm=True)

    assert report["action"] == "none"
    assert (install / "models" / "model.gguf").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unavailable")
def test_retention_rejects_linked_user_directory(tmp_path):
    install, data = _make_install(tmp_path)
    target = tmp_path / "real-models"
    target.mkdir()
    (install / "models" / "model.gguf").unlink()
    (install / "models").rmdir()
    try:
        (install / "models").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")

    with pytest.raises(DataRetentionError, match="linked"):
        retain_user_data(install, data, confirm=True)


def _force_cross_volume(monkeypatch):
    monkeypatch.setattr(retention, "_same_filesystem", lambda first, second: False)


def test_cross_volume_retention_and_reassociation_round_trip(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    nested = install / "local_docs" / "guides" / "zh"
    nested.mkdir(parents=True)
    (nested / "start.md").write_text("跨卷保留\n", encoding="utf-8")
    _force_cross_volume(monkeypatch)

    retained = retain_user_data(install, data, confirm=True)

    assert retained["transfer_mode"] == "verified-copy"
    assert retained["recovered"] is False
    assert not (data / RETENTION_JOURNAL).exists()
    assert not (install / "models").exists()
    assert (data / "models" / "model.gguf").read_bytes() == b"model"
    assert (data / "local_docs" / "guides" / "zh" / "start.md").read_text(encoding="utf-8") == "跨卷保留\n"

    for name in USER_DIRS:
        (install / name).mkdir()
    reassociated = reassociate_user_data(install, data, confirm=True)

    assert reassociated["transfer_mode"] == "verified-copy"
    assert not (data / RETENTION_JOURNAL).exists()
    assert (install / "models" / "model.gguf").read_bytes() == b"model"
    assert not (data / "models").exists()
    marker = json.loads((data / RETENTION_MARKER).read_text(encoding="utf-8"))
    assert marker["state"] == "reassociated"


def test_cross_volume_preflight_preserves_source_when_space_is_insufficient(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    _force_cross_volume(monkeypatch)
    monkeypatch.setattr(retention.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 0})())

    with pytest.raises(DataRetentionError, match="insufficient destination space"):
        retain_user_data(install, data, confirm=True)

    assert (install / "models" / "model.gguf").read_bytes() == b"model"
    assert not (data / "models").exists()
    assert not (data / RETENTION_JOURNAL).exists()


def test_cross_volume_deletion_phase_resumes_without_recopy(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    _force_cross_volume(monkeypatch)
    original_delete = retention._delete_committed_sources

    def interrupt_delete(payload, source_root, destination_root):
        raise DataRetentionError("simulated deletion interruption")

    monkeypatch.setattr(retention, "_delete_committed_sources", interrupt_delete)
    with pytest.raises(DataRetentionError, match="simulated deletion interruption"):
        retain_user_data(install, data, confirm=True)

    assert (install / "models" / "model.gguf").exists()
    assert (data / "models" / "model.gguf").exists()
    assert json.loads((data / RETENTION_JOURNAL).read_text(encoding="utf-8"))["phase"] == "deleting_source"
    assert retention_status(install, data)["transaction"]["phase"] == "deleting_source"

    monkeypatch.setattr(retention, "_delete_committed_sources", original_delete)
    recovered = retain_user_data(install, data, confirm=True)

    assert recovered["recovered"] is True
    assert not (install / "models").exists()
    assert (data / "models" / "model.gguf").read_bytes() == b"model"
    assert not (data / RETENTION_JOURNAL).exists()


def test_cross_volume_staging_tamper_fails_before_source_deletion(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    _force_cross_volume(monkeypatch)
    original_commit = retention._commit_staged_directories

    def interrupt_commit(payload, source_root, destination_root):
        raise DataRetentionError("simulated commit interruption")

    monkeypatch.setattr(retention, "_commit_staged_directories", interrupt_commit)
    with pytest.raises(DataRetentionError, match="simulated commit interruption"):
        retain_user_data(install, data, confirm=True)

    journal = json.loads((data / RETENTION_JOURNAL).read_text(encoding="utf-8"))
    staged_model = data / f".qlh-retention-staging-{journal['transaction_id']}" / "models" / "model.gguf"
    staged_model.write_bytes(b"tampered")
    monkeypatch.setattr(retention, "_commit_staged_directories", original_commit)

    with pytest.raises(DataRetentionError, match="inventory does not match"):
        retain_user_data(install, data, confirm=True)
    assert (install / "models" / "model.gguf").read_bytes() == b"model"


def test_cross_volume_journal_root_tamper_is_rejected(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    _force_cross_volume(monkeypatch)
    monkeypatch.setattr(
        retention,
        "_commit_staged_directories",
        lambda payload, source_root, destination_root: (_ for _ in ()).throw(
            DataRetentionError("simulated commit interruption")
        ),
    )
    with pytest.raises(DataRetentionError, match="simulated commit interruption"):
        retain_user_data(install, data, confirm=True)

    journal_path = data / RETENTION_JOURNAL
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["install_root"] = str(tmp_path / "different-install")
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(DataRetentionError, match="roots do not match"):
        retain_user_data(install, data, confirm=True)
    assert (install / "models" / "model.gguf").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink unavailable")
def test_cross_volume_rejects_nested_link_before_copy(tmp_path, monkeypatch):
    install, data = _make_install(tmp_path)
    target = tmp_path / "linked-target"
    target.mkdir()
    try:
        (install / "local_docs" / "linked").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    _force_cross_volume(monkeypatch)

    with pytest.raises(DataRetentionError, match="nested user-data directory is unsafe"):
        retain_user_data(install, data, confirm=True)

    assert (install / "models" / "model.gguf").exists()
    assert not (data / RETENTION_JOURNAL).exists()
