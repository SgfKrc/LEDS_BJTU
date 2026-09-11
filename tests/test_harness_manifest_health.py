import hashlib
import json
from pathlib import Path

from harness_workbench.tools import scan_manifest_health


def test_manifest_health_verifies_sidecar_and_respects_repo_gitignore(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "models"
    models.mkdir(parents=True)
    payload = b"small fixture artifact"
    artifact = models / "fixture.gguf"
    artifact.write_bytes(payload)
    artifact.with_name(artifact.name + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "  fixture.gguf\n", encoding="utf-8"
    )
    (repo / ".gitignore").write_text("models/*.gguf\n", encoding="utf-8")

    report = scan_manifest_health(models, gitignore=repo / ".gitignore", verify_hash=True)

    assert report.valid is True
    assert report.read_only is True
    assert report.network_used is False
    assert report.weights_loaded is False
    assert report.ignored_file_count == 1
    assert report.unignored_model_paths == ()
    assert report.artifacts[0]["ignored"] is True
    assert report.artifacts[0]["sha256_status"] == "verified"
    assert report.digest == scan_manifest_health(models, gitignore=repo / ".gitignore", verify_hash=True).digest


def test_manifest_health_checks_manifest_entries_without_loading_weights(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "weights.bin"
    artifact.write_bytes(b"abc")
    manifest = models / "weights.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fixture.v1",
                "files": [
                    {
                        "path": artifact.name,
                        "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = scan_manifest_health(models, verify_hash=True)

    assert report.valid is True
    assert report.manifests[0]["valid"] is True
    assert report.manifests[0]["entries_checked"] == 1
    assert report.manifests[0]["hashes_checked"] == 1
    assert report.weights_loaded is False


def test_manifest_health_rejects_unsafe_manifest_paths(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "bad.manifest.json").write_text(
        json.dumps({"files": [{"path": "../outside.bin"}]}), encoding="utf-8"
    )

    report = scan_manifest_health(models)

    assert report.valid is False
    assert "bad.manifest.json:files[0]:unsafe_path" in report.errors


def test_manifest_health_default_mode_only_reads_declarations(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "fixture.gguf"
    artifact.write_bytes(b"fixture")
    artifact.with_name(artifact.name + ".sha256").write_text("0" * 64, encoding="utf-8")
    (models / "model.sha256.meta.json").write_text(json.dumps([{"file": artifact.name}]), encoding="utf-8")

    report = scan_manifest_health(models)

    assert report.verify_hash is False
    assert report.artifacts[0]["sha256_status"] == "declared"
    assert report.artifacts[0]["expected_sha256"] == "0" * 64
    assert report.manifests == ()
    assert report.network_used is False
    assert report.weights_loaded is False


def test_manifest_health_fails_explicit_hash_mismatch(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "fixture.gguf"
    artifact.write_bytes(b"fixture")
    artifact.with_name(artifact.name + ".sha256").write_text("0" * 64, encoding="utf-8")

    report = scan_manifest_health(models, verify_hash=True)

    assert report.valid is False
    assert report.artifacts[0]["sha256_status"] == "mismatch"
    assert "mismatch:fixture.gguf" in report.errors


def test_manifest_health_missing_root_is_structured(tmp_path):
    report = scan_manifest_health(tmp_path / "missing-models")

    assert report.valid is False
    assert report.errors == ("model_root_missing",)
    assert report.as_dict()["schema"] == "qlh.harness.manifest_health.v1"
