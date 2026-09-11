import hashlib
import json

import pytest

from harness_workbench.tools import (
    DOWNLOAD_MANIFEST_SCHEMA,
    DownloadError,
    DownloadFilePin,
    DownloadManifest,
    DownloadRunner,
    MemoryDownloadTransport,
)
from harness_workbench.tools.download_runner import DownloadResponse, main


def _manifest(payload: bytes, *, name: str = "fixture.gguf", target_root: str = "fixture-model"):
    url = "https://fixture.invalid/models/" + name
    pin = DownloadFilePin(name, url, hashlib.sha256(payload).hexdigest(), len(payload))
    return DownloadManifest("fixture-model", "a" * 40, target_root, (pin,), source_ref="fixture")


def test_manifest_schema_requires_nonfloating_revision_and_safe_paths():
    with pytest.raises(DownloadError, match="non-floating"):
        DownloadManifest("model", "main", "target", (DownloadFilePin("a.gguf", "https://x.invalid/a", "0" * 64, 0),))
    with pytest.raises(DownloadError, match="inside"):
        DownloadFilePin("../a.gguf", "https://x.invalid/a", "0" * 64, 0)


def test_manifest_loader_builds_urls_from_https_base(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": DOWNLOAD_MANIFEST_SCHEMA,
        "model_id": "model",
        "revision": "b" * 40,
        "target_root": "model",
        "source_base_url": "https://fixture.invalid/repo",
        "files": [{"path": "weights.gguf", "sha256": "1" * 64, "size_bytes": 4}],
    }), encoding="utf-8")

    from harness_workbench.tools.download_runner import load_download_manifest

    manifest = load_download_manifest(path)
    assert manifest.files[0].url == "https://fixture.invalid/repo/weights.gguf"
    assert manifest.pinned_sha_revision is True


def test_plan_is_offline_and_does_not_create_target(tmp_path):
    payload = b"GGUF offline plan"
    manifest = _manifest(payload)

    report = DownloadRunner(tmp_path / "models").run(manifest, dry_run=True)

    assert report.valid is True
    assert report.status == "planned"
    assert report.runner_kind == "plan"
    assert report.network_used is False
    assert not (tmp_path / "models").exists()


def test_download_writes_sidecar_atomically_and_runs_health(tmp_path):
    payload = b"GGUF fixture bytes"
    manifest = _manifest(payload)
    transport = MemoryDownloadTransport({manifest.files[0].url: payload})
    root = tmp_path / "models"

    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    target = root / manifest.target_root / manifest.files[0].path
    sidecar = target.with_name(target.name + ".sha256")
    assert report.valid is True
    assert report.status == "ready"
    assert report.network_used is False
    assert report.health_summary["valid"] is True
    assert target.read_bytes() == payload
    assert manifest.files[0].sha256 in sidecar.read_text(encoding="utf-8")
    assert not target.with_name(target.name + ".part").exists()


def test_resume_uses_existing_part_and_range_header(tmp_path):
    payload = b"0123456789abcdef"
    manifest = _manifest(payload)
    root = tmp_path / "models"
    target = root / manifest.target_root / manifest.files[0].path
    target.parent.mkdir(parents=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(payload[:6])
    part.with_name(part.name + ".json").write_text(json.dumps({
        "schema": "qlh.harness.download_runner.v1",
        "manifest_id": manifest.manifest_id,
        "path": manifest.files[0].path,
        "url": manifest.files[0].url,
        "expected_sha256": manifest.files[0].sha256,
        "expected_size": len(payload),
        "downloaded_bytes": 6,
    }), encoding="utf-8")
    transport = MemoryDownloadTransport({manifest.files[0].url: payload})

    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    assert report.valid is True
    assert report.files[0].resumed is True
    assert any("Range" in headers for _, headers in transport.calls)
    assert target.read_bytes() == payload


def test_retryable_transport_failure_is_retried(tmp_path):
    payload = b"retry me"
    manifest = _manifest(payload)
    transport = MemoryDownloadTransport({manifest.files[0].url: payload}, failures={manifest.files[0].url: ["500"]})

    report = DownloadRunner(tmp_path / "models", transport=transport, max_attempts=3).run(manifest, allow_transport=True)

    assert report.valid is True
    assert report.files[0].attempts == 2
    assert len(transport.calls) == 2


def test_large_payload_uses_multiple_bounded_ranges(tmp_path):
    payload = b"x" * 2500
    manifest = _manifest(payload)
    transport = MemoryDownloadTransport({manifest.files[0].url: payload})

    report = DownloadRunner(tmp_path / "models", transport=transport, chunk_bytes=1024, max_response_bytes=4096).run(manifest, allow_transport=True)

    assert report.valid is True
    assert report.files[0].attempts == 3
    assert len(transport.calls) == 3
    assert all("Range" in headers for _, headers in transport.calls)


def test_existing_matching_target_is_idempotent(tmp_path):
    payload = b"already present"
    manifest = _manifest(payload)
    root = tmp_path / "models"
    target = root / manifest.target_root / manifest.files[0].path
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    transport = MemoryDownloadTransport({manifest.files[0].url: b"must not be fetched"})

    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    assert report.valid is True
    assert report.status == "already_ready"
    assert report.files[0].status == "already_ready"
    assert transport.calls == []


def test_hash_mismatch_does_not_publish_target(tmp_path):
    payload = b"wr0ng!!!"
    manifest = _manifest(b"expected")
    transport = MemoryDownloadTransport({manifest.files[0].url: payload})
    root = tmp_path / "models"

    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    target = root / manifest.target_root / manifest.files[0].path
    assert report.valid is False
    assert report.status == "failed"
    assert report.files[0].error_code == "sha256_mismatch"
    assert not target.exists()
    assert target.with_name(target.name + ".part").exists()


def test_range_ignored_by_server_restarts_from_zero(tmp_path):
    payload = b"server ignores range"
    manifest = _manifest(payload)
    root = tmp_path / "models"
    target = root / manifest.target_root / manifest.files[0].path
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(payload[:4])
    target.with_name(target.name + ".part.json").write_text(json.dumps({
        "manifest_id": manifest.manifest_id,
        "path": manifest.files[0].path,
        "url": manifest.files[0].url,
        "expected_sha256": manifest.files[0].sha256,
        "expected_size": len(payload),
    }), encoding="utf-8")

    class IgnoreRangeTransport(MemoryDownloadTransport):
        def get(self, url, *, headers, timeout_seconds, max_bytes):
            if "Range" in headers:
                self.calls.append((url, dict(headers)))
                return DownloadResponse(200, {"Content-Length": str(len(payload))}, payload)
            return super().get(url, headers=headers, timeout_seconds=timeout_seconds, max_bytes=max_bytes)

    transport = IgnoreRangeTransport({manifest.files[0].url: payload})
    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    assert report.valid is True
    assert report.files[0].resumed is False
    assert target.read_bytes() == payload


def test_network_disabled_blocks_without_staging(tmp_path):
    payload = b"network blocked"
    manifest = _manifest(payload)
    root = tmp_path / "models"

    report = DownloadRunner(root).run(manifest)

    assert report.valid is False
    assert report.status == "blocked"
    assert report.files[0].error_code == "network_disabled"
    assert not root.exists()


def test_existing_mismatched_target_is_not_overwritten(tmp_path):
    expected = b"expected"
    manifest = _manifest(expected)
    root = tmp_path / "models"
    target = root / manifest.target_root / manifest.files[0].path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old user file")
    transport = MemoryDownloadTransport({manifest.files[0].url: expected})

    report = DownloadRunner(root, transport=transport).run(manifest, allow_transport=True)

    assert report.valid is False
    assert report.files[0].error_code == "target_hash_mismatch"
    assert target.read_bytes() == b"old user file"
    assert transport.calls == []


def test_cli_defaults_to_plan_and_writes_reports(tmp_path):
    payload = b"cli payload"
    manifest = _manifest(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    result = main(["--manifest", str(manifest_path), "--root", str(tmp_path / "models"), "--json", str(json_path), "--markdown", str(markdown_path)])

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["status"] == "planned"
    assert report["network_used"] is False
    assert markdown_path.read_text(encoding="utf-8").startswith("# TOOL-DL-RUNNER-01 download report")
