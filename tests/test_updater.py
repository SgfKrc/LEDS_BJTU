import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import update_core
import updater


def _asset(name: str, payload: bytes = b"payload") -> dict:
    return {
        "name": name,
        "url": name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "platform": "windows",
        "variant": "cpu",
        "arch": "x86_64",
    }


def test_project_version_supports_four_numeric_parts_and_prerelease():
    assert update_core.version_key("v0.1.8") < update_core.version_key("0.1.8.1")
    assert update_core.version_key("0.1.8.1-rc.1") < update_core.version_key("0.1.8.1")
    with pytest.raises(update_core.ManifestError):
        update_core.version_key("0.1")


def test_manifest_signature_presence_never_means_verified():
    manifest = update_core.UpdateManifest.from_mapping({
        "schema_version": 1,
        "tag": "0.1.8.1",
        "signature": "untrusted text",
        "assets": [_asset("setup.exe")],
    })
    assert manifest.signature_present is True
    assert manifest.signature_verified is False


def test_fetch_latest_uses_highest_version_and_keeps_failed_source():
    def fetcher(url, timeout):
        assert timeout == 1
        if "failed" in url:
            raise update_core.UpdateError("offline")
        tag = "0.1.8.1" if "new" in url else "0.1.8"
        return update_core.UpdateManifest.from_mapping(
            {"schema_version": 1, "tag": tag, "assets": [_asset(f"{tag}.exe")]},
            source_url=url,
        )

    manifest, failures = update_core.fetch_latest(
        ["https://old", "https://failed", "https://new"],
        timeout=1, fetcher=fetcher,
    )
    assert manifest.tag == "0.1.8.1"
    assert failures == ("https://failed: offline",)


def test_select_asset_prefers_exact_arch_over_any():
    common = _asset("any.exe")
    common["arch"] = "any"
    exact = _asset("exact.exe")
    manifest = update_core.UpdateManifest.from_mapping({
        "schema_version": 1, "tag": "0.1.8.1", "assets": [common, exact],
    })
    selected = update_core.select_asset(
        manifest, platform="windows", variant="cpu", arch="amd64",
    )
    assert selected.name == "exact.exe"


def test_download_is_atomic_and_reuses_verified_file(tmp_path):
    payload = b"verified installer"
    asset = update_core.UpdateAsset.from_mapping(_asset("setup.exe", payload))
    calls = 0

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def opener(_url, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 2
        return Response(payload)

    first = update_core.download_asset(asset, tmp_path, timeout=2, opener=opener)
    second = update_core.download_asset(asset, tmp_path, timeout=2, opener=opener)
    assert first == second
    assert first.read_bytes() == payload
    assert calls == 1
    assert not (tmp_path / "setup.exe.part").exists()


def test_manifest_rejects_non_object_assets_and_unbounded_size():
    with pytest.raises(update_core.ManifestError, match="must be an object"):
        update_core.UpdateManifest.from_mapping({
            "schema_version": 1, "tag": "0.1.8.1", "assets": ["bad"],
        })
    oversized = _asset("oversized.exe")
    oversized["size"] = 33 * 1024 * 1024 * 1024
    with pytest.raises(update_core.ManifestError, match="out of range"):
        update_core.UpdateAsset.from_mapping(oversized)


def test_configured_sources_keeps_multiple_persisted_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("QLH_LAUNCHER_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("QLH_UPDATE_SOURCE", raising=False)
    update_core.save_json_state(
        tmp_path / "launcher.json",
        {"update_source": ["https://a.example/latest.json", "https://b.example/latest.json"]},
    )
    assert updater.configured_sources() == [
        "https://a.example/latest.json", "https://b.example/latest.json",
    ]


def test_detect_current_version_reads_source_tree_without_importing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("QLH_CURRENT_VERSION", raising=False)
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text('__version__ = "0.1.8.1"\n', encoding="utf-8")
    assert updater.detect_current_version(tmp_path, fallback="0.0.0") == "0.1.8.1"
