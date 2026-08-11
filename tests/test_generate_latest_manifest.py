"""Release-to-latest.json generation tests without GitHub network access."""

import json
import sys
from pathlib import Path

import pytest

PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import generate_latest_manifest as generator
import signing


def _keyring(tmp_path):
    keys = tmp_path / "keys"
    signing.generate_keypair(keys, key_id="root", role="root")
    signing.generate_keypair(keys, key_id="release-test", role="release")
    signing.authorize_new_key(
        keys / "release-test.pub.json",
        issuer_private_path=keys / "root.key",
        issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes((keys / "root.pub.json").read_bytes())
    (pubkeys / "release-test.pub.json").write_bytes(
        (keys / "release-test.pub.json").read_bytes()
    )
    return keys / "release-test.key", pubkeys


def _release(*assets, tag="0.1.8.2", draft=False):
    return {"tag_name": tag, "draft": draft, "assets": list(assets)}


def _asset(name, *, digest="a" * 64, size=123, url=None):
    return {
        "name": name,
        "digest": f"sha256:{digest}",
        "size": size,
        "browser_download_url": url or f"https://github.com/example/repo/releases/download/0.1.8.2/{name}",
    }


def test_generates_signs_verifies_and_writes_manifest(tmp_path):
    release_key, pubkeys = _keyring(tmp_path)
    release = _release(
        _asset("QLH-Launcher-v0.1.8.2.zip", digest="b" * 64, size=27),
        _asset("QLH-Edge-Inference-Setup-v0.1.8.2.exe", digest="c" * 64, size=91),
        _asset("notes.txt"),
    )

    manifest = generator.build_manifest_from_release(
        release,
        tag="v0.1.8.2",
        generated_at="2026-08-12T00:00:00+00:00",
    )
    assert manifest["tag"] == "0.1.8.2"
    assert [asset["kind"] for asset in manifest["assets"]] == ["installer", "launcher"]
    signed = generator.sign_and_verify_manifest(
        manifest, private_key_path=release_key, trusted_keys_dir=pubkeys,
    )
    output = tmp_path / "latest.json"
    generator.write_manifest(output, signed)

    written = json.loads(output.read_text(encoding="utf-8"))
    verified, reason = signing.verify_manifest_signature(
        written, trusted_keys_dir=pubkeys,
    )
    assert verified is True, reason
    assert written["assets"][0]["sha256"] == "c" * 64


def test_rejects_asset_without_github_sha256_digest():
    release = _release(_asset("QLH-Launcher-v0.1.8.2.zip"))
    release["assets"][0]["digest"] = ""

    with pytest.raises(generator.ManifestGenerationError, match="SHA-256"):
        generator.build_manifest_from_release(release, tag="0.1.8.2")


def test_rejects_duplicate_update_targets():
    release = _release(
        _asset("QLH-Launcher-v0.1.8.2.zip"),
        _asset("QLH-Launcher-v0.1.8.2-backup.zip", digest="b" * 64),
    )

    with pytest.raises(generator.ManifestGenerationError, match="same update target"):
        generator.build_manifest_from_release(release, tag="0.1.8.2")


def test_rejects_draft_or_mismatched_release():
    with pytest.raises(generator.ManifestGenerationError, match="draft"):
        generator.build_manifest_from_release(
            _release(_asset("QLH-Launcher-v0.1.8.2.zip"), draft=True),
            tag="0.1.8.2",
        )
    with pytest.raises(generator.ManifestGenerationError, match="does not match"):
        generator.build_manifest_from_release(
            _release(_asset("QLH-Launcher-v0.1.8.2.zip"), tag="0.1.8.3"),
            tag="0.1.8.2",
        )
