"""UP-N2 trusted publishing tests: Ed25519 manifest signatures, key rotation
and the fail-closed matrix (tampered manifest/assets, unknown/forged keys)."""

import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import signing
import update_core


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


@pytest.fixture()
def keyring(tmp_path):
    """root keypair + release keypair authorized by root, plus a pubkeys dir."""
    root_key = tmp_path / "keys" / "root.key"
    root_key.parent.mkdir()
    signing.generate_keypair(root_key.parent, key_id="root", role="root")
    release_dir = tmp_path / "keys"
    signing.generate_keypair(release_dir, key_id="release-20260809", role="release")
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes(
        (release_dir / "root.pub.json").read_bytes()
    )
    signing.authorize_new_key(
        release_dir / "release-20260809.pub.json",
        issuer_private_path=root_key, issuer_key_id="root",
    )
    (pubkeys / "release-20260809.pub.json").write_bytes(
        (release_dir / "release-20260809.pub.json").read_bytes()
    )
    return {
        "root_key": root_key,
        "release_key": release_dir / "release-20260809.key",
        "release_pub": release_dir / "release-20260809.pub.json",
        "pubkeys": pubkeys,
        "other_key": release_dir / "other.key",
    }


def _signed_manifest(keyring, **overrides) -> dict:
    mapping = {
        "schema_version": 1,
        "tag": "0.1.8.1",
        "channel": "stable",
        "generated_at": "2026-08-08T00:00:00+00:00",
        "assets": [_asset("setup.exe")],
    }
    mapping.update(overrides)
    return signing.sign_manifest(
        mapping,
        private_key_path=keyring["release_key"],
        key_id=overrides.get("key_id") or "release-20260809",
        signed_at=overrides.get("signed_at"),
    )


# --------------------------------------------------------------------------
# keygen
# --------------------------------------------------------------------------

def test_keygen_writes_private_and_public_files_with_roles(tmp_path):
    key_dir = tmp_path / "keys"
    public_path = signing.generate_keypair(
        key_dir, key_id="release-20260809", role="release",
    )
    assert public_path.name == "release-20260809.pub.json"
    assert (key_dir / "release-20260809.key").is_file()
    public = json.loads(public_path.read_text(encoding="utf-8"))
    assert public["key_id"] == "release-20260809"
    assert public["role"] == "release"
    assert len(base64.b64decode(public["public_key"])) == 32
    # 私钥可重新加载并对应同一公钥
    _, private = signing.load_private_key(key_dir / "release-20260809.key")
    assert signing._public_key_to_file_bytes(private.public_key()) == base64.b64decode(
        public["public_key"]
    )
    if hasattr(Path, "stat") and not sys.platform.startswith("win"):
        assert (key_dir / "release-20260809.key").stat().st_mode & 0o777 == 0o600


def test_keygen_rejects_key_id_with_path_characters(tmp_path):
    with pytest.raises(signing.SigningError, match="key_id"):
        signing.generate_keypair(tmp_path, key_id="a/b")


# --------------------------------------------------------------------------
# sign / verify round trips
# --------------------------------------------------------------------------

def test_signed_manifest_verifies_against_trusted_keys(keyring):
    manifest = _signed_manifest(keyring)
    assert manifest["key_id"] == "release-20260809"
    assert manifest["signature"]
    assert manifest["signed_at"]
    verified, reason = signing.verify_manifest_signature(
        manifest, trusted_keys_dir=keyring["pubkeys"],
    )
    assert verified is True, reason


def test_signature_binds_key_id_and_signed_at(keyring):
    """key_id/signed_at 在签名正文内，篡改任何一项都会失效。"""
    manifest = _signed_manifest(keyring, key_id="release-20260809", signed_at="2026-08-08T00:00:00+00:00")
    for field, value in (
        ("key_id", "release-20260809-evil"),
        ("signed_at", "2026-08-09T00:00:00+00:00"),
    ):
        tampered = dict(manifest)
        tampered[field] = value
        verified, _reason = signing.verify_manifest_signature(
            tampered, trusted_keys_dir=keyring["pubkeys"],
        )
        assert verified is False


def test_tampered_manifest_body_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    tampered = dict(manifest)
    tampered["assets"] = [_asset("setup.exe", b"evil payload")]
    result = signing.verify_manifest_signature_details(
        tampered, trusted_keys_dir=keyring["pubkeys"],
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_INVALID"


def test_tampered_signature_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    tampered = dict(manifest)
    tampered["signature"] = base64.b64encode(b"x" * 64).decode("ascii")
    result = signing.verify_manifest_signature_details(
        tampered, trusted_keys_dir=keyring["pubkeys"],
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_INVALID"


# --------------------------------------------------------------------------
# fail-closed matrix
# --------------------------------------------------------------------------

def test_missing_signature_is_not_verified(keyring):
    manifest = {
        "schema_version": 1, "tag": "0.1.8.1", "assets": [_asset("setup.exe")],
    }
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=keyring["pubkeys"],
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_MISSING"


def test_unknown_key_id_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    manifest["key_id"] = "release-evil"
    manifest["signature"] = base64.b64encode(b"y" * 64).decode("ascii")
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=keyring["pubkeys"],
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_KEY_UNKNOWN"


def test_missing_signed_at_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    del manifest["signed_at"]
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=keyring["pubkeys"],
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_SIGNED_AT_INVALID"


def test_malformed_signed_at_fails_closed(keyring):
    for bad in ("2026-08-08", "not-a-date", "2026-08-08T00:00:00"):
        manifest = _signed_manifest(keyring, signed_at=bad)
        result = signing.verify_manifest_signature_details(
            manifest, trusted_keys_dir=keyring["pubkeys"],
        )
        assert result["error_code"] == "SIGNATURE_SIGNED_AT_INVALID", bad


def test_missing_trusted_keys_dir_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=None,
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_TRUST_STORE_MISSING"


def test_missing_root_key_file_fails_closed(keyring, tmp_path):
    empty = tmp_path / "empty-pubkeys"
    empty.mkdir()
    manifest = _signed_manifest(keyring)
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=empty,
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_ROOT_KEY_MISSING"


# --------------------------------------------------------------------------
# key rotation
# --------------------------------------------------------------------------

def test_rotated_key_authorized_by_previous_release_key_is_trusted(keyring, tmp_path):
    """新发布 key 由旧发布 key 授权（链式轮换），签名可被验证。"""
    keys = tmp_path / "keys2"
    keys.mkdir()
    signing.generate_keypair(keys, key_id="release-20260901", role="release")
    # 旧 key 先由 root 授权并放入信任集
    signing.authorize_new_key(
        keys / "release-20260901.pub.json",
        issuer_private_path=keyring["root_key"], issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys2"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes(
        (keyring["root_key"].parent / "root.pub.json").read_bytes()
    )
    (pubkeys / "release-20260901.pub.json").write_bytes(
        (keys / "release-20260901.pub.json").read_bytes()
    )
    # 新 key（release-20260809）由旧发布 key 授权
    signing.authorize_new_key(
        keyring["release_pub"],
        issuer_private_path=keys / "release-20260901.key", issuer_key_id="release-20260901",
    )
    (pubkeys / "release-20260809.pub.json").write_bytes(
        keyring["release_pub"].read_bytes()
    )
    manifest = _signed_manifest(keyring)
    verified, reason = signing.verify_manifest_signature(
        manifest, trusted_keys_dir=pubkeys,
    )
    assert verified is True, reason


def test_forged_authorization_fails_closed(keyring, tmp_path):
    """用错误的私钥签发授权 → 新 key 不可信。"""
    other = tmp_path / "other"
    other.mkdir()
    signing.generate_keypair(other, key_id="attacker", role="release")
    forged = dict(json.loads(keyring["release_pub"].read_text(encoding="utf-8")))
    forged["authorized_by"] = "root"
    forged["authorization"] = signing.sign_authorization(
        forged, issuer_private_key=signing.load_private_key(other / "attacker.key")[1],
    )
    pubkeys = tmp_path / "pubkeys-forged"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes(
        (keyring["root_key"].parent / "root.pub.json").read_bytes()
    )
    (pubkeys / "release-20260809.pub.json").write_text(
        json.dumps(forged, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    manifest = _signed_manifest(keyring)
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=pubkeys,
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_AUTHORIZATION_INVALID"


def test_unknown_issuer_fails_closed(keyring, tmp_path):
    """authorized_by 指向不受信任的 key → 新 key 不可信。"""
    forged = dict(json.loads(keyring["release_pub"].read_text(encoding="utf-8")))
    forged["authorized_by"] = "release-ghost"
    pubkeys = tmp_path / "pubkeys-ghost"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes(
        (keyring["root_key"].parent / "root.pub.json").read_bytes()
    )
    (pubkeys / "release-20260809.pub.json").write_text(
        json.dumps(forged, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    manifest = _signed_manifest(keyring)
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=pubkeys,
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_ISSUER_UNTRUSTED"


def test_expired_key_fails_closed(keyring, tmp_path):
    expired = dict(json.loads(keyring["release_pub"].read_text(encoding="utf-8")))
    expired["valid_until"] = "2020-01-01T00:00:00+00:00"
    expired["authorized_by"] = "root"
    expired["authorization"] = signing.sign_authorization(
        expired, issuer_private_key=signing.load_private_key(keyring["root_key"])[1],
    )
    pubkeys = tmp_path / "pubkeys-expired"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes(
        (keyring["root_key"].parent / "root.pub.json").read_bytes()
    )
    (pubkeys / "release-20260809.pub.json").write_text(
        json.dumps(expired, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    manifest = _signed_manifest(keyring)
    result = signing.verify_manifest_signature_details(
        manifest, trusted_keys_dir=pubkeys,
    )
    assert result["verified"] is False
    assert result["error_code"] == "SIGNATURE_KEY_EXPIRED"


# --------------------------------------------------------------------------
# update_core / updater integration (fetch gate)
# --------------------------------------------------------------------------

class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, _limit: int) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


def test_fetch_manifest_verifies_signed_manifest_end_to_end(keyring):
    manifest = _signed_manifest(keyring)
    encoded = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    fetched = update_core.fetch_manifest(
        "https://example.invalid/latest.json",
        opener=lambda _url, timeout: _Response(encoded),
        trusted_keys_dir=keyring["pubkeys"],
    )
    assert fetched.signature_present is True
    assert fetched.signature_verified is True
    assert fetched.signature_key_id == "release-20260809"
    assert fetched.signature_error_code == ""
    assert fetched.signature_error == ""


def test_fetch_manifest_tampered_body_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    manifest["assets"] = [_asset("setup.exe", b"evil")]
    encoded = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    fetched = update_core.fetch_manifest(
        "https://example.invalid/latest.json",
        opener=lambda _url, timeout: _Response(encoded),
        trusted_keys_dir=keyring["pubkeys"],
    )
    assert fetched.signature_present is True
    assert fetched.signature_verified is False
    assert fetched.signature_error_code == "SIGNATURE_INVALID"
    assert fetched.signature_error


def test_fetch_manifest_unknown_key_fails_closed(keyring):
    manifest = _signed_manifest(keyring)
    manifest["key_id"] = "release-evil"
    manifest["signature"] = base64.b64encode(b"z" * 64).decode("ascii")
    encoded = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    fetched = update_core.fetch_manifest(
        "https://example.invalid/latest.json",
        opener=lambda _url, timeout: _Response(encoded),
        trusted_keys_dir=keyring["pubkeys"],
    )
    assert fetched.signature_verified is False
    assert fetched.signature_error_code == "SIGNATURE_KEY_UNKNOWN"


def test_fetch_manifest_unsigned_manifest_is_reported_but_usable(keyring):
    mapping = {
        "schema_version": 1, "tag": "0.1.8.1", "assets": [_asset("setup.exe")],
    }
    encoded = json.dumps(mapping, ensure_ascii=False).encode("utf-8")
    fetched = update_core.fetch_manifest(
        "https://example.invalid/latest.json",
        opener=lambda _url, timeout: _Response(encoded),
        trusted_keys_dir=keyring["pubkeys"],
    )
    assert fetched.signature_present is False
    assert fetched.signature_verified is False
    assert fetched.signature_error == ""


def test_downloaded_asset_tampering_fails_closed(tmp_path):
    """资产被篡改（大小或 SHA-256 不符）→ 下载事务失败，不留缓存文件。"""
    asset = update_core.UpdateAsset.from_mapping(_asset("setup.exe", b"good"))
    calls = 0

    def opener(_url, timeout):
        nonlocal calls
        calls += 1
        return _Response(b"evil")

    with pytest.raises(update_core.DownloadError) as download_error:
        update_core.download_asset(asset, tmp_path, timeout=1, opener=opener)
    assert download_error.value.code == "UPDATE_DOWNLOAD_FAILED"
    assert not (tmp_path / "setup.exe").exists()
    assert not (tmp_path / "setup.exe.part").exists()


# --------------------------------------------------------------------------
# updater install gate
# --------------------------------------------------------------------------

def test_install_proceeds_without_allow_unsigned_when_verified(
    keyring, tmp_path, monkeypatch,
):
    import updater
    from dataclasses import replace

    manifest = update_core.UpdateManifest.from_mapping(
        _signed_manifest(keyring), source_url="https://example.invalid/latest.json",
    )
    manifest = replace(manifest, signature_verified=True)
    monkeypatch.setattr(
        updater, "_manifest_for_request", lambda *_a, **_k: (manifest, ()),
    )
    monkeypatch.setattr(
        updater, "detect_profile",
        lambda *_a, **_k: {"platform": "windows", "arch": "x86_64", "variant": "cpu"},
    )
    monkeypatch.setattr(
        updater, "download_asset",
        lambda *_a, **_k: tmp_path / "setup.exe",
    )
    monkeypatch.setattr(updater, "_launch_installer", lambda _path: 0)
    monkeypatch.setattr(updater, "detect_current_version", lambda *a, **k: "0.1.8")
    monkeypatch.setattr(
        updater, "default_state_dir", lambda: tmp_path / "state",
    )
    code = updater.main(["install", "--yes", "--source", "https://example.invalid/latest.json"])
    assert code == 0


def test_install_refused_without_allow_unsigned_when_unverified(
    keyring, tmp_path, monkeypatch,
):
    import updater

    manifest = update_core.UpdateManifest.from_mapping(
        _signed_manifest(keyring), source_url="https://example.invalid/latest.json",
    )
    monkeypatch.setattr(
        updater, "_manifest_for_request", lambda *_a, **_k: (manifest, ()),
    )
    monkeypatch.setattr(
        updater, "detect_profile",
        lambda *_a, **_k: {"platform": "windows", "arch": "x86_64", "variant": "cpu"},
    )
    monkeypatch.setattr(
        updater, "download_asset",
        lambda *_a, **_k: tmp_path / "setup.exe",
    )
    monkeypatch.setattr(updater, "detect_current_version", lambda *a, **k: "0.1.8")
    printed = {}
    monkeypatch.setattr(updater, "_print", lambda value, *, as_json: printed.update(value))
    code = updater.main(["install", "--yes", "--source", "https://example.invalid/latest.json"])
    assert code == 2
    # 有签名但未验证 → 同样拒绝，必须显式 --allow-unsigned
    assert "签名尚未验证" in printed["error"]


def test_install_refuses_unsigned_manifest_even_with_present_signature_field(
    keyring, tmp_path, monkeypatch,
):
    import updater

    manifest = update_core.UpdateManifest.from_mapping(
        {
            "schema_version": 1, "tag": "0.1.8.1", "assets": [_asset("setup.exe")],
        },
        source_url="https://example.invalid/latest.json",
    )
    monkeypatch.setattr(
        updater, "_manifest_for_request", lambda *_a, **_k: (manifest, ()),
    )
    monkeypatch.setattr(
        updater, "detect_profile",
        lambda *_a, **_k: {"platform": "windows", "arch": "x86_64", "variant": "cpu"},
    )
    monkeypatch.setattr(
        updater, "download_asset",
        lambda *_a, **_k: tmp_path / "setup.exe",
    )
    monkeypatch.setattr(updater, "detect_current_version", lambda *a, **k: "0.1.8")
    printed = {}
    monkeypatch.setattr(updater, "_print", lambda value, *, as_json: printed.update(value))
    code = updater.main(["install", "--yes", "--source", "https://example.invalid/latest.json"])
    assert code == 2
    assert "清单没有签名" in printed["error"]


def test_install_still_allowed_with_explicit_allow_unsigned(keyring, tmp_path, monkeypatch):
    import updater

    manifest = update_core.UpdateManifest.from_mapping(
        _signed_manifest(keyring), source_url="https://example.invalid/latest.json",
    )
    monkeypatch.setattr(
        updater, "_manifest_for_request", lambda *_a, **_k: (manifest, ()),
    )
    monkeypatch.setattr(
        updater, "detect_profile",
        lambda *_a, **_k: {"platform": "windows", "arch": "x86_64", "variant": "cpu"},
    )
    monkeypatch.setattr(
        updater, "download_asset",
        lambda *_a, **_k: tmp_path / "setup.exe",
    )
    monkeypatch.setattr(updater, "_launch_installer", lambda _path: 0)
    monkeypatch.setattr(updater, "detect_current_version", lambda *a, **k: "0.1.8")
    monkeypatch.setattr(updater, "default_state_dir", lambda: tmp_path / "state")
    code = updater.main(
        ["install", "--yes", "--allow-unsigned", "--source", "https://example.invalid/latest.json"]
    )
    assert code == 0


# --------------------------------------------------------------------------
# serve.py signed /latest.json
# --------------------------------------------------------------------------

def test_serve_build_update_manifest_signed_when_signer_configured(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "packaging_serve_for_test", PACKAGING_DIR / "serve.py",
    )
    serve = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(serve)

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "QLH-Edge-Inference-Setup-v0.1.8.1.exe").write_bytes(b"current")
    monkeypatch.setattr(serve, "_project_version", lambda: "0.1.8.1")
    serve._SHA256_CACHE.clear()

    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    signing.generate_keypair(key_dir, key_id="root", role="root")
    signing.generate_keypair(key_dir, key_id="release-20260809", role="release")
    signing.authorize_new_key(
        key_dir / "release-20260809.pub.json",
        issuer_private_path=key_dir / "root.key", issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    (pubkeys / "root.pub.json").write_bytes((key_dir / "root.pub.json").read_bytes())
    (pubkeys / "release-20260809.pub.json").write_bytes(
        (key_dir / "release-20260809.pub.json").read_bytes()
    )

    manifest = serve.build_update_manifest(
        str(dist_dir), signer=serve.Signer(str(key_dir / "release-20260809.key")),
    )
    assert manifest["key_id"] == "release-20260809"
    assert manifest["signature"]
    assert manifest["signed_at"]
    verified, reason = signing.verify_manifest_signature(
        manifest, trusted_keys_dir=pubkeys,
    )
    assert verified is True, reason
