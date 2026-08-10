import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import install_manifest
import repair
import signing
import update_core


@pytest.fixture()
def install_keyring(tmp_path):
    keys = tmp_path / "keys"
    keys.mkdir()
    signing.generate_keypair(keys, key_id="root", role="root")
    signing.generate_keypair(keys, key_id="release-repair", role="release")
    signing.authorize_new_key(
        keys / "release-repair.pub.json",
        issuer_private_path=keys / "root.key", issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    shutil.copy(keys / "root.pub.json", pubkeys / "root.pub.json")
    shutil.copy(keys / "release-repair.pub.json", pubkeys / "release-repair.pub.json")
    return {"key": keys / "release-repair.key", "pubkeys": pubkeys}


def _application_tree(root: Path, keyring) -> Path:
    root.mkdir()
    (root / "QLH-Edge-Inference.exe").write_bytes(b"application")
    (root / "_internal").mkdir()
    (root / "_internal" / "runtime.dll").write_bytes(b"runtime")
    (root / "frontend").mkdir()
    (root / "frontend" / "index.html").write_text("frontend", encoding="utf-8")
    (root / "version.txt").write_text("0.1.8.1\n", encoding="utf-8")
    shutil.copytree(keyring["pubkeys"], root / "pubkeys")
    for reserved in install_manifest.RESERVED_USER_DATA_ROOTS:
        directory = root / reserved
        directory.mkdir()
        (directory / "user-owned.bin").write_bytes(b"do not touch")
    install_manifest.write_signed_install_manifest(
        root,
        app_id="qlh-edge-inference",
        version="0.1.8.1",
        platform="windows",
        variant="cpu",
        package_kind="application",
        private_key_path=keyring["key"],
        trusted_keys_dir=keyring["pubkeys"],
        generated_at="2026-08-11T00:00:00+00:00",
        signed_at="2026-08-11T00:00:01+00:00",
    )
    return root


def _repair_source(root: Path, keyring, tmp_path):
    payloads = tmp_path / "payloads"
    index = tmp_path / "QLH-Edge-Inference-Repair-v0.1.8.1-windows-cpu.json"
    repair.build_repair_index(
        root,
        output=index,
        payload_dir=payloads,
        url_prefix="/repair/0.1.8.1/windows/cpu",
        trusted_keys_dir=keyring["pubkeys"],
    )
    payload = index.read_bytes()
    asset = {
        "name": index.name,
        "url": "https://updates.example/" + index.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "platform": "windows",
        "variant": "cpu",
        "arch": "x86_64",
        "kind": "repair-index",
    }
    manifest = update_core.UpdateManifest.from_mapping(
        {"schema_version": 1, "tag": "0.1.8.1", "assets": [asset]},
        source_url="https://updates.example/latest.json",
    )
    return index, payloads, replace(manifest, signature_present=True, signature_verified=True)


def _downloader(index: Path, payloads: Path):
    def download(asset, destination, *, timeout):
        del timeout
        target_dir = Path(destination)
        target_dir.mkdir(parents=True, exist_ok=True)
        source = index if asset.name == index.name else payloads / asset.sha256
        target = target_dir / asset.name
        shutil.copyfile(source, target)
        return target

    return download


def _profile():
    return {"platform": "windows", "variant": "cpu", "arch": "x86_64"}


def test_repair_restores_one_to_three_corrupted_files_and_preserves_user_data(
    tmp_path, install_keyring,
):
    root = _application_tree(tmp_path / "app", install_keyring)
    index, payloads, manifest = _repair_source(root, install_keyring, tmp_path)
    (root / "QLH-Edge-Inference.exe").unlink()
    (root / "_internal" / "runtime.dll").write_bytes(b"corrupt")
    (root / "frontend" / "index.html").write_text("bad", encoding="utf-8")
    before_data = {
        reserved: (root / reserved / "user-owned.bin").read_bytes()
        for reserved in install_manifest.RESERVED_USER_DATA_ROOTS
    }

    report = repair.repair_install(
        root,
        sources=["https://updates.example/latest.json"],
        profile=_profile(),
        trusted_keys_dir=install_keyring["pubkeys"],
        fetcher=lambda _url, timeout: manifest,
        downloader=_downloader(index, payloads),
        download_dir=tmp_path / "downloads",
        backup_dir=tmp_path / "backups",
    )

    assert report["ok"] is True
    assert report["action"] == "repaired"
    assert set(report["repaired"]) == {
        "QLH-Edge-Inference.exe", "_internal/runtime.dll", "frontend/index.html",
    }
    assert install_manifest.verify_install_tree(
        root, level="deep", trusted_keys_dir=install_keyring["pubkeys"],
    )["ok"] is True
    assert all(
        (root / reserved / "user-owned.bin").read_bytes() == value
        for reserved, value in before_data.items()
    )
    assert len(list((tmp_path / "backups").glob("*.bak"))) == 2


def test_repair_threshold_escalates_without_fetch_or_write(tmp_path, install_keyring, monkeypatch):
    root = _application_tree(tmp_path / "app", install_keyring)
    monkeypatch.setattr(repair, "MAX_REPAIR_FILES", 1)
    (root / "QLH-Edge-Inference.exe").write_bytes(b"bad")
    (root / "_internal" / "runtime.dll").write_bytes(b"bad")
    original = (root / "frontend" / "index.html").read_bytes()

    report = repair.repair_install(
        root,
        sources=["https://updates.example/latest.json"],
        profile=_profile(),
        trusted_keys_dir=install_keyring["pubkeys"],
        fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    assert report["ok"] is False
    assert report["action"] == "escalate"
    assert (root / "QLH-Edge-Inference.exe").read_bytes() == b"bad"
    assert (root / "_internal" / "runtime.dll").read_bytes() == b"bad"
    assert (root / "frontend" / "index.html").read_bytes() == original


def test_repair_refuses_untrusted_local_baseline_before_network(tmp_path, install_keyring):
    root = _application_tree(tmp_path / "app", install_keyring)
    manifest_path = root / install_manifest.MANIFEST_RELATIVE_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "0.1.8.2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(repair.RepairError, match="not trusted"):
        repair.repair_install(
            root,
            sources=["https://updates.example/latest.json"],
            profile=_profile(),
            trusted_keys_dir=install_keyring["pubkeys"],
            fetcher=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not fetch")),
        )


def test_repair_rejects_index_path_outside_signed_program_files(tmp_path, install_keyring):
    root = _application_tree(tmp_path / "app", install_keyring)
    index, payloads, manifest = _repair_source(root, install_keyring, tmp_path)
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["files"][0]["path"] = "models/should-not-be-written.bin"
    index.write_text(json.dumps(index_value), encoding="utf-8")
    updated = index.read_bytes()
    tampered_asset = dict(manifest.assets[0].__dict__)
    tampered_asset["size"] = len(updated)
    tampered_asset["sha256"] = hashlib.sha256(updated).hexdigest()
    tampered_manifest = update_core.UpdateManifest.from_mapping(
        {"schema_version": 1, "tag": "0.1.8.1", "assets": [tampered_asset]},
        source_url=manifest.source_url,
    )
    tampered_manifest = replace(tampered_manifest, signature_present=True, signature_verified=True)
    (root / "QLH-Edge-Inference.exe").unlink()

    with pytest.raises(repair.RepairError, match="outside the local baseline"):
        repair.repair_install(
            root,
            sources=["https://updates.example/latest.json"],
            profile=_profile(),
            trusted_keys_dir=install_keyring["pubkeys"],
            fetcher=lambda _url, timeout: tampered_manifest,
            downloader=_downloader(index, payloads),
            download_dir=tmp_path / "downloads",
        )
    assert not (root / "models" / "should-not-be-written.bin").exists()


def test_repair_rejects_unsafe_payload_url_before_replacement(tmp_path, install_keyring):
    root = _application_tree(tmp_path / "app", install_keyring)
    index, payloads, manifest = _repair_source(root, install_keyring, tmp_path)
    index_value = json.loads(index.read_text(encoding="utf-8"))
    index_value["files"][0]["url"] = "/repair/../outside"
    index.write_text(json.dumps(index_value), encoding="utf-8")
    updated = index.read_bytes()
    asset = dict(manifest.assets[0].__dict__)
    asset["size"] = len(updated)
    asset["sha256"] = hashlib.sha256(updated).hexdigest()
    unsafe_manifest = update_core.UpdateManifest.from_mapping(
        {"schema_version": 1, "tag": "0.1.8.1", "assets": [asset]},
        source_url=manifest.source_url,
    )
    unsafe_manifest = replace(unsafe_manifest, signature_present=True, signature_verified=True)
    (root / "QLH-Edge-Inference.exe").unlink()

    with pytest.raises(repair.RepairError, match="URL is unsafe"):
        repair.repair_install(
            root,
            sources=["https://updates.example/latest.json"],
            profile=_profile(),
            trusted_keys_dir=install_keyring["pubkeys"],
            fetcher=lambda _url, timeout: unsafe_manifest,
            downloader=_downloader(index, payloads),
            download_dir=tmp_path / "downloads",
        )
    assert not (root / "QLH-Edge-Inference.exe").exists()


def test_repair_rolls_back_replacements_when_post_verification_fails(
    tmp_path, install_keyring, monkeypatch,
):
    root = _application_tree(tmp_path / "app", install_keyring)
    index, payloads, manifest = _repair_source(root, install_keyring, tmp_path)
    target = root / "_internal" / "runtime.dll"
    target.write_bytes(b"corrupt-before-repair")
    real_verify = repair.verify_install_tree
    calls = 0

    def verify_after_repair(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"ok": False, "summary": {"checked": 1, "failed": 1}, "failed": []}
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(repair, "verify_install_tree", verify_after_repair)
    report = repair.repair_install(
        root,
        sources=["https://updates.example/latest.json"],
        profile=_profile(),
        trusted_keys_dir=install_keyring["pubkeys"],
        fetcher=lambda _url, timeout: manifest,
        downloader=_downloader(index, payloads),
        download_dir=tmp_path / "downloads",
        backup_dir=tmp_path / "backups",
    )

    assert report["ok"] is False
    assert report["action"] == "failed"
    assert target.read_bytes() == b"corrupt-before-repair"
    assert len(list((tmp_path / "backups").glob("*.bak"))) == 1


def test_repair_index_builder_only_emits_signed_program_paths(tmp_path, install_keyring):
    root = _application_tree(tmp_path / "app", install_keyring)
    index = repair.build_repair_index(
        root,
        output=tmp_path / "index.json",
        payload_dir=tmp_path / "payloads",
        url_prefix="/repair/test",
        trusted_keys_dir=install_keyring["pubkeys"],
    )
    value = json.loads(index.read_text(encoding="utf-8"))

    assert value["manifest_type"] == repair.REPAIR_INDEX_TYPE
    assert all(
        item["path"].split("/", 1)[0] not in install_manifest.RESERVED_USER_DATA_ROOTS
        for item in value["files"]
    )
    assert all((tmp_path / "payloads" / item["sha256"]).is_file() for item in value["files"])
