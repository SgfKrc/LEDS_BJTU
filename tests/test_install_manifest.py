import copy
import json
import shutil
import sys
from pathlib import Path

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import install_manifest
import signing
from launcher_slots import LauncherSlotError, LauncherSlotStore
from version_store import VersionStore, VersionStoreError


@pytest.fixture()
def install_keyring(tmp_path):
    keys = tmp_path / "keys"
    keys.mkdir()
    signing.generate_keypair(keys, key_id="root", role="root")
    signing.generate_keypair(keys, key_id="release-install", role="release")
    signing.authorize_new_key(
        keys / "release-install.pub.json",
        issuer_private_path=keys / "root.key",
        issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    shutil.copy(keys / "root.pub.json", pubkeys / "root.pub.json")
    shutil.copy(
        keys / "release-install.pub.json",
        pubkeys / "release-install.pub.json",
    )
    return {
        "release_key": keys / "release-install.key",
        "pubkeys": pubkeys,
    }


def _application_tree(root: Path) -> Path:
    root.mkdir()
    (root / "QLH-Edge-Inference.exe").write_bytes(b"application")
    (root / "_internal").mkdir()
    (root / "_internal" / "runtime.dll").write_bytes(b"runtime")
    (root / "frontend").mkdir()
    (root / "frontend" / "index.html").write_text("app", encoding="utf-8")
    for reserved in install_manifest.RESERVED_USER_DATA_ROOTS:
        directory = root / reserved
        directory.mkdir()
        (directory / "user-owned.bin").write_bytes(b"never hash me")
    return root


def _write_signed(
    root: Path,
    keyring,
    *,
    app_id: str = "qlh-edge-inference",
    version: str = "0.1.8.1",
    variant: str = "cpu",
    package_kind: str = "application",
    includes=(),
) -> Path:
    return install_manifest.write_signed_install_manifest(
        root,
        app_id=app_id,
        version=version,
        platform="windows",
        variant=variant,
        package_kind=package_kind,
        private_key_path=keyring["release_key"],
        trusted_keys_dir=keyring["pubkeys"],
        includes=includes,
        generated_at="2026-08-11T00:00:00+00:00",
        signed_at="2026-08-11T00:00:01+00:00",
    )


def test_signed_install_manifest_has_strict_fields_and_excludes_user_data(
    tmp_path, install_keyring,
):
    root = _application_tree(tmp_path / "app")
    external_version = tmp_path / "version.txt"
    external_version.write_text("0.1.8.1\n", encoding="utf-8")
    path = _write_signed(
        root,
        install_keyring,
        includes=(
            install_manifest.IncludeSource(external_version, "version.txt"),
        ),
    )

    mapping = install_manifest.load_install_manifest(path)
    verified = install_manifest.verify_install_manifest(
        mapping, trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert path == root / "manifest" / "install-manifest.json"
    assert verified["schema_version"] == 1
    assert verified["manifest_type"] == "qlh_install"
    assert verified["scope"] == "application_files_only"
    assert verified["app_id"] == "qlh-edge-inference"
    assert verified["version"] == "0.1.8.1"
    assert verified["platform"] == "windows"
    assert verified["variant"] == "cpu"
    assert verified["package_kind"] == "application"
    assert verified["key_id"] == "release-install"
    paths = [entry["path"] for entry in verified["files"]]
    assert paths == sorted(paths, key=lambda value: (value.casefold(), value))
    assert "QLH-Edge-Inference.exe" in paths
    assert "_internal/runtime.dll" in paths
    assert "frontend/index.html" in paths
    assert "version.txt" in paths
    assert install_manifest.MANIFEST_RELATIVE_PATH not in paths
    assert all(
        path.split("/", 1)[0].casefold()
        not in install_manifest.RESERVED_USER_DATA_ROOTS
        for path in paths
    )
    assert all(set(entry) == {"path", "size", "sha256", "kind"} for entry in verified["files"])


def test_manifest_tampering_is_rejected_before_persist(tmp_path, install_keyring):
    source_root = _application_tree(tmp_path / "source")
    source = _write_signed(source_root, install_keyring)
    install_root = tmp_path / "installed"
    install_root.mkdir()
    destination = install_manifest.persist_verified_install_manifest(
        source,
        install_root=install_root,
        trusted_keys_dir=install_keyring["pubkeys"],
    )
    original = destination.read_bytes()

    tampered = install_manifest.load_install_manifest(source)
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(install_manifest.InstallManifestError, match="signature rejected"):
        install_manifest.persist_verified_install_manifest(
            tampered,
            install_root=install_root,
            trusted_keys_dir=install_keyring["pubkeys"],
        )
    assert destination.read_bytes() == original


def test_even_signed_manifest_cannot_claim_reserved_user_data(tmp_path, install_keyring):
    root = _application_tree(tmp_path / "app")
    path = _write_signed(root, install_keyring)
    mapping = install_manifest.load_install_manifest(path)
    mapping["files"].append(
        {
            "path": "models/user.gguf",
            "size": 1,
            "sha256": "0" * 64,
            "kind": "application",
        }
    )
    mapping["files"].sort(key=lambda entry: (entry["path"].casefold(), entry["path"]))
    resigned = signing.sign_manifest(
        mapping,
        private_key_path=install_keyring["release_key"],
        signed_at="2026-08-11T00:00:02+00:00",
    )
    with pytest.raises(install_manifest.InstallManifestError, match="user data path"):
        install_manifest.verify_install_manifest(
            resigned, trusted_keys_dir=install_keyring["pubkeys"],
        )


@pytest.mark.parametrize("field,value", [("platform", []), ("package_kind", {}), ("kind", [])])
def test_install_manifest_rejects_malformed_json_types(tmp_path, install_keyring, field, value):
    root = _application_tree(tmp_path / "app")
    mapping = install_manifest.load_install_manifest(_write_signed(root, install_keyring))
    if field == "kind":
        mapping["files"][0][field] = value
    else:
        mapping[field] = value
    with pytest.raises(install_manifest.InstallManifestError):
        install_manifest.validate_install_manifest(mapping)


@pytest.mark.parametrize(
    "path",
    ["../escape.dll", "/absolute.dll", "C:/windows.dll", "a\\b.dll", "a//b.dll"],
)
def test_install_manifest_rejects_unsafe_paths(path):
    with pytest.raises(install_manifest.InstallManifestError):
        install_manifest.normalize_relative_path(path)


def test_install_manifest_rejects_case_colliding_includes(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    with pytest.raises(install_manifest.InstallManifestError, match="case-colliding"):
        install_manifest.collect_install_files(
            root,
            includes=(
                install_manifest.IncludeSource(payload, "Docs/Readme.txt"),
                install_manifest.IncludeSource(payload, "docs/readme.txt"),
            ),
        )


def _launcher_tree(root: Path, keyring, *, version: str = "0.1.8.1") -> Path:
    root.mkdir()
    (root / "QLH-Launcher.exe").write_bytes(b"launcher")
    (root / "health.ok").write_text("healthy", encoding="utf-8")
    (root / "version.txt").write_text(f"{version}\n", encoding="utf-8")
    shutil.copytree(keyring["pubkeys"], root / "pubkeys")
    _write_signed(
        root,
        keyring,
        app_id="qlh-launcher",
        version=version,
        variant="any",
        package_kind="launcher",
    )
    return root


def test_launcher_staging_requires_signed_manifest_and_matching_version(
    tmp_path, install_keyring,
):
    bundle = _launcher_tree(tmp_path / "launcher", install_keyring)
    store = LauncherSlotStore(tmp_path / "slots")
    staged = store.stage_directory(
        bundle, "0.1.8.1", require_install_manifest=True,
    )
    assert staged.version == "0.1.8.1"

    missing = tmp_path / "legacy"
    missing.mkdir()
    (missing / "QLH-Launcher.exe").write_bytes(b"legacy")
    with pytest.raises(LauncherSlotError, match="no signed install-manifest"):
        LauncherSlotStore(tmp_path / "missing-store").stage_directory(
            missing, "0.1.8.1", require_install_manifest=True,
        )
    with pytest.raises(LauncherSlotError, match="version mismatch"):
        LauncherSlotStore(tmp_path / "wrong-version-store").stage_directory(
            bundle, "0.1.8.2", require_install_manifest=True,
        )


def test_application_version_store_rejects_signed_identity_mismatch(
    tmp_path, install_keyring,
):
    bundle = _application_tree(tmp_path / "app")
    shutil.copytree(install_keyring["pubkeys"], bundle / "pubkeys")
    _write_signed(bundle, install_keyring, variant="cpu")
    store = VersionStore(tmp_path / "store")
    target = store.stage_directory(bundle, "0.1.8.1", "cpu")
    assert target.is_dir()

    with pytest.raises(VersionStoreError, match="identity mismatch"):
        VersionStore(tmp_path / "other-store").stage_directory(
            bundle, "0.1.8.1", "cuda",
        )


def test_install_manifest_cli_build_and_validate(tmp_path, install_keyring, capsys):
    root = _application_tree(tmp_path / "cli-app")
    assert install_manifest.main([
        "build",
        "--root", str(root),
        "--app-id", "qlh-edge-inference",
        "--version", "0.1.8.1",
        "--platform", "windows",
        "--variant", "cpu",
        "--package-kind", "application",
        "--key", str(install_keyring["release_key"]),
        "--trusted-keys-dir", str(install_keyring["pubkeys"]),
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["file_count"] == 3
    assert output["key_id"] == "release-install"
    assert install_manifest.main([
        "validate",
        "--manifest", str(root / install_manifest.MANIFEST_RELATIVE_PATH),
        "--trusted-keys-dir", str(install_keyring["pubkeys"]),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_install_manifest_cli_refuses_unsigned_release_build(tmp_path, capsys):
    root = tmp_path / "app"
    root.mkdir()
    (root / "app.exe").write_bytes(b"app")
    assert install_manifest.main([
        "build",
        "--root", str(root),
        "--app-id", "qlh-edge-inference",
        "--version", "0.1.8.1",
        "--platform", "windows",
        "--variant", "cpu",
        "--package-kind", "application",
    ]) == 2
    assert "requires --key" in capsys.readouterr().err


def _verified_application_tree(root: Path, keyring) -> Path:
    root = _application_tree(root)
    (root / "version.txt").write_text("0.1.8.1\n", encoding="utf-8")
    _write_signed(root, keyring)
    return root


def test_runtime_quick_verify_hashes_critical_files_and_ignores_user_data(
    tmp_path, install_keyring,
):
    root = _verified_application_tree(tmp_path / "app", install_keyring)
    report = install_manifest.verify_install_tree(
        root, level="quick", trusted_keys_dir=install_keyring["pubkeys"],
    )

    assert report["ok"] is True
    assert report["level"] == "quick"
    assert {item["path"] for item in report["passed"]} >= {
        install_manifest.MANIFEST_RELATIVE_PATH,
        "QLH-Edge-Inference.exe",
        "version.txt",
    }
    assert report["summary"]["hash_verified"] >= 2
    assert all(
        item["path"].split("/", 1)[0].casefold()
        not in install_manifest.RESERVED_USER_DATA_ROOTS
        for item in report["passed"] + report["failed"]
        if item["path"] not in {".", install_manifest.MANIFEST_RELATIVE_PATH}
    )


@pytest.mark.parametrize("level", ["full", "deep"])
def test_runtime_full_and_deep_reject_a_signed_baseline_without_entrypoint(
    tmp_path, install_keyring, level,
):
    root = tmp_path / "incomplete-app"
    root.mkdir()
    (root / "version.txt").write_text("0.1.8.1\n", encoding="utf-8")
    _write_signed(root, install_keyring)

    report = install_manifest.verify_install_tree(
        root, level=level, trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert report["ok"] is False
    assert any(
        item["path"] == "QLH-Edge-Inference.exe"
        and item["category"] == "missing"
        and item["actual"] == "not listed"
        for item in report["failed"]
    )


def test_runtime_verify_reports_missing_hash_and_health_failures(tmp_path, install_keyring):
    missing_root = _verified_application_tree(tmp_path / "missing", install_keyring)
    (missing_root / "QLH-Edge-Inference.exe").unlink()
    missing = install_manifest.verify_install_tree(
        missing_root, trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert missing["ok"] is False
    assert any(
        item["path"] == "QLH-Edge-Inference.exe" and item["category"] == "missing"
        for item in missing["failed"]
    )
    assert missing["advice"]

    hash_root = _verified_application_tree(tmp_path / "hash", install_keyring)
    (hash_root / "_internal" / "runtime.dll").write_bytes(b"RUNTIME")
    hashed = install_manifest.verify_install_tree(
        hash_root, level="deep", trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert any(
        item["path"] == "_internal/runtime.dll" and item["category"] == "hash"
        for item in hashed["failed"]
    )

    launcher_root = _launcher_tree(tmp_path / "launcher-verify", install_keyring)
    (launcher_root / "health.ok").unlink()
    health = install_manifest.verify_install_tree(
        launcher_root, trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert any(
        item["path"] == "health.ok" and item["category"] == "missing"
        for item in health["failed"]
    )


def test_runtime_full_sampling_is_explicit_and_deep_detects_large_file_tamper(
    tmp_path, install_keyring, monkeypatch,
):
    monkeypatch.setattr(install_manifest, "LARGE_FILE_BYTES", 8)
    monkeypatch.setattr(install_manifest, "SAMPLE_WINDOW_BYTES", 4)
    root = _application_tree(tmp_path / "app")
    (root / "version.txt").write_text("0.1.8.1\n", encoding="utf-8")
    payload = root / "large.bin"
    payload.write_bytes(b"0123456789abcdef")
    _write_signed(root, install_keyring)

    full = install_manifest.verify_install_tree(
        root, level="full", trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert full["ok"] is True
    assert full["summary"]["sampled"] >= 1
    assert {item["path"]: item["check"] for item in full["passed"]}["large.bin"] == "size+sample"

    payload.write_bytes(b"01234X6789abcdef")
    full_after_tamper = install_manifest.verify_install_tree(
        root, level="full", trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert full_after_tamper["ok"] is True
    deep = install_manifest.verify_install_tree(
        root, level="deep", trusted_keys_dir=install_keyring["pubkeys"],
    )
    assert any(
        item["path"] == "large.bin" and item["category"] == "hash"
        for item in deep["failed"]
    )


def test_runtime_verify_signature_failure_and_json_cli_exit_code(tmp_path, install_keyring, capsys):
    root = _verified_application_tree(tmp_path / "app", install_keyring)
    manifest_path = root / install_manifest.MANIFEST_RELATIVE_PATH
    manifest = install_manifest.load_install_manifest(manifest_path)
    manifest["version"] = "0.1.8.2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert install_manifest.main([
        "verify", "--root", str(root), "--trusted-keys-dir", str(install_keyring["pubkeys"]), "--json",
    ]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["failed"][0]["category"] == "signature"


def test_release_build_entrypoints_require_and_package_signed_baseline():
    project = PACKAGING_DIR.parent
    for name in ("build-cpu.bat", "build-cuda.bat", "build-launcher.bat"):
        text = (PACKAGING_DIR / name).read_text(encoding="utf-8")
        assert "QLH_SIGNING_KEY" in text
        assert "install_manifest.py build" in text
        assert "--trusted-keys-dir" in text
        assert "qlh-install-manifest.spec" in text
        assert "QLH-Install-Manifest.exe" in text
        assert "packaging\\pubkeys" in text
    installer = (PACKAGING_DIR / "build-installer.bat").read_text(encoding="utf-8")
    assert "install_manifest.py\" validate" in installer
    assert "install-manifest.json" in installer

    linux_build = (PACKAGING_DIR / "linux" / "build-deb.sh").read_text(encoding="utf-8")
    linux_postinst = (PACKAGING_DIR / "linux" / "postinst").read_text(encoding="utf-8")
    assert "QLH_SIGNING_KEY" in linux_build
    assert 'install_manifest.py" build' in linux_build
    assert '"$MANIFEST_TOOL" validate' in linux_postinst
    assert 'manifest/install-manifest.json' in linux_postinst

    launcher_setup = (PACKAGING_DIR / "setup-launcher.iss").read_text(encoding="utf-8")
    assert 'Source: "{#MySourceDir}\\*"' in launcher_setup
    assert 'Source: "version.txt"' not in launcher_setup
    for name in ("setup.iss", "setup-cuda.iss", "setup-launcher.iss"):
        setup = (PACKAGING_DIR / name).read_text(encoding="utf-8")
        assert "QLH-Install-Manifest.exe" in setup
        assert "RaiseException" in setup
    for name in ("setup.iss", "setup-cuda.iss"):
        setup = (PACKAGING_DIR / name).read_text(encoding="utf-8")
        assert 'verify --root "' in setup
        assert "--level deep" in setup
        assert "validate --manifest" not in setup
        assert 'Source: "..\\' not in setup
        assert 'Source: "version.txt"' not in setup
        assert 'Source: "scripts\\convert_to_gguf.py"' not in setup
        assert "RetainUserData" in setup
        assert "ReassociateRetainedData" in setup
        assert "RunDataRetention" in setup
        assert "QLH-Data-Retention.exe" in setup
        assert "--data-root" in setup and "--yes --json" in setup
        assert "RenameFile(Sources" not in setup
        assert "{localappdata}\\QLH-Edge-Inference\\data" in setup
        assert "DelTree(ExpandConstant('{app}\\models')" not in setup
        assert "[UninstallDelete]" not in setup
    launcher_setup = (PACKAGING_DIR / "setup-launcher.iss").read_text(encoding="utf-8")
    assert "install-manifest.json" in launcher_setup
    linux_prerm = (PACKAGING_DIR / "linux" / "prerm").read_text(encoding="utf-8")
    linux_postinst = (PACKAGING_DIR / "linux" / "postinst").read_text(encoding="utf-8")
    assert 'RETENTION_TOOL="$APP_DIR/bin/data_retention.py"' in linux_prerm
    assert '"$RETENTION_TOOL" retain' in linux_prerm
    assert 'RETENTION_TOOL="$APP_DIR/bin/data_retention.py"' in linux_postinst
    assert '"$RETENTION_TOOL" reassociate' in linux_postinst
    verifier_spec = (PACKAGING_DIR / "qlh-install-manifest.spec").read_text(encoding="utf-8")
    assert 'name="QLH-Install-Manifest"' in verifier_spec
    assert '"torch"' in verifier_spec and "excludes=" in verifier_spec
    retention_spec = (PACKAGING_DIR / "qlh-data-retention.spec").read_text(encoding="utf-8")
    assert 'name="QLH-Data-Retention"' in retention_spec
    assert '"torch"' in retention_spec and "excludes=" in retention_spec
    for name in ("build-cpu.bat", "build-cuda.bat"):
        build = (PACKAGING_DIR / name).read_text(encoding="utf-8")
        helper_build = build.index("packaging\\qlh-data-retention.spec")
        manifest_build = build.index("packaging\\install_manifest.py build")
        assert helper_build < manifest_build
        assert "QLH-Data-Retention.exe" in build
    assert (project / "packaging" / "install_manifest.py").is_file()
