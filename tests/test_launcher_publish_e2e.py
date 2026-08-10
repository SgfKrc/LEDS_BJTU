"""UP-N4 发布链路端到端测试：真实签名源站 → 验签 → launcher 资产选中 →
下载 → A/B 槽 staging → 隔离子进程健康探针 → 激活 → 回滚。

该测试不 mock 传输层：serve 的 HTTP handler、Ed25519 验签、下载事务、
`--health-check` 子进程和槽位指针切换全部走真实代码路径（ZIP 内为
packaging 源码模式 bundle，健康探针用本机 Python 执行）。
"""

import json
import shutil
import sys
import threading
import zipfile
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

PACKAGING_DIR = Path(__file__).resolve().parents[1] / "packaging"
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))

import launcher_slots
import install_manifest
import serve
import signing
import updater
import update_core

_BUNDLE_MODULES = (
    "qlh_launcher.py",
    "diagnose.py",
    "repair.py",
    "update_core.py",
    "updater.py",
    "signing.py",
    "install_manifest.py",
    "launcher_slots.py",
    "version_store.py",
)


def _build_launcher_zip(
    zip_path: Path,
    *,
    bundle_root: Path,
    pubkeys: Path,
    release_key: Path,
    version: str,
) -> None:
    shutil.rmtree(bundle_root, ignore_errors=True)
    bundle_root.mkdir()
    for name in _BUNDLE_MODULES:
        shutil.copy(PACKAGING_DIR / name, bundle_root / name)
    shutil.copytree(pubkeys, bundle_root / "pubkeys")
    (bundle_root / "health.ok").write_text("QLH Launcher e2e\n", encoding="utf-8")
    install_manifest.write_signed_install_manifest(
        bundle_root,
        app_id="qlh-launcher",
        version=version,
        platform="windows",
        variant="any",
        package_kind="launcher",
        private_key_path=release_key,
        trusted_keys_dir=pubkeys,
        generated_at="2026-08-11T00:00:00+00:00",
        signed_at="2026-08-11T00:00:01+00:00",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(bundle_root).as_posix())


class _PublishHandler(serve.QuietHTTPRequestHandler):
    """/latest.json 走签名清单；其余请求由 dist 目录静态服务（ZIP 下载）。"""

    signer = None

    def do_GET(self):
        if unquote(urlparse(self.path).path) == "/latest.json":
            manifest = serve.build_update_manifest(
                self.directory, signer=self.signer,
            )
            encoded = json.dumps(
                manifest, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        super().do_GET()


@pytest.fixture()
def publish_server(tmp_path):
    """测试密钥 + 真实源码模式 Launcher ZIP + 签名 HTTP 源站。"""
    keys = tmp_path / "keys"
    keys.mkdir()
    signing.generate_keypair(keys, key_id="root", role="root")
    signing.generate_keypair(keys, key_id="release-e2e", role="release")
    signing.authorize_new_key(
        keys / "release-e2e.pub.json",
        issuer_private_path=keys / "root.key", issuer_key_id="root",
    )
    pubkeys = tmp_path / "pubkeys"
    pubkeys.mkdir()
    shutil.copy(keys / "root.pub.json", pubkeys / "root.pub.json")
    shutil.copy(keys / "release-e2e.pub.json", pubkeys / "release-e2e.pub.json")

    dist = tmp_path / "dist"
    dist.mkdir()
    version = serve._project_version()
    zip_path = dist / f"QLH-Launcher-v{version}.zip"
    _build_launcher_zip(
        zip_path,
        bundle_root=tmp_path / "launcher-bundle",
        pubkeys=pubkeys,
        release_key=keys / "release-e2e.key",
        version=version,
    )

    _PublishHandler.signer = serve.Signer(str(keys / "release-e2e.key"))
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(_PublishHandler, directory=str(dist)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield {
            "base": base,
            "pubkeys": str(pubkeys),
            "version": version,
            "zip_path": zip_path,
            "release_key": keys / "release-e2e.key",
        }
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_publish_chain_check_download_verifies_through_real_http(publish_server, tmp_path):
    base = publish_server["base"]
    trusted = publish_server["pubkeys"]
    version = publish_server["version"]

    manifest = updater.fetch_manifest_with_keys(
        base + "/latest.json", timeout=8, trusted_keys_dir=trusted,
    )
    assert manifest.signature_verified is True
    assert manifest.signature_key_id == "release-e2e"

    result = updater.check_launcher_updates(
        [base + "/latest.json"],
        profile={"platform": "windows", "arch": "x86_64", "variant": "cpu"},
        current_version="0.1.8",
        trusted_keys_dir=trusted,
    )
    assert result["signature_verified"] is True
    assert result["update_available"] is True
    assert result["asset_kind"] == "launcher"
    assert result["asset"]["name"] == f"QLH-Launcher-v{version}.zip"

    asset = update_core.select_asset(
        manifest, platform="windows", variant="any",
        arch="x86_64", kind="launcher",
    )
    downloaded = update_core.download_asset(asset, tmp_path, timeout=30)
    assert downloaded.name == f"QLH-Launcher-v{version}.zip"
    assert downloaded.stat().st_size == asset.size
    assert update_core.verify_file(downloaded, asset)


def test_launcher_install_command_requires_signed_install_baseline(
    publish_server, tmp_path,
):
    store = tmp_path / "launcher-slots"
    code = updater.main([
        "launcher-install",
        "--source", publish_server["base"] + "/latest.json",
        "--trusted-keys-dir", publish_server["pubkeys"],
        "--launcher-store", str(store),
        "--download-dir", str(tmp_path / "downloads"),
        "--variant", "cpu",
        "--yes",
        "--json",
    ])
    assert code == 0
    active = launcher_slots.LauncherSlotStore(store)
    assert active.current().version == publish_server["version"]
    installed = active.slot_path(active.current().slot)
    assert (installed / "manifest" / "install-manifest.json").is_file()


def test_publish_chain_a_b_slots_activate_health_and_rollback(publish_server, tmp_path):
    """真实下载 → stage → 健康探针（真实子进程）→ activate → 再更新 →
    回滚 → 指针恢复旧版。"""
    base = publish_server["base"]
    trusted = publish_server["pubkeys"]
    version = publish_server["version"]

    manifest = updater.fetch_manifest_with_keys(
        base + "/latest.json", timeout=8, trusted_keys_dir=trusted,
    )
    asset = update_core.select_asset(
        manifest, platform="windows", variant="any",
        arch="x86_64", kind="launcher",
    )
    zip_path = update_core.download_asset(asset, tmp_path, timeout=30)

    store = launcher_slots.LauncherSlotStore(tmp_path / "launcher-slots")

    first = store.stage_archive(zip_path, version)
    first = store.activate(first.version, health_check=updater._launcher_health)
    assert store.current().version == version
    assert (store.slot_path(first.slot) / "qlh_launcher.py").is_file()

    # 新版本更新：独立签名的新版本 bundle 入槽 → 槽位翻转 → 回滚回旧版
    second_zip = tmp_path / "QLH-Launcher-v0.1.8.2.zip"
    _build_launcher_zip(
        second_zip,
        bundle_root=tmp_path / "launcher-bundle-v2",
        pubkeys=Path(publish_server["pubkeys"]),
        release_key=Path(publish_server["release_key"]),
        version="0.1.8.2",
    )
    second = store.stage_archive(second_zip, "0.1.8.2")
    second = store.activate(second.version, health_check=updater._launcher_health)
    assert store.current().version == "0.1.8.2"
    assert store.current().slot != first.slot

    rolled = store.rollback(health_check=updater._launcher_health)
    assert rolled.version == version
    assert store.current().version == version

    # 指针损坏恢复：current.json 指向不存在的槽 → recover 回到 previous
    # （previous.json 保持 rollback 后的真实指针，仅破坏 current）
    store.current_file.write_text(
        json.dumps({"schema_version": 1, "slot": "b", "version": "9.9.9",
                    "entrypoint": "QLH-Launcher.exe"}), encoding="utf-8",
    )
    recovered = store.recover()
    # rollback 后 previous 指向 slot b/0.1.8.2（槽真实存在），recover 恢复它
    assert recovered is not None and recovered.version == "0.1.8.2"


def test_publish_chain_tampered_asset_fails_closed(publish_server, tmp_path):
    """篡改已发布 ZIP 内容 → 下载事务 SHA-256 校验失败。"""
    base = publish_server["base"]
    trusted = publish_server["pubkeys"]

    manifest = updater.fetch_manifest_with_keys(
        base + "/latest.json", timeout=8, trusted_keys_dir=trusted,
    )
    asset = update_core.select_asset(
        manifest, platform="windows", variant="any",
        arch="x86_64", kind="launcher",
    )
    # 从源站下载后篡改内容再重放校验
    good = update_core.download_asset(asset, tmp_path, timeout=30)
    evil = tmp_path / ("evil-" + asset.name)
    evil.write_bytes(good.read_bytes() + b"tampered")
    from update_core import verify_file
    assert verify_file(evil, asset) is False
