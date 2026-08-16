import importlib.util
import hashlib
from pathlib import Path


def _load_serve_module():
    module_path = Path(__file__).resolve().parents[1] / "packaging" / "serve.py"
    spec = importlib.util.spec_from_file_location("packaging_serve_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_bundle_scan_merges_pc_volumes_and_android(tmp_path, monkeypatch):
    serve = _load_serve_module()
    monkeypatch.setattr(serve, "BUNDLE_DIR", str(tmp_path))

    # PC 分卷 .001-.003 + 安卓单卷
    for vol in ("qlh-models-pc-v1.7z.001", "qlh-models-pc-v1.7z.002",
                "qlh-models-pc-v1.7z.003"):
        (tmp_path / vol).write_bytes(b"vol")
    (tmp_path / "qlh-models-android-v1.7z").write_bytes(b"android")
    (tmp_path / "unrelated.bin").write_bytes(b"x")

    entries = serve._scan_model_bundles()
    display_names = [entry[1] for entry in entries]

    # PC 分卷合并为一个代表项；安卓单卷一项；无关文件不出现
    assert any("(.001-003)" in n for n in display_names)
    assert "qlh-models-android-v1.7z" in display_names
    assert len(entries) == 2
    # 代表项 href 指向 .001
    pc_entry = next(e for e in entries if e[0] == "pc-bundle")
    assert pc_entry[2].endswith("qlh-models-pc-v1.7z.001")


def test_model_bundle_path_resolution_rejects_traversal_and_missing(tmp_path, monkeypatch):
    serve = _load_serve_module()
    monkeypatch.setattr(serve, "BUNDLE_DIR", str(tmp_path))

    (tmp_path / "qlh-models-android-v1.7z").write_bytes(b"android")
    (tmp_path / "qlh-models-pc-v1.7z.001").write_bytes(b"vol1")
    (tmp_path / "secret.7z").write_bytes(b"secret")

    assert serve._resolve_model_bundle_path(
        "/models-bundle/qlh-models-android-v1.7z") == str(
        tmp_path / "qlh-models-android-v1.7z")
    assert serve._resolve_model_bundle_path(
        "/models-bundle/qlh-models-pc-v1.7z.001") == str(
        tmp_path / "qlh-models-pc-v1.7z.001")
    # 穿越防护与不存在
    assert serve._resolve_model_bundle_path(
        "/models-bundle/../secret.7z") is None
    assert serve._resolve_model_bundle_path(
        "/models-bundle/nope.7z") is None
    assert serve._resolve_model_bundle_path("/models-pc/legacy.7z") is None


def test_pc_installer_scan_excludes_android_packages_in_dist(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "packaging" / "dist"
    dist_dir.mkdir(parents=True)
    monkeypatch.setattr(serve, "DIST_DIR", str(dist_dir))

    installer = dist_dir / "QLH-Edge-Inference-Setup-v0.1.8.1.exe"
    launcher_setup = dist_dir / "QLH-Launcher-Setup-v0.1.8.1.exe"
    full_apk = dist_dir / "QLH-Inference-v0.1.8.1-full-release.apk"
    lite_apk = dist_dir / "QLH-Inference-v0.1.8.1-lite-release.apk"
    installer.write_bytes(b"exe")
    launcher_setup.write_bytes(b"exe")
    full_apk.write_bytes(b"apk")
    lite_apk.write_bytes(b"apk")

    entries = serve._scan_pc_installers()

    assert entries == [
        ("QLH-Edge-Inference-Setup-v0.1.8.1.exe", "/QLH-Edge-Inference-Setup-v0.1.8.1.exe", str(installer))
    ]


def test_launcher_scan_finds_setup_and_zip_excluding_main_installer(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "packaging" / "dist"
    dist_dir.mkdir(parents=True)
    monkeypatch.setattr(serve, "DIST_DIR", str(dist_dir))

    launcher_setup = dist_dir / "QLH-Launcher-Setup-v0.1.8.1.exe"
    launcher_zip = dist_dir / "QLH-Launcher-v0.1.8.1.zip"
    main_installer = dist_dir / "QLH-Edge-Inference-Setup-v0.1.8.1.exe"
    launcher_setup.write_bytes(b"exe")
    launcher_zip.write_bytes(b"zip")
    main_installer.write_bytes(b"exe")

    entries = serve._scan_launcher_assets()

    assert entries == [
        ("QLH-Launcher-Setup-v0.1.8.1.exe", "/QLH-Launcher-Setup-v0.1.8.1.exe", str(launcher_setup)),
        ("QLH-Launcher-v0.1.8.1.zip", "/QLH-Launcher-v0.1.8.1.zip", str(launcher_zip)),
    ]


def test_android_download_scan_includes_dist_and_gradle_outputs(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "packaging" / "dist"
    android_outputs = tmp_path / "android" / "app" / "build" / "outputs"
    dist_dir.mkdir(parents=True)
    gradle_release_dir = android_outputs / "apk" / "full" / "release"
    gradle_debug_dir = android_outputs / "apk" / "full" / "debug"
    gradle_release_dir.mkdir(parents=True)
    gradle_debug_dir.mkdir(parents=True)
    monkeypatch.setattr(serve, "DIST_DIR", str(dist_dir))
    monkeypatch.setattr(serve, "ANDROID_OUTPUT_DIR", str(android_outputs))

    full_apk = dist_dir / "QLH-Inference-v0.1.8.1-full-release.apk"
    lite_apk = dist_dir / "QLH-Inference-v0.1.8.1-lite-release.apk"
    gradle_apk = gradle_release_dir / "app-full-release.apk"
    debug_apk = gradle_debug_dir / "app-full-debug.apk"
    full_apk.write_bytes(b"full")
    lite_apk.write_bytes(b"lite")
    gradle_apk.write_bytes(b"gradle")
    debug_apk.write_bytes(b"debug")

    entries = serve._scan_android_downloads()

    assert entries == [
        ("QLH-Inference-v0.1.8.1-full-release.apk", "/QLH-Inference-v0.1.8.1-full-release.apk", str(full_apk)),
        ("QLH-Inference-v0.1.8.1-lite-release.apk", "/QLH-Inference-v0.1.8.1-lite-release.apk", str(lite_apk)),
        (
            "android/app/build/outputs/apk/full/release/app-full-release.apk",
            "/android/apk/full/release/app-full-release.apk",
            str(gradle_apk),
        ),
    ]


def test_reconfigure_utf8_converts_gbk_streams_without_crashing():
    """serve.py 在 Windows GBK 控制台输出 emoji 曾直接崩溃（回归）。"""
    import io

    serve = _load_serve_module()
    out = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    err = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    serve._reconfigure_utf8((out, err))
    assert out.encoding.lower() == "utf-8"
    assert err.encoding.lower() == "utf-8"
    # GBK 无法编码的字符在 UTF-8 下可正常写出
    out.write("📦 QLH 文件分发服务\n")
    out.flush()


def test_serve_dist_dir_matches_launcher_zip_output():
    """serve.py 扫描目录必须与 build-launcher.bat 的 ZIP 落点一致。"""
    serve = _load_serve_module()
    expected = str(Path(__file__).resolve().parents[1] / "packaging" / "dist")
    assert serve.DIST_DIR.replace("/", "\\") == expected.replace("/", "\\")
    bat = (Path(__file__).resolve().parents[1] / "packaging" / "build-launcher.bat").read_text(
        encoding="utf-8"
    )
    assert 'LAUNCHER_ZIP=packaging\\dist\\QLH-Launcher-v' in bat


def test_distribution_url_host_formats_ipv6():
    serve = _load_serve_module()

    assert serve._url_host("100.64.0.1") == "100.64.0.1"
    assert serve._url_host("fd7a:115c:a1e0::1") == "[fd7a:115c:a1e0::1]"


def test_distribution_server_accepts_ipv4_and_ipv6():
    import socket

    if not socket.has_ipv6:
        return
    serve = _load_serve_module()
    server = serve.create_distribution_server(
        "::", 0, serve.http.server.SimpleHTTPRequestHandler
    )
    try:
        port = server.server_address[1]
        assert server.socket.getsockopt(
            socket.IPPROTO_IPV6, socket.IPV6_V6ONLY
        ) == 0
        for host in ("127.0.0.1", "::1"):
            client = socket.create_connection((host, port), timeout=3)
            client.close()
    finally:
        server.server_close()


def test_update_manifest_only_exposes_current_version_and_classifies_assets(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    current = dist_dir / "QLH-Edge-Inference-Setup-v0.1.8.1.exe"
    old = dist_dir / "QLH-Edge-Inference-Setup-v0.1.8.exe"
    full = dist_dir / "QLH-Inference-v0.1.8.1-full-release.apk"
    launcher = dist_dir / "QLH-Launcher-v0.1.8.1.zip"
    current.write_bytes(b"current")
    old.write_bytes(b"old")
    full.write_bytes(b"full")
    launcher.write_bytes(b"launcher")
    monkeypatch.setattr(serve, "_project_version", lambda: "0.1.8.1")
    serve._SHA256_CACHE.clear()

    manifest = serve.build_update_manifest(str(dist_dir))

    assert manifest["schema_version"] == 1
    assert manifest["tag"] == "0.1.8.1"
    assert [item["name"] for item in manifest["assets"]] == [
        current.name, full.name, launcher.name,
    ]
    assert manifest["assets"][0]["variant"] == "cpu"
    assert manifest["assets"][0]["sha256"] == hashlib.sha256(b"current").hexdigest()
    assert manifest["assets"][1]["variant"] == "full"
    assert manifest["assets"][2]["kind"] == "launcher"
    assert manifest["assets"][2]["variant"] == "any"


def test_update_manifest_classifies_launcher_setup_exe(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    setup = dist_dir / "QLH-Launcher-Setup-v0.1.8.1.exe"
    bundle = dist_dir / "QLH-Launcher-v0.1.8.1.zip"
    setup.write_bytes(b"setup")
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(serve, "_project_version", lambda: "0.1.8.1")
    serve._SHA256_CACHE.clear()

    manifest = serve.build_update_manifest(str(dist_dir))

    by_name = {item["name"]: item for item in manifest["assets"]}
    assert by_name[setup.name]["kind"] == "launcher-setup"
    assert by_name[setup.name]["variant"] == "any"
    assert by_name[setup.name]["platform"] == "windows"
    assert by_name[bundle.name]["kind"] == "launcher"


def test_update_manifest_classifies_repair_index(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    index = dist_dir / "QLH-Edge-Inference-Repair-v0.1.8.1-windows-cpu.json"
    index.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(serve, "_project_version", lambda: "0.1.8.1")
    serve._SHA256_CACHE.clear()

    manifest = serve.build_update_manifest(str(dist_dir))

    assert manifest["assets"] == [{
        "name": index.name,
        "url": "/" + index.name,
        "size": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "platform": "windows",
        "variant": "cpu",
        "arch": "x86_64",
        "kind": "repair-index",
    }]
