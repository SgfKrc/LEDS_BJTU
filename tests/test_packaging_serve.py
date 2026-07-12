import importlib.util
from pathlib import Path


def _load_serve_module():
    module_path = Path(__file__).resolve().parents[1] / "packaging" / "serve.py"
    spec = importlib.util.spec_from_file_location("packaging_serve_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_archive_scan_uses_pc_and_android_packages(tmp_path, monkeypatch):
    serve = _load_serve_module()
    monkeypatch.setattr(serve, "PROJECT_ROOT", str(tmp_path))

    (tmp_path / "model.7z").write_bytes(b"legacy")
    (tmp_path / "models_pc.7z").write_bytes(b"pc-root")
    android_dir = tmp_path / "models_android" / "qwen"
    android_dir.mkdir(parents=True)
    (android_dir / "android-gguf.7z").write_bytes(b"android")

    entries = serve._scan_model_archives()
    display_names = [entry[1] for entry in entries]
    hrefs = [entry[2] for entry in entries]

    assert "models_pc.7z" in display_names
    assert "models_android/qwen/android-gguf.7z" in display_names
    assert "/models_pc.7z" in hrefs
    assert "/models-android/qwen/android-gguf.7z" in hrefs
    assert "model.7z" not in display_names


def test_model_archive_path_resolution_rejects_legacy_and_traversal(tmp_path, monkeypatch):
    serve = _load_serve_module()
    monkeypatch.setattr(serve, "PROJECT_ROOT", str(tmp_path))

    (tmp_path / "models_pc.7z").write_bytes(b"pc-root")
    pc_dir = tmp_path / "models_pc"
    pc_dir.mkdir()
    (pc_dir / "nested.7z").write_bytes(b"nested")
    (tmp_path / "secret.7z").write_bytes(b"secret")
    (tmp_path / "model.7z").write_bytes(b"legacy")

    assert serve._resolve_model_archive_path("/models_pc.7z") == str(tmp_path / "models_pc.7z")
    assert serve._resolve_model_archive_path("/models-pc/nested.7z") == str(pc_dir / "nested.7z")
    assert serve._resolve_model_archive_path("/model.7z") is None
    assert serve._resolve_model_archive_path("/models-pc/../secret.7z") is None
    assert serve._resolve_model_archive_path("/models-pc/not-a-model.zip") is None


def test_pc_installer_scan_excludes_android_packages_in_dist(tmp_path, monkeypatch):
    serve = _load_serve_module()
    dist_dir = tmp_path / "packaging" / "dist"
    dist_dir.mkdir(parents=True)
    monkeypatch.setattr(serve, "DIST_DIR", str(dist_dir))

    installer = dist_dir / "QLH-Edge-Inference-Setup-v0.1.7.exe"
    full_apk = dist_dir / "QLH-Inference-v0.1.7-full-release.apk"
    lite_apk = dist_dir / "QLH-Inference-v0.1.7-lite-release.apk"
    installer.write_bytes(b"exe")
    full_apk.write_bytes(b"apk")
    lite_apk.write_bytes(b"apk")

    entries = serve._scan_pc_installers()

    assert entries == [
        ("QLH-Edge-Inference-Setup-v0.1.7.exe", "/QLH-Edge-Inference-Setup-v0.1.7.exe", str(installer))
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

    full_apk = dist_dir / "QLH-Inference-v0.1.7-full-release.apk"
    lite_apk = dist_dir / "QLH-Inference-v0.1.7-lite-release.apk"
    gradle_apk = gradle_release_dir / "app-full-release.apk"
    debug_apk = gradle_debug_dir / "app-full-debug.apk"
    full_apk.write_bytes(b"full")
    lite_apk.write_bytes(b"lite")
    gradle_apk.write_bytes(b"gradle")
    debug_apk.write_bytes(b"debug")

    entries = serve._scan_android_downloads()

    assert entries == [
        ("QLH-Inference-v0.1.7-full-release.apk", "/QLH-Inference-v0.1.7-full-release.apk", str(full_apk)),
        ("QLH-Inference-v0.1.7-lite-release.apk", "/QLH-Inference-v0.1.7-lite-release.apk", str(lite_apk)),
        (
            "android/app/build/outputs/apk/full/release/app-full-release.apk",
            "/android/apk/full/release/app-full-release.apk",
            str(gradle_apk),
        ),
    ]
