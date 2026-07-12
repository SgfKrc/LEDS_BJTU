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
