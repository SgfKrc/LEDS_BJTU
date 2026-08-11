"""SD 1.5 离线资产包打包与签名源站分类测试（阶段 1）。"""

import importlib.util
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diffusion.assets import DiffusionAssetFile, DiffusionAssetSpec  # noqa: E402


def _load_module(name: str, rel_path: str):
    module_path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


serve = _load_module("packaging_serve_for_test", "packaging/serve.py")
pack = _load_module("scripts_package_sd15_assets", "scripts/package_sd15_assets.py")


@pytest.fixture
def synthetic_spec(tmp_path, monkeypatch):
    """合成资产 spec：指向 tmp_path 下的假模型目录，frozen dataclass。"""
    import diffusion.assets as assets_mod

    local_dir = tmp_path / "models" / "sd15-synthetic-v1"
    (local_dir / "unet").mkdir(parents=True)
    (local_dir / "README.md").write_text("# Synthetic model card\nlicense: mit\n", encoding="utf-8")
    tiny = b"weight-bytes-abcdef"
    (local_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(tiny)
    (local_dir / "model_index.json").write_text("{}", encoding="utf-8")

    files = (
        DiffusionAssetFile(
            "README.md", (local_dir / "README.md").stat().st_size,
            "example/synthetic", "abc123",
            pack._sha256(local_dir / "README.md"),
        ),
        DiffusionAssetFile(
            "model_index.json", (local_dir / "model_index.json").stat().st_size,
            "example/synthetic", "abc123",
            pack._sha256(local_dir / "model_index.json"),
        ),
        DiffusionAssetFile(
            "unet/diffusion_pytorch_model.safetensors", len(tiny),
            "example/synthetic", "abc123",
            pack._sha256(local_dir / "unet" / "diffusion_pytorch_model.safetensors"),
        ),
    )
    spec = DiffusionAssetSpec(
        asset_id="sd15_synthetic_v1",
        artifact_id="synthetic-artifact",
        name="Synthetic SD 1.5",
        repo_id="example/synthetic",
        revision="abc123",
        local_dir="sd15-synthetic-v1",
        license_id="mit",
        model_card_url="https://huggingface.co/example/synthetic",
        preset_id="sd15_synthetic_v1",
        files=files,
    )
    # 资产根目录映射到 tmp_path/models（target_path() = 根 + local_dir）
    monkeypatch.setattr(assets_mod, "_app_root", lambda: tmp_path / "models")
    return spec


class TestPackageAsset:
    def test_package_builds_valid_zip(self, synthetic_spec, tmp_path):
        out = tmp_path / "out"
        result = pack.package_asset(synthetic_spec, out, check_hashes=True, dry_run=False)

        assert result["dry_run"] is False
        pkg = out / result["package"]
        assert pkg.is_file()
        sidecar = out / f"{result['package']}.sha256"
        assert sidecar.read_text(encoding="utf-8").startswith(result["package_sha256"])

        with zipfile.ZipFile(pkg) as zf:
            names = set(zf.namelist())
            assert {"manifest.json", "LICENSE.txt", "MODEL_CARD.md", "IMPORT.md"} <= names
            assert "unet/diffusion_pytorch_model.safetensors" in names
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["asset_id"] == "sd15_synthetic_v1"
            assert manifest["license_id"] == "mit"
            assert manifest["revision"] == "abc123"
            assert manifest["format"] == "qlh.sd15-asset.v1"
            assert len(manifest["files"]) == 3
            for entry in manifest["files"]:
                assert entry["sha256"]
            assert zf.read("LICENSE.txt").startswith(b"MIT License")
            assert b"Synthetic model card" in zf.read("MODEL_CARD.md")
            assert "离线导入".encode() in zf.read("IMPORT.md")
            # 包内文件与源目录逐字节一致
            for entry in manifest["files"]:
                source = tmp_path / "models" / "sd15-synthetic-v1" / entry["path"]
                assert zf.read(entry["path"]) == source.read_bytes()

    def test_missing_license_fails_closed(self, synthetic_spec, tmp_path):
        spec = replace(synthetic_spec, license_id="openrail")  # 原文待补 → 必须拒绝打包
        with pytest.raises(SystemExit, match="许可证原文缺失"):
            pack.package_asset(spec, tmp_path / "out", check_hashes=True, dry_run=False)

    def test_missing_file_fails_closed(self, synthetic_spec, tmp_path):
        (tmp_path / "models" / "sd15-synthetic-v1" / "model_index.json").unlink()
        with pytest.raises(SystemExit, match="缺失文件"):
            pack.package_asset(synthetic_spec, tmp_path / "out", check_hashes=True, dry_run=False)

    def test_tampered_hash_fails_closed(self, synthetic_spec, tmp_path):
        (tmp_path / "models" / "sd15-synthetic-v1" / "model_index.json").write_text(
            "ab", encoding="utf-8")  # 等长篡改：size 不变、SHA 变化
        with pytest.raises(SystemExit, match="SHA-256 不一致"):
            pack.package_asset(synthetic_spec, tmp_path / "out", check_hashes=True, dry_run=False)

    def test_size_mismatch_fails_closed(self, synthetic_spec, tmp_path):
        (tmp_path / "models" / "sd15-synthetic-v1" / "model_index.json").write_text(
            "{}", encoding="utf-8")  # 内容不变但重写时间不影响 size——改为追加
        with open(tmp_path / "models" / "sd15-synthetic-v1" / "model_index.json", "ab") as fh:
            fh.write(b"x")
        with pytest.raises(SystemExit, match="大小不一致"):
            pack.package_asset(synthetic_spec, tmp_path / "out", check_hashes=True, dry_run=False)

    def test_dry_run_writes_nothing(self, synthetic_spec, tmp_path):
        out = tmp_path / "out"
        result = pack.package_asset(synthetic_spec, out, check_hashes=True, dry_run=True)
        assert result["dry_run"] is True
        assert not (out / result["package"]).exists()

    def test_unknown_license_id_fails(self, synthetic_spec, tmp_path):
        spec = replace(synthetic_spec, license_id="no-such-license")
        with pytest.raises(SystemExit, match="未知 license_id"):
            pack.package_asset(spec, tmp_path / "out", check_hashes=True, dry_run=False)


class TestServeClassification:
    def test_sd15_asset_zip_classified(self):
        assert serve._classify_update_asset(
            "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip"
        ) == ("any", "any", "any", "sd15-asset")

    def test_launcher_unaffected(self):
        assert serve._classify_update_asset("QLH-Launcher-v0.1.8.1.zip") == (
            "windows", "any", "x86_64", "launcher")

    def test_installer_unaffected(self):
        assert serve._classify_update_asset("QLH-Edge-Inference-Setup-v0.1.8.1-CUDA.exe") == (
            "windows", "cuda", "x86_64", "installer")


class TestPackageName:
    def test_package_name(self):
        assert pack.package_name("sd15_original_v1") == (
            "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip")
