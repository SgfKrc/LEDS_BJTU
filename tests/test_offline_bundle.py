"""离线资产一键整合包打包器（M1）单测。

用临时目录 + monkeypatch 的 REPO_ROOT/OUT_DIR 构造合成资产 fixture，
覆盖：PC/安卓双版打包、manifest/SHA 结构、缺失资产 fail-closed 与获取命令、
篡改拒绝、--verify 闭环。不触碰真实 models/ 与 build/。
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_offline_bundle as b  # noqa: E402


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    """构造合成资产树 + 重定向 REPO_ROOT/OUT_DIR。"""
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    # 资产
    (repo / "models" / "qwen-1_8b-chat").mkdir(parents=True)
    (repo / "models" / "qwen-1_8b-chat" / "model.safetensors").write_bytes(b"fake-sd" * 100)
    (repo / "models" / "Qwen-1_8B-Chat.Q4_K_M.gguf").write_bytes(b"fake-gguf-18b" * 100)
    (repo / "models" / "qwen3-4b-gguf").mkdir(parents=True)
    (repo / "models" / "qwen3-4b-gguf" / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"fake-gguf-3b" * 100)
    # gemma4 lock
    gm = repo / "models" / "gemma4-native"
    gm.mkdir(parents=True)
    (gm / "model.gguf").write_bytes(b"fake-gemma" * 100)
    (gm / "mmproj-f16.gguf").write_bytes(b"fake-mmproj" * 50)
    (gm / "gemma4-native.lock.json").write_text(json.dumps({
        "artifacts": {"model.gguf": {"size": 1000}, "mmproj-f16.gguf": {"size": 500}},
    }), encoding="utf-8")
    # sd15 离线包 + 侧车
    sd = repo / "build" / "sd15-assets"
    sd.mkdir(parents=True)
    pkg = sd / "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip"
    pkg.write_bytes(b"fake-sd15-pkg" * 100)
    pkg_sha = hashlib.sha256(pkg.read_bytes()).hexdigest()
    (sd / (pkg.name + ".sha256")).write_text(f"{pkg_sha}  {pkg.name}\n", encoding="utf-8")

    monkeypatch.setattr(b, "REPO_ROOT", repo)
    monkeypatch.setattr(b, "OUT_DIR", out)
    return repo, out


def test_missing_lists_fetch_commands(fake_assets):
    repo, _ = fake_assets
    # 删一个资产 -> --missing 应列出 + 获取命令
    (repo / "models" / "qwen3-4b-gguf" / "Qwen3-4B-Q4_K_M.gguf").unlink()
    missing_pc = b._missing_assets(b.PC_ASSETS)
    assert "qwen3-4b-gguf" in missing_pc
    assert "下载" in b.ASSETS["qwen3-4b-gguf"]["fetch"]


def test_missing_all_present(fake_assets):
    assert b._missing_assets(b.PC_ASSETS) == []
    assert b._missing_assets(b.ANDROID_ASSETS) == []


def test_pc_bundle_structure_and_manifest(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc")
    assert bundle.name == f"qlh-models-pc-{b.BUNDLE_VERSION}.zip"
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        root = bundle.name.removesuffix(".zip")
        # 关键文件都在
        assert f"{root}/MANIFEST.json" in names
        assert f"{root}/CHECKSUMS.sha256" in names
        assert f"{root}/README-导入说明.md" in names
        assert any("qwen-1_8b-chat/model.safetensors" in n for n in names)
        assert any("gemma4-native/model.gguf" in n for n in names)
        assert any("sd15-assets/QLH-SD15-Assets-" in n for n in names)
        # manifest 字段
        manifest = json.loads(zf.read(f"{root}/MANIFEST.json"))
        assert manifest["variant"] == "pc"
        assert set(manifest["asset_ids"]) == set(b.PC_ASSETS)
        # 所有 manifest 文件都在包内
        for f in manifest["files"]:
            assert f"{root}/{f['path']}" in names
            assert f["sha256"] and f["size"] > 0


def test_android_bundle_only_gguf(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android")
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        # 只有 GGUF + 清单文件
        assert any("Qwen-1_8B-Chat.Q4_K_M.gguf" in n for n in names)
        assert any("Qwen3-4B-Q4_K_M.gguf" in n for n in names)
        assert not any("safetensors" in n or "gemma" in n.lower()
                       or "sd15" in n for n in names)


def test_verify_roundtrip(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc")
    b._verify_bundle(bundle)  # 不应抛异常


def test_verify_detects_tamper(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc")
    # 篡改包内一个文件后 verify 必须失败
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(tmp)
        root = tmp / bundle.name.removesuffix(".zip")
        target = next(root.rglob("model.safetensors"))
        target.write_bytes(b"tampered")
        tampered = tmp / "tampered.zip"
        with zipfile.ZipFile(tampered, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(root.rglob("*")):
                zf.write(p, p.relative_to(tmp).as_posix())
        with pytest.raises(b.BundleError, match="SHA 不匹配|缺文件"):
            b._verify_bundle(tampered)


def test_sd15_sidecar_mismatch_fails_closed(fake_assets):
    repo, _ = fake_assets
    sidecar = repo / "build" / "sd15-assets" / "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip.sha256"
    sidecar.write_text("deadbeef" * 8 + "  x.zip\n", encoding="utf-8")
    with pytest.raises(b.BundleError, match="SHA 不匹配"):
        b._collect_asset("sd15-assets")


def test_missing_asset_fails_closed_with_fetch(fake_assets):
    repo, _ = fake_assets
    (repo / "models" / "Qwen-1_8B-Chat.Q4_K_M.gguf").unlink()
    with pytest.raises(b.BundleError, match="获取"):
        b._collect_asset("qwen18b-gguf")
