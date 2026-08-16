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

_TEST_TMP = Path("D:/qlh_tmp")

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
    # sd15 离线包（真 zip，含 models/ 顶层）+ 侧车；两个包共享一个文件（测去重）
    sd = repo / "build" / "sd15-assets"
    sd.mkdir(parents=True)
    shared_bytes = b"shared-vae" * 50
    for pkg_name, extra in (
            ("QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip", b"orig-unet"),
            ("QLH-SD15-Assets-sd15_inpaint_v1-v0.1.0.zip", b"inpaint-unet")):
        pkg = sd / pkg_name
        with zipfile.ZipFile(pkg, "w") as zf:
            zf.writestr("models/sd15-original-v1/vae/model.fp16.safetensors"
                        if "original" in pkg_name
                        else "models/sd15-inpaint-v1/vae/model.fp16.safetensors",
                        shared_bytes)
            zf.writestr("models/sd15-original-v1/unet/model.fp16.safetensors"
                        if "original" in pkg_name
                        else "models/sd15-inpaint-v1/unet/model.fp16.safetensors",
                        extra)
        pkg_sha = hashlib.sha256(pkg.read_bytes()).hexdigest()
        (sd / (pkg_name + ".sha256")).write_text(
            f"{pkg_sha}  {pkg_name}\n", encoding="utf-8")

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
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    assert bundle.name == f"qlh-models-pc-{b.BUNDLE_VERSION}.zip"
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        # 关键文件都在（解压到项目根即就位，无顶层目录）
        assert "MANIFEST.json" in names
        assert "CHECKSUMS.sha256" in names
        assert "README-导入说明.md" in names
        assert any("qwen-1_8b-chat/model.safetensors" in n for n in names)
        assert any("gemma4-native/model.gguf" in n for n in names)
        # SD 已解包重组到 models/sd15-*/（不再收 zip）；共享文件只存一份
        assert any("models/sd15-original-v1/unet/model.fp16.safetensors"
                   in n for n in names)
        assert not any("sd15-assets/" in n for n in names)
        # manifest 字段
        manifest = json.loads(zf.read("MANIFEST.json"))
        assert manifest["variant"] == "pc"
        assert set(manifest["asset_ids"]) == set(b.PC_ASSETS)
        # 所有非去重条目都在包内
        for f in manifest["files"]:
            if f.get("dedup_of"):
                continue  # 去重条目由 restore 恢复，不在包内
            assert f["path"] in names
            assert f["sha256"] and f["size"] > 0


def test_android_bundle_only_gguf(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="zip")
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        # 只有 GGUF + 清单文件
        assert any("Qwen-1_8B-Chat.Q4_K_M.gguf" in n for n in names)
        assert any("Qwen3-4B-Q4_K_M.gguf" in n for n in names)
        assert not any("safetensors" in n or "gemma" in n.lower()
                       or "sd15" in n for n in names)


def test_verify_roundtrip(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    b._verify_bundle(bundle)  # 不应抛异常


@pytest.mark.skipif(b._seven_zip() is None, reason="7-Zip 未安装")
def test_7z_build_and_verify_roundtrip(fake_assets):
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="7z")
    assert bundle.suffix == ".7z"
    b._verify_bundle(bundle)  # 7z 解包逐文件比对，不应抛异常


def test_sd15_dedup_stores_shared_file_once(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
        vae_orig = "models/sd15-original-v1/vae/model.fp16.safetensors"
        vae_inp = "models/sd15-inpaint-v1/vae/model.fp16.safetensors"
        # 共享 vae 只存一次（打包顺序决定主/副），另一个记 dedup_of
        assert (vae_orig in names) != (vae_inp in names), "vae 应只存一份"
        manifest = json.loads(zf.read("MANIFEST.json"))
        entries = {f["path"]: f for f in manifest["files"]}
        stored, shadow = ((vae_orig, vae_inp) if vae_orig in names
                          else (vae_inp, vae_orig))
        assert entries[shadow]["dedup_of"] == stored
        assert "dedup_of" not in entries[stored]


def test_verify_restores_dedup_links(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    b._verify_bundle(bundle)  # 内部会 restore dedup 后逐文件校验


def test_volume_split_and_verify(fake_assets):
    # 小 bundle 用 512k 卷切分 -> verify 用 .001 入口
    bundle = b._build_bundle(b.ANDROID_ASSETS, "android", fmt="7z",
                             volume="512k")
    assert not bundle.exists()  # 单卷已删
    vols = sorted(p for p in b.OUT_DIR.glob("qlh-models-android-v1.7z.*"))
    assert vols and vols[0].name.endswith(".001")
    assert all(v.stat().st_size <= 512 * 1024 for v in vols)
    b._verify_bundle(bundle)  # 自动找 .001 解包比对


def test_parse_volume_size():
    assert b._parse_volume_size("4g") == 4 * (1 << 30)
    assert b._parse_volume_size("512m") == 512 * (1 << 20)
    with pytest.raises(b.BundleError):
        b._parse_volume_size("xyz")


def test_verify_detects_tamper(fake_assets):
    bundle = b._build_bundle(b.PC_ASSETS, "pc", fmt="zip")
    # 篡改包内一个文件后 verify 必须失败
    with tempfile.TemporaryDirectory(dir=_TEST_TMP) as td:
        tmp = Path(td)
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(tmp)
        target = next(Path(tmp).rglob("model.safetensors"))
        target.write_bytes(b"tampered")
        tampered = tmp / "tampered.zip"
        with zipfile.ZipFile(tampered, "w", zipfile.ZIP_STORED) as zf:
            for p in sorted(Path(tmp).rglob("*")):
                if p == tampered:
                    continue  # 不把输出包自己压进去
                zf.write(p, p.relative_to(tmp).as_posix())
        with pytest.raises(b.BundleError, match="SHA 不匹配|缺文件"):
            b._verify_bundle(tampered)


def test_sd15_sidecar_mismatch_fails_closed(fake_assets):
    repo, out = fake_assets
    sidecar = repo / "build" / "sd15-assets" / "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip.sha256"
    sidecar.write_text("deadbeef" * 8 + "  x.zip\n", encoding="utf-8")
    staging = out / ".staging-test"
    staging.mkdir(parents=True)
    with pytest.raises(b.BundleError, match="SHA 不匹配"):
        b._collect_asset("sd15-assets", staging_sd=staging)


def _make_sd15_package(dirpath: Path, pkg_name: str) -> Path:
    """构造最小 SD zip + 侧车（供独立测试用）。"""
    import hashlib as _h
    import zipfile as _zf
    pkg = dirpath / pkg_name
    with _zf.ZipFile(pkg, "w") as zf:
        zf.writestr("models/sd15-original-v1/unet/model.fp16.safetensors", b"x" * 10)
    (dirpath / (pkg_name + ".sha256")).write_text(
        _h.sha256(pkg.read_bytes()).hexdigest() + "  " + pkg_name + chr(10),
        encoding="utf-8")
    return pkg


def test_prune_skips_junction_archive(tmp_path, monkeypatch):
    """归档 junction 源：--prune 绝不删除归档文件。"""
    import os
    import shutil as _sh
    import subprocess as _sp
    if os.name != "nt":
        pytest.skip("junction 仅 Windows")
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    real = repo / "build" / "sd15-assets"
    real.mkdir(parents=True)
    _make_sd15_package(real, "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip")
    # 归档备份 + junction 替换源目录
    archive = tmp_path / "archive-backup"
    archive.mkdir()
    for z in real.glob("*.zip"):
        (archive / z.name).write_bytes(z.read_bytes())
    _sh.rmtree(real)
    r = _sp.run(["cmd", "/c", "mklink", "/J",
                 str(real).replace("/", "\\"), str(archive).replace("/", "\\")],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert real.is_junction(), r.stderr
    monkeypatch.setattr(b, "REPO_ROOT", repo)
    monkeypatch.setattr(b, "OUT_DIR", out)
    b.PRUNE_SD_ZIPS = True
    try:
        b._collect_asset("sd15-assets", staging_sd=out / ".staging-test")
    finally:
        b.PRUNE_SD_ZIPS = False
    assert list(archive.glob("*.zip")), "归档被 prune 删除了！"


def test_prune_dry_run_does_not_delete(tmp_path, monkeypatch):
    """dry-run：只打印将删列表，不执行删除。"""
    repo = tmp_path / "repo"
    out = tmp_path / "out"
    real = repo / "build" / "sd15-assets"
    real.mkdir(parents=True)
    _make_sd15_package(real, "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip")
    monkeypatch.setattr(b, "REPO_ROOT", repo)
    monkeypatch.setattr(b, "OUT_DIR", out)
    b.PRUNE_SD_ZIPS = True
    b.PRUNE_DRY_RUN = True
    try:
        b._collect_asset("sd15-assets", staging_sd=out / ".staging-test")
    finally:
        b.PRUNE_SD_ZIPS = False
        b.PRUNE_DRY_RUN = False
    target = real / "QLH-SD15-Assets-sd15_original_v1-v0.1.0.zip"
    assert target.is_file(), "dry-run 不应删除文件"


def test_missing_asset_fails_closed_with_fetch(fake_assets):
    repo, _ = fake_assets
    (repo / "models" / "Qwen-1_8B-Chat.Q4_K_M.gguf").unlink()
    with pytest.raises(b.BundleError, match="获取"):
        b._collect_asset("qwen18b-gguf")
