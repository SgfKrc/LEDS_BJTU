#!/usr/bin/env python3
"""SD 1.5 离线资产包阶段 2 验证：包级校验 + 离线导入闭环（不跑 GPU）。

用法:
    .venv-packaging-cuda\\Scripts\\python.exe scripts/validate_sd15_offline_packages.py \
        --packages-dir build/sd15-assets [--json]

对每个包:
1. 校验 .sha256 侧车与包级摘要一致；
2. 解包到干净临时目录（打包机路径）；
3. DiffusionAssetManager.import_asset 官方导入（verify_asset_directory + manifest 写入）→ valid；
4. 全程不访问网络（HF_HUB_OFFLINE=1）、不加载模型、不使用 GPU。

GPU 冒烟由 --smoke-asset 单独执行（见 validate_sd15_offline_smoke 分支，本脚本不含）。
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os_ = __import__("os")
os_.environ.setdefault("HF_HUB_OFFLINE", "1")
os_.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(pkg: Path) -> dict:
    sidecar = Path(f"{pkg}.sha256")
    if not sidecar.is_file():
        raise SystemExit(f"[error] 缺少包级侧车: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    actual = _sha256(pkg)
    if actual != expected:
        raise SystemExit(f"[error] 包级 SHA-256 不一致: {pkg.name} 期望 {expected} 实际 {actual}")

    with zipfile.ZipFile(pkg) as zf:
        names = set(zf.namelist())
        for required in ("manifest.json", "LICENSE.txt", "MODEL_CARD.md", "IMPORT.md"):
            if required not in names:
                raise SystemExit(f"[error] {pkg.name} 缺少 {required}")
        manifest = json.loads(zf.read("manifest.json"))
        asset_id = manifest["asset_id"]

    from diffusion.assets import get_asset_spec

    spec = get_asset_spec(asset_id)

    # 解包到干净临时目录后走官方导入
    with tempfile.TemporaryDirectory(prefix="qlh-sd15-import-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(pkg) as zf:
            zf.extractall(root)
        # 解包内容物与 manifest 逐文件比对（防打包/传输不一致）
        for entry in manifest["files"]:
            f = root / spec.local_dir / entry["path"]
            if not f.is_file():
                raise SystemExit(f"[error] {pkg.name} 解包缺失 {entry['path']}")
            if f.stat().st_size != entry["size_bytes"]:
                raise SystemExit(f"[error] {pkg.name} 解包大小不符 {entry['path']}")
            digest = _sha256(f)
            if digest != entry["sha256"]:
                raise SystemExit(f"[error] {pkg.name} 解包 SHA 不符 {entry['path']}")

        # 官方离线导入：verify_asset_directory + manifest 写入 + ready 通知
        from diffusion.assets import DiffusionAssetManager

        spec = get_asset_spec(asset_id)
        ready: list[tuple[str, Path]] = []
        manager = DiffusionAssetManager(
            root=root,
            on_ready=lambda s, p: ready.append((s.asset_id, Path(p))),
        )
        report = manager.import_asset(asset_id, str(root / spec.local_dir), license_accepted=True)
        if not report["valid"]:
            raise SystemExit(f"[error] {pkg.name} 离线导入校验未通过: {report}")
        assert ready and ready[0][0] == asset_id, f"{pkg.name} 未触发 ready 通知"
        return {
            "asset_id": asset_id,
            "package": pkg.name,
            "package_sha256": actual,
            "files": len(manifest["files"]),
            "import_valid": True,
            "ready_notified": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="SD 1.5 离线资产包阶段 2 验证（不含 GPU 冒烟）")
    parser.add_argument("--packages-dir", default=str(PROJECT_ROOT / "build" / "sd15-assets"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    packages_dir = Path(args.packages_dir)
    pkgs = sorted(packages_dir.glob("QLH-SD15-Assets-*.zip"))
    if not pkgs:
        raise SystemExit(f"[error] 未找到离线资产包: {packages_dir}")
    results = [validate_package(pkg) for pkg in pkgs]
    for row in results:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps({"validated": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
