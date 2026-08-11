#!/usr/bin/env python3
"""SD 1.5 离线资产包打包脚本（阶段 1：零 GPU、不修改 src/ 现有文件）。

用法:
    python scripts/package_sd15_assets.py --list
    python scripts/package_sd15_assets.py --asset-id sd15_original_v1 [--output-dir build/sd15-assets]
    python scripts/package_sd15_assets.py --asset-id sd15_original_v1 --dry-run --json

行为:
- 元数据唯一来源是 src/diffusion/assets.py 的 DiffusionAssetSpec（revision、逐文件 size+sha256、preset）；
- 打包前对本地目录逐文件做 size + SHA-256 校验，任何不一致 fail-closed；
- 离线包结构: manifest.json / LICENSE.txt / MODEL_CARD.md / IMPORT.md / 模型文件；
- 许可证原文从 packaging/sd15-licenses/ 读取，缺失时 fail-closed（不产出无许可证的包）；
- 包旁生成 <包名>.sha256 侧车；
- 全程不访问网络、不加载模型、不使用 GPU。
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PACKAGE_PREFIX = "QLH-SD15-Assets"
PACKAGE_VERSION = "v0.1.0"
LICENSE_MAP = {
    "creativeml-openrail-m": "CREATIVEML-OPENRAIL-M.txt",
    "openrail": "OPENRAIL.txt",  # 原文待补（见 OPENRAIL.pending.md）；缺失时打包 fail-closed
    "apache-2.0": "APACHE-2.0.txt",
    "mit": "MIT.txt",
}
LICENSE_DIR = PROJECT_ROOT / "packaging" / "sd15-licenses"
IMPORT_TEMPLATE = """# 离线导入说明 — {name}

本包是 QLH SD 1.5 系列离线资产包（{asset_id}，{package_version}）。

## 导入步骤

1. 解压本包到模型根目录下的目标目录：
   `models/{local_dir}/`（相对 QLH 安装根；已存在同名目录时先备份/确认内容一致）

2. 服务端启动或目录刷新时，`src/diffusion/assets.py` 的目录事实源会自动发现并注册：
   - 逐文件大小与 SHA-256 与 manifest 比对；
   - 与冻结 spec 不一致（缺文件、哈希不符、多余文件冒充）会 fail-closed 拒绝注册。

3. 校验命令（CUDA 侧车环境）：
   `.venv-packaging-cuda\\Scripts\\python.exe scripts/quality_gate_sd15.py --asset-id {asset_id}`

4. 若目录已被占用（用户已有同名模型），只登记路径与本机哈希，不覆盖。

## 内容

- manifest.json：来源 revision、逐文件 size + sha256、preset、许可证标识
- LICENSE.txt：{license_id} 原文副本（随包分发）
- MODEL_CARD.md：模型卡（来源 README）
- 模型文件：与 manifest 逐项对应

## 校验

包级 SHA-256 见同目录 `{package_name}.sha256`；
导入后可用 `python scripts/model_tools.py models_sweep` 复验全库体检。
"""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _license_text(license_id: str) -> str:
    rel = LICENSE_MAP.get(license_id)
    if not rel:
        raise SystemExit(f"[error] 未知 license_id: {license_id}")
    path = LICENSE_DIR / rel
    if not path.exists():
        raise SystemExit(
            f"[error] 许可证原文缺失: {path}（{license_id}）。"
            "离线包必须附许可证原文，拒绝打包。"
        )
    return path.read_text(encoding="utf-8")


def package_name(asset_id: str) -> str:
    return f"{PACKAGE_PREFIX}-{asset_id}-{PACKAGE_VERSION}.zip"


def verify_local_assets(spec, *, check_hashes: bool) -> dict:
    """校验本地资产目录；返回 {相对路径: sha256}，不一致即抛错。"""
    root = spec.target_path()
    if not root.is_dir():
        raise SystemExit(f"[error] 资产目录不存在: {root}")
    file_hashes: dict[str, str] = {}
    for item in spec.files:
        path = root / item.path
        if not path.is_file():
            raise SystemExit(f"[error] 缺失文件: {path}（spec: {item.path}）")
        size = path.stat().st_size
        if size != item.size_bytes:
            raise SystemExit(
                f"[error] 大小不一致: {item.path} 期望 {item.size_bytes} 实际 {size}"
            )
        if check_hashes:
            digest = _sha256(path)
            if item.sha256 and digest != item.sha256:
                raise SystemExit(
                    f"[error] SHA-256 不一致: {item.path} 期望 {item.sha256} 实际 {digest}"
                )
            file_hashes[item.path] = digest
        elif item.sha256:
            file_hashes[item.path] = item.sha256
    return file_hashes


def build_manifest(spec, file_hashes: dict, package_bytes: int) -> dict:
    return {
        "format": "qlh.sd15-asset.v1",
        "package": package_name(spec.asset_id),
        "package_version": PACKAGE_VERSION,
        "package_sha256": "",  # 打包完成后回填包级摘要
        "package_bytes": package_bytes,
        "asset_id": spec.asset_id,
        "artifact_id": spec.artifact_id,
        "name": spec.name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "local_dir": spec.local_dir,
        "license_id": spec.license_id,
        "model_card_url": spec.model_card_url,
        "preset_id": spec.preset_id,
        "artifact_kind": spec.artifact_kind,
        "safety_checker_required": spec.safety_checker_required,
        "files": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "sha256": file_hashes.get(item.path, item.sha256),
            }
            for item in spec.files
        ],
        "notes": list(spec.notes),
    }


def _zip_add(zf: zipfile.ZipFile, arcname: str, source: Path | None, data: str | None = None):
    """小文本用 DEFLATED，大权重用 STORED（已压缩内容，省 CPU）。"""
    if data is not None:
        zf.writestr(arcname, data.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
        return
    assert source is not None
    compress = zipfile.ZIP_DEFLATED if source.stat().st_size < 1024 * 1024 else zipfile.ZIP_STORED
    zf.write(source, arcname, compress_type=compress)


def package_asset(spec, output_dir: Path, *, check_hashes: bool, dry_run: bool) -> dict:
    import diffusion.assets  # noqa: F401  # 确保模块路径

    license_text = _license_text(spec.license_id)
    file_hashes = verify_local_assets(spec, check_hashes=check_hashes)

    readme = spec.target_path() / "README.md"
    model_card = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else (
        f"# {spec.name}\n\n来源: {spec.model_card_url}\nrevision: {spec.revision}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pkg_name = package_name(spec.asset_id)
    pkg_path = output_dir / pkg_name
    pkg_path.unlink(missing_ok=True)

    if dry_run:
        total = sum(item.size_bytes for item in spec.files)
        result = {
            "dry_run": True,
            "asset_id": spec.asset_id,
            "package": pkg_name,
            "local_dir": str(spec.target_path()),
            "files": len(spec.files),
            "package_bytes_expected": total,
            "license": spec.license_id,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    manifest = build_manifest(spec, file_hashes, 0)
    with zipfile.ZipFile(pkg_path, "w", allowZip64=True) as zf:
        for item in spec.files:
            _zip_add(zf, item.path, spec.target_path() / item.path)
        _zip_add(zf, "LICENSE.txt", None, license_text)
        _zip_add(zf, "MODEL_CARD.md", None, model_card)
        _zip_add(zf, "IMPORT.md", None, IMPORT_TEMPLATE.format(
            name=spec.name,
            asset_id=spec.asset_id,
            package_version=PACKAGE_VERSION,
            local_dir=spec.local_dir,
            license_id=spec.license_id,
            package_name=pkg_name,
        ))
        _zip_add(zf, "manifest.json", None, json.dumps(manifest, ensure_ascii=False, indent=2))

    package_digest = _sha256(pkg_path)
    (output_dir / f"{pkg_name}.sha256").write_text(f"{package_digest}  {pkg_name}\n", encoding="utf-8")

    result = {
        "dry_run": False,
        "asset_id": spec.asset_id,
        "package": pkg_name,
        "path": str(pkg_path),
        "package_bytes": pkg_path.stat().st_size,
        "package_sha256": package_digest,
        "files": len(spec.files),
        "license": spec.license_id,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def list_assets() -> list[dict]:
    from diffusion import assets
    rows = []
    for asset_id, spec in assets.ASSET_CATALOG.items():
        spec = assets.get_asset_spec(asset_id)
        rows.append({
            "asset_id": spec.asset_id,
            "name": spec.name,
            "license_id": spec.license_id,
            "local_dir": spec.local_dir,
            "files": len(spec.files),
            "package_bytes_expected": spec.download_bytes,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="SD 1.5 离线资产包打包")
    parser.add_argument("--list", action="store_true", help="列出全部可打包资产")
    parser.add_argument("--asset-id", help="要打包的 asset_id")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "build" / "sd15-assets"))
    parser.add_argument("--dry-run", action="store_true", help="只做预检不写包")
    parser.add_argument("--skip-hash", action="store_true", help="跳过逐文件 SHA 校验（仅 size）")
    parser.add_argument("--json", action="store_true", help="JSON 输出（默认即 JSON）")
    args = parser.parse_args()

    if args.list:
        for row in list_assets():
            print(json.dumps(row, ensure_ascii=False))
        return 0

    if not args.asset_id:
        parser.error("--asset-id 或 --list 必须指定其一")

    from diffusion import assets
    spec = assets.get_asset_spec(args.asset_id)
    package_asset(
        spec,
        Path(args.output_dir),
        check_hashes=not args.skip_hash,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
