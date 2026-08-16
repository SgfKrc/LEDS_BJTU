#!/usr/bin/env python3
"""离线资产一键整合包打包器（M1/M2）——按《离线资产一键整合包设计》。

一键把本机已下载的离线资产打包成两个整合包：
  --pc       PC 版（全量：Qwen 双格式 + Qwen3-4B + Gemma4 原生 + SD 五资产包）
  --android  安卓版（纯 GGUF：Qwen-1.8B Q4 + Qwen3-4B Q4，SAF 目录直接可用）

格式：
  --format 7z（默认，7-Zip LZMA2，体积最小；增量追加免复制 30GB）
  --format zip（ZIP_STORED 快速模式，不压缩）

原则：
- 只读资产源（models/、build/sd15-assets/），不重新下载、不新增事实源；
- 逐文件 SHA-256 校验（复用既有 lock/manifest/SHA 侧车）；
- 缺失资产 fail-closed 并给出获取命令（--missing 只列缺失）。

用法：
  python scripts/build_offline_bundle.py --pc
  python scripts/build_offline_bundle.py --android
  python scripts/build_offline_bundle.py --all --verify
  python scripts/build_offline_bundle.py --missing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "build" / "offline-bundles"
BUNDLE_VERSION = "v1"

SEVEN_ZIP_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
)

# 资产注册表：asset_id -> 相对路径（目录或文件）+ 说明
ASSETS: dict[str, dict] = {
    "qwen18b-safetensors": {
        "path": "models/qwen-1_8b-chat", "kind": "dir", "scope": "pc",
        "desc": "Qwen-1.8B-Chat Safetensors（默认模型，独显/分布式）",
        "fetch": "python -c \"from modelscope import snapshot_download; "
                 "snapshot_download('Qwen/Qwen-1.8B-Chat', local_dir='models/qwen-1_8b-chat')\"",
    },
    "qwen18b-gguf": {
        "path": "models/Qwen-1_8B-Chat.Q4_K_M.gguf", "kind": "file", "scope": "both",
        "desc": "Qwen-1.8B-Chat GGUF Q4_K_M（CPU/Android）",
        "fetch": "huggingface-cli download RichardErkhov/Qwen_-_Qwen-1_8B-Chat-gguf "
                 "Qwen-1_8B-Chat-Q4_K_M.gguf --local-dir models/",
    },
    "qwen3-4b-gguf": {
        "path": "models/qwen3-4b-gguf", "kind": "dir", "scope": "both",
        "desc": "Qwen3-4B GGUF Q4_K_M（判题模型/实验）",
        "fetch": "MODEL-TOOLS 受管下载（Qwen/Qwen3-4B-GGUF Q4_K_M）",
    },
    "gemma4-native": {
        "path": "models/gemma4-native", "kind": "lock", "scope": "pc",
        "lock_file": "models/gemma4-native/gemma4-native.lock.json",
        "desc": "Gemma 4 12B 原生绑定（GGUF + mmproj，图像理解）",
        "fetch": "按 models/gemma4-native/gemma4-native.lock.json 受管下载",
    },
    "sd15-assets": {
        "path": "build/sd15-assets", "kind": "sd15", "scope": "pc",
        "desc": "SD 1.5 五资产离线包（原版/90s/IP-Adapter/inpaint/InstructPix2Pix）",
        "fetch": "python scripts/package_sd15_assets.py --asset-id <id> "
                 "--output-dir build/sd15-assets（验收后清理过，需重新打包）",
    },
}

PRUNE_SD_ZIPS = False  # --prune-sd-zips 时置 True
PRUNE_DRY_RUN = False  # --prune-dry-run：只打印将删文件不执行
_PRUNE_WARNED = False


def _set_prune_warned() -> None:
    global _PRUNE_WARNED
    _PRUNE_WARNED = True

# 临时目录放 D 盘（系统盘只有 ~11GB，verify/staging 会写几十 GB）
TMP_BASE = Path(os.environ.get("QLH_BUNDLE_TMP", "D:/qlh_tmp"))
try:
    TMP_BASE.mkdir(parents=True, exist_ok=True)
except OSError:
    TMP_BASE = Path(tempfile.gettempdir())  # 兜底系统临时目录


PC_ASSETS = ("qwen18b-safetensors", "qwen18b-gguf", "qwen3-4b-gguf",
             "gemma4-native", "sd15-assets")
ANDROID_ASSETS = ("qwen18b-gguf", "qwen3-4b-gguf")


class BundleError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(asset_id: str) -> Path:
    return (REPO_ROOT / ASSETS[asset_id]["path"]).resolve()


def _collect_asset(asset_id: str, staging_sd: Path | None = None) -> dict:
    """收集资产文件清单（src=仓库绝对、path=bundle 内相对），校验 SHA。

    sd15 资产会解包去重到 staging_sd（同 SHA 文件只写一份，重复条目记
    dedup_of）；其余资产直接引用仓库路径。
    """
    spec = ASSETS[asset_id]
    raw_root = REPO_ROOT / spec["path"]
    root = raw_root.resolve()  # 解析 junction 到真实位置（读文件用）
    is_archive_junction = raw_root.is_junction()  # 必须在 resolve 前检测
    files: list[dict] = []

    def add(path_in_bundle: str, src: Path) -> None:
        files.append({"src": str(src), "path": path_in_bundle,
                      "size": src.stat().st_size, "sha256": _sha256_file(src)})

    if spec["kind"] == "file":
        if not root.is_file():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        add(f"models/{root.name}", root)
    elif spec["kind"] == "dir":
        if not root.is_dir():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        for fp in sorted(root.rglob("*")):
            if fp.is_file():
                add(f"{spec['path']}/{fp.relative_to(root).as_posix()}", fp)
        if not files:
            raise BundleError(f"缺失资产 {asset_id}: {root} 为空\n  获取: {spec['fetch']}")
    elif spec["kind"] == "lock":
        lock = root / "gemma4-native.lock.json"
        if not lock.is_file():
            raise BundleError(f"缺失资产 {asset_id}: {lock}\n  获取: {spec['fetch']}")
        lock_data = json.loads(lock.read_text(encoding="utf-8"))
        declared = lock_data.get("artifacts") or {}
        for name, info in declared.items():
            filename = (info.get("filename") if isinstance(info, dict)
                        else None) or name
            fp = root / filename
            if not fp.is_file():
                raise BundleError(f"缺失资产 {asset_id} 文件 {filename}: {fp}")
            actual = _sha256_file(fp)
            declared_sha = info.get("sha256") if isinstance(info, dict) else None
            if declared_sha and actual != declared_sha:
                raise BundleError(
                    f"资产 {asset_id} 文件 {filename} SHA 不匹配"
                    f"（lock {declared_sha} != 实际 {actual}）")
            add(f"models/gemma4-native/{filename}", fp)
    elif spec["kind"] == "sd15":
        if not root.is_dir():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        if staging_sd is None:
            raise BundleError("sd15 资产需要 staging 目录（去重重组）")
        seen_sha: dict[str, str] = {}  # sha -> 已存储的 bundle 路径
        for fp in sorted(root.glob("*.zip")):
            sidecar = fp.with_suffix(".zip.sha256")
            declared_sha = None
            if sidecar.is_file():
                declared_sha = sidecar.read_text(encoding="utf-8").strip().split()[0]
            actual = _sha256_file(fp)
            if declared_sha and actual != declared_sha:
                raise BundleError(
                    f"资产 {asset_id} 包 {fp.name} SHA 不匹配（{declared_sha} != {actual}）")
            with zipfile.ZipFile(fp) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.startswith("models/"):
                        continue
                    bundle_path = info.filename
                    entry_sha = hashlib.sha256()
                    with zf.open(info) as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            entry_sha.update(chunk)
                    digest = entry_sha.hexdigest()
                    if digest in seen_sha:
                        # 跨包重复：只记映射，不重复写
                        files.append({
                            "src": "", "path": bundle_path,
                            "size": info.file_size, "sha256": digest,
                            "dedup_of": seen_sha[digest],
                        })
                        continue
                    seen_sha[digest] = bundle_path
                    staging_target = staging_sd / bundle_path
                    staging_target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as fh, open(staging_target, "wb") as out:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            out.write(chunk)
                    files.append({
                        "src": str(staging_target), "path": bundle_path,
                        "size": info.file_size, "sha256": digest,
                    })
            if PRUNE_SD_ZIPS:
                # 源 zip 可重建（package_sd15_assets.py），处理完即删以省空间
                if is_archive_junction:
                    # 归档 junction（如 H:/qlh-archives）只读：绝不删归档文件
                    if not _PRUNE_WARNED:
                        print("  [prune] 源为归档 junction，只读保护——跳过删除")
                        _set_prune_warned()
                elif PRUNE_DRY_RUN:
                    print(f"  [prune-dry-run] 将删除: {fp.name} + 侧车")
                else:
                    fp.unlink()
                    sidecar.unlink(missing_ok=True)
        if not files:
            raise BundleError(
                f"资产 {asset_id}: {root} 无离线包（先跑 download_sd15 / 导入离线包）")
    else:  # pragma: no cover
        raise BundleError(f"未知资产 kind: {spec['kind']}")

    return {"asset_id": asset_id, "desc": spec["desc"], "files": files}


def _missing_assets(scope: tuple[str, ...]) -> list[str]:
    missing = []
    for asset_id in scope:
        try:
            if ASSETS[asset_id]["kind"] == "sd15":
                # 缺失检查只验证 zip 存在（深解包去重校验在打包时做）
                root = _resolve(asset_id)
                if not root.is_dir() or not list(root.glob("*.zip")):
                    raise BundleError("sd15 无离线包")
            else:
                _collect_asset(asset_id)
        except BundleError:
            missing.append(asset_id)
    return missing


def _seven_zip() -> str | None:
    exe = shutil.which("7z") or shutil.which("7za")
    if exe:
        return exe
    for cand in SEVEN_ZIP_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None


def _bundle_root_name(bundle_path: Path) -> str:
    return bundle_path.name.removesuffix(".7z").removesuffix(".zip")


def _build_bundle(scope: tuple[str, ...], variant: str,
                  fmt: str = "7z", volume: str | None = None,
                  threads: int = 4) -> Path:
    manifest_files: list[dict] = []
    checksum_lines: list[str] = []
    collected_all: list[dict] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ext = ".7z" if fmt == "7z" else ".zip"
    bundle_path = OUT_DIR / f"qlh-models-{variant}-{BUNDLE_VERSION}{ext}"

    staging_sd = None
    if "sd15-assets" in scope:
        staging_sd = OUT_DIR / f".staging-{variant}"
        if staging_sd.exists():
            shutil.rmtree(staging_sd)
        staging_sd.mkdir(parents=True)

    for asset_id in scope:
        collected = _collect_asset(asset_id, staging_sd=staging_sd)
        collected_all.append(collected)
        for f in collected["files"]:
            manifest_files.append({k: v for k, v in f.items() if k != "src"})
            if not f.get("dedup_of"):
                checksum_lines.append(f"{f['sha256']}  {f['path']}")

    manifest = {
        "schema_version": "qlh.offline_bundle.v1",
        "bundle_version": BUNDLE_VERSION,
        "variant": variant,
        "format": fmt,
        "volume": volume or "",
        "asset_ids": list(scope),
        "files": manifest_files,
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=1,
                               sort_keys=True)

    try:
        if fmt == "7z":
            _build_bundle_7z(bundle_path, collected_all, manifest_json,
                             checksum_lines, variant, volume, threads)
        else:
            _build_bundle_zip(bundle_path, collected_all,
                              manifest_json, checksum_lines, variant)
    finally:
        # staging 是中间产物：打包完（或失败）立即删，降低磁盘峰值
        if staging_sd is not None and staging_sd.exists():
            shutil.rmtree(staging_sd, ignore_errors=True)
    return bundle_path


def _build_bundle_zip(bundle_path, collected_all, manifest_json,
                      checksum_lines, variant) -> None:
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_STORED) as zf:
        for collected in collected_all:
            for f in collected["files"]:
                if f.get("dedup_of"):
                    continue  # 去重条目由 restore 阶段恢复
                zf.write(f["src"], f["path"])
        zf.writestr("MANIFEST.json", manifest_json)
        zf.writestr("CHECKSUMS.sha256",
                    "\n".join(checksum_lines) + "\n")
        zf.writestr("README-导入说明.md",
                    _import_readme(variant, manifest_json))


def _build_bundle_7z(bundle_path, collected_all, manifest_json,
                     checksum_lines, variant, volume=None,
                     threads: int = 4) -> None:
    seven_zip = _seven_zip()
    if seven_zip is None:
        raise BundleError("7z 模式需要 7-Zip（winget install 7zip.7zip）")
    root = _bundle_root_name(bundle_path)
    # 分段增量追加（免复制）：models/（cwd=REPO_ROOT）；sd15 staging
    # （cwd=staging）；清单（临时目录）
    parts: list[tuple[Path, list[str]]] = []
    models_files = [f for c in collected_all for f in c["files"]
                    if f["path"].startswith("models/") and not f.get("dedup_of")]
    if models_files:
        parts.append((REPO_ROOT, [f["path"] for f in models_files]))
    sd15_files = [f for c in collected_all for f in c["files"]
                  if f["path"].startswith("models/sd15-") and not f.get("dedup_of")]
    if sd15_files:
        parts.append((OUT_DIR / f".staging-{variant}", [f["path"] for f in sd15_files]))
    if bundle_path.exists():
        bundle_path.unlink()
    with tempfile.TemporaryDirectory(dir=TMP_BASE) as td:
        staging = Path(td)
        (staging / "MANIFEST.json").write_text(manifest_json, encoding="utf-8")
        (staging / "CHECKSUMS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8")
        (staging / "README-导入说明.md").write_text(
            _import_readme(variant, manifest_json), encoding="utf-8")
        parts.append((staging, ["MANIFEST.json", "CHECKSUMS.sha256",
                                "README-导入说明.md"]))
        for cwd, paths in parts:
            # 单卷三段追加（分卷后置物理切分——7z 不支持对分卷追加）
            cmd = [seven_zip, "a", "-t7z", "-mx=1", f"-mmt={threads}",
                   "-y", str(bundle_path), *paths]
            r = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise BundleError(
                    f"7z 打包失败: {r.stderr[-300:] or r.stdout[-300:]}")
    if volume:
        _split_volumes(bundle_path, volume)


def _parse_volume_size(volume: str) -> int:
    """'4g'/'2g'/'512m' -> 字节。"""
    m = re.fullmatch(r"(\d+)([gmk])", volume.strip().lower())
    if not m:
        raise BundleError(f"分卷大小格式错误: {volume}（示例 4g/2g/512m）")
    n = int(m.group(1))
    return n * {"g": 1 << 30, "m": 1 << 20, "k": 1 << 10}[m.group(2)]


def _split_volumes(bundle_path: Path, volume: str) -> None:
    """7z 分卷 = 纯字节切割：.001 含完整 header，后续为数据段。

    单卷打包完成后按大小切块命名 name.7z.001/.002/...，删除单卷。
    """
    size = _parse_volume_size(volume)
    for stale in OUT_DIR.glob(f"{bundle_path.name}.*"):
        stale.unlink()  # 清理上次中断残留的旧分卷
    data = bundle_path.read_bytes() if bundle_path.stat().st_size < (1 << 30) else None
    if data is not None:
        # 小文件（测试）直接内存切
        parts = [data[i:i + size] for i in range(0, len(data), size)]
        for idx, chunk in enumerate(parts, 1):
            (Path(str(bundle_path) + f".{idx:03d}")).write_bytes(chunk)
    else:
        # 大文件流式切
        vol_idx = 1
        with open(bundle_path, "rb") as src:
            while True:
                chunk = src.read(size)
                if not chunk:
                    break
                (Path(str(bundle_path) + f".{vol_idx:03d}")).write_bytes(chunk)
                vol_idx += 1
    bundle_path.unlink()


def _import_readme(variant: str, manifest_json: str) -> str:
    if variant == "android":
        return (
            "# 安卓版整合包导入说明\n\n"
            f"> 版本：{BUNDLE_VERSION} | 仅含 GGUF（SAF 目录直接可用）\n\n"
            "1. 解压本包，得到 `gguf/` 目录（Qwen-1.8B Q4_K_M、Qwen3-4B Q4_K_M）\n"
            "2. 在安卓 Full 模式设置中选择该目录（SAF 授权）→ 扫描 → 选模型\n"
            "3. 校验：`sha256sum -c CHECKSUMS.sha256`\n\n"
            "说明：安卓本地只跑 GGUF（llama.cpp CPU）；判题/图像生成均走 PC 主节点远程推理。\n"
        )
    return (
        "# PC 版整合包导入说明\n\n"
        f"> 版本：{BUNDLE_VERSION} | 全量资产，解压到项目根即就位\n\n"
        "1. 解压到 QLH 项目根目录（`models/`、`sd15-assets/` 就位）\n"
        "2. 校验：`sha256sum -c CHECKSUMS.sha256`\n"
        "3. SD 资产导入：serve/图像工作区自动发现，或 `import_asset` 逐包导入\n"
        "4. 判题模型（Qwen3-4B）与 Gemma4 原生工件按 MANIFEST.json 校验后可用\n\n"
        "清单（自动生成）：\n```json\n" + manifest_json + "\n```\n"
    )


def _verify_bundle(bundle_path: Path) -> None:
    """解包到临时目录，恢复 dedup 链接后逐文件 SHA 比对（7z/分卷/zip）。"""
    with tempfile.TemporaryDirectory(dir=TMP_BASE) as td:
        tmp = Path(td)
        if bundle_path.suffix == ".7z":
            seven_zip = _seven_zip()
            if seven_zip is None:
                raise BundleError("verify 7z 需要 7-Zip")
            # 分卷时用 .001 作为入口
            target = bundle_path
            if not target.exists():
                first_vol = Path(str(target) + ".001")
                if first_vol.exists():
                    target = first_vol
            r = subprocess.run(
                [seven_zip, "x", "-y", f"-o{tmp}", str(target)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace")
            if r.returncode != 0:
                raise BundleError(f"7z 解包失败: {r.stderr[-300:]}")
            root_dir = tmp
        else:
            with zipfile.ZipFile(bundle_path) as zf:
                zf.extractall(tmp)
                root_dir = tmp
        # 先恢复 dedup 链接（MANIFEST 里的 dedup_of 映射）
        manifest_path = root_dir / "MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            restored = _restore_dedup(root_dir, manifest)
            if restored:
                print(f"  verify 恢复 {restored} 个去重链接")
        checksums = (root_dir / "CHECKSUMS.sha256").read_text(
            encoding="utf-8").splitlines()
        checked = 0
        for line in checksums:
            sha, _, rel = line.partition("  ")
            p = root_dir / rel
            if not p.is_file():
                raise BundleError(f"verify 失败：解包缺文件 {rel}")
            if _sha256_file(p) != sha:
                raise BundleError(f"verify 失败：{rel} SHA 不匹配")
            checked += 1
        print(f"  verify OK：{checked} 文件一致")


def _restore_dedup(root_dir: Path, manifest: dict) -> int:
    """按 MANIFEST 的 dedup_of 恢复链接/复制（Windows hardlink、posix symlink）。"""
    restored = 0
    for f in manifest.get("files", []):
        dedup_of = f.get("dedup_of")
        if not dedup_of:
            continue
        target = root_dir / f["path"]
        source = root_dir / dedup_of
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            try:
                os.link(source, target)  # 同卷 hardlink
            except OSError:
                shutil.copy2(source, target)
        else:
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        restored += 1
    return restored


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线资产一键整合包打包器")
    ap.add_argument("--pc", action="store_true", help="构建 PC 版")
    ap.add_argument("--android", action="store_true", help="构建安卓版")
    ap.add_argument("--all", action="store_true", help="构建双版")
    ap.add_argument("--verify", action="store_true", help="打包后自动解包校验")
    ap.add_argument("--missing", action="store_true",
                    help="只列出缺失资产与获取命令（不打包）")
    ap.add_argument("--format", choices=("7z", "zip"), default="7z",
                    help="打包格式（默认 7z，体积最小；zip=ZIP_STORED 快速）")
    ap.add_argument("--volume", metavar="SIZE", default=None,
                    help="7z 分卷大小（如 4g/2g；PC 版建议 4g 便于 U 盘/局域网分批）")
    ap.add_argument("--prune-sd-zips", action="store_true",
                    help="SD 源 zip 处理完即删（可重建；省磁盘，峰值约降 15GB）")
    ap.add_argument("--prune-dry-run", action="store_true",
                    help="只打印将删除的 SD 源 zip，不执行删除（先看再删）")
    ap.add_argument("--threads", type=int, default=4, metavar="N",
                    help="7z 压缩线程数（默认 4；本机 20 核默认会吃满，限制后留余量）")
    args = ap.parse_args(argv)
    if args.prune_sd_zips:
        global PRUNE_SD_ZIPS, PRUNE_DRY_RUN
        PRUNE_SD_ZIPS = True
        PRUNE_DRY_RUN = args.prune_dry_run

    if args.prune_dry_run:
        # 独立预览：列出 SD 源将删除的 zip（不打包、不删除）
        root = _resolve("sd15-assets")
        raw = REPO_ROOT / ASSETS["sd15-assets"]["path"]
        if raw.is_junction():
            print(f"[prune-dry-run] SD 源是归档 junction（{root}）——只读保护，永不删除")
        else:
            zips = sorted((root if root.is_dir() else Path()).glob("*.zip"))
            print(f"[prune-dry-run] 将删除 {len(zips)} 个可重建 SD 源 zip：")
            for z in zips:
                print(f"  - {z.name}（{z.stat().st_size / 1e9:.1f} GB）")
        return 0

    if args.missing:
        for scope, variant in ((PC_ASSETS, "pc"), (ANDROID_ASSETS, "android")):
            missing = _missing_assets(scope)
            print(f"[{variant}] 缺失 {len(missing)} 项：")
            for asset_id in missing:
                spec = ASSETS[asset_id]
                print(f"  - {asset_id}（{spec['desc']}）\n    获取: {spec['fetch']}")
            if not missing:
                print(f"  - 全部齐备 ✓")
        return 0

    if args.all:
        args.pc = args.android = True
    if not (args.pc or args.android):
        ap.error("至少指定 --pc / --android / --all / --missing 之一")

    if args.pc:
        print(f"构建 PC 版（全量，{args.format}）...")
        bundle = _build_bundle(PC_ASSETS, "pc", args.format, args.volume,
                              args.threads)
        _print_bundle_result(bundle, args.volume)
        if args.verify:
            _verify_bundle(bundle)
    if args.android:
        print(f"构建安卓版（纯 GGUF，{args.format}）...")
        bundle = _build_bundle(ANDROID_ASSETS, "android", args.format,
                              args.volume, args.threads)
        _print_bundle_result(bundle, args.volume)
        if args.verify:
            _verify_bundle(bundle)
    return 0


def _print_bundle_result(bundle_path: Path, volume: str | None) -> None:
    """打印产物信息（分卷后单卷文件已删，需按卷统计）。"""
    if volume:
        vols = sorted(OUT_DIR.glob(f"{bundle_path.name}.*"))
        total = sum(v.stat().st_size for v in vols)
        print(f"  -> {bundle_path.name} 分卷 {len(vols)} 个"
              f"（共 {total / 1e9:.1f} GB，每卷 {volume}）")
    else:
        print(f"  -> {bundle_path}（{bundle_path.stat().st_size / 1e9:.1f} GB）")


if __name__ == "__main__":
    sys.exit(main())
