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
from pathlib import PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get(
    "QLH_BUNDLE_OUT", str(REPO_ROOT / "build" / "offline-bundles")))
BUNDLE_VERSION = "v2"

SEVEN_ZIP_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
)

# 资产注册表：asset_id -> 相对路径（目录或文件）+ 说明
ASSETS: dict[str, dict] = {
    "qwen3-4b-safetensors": {
        "path": "models/qwen3-4b", "kind": "dir", "scope": "pc",
        "desc": "Qwen3-4B Safetensors（实验/训练）",
        "fetch": "ModelScope Qwen/Qwen3-4B（受管下载）",
    },
    "qwen3vl-4b-safetensors": {
        "path": "models/qwen3-vl-4b-instruct", "kind": "dir", "scope": "pc",
        "desc": "Qwen3-VL-4B Safetensors（多模态）",
        "fetch": "ModelScope Qwen/Qwen3-VL-4B-Instruct",
    },
    "qwen3vl-4b-gguf": {
        "path": "models/qwen3-vl-4b-instruct-gguf", "kind": "dir", "scope": "pc",
        "desc": "Qwen3-VL-4B GGUF + mmproj（多模态 CPU）",
        "fetch": "ModelScope Qwen/Qwen3-VL-4B-Instruct-GGUF",
    },
    "qwen35-2b-safetensors": {
        "path": "models/qwen3-5-2b", "kind": "dir", "scope": "pc",
        "desc": "Qwen3.5-2B Safetensors",
        "fetch": "ModelScope Qwen/Qwen3.5-2B",
    },
    "qwen35-2b-gguf": {
        "path": "models/qwen3-5-2b-gguf", "kind": "dir", "scope": "both",
        "desc": "Qwen3.5-2B GGUF Q4_K_M（本地转换）",
        "fetch": "本地 gguf-convert（官方无 GGUF）",
    },
    "qwen35-9b-safetensors": {
        "path": "models/qwen3-5-9b", "kind": "dir", "scope": "pc",
        "desc": "Qwen3.5-9B Safetensors（蒸馏主选学生）",
        "fetch": "ModelScope Qwen/Qwen3.5-9B",
    },
    "qwen35-9b-gguf": {
        "path": "models/qwen3-5-9b-gguf", "kind": "dir", "scope": "both",
        "desc": "Qwen3.5-9B GGUF Q4_K_M（unsloth）",
        "fetch": "ModelScope unsloth/Qwen3.5-9B-GGUF",
    },
    "gemma4-12b-safetensors": {
        "path": "models/gemma4-12b-safetensors", "kind": "dir", "scope": "pc",
        "desc": "Gemma4 12B Safetensors 权重（原生绑定对照）",
        "fetch": "ModelScope google/gemma-4-12b-it",
    },
    "deepseek-7b-safetensors": {
        "path": "models/deepseek-r1-distill-qwen-7b", "kind": "dir", "scope": "pc",
        "desc": "DeepSeek-R1-Distill-7B Safetensors",
        "fetch": "受管下载",
    },
    "deepseek-7b-gguf": {
        "path": "models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", "kind": "file",
        "scope": "pc", "desc": "DeepSeek-R1-Distill-7B GGUF Q4_K_M",
        "fetch": "受管下载",
    },
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
_CAPACITY_HEADROOM_BYTES = 64 << 20


def _set_prune_warned() -> None:
    global _PRUNE_WARNED
    _PRUNE_WARNED = True

# 临时目录放 D 盘（系统盘只有 ~11GB，verify/staging 会写几十 GB）
TMP_BASE = Path(os.environ.get("QLH_BUNDLE_TMP", "D:/qlh_tmp"))
try:
    TMP_BASE.mkdir(parents=True, exist_ok=True)
except OSError:
    TMP_BASE = Path(tempfile.gettempdir())  # 兜底系统临时目录


PC_ASSETS = (
    "qwen18b-safetensors", "qwen18b-gguf",
    "qwen3-4b-safetensors", "qwen3-4b-gguf",
    "qwen3vl-4b-safetensors", "qwen3vl-4b-gguf",
    "qwen35-2b-safetensors", "qwen35-2b-gguf",
    "qwen35-9b-safetensors", "qwen35-9b-gguf",
    "gemma4-native", "gemma4-12b-safetensors",
    "deepseek-7b-safetensors", "deepseek-7b-gguf",
    "sd15-assets",
)
ANDROID_ASSETS = ("qwen18b-gguf", "qwen3-4b-gguf",
                  "qwen35-9b-gguf", "qwen35-2b-gguf")


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
        # lock 本身是原生 Gemma4 工件身份的一部分，必须随包分发，不能只用它
        # 做打包时的本地校验。
        add("models/gemma4-native/gemma4-native.lock.json", lock)
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


def _existing_parent(path: Path) -> Path:
    """返回可用于 disk_usage 的现有目录，不创建目录。"""
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _storage_key(path: Path) -> tuple[str, str | int]:
    """标识 path 所在卷；Windows 优先使用盘符，POSIX 使用 st_dev。"""
    existing = _existing_parent(path)
    if os.name == "nt":
        drive = existing.drive or existing.anchor
        return "drive", drive.casefold()
    return "dev", existing.stat().st_dev


def _estimate_asset_bytes(asset_id: str) -> int:
    """只读估算某资产进入整合包前的未压缩字节数。

    不计算 SHA、不创建 staging。SD zip 使用 central directory 的未压缩大小；
    这会忽略跨包去重，因此预检结果刻意偏保守。
    """
    spec = ASSETS[asset_id]
    root = _resolve(asset_id)
    kind = spec["kind"]
    if kind == "file":
        if not root.is_file():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        return root.stat().st_size
    if kind == "dir":
        if not root.is_dir():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        files = [path for path in root.rglob("*") if path.is_file()]
        if not files:
            raise BundleError(f"缺失资产 {asset_id}: {root} 为空\n  获取: {spec['fetch']}")
        return sum(path.stat().st_size for path in files)
    if kind == "lock":
        lock = root / "gemma4-native.lock.json"
        if not lock.is_file():
            raise BundleError(f"缺失资产 {asset_id}: {lock}\n  获取: {spec['fetch']}")
        try:
            declared = json.loads(lock.read_text(encoding="utf-8")).get("artifacts") or {}
        except json.JSONDecodeError as exc:
            raise BundleError(f"资产 {asset_id} lock JSON 无效: {lock}") from exc
        total = lock.stat().st_size
        for name, info in declared.items():
            filename = (info.get("filename") if isinstance(info, dict) else None) or name
            artifact = root / filename
            if not artifact.is_file():
                raise BundleError(f"缺失资产 {asset_id} 文件 {filename}: {artifact}")
            total += artifact.stat().st_size
        return total
    if kind == "sd15":
        if not root.is_dir():
            raise BundleError(f"缺失资产 {asset_id}: {root}\n  获取: {spec['fetch']}")
        packages = sorted(root.glob("*.zip"))
        if not packages:
            raise BundleError(f"资产 {asset_id}: {root} 无离线包")
        total = 0
        for package in packages:
            try:
                with zipfile.ZipFile(package) as archive:
                    total += sum(
                        info.file_size for info in archive.infolist()
                        if not info.is_dir() and info.filename.startswith("models/")
                    )
            except zipfile.BadZipFile as exc:
                raise BundleError(f"资产 {asset_id} 离线包损坏: {package}") from exc
        if total <= 0:
            raise BundleError(f"资产 {asset_id}: {root} 无 models/ 内容")
        return total
    raise BundleError(f"未知资产 kind: {kind}")


def _capacity_preflight(scope: tuple[str, ...], variant: str, fmt: str,
                        *, verify: bool = False) -> dict:
    """返回构建峰值的保守磁盘预算，不写入文件。

    输出包始终以未压缩 payload 估算，因而对 ZIP_STORED 精确、对 7z 保守。
    SD 解包 staging 先于压缩存在；去重收益只在真正收集后计算，预检不会把它
    当作可用空间，避免空间恰好不足时留下半成品。
    """
    by_asset = {asset_id: _estimate_asset_bytes(asset_id) for asset_id in scope}
    payload_bytes = sum(by_asset.values())
    sd_staging_bytes = by_asset.get("sd15-assets", 0)
    output_bytes = payload_bytes + _CAPACITY_HEADROOM_BYTES
    build_peak_bytes = output_bytes + sd_staging_bytes
    verify_extract_bytes = payload_bytes + _CAPACITY_HEADROOM_BYTES if verify else 0

    out_parent = _existing_parent(OUT_DIR)
    tmp_parent = _existing_parent(TMP_BASE)
    out_key = _storage_key(out_parent)
    tmp_key = _storage_key(tmp_parent)
    demands: dict[tuple[str, str | int], dict] = {
        out_key: {
            "path": str(out_parent),
            "required_bytes": build_peak_bytes,
            "purpose": ["bundle_output", "sd15_staging"] if sd_staging_bytes else ["bundle_output"],
        }
    }
    if verify:
        if tmp_key == out_key:
            # verify 发生在 bundle 留在输出目录时，因此同卷峰值要保留输出包。
            demands[out_key]["required_bytes"] = max(
                demands[out_key]["required_bytes"], output_bytes + verify_extract_bytes,
            )
            demands[out_key]["purpose"].append("verify_extract")
        else:
            demands[tmp_key] = {
                "path": str(tmp_parent),
                "required_bytes": verify_extract_bytes,
                "purpose": ["verify_extract"],
            }

    volumes = []
    for demand in demands.values():
        free_bytes = shutil.disk_usage(demand["path"]).free
        volumes.append({
            **demand,
            "free_bytes": free_bytes,
            "admitted": free_bytes >= demand["required_bytes"],
        })
    return {
        "schema_version": "qlh.offline_bundle.capacity.v1",
        "variant": variant,
        "format": fmt,
        "verify": verify,
        "asset_bytes": by_asset,
        "payload_bytes": payload_bytes,
        "sd15_staging_bytes": sd_staging_bytes,
        "output_bytes": output_bytes,
        "volumes": volumes,
        "admitted": all(volume["admitted"] for volume in volumes),
    }


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def _print_capacity_preflight(report: dict) -> None:
    print(
        f"容量预检 {report['variant']} ({report['format']})："
        f"payload {_format_bytes(report['payload_bytes'])}，"
        f"SD staging {_format_bytes(report['sd15_staging_bytes'])}"
    )
    for volume in report["volumes"]:
        verdict = "通过" if volume["admitted"] else "不足"
        purpose = "+".join(volume["purpose"])
        print(
            f"  [{verdict}] {volume['path']}：需 {_format_bytes(volume['required_bytes'])}，"
            f"可用 {_format_bytes(volume['free_bytes'])} ({purpose})"
        )


def _require_capacity(scope: tuple[str, ...], variant: str, fmt: str,
                      *, verify: bool = False) -> dict:
    report = _capacity_preflight(scope, variant, fmt, verify=verify)
    if report["admitted"]:
        return report
    failed = [
        f"{volume['path']}（需 {_format_bytes(volume['required_bytes'])}，"
        f"可用 {_format_bytes(volume['free_bytes'])}）"
        for volume in report["volumes"] if not volume["admitted"]
    ]
    raise BundleError("磁盘容量预检失败，拒绝开始打包: " + "; ".join(failed))


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
                  threads: int = 4, *, verify: bool = False) -> Path:
    manifest_files: list[dict] = []
    checksum_lines: list[str] = []
    collected_all: list[dict] = []

    _require_capacity(scope, variant, fmt, verify=verify)
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
        "payload_bytes": sum(item["size"] for item in manifest_files),
        "stored_bytes": sum(item["size"] for item in manifest_files
                            if not item.get("dedup_of")),
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
    _write_archive_checksums(bundle_path, volume=volume)
    return bundle_path


def _build_bundle_zip(bundle_path, collected_all, manifest_json,
                      checksum_lines, variant) -> None:
    partial = Path(str(bundle_path) + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as zf:
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
        os.replace(partial, bundle_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


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
    # SD 内容来自 OUT_DIR/.staging-*，不是 REPO_ROOT；把它混入这一段会让
    # 7z 在项目根目录寻找尚不存在的 models/sd15-* 文件。
    models_files = [f for c in collected_all for f in c["files"]
                    if f["path"].startswith("models/")
                    and not f["path"].startswith("models/sd15-")
                    and not f.get("dedup_of")]
    if models_files:
        parts.append((REPO_ROOT, [f["path"] for f in models_files]))
    sd15_files = [f for c in collected_all for f in c["files"]
                  if f["path"].startswith("models/sd15-") and not f.get("dedup_of")]
    if sd15_files:
        parts.append((OUT_DIR / f".staging-{variant}", [f["path"] for f in sd15_files]))
    partial = Path(str(bundle_path) + ".partial")
    partial.unlink(missing_ok=True)
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
                   "-y", str(partial), *paths]
            r = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise BundleError(
                    f"7z 打包失败: {r.stderr[-300:] or r.stdout[-300:]}")
    if volume:
        _split_volumes(partial, volume)
        _publish_split_volumes(partial, bundle_path)
    else:
        os.replace(partial, bundle_path)


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
    for stale in bundle_path.parent.glob(f"{bundle_path.name}.*"):
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


def _publish_split_volumes(partial: Path, bundle_path: Path) -> None:
    """在全部临时分卷成功后再替换对外分卷，避免构建失败污染旧产物。"""
    partial_volumes = sorted(partial.parent.glob(f"{partial.name}.*"))
    if not partial_volumes:
        raise BundleError("7z 分卷构建未产生任何临时分卷")
    prefix = partial.name + "."
    for stale in bundle_path.parent.glob(f"{bundle_path.name}.*"):
        # partial 分卷与正式分卷共享前缀，不能把刚构建好的临时卷删掉。
        if stale.name.startswith(prefix):
            continue
        stale.unlink()
    for volume in partial_volumes:
        suffix = volume.name.removeprefix(prefix)
        os.replace(volume, Path(str(bundle_path) + f".{suffix}"))


def _archive_parts(bundle_path: Path, *, volume: str | None = None) -> list[Path]:
    if volume:
        parts = sorted(
            (
                path for path in bundle_path.parent.glob(f"{bundle_path.name}.*")
                if re.fullmatch(r"\.\d{3}", path.name.removeprefix(bundle_path.name))
            ),
        )
        if not parts:
            raise BundleError(f"未找到整合包分卷: {bundle_path}.001")
        return parts
    if not bundle_path.is_file():
        raise BundleError(f"未找到整合包: {bundle_path}")
    return [bundle_path]


def _write_archive_checksums(bundle_path: Path, *, volume: str | None = None) -> Path:
    """写 archive 级 SHA-256 侧车；分卷逐卷列出，供下载端先验文件完整性。"""
    parts = _archive_parts(bundle_path, volume=volume)
    sidecar = Path(str(bundle_path) + ".sha256")
    partial = Path(str(sidecar) + ".partial")
    content = "".join(f"{_sha256_file(part)}  {part.name}\n" for part in parts)
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, sidecar)
    return sidecar


def _verify_archive_checksums(bundle_path: Path) -> None:
    """若存在 archive 侧车则先验证；兼容历史包没有侧车的情况。"""
    sidecar = Path(str(bundle_path) + ".sha256")
    if not sidecar.is_file():
        return
    declared = sidecar.read_text(encoding="utf-8").splitlines()
    if not declared:
        raise BundleError("verify 失败：archive SHA 侧车为空")
    seen: set[str] = set()
    for line in declared:
        digest, separator, name = line.partition("  ")
        if (not separator or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not name or Path(name).name != name or name in seen):
            raise BundleError("verify 失败：archive SHA 侧车格式无效")
        seen.add(name)
        part = bundle_path.parent / name
        if not part.is_file():
            raise BundleError(f"verify 失败：archive 分卷缺失 {name}")
        if _sha256_file(part) != digest:
            raise BundleError(f"verify 失败：archive 分卷 {name} SHA 不匹配")
    if bundle_path.is_file() and seen != {bundle_path.name}:
        raise BundleError("verify 失败：archive SHA 侧车未绑定当前整合包")
    if not bundle_path.is_file() and f"{bundle_path.name}.001" not in seen:
        raise BundleError("verify 失败：archive SHA 侧车缺少首分卷")


def _import_readme(variant: str, manifest_json: str) -> str:
    if variant == "android":
        return (
            "# 安卓版整合包导入说明\n\n"
            f"> 版本：{BUNDLE_VERSION} | 仅含 GGUF（SAF 目录直接可用）\n\n"
            "1. 解压前先校验同名 archive `.sha256` 侧车（分卷需保留全部卷）\n"
            "2. 解压本包，得到 `models/` 目录（仅为 Android 注册表中的 GGUF）\n"
            "3. 在安卓 Full 模式设置中选择 `models/`（SAF 授权）→ 扫描 → 选模型\n"
            "4. 解压后校验：`sha256sum -c CHECKSUMS.sha256`\n\n"
            "说明：安卓本地只跑 GGUF（llama.cpp CPU）；判题/图像生成均走 PC 主节点远程推理。\n"
        )
    return (
        "# PC 版整合包导入说明\n\n"
        f"> 版本：{BUNDLE_VERSION} | 全量资产，解压到项目根即就位\n\n"
        "1. 解压前先校验同名 archive `.sha256` 侧车（分卷需保留全部卷）\n"
        "2. 解压到 QLH 项目根目录，`models/` 目录即就位\n"
        "3. 解压后校验：`sha256sum -c CHECKSUMS.sha256`\n"
        "4. 图像工作区会发现 `models/sd15-*`；判题模型与 Gemma4 工件按 MANIFEST.json 校验后可用\n\n"
        "清单（自动生成）：\n```json\n" + manifest_json + "\n```\n"
    )


def _verify_bundle(bundle_path: Path) -> None:
    """解包到临时目录，恢复 dedup 链接后逐文件 SHA 比对（7z/分卷/zip）。"""
    _verify_archive_checksums(bundle_path)
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
            _safe_extract_zip(bundle_path, tmp)
            root_dir = tmp
        # 先恢复 dedup 链接（MANIFEST 里的 dedup_of 映射）
        manifest_path = root_dir / "MANIFEST.json"
        if not manifest_path.is_file():
            raise BundleError("verify 失败：解包缺少 MANIFEST.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleError("verify 失败：MANIFEST.json 无效") from exc
        entries = _validate_manifest(manifest)
        restored = _restore_dedup(root_dir, manifest)
        if restored:
            print(f"  verify 恢复 {restored} 个去重链接")
        _validate_checksums(root_dir, entries)
        checked = 0
        for rel, entry in entries.items():
            p = _bundle_path(root_dir, rel)
            if not p.is_file():
                raise BundleError(f"verify 失败：解包缺文件 {rel}")
            if p.stat().st_size != entry["size"]:
                raise BundleError(f"verify 失败：{rel} 大小不匹配")
            if _sha256_file(p) != entry["sha256"]:
                raise BundleError(f"verify 失败：{rel} SHA 不匹配")
            checked += 1
        print(f"  verify OK：{checked} 文件一致")


def _safe_bundle_relpath(value: str) -> str:
    """校验 archive/manifest 内的 POSIX 相对路径，拒绝 Zip Slip 和歧义路径。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise BundleError(f"整合包包含非法路径: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BundleError(f"整合包包含越界路径: {value!r}")
    if any(":" in part for part in pure.parts):
        raise BundleError(f"整合包包含非法路径: {value!r}")
    return pure.as_posix()


def _bundle_path(root_dir: Path, relative: str) -> Path:
    return root_dir.joinpath(*PurePosixPath(_safe_bundle_relpath(relative)).parts)


def _safe_extract_zip(bundle_path: Path, target_dir: Path) -> None:
    """受限解压 ZIP，避免未经验证的归档写出临时根目录。"""
    try:
        archive = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"verify 失败：ZIP 损坏: {bundle_path}") from exc
    with archive:
        seen: set[str] = set()
        for info in archive.infolist():
            raw_name = info.filename[:-1] if info.is_dir() else info.filename
            relative = _safe_bundle_relpath(raw_name)
            if relative in seen:
                raise BundleError(f"verify 失败：ZIP 含重复路径 {relative}")
            seen.add(relative)
            destination = _bundle_path(target_dir, relative)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)


def _validate_manifest(manifest: dict) -> dict[str, dict]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "qlh.offline_bundle.v1":
        raise BundleError("verify 失败：MANIFEST schema 不受支持")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise BundleError("verify 失败：MANIFEST 缺少 files")
    entries: dict[str, dict] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise BundleError("verify 失败：MANIFEST 文件条目无效")
        relative = _safe_bundle_relpath(entry.get("path"))
        digest = entry.get("sha256")
        size = entry.get("size")
        if relative in entries:
            raise BundleError(f"verify 失败：MANIFEST 含重复路径 {relative}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleError(f"verify 失败：{relative} SHA 声明无效")
        if not isinstance(size, int) or size < 0:
            raise BundleError(f"verify 失败：{relative} 大小声明无效")
        dedup_of = entry.get("dedup_of")
        if dedup_of is not None:
            dedup_of = _safe_bundle_relpath(dedup_of)
            entry = {**entry, "dedup_of": dedup_of}
        entries[relative] = entry
    for relative, entry in entries.items():
        dedup_of = entry.get("dedup_of")
        if dedup_of and dedup_of not in entries:
            raise BundleError(f"verify 失败：{relative} 引用不存在的 dedup_of")
        if dedup_of == relative:
            raise BundleError(f"verify 失败：{relative} 不能引用自身 dedup_of")
    return entries


def _validate_checksums(root_dir: Path, entries: dict[str, dict]) -> None:
    checksums_path = root_dir / "CHECKSUMS.sha256"
    if not checksums_path.is_file():
        raise BundleError("verify 失败：解包缺少 CHECKSUMS.sha256")
    expected = {
        relative: entry["sha256"]
        for relative, entry in entries.items() if not entry.get("dedup_of")
    }
    declared: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator:
            raise BundleError("verify 失败：CHECKSUMS.sha256 格式无效")
        relative = _safe_bundle_relpath(relative)
        if relative in declared or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleError("verify 失败：CHECKSUMS.sha256 条目无效")
        declared[relative] = digest
    if declared != expected:
        raise BundleError("verify 失败：CHECKSUMS 与 MANIFEST 不一致")


def _restore_dedup(root_dir: Path, manifest: dict) -> int:
    """按 MANIFEST 的 dedup_of 恢复链接/复制（Windows hardlink、posix symlink）。"""
    restored = 0
    for f in manifest.get("files", []):
        dedup_of = f.get("dedup_of")
        if not dedup_of:
            continue
        target = _bundle_path(root_dir, f["path"])
        source = _bundle_path(root_dir, dedup_of)
        if not source.is_file():
            raise BundleError(f"verify 失败：dedup 源不存在 {dedup_of}")
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
    ap.add_argument("--preflight", action="store_true",
                    help="仅执行只读容量预检，不创建整合包")
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
    ap.add_argument("--out-dir", metavar="DIR", default=None,
                    help="输出目录（默认 build/offline-bundles；大包可用 D 盘如 D:/qlh-bundles）")
    args = ap.parse_args(argv)
    if args.out_dir:
        global OUT_DIR
        OUT_DIR = Path(args.out_dir)
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

    if args.preflight:
        reports = []
        if args.pc:
            reports.append(_capacity_preflight(
                PC_ASSETS, "pc", args.format, verify=args.verify))
        if args.android:
            reports.append(_capacity_preflight(
                ANDROID_ASSETS, "android", args.format, verify=args.verify))
        for report in reports:
            _print_capacity_preflight(report)
        return 0 if all(report["admitted"] for report in reports) else 2

    if args.pc:
        print(f"构建 PC 版（全量，{args.format}）...")
        bundle = _build_bundle(PC_ASSETS, "pc", args.format, args.volume,
                              args.threads, verify=args.verify)
        _print_bundle_result(bundle, args.volume)
        if args.verify:
            _verify_bundle(bundle)
    if args.android:
        print(f"构建安卓版（纯 GGUF，{args.format}）...")
        bundle = _build_bundle(ANDROID_ASSETS, "android", args.format,
                              args.volume, args.threads, verify=args.verify)
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
