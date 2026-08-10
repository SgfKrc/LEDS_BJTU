"""Read-only Launcher symptom diagnosis for UP-N6.2.

The module turns signed-install verification and small bounded local probes
into actionable, non-destructive troubleshooting guidance.  It intentionally
does not contact update sources, enumerate user data, or repair any files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from install_manifest import verify_install_tree


DIAGNOSE_SCHEMA_VERSION = 1
LOW_DISK_BYTES = 2 * 1024 * 1024 * 1024
MAX_FAILURE_PATHS = 20
_DRIVER_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")

# This is deliberately a finite, local knowledge base rather than remotely
# supplied advice.  Remote support content belongs to a separately trusted
# update protocol, not a troubleshooting command.
KNOWLEDGE_BASE: tuple[dict[str, str], ...] = (
    {
        "id": "install_signature_invalid",
        "title": "安装完整性基线不可信",
        "severity": "critical",
        "steps": "停止使用该安装目录；从可信更新源覆盖安装，勿手工替换程序文件。",
    },
    {
        "id": "program_file_integrity",
        "title": "程序文件缺失或损坏",
        "severity": "error",
        "steps": "记录下方失败文件；当前版本请覆盖安装。单文件自动修复属于后续 UP-N6.3。",
    },
    {
        "id": "version_mismatch",
        "title": "安装版本不一致",
        "severity": "error",
        "steps": "使用 bjtu update 更新，或使用版本回滚恢复与签名清单匹配的版本。",
    },
    {
        "id": "dll_or_pyd_load_failure",
        "title": "DLL 或 Python 扩展加载失败",
        "severity": "error",
        "steps": "确认安装包 CPU/CUDA 变体正确；Windows 请修复 Microsoft Visual C++ Redistributable 后覆盖安装。",
    },
    {
        "id": "cuda_driver_problem",
        "title": "NVIDIA 驱动或 CUDA 环境不可用",
        "severity": "warning",
        "steps": "确认 NVIDIA 驱动可用、CUDA 安装包与设备匹配；无 NVIDIA GPU 时改用 CPU 安装包。",
    },
    {
        "id": "low_disk_space",
        "title": "可用磁盘空间不足",
        "severity": "warning",
        "steps": "清理磁盘后重试；模型目录请先运行 models_clean 的 dry-run，勿手工删除未知模型文件。",
    },
    {
        "id": "update_or_network_failure",
        "title": "更新或网络连接失败",
        "severity": "warning",
        "steps": "检查网络和更新源；校园网 UDP 受限时使用 QLH 的 DERP/WSS 路径或手动下载安装包。",
    },
    {
        "id": "permission_problem",
        "title": "安装目录权限不足",
        "severity": "warning",
        "steps": "确认当前用户对安装目录有读取权限；需要更新时以可写用户目录安装或使用管理员权限。",
    },
    {
        "id": "antivirus_interference",
        "title": "可能被安全软件隔离",
        "severity": "warning",
        "steps": "检查安全软件隔离记录并为 QLH 安装目录加白；恢复后重新运行 verify，勿从不可信来源下载 DLL。",
    },
    {
        "id": "model_asset_problem",
        "title": "模型资产缺失或损坏",
        "severity": "warning",
        "steps": "运行 gguf_verify 或 models_sweep 检查模型资产；诊断不会扫描、下载或修改模型。",
    },
)
_KNOWLEDGE_BY_ID = {entry["id"]: entry for entry in KNOWLEDGE_BASE}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _issue(issue_id: str, *, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entry = _KNOWLEDGE_BY_ID[issue_id]
    return {
        "id": entry["id"],
        "title": entry["title"],
        "severity": entry["severity"],
        "auto_repair_available": False,
        "manual_steps": [entry["steps"]],
        "evidence": dict(evidence or {}),
    }


def _safe_driver_version(value: str) -> str | None:
    candidate = value.strip().splitlines()[0].strip() if value.strip() else ""
    return candidate if _DRIVER_VERSION_RE.fullmatch(candidate) else None


def probe_nvidia_smi() -> dict[str, Any]:
    """Return a bounded, non-shell CUDA driver capability probe."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"probed": True, "available": False, "reason": "not_found"}
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"probed": True, "available": False, "reason": "unavailable"}
    if completed.returncode != 0:
        return {"probed": True, "available": False, "reason": "unavailable"}
    return {
        "probed": True,
        "available": True,
        "driver_version": _safe_driver_version(completed.stdout),
    }


def _error_flags(error: str | None) -> dict[str, bool]:
    value = (error or "").casefold()
    return {
        "dll": any(token in value for token in ("dll", ".pyd", "winerror 126", "module could not")),
        "cuda": any(token in value for token in ("cuda", "nvidia", "driver", "cudnn")),
        "disk": any(token in value for token in ("no space", "disk full", "enospc", "磁盘")),
        "network": any(token in value for token in ("timeout", "timed out", "network", "connection", "dns", "http", "udp")),
        "permission": any(token in value for token in ("permission denied", "access is denied", "eacces", "权限")),
        "model": any(token in value for token in ("model", "gguf", "safetensors", "tokenizer", "权重")),
    }


def _system_snapshot(root: Path, *, probe_gpu: bool) -> dict[str, Any]:
    disk: dict[str, Any]
    try:
        usage = shutil.disk_usage(root)
        disk = {"available": True, "free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        disk = {"available": False}
    return {
        "disk": disk,
        "directory_writable": bool(os.access(root, os.W_OK)) if root.is_dir() else False,
        "gpu": probe_nvidia_smi() if probe_gpu else {"probed": False},
    }


def _integrity_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    failed = report.get("failed", [])
    failures = [item for item in failed if isinstance(item, Mapping)] if isinstance(failed, list) else []
    categories = sorted({str(item.get("category", "unknown")) for item in failures})
    paths = [str(item.get("path", "")) for item in failures if item.get("path")]
    return {
        "ok": bool(report.get("ok")),
        "level": str(report.get("level", "quick")),
        "summary": dict(report.get("summary", {})) if isinstance(report.get("summary"), Mapping) else {},
        "manifest": dict(report.get("manifest", {})) if isinstance(report.get("manifest"), Mapping) else None,
        "failure_categories": categories,
        "failed_paths": paths[:MAX_FAILURE_PATHS],
    }


def diagnose_install(
    root: str | os.PathLike[str],
    *,
    error: str | None = None,
    integrity_report: Mapping[str, Any] | None = None,
    trusted_keys_dir: str | os.PathLike[str] | None = None,
    probe_gpu: bool = True,
) -> dict[str, Any]:
    """Build a local, read-only diagnosis report for an installed app tree."""
    install_root = Path(root).expanduser()
    integrity = dict(integrity_report) if integrity_report is not None else verify_install_tree(
        install_root, level="quick", trusted_keys_dir=trusted_keys_dir,
    )
    summary = _integrity_summary(integrity)
    manifest = summary.get("manifest") or {}
    variant = str(manifest.get("variant", ""))
    system = _system_snapshot(install_root, probe_gpu=probe_gpu and variant == "cuda")
    categories = set(summary["failure_categories"])
    flags = _error_flags(error)
    issues: list[dict[str, Any]] = []

    if "signature" in categories:
        issues.append(_issue("install_signature_invalid", evidence={"categories": ["signature"]}))
    integrity_categories = sorted(categories & {"missing", "size", "hash", "unsafe", "io"})
    if integrity_categories:
        issues.append(_issue(
            "program_file_integrity",
            evidence={"categories": integrity_categories, "paths": summary["failed_paths"]},
        ))
    if "version" in categories:
        issues.append(_issue("version_mismatch", evidence={"categories": ["version"]}))
    if flags["dll"]:
        issues.append(_issue("dll_or_pyd_load_failure", evidence={"error_symptom": True}))
    gpu = system["gpu"]
    if variant == "cuda" and (flags["cuda"] or not gpu.get("available", False)):
        issues.append(_issue(
            "cuda_driver_problem",
            evidence={"gpu_available": bool(gpu.get("available")), "probe_reason": gpu.get("reason")},
        ))
    disk = system["disk"]
    if flags["disk"] or (disk.get("available") and int(disk.get("free_bytes", 0)) < LOW_DISK_BYTES):
        issues.append(_issue(
            "low_disk_space",
            evidence={"free_bytes": disk.get("free_bytes"), "threshold_bytes": LOW_DISK_BYTES},
        ))
    if flags["network"]:
        issues.append(_issue("update_or_network_failure", evidence={"error_symptom": True}))
    if flags["permission"] or not system["directory_writable"]:
        issues.append(_issue(
            "permission_problem",
            evidence={"directory_writable": bool(system["directory_writable"])},
        ))
    if "missing" in categories and "signature" not in categories:
        issues.append(_issue("antivirus_interference", evidence={"missing_files": True}))
    if flags["model"]:
        issues.append(_issue("model_asset_problem", evidence={"error_symptom": True}))

    return {
        "schema_version": DIAGNOSE_SCHEMA_VERSION,
        "command": "diagnose",
        "created_at": _utc_now(),
        "root": str(install_root),
        "integrity": summary,
        "system": system,
        "issues": issues,
        "ok": not issues,
    }


def diagnosis_bundle_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the privacy-reduced JSON record suitable for a support ZIP."""
    integrity = report.get("integrity", {})
    system = report.get("system", {})
    return {
        "schema_version": DIAGNOSE_SCHEMA_VERSION,
        "command": "diagnose",
        "created_at": report.get("created_at"),
        "integrity": {
            "ok": integrity.get("ok"),
            "level": integrity.get("level"),
            "summary": integrity.get("summary"),
            "failure_categories": integrity.get("failure_categories"),
        },
        "system": {
            "disk": system.get("disk"),
            "directory_writable": system.get("directory_writable"),
            "gpu": system.get("gpu"),
        },
        "issues": report.get("issues", []),
    }


def write_diagnosis_report(
    report: Mapping[str, Any], output: str | os.PathLike[str], *, bundle_safe: bool = False,
) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    value: Mapping[str, Any] = diagnosis_bundle_summary(report) if bundle_safe else report
    fd, raw = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def format_diagnosis(report: Mapping[str, Any]) -> str:
    issues = report.get("issues", [])
    if not isinstance(issues, list) or not issues:
        return "诊断完成：未发现内置症状规则匹配的问题。"
    lines = [f"诊断完成：发现 {len(issues)} 项需要处理的问题。"]
    for item in issues:
        if not isinstance(item, Mapping):
            continue
        steps = item.get("manual_steps", [])
        step = steps[0] if isinstance(steps, list) and steps else "请查看 JSON 诊断结果。"
        lines.append(f"- [{item.get('severity', 'warning')}] {item.get('title', '未知问题')}：{step}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlh-diagnose", description="Read-only QLH installation diagnosis")
    parser.add_argument("--root", required=True)
    parser.add_argument("--error", help="optional local startup error text; never persisted verbatim")
    parser.add_argument("--trusted-keys-dir")
    parser.add_argument("--no-gpu-probe", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--output", help="atomically persist the local diagnosis JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = diagnose_install(
        args.root,
        error=args.error,
        trusted_keys_dir=args.trusted_keys_dir,
        probe_gpu=not args.no_gpu_probe,
    )
    if args.output:
        write_diagnosis_report(report, args.output)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(format_diagnosis(report))
    return 0 if report["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
