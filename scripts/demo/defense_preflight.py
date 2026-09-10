"""Model-free, loopback-only preflight for the DEF-A3 defense checklist."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("checklist.json")
DEFAULT_REPORT = ROOT / "build" / "defense-preflight" / "latest.json"
EXPECTED_CATEGORIES = {"device", "power", "network", "model", "ports", "evidence", "recovery"}
EXPECTED_GUARD = {
    "external_network_required": False,
    "model_load_required": False,
    "real_model_claim_allowed": False,
    "physical_dual_host_claim_allowed": False,
}
KNOWN_HANDLERS = {
    "python_runtime", "node_runtime", "edge_browser", "free_storage", "power_supply",
    "loopback", "frontend_port", "api_port", "cluster_port", "model_scope",
    "evidence_contracts", "recovery_entrypoints",
}


class PreflightError(ValueError):
    """A stable preflight configuration or checklist-contract error."""


@dataclass(frozen=True)
class PreflightConfig:
    frontend_port: int = 5174
    api_port: int = 8000
    cluster_port: int = 8001
    minimum_free_gb: float = 1.0
    minimum_battery_percent: int = 40
    strict_warnings: bool = False
    report_path: Path = DEFAULT_REPORT


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightError(f"{label} 不能为空")
    return value.strip()


def _repo_path(value: object, label: str) -> tuple[str, Path]:
    relative = _text(value, label).replace("\\", "/")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PreflightError(f"{label} 必须是仓库内相对路径")
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PreflightError(f"{label} 越出仓库") from exc
    return relative, resolved


def _report_ref(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise PreflightError("报告路径必须位于仓库内") from exc


def _validate_config(config: PreflightConfig) -> None:
    ports = (config.frontend_port, config.api_port, config.cluster_port)
    if any(isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 for port in ports):
        raise PreflightError("端口必须是 1 到 65535 的整数")
    if len(set(ports)) != 3:
        raise PreflightError("frontend/api/cluster 端口必须互不相同")
    if not 0.1 <= config.minimum_free_gb <= 1024:
        raise PreflightError("最小可用磁盘必须在 0.1 到 1024 GB 之间")
    if not 1 <= config.minimum_battery_percent <= 100:
        raise PreflightError("最低电量必须在 1 到 100 之间")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError(f"检查清单不可读取：{type(exc).__name__}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "qlh.defense_checklist.v1":
        raise PreflightError("检查清单 schema 无效")
    if payload.get("ticket") != "DEF-A3" or payload.get("profile") != "fixture-defense":
        raise PreflightError("检查清单票号或 profile 无效")
    if set(payload.get("categories", [])) != EXPECTED_CATEGORIES:
        raise PreflightError("检查清单类别覆盖不完整")
    if payload.get("claim_guard") != EXPECTED_GUARD:
        raise PreflightError("检查清单声明边界无效")
    _, document_path = _repo_path(payload.get("document"), "检查清单文档")
    if not document_path.is_file():
        raise PreflightError("检查清单文档不存在")
    document = document_path.read_text(encoding="utf-8")

    automated = payload.get("automated_requirements")
    manual = payload.get("manual_requirements")
    recovery = payload.get("recovery_playbook")
    if not isinstance(automated, list) or not automated:
        raise PreflightError("自动检查项为空")
    if not isinstance(manual, list) or not manual:
        raise PreflightError("人工检查项为空")
    if not isinstance(recovery, list) or not recovery:
        raise PreflightError("恢复动作为空")

    all_ids: set[str] = set()
    for item in automated:
        if not isinstance(item, dict) or set(item) != {"id", "category", "handler", "blocking"}:
            raise PreflightError("自动检查项字段无效")
        item_id = _text(item.get("id"), "自动检查 ID")
        if item_id in all_ids:
            raise PreflightError(f"检查 ID 重复：{item_id}")
        if item.get("category") not in EXPECTED_CATEGORIES or item.get("handler") not in KNOWN_HANDLERS:
            raise PreflightError(f"自动检查 {item_id} 类别或 handler 无效")
        if not isinstance(item.get("blocking"), bool):
            raise PreflightError(f"自动检查 {item_id} blocking 无效")
        if f"`{item_id}`" not in document:
            raise PreflightError(f"检查清单文档缺少 {item_id}")
        all_ids.add(item_id)

    for item in manual:
        if not isinstance(item, dict) or set(item) != {"id", "category", "instruction"}:
            raise PreflightError("人工检查项字段无效")
        item_id = _text(item.get("id"), "人工检查 ID")
        if item_id in all_ids or item.get("category") not in EXPECTED_CATEGORIES:
            raise PreflightError(f"人工检查 {item_id} 重复或类别无效")
        _text(item.get("instruction"), f"人工检查 {item_id} 说明")
        if f"`{item_id}`" not in document:
            raise PreflightError(f"检查清单文档缺少 {item_id}")
        all_ids.add(item_id)

    for item in recovery:
        if not isinstance(item, dict) or set(item) != {"id", "trigger", "action", "evidence"}:
            raise PreflightError("恢复动作字段无效")
        item_id = _text(item.get("id"), "恢复动作 ID")
        if item_id in all_ids:
            raise PreflightError(f"恢复动作 ID 重复：{item_id}")
        _text(item.get("trigger"), f"恢复动作 {item_id} 触发条件")
        _text(item.get("action"), f"恢复动作 {item_id} 动作")
        evidence_ref, evidence_path = _repo_path(item.get("evidence"), f"恢复动作 {item_id} 证据")
        if not evidence_ref.startswith("build/") and not evidence_path.is_file():
            raise PreflightError(f"恢复动作 {item_id} 的仓库内证据不存在")
        if f"`{item_id}`" not in document:
            raise PreflightError(f"检查清单文档缺少 {item_id}")
        all_ids.add(item_id)
    return payload


def _result(requirement: dict[str, Any], state: str, detail: str) -> dict[str, Any]:
    return {
        "id": requirement["id"],
        "category": requirement["category"],
        "state": state,
        "blocking": requirement["blocking"],
        "detail": detail,
    }


def _command_version(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = (completed.stdout or completed.stderr).strip().splitlines()
    return line[0][:80] if line else "available"


def _check_python(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return _result(requirement, "passed", f"python={version}; os={platform.system() or 'unknown'}")


def _check_node(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    node = _command_version(["node", "--version"])
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    npm = _command_version([npm_command, "--version"])
    vite = ROOT / "frontend_cybergothic" / "node_modules" / "vite" / "package.json"
    package = ROOT / "frontend_cybergothic" / "package.json"
    if node is None or npm is None or not vite.is_file() or not package.is_file():
        return _result(requirement, "failed", "node/npm/current-frontend-vite unavailable")
    return _result(requirement, "passed", f"node={node}; npm={npm}; frontend=frontend_cybergothic")


def _edge_available() -> bool:
    if shutil.which("msedge") or shutil.which("microsoft-edge") or shutil.which("microsoft-edge-stable"):
        return True
    candidates: list[Path] = []
    if os.name == "nt":
        for env_name in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
            base = os.environ.get(env_name)
            if base:
                candidates.append(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    elif platform.system() == "Darwin":
        candidates.append(Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"))
    return any(path.is_file() for path in candidates)


def _check_edge(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    available = _edge_available()
    return _result(requirement, "passed" if available else "failed", "edge=available" if available else "edge=unavailable")


def _check_storage(requirement: dict[str, Any], config: PreflightConfig) -> dict[str, Any]:
    free_gb = shutil.disk_usage(ROOT).free / 1024 ** 3
    state = "passed" if free_gb >= config.minimum_free_gb else "failed"
    return _result(requirement, state, f"free_gb={free_gb:.2f}; required_gb={config.minimum_free_gb:.2f}")


def _windows_power() -> tuple[bool | None, bool | None, int | None]:
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line_status", ctypes.c_ubyte),
            ("battery_flag", ctypes.c_ubyte),
            ("battery_percent", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("battery_life_time", ctypes.c_uint32),
            ("battery_full_life_time", ctypes.c_uint32),
        ]

    status = SystemPowerStatus()
    try:
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    except (AttributeError, OSError):
        return None, None, None
    if not ok:
        return None, None, None
    ac = None if status.ac_line_status == 255 else status.ac_line_status == 1
    present = None if status.battery_flag == 255 else not bool(status.battery_flag & 128)
    percent = None if status.battery_percent == 255 else int(status.battery_percent)
    return ac, present, percent


def _linux_power() -> tuple[bool | None, bool | None, int | None]:
    power_root = Path("/sys/class/power_supply")
    if not power_root.is_dir():
        return None, None, None
    batteries = []
    for entry in power_root.iterdir():
        try:
            if (entry / "type").read_text(encoding="utf-8").strip().lower() == "battery":
                batteries.append(entry)
        except OSError:
            continue
    if not batteries:
        return None, False, None
    battery = batteries[0]
    try:
        percent = int((battery / "capacity").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        percent = None
    try:
        state = (battery / "status").read_text(encoding="utf-8").strip().lower()
        ac = state in {"charging", "full"}
    except OSError:
        ac = None
    return ac, True, percent


def _power_status() -> tuple[bool | None, bool | None, int | None]:
    if os.name == "nt":
        return _windows_power()
    if platform.system() == "Linux":
        return _linux_power()
    return None, None, None


def _check_power(requirement: dict[str, Any], config: PreflightConfig) -> dict[str, Any]:
    ac, present, percent = _power_status()
    if present is False:
        return _result(requirement, "passed", "battery=not-present; manual_ac_check=required")
    if present is None or percent is None:
        return _result(requirement, "warning", "battery=unknown; manual_check=required")
    if ac is True:
        return _result(requirement, "passed", f"battery_percent={percent}; ac=connected")
    if percent < config.minimum_battery_percent:
        return _result(requirement, "warning", f"battery_percent={percent}; ac=not-connected; minimum={config.minimum_battery_percent}")
    return _result(requirement, "passed", f"battery_percent={percent}; ac={'unknown' if ac is None else 'not-connected'}")


def _check_loopback(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    accepted: socket.socket | None = None
    try:
        server.settimeout(2.0)
        client.settimeout(2.0)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client.connect(("127.0.0.1", server.getsockname()[1]))
        accepted, _ = server.accept()
        client.sendall(b"ok")
        if accepted.recv(2) != b"ok":
            raise OSError("loopback marker mismatch")
    except OSError:
        return _result(requirement, "failed", "tcp_loopback=unavailable; external_probe=false")
    finally:
        if accepted is not None:
            accepted.close()
        client.close()
        server.close()
    return _result(requirement, "passed", "tcp_loopback=available; external_probe=false")


def _port_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _check_port(requirement: dict[str, Any], port: int) -> dict[str, Any]:
    available = _port_available(port)
    if available:
        return _result(requirement, "passed", f"port={port}; state=available; bind=loopback")
    state = "failed" if requirement["blocking"] else "warning"
    return _result(requirement, state, f"port={port}; state=in-use; owner=not-inspected")


def _check_frontend_port(requirement: dict[str, Any], config: PreflightConfig) -> dict[str, Any]:
    return _check_port(requirement, config.frontend_port)


def _check_api_port(requirement: dict[str, Any], config: PreflightConfig) -> dict[str, Any]:
    return _check_port(requirement, config.api_port)


def _check_cluster_port(requirement: dict[str, Any], config: PreflightConfig) -> dict[str, Any]:
    return _check_port(requirement, config.cluster_port)


def _check_model_scope(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    return _result(requirement, "deferred", "model_load=not-required; real_model_claim=false; tickets=AUD-RT-01,AUD-SW-01")


def _check_evidence(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    required = (
        "scripts/demo/checklist.json",
        "scripts/demo/defense_preflight.py",
        "scripts/demo/storyline.json",
        "scripts/demo/scenarios.json",
        "frontend_cybergothic/src/data/defense-topology.json",
        "docs/答辩现场检查清单-2026-09-10.md",
        "docs/答辩演示-5分钟故事线-2026-09-10.md",
        "docs/答辩演示-性能结题表-2026-09-10.md",
        "tests/test_defense_demo.py",
        "tests/test_defense_performance_report.py",
        "tests/test_defense_preflight.py",
    )
    missing = [item for item in required if not (ROOT / item).is_file()]
    return _result(requirement, "failed" if missing else "passed", f"required={len(required)}; missing={len(missing)}")


def _check_recovery(requirement: dict[str, Any], _config: PreflightConfig) -> dict[str, Any]:
    required = (
        "scripts/demo/demo.bat",
        "scripts/demo/demo.sh",
        "scripts/demo/benchmark.bat",
        "scripts/demo/benchmark.sh",
        "scripts/demo/performance_report.bat",
        "scripts/demo/performance_report.sh",
        "scripts/demo/story.bat",
        "scripts/demo/story.sh",
        "scripts/demo/checklist.bat",
        "scripts/demo/checklist.sh",
    )
    missing = [item for item in required if not (ROOT / item).is_file()]
    return _result(requirement, "failed" if missing else "passed", f"entrypoints={len(required)}; missing={len(missing)}")


HANDLERS: dict[str, Callable[[dict[str, Any], PreflightConfig], dict[str, Any]]] = {
    "python_runtime": _check_python,
    "node_runtime": _check_node,
    "edge_browser": _check_edge,
    "free_storage": _check_storage,
    "power_supply": _check_power,
    "loopback": _check_loopback,
    "frontend_port": _check_frontend_port,
    "api_port": _check_api_port,
    "cluster_port": _check_cluster_port,
    "model_scope": _check_model_scope,
    "evidence_contracts": _check_evidence,
    "recovery_entrypoints": _check_recovery,
}


def run_preflight(config: PreflightConfig, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    _validate_config(config)
    manifest = manifest or load_manifest()
    checks = [HANDLERS[item["handler"]](item, config) for item in manifest["automated_requirements"]]
    blocking_failures = [item["id"] for item in checks if item["blocking"] and item["state"] == "failed"]
    warnings = [item["id"] for item in checks if item["state"] == "warning" or (not item["blocking"] and item["state"] == "failed")]
    if blocking_failures:
        status = "blocked"
    elif warnings:
        status = "warning"
    else:
        status = "ready_for_manual_checks"
    return {
        "schema": "qlh.defense_preflight.v1",
        "status": status,
        "profile": manifest["profile"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_guard": {
            **manifest["claim_guard"],
            "external_network_probed": False,
            "model_files_enumerated": False,
            "model_loaded": False,
        },
        "ports": {
            "frontend": config.frontend_port,
            "api_advisory": config.api_port,
            "cluster_advisory": config.cluster_port,
        },
        "checks": checks,
        "blocking_failures": blocking_failures,
        "warnings": warnings,
        "manual_pending": [item["id"] for item in manifest["manual_requirements"]],
        "recovery_actions": [item["id"] for item in manifest["recovery_playbook"]],
        "document": manifest["document"],
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    _report_ref(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("端口必须是整数") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1 到 65535 之间")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the model-free DEF-A3 defense preflight")
    parser.add_argument("--frontend-port", type=_port, default=5174)
    parser.add_argument("--api-port", type=_port, default=8000)
    parser.add_argument("--cluster-port", type=_port, default=8001)
    parser.add_argument("--minimum-free-gb", type=float, default=1.0)
    parser.add_argument("--minimum-battery-percent", type=int, default=40)
    parser.add_argument("--strict-warnings", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PreflightConfig(
        frontend_port=args.frontend_port,
        api_port=args.api_port,
        cluster_port=args.cluster_port,
        minimum_free_gb=args.minimum_free_gb,
        minimum_battery_percent=args.minimum_battery_percent,
        strict_warnings=args.strict_warnings,
        report_path=args.report,
    )
    try:
        report = run_preflight(config)
        write_report(report, config.report_path)
    except PreflightError as exc:
        print(f"[QLH-PREFLIGHT] ERROR {exc}", file=sys.stderr)
        return 2
    for item in report["checks"]:
        print(f"[QLH-PREFLIGHT] {item['state'].upper():8s} {item['id']} {item['detail']}")
    print(
        f"[QLH-PREFLIGHT] {report['status'].upper()} "
        f"manual_pending={len(report['manual_pending'])} warnings={len(report['warnings'])}"
    )
    if report["status"] == "blocked":
        return 2
    if config.strict_warnings and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
