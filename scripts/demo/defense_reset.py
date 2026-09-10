"""Fail-closed cleanup for processes started by the defense-demo tickets.

The default is a dry run.  Even with --apply, receipt data is never treated as
authority: every PID is revalidated by creation time, command allowlist, and
working directory immediately before it can be stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from demo_ownership import OWNERSHIP_ROOT, ROOT, inspect_identity, load_receipt


DEFAULT_REPORT = ROOT / "build" / "defense-reset" / "latest.json"
TEMP_NAME = re.compile(r"^[0-9a-f]{24}\.tmp$")


class ResetError(ValueError):
    """A stable reset configuration error."""


@dataclass(frozen=True)
class ResetConfig:
    apply: bool = False
    timeout: float = 5.0
    report_path: Path = DEFAULT_REPORT


def _validate_config(config: ResetConfig) -> None:
    if not 0.5 <= config.timeout <= 30.0:
        raise ResetError("timeout 必须在 0.5 到 30 秒之间")
    try:
        config.report_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ResetError("报告路径必须位于仓库内") from exc


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


def _receipt_paths(ownership_root: Path) -> list[Path]:
    if not ownership_root.is_dir():
        return []
    return sorted(
        path for path in ownership_root.iterdir()
        if path.is_file() and path.suffix == ".json"
    )


def _inspect_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = load_receipt(path)
    except ValueError:
        return {"path": path, "payload": None, "state": "invalid", "identities": []}
    inspected = [(identity, *inspect_identity(identity)) for identity in payload["processes"]]
    states = [state for _identity, state, _process in inspected]
    if any(state in {"invalid", "mismatch"} for state in states):
        state = "review_required"
    elif any(state == "matched" for state in states):
        state = "actionable"
    else:
        state = "stale"
    return {"path": path, "payload": payload, "state": state, "identities": inspected}


def _stop_identities(
    identities: list[tuple[dict[str, Any], str, psutil.Process | None]],
    timeout: float,
) -> tuple[int, int, int]:
    """Return stopped, forced, and review counts after point-of-action checks."""
    verified: list[tuple[dict[str, Any], psutil.Process]] = []
    for identity, _old_state, _old_process in identities:
        state, process = inspect_identity(identity)
        if state == "matched" and process is not None:
            verified.append((identity, process))
        elif state == "mismatch":
            return 0, 0, 1

    # Stop descendants before their launcher shim, keeping the tree observable.
    verified.sort(key=lambda item: item[0]["pid"] == item[0]["root_pid"])
    for _identity, process in verified:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except (psutil.AccessDenied, OSError):
            return 0, 0, 1

    _, alive = psutil.wait_procs([process for _identity, process in verified], timeout=timeout)
    forced = 0
    review = 0
    for process in alive:
        identity = next(item for item, candidate in verified if candidate.pid == process.pid)
        state, current = inspect_identity(identity)
        if state == "matched" and current is not None:
            try:
                current.kill()
                forced += 1
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
            except (psutil.AccessDenied, OSError):
                review += 1
        elif state == "mismatch":
            review += 1
    if alive:
        psutil.wait_procs([process for process in alive], timeout=timeout)

    remaining = 0
    stopped = 0
    for identity, _process in verified:
        state, _current = inspect_identity(identity)
        if state == "dead":
            stopped += 1
        else:
            remaining += 1
    return stopped, forced, review + remaining


def run_reset(
    config: ResetConfig,
    *,
    ownership_root: Path = OWNERSHIP_ROOT,
) -> dict[str, Any]:
    _validate_config(config)
    receipt_paths = _receipt_paths(ownership_root)
    inspections = [_inspect_receipt(path) for path in receipt_paths]
    valid = [item for item in inspections if item["payload"] is not None]
    invalid_count = sum(item["state"] == "invalid" for item in inspections)
    mismatch_count = sum(
        state == "mismatch"
        for item in valid
        for _identity, state, _process in item["identities"]
    )
    invalid_process_count = sum(
        state == "invalid"
        for item in valid
        for _identity, state, _process in item["identities"]
    )
    matched_count = sum(
        state == "matched"
        for item in valid
        for _identity, state, _process in item["identities"]
    )
    dead_count = sum(
        state == "dead"
        for item in valid
        for _identity, state, _process in item["identities"]
    )
    owned_ports = sorted({port for item in valid for port in item["payload"]["ports"]})
    actionable_ports = {
        port
        for item in valid
        if item["state"] == "actionable"
        for port in item["payload"]["ports"]
    }
    stopped_count = 0
    forced_count = 0
    action_review_count = 0

    if config.apply:
        seen: set[tuple[int, float]] = set()
        for item in valid:
            if item["state"] != "actionable":
                continue
            unique = []
            for identity, state, process in item["identities"]:
                key = (identity["pid"], float(identity["create_time"]))
                if key not in seen:
                    seen.add(key)
                    unique.append((identity, state, process))
            stopped, forced, review = _stop_identities(unique, config.timeout)
            stopped_count += stopped
            forced_count += forced
            action_review_count += review

    ports = []
    for port in owned_ports:
        if _port_available(port):
            state = "available"
        elif not config.apply and port in actionable_ports:
            state = "in_use_by_registered_candidate"
        else:
            state = "in_use_unowned_or_unresolved"
        ports.append({"port": port, "state": state})
    unresolved_ports = sum(item["state"] == "in_use_unowned_or_unresolved" for item in ports)
    removed_receipts = 0
    removed_temp_files = 0
    if config.apply:
        for item in valid:
            states = [inspect_identity(identity)[0] for identity, _state, _process in item["identities"]]
            receipt_ports = item["payload"]["ports"]
            ports_clear = all(_port_available(port) for port in receipt_ports)
            if states and all(state == "dead" for state in states) and ports_clear:
                item["path"].unlink(missing_ok=True)
                removed_receipts += 1
        if ownership_root.is_dir():
            for path in ownership_root.iterdir():
                if path.is_file() and TEMP_NAME.fullmatch(path.name):
                    path.unlink(missing_ok=True)
                    removed_temp_files += 1

    review_count = invalid_count + invalid_process_count + mismatch_count + action_review_count + unresolved_ports
    if review_count:
        status = "review_required"
    elif config.apply:
        status = "reset_complete"
    elif matched_count:
        status = "dry_run"
    else:
        status = "nothing_to_reset"
    return {
        "schema": "qlh.defense_reset.v1",
        "status": status,
        "mode": "apply" if config.apply else "dry_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_guard": {
            "registered_processes_only": True,
            "identity_revalidated": True,
            "unregistered_processes_stopped": False,
            "unregistered_ports_released": False,
            "model_files_touched": False,
            "database_touched": False,
            "user_configuration_touched": False,
            "evidence_reports_preserved": True,
        },
        "summary": {
            "receipt_files": len(receipt_paths),
            "valid_receipts": len(valid),
            "invalid_receipts": invalid_count,
            "matched_processes": matched_count,
            "dead_processes": dead_count,
            "invalid_processes": invalid_process_count,
            "mismatched_processes": mismatch_count,
            "stopped_processes": stopped_count,
            "forced_processes": forced_count,
            "removed_receipts": removed_receipts,
            "removed_temp_files": removed_temp_files,
            "review_items": review_count,
        },
        "owned_ports": ports,
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reset only registered QLH defense-demo runtime state")
    parser.add_argument("--apply", action="store_true", help="stop revalidated owned processes and clear their receipts")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ResetConfig(apply=args.apply, timeout=args.timeout, report_path=args.report)
    try:
        report = run_reset(config)
        write_report(report, config.report_path)
    except ResetError as exc:
        print(f"[QLH-RESET] ERROR {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        f"[QLH-RESET] {report['status'].upper()} mode={report['mode']} "
        f"matched={summary['matched_processes']} stopped={summary['stopped_processes']} "
        f"review={summary['review_items']}"
    )
    print(f"[QLH-RESET] REPORT {config.report_path}")
    return 2 if report["status"] == "review_required" else 0


if __name__ == "__main__":
    raise SystemExit(main())
