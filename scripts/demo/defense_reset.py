"""QLH CLI adapter for spawnledger's fail-closed reset engine."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from demo_ownership import OWNERSHIP_ROOT, ROOT, inspect_identity, load_receipt
from spawnledger import (
    TEMP_NAME_PATTERN,
    ResetConfig as CoreResetConfig,
    inspect_receipt as core_inspect_receipt,
    port_available as core_port_available,
    receipt_paths as core_receipt_paths,
    run_reset as core_run_reset,
    stop_identities as core_stop_identities,
    write_report as core_write_report,
)


DEFAULT_REPORT = ROOT / "build" / "defense-reset" / "latest.json"
TEMP_NAME = TEMP_NAME_PATTERN


class ResetError(ValueError):
    """A stable reset configuration error."""


@dataclass(frozen=True)
class ResetConfig:
    apply: bool = False
    timeout: float = 5.0
    report_path: Path = DEFAULT_REPORT


def _validate_config(config: ResetConfig) -> None:
    if not isinstance(config.apply, bool):
        raise ResetError("apply 必须是布尔值")
    if (
        isinstance(config.timeout, bool)
        or not isinstance(config.timeout, (int, float))
        or not 0.5 <= config.timeout <= 30.0
    ):
        raise ResetError("timeout 必须在 0.5 到 30 秒之间")
    try:
        config.report_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ResetError("报告路径必须位于仓库内") from exc


def _port_available(port: int) -> bool:
    return core_port_available(port)


def _receipt_paths(ownership_root: Path) -> list[Path]:
    return core_receipt_paths(ownership_root)


def _inspect_receipt(path: Path) -> dict[str, Any]:
    return core_inspect_receipt(
        path,
        receipt_loader=load_receipt,
        identity_inspector=inspect_identity,
    )


def _stop_identities(
    identities: list[tuple[dict[str, Any], str, psutil.Process | None]],
    timeout: float,
) -> tuple[int, int, int]:
    return core_stop_identities(identities, timeout, identity_inspector=inspect_identity)


def run_reset(config: ResetConfig, *, ownership_root: Path = OWNERSHIP_ROOT) -> dict[str, Any]:
    _validate_config(config)
    return core_run_reset(
        CoreResetConfig(apply=config.apply, timeout=config.timeout),
        ownership_root=ownership_root,
        receipt_loader=load_receipt,
        identity_inspector=inspect_identity,
        stopper=_stop_identities,
        port_checker=_port_available,
        report_schema="qlh.defense_reset.v1",
        scope_guard={
            "model_files_touched": False,
            "database_touched": False,
            "user_configuration_touched": False,
            "evidence_reports_preserved": True,
        },
    )


def write_report(report: dict[str, Any], report_path: Path) -> None:
    core_write_report(report, report_path)


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
