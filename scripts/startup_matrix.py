"""Run the startup regression matrix for full, edge, no-Torch, and TUI paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTEST_PROFILES = ("full", "no-torch", "tui")
PROFILES = ("all", "edge", *PYTEST_PROFILES)
OUTPUT_TAIL = 4000

FULL_TESTS = (
    "tests/test_api_cold_start.py::test_lazy_model_manager_defers_construction",
    "tests/test_api_cold_start.py::test_available_models_does_not_touch_model_manager_when_unloaded",
    "tests/test_api_cold_start.py::test_frontend_bootstrap_endpoints_keep_model_manager_lazy",
    "tests/test_api_cold_start.py::test_health_is_available_while_runtime_startup_is_still_running",
    "tests/test_api_cold_start.py::test_reserved_pipeline_worker_does_not_auto_load_full_model",
    "tests/test_api_cold_start.py::test_reserved_worker_rejects_local_model_even_when_already_loaded",
    "tests/test_api_contract_keys.py",
)
NO_TORCH_TESTS = (
    "tests/test_api_cold_start.py::test_l_tier_cold_start_gguf_inference_and_tui_without_torch",
    "tests/test_api_cold_start.py::test_paged_kv_cache_import_keeps_torch_out_of_default_gguf_process",
    "tests/test_api_cold_start.py::test_default_gguf_api_import_and_bootstrap_queries_do_not_load_torch",
)
TUI_TESTS = (
    "tests/test_tui_backend.py",
    "tests/test_tui_textual.py",
    "tests/test_tui_e2e_flow.py",
)


def _default_edge_python() -> Path:
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return ROOT / ".venv-edge" / executable


def _is_virtual_environment(python_executable: Path) -> bool:
    """Check the target interpreter without importing the project's runtime."""
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", "import sys; print(sys.prefix != sys.base_prefix)"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0 and completed.stdout.strip().lower() == "true"


def _pytest_command(
    python_executable: Path,
    tests: Sequence[str],
) -> list[str]:
    command = [str(python_executable), "-m", "pytest", *tests, "-q"]
    # pytest.ini enables xdist for the normal unit channel. Matrix profiles are
    # independent already, so keep each profile serial and avoid nested workers.
    if importlib.util.find_spec("xdist") is not None:
        command.extend(["-n", "0"])
    return command


def _tail(value: str) -> str:
    return value[-OUTPUT_TAIL:]


def _run_subprocess(
    command: Sequence[str],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    value for value in (str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if value
                ),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "returncode": -1,
            "duration_s": round(time.perf_counter() - started, 4),
            "command": list(command),
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -2,
            "duration_s": round(time.perf_counter() - started, 4),
            "command": list(command),
            "stdout": _tail(exc.stdout or ""),
            "stderr": _tail(exc.stderr or "") + f"\ntimeout after {timeout_s}s",
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_s": round(time.perf_counter() - started, 4),
        "command": list(command),
        "stdout": _tail(completed.stdout),
        "stderr": _tail(completed.stderr),
    }


def _run_edge_profile(edge_python: Path, *, timeout_s: float) -> dict[str, Any]:
    command = [
        str(edge_python),
        str(ROOT / "scripts" / "edge_preflight.py"),
        "--python",
        str(edge_python),
        "--json",
    ]
    result = _run_subprocess(command, timeout_s=timeout_s)
    if result["returncode"] == -1:
        result["profile"] = "edge"
        return result

    try:
        preflight = json.loads(result.get("stdout", ""))
    except json.JSONDecodeError as exc:
        result.update(
            {
                "ok": False,
                "error": f"edge_preflight did not emit JSON: {exc}",
            }
        )
    else:
        result["ok"] = result["ok"] and preflight.get("ok") is True
        result["preflight"] = preflight
    result["profile"] = "edge"
    return result


def _run_pytest_profile(
    profile: str,
    test_python: Path,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    tests = {
        "full": FULL_TESTS,
        "no-torch": NO_TORCH_TESTS,
        "tui": TUI_TESTS,
    }[profile]
    result = _run_subprocess(
        _pytest_command(test_python, tests),
        timeout_s=timeout_s,
    )
    result["profile"] = profile
    result["tests"] = list(tests)
    return result


def run_matrix(
    profile: str = "all",
    *,
    test_python: str | os.PathLike[str] | None = None,
    edge_python: str | os.PathLike[str] | None = None,
    allow_system_python: bool = False,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown startup matrix profile: {profile}")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    selected = (*PYTEST_PROFILES, "edge") if profile == "all" else (profile,)
    selected_test_python = Path(test_python or sys.executable)
    selected_edge_python = Path(edge_python or _default_edge_python())
    results: list[dict[str, Any]] = []

    if any(item in PYTEST_PROFILES for item in selected):
        if not selected_test_python.is_file():
            error = f"test Python does not exist: {selected_test_python}"
            results.extend(
                {
                    "profile": item,
                    "ok": False,
                    "returncode": -1,
                    "error": error,
                }
                for item in selected
                if item in PYTEST_PROFILES
            )
        elif not allow_system_python and not _is_virtual_environment(selected_test_python):
            error = (
                "refusing to run pytest with a system Python; use .venv-test "
                "or pass --allow-system-python for a disposable environment"
            )
            results.extend(
                {
                    "profile": item,
                    "ok": False,
                    "returncode": -1,
                    "error": error,
                }
                for item in selected
                if item in PYTEST_PROFILES
            )
        else:
            for item in selected:
                if item in PYTEST_PROFILES:
                    results.append(
                        _run_pytest_profile(
                            item,
                            selected_test_python,
                            timeout_s=timeout_s,
                        )
                    )

    if "edge" in selected:
        results.append(_run_edge_profile(selected_edge_python, timeout_s=timeout_s))

    return {
        "ok": bool(results) and all(item.get("ok") is True for item in results),
        "profile": profile,
        "test_python": str(selected_test_python),
        "edge_python": str(selected_edge_python),
        "results": results,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default="all",
        help="matrix slice to run (default: all)",
    )
    parser.add_argument(
        "--test-python",
        default=None,
        help="pytest interpreter; defaults to the current interpreter",
    )
    parser.add_argument(
        "--edge-python",
        default=str(_default_edge_python()),
        help="Edge interpreter used by edge_preflight.py",
    )
    parser.add_argument(
        "--allow-system-python",
        action="store_true",
        help="allow pytest profiles outside a virtual environment",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="timeout for each profile subprocess (default: 600)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the complete matrix report as JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="also write the JSON matrix report to this path",
    )
    return parser.parse_args(argv)


def _print_summary(report: dict[str, Any]) -> None:
    for result in report["results"]:
        status = "PASS" if result.get("ok") else "FAIL"
        duration = result.get("duration_s", "n/a")
        print(f"[{status}] {result['profile']} ({duration}s)")
        if result.get("error"):
            print(f"  {result['error']}")
    status = "PASS" if report["ok"] else "FAIL"
    print(f"startup matrix: {status}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_matrix(
        args.profile,
        test_python=args.test_python,
        edge_python=args.edge_python,
        allow_system_python=args.allow_system_python,
        timeout_s=args.timeout_s,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
