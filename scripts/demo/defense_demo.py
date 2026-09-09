"""QLH defense demo launcher.

The default fixture mode starts only the product frontend and is safe to run
without a model, network, or worker. Live mode must be explicitly selected and
starts the local API plus the frontend on loopback addresses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = ROOT / "frontend_cybergothic"
BUILD_ROOT = ROOT / "build" / "defense-demo"
DEFAULT_REPORT = BUILD_ROOT / "latest.json"


@dataclass(frozen=True)
class DemoConfig:
    mode: str = "fixtures"
    api_port: int = 8000
    frontend_port: int = 5174
    startup_timeout: float = 30.0
    duration: float = 0.0
    open_browser: bool = False
    skip_frontend: bool = False
    report_path: Path = DEFAULT_REPORT


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[str]
    log_path: Path


@dataclass
class DemoRun:
    config: DemoConfig
    processes: list[ManagedProcess] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append({"name": name, "ok": ok, "detail": detail})

    def stop(self) -> None:
        for managed in reversed(self.processes):
            process = managed.process
            if os.name == "nt":
                # npm/python on Windows can be launcher shims. Kill the tree
                # so a child server cannot survive the demo process.
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                continue
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        self.processes.clear()


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _python_command() -> str:
    executable = "python.exe" if os.name == "nt" else "python"
    candidates = (
        ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv-test" / ("Scripts" if os.name == "nt" else "bin") / executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _live_python_command() -> str:
    executable = "python.exe" if os.name == "nt" else "python"
    candidates = (
        ROOT / ".venv-packaging-cuda" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / executable,
        ROOT / ".venv-test" / ("Scripts" if os.name == "nt" else "bin") / executable,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _loopback_url(port: int, path: str = "") -> str:
    return f"http://127.0.0.1:{port}{path}"


def _cluster_port(config: DemoConfig) -> int:
    port = config.api_port + 1
    if port == config.frontend_port:
        port += 1
    return port


def _public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _http_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    return payload


def _http_ready(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 400:
            raise RuntimeError(f"HTTP {response.status}")


def wait_for_health(url: str, timeout: float, *, sleep_seconds: float = 0.25) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "unavailable"
    while time.monotonic() < deadline:
        try:
            payload = _http_json(url, min(2.0, max(0.2, deadline - time.monotonic())))
            if payload.get("status") == "ok":
                return payload
            last_error = "health response status is not ok"
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(sleep_seconds)
    raise RuntimeError(f"health check timed out: {last_error}")


def wait_for_ready(url: str, timeout: float, *, sleep_seconds: float = 0.25) -> None:
    deadline = time.monotonic() + timeout
    last_error = "unavailable"
    while time.monotonic() < deadline:
        try:
            _http_ready(url, min(2.0, max(0.2, deadline - time.monotonic())))
            return
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = type(exc).__name__
        time.sleep(sleep_seconds)
    raise RuntimeError(f"frontend check timed out: {last_error}")


def _start_process(run: DemoRun, name: str, command: list[str], env: dict[str, str], log_name: str) -> None:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = BUILD_ROOT / log_name
    handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT if name == "backend" else FRONTEND_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except Exception:
        handle.close()
        raise
    handle.close()
    run.processes.append(ManagedProcess(name=name, process=process, log_path=log_path))


def _validate_layout(config: DemoConfig) -> list[str]:
    errors: list[str] = []
    if not (ROOT / "src" / "api_server.py").is_file():
        errors.append("src/api_server.py is missing")
    if not FRONTEND_ROOT.is_dir():
        errors.append("frontend_cybergothic is missing")
    if not (FRONTEND_ROOT / "package.json").is_file():
        errors.append("frontend_cybergothic/package.json is missing")
    if config.mode == "live" and not shutil.which(_live_python_command()):
        errors.append("python runtime is unavailable")
    if not config.skip_frontend and not shutil.which(_npm_command()):
        errors.append("npm runtime is unavailable")
    if config.mode == "live" and config.api_port == config.frontend_port:
        errors.append("api and frontend ports must be different")
    if config.mode == "live" and _cluster_port(config) > 65535:
        errors.append("api port leaves no room for the local cluster port")
    return errors


def run_demo(config: DemoConfig) -> int:
    run = DemoRun(config)
    errors = _validate_layout(config)
    if errors:
        for error in errors:
            run.record("preflight", False, error)
        _write_report(run)
        for error in errors:
            print(f"[QLH-DEMO] ERROR {error}", file=sys.stderr)
        return 2

    try:
        run.record("preflight", True, f"mode={config.mode}")
        if config.mode == "live":
            backend_env = os.environ.copy()
            backend_env["QLH_API_PORT"] = str(config.api_port)
            backend_env["QLH_BACKEND_HOST"] = "127.0.0.1"
            backend_env["QLH_SERVER_IP"] = "127.0.0.1"
            backend_env["QLH_SERVER_PORT"] = str(_cluster_port(config))
            _start_process(
                run,
                "backend",
                [_live_python_command(), "-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1", "--port", str(config.api_port)],
                backend_env,
                "backend.log",
            )
            wait_for_health(_loopback_url(config.api_port, "/api/health"), config.startup_timeout)
            run.record("start-backend", True, "health=ok")

        if not config.skip_frontend:
            frontend_env = os.environ.copy()
            if config.mode == "live":
                frontend_env["QLH_VITE_API_TARGET"] = _loopback_url(config.api_port)
            _start_process(
                run,
                "frontend",
                [_npm_command(), "run", "dev", "--", "--host", "127.0.0.1", "--port", str(config.frontend_port)],
                frontend_env,
                "frontend.log",
            )
            wait_for_ready(_loopback_url(config.frontend_port), config.startup_timeout)
            run.record("start-frontend", True, "vite=ready")

        frontend_url = _loopback_url(config.frontend_port, "/?fixtures=1" if config.mode == "fixtures" else "")
        run.record("demo-url", True, frontend_url)
        _write_report(run)
        print(f"[QLH-DEMO] OK mode={config.mode}")
        print(f"[QLH-DEMO] URL {frontend_url}")
        if config.open_browser:
            webbrowser.open(frontend_url)
        if config.duration > 0:
            time.sleep(config.duration)
        elif run.processes:
            print("[QLH-DEMO] Press Ctrl+C to stop.")
            while all(item.process.poll() is None for item in run.processes):
                time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        run.record("stopped", True, "interrupted")
        return 0
    except Exception as exc:  # noqa: BLE001
        run.record("runtime", False, type(exc).__name__)
        print(f"[QLH-DEMO] ERROR {exc}", file=sys.stderr)
        return 1
    finally:
        _write_report(run)
        run.stop()


def _write_report(run: DemoRun) -> None:
    path = run.config.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "qlh.defense_demo.v1",
        "mode": run.config.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frontend_url": _loopback_url(run.config.frontend_port, "/?fixtures=1" if run.config.mode == "fixtures" else ""),
        "api_url": _loopback_url(run.config.api_port, "/api/health") if run.config.mode == "live" else None,
        "steps": run.steps,
        "logs": [_public_path(item.log_path) for item in run.processes],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLH defense demo launcher")
    parser.add_argument("--mode", choices=("fixtures", "live"), default="fixtures")
    parser.add_argument("--api-port", type=_port, default=8000)
    parser.add_argument("--frontend-port", type=_port, default=5174)
    parser.add_argument("--timeout", type=float, default=30.0, dest="startup_timeout")
    parser.add_argument("--duration", type=float, default=0.0, help="stop automatically after N seconds")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="open the local demo URL")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DemoConfig(
        mode=args.mode,
        api_port=args.api_port,
        frontend_port=args.frontend_port,
        startup_timeout=args.startup_timeout,
        duration=args.duration,
        open_browser=args.open_browser,
        skip_frontend=args.skip_frontend,
        report_path=args.report,
    )
    return run_demo(config)


if __name__ == "__main__":
    raise SystemExit(main())
