"""Validate an installed QLH Edge L runtime without importing the full stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "pandas",
    "einops",
    "tiktoken",
)
REQUIRED_MODULES = {
    "llama_cpp": "llama-cpp-python",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "psutil": "psutil",
    "httpx": "httpx",
}
REQUIRED_ROUTES = ("/health", "/status", "/generate")
PROBE_MARKER = "QLH_EDGE_PREFLIGHT="


def _venv_root(python_executable: Path) -> Path:
    executable = python_executable.resolve()
    if executable.parent.name.lower() in {"scripts", "bin"}:
        return executable.parent.parent
    return executable.parent


def _directory_size_mb(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    return round(total / 2**20, 2)


def _probe_code() -> str:
    forbidden = repr(list(FORBIDDEN_MODULES))
    required = repr(list(REQUIRED_MODULES))
    return f"""
import importlib.util
import json
import sys
import time

started = time.perf_counter()
payload = {{}}
try:
    import qlh_edge

    payload["import_elapsed_s"] = round(time.perf_counter() - started, 4)
    payload["routes"] = sorted({{route.path for route in qlh_edge.app.routes}})
    payload["forbidden_imported"] = [name for name in {forbidden} if name in sys.modules]
    payload["forbidden_installed"] = []
    for name in {forbidden}:
        try:
            if importlib.util.find_spec(name) is not None:
                payload["forbidden_installed"].append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
    payload["missing_required"] = []
    for name in {required}:
        try:
            if importlib.util.find_spec(name) is None:
                payload["missing_required"].append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            payload["missing_required"].append(name)
    payload["ok"] = True
except Exception as exc:
    payload = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}

print({PROBE_MARKER!r} + json.dumps(payload, sort_keys=True))
"""


def _run_probe(python_executable: Path, repository_root: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    python_path = [str(repository_root), str(repository_root / "src")]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    try:
        completed = subprocess.run(
            [str(python_executable), "-c", _probe_code()],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {"ok": False, "error": f"cannot start edge Python: {exc}", "probe_returncode": -1}
    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_MARKER):
            payload = json.loads(line[len(PROBE_MARKER) :])
            break
    if payload is None:
        payload = {
            "ok": False,
            "error": "edge import probe produced no result",
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    payload["probe_returncode"] = completed.returncode
    return payload


def run_preflight(
    python_executable: str | os.PathLike[str],
    repository_root: str | os.PathLike[str] | None = None,
    *,
    max_size_mb: float = 300.0,
    max_startup_s: float = 15.0,
) -> dict[str, Any]:
    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    python_path = Path(python_executable).resolve()
    venv_root = _venv_root(python_path)
    size_mb = _directory_size_mb(venv_root)
    probe = _run_probe(python_path, root)
    routes = set(probe.get("routes", []))
    import_elapsed_s = probe.get("import_elapsed_s")

    checks = {
        "python_exists": python_path.is_file(),
        "venv_size": size_mb <= max_size_mb,
        "cold_start": isinstance(import_elapsed_s, (int, float)) and import_elapsed_s <= max_startup_s,
        "required_modules": not probe.get("missing_required"),
        "no_forbidden_import": not probe.get("forbidden_imported"),
        "no_forbidden_install": not probe.get("forbidden_installed"),
        "required_routes": set(REQUIRED_ROUTES).issubset(routes),
        "probe": probe.get("ok") is True and probe.get("probe_returncode") == 0,
    }
    return {
        "ok": all(checks.values()),
        "python": str(python_path),
        "venv_root": str(venv_root),
        "venv_size_mb": size_mb,
        "max_size_mb": max_size_mb,
        "max_startup_s": max_startup_s,
        "checks": checks,
        "probe": probe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the QLH Edge L runtime contract.")
    parser.add_argument("--python", default=sys.executable, help="Edge venv Python executable")
    parser.add_argument("--max-size-mb", type=float, default=300.0)
    parser.add_argument("--max-startup-s", type=float, default=15.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result = run_preflight(
        args.python,
        max_size_mb=args.max_size_mb,
        max_startup_s=args.max_startup_s,
    )
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"edge python: {result['python']}")
        print(f"venv size: {result['venv_size_mb']} MB / {result['max_size_mb']} MB")
        print(f"cold start: {result['probe'].get('import_elapsed_s', 'n/a')} s / {result['max_startup_s']} s")
        for name, passed in result["checks"].items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        if result["probe"].get("error"):
            print(result["probe"]["error"], file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
