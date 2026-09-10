"""Crash-resilient ownership receipts for defense-demo child processes.

Receipts are hints, never authority: reset code must re-check PID creation time,
working directory, and a hard-coded command signature before stopping anything.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = ROOT / "frontend_cybergothic"
OWNERSHIP_ROOT = ROOT / "build" / "defense-runtime" / "ownership"
RECEIPT_SCHEMA = "qlh.defense_process_ownership.v1"
ALLOWED_KINDS = {"frontend", "backend", "failure-worker", "benchmark-worker"}
RECEIPT_KEYS = {"schema", "run_id", "source", "created_by_pid", "ports", "processes"}
IDENTITY_KEYS = {"pid", "create_time", "kind", "root_pid"}
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def _same_path(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, RuntimeError):
        return False


def _under_path(value: str | Path, parent: Path) -> bool:
    try:
        Path(value).resolve().relative_to(parent.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _has_pair(command: list[str], option: str, value: str) -> bool:
    lowered = [item.lower() for item in command]
    return any(
        lowered[index] == option and lowered[index + 1] == value
        for index in range(len(lowered) - 1)
    )


def _command_matches(kind: str, command: list[str], cwd: str) -> bool:
    """Hard-coded allowlist; receipt content cannot expand this authority."""
    if not command:
        return False
    executable_name = Path(command[0]).name.lower()
    lowered = [item.lower() for item in command]
    if kind == "frontend":
        if not _same_path(cwd, FRONTEND_ROOT):
            return False
        if _has_pair(command, "--host", "127.0.0.1"):
            if executable_name in {"cmd", "cmd.exe"}:
                is_npm = any(Path(item).name.lower() in {"npm", "npm.cmd"} for item in command[1:])
                is_vite = any(item.lower() == "vite" for item in command[1:])
                if is_vite or (is_npm and "run" in lowered and "dev" in lowered):
                    return True
            if executable_name in {"node", "node.exe"} and len(command) >= 2:
                script = Path(command[1])
                if script.name.lower() == "npm-cli.js" and "run" in lowered and "dev" in lowered:
                    return True
                if (
                    script.name.lower() == "vite.js"
                    and _under_path(script, FRONTEND_ROOT / "node_modules" / "vite")
                ):
                    return True
        # Vite owns one esbuild service child.  Accept only its installed
        # binary under this frontend and its exact service/ping argument form.
        if len(command) != 3:
            return False
        try:
            executable = Path(command[0]).resolve()
            relative = executable.relative_to((FRONTEND_ROOT / "node_modules").resolve())
        except (OSError, RuntimeError, ValueError):
            return False
        executable_name = relative.name.lower()
        esbuild_package = "@esbuild" in {part.lower() for part in relative.parts} or relative.as_posix().lower().startswith("esbuild/bin/")
        return (
            executable_name in {"esbuild", "esbuild.exe"}
            and esbuild_package
            and re.fullmatch(r"--service=\d+\.\d+\.\d+", command[1]) is not None
            and command[2] == "--ping"
        )
    if kind == "backend":
        return (
            _same_path(cwd, ROOT)
            and executable_name in {"python", "python.exe"}
            and any(lowered[index:index + 2] == ["-m", "uvicorn"] for index in range(len(lowered) - 1))
            and "src.api_server:app" in lowered
            and _has_pair(command, "--host", "127.0.0.1")
        )
    if kind == "failure-worker":
        return (
            _same_path(cwd, ROOT)
            and executable_name in {"python", "python.exe"}
            and len(command) >= 2
            and _same_path(command[1], ROOT / "scripts" / "demo" / "failure_worker.py")
        )
    if kind == "benchmark-worker":
        return (
            _same_path(cwd, ROOT)
            and executable_name in {"python", "python.exe"}
            and len(command) >= 2
            and _same_path(command[1], ROOT / "scripts" / "demo" / "benchmark_worker.py")
        )
    return False


def inspect_identity(identity: object) -> tuple[str, psutil.Process | None]:
    """Return matched/dead/mismatch/invalid without exposing the command line."""
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        return "invalid", None
    pid = identity.get("pid")
    create_time = identity.get("create_time")
    kind = identity.get("kind")
    root_pid = identity.get("root_pid")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(root_pid, bool)
        or not isinstance(root_pid, int)
        or root_pid <= 0
        or isinstance(create_time, bool)
        or not isinstance(create_time, (int, float))
        or kind not in ALLOWED_KINDS
    ):
        return "invalid", None
    try:
        process = psutil.Process(pid)
        actual_create_time = process.create_time()
    except psutil.NoSuchProcess:
        return "dead", None
    except (psutil.AccessDenied, OSError):
        return "mismatch", None
    if abs(actual_create_time - float(create_time)) > 0.05:
        return "mismatch", None
    try:
        command = process.cmdline()
        cwd = process.cwd()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "dead", None
    except (psutil.AccessDenied, OSError):
        return "mismatch", None
    if not _command_matches(str(kind), command, cwd):
        return "mismatch", None
    return "matched", process


def _identity(process: psutil.Process, *, kind: str, root_pid: int) -> dict[str, Any] | None:
    try:
        create_time = process.create_time()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
        return None
    return {
        "pid": process.pid,
        "create_time": create_time,
        "kind": kind,
        "root_pid": root_pid,
    }


class OwnershipLedger:
    """Persist identities for children that may outlive an abruptly killed launcher."""

    def __init__(self, source: str, *, ports: list[int] | None = None, ownership_root: Path = OWNERSHIP_ROOT):
        self.source = source
        self.ports = sorted(set(ports or []))
        self.run_id = secrets.token_hex(12)
        self.ownership_root = ownership_root
        self.path = ownership_root / f"{self.run_id}.json"
        self._roots: dict[int, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, kind: str, pid: int) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unapproved demo process kind: {kind}")
        with self._lock:
            self._roots[pid] = kind
            self._refresh_locked()
            if self._thread is None:
                self._thread = threading.Thread(target=self._monitor, name="demo-ownership", daemon=True)
                self._thread.start()

    def _monitor(self) -> None:
        while not self._stop.wait(0.2):
            with self._lock:
                self._refresh_locked()

    def refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        identities: dict[tuple[int, float], dict[str, Any]] = {}
        for root_pid, kind in list(self._roots.items()):
            try:
                root_process = psutil.Process(root_pid)
                processes = [root_process, *root_process.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, OSError):
                continue
            for process in processes:
                item = _identity(process, kind=kind, root_pid=root_pid)
                if item is not None:
                    identities[(item["pid"], item["create_time"])] = item
        if not identities:
            return
        payload = {
            "schema": RECEIPT_SCHEMA,
            "run_id": self.run_id,
            "source": self.source,
            "created_by_pid": os.getpid(),
            "ports": self.ports,
            "processes": sorted(identities.values(), key=lambda item: (item["root_pid"], item["pid"])),
        }
        self.ownership_root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._refresh_locked()
            if self.path.is_file():
                try:
                    payload = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    return
                states = [inspect_identity(item)[0] for item in payload.get("processes", [])]
                if all(state == "dead" for state in states):
                    self.path.unlink(missing_ok=True)


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"ownership receipt unreadable: {type(exc).__name__}") from exc
    if not isinstance(payload, dict) or set(payload) != RECEIPT_KEYS or payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("ownership receipt schema invalid")
    if not RUN_ID_PATTERN.fullmatch(path.stem) or path.stem != payload.get("run_id"):
        raise ValueError("ownership receipt run id mismatch")
    if payload.get("source") not in {"defense_demo", "defense_benchmark"}:
        raise ValueError("ownership receipt source invalid")
    created_by_pid = payload.get("created_by_pid")
    if isinstance(created_by_pid, bool) or not isinstance(created_by_pid, int) or created_by_pid <= 0:
        raise ValueError("ownership receipt creator invalid")
    ports = payload.get("ports")
    if (
        not isinstance(ports, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 65535 for item in ports)
        or len(set(ports)) != len(ports)
    ):
        raise ValueError("ownership receipt ports invalid")
    processes = payload.get("processes")
    if not isinstance(processes, list) or not processes:
        raise ValueError("ownership receipt processes invalid")
    if any(not isinstance(item, dict) or set(item) != IDENTITY_KEYS for item in processes):
        raise ValueError("ownership receipt process identity invalid")
    for item in processes:
        if (
            isinstance(item.get("pid"), bool)
            or not isinstance(item.get("pid"), int)
            or item["pid"] <= 0
            or isinstance(item.get("root_pid"), bool)
            or not isinstance(item.get("root_pid"), int)
            or item["root_pid"] <= 0
            or isinstance(item.get("create_time"), bool)
            or not isinstance(item.get("create_time"), (int, float))
            or item.get("kind") not in ALLOWED_KINDS
        ):
            raise ValueError("ownership receipt process identity invalid")
    identity_keys = [(item.get("pid"), item.get("create_time")) for item in processes]
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("ownership receipt process identity duplicated")
    process_pids = {item.get("pid") for item in processes}
    if any(item.get("root_pid") not in process_pids for item in processes):
        raise ValueError("ownership receipt root identity missing")
    root_kinds: dict[object, object] = {}
    for item in processes:
        root_pid = item.get("root_pid")
        kind = item.get("kind")
        if root_pid in root_kinds and root_kinds[root_pid] != kind:
            raise ValueError("ownership receipt root kind mismatch")
        root_kinds[root_pid] = kind
    return payload
