"""Optional native llama.cpp RPC worker for the Edge runtime.

The Edge process always owns local GGUF inference.  This module only manages
an explicitly configured native ``rpc-server`` child; it never turns local
generation into an HTTP proxy and never imports the PyTorch control stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any


class EdgeRpcWorkerError(RuntimeError):
    """Raised when the native RPC worker cannot be started or stopped."""


def _rpc_port() -> int:
    raw = os.environ.get("QLH_EDGE_RPC_PORT", "50052").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise EdgeRpcWorkerError("QLH_EDGE_RPC_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise EdgeRpcWorkerError("QLH_EDGE_RPC_PORT must be between 1 and 65535")
    return port


def _rpc_host() -> str:
    host = os.environ.get("QLH_EDGE_RPC_HOST", "127.0.0.1").strip()
    if not host or len(host) > 253 or any(char.isspace() for char in host):
        raise EdgeRpcWorkerError("QLH_EDGE_RPC_HOST is invalid")
    return host


class EdgeRpcWorker:
    """Own one optional ``llama.cpp`` RPC worker process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[Any] | None = None

    @staticmethod
    def _configured_command() -> str:
        return os.environ.get("QLH_EDGE_RPC_SERVER", "").strip()

    def _live_process(self) -> subprocess.Popen[Any] | None:
        process = self._process
        if process is not None and process.poll() is not None:
            self._process = None
            process = None
        return process

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            process = self._live_process()
            configured = bool(self._configured_command())
            host = os.environ.get("QLH_EDGE_RPC_HOST", "127.0.0.1").strip()
            port = os.environ.get("QLH_EDGE_RPC_PORT", "50052").strip()
            return {
                "transport": "llama.cpp-rpc",
                "role": "rpc_worker",
                "configured": configured,
                "running": process is not None,
                "pid": process.pid if process is not None else None,
                "address": f"{host}:{port}" if configured else None,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._live_process() is not None:
                return self.snapshot()

            command = self._configured_command()
            if not command:
                raise EdgeRpcWorkerError(
                    "QLH_EDGE_RPC_SERVER is required to start the native RPC worker"
                )
            executable = str(Path(command).expanduser())
            resolved = shutil.which(executable)
            if resolved is None:
                raise EdgeRpcWorkerError("configured QLH_EDGE_RPC_SERVER is not executable")
            host = _rpc_host()
            port = _rpc_port()
            try:
                self._process = subprocess.Popen(
                    [resolved, "--host", host, "--port", str(port)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=os.name != "nt",
                )
            except OSError as exc:
                raise EdgeRpcWorkerError("native RPC worker failed to start") from exc
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._live_process()
            if process is None:
                return self.snapshot()
            try:
                process.terminate()
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            except OSError as exc:
                raise EdgeRpcWorkerError("native RPC worker failed to stop") from exc
            finally:
                self._process = None
            return self.snapshot()

