"""QLH policy adapter for the reusable spawnledger ownership package."""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = ROOT / "frontend_cybergothic"
OWNERSHIP_ROOT = ROOT / "build" / "defense-runtime" / "ownership"
_PACKAGE_ROOT = ROOT / "packages" / "spawnledger"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from spawnledger import (  # noqa: E402
    IDENTITY_KEYS,
    RECEIPT_KEYS,
    RUN_ID_PATTERN,
    OwnershipLedger as BaseOwnershipLedger,
    OwnershipPolicy,
    inspect_identity as inspect_with_policy,
    load_receipt as load_with_policy,
)


RECEIPT_SCHEMA = "qlh.defense_process_ownership.v1"
ALLOWED_KINDS = {"frontend", "backend", "failure-worker", "benchmark-worker"}


def _npm_candidates() -> tuple[frozenset[Path], frozenset[Path]]:
    launchers: set[Path] = set()
    cli_scripts: set[Path] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for executable in ("npm", "npm.cmd"):
            launcher = Path(directory) / executable
            if not launcher.is_file():
                continue
            resolved = launcher.resolve()
            launchers.add(resolved)
            if resolved.name.lower() == "npm-cli.js":
                cli_scripts.add(resolved)
            candidate = (launcher.parent / "node_modules" / "npm" / "bin" / "npm-cli.js").resolve()
            if candidate.is_file():
                cli_scripts.add(candidate)
    return frozenset(launchers), frozenset(cli_scripts)


NPM_LAUNCHER_CANDIDATES, NPM_CLI_CANDIDATES = _npm_candidates()


def _path_launchers(*names: str) -> frozenset[Path]:
    candidates: set[Path] = set()
    for name in names:
        located = shutil.which(name)
        if located:
            candidates.add(Path(located).resolve())
    return frozenset(candidates)


CMD_LAUNCHER_CANDIDATES = _path_launchers("cmd", "cmd.exe")
NODE_LAUNCHER_CANDIDATES = _path_launchers("node", "node.exe")
PYTHON_LAUNCHER = Path(sys.executable).resolve()


def _trusted_launcher(value: str, candidates: frozenset[Path]) -> bool:
    path = Path(value.strip('"'))
    if path.parent == Path("."):
        located = shutil.which(str(path))
        return located is not None and any(_same_path(located, candidate) for candidate in candidates)
    return any(_same_path(path, candidate) for candidate in candidates)


def _trusted_npm_launcher(value: str) -> bool:
    return _trusted_launcher(value, NPM_LAUNCHER_CANDIDATES)


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


def _has_pair(command: Sequence[str], option: str, value: str) -> bool:
    lowered = [item.lower() for item in command]
    return any(
        lowered[index] == option and lowered[index + 1] == value
        for index in range(len(lowered) - 1)
    )


def _vite_arguments(arguments: Sequence[str]) -> bool:
    normalized = [item.lower() for item in arguments]
    if normalized[:1] == ["--"]:
        normalized = normalized[1:]
    if normalized == ["--host", "127.0.0.1"]:
        return True
    if len(normalized) != 4 or normalized[:3] != ["--host", "127.0.0.1", "--port"]:
        return False
    try:
        port = int(normalized[3])
    except ValueError:
        return False
    return str(port) == normalized[3] and 1 <= port <= 65535


def _command_matches(kind: str, command: Sequence[str], cwd: str) -> bool:
    """QLH command allowlist; receipt content cannot expand this authority."""
    if not command:
        return False
    executable_name = Path(command[0]).name.lower()
    lowered = [item.lower() for item in command]
    if kind == "frontend":
        if not _same_path(cwd, FRONTEND_ROOT):
            return False
        if _has_pair(command, "--host", "127.0.0.1"):
            if executable_name in {"cmd", "cmd.exe"} and _trusted_launcher(command[0], CMD_LAUNCHER_CANDIDATES):
                has_shell_operator = any(item in {"&", "&&", "|", "||"} for item in command)
                command_indexes = [
                    index for index, item in enumerate(lowered[:-1])
                    if item in {"/c", "/k"}
                ]
                if command_indexes and not has_shell_operator:
                    payload = command[command_indexes[-1] + 1:]
                    payload_lowered = [item.lower() for item in payload]
                    launcher = Path(payload[0].strip('"')).name.lower() if payload else ""
                    if launcher in {"vite", "vite.cmd"} and _vite_arguments(payload[1:]):
                        return True
                    if (
                        launcher in {"npm", "npm.cmd"}
                        and _trusted_npm_launcher(payload[0])
                        and payload_lowered[1:3] == ["run", "dev"]
                        and _vite_arguments(payload[3:])
                    ):
                        return True
            if (
                executable_name in {"node", "node.exe"}
                and _trusted_launcher(command[0], NODE_LAUNCHER_CANDIDATES)
                and len(command) >= 2
            ):
                script = Path(command[1])
                if (
                    script.name.lower() == "npm-cli.js"
                    and Path(script).resolve() in NPM_CLI_CANDIDATES
                    and lowered[2:4] == ["run", "dev"]
                    and _vite_arguments(command[4:])
                ):
                    return True
                if (
                    script.name.lower() == "vite.js"
                    and _under_path(script, FRONTEND_ROOT / "node_modules" / "vite")
                    and _vite_arguments(command[2:])
                ):
                    return True
        if len(command) != 3:
            return False
        try:
            executable = Path(command[0]).resolve()
            relative = executable.relative_to((FRONTEND_ROOT / "node_modules").resolve())
        except (OSError, RuntimeError, ValueError):
            return False
        executable_name = relative.name.lower()
        esbuild_package = "@esbuild" in {
            part.lower() for part in relative.parts
        } or relative.as_posix().lower().startswith("esbuild/bin/")
        return (
            executable_name in {"esbuild", "esbuild.exe"}
            and esbuild_package
            and re.fullmatch(r"--service=\d+\.\d+\.\d+", command[1]) is not None
            and command[2] == "--ping"
        )
    if kind == "backend":
        return (
            _same_path(cwd, ROOT)
            and _same_path(command[0], PYTHON_LAUNCHER)
            and lowered[1:4] == ["-m", "uvicorn", "src.api_server:app"]
            and _has_pair(command, "--host", "127.0.0.1")
        )
    if kind == "failure-worker":
        return (
            _same_path(cwd, ROOT)
            and _same_path(command[0], PYTHON_LAUNCHER)
            and len(command) >= 2
            and _same_path(command[1], ROOT / "scripts" / "demo" / "failure_worker.py")
        )
    if kind == "benchmark-worker":
        return (
            _same_path(cwd, ROOT)
            and _same_path(command[0], PYTHON_LAUNCHER)
            and len(command) >= 2
            and _same_path(command[1], ROOT / "scripts" / "demo" / "benchmark_worker.py")
        )
    return False


QLH_OWNERSHIP_POLICY = OwnershipPolicy(
    schema=RECEIPT_SCHEMA,
    allowed_sources=frozenset({"defense_demo", "defense_benchmark"}),
    matchers={
        kind: (lambda command, cwd, selected=kind: _command_matches(selected, command, cwd))
        for kind in ALLOWED_KINDS
    },
)


class OwnershipLedger(BaseOwnershipLedger):
    """Backward-compatible QLH ledger with the defense-demo policy prebound."""

    def __init__(
        self,
        source: str,
        *,
        ports: list[int] | None = None,
        ownership_root: Path = OWNERSHIP_ROOT,
    ) -> None:
        super().__init__(
            source,
            policy=QLH_OWNERSHIP_POLICY,
            ports=ports,
            ownership_root=ownership_root,
        )


def inspect_identity(identity: object):
    return inspect_with_policy(identity, QLH_OWNERSHIP_POLICY)


def load_receipt(path: Path) -> dict[str, Any]:
    return load_with_policy(path, QLH_OWNERSHIP_POLICY)
