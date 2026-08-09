"""Cross-process SQLite ownership gate for M1.2.

The gate uses a fresh temporary directory for every scenario. It never opens
the configured user database. Run it once on Windows and once inside a Linux
environment with a native Node.js binary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_NODE_PROBE = ROOT / "control" / "dist" / "storage-environment-probe.js"


def _clean_environment(sqlite_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in list(environment):
        if name == "DATABASE_URL" or name.startswith("QLH_DB_"):
            environment.pop(name, None)
    environment["QLH_SQLITE_PATH"] = str(sqlite_path)
    environment["QLH_STATE_DIR"] = str(sqlite_path.parent)
    return environment


def _run(command: list[str], environment: dict[str, str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command returned no JSON: {' '.join(command)}")
    return json.loads(lines[-1])


def _python_probe(mode: str, sqlite_path: Path, environment: dict[str, str]) -> dict:
    return _run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--python-mode",
            mode,
            "--sqlite-path",
            str(sqlite_path),
        ],
        environment,
    )


def _node_probe(
    node_binary: str,
    node_probe: Path,
    mode: str,
    sqlite_path: Path,
    environment: dict[str, str],
) -> dict:
    return _run(
        [node_binary, str(node_probe), "--mode", mode, "--path", str(sqlite_path)],
        environment,
    )


def _seed_and_verify(
    sqlite_path: Path,
    node_binary: str,
    node_probe: Path,
) -> list[dict]:
    environment = _clean_environment(sqlite_path)
    return [
        _python_probe("seed", sqlite_path, environment),
        _node_probe(node_binary, node_probe, "seed", sqlite_path, environment),
        _python_probe("verify", sqlite_path, environment),
        _node_probe(node_binary, node_probe, "verify", sqlite_path, environment),
    ]


def _assert_unavailable(
    sqlite_path: Path,
    node_binary: str,
    node_probe: Path,
) -> list[dict]:
    environment = _clean_environment(sqlite_path)
    return [
        _python_probe("assert-unavailable", sqlite_path, environment),
        _node_probe(
            node_binary,
            node_probe,
            "assert-unavailable",
            sqlite_path,
            environment,
        ),
    ]


def _corrupt_database(sqlite_path: Path) -> None:
    with sqlite_path.open("r+b") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise RuntimeError("SQLite file is too short to corrupt safely")
        handle.seek(0)
        handle.write(b"QLH-CORRUPTED!!!")
        handle.flush()
        os.fsync(handle.fileno())


def _windows_identity() -> str:
    completed = subprocess.run(
        ["whoami"], capture_output=True, text=True, check=True, timeout=10,
    )
    identity = completed.stdout.strip()
    if not identity:
        raise RuntimeError("whoami returned an empty identity")
    return identity


def _set_read_only(directory: Path) -> object:
    if os.name == "nt":
        identity = _windows_identity()
        completed = subprocess.run(
            [
                "icacls",
                str(directory),
                "/deny",
                f"{identity}:(OI)(CI)(W)",
                "/T",
                "/C",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"icacls deny failed: {completed.stdout} {completed.stderr}")
        return identity

    if os.geteuid() == 0:
        raise RuntimeError("Linux read-only gate must run as a non-root user")
    files = [path for path in directory.iterdir() if path.is_file()]
    for path in files:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    directory.chmod(
        stat.S_IRUSR | stat.S_IXUSR
        | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH
    )
    return files


def _restore_writable(directory: Path, token: object) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["icacls", str(directory), "/remove:d", str(token), "/T", "/C"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"icacls restore failed: {completed.stdout} {completed.stderr}"
            )
        return

    directory.chmod(stat.S_IRWXU)
    for path in token:
        if Path(path).exists():
            Path(path).chmod(stat.S_IRUSR | stat.S_IWUSR)


def _scenario_clean(node_binary: str, node_probe: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="qlh-storage-clean-") as raw:
        sqlite_path = Path(raw) / "qlh-control.sqlite3"
        probes = _seed_and_verify(sqlite_path, node_binary, node_probe)
        return {"scenario": "clean", "passed": True, "probes": probes}


def _scenario_corrupt(node_binary: str, node_probe: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="qlh-storage-corrupt-") as raw:
        sqlite_path = Path(raw) / "qlh-control.sqlite3"
        _seed_and_verify(sqlite_path, node_binary, node_probe)
        _corrupt_database(sqlite_path)
        probes = _assert_unavailable(sqlite_path, node_binary, node_probe)
        return {"scenario": "corrupt", "passed": True, "probes": probes}


def _scenario_read_only(node_binary: str, node_probe: Path) -> dict:
    raw = tempfile.mkdtemp(prefix="qlh-storage-readonly-")
    directory = Path(raw)
    sqlite_path = directory / "qlh-control.sqlite3"
    token = None
    try:
        _seed_and_verify(sqlite_path, node_binary, node_probe)
        token = _set_read_only(directory)
        probes = _assert_unavailable(sqlite_path, node_binary, node_probe)
        return {"scenario": "read_only", "passed": True, "probes": probes}
    finally:
        if token is not None:
            _restore_writable(directory, token)
        shutil.rmtree(directory, ignore_errors=False)


def _python_mode(mode: str, sqlite_path: Path) -> None:
    os.environ.update(_clean_environment(sqlite_path))
    sys.path.insert(0, str(SRC))
    import local_store

    if mode == "assert-unavailable":
        try:
            local_store.initialize_local_store()
            health = local_store.local_store_health()
        except Exception as error:
            result = {
                "unavailable": True,
                "rejected_during_open": True,
                "error": str(error),
            }
        else:
            if health.get("status") != "unavailable" or health.get("writable"):
                raise RuntimeError("SQLite unexpectedly remained writable")
            result = {"unavailable": True, "health": health}
        print(json.dumps({"mode": mode, "path": str(sqlite_path), **result}))
        return

    if mode == "seed":
        local_store.initialize_local_store()
        local_store.create_local_session("python-clean", "Python clean seed")
        local_store.clear_local_conversation("python-clean")
        local_store.save_local_message("python-clean", "user", "python question")
        local_store.save_local_message(
            "python-clean",
            "assistant",
            "python answer",
            {"source": "python"},
        )
        local_store.increment_local_session_message_count("python-clean")
        local_store.set_local_user_settings({"saveHistory": True, "gate": "clean"})
        print(json.dumps({"mode": mode, "path": str(sqlite_path), "seeded": True}))
        return

    if mode == "verify":
        local_store.initialize_local_store()
        node_session = local_store.get_local_session("node-clean")
        node_messages = local_store.load_local_conversation("node-clean")
        settings = local_store.get_local_user_settings()
        connection = sqlite3.connect(sqlite_path)
        try:
            schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
            marker = connection.execute(
                "SELECT value FROM cluster_settings WHERE key = 'environment_gate_node'"
            ).fetchone()
        finally:
            connection.close()
        if not node_session or len(node_messages) != 1:
            raise RuntimeError("Node assets are not visible to Python")
        if settings.get("gate") != "clean" or marker != ("ready",):
            raise RuntimeError("shared settings are not intact")
        if schema_version != 5:
            raise RuntimeError(f"unexpected schema version {schema_version}")
        print(json.dumps({
            "mode": mode,
            "path": str(sqlite_path),
            "schema_version": schema_version,
            "verified": True,
        }))
        return

    raise RuntimeError(f"unsupported Python probe mode {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("all", "clean", "corrupt", "read-only"),
        default="all",
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--node-probe", type=Path, default=DEFAULT_NODE_PROBE)
    parser.add_argument(
        "--python-mode",
        choices=("seed", "verify", "assert-unavailable"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--sqlite-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.python_mode:
        if args.sqlite_path is None:
            parser.error("--sqlite-path is required with --python-mode")
        _python_mode(args.python_mode, args.sqlite_path.resolve())
        return 0

    node_probe = args.node_probe.resolve()
    if not node_probe.is_file():
        raise RuntimeError(f"Node probe is missing: {node_probe}; run control build first")

    scenarios = []
    if args.scenario in ("all", "clean"):
        scenarios.append(_scenario_clean(args.node, node_probe))
    if args.scenario in ("all", "corrupt"):
        scenarios.append(_scenario_corrupt(args.node, node_probe))
    if args.scenario in ("all", "read-only"):
        scenarios.append(_scenario_read_only(args.node, node_probe))
    print(json.dumps({
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "node": args.node,
        "passed": all(item["passed"] for item in scenarios),
        "scenarios": scenarios,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
