#!/usr/bin/env python3
"""Controlled OpenSSH client for the optional SYNC-SSH-S2 transport.

This module deliberately exposes only the S0 fixed helper protocol.  It never
opens a shell, accepts a password, or decides which commit should be applied.
The caller supplies a signed patch frame produced by the existing dispatcher;
the remote helper remains responsible for verifying that signature.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ssh_sync_contract import (
    HELPER_RESULT_SCHEMA,
    HELPER_SCHEMA,
    SshSyncContractError,
    SshSyncProfile,
    build_helper_request,
    build_helper_result,
    build_ssh_argv,
    load_profile,
    validate_helper_result,
)


DEFAULT_TIMEOUT_SECONDS = 20
MAX_HELPER_RESULT_BYTES = 8 * 1024
_ACTIONS = frozenset({"status", "fetch", "apply", "verify"})


class SshSyncClientError(ValueError):
    """Raised before any SSH process starts for an invalid local request."""


Runner = Callable[[Sequence[str], int], subprocess.CompletedProcess[str]]


def write_managed_known_hosts(profile: SshSyncProfile, path: str | os.PathLike[str]) -> Path:
    """Atomically replace a dedicated known-hosts file with this profile's key."""
    target = Path(path).expanduser().resolve()
    if not target.name:
        raise SshSyncClientError("known_hosts path is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=target.parent, delete=False,
        ) as handle:
            handle.write(profile.known_hosts_entry())
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise SshSyncClientError("managed known_hosts cannot be written") from exc
    return target


def normalize_request(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a request through the S0 contract before rendering SSH argv."""
    if not isinstance(raw, Mapping) or raw.get("schema") != HELPER_SCHEMA:
        raise SshSyncClientError("helper request schema is invalid")
    action = raw.get("action")
    if action not in _ACTIONS:
        raise SshSyncClientError("helper action is invalid")
    try:
        if action == "status":
            if set(raw) != {"schema", "action"}:
                raise SshSyncClientError("status request has unsupported fields")
            return build_helper_request("status")
        if action in {"fetch", "verify"}:
            if set(raw) != {"schema", "action", "commit_sha"}:
                raise SshSyncClientError(f"{action} request has unsupported fields")
            return build_helper_request(action, {"commit_sha": raw.get("commit_sha")})
        if set(raw) != {"schema", "action", "frame", "maintenance_lock"}:
            raise SshSyncClientError("apply request has unsupported fields")
        lock = raw.get("maintenance_lock")
        return build_helper_request("apply", {
            "frame": raw.get("frame"),
            "operation_id": lock.get("operation_id") if isinstance(lock, Mapping) else None,
        })
    except SshSyncContractError as exc:
        raise SshSyncClientError(str(exc)) from exc


def _default_runner(argv: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), shell=False, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_seconds,
    )


def _transport_failure(action: str) -> dict[str, Any]:
    return build_helper_result(action, status="error", error_code="transport_unavailable")


def _remote_rejection(action: str) -> dict[str, Any]:
    return build_helper_result(action, status="error", error_code="remote_helper_rejected")


def _parse_result(stdout: str, *, action: str) -> dict[str, Any] | None:
    encoded = stdout.encode("utf-8", errors="replace")
    if not stdout or len(encoded) > MAX_HELPER_RESULT_BYTES:
        return None
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        raw = json.loads(lines[0])
        return validate_helper_result(raw, expected_action=action)
    except (TypeError, ValueError, json.JSONDecodeError, SshSyncContractError):
        return None


def execute_helper(
    profile: SshSyncProfile,
    request: Mapping[str, Any],
    *,
    known_hosts_file: str | os.PathLike[str],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner = _default_runner,
) -> dict[str, Any]:
    """Run exactly one fixed helper action and return a contract-valid result.

    Transport failures do not expose process output because it can contain
    paths or server banners.  Valid helper error responses are preserved so
    callers can follow the fixed retry/fallback matrix.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 120:
        raise SshSyncClientError("SSH timeout must be an integer in 1..120")
    normalized = normalize_request(request)
    action = normalized["action"]
    known_hosts = write_managed_known_hosts(profile, known_hosts_file)
    argv = build_ssh_argv(profile, normalized, known_hosts_file=known_hosts)
    try:
        completed = runner(argv, timeout_seconds)
    except (OSError, subprocess.TimeoutExpired):
        return _transport_failure(action)
    result = _parse_result(completed.stdout, action=action)
    if result is None:
        # OpenSSH reserves 255 for client/transport failures.  A different
        # non-zero status means the authenticated remote command ran but did
        # not expose the fixed helper (or returned an invalid protocol frame).
        return _transport_failure(action) if completed.returncode == 255 else _remote_rejection(action)
    if completed.returncode and result["status"] == "ok":
        return _remote_rejection(action)
    return result


def build_acceptance_requests(
    *, commit_sha: str, frame: Mapping[str, Any], operation_id: str,
) -> list[dict[str, Any]]:
    """Return the explicit S2 sequence; callers choose whether to execute it."""
    return [
        build_helper_request("status"),
        build_helper_request("fetch", {"commit_sha": commit_sha}),
        build_helper_request("apply", {"frame": frame, "operation_id": operation_id}),
        build_helper_request("verify", {"commit_sha": commit_sha}),
    ]


def _load_json(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SshSyncClientError("JSON input cannot be read") from exc
    if not isinstance(value, dict):
        raise SshSyncClientError("JSON input must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QLH controlled SSH patch client")
    parser.add_argument("--profile", required=True, help="S0 SSH profile JSON, outside the checkout")
    parser.add_argument("--known-hosts", required=True, help="managed known_hosts path, outside the checkout")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("status")
    for name in ("fetch", "verify"):
        command = actions.add_parser(name)
        command.add_argument("--commit-sha", required=True)
    apply = actions.add_parser("apply")
    apply.add_argument("--frame", required=True, help="signed qlh.patch_frame.v1 JSON")
    apply.add_argument("--operation-id", required=True)
    args = parser.parse_args(argv)
    try:
        profile = load_profile(_load_json(args.profile))
        if args.action == "status":
            request = build_helper_request("status")
        elif args.action in {"fetch", "verify"}:
            request = build_helper_request(args.action, {"commit_sha": args.commit_sha})
        else:
            request = build_helper_request("apply", {
                "frame": _load_json(args.frame), "operation_id": args.operation_id,
            })
        result = execute_helper(
            profile, request, known_hosts_file=args.known_hosts, timeout_seconds=args.timeout,
        )
    except (SshSyncClientError, SshSyncContractError, ValueError) as exc:
        print(json.dumps({"schema": HELPER_RESULT_SCHEMA, "status": "error", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
