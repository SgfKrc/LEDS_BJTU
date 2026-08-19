#!/usr/bin/env python3
"""Fixed, local-only helper used by SYNC-SSH-S1/S2.

The helper is a library and a JSONL command-line entry point. It never invokes
an SSH client or a shell. A future forced-command SSH account may expose only
``python -m ssh_sync_helper --repo ...`` and pass one JSON request per line.
All state is kept outside the managed checkout.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packaging"))

from ssh_sync_contract import (
    FAILURE_MATRIX,
    HELPER_LEDGER_SCHEMA,
    HELPER_RESULT_SCHEMA,
    HELPER_SCHEMA,
    LOCK_SCHEMA,
    REMOTE_HELPER,
    build_helper_result,
    build_helper_request,
    build_maintenance_lock,
    validate_helper_result,
)

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FRAME_SCHEMA = "qlh.patch_frame.v1"
_MAX_REQUEST_BYTES = 32 * 1024
_HELPER_CONFIG_SCHEMA = "qlh.sync_ssh.helper-config.v1"
_HELPER_CONFIG_ENV = "QLH_SSH_SYNC_HELPER_CONFIG"


class HelperError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class HelperLedger:
    """Durable operation ledger; only a digest and terminal status are stored."""

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.path = Path(state_dir).expanduser().resolve() / "ssh-sync-helper-ledger.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": HELPER_LEDGER_SCHEMA, "operations": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HelperError("remote_helper_rejected", "helper ledger unreadable") from exc
        if not isinstance(value, dict) or value.get("schema") != HELPER_LEDGER_SCHEMA or not isinstance(value.get("operations"), dict):
            raise HelperError("remote_helper_rejected", "helper ledger schema invalid")
        return value

    def _write(self, value: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
                json.dump(value, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise HelperError("remote_helper_rejected", "helper ledger cannot be persisted") from exc

    def get(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read()["operations"].get(operation_id)

    def put(self, operation_id: str, result: Mapping[str, Any]) -> None:
        with self._lock:
            value = self._read()
            value["operations"][operation_id] = {
                "status": result.get("status"),
                "action": result.get("action"),
                "commit_sha": result.get("commit_sha", ""),
                "error_code": result.get("error_code", ""),
            }
            self._write(value)


class MaintenanceLock:
    """Exclusive process/file lock for a managed checkout mutation."""

    _thread_lock = threading.Lock()

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.path = Path(state_dir).expanduser().resolve() / "ssh-sync-maintenance.lock"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, operation_id: str):
        with self._thread_lock:
            if self.path.exists():
                try:
                    current = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise HelperError("remote_helper_rejected", "maintenance lock is unreadable") from exc
                if not isinstance(current, dict) or current.get("schema") != LOCK_SCHEMA:
                    raise HelperError("remote_helper_rejected", "maintenance lock schema invalid")
                raise HelperError("maintenance_lock_held", "maintenance lock is already held")
            try:
                with self.path.open("x", encoding="utf-8") as handle:
                    json.dump(build_maintenance_lock(operation_id), handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise HelperError("maintenance_lock_held", "maintenance lock is already held") from exc
        try:
            yield
        finally:
            with self._thread_lock:
                try:
                    current = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    current = None
                if isinstance(current, dict) and current.get("operation_id") == operation_id:
                    self.path.unlink(missing_ok=True)


def _git(repo: Path, args: list[str], *, check: bool = True) -> str:
    if any(not isinstance(arg, str) or "\x00" in arg for arg in args):
        raise HelperError("remote_helper_rejected", "git argument contains invalid bytes")
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    if check and result.returncode != 0:
        raise HelperError("remote_helper_rejected", "managed git operation failed")
    return result.stdout.strip()


def _validate_repo(repo: Path) -> None:
    if not repo.is_dir() or not (repo / ".git").exists():
        raise HelperError("repository_unmanaged", "target is not a managed git checkout")
    if _git(repo, ["rev-parse", "--is-inside-work-tree"]) != "true":
        raise HelperError("repository_unmanaged", "target is not an ordinary work tree")


def _validate_external_path(path: Path, repo: Path, *, label: str) -> None:
    """Prevent helper state and trust material from being reset with the code."""
    try:
        path.relative_to(repo)
    except ValueError:
        return
    raise HelperError("repository_unmanaged", f"{label} must be outside the managed checkout")


def _verify_frame(frame: Mapping[str, Any], public_key_path: Path, repo: Path) -> None:
    """Verify the existing signed patch frame before any fetch/reset."""
    if frame.get("schema") != _FRAME_SCHEMA or frame.get("branch") != "dev":
        raise HelperError("patch_signature_invalid", "patch frame schema or branch invalid")
    commit = frame.get("commit_sha")
    if not isinstance(commit, str) or not _COMMIT_SHA_RE.fullmatch(commit):
        raise HelperError("patch_signature_invalid", "patch frame commit SHA invalid")
    signature = frame.get("signature")
    key_id = frame.get("key_id")
    if not isinstance(signature, str) or not isinstance(key_id, str):
        raise HelperError("patch_signature_invalid", "patch frame signature is missing")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from signing import canonical_json, load_public_key_file
        key_info = load_public_key_file(public_key_path)
        if key_info.get("key_id") and key_id != key_info["key_id"]:
            raise HelperError("patch_signature_invalid", "patch frame key id mismatch")
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(key_info["public_key"], validate=True))
        signed = {key: value for key, value in frame.items() if key not in {"signature", "key_id"}}
        public_key.verify(base64.b64decode(signature, validate=True), canonical_json(signed))
    except HelperError:
        raise
    except Exception as exc:
        raise HelperError("patch_signature_invalid", "patch frame signature verification failed") from exc
    expected_repo = _git(repo, ["remote", "get-url", "origin"])
    if not isinstance(frame.get("repo"), str) or frame["repo"] != expected_repo:
        raise HelperError("patch_signature_invalid", "patch frame repository does not match managed origin")


def handle_request(
    request: Mapping[str, Any], *, repo: str | os.PathLike[str], state_dir: str | os.PathLike[str],
    verify_key: str | os.PathLike[str], fault: str = "",
) -> dict[str, Any]:
    """Execute exactly one helper request in a managed checkout."""
    if not isinstance(request, Mapping) or request.get("schema") != HELPER_SCHEMA:
        return build_helper_result("status", status="error", error_code="helper_action_invalid")
    action = request.get("action")
    if action not in {"status", "fetch", "apply", "verify"}:
        return build_helper_result("status", status="error", error_code="helper_action_invalid")
    operation_id = ""
    if action == "apply":
        lock = request.get("maintenance_lock")
        operation_id = str(lock.get("operation_id", "")) if isinstance(lock, Mapping) else ""
        try:
            build_maintenance_lock(operation_id)
        except Exception:
            return build_helper_result(action, status="error", error_code="helper_action_invalid")
    if fault == "disconnect_before":
        raise ConnectionError("simulated SSH disconnect before helper response")
    target_repo = Path(repo).expanduser().resolve()
    try:
        _validate_repo(target_repo)
        external_state_dir = Path(state_dir).expanduser().resolve()
        external_verify_key = Path(verify_key).expanduser().resolve()
        _validate_external_path(external_state_dir, target_repo, label="helper state directory")
        _validate_external_path(external_verify_key, target_repo, label="helper verification key")
        if not external_verify_key.is_file():
            raise HelperError("patch_signature_invalid", "helper verification key is unavailable")
        ledger = HelperLedger(external_state_dir)
        if action == "status":
            head = _git(target_repo, ["rev-parse", "HEAD"])
            result = build_helper_result(action, commit_sha=head)
        elif action in {"fetch", "verify"}:
            commit = request.get("commit_sha")
            if not isinstance(commit, str) or not _COMMIT_SHA_RE.fullmatch(commit):
                raise HelperError("helper_action_invalid", "commit SHA invalid")
            if action == "fetch":
                _git(target_repo, ["fetch", "origin", "dev"])
            if action == "verify" and _git(target_repo, ["rev-parse", "HEAD"]) != commit:
                raise HelperError("remote_helper_rejected", "HEAD does not match requested commit")
            if action == "fetch":
                remote_tip = _git(target_repo, ["rev-parse", "--verify", "origin/dev"])
                resolved = _git(target_repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
                if resolved != commit or _git(target_repo, ["merge-base", commit, remote_tip]) != commit:
                    raise HelperError("remote_helper_rejected", "requested commit is not reachable from origin/dev")
            result = build_helper_result(action, commit_sha=commit)
        else:
            frame = request.get("frame")
            if not isinstance(frame, Mapping):
                raise HelperError("patch_signature_invalid", "signed patch frame missing")
            _verify_frame(frame, external_verify_key, target_repo)
            commit = str(frame["commit_sha"])
            with MaintenanceLock(external_state_dir).acquire(operation_id):
                prior = ledger.get(operation_id)
                if prior is not None:
                    if (
                        prior.get("status") != "ok"
                        or prior.get("action") != action
                        or prior.get("commit_sha") != commit
                    ):
                        raise HelperError("remote_helper_rejected", "helper ledger operation is invalid")
                    return build_helper_result(action, commit_sha=prior.get("commit_sha") or None)
                if fault == "disconnect_during_apply":
                    raise ConnectionError("simulated SSH disconnect during apply")
                if _git(target_repo, ["status", "--porcelain"]):
                    raise HelperError("workspace_dirty", "managed checkout has local changes")
                _git(target_repo, ["fetch", "origin", "dev"])
                remote_tip = _git(target_repo, ["rev-parse", "--verify", "origin/dev"])
                resolved = _git(target_repo, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
                if resolved != commit or _git(target_repo, ["merge-base", commit, remote_tip]) != commit:
                    raise HelperError("remote_helper_rejected", "signed target is not reachable from origin/dev")
                _git(target_repo, ["reset", "--hard", commit])
                result = build_helper_result(action, commit_sha=_git(target_repo, ["rev-parse", "HEAD"]))
                ledger.put(operation_id, result)
                if fault == "disconnect_after_apply":
                    raise ConnectionError("simulated SSH disconnect after apply before ACK")
        return result
    except ConnectionError:
        raise
    except HelperError as exc:
        result = build_helper_result(action, status="error", error_code=exc.code)
        return result


class FakeSshDisconnected(ConnectionError):
    """S1-only signal that the fake SSH connection closed without a result."""


class FakeSshServer:
    """In-process, byte-bounded SSH helper loopback used only by S1 tests.

    It serializes both directions like a real command channel, but intentionally
    opens no TCP socket and has no authentication behavior. S2 must still prove
    the OpenSSH/Tailscale path with the S0 profile and known-hosts gate.
    """

    _FAULTS = {"", "disconnect_before", "disconnect_during_apply", "disconnect_after_apply"}

    def __init__(self, *, repo: str | os.PathLike[str], state_dir: str | os.PathLike[str], verify_key: str | os.PathLike[str]):
        self.repo = repo
        self.state_dir = state_dir
        self.verify_key = verify_key

    def exchange(self, request: Mapping[str, Any], *, fault: str = "") -> dict[str, Any]:
        if fault not in self._FAULTS:
            raise ValueError("unknown fake SSH fault")
        wire = json.dumps(dict(request), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(wire) > _MAX_REQUEST_BYTES:
            return build_helper_result("status", status="error", error_code="helper_payload_too_large")
        parsed = json.loads(wire.decode("utf-8"))
        action = parsed.get("action") if isinstance(parsed, dict) else "status"
        if action not in {"status", "fetch", "apply", "verify"}:
            action = "status"
        try:
            if action == "status":
                if set(parsed) != {"schema", "action"}:
                    raise ValueError("status fields")
                parsed = build_helper_request("status")
            elif action in {"fetch", "verify"}:
                if set(parsed) != {"schema", "action", "commit_sha"}:
                    raise ValueError("fetch/verify fields")
                parsed = build_helper_request(action, {"commit_sha": parsed.get("commit_sha")})
            else:
                if set(parsed) != {"schema", "action", "frame", "maintenance_lock"}:
                    raise ValueError("apply fields")
                lock = parsed.get("maintenance_lock")
                parsed = build_helper_request("apply", {
                    "frame": parsed.get("frame"),
                    "operation_id": lock.get("operation_id") if isinstance(lock, Mapping) else None,
                })
        except Exception:
            return build_helper_result(action, status="error", error_code="helper_action_invalid")
        try:
            result = handle_request(
                parsed, repo=self.repo, state_dir=self.state_dir,
                verify_key=self.verify_key, fault=fault,
            )
        except ConnectionError as exc:
            raise FakeSshDisconnected("fake SSH connection closed") from exc
        response_wire = json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        response = json.loads(response_wire.decode("utf-8"))
        return validate_helper_result(response, expected_action=action)


def _load_helper_config() -> dict[str, str]:
    """Load deployment-owned config; the remote client cannot set these paths."""
    configured = os.environ.get(_HELPER_CONFIG_ENV, "").strip()
    if not configured:
        raise HelperError("remote_helper_rejected", f"{_HELPER_CONFIG_ENV} is not configured")
    config_path = Path(configured).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HelperError("remote_helper_rejected", "fixed helper config is unreadable") from exc
    allowed = {"schema", "repo", "state_dir", "verify_key"}
    if not isinstance(raw, dict) or raw.get("schema") != _HELPER_CONFIG_SCHEMA or any(key not in allowed for key in raw):
        raise HelperError("remote_helper_rejected", "fixed helper config schema is invalid")
    result = {key: raw.get(key, "") for key in ("repo", "state_dir", "verify_key")}
    if any(not isinstance(value, str) or not value.strip() for value in result.values()):
        raise HelperError("remote_helper_rejected", "fixed helper config has invalid paths")
    _validate_external_path(config_path, Path(result["repo"]).expanduser().resolve(), label="helper config")
    return result


def _decode_frame(value: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise HelperError("patch_signature_invalid", "helper frame encoding is invalid")
    try:
        padded = value + "=" * (-len(value) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HelperError("patch_signature_invalid", "helper frame encoding is invalid") from exc
    if not isinstance(decoded, dict) or len(raw) > _MAX_REQUEST_BYTES:
        raise HelperError("patch_signature_invalid", "helper frame is invalid")
    return decoded


def _print_result(result: Mapping[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")), flush=True)


def _main_command(argv: list[str]) -> int:
    """Fixed command entry point installed as ``qlh-patch-helper`` in S2.

    Repo, state and verification-key paths come exclusively from the deployment
    configuration. The SSH caller can only select one contract action and its
    bounded contract arguments.
    """
    parser = argparse.ArgumentParser(description="QLH fixed SSH patch helper")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("action", choices=("status", "fetch", "apply", "verify"))
    parser.add_argument("--commit-sha")
    parser.add_argument("--frame-b64url")
    parser.add_argument("--maintenance-operation")
    args = parser.parse_args(argv)
    if args.protocol != HELPER_SCHEMA:
        _print_result(build_helper_result(args.action, status="error", error_code="helper_action_invalid"))
        return 2
    try:
        if args.action == "status":
            if any(value is not None for value in (args.commit_sha, args.frame_b64url, args.maintenance_operation)):
                raise HelperError("helper_action_invalid", "status has no arguments")
            request = build_helper_request("status")
        elif args.action in {"fetch", "verify"}:
            if args.frame_b64url is not None or args.maintenance_operation is not None:
                raise HelperError("helper_action_invalid", "fetch/verify have only commit_sha")
            request = build_helper_request(args.action, {"commit_sha": args.commit_sha})
        else:
            if args.commit_sha is not None:
                raise HelperError("helper_action_invalid", "apply cannot receive commit_sha separately")
            request = build_helper_request("apply", {
                "frame": _decode_frame(args.frame_b64url or ""),
                "operation_id": args.maintenance_operation,
            })
        config = _load_helper_config()
        result = handle_request(request, **config)
    except HelperError as exc:
        code = exc.code if exc.code in FAILURE_MATRIX else "remote_helper_rejected"
        result = build_helper_result(args.action, status="error", error_code=code)
    except ValueError:
        result = build_helper_result(args.action, status="error", error_code="helper_action_invalid")
    _print_result(result)
    return 0 if result["status"] == "ok" else 1


def forced_command_main() -> int:
    """Run under sshd ForceCommand, accepting only S0's exact remote command."""
    original = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    try:
        tokens = shlex.split(original, posix=True)
    except ValueError:
        tokens = []
    if not tokens or tokens[0] != REMOTE_HELPER:
        _print_result(build_helper_result("status", status="error", error_code="helper_action_invalid"))
        return 2
    return _main_command(tokens[1:])


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--forced-command"]:
        return forced_command_main()
    return _main_command(values)


if __name__ == "__main__":
    raise SystemExit(main())
