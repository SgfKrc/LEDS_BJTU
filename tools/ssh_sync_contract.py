#!/usr/bin/env python3
"""安全的 SSH 补丁同步契约（SYNC-SSH-S0）。

本模块刻意不建立连接、不执行 git，也不应用补丁。它只冻结 SSH 传输层
的受信主机、Ed25519 身份、固定远程 helper 与维护锁协议，使后续的 SSH
客户端和服务端都不能退化为任意远程 shell。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROFILE_SCHEMA = "qlh.sync_ssh.profile.v1"
HELPER_SCHEMA = "qlh.sync_ssh.helper.v1"
HELPER_RESULT_SCHEMA = "qlh.sync_ssh.helper-result.v1"
HELPER_LEDGER_SCHEMA = "qlh.sync_ssh.helper-ledger.v1"
LOCK_SCHEMA = "qlh.sync_ssh.maintenance-lock.v1"
REMOTE_HELPER = "qlh-patch-helper"
MAX_HELPER_PAYLOAD_BYTES = 32 * 1024
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_DNS_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_FINGERPRINT_RE = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
_ACTIONS = frozenset({"status", "fetch", "apply", "verify"})
FAILURE_MATRIX = {
    "host_invalid": {"retryable": False, "next_action": "fix_profile"},
    "host_key_invalid": {"retryable": False, "next_action": "reenroll_host_key"},
    "host_fingerprint_mismatch": {"retryable": False, "next_action": "reenroll_host_key"},
    "identity_key_invalid": {"retryable": False, "next_action": "fix_identity"},
    "password_auth_forbidden": {"retryable": False, "next_action": "fix_profile"},
    "helper_action_invalid": {"retryable": False, "next_action": "reject_request"},
    "helper_payload_too_large": {"retryable": False, "next_action": "reject_request"},
    "maintenance_lock_held": {"retryable": True, "next_action": "retry_later"},
    "repository_unmanaged": {"retryable": False, "next_action": "manual_review"},
    "workspace_dirty": {"retryable": False, "next_action": "manual_review"},
    "patch_signature_invalid": {"retryable": False, "next_action": "reject_request"},
    "remote_helper_rejected": {"retryable": False, "next_action": "manual_review"},
    "transport_unavailable": {"retryable": True, "next_action": "fallback_pull_push"},
}
_FORBIDDEN_PROFILE_FIELDS = frozenset({
    "password", "passphrase", "proxy_command", "remote_command",
    "command", "auto_accept_host_key", "strict_host_key_checking",
})


class SshSyncContractError(ValueError):
    """A fail-closed S0 validation error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _error(code: str, message: str) -> None:
    raise SshSyncContractError(code, message)


def normalize_host(value: object) -> str:
    """Normalize a bare IPv4/IPv6 address or DNS/MagicDNS host.

    Port and URI/user syntax are deliberately rejected: they have separate
    profile fields and otherwise become a shell/authority ambiguity.
    """
    if not isinstance(value, str):
        _error("host_invalid", "SSH host must be a string")
    host = value.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host or any(token in host for token in ("/", "@", " ", "\t", "\\")):
        _error("host_invalid", "SSH host is not a bare address or DNS name")
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        if not _DNS_HOST_RE.fullmatch(host):
            _error("host_invalid", "SSH host contains unsupported characters")
        return host


def _normalize_port(value: object) -> int:
    if isinstance(value, bool):
        _error("port_invalid", "SSH port must be an integer")
    try:
        port = int(value)
    except (TypeError, ValueError):
        _error("port_invalid", "SSH port must be an integer")
    if not 1 <= port <= 65535:
        _error("port_invalid", "SSH port is outside 1..65535")
    return port


def _parse_ed25519_public_key(value: object, *, code: str) -> tuple[str, bytes]:
    if not isinstance(value, str):
        _error(code, "SSH public key must be an OpenSSH string")
    tokens = value.strip().split()
    if len(tokens) < 2 or tokens[0] != "ssh-ed25519":
        _error(code, "only ssh-ed25519 public keys are accepted")
    try:
        wire = base64.b64decode(tokens[1].encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise SshSyncContractError(code, "SSH public key base64 is invalid") from exc
    try:
        algorithm_length = int.from_bytes(wire[:4], "big")
        cursor = 4
        algorithm = wire[cursor:cursor + algorithm_length]
        cursor += algorithm_length
        key_length = int.from_bytes(wire[cursor:cursor + 4], "big")
        cursor += 4
        public = wire[cursor:cursor + key_length]
    except (IndexError, ValueError) as exc:
        raise SshSyncContractError(code, "SSH public key wire encoding is invalid") from exc
    if algorithm != b"ssh-ed25519" or len(public) != 32 or cursor + key_length != len(wire):
        _error(code, "SSH public key is not a complete Ed25519 key")
    return f"ssh-ed25519 {tokens[1]}", wire


def host_fingerprint(public_key: str) -> str:
    """Return OpenSSH's SHA256 fingerprint format for an Ed25519 public key."""
    _normalized, wire = _parse_ed25519_public_key(public_key, code="host_key_invalid")
    digest = base64.b64encode(hashlib.sha256(wire).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def _normalize_fingerprint(value: object) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value.strip()):
        _error("host_fingerprint_invalid", "host fingerprint must use SSH SHA256 format")
    return value.strip()


@dataclass(frozen=True)
class SshSyncProfile:
    host: str
    port: int
    user: str
    host_public_key: str
    host_fingerprint: str
    identity_file: str
    identity_public_key: str

    @property
    def authority(self) -> str:
        return f"[{self.host}]:{self.port}" if ":" in self.host else f"{self.host}:{self.port}"

    def known_hosts_entry(self) -> str:
        host_token = self.host if self.port == 22 else f"[{self.host}]:{self.port}"
        return f"{host_token} {self.host_public_key}\n"


def load_profile(raw: Mapping[str, Any]) -> SshSyncProfile:
    """Validate a portable profile without loading its private identity file."""
    if not isinstance(raw, Mapping) or raw.get("schema") != PROFILE_SCHEMA:
        _error("profile_schema_invalid", f"profile schema must be {PROFILE_SCHEMA}")
    forbidden = sorted(field for field in _FORBIDDEN_PROFILE_FIELDS if field in raw)
    if forbidden:
        _error("password_auth_forbidden", "forbidden SSH profile fields: " + ", ".join(forbidden))
    allowed = {
        "schema", "host", "port", "user", "host_public_key", "host_fingerprint",
        "identity_file", "identity_public_key",
    }
    extras = sorted(str(key) for key in raw if key not in allowed)
    if extras:
        _error("profile_field_unsupported", "unsupported SSH profile fields: " + ", ".join(extras))
    host = normalize_host(raw.get("host"))
    port = _normalize_port(raw.get("port", 22))
    user = raw.get("user")
    if not isinstance(user, str) or not _USER_RE.fullmatch(user):
        _error("user_invalid", "SSH user must be a conservative local account name")
    host_key, _wire = _parse_ed25519_public_key(raw.get("host_public_key"), code="host_key_invalid")
    fingerprint = _normalize_fingerprint(raw.get("host_fingerprint"))
    if host_fingerprint(host_key) != fingerprint:
        _error("host_fingerprint_mismatch", "configured host key does not match its fingerprint")
    identity_file = raw.get("identity_file")
    if not isinstance(identity_file, str) or not identity_file.strip():
        _error("identity_file_invalid", "an Ed25519 identity file path is required")
    identity_key, _identity_wire = _parse_ed25519_public_key(
        raw.get("identity_public_key"), code="identity_key_invalid",
    )
    return SshSyncProfile(
        host=host,
        port=port,
        user=user,
        host_public_key=host_key,
        host_fingerprint=fingerprint,
        identity_file=identity_file.strip(),
        identity_public_key=identity_key,
    )


def build_maintenance_lock(operation_id: object) -> dict[str, str]:
    """Describe the lock a remote helper must acquire before an apply request."""
    if not isinstance(operation_id, str) or not re.fullmatch(r"[0-9a-f]{64}", operation_id):
        _error("operation_id_invalid", "maintenance operation_id must be a SHA-256 hex digest")
    return {
        "schema": LOCK_SCHEMA,
        "operation_id": operation_id,
        "scope": "managed_checkout",
        "owner": "remote_helper",
    }


def failure_semantics(code: object) -> dict[str, Any]:
    if not isinstance(code, str) or code not in FAILURE_MATRIX:
        _error("failure_code_invalid", "SSH helper returned an unknown failure code")
    return {"code": code, **FAILURE_MATRIX[code]}


def validate_helper_result(raw: Mapping[str, Any], *, expected_action: str) -> dict[str, Any]:
    """Validate a path-free helper result and its stable retry/fallback semantics."""
    if expected_action not in _ACTIONS:
        _error("helper_action_invalid", "expected helper action is not permitted")
    if not isinstance(raw, Mapping) or raw.get("schema") != HELPER_RESULT_SCHEMA:
        _error("helper_result_invalid", "SSH helper result schema is invalid")
    allowed = {"schema", "action", "status", "commit_sha", "error_code", "retryable", "next_action"}
    if any(key not in allowed for key in raw):
        _error("helper_result_invalid", "SSH helper result contains unsupported fields")
    if raw.get("action") != expected_action or raw.get("status") not in {"ok", "error"}:
        _error("helper_result_invalid", "SSH helper result action or status is invalid")
    result = dict(raw)
    if "commit_sha" in result:
        result["commit_sha"] = _commit_sha(result["commit_sha"])
    if result["status"] == "ok":
        if any(field in result for field in ("error_code", "retryable", "next_action")):
            _error("helper_result_invalid", "successful SSH helper result carries failure fields")
        return result
    semantics = failure_semantics(result.get("error_code"))
    if result.get("retryable") is not semantics["retryable"] or result.get("next_action") != semantics["next_action"]:
        _error("helper_result_invalid", "SSH helper failure semantics do not match the contract")
    return result


def build_helper_result(
    action: object,
    *,
    status: str = "ok",
    commit_sha: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Build a path-free, contract-valid helper result."""
    if not isinstance(action, str) or action not in _ACTIONS:
        _error("helper_action_invalid", "helper result action is not permitted")
    if status not in {"ok", "error"}:
        _error("helper_result_invalid", "helper result status is invalid")
    result: dict[str, Any] = {
        "schema": HELPER_RESULT_SCHEMA,
        "action": action,
        "status": status,
    }
    if commit_sha is not None:
        result["commit_sha"] = _commit_sha(commit_sha)
    if status == "error":
        semantics = failure_semantics(error_code)
        result.update({
            "error_code": semantics["code"],
            "retryable": semantics["retryable"],
            "next_action": semantics["next_action"],
        })
    return validate_helper_result(result, expected_action=action)


def _commit_sha(value: object) -> str:
    if not isinstance(value, str) or not _COMMIT_SHA_RE.fullmatch(value):
        _error("commit_sha_invalid", "commit_sha must be a lowercase 40-character SHA-1")
    return value


def build_helper_request(action: object, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a fixed helper request; it never contains a shell command string."""
    if not isinstance(action, str) or action not in _ACTIONS:
        _error("helper_action_invalid", "only status, fetch, apply and verify are permitted")
    data = dict(payload or {})
    if action == "status":
        if data:
            _error("helper_payload_invalid", "status does not accept arguments")
        return {"schema": HELPER_SCHEMA, "action": action}
    if action in {"fetch", "verify"}:
        if set(data) != {"commit_sha"}:
            _error("helper_payload_invalid", f"{action} requires only commit_sha")
        return {"schema": HELPER_SCHEMA, "action": action, "commit_sha": _commit_sha(data["commit_sha"])}
    # The signed frame remains opaque to this transport layer.  It is bounded,
    # cannot request restart, and must be re-verified by the remote helper.
    if set(data) != {"frame", "operation_id"} or not isinstance(data["frame"], Mapping):
        _error("helper_payload_invalid", "apply requires a signed frame and operation_id")
    frame = dict(data["frame"])
    if frame.get("schema") != "qlh.patch_frame.v1":
        _error("patch_frame_invalid", "apply only accepts qlh.patch_frame.v1")
    if frame.get("restart_requested") is True:
        _error("restart_forbidden", "SSH apply cannot request an automatic restart")
    _commit_sha(frame.get("commit_sha"))
    encoded = json.dumps(frame, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_HELPER_PAYLOAD_BYTES:
        _error("helper_payload_too_large", "signed patch frame exceeds SSH helper limit")
    lock = build_maintenance_lock(data["operation_id"])
    return {
        "schema": HELPER_SCHEMA,
        "action": "apply",
        "frame": frame,
        "maintenance_lock": lock,
    }


def _helper_arguments(request: Mapping[str, Any]) -> list[str]:
    action = request["action"]
    arguments = [REMOTE_HELPER, "--protocol", HELPER_SCHEMA, action]
    if action in {"fetch", "verify"}:
        arguments.extend(["--commit-sha", request["commit_sha"]])
    elif action == "apply":
        encoded = base64.urlsafe_b64encode(
            json.dumps(request["frame"], ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).decode("ascii").rstrip("=")
        arguments.extend([
            "--frame-b64url", encoded,
            "--maintenance-operation", request["maintenance_lock"]["operation_id"],
        ])
    return arguments


def build_ssh_argv(profile: SshSyncProfile, request: Mapping[str, Any], *, known_hosts_file: str | Path) -> list[str]:
    """Render a non-shell argv for OpenSSH with strict host-key verification."""
    if not isinstance(request, Mapping) or request.get("schema") != HELPER_SCHEMA:
        _error("helper_request_invalid", "remote helper request has an unknown schema")
    # Rebuild from input to avoid a caller splicing arbitrary helper arguments.
    action = request.get("action")
    if action == "apply":
        request = build_helper_request("apply", {
            "frame": request.get("frame"),
            "operation_id": request.get("maintenance_lock", {}).get("operation_id"),
        })
    elif action in {"status", "fetch", "verify"}:
        payload = {} if action == "status" else {"commit_sha": request.get("commit_sha")}
        request = build_helper_request(action, payload)
    else:
        _error("helper_action_invalid", "remote helper action is not permitted")
    if not isinstance(known_hosts_file, (str, Path)) or not str(known_hosts_file).strip():
        _error("known_hosts_invalid", "a managed known_hosts path is required")
    known_hosts = str(Path(known_hosts_file))
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "PubkeyAuthentication=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", f"GlobalKnownHostsFile={os.devnull}",
        "-i", profile.identity_file,
        "-p", str(profile.port),
        f"{profile.user}@{profile.host}",
        *_helper_arguments(request),
    ]


def protocol_fixture() -> dict[str, Any]:
    """Stable non-secret protocol fixture used by docs and regression tests."""
    return {
        "profile_schema": PROFILE_SCHEMA,
        "helper_schema": HELPER_SCHEMA,
        "helper_result_schema": HELPER_RESULT_SCHEMA,
        "helper_ledger_schema": HELPER_LEDGER_SCHEMA,
        "maintenance_lock_schema": LOCK_SCHEMA,
        "remote_helper": REMOTE_HELPER,
        "actions": sorted(_ACTIONS),
        "failure_matrix": FAILURE_MATRIX,
        "prohibited": [
            "password_authentication", "accept_new_host_keys", "arbitrary_remote_shell",
            "sqlite_transfer", "model_weight_transfer", "attachment_transfer", "automatic_restart",
        ],
    }
