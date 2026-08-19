"""One-time, client-only cluster join grants.

This module deliberately stops at the local authorization contract.  Auth App
verification happens in the control plane; this layer receives only the
boolean approval result and never handles a TOTP seed, code, or email secret.
The resulting short-lived Ed25519 grant can be rendered as both a text code
and a QR payload.  A SQLite ledger makes nonce consumption survive restart.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except Exception as exc:  # pragma: no cover - depends on runtime packaging
    InvalidSignature = Exception  # type: ignore[assignment,misc]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]
    serialization = None  # type: ignore[assignment]
    _CRYPTO_IMPORT_ERROR = exc
else:
    _CRYPTO_IMPORT_ERROR = None


SCHEMA_VERSION = "qlh.cluster.join-grant.v1"
GRANT_PREFIX = "qlhjoin1"
MAX_GRANT_LENGTH = 16 * 1024
DEFAULT_CAPABILITIES = ("presence", "task")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class JoinContractError(ValueError):
    """Fail-closed local join contract error with a stable reason code."""

    def __init__(self, message: str, *, code: str = "join_invalid") -> None:
        self.code = code
        super().__init__(message)


def _require_crypto() -> None:
    if Ed25519PrivateKey is None:
        raise JoinContractError(
            f"cryptography is unavailable: {_CRYPTO_IMPORT_ERROR}",
            code="crypto_unavailable",
        )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: Any, *, field: str, expected_length: int | None = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise JoinContractError(f"{field} must be base64url text", code="invalid_encoding")
    try:
        raw = base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
    except (UnicodeEncodeError, ValueError, TypeError) as exc:
        raise JoinContractError(f"{field} is not valid base64url", code="invalid_encoding") from exc
    if expected_length is not None and len(raw) != expected_length:
        raise JoinContractError(f"{field} has invalid length", code="invalid_encoding")
    return raw


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _safe_id(value: Any, *, field: str, pattern: re.Pattern[str] = _SAFE_ID) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise JoinContractError(f"{field} is invalid", code="invalid_field")
    return value


def _normalize_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise JoinContractError("master_endpoint is invalid", code="invalid_endpoint")
    parsed = urlsplit(f"//{value}")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise JoinContractError("master_endpoint must be host:port", code="invalid_endpoint")
    try:
        port = parsed.port
    except ValueError as exc:
        raise JoinContractError("master_endpoint port is invalid", code="invalid_endpoint") from exc
    if not parsed.hostname or port is None or not 1 <= port <= 65535:
        raise JoinContractError("master_endpoint must include a valid port", code="invalid_endpoint")
    host = parsed.hostname.lower()
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def _now_seconds(value: int | float | None) -> int:
    if value is None:
        return int(time.time())
    if isinstance(value, bool):
        raise JoinContractError("time value is invalid", code="invalid_time")
    return int(value)


def _utc_text(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class JoinKeyPair:
    """Raw Ed25519 key material encoded for local config/QR-safe transport."""

    private_key: str
    public_key: str


def generate_join_keypair() -> JoinKeyPair:
    """Generate the target node keypair; the private key never enters a grant."""
    _require_crypto()
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    return JoinKeyPair(
        private_key=_b64(
            private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
        ),
        public_key=_b64(
            public.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ),
    )


def _normalize_capabilities(value: Any) -> list[str]:
    if value is None:
        value = DEFAULT_CAPABILITIES
    if not isinstance(value, (list, tuple)) or not value or len(value) > 16:
        raise JoinContractError("capabilities are invalid", code="invalid_capabilities")
    result = []
    for item in value:
        item = _safe_id(item, field="capability", pattern=re.compile(r"^[a-z][a-z0-9._:-]{0,31}$"))
        if item == "master" or item.startswith("admin"):
            raise JoinContractError("client grant requests forbidden capability", code="role_escalation")
        result.append(item)
    if len(set(result)) != len(result):
        raise JoinContractError("capabilities must be unique", code="invalid_capabilities")
    return sorted(result)


def build_join_request(
    *,
    master_endpoint: str,
    cluster_id: str,
    target_node_id: str,
    target_public_key: str,
    requested_at: int | float | None = None,
    request_ttl_seconds: int = 600,
    capabilities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the target-side request that an administrator approves."""
    requested = _now_seconds(requested_at)
    if isinstance(request_ttl_seconds, bool) or not 60 <= int(request_ttl_seconds) <= 3600:
        raise JoinContractError("request TTL is outside the allowed window", code="invalid_ttl")
    _unb64(target_public_key, field="target_public_key", expected_length=32)
    request = {
        "schema_version": SCHEMA_VERSION,
        "master_endpoint": _normalize_endpoint(master_endpoint),
        "cluster_id": _safe_id(cluster_id, field="cluster_id"),
        "target_node_id": _safe_id(target_node_id, field="target_node_id"),
        "target_public_key": target_public_key,
        "capabilities": _normalize_capabilities(capabilities),
        "requested_at": requested,
        "request_expires_at": requested + int(request_ttl_seconds),
        "request_nonce": _b64(secrets.token_bytes(18)),
    }
    request["request_digest"] = hashlib.sha256(_canonical(request)).hexdigest()
    return request


def issue_join_grant(
    request: Mapping[str, Any],
    *,
    issuer_key_id: str,
    issuer_private_key: Ed25519PrivateKey,
    auth_verified: bool,
    now: int | float | None = None,
    ttl_seconds: int = 300,
) -> dict[str, Any]:
    """Issue a short-lived grant after the control plane verified Auth App/TOTP."""
    if auth_verified is not True:
        raise JoinContractError("Auth App approval is required", code="auth_required")
    if not _SAFE_KEY_ID.fullmatch(str(issuer_key_id)):
        raise JoinContractError("issuer_key_id is invalid", code="invalid_field")
    _require_crypto()
    if not isinstance(issuer_private_key, Ed25519PrivateKey):
        raise JoinContractError("issuer private key is invalid", code="invalid_key")
    required = {
        "schema_version", "master_endpoint", "cluster_id", "target_node_id",
        "target_public_key", "capabilities", "request_expires_at", "request_digest",
    }
    if not required.issubset(request):
        raise JoinContractError("join request is incomplete", code="invalid_request")
    issued_at = _now_seconds(now)
    if issued_at >= int(request["request_expires_at"]):
        raise JoinContractError("join request has expired", code="request_expired")
    if isinstance(ttl_seconds, bool) or not 60 <= int(ttl_seconds) <= 900:
        raise JoinContractError("grant TTL is outside the allowed window", code="invalid_ttl")
    request_copy = dict(request)
    digest = hashlib.sha256(
        _canonical({key: value for key, value in request_copy.items() if key != "request_digest"})
    ).hexdigest()
    if digest != request_copy.get("request_digest"):
        raise JoinContractError("join request digest mismatch", code="request_tampered")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "grant_type": "client_only",
        "master_endpoint": _normalize_endpoint(request["master_endpoint"]),
        "cluster_id": _safe_id(request["cluster_id"], field="cluster_id"),
        "target_node_id": _safe_id(request["target_node_id"], field="target_node_id"),
        "target_public_key": request["target_public_key"],
        "role": "client",
        "capabilities": _normalize_capabilities(request["capabilities"]),
        "issued_at": issued_at,
        "expires_at": issued_at + int(ttl_seconds),
        "nonce": _b64(secrets.token_bytes(18)),
        "request_digest": request_copy["request_digest"],
        "issuer_key_id": str(issuer_key_id),
        "auth_method": "totp",
    }
    signature = _b64(issuer_private_key.sign(_canonical(payload)))
    return {
        "schema_version": SCHEMA_VERSION,
        "payload": payload,
        "signature": signature,
    }


def encode_join_grant(grant: Mapping[str, Any]) -> str:
    """Return the same compact string for manual entry and QR encoding."""
    if not isinstance(grant, Mapping) or set(grant) != {"schema_version", "payload", "signature"}:
        raise JoinContractError("grant envelope is invalid", code="invalid_grant")
    if grant.get("schema_version") != SCHEMA_VERSION:
        raise JoinContractError("grant schema is unsupported", code="unsupported_schema")
    payload = grant.get("payload")
    signature = grant.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        raise JoinContractError("grant envelope is invalid", code="invalid_grant")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("grant_type") != "client_only":
        raise JoinContractError("grant type is unsupported", code="invalid_grant")
    if payload.get("role") != "client":
        raise JoinContractError("grant is not client-only", code="role_escalation")
    _normalize_endpoint(payload.get("master_endpoint"))
    _safe_id(payload.get("cluster_id"), field="cluster_id")
    _safe_id(payload.get("target_node_id"), field="target_node_id")
    _unb64(payload.get("target_public_key"), field="target_public_key", expected_length=32)
    _normalize_capabilities(payload.get("capabilities"))
    for field in ("issued_at", "expires_at"):
        if not isinstance(payload.get(field), int) or payload[field] < 0:
            raise JoinContractError(f"{field} is invalid", code="invalid_grant")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("request_digest", ""))):
        raise JoinContractError("request_digest is invalid", code="invalid_grant")
    _safe_id(payload.get("issuer_key_id"), field="issuer_key_id", pattern=_SAFE_KEY_ID)
    _unb64(payload.get("nonce"), field="nonce", expected_length=18)
    _unb64(signature, field="signature", expected_length=64)
    encoded = f"{GRANT_PREFIX}.{_b64(_canonical(dict(payload)))}.{signature}"
    if len(encoded) > MAX_GRANT_LENGTH:
        raise JoinContractError("grant is too large for manual/QR transport", code="grant_too_large")
    return encoded


def decode_join_grant(value: str) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > MAX_GRANT_LENGTH or not value.startswith(f"{GRANT_PREFIX}."):
        raise JoinContractError("grant prefix is invalid", code="invalid_grant")
    parts = value.split(".")
    if len(parts) != 3:
        raise JoinContractError("grant encoding is invalid", code="invalid_grant")
    try:
        payload = json.loads(_unb64(parts[1], field="payload").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoinContractError("grant payload is not valid JSON", code="invalid_grant") from exc
    if not isinstance(payload, dict):
        raise JoinContractError("grant payload is invalid", code="invalid_grant")
    _unb64(parts[2], field="signature", expected_length=64)
    return {"schema_version": SCHEMA_VERSION, "payload": payload, "signature": parts[2]}


class JoinGrantLedger:
    """Durable nonce ledger owned by the user's main node."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_join_grant_nonces (
                  nonce TEXT PRIMARY KEY,
                  grant_digest TEXT NOT NULL,
                  consumed_at INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def consume(self, *, nonce: str, grant_digest: str, consumed_at: int | float | None = None) -> None:
        _unb64(nonce, field="nonce", expected_length=18)
        if not re.fullmatch(r"[0-9a-f]{64}", grant_digest):
            raise JoinContractError("grant digest is invalid", code="invalid_grant")
        timestamp = _now_seconds(consumed_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT grant_digest FROM cluster_join_grant_nonces WHERE nonce = ?",
                (nonce,),
            ).fetchone()
            if existing is not None:
                raise JoinContractError("grant nonce was already consumed", code="nonce_replayed")
            connection.execute(
                "INSERT INTO cluster_join_grant_nonces(nonce, grant_digest, consumed_at) VALUES (?, ?, ?)",
                (nonce, grant_digest, timestamp),
            )


def verify_and_consume_join_grant(
    encoded_grant: str,
    *,
    issuer_public_key: str,
    expected_request: Mapping[str, Any],
    ledger: JoinGrantLedger,
    now: int | float | None = None,
) -> dict[str, Any]:
    """Verify signature, binding, TTL and role, then atomically consume nonce."""
    _require_crypto()
    grant = decode_join_grant(encoded_grant)
    # Validate the full envelope before any nonce state is consumed.
    encode_join_grant(grant)
    payload = grant["payload"]
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("grant_type") != "client_only":
        raise JoinContractError("grant type is unsupported", code="invalid_grant")
    try:
        public = Ed25519PublicKey.from_public_bytes(
            _unb64(issuer_public_key, field="issuer_public_key", expected_length=32)
        )
        public.verify(
            _unb64(grant["signature"], field="signature", expected_length=64),
            _canonical(payload),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise JoinContractError("grant signature is invalid", code="signature_invalid") from exc
    if payload.get("role") != "client":
        raise JoinContractError("grant is not client-only", code="role_escalation")
    current = _now_seconds(now)
    if current < int(payload.get("issued_at", 0)) or current >= int(payload.get("expires_at", 0)):
        raise JoinContractError("grant is expired or not yet valid", code="grant_expired")
    expected_digest = hashlib.sha256(
        _canonical({key: value for key, value in dict(expected_request).items() if key != "request_digest"})
    ).hexdigest()
    if payload.get("request_digest") != expected_digest:
        raise JoinContractError("grant does not match this join request", code="request_mismatch")
    for field in ("master_endpoint", "cluster_id", "target_node_id", "target_public_key"):
        expected = expected_request.get(field)
        if field == "master_endpoint":
            expected = _normalize_endpoint(expected)
        if payload.get(field) != expected:
            raise JoinContractError(f"grant {field} mismatch", code="request_mismatch")
    _unb64(payload.get("target_public_key"), field="target_public_key", expected_length=32)
    nonce = payload.get("nonce")
    digest = hashlib.sha256(encoded_grant.encode("ascii")).hexdigest()
    ledger.consume(nonce=nonce, grant_digest=digest, consumed_at=current)
    return dict(payload)
