"""Bounded artifact transfer primitives for the Qwen3 pipeline.

The data plane streams controller-owned artifacts into a scoped ``.part``
file.  Tickets and receipts contain metadata only; tensor bytes, local paths,
and model inputs never enter control frames or SQLite state.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)
import uuid

from qwen3_pipeline_loopback import Qwen3LoopbackError, validate_loopback_base_url


QWEN3_TRANSFER_SCHEMA_VERSION = 1
QWEN3_TRANSFER_PREFIX = "/internal/v1/qwen3/artifact-transfer"
MAX_TRANSFER_CHUNK_BYTES = 4 * 1024 * 1024
MAX_TRANSFER_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
MAX_TRANSFER_TICKET_SECONDS = 300
MAX_CONTROL_RESPONSE_BYTES = 64 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRANSFER_ID = re.compile(r"^qtx_[0-9a-f]{32}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_TEMP_FILE = re.compile(r"^\.qtx_[0-9a-f]{32}\.part$")
_FINAL_FILE = re.compile(r"^qtx_[0-9a-f]{32}\.pt$")


class Qwen3TransferError(RuntimeError):
    """Stable fail-closed error for transfer clients and HTTP adapters."""

    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)[:128]
        self.reason = str(reason)[:1024]
        super().__init__(self.reason)


def _canonical_bytes(value: Any, maximum: int = 8192) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3TransferError(
            "qwen3_transfer_invalid_json", "transfer metadata is not strict JSON",
        ) from exc
    if len(encoded) > maximum:
        raise Qwen3TransferError(
            "qwen3_transfer_metadata_oversize", "transfer metadata exceeds its size limit",
        )
    return encoded


def _safe_node_id(value: Any) -> str:
    result = str(value or "")
    if _NODE_ID.fullmatch(result) is None:
        raise Qwen3TransferError(
            "qwen3_transfer_identity_invalid", "transfer peer identity is invalid",
        )
    return result


def _safe_dimensions(
    *, chain_id: Any, generation: Any, phase: Any,
    from_segment: Any, to_segment: Any, size_bytes: Any, sha256: Any,
    peer_epoch: Any = 0,
) -> dict[str, Any]:
    chain = str(chain_id or "").lower()
    digest = str(sha256 or "").lower()
    if _SHA256.fullmatch(chain) is None:
        raise Qwen3TransferError(
            "qwen3_transfer_contract_invalid", "transfer chain digest is invalid",
        )
    if _SHA256.fullmatch(digest) is None:
        raise Qwen3TransferError(
            "qwen3_transfer_contract_invalid", "transfer artifact digest is invalid",
        )
    if not isinstance(phase, str) or phase not in {"prefill", "decode"}:
        raise Qwen3TransferError(
            "qwen3_transfer_contract_invalid", "transfer phase is invalid",
        )
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (generation, from_segment, to_segment, size_bytes, peer_epoch)
    ):
        raise Qwen3TransferError(
            "qwen3_transfer_contract_invalid", "transfer dimensions are invalid",
        )
    generation_value = generation
    source = from_segment
    destination = to_segment
    size = size_bytes
    epoch = peer_epoch
    if (
        generation_value < 0
        or source not in {0, 1}
        or destination != source + 1
        or destination not in {1, 2}
        or not 0 < size <= MAX_TRANSFER_ARTIFACT_BYTES
        or epoch < 0
    ):
        raise Qwen3TransferError(
            "qwen3_transfer_contract_invalid", "transfer dimensions are outside limits",
        )
    return {
        "chain_id": chain,
        "generation": generation_value,
        "phase": str(phase),
        "from_segment": source,
        "to_segment": destination,
        "size_bytes": size,
        "sha256": digest,
        "peer_epoch": epoch,
    }


def _file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


class Qwen3TransferTicketSigner:
    """Issue one-session HMAC tickets bound to the complete handoff contract."""

    def __init__(
        self, secret: str | bytes, *, clock: Callable[[], float] = time.time,
    ) -> None:
        key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(key) < 32:
            raise ValueError("Qwen3 transfer secret must contain at least 32 bytes")
        self._key = key
        self._clock = clock

    @staticmethod
    def _encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    @staticmethod
    def _decode(value: str) -> bytes:
        try:
            return base64.b64decode(
                value + "=" * (-len(value) % 4),
                altchars=b"-_",
                validate=True,
            )
        except (TypeError, ValueError) as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket is malformed",
            ) from exc

    def issue(
        self,
        *,
        transfer_id: str,
        peer_node_id: str,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        size_bytes: int,
        sha256: str,
        ttl_seconds: float,
        nonce: str | None = None,
        peer_epoch: int = 0,
    ) -> str:
        if _TRANSFER_ID.fullmatch(str(transfer_id or "")) is None:
            raise Qwen3TransferError(
                "qwen3_transfer_identity_invalid", "transfer identifier is invalid",
            )
        peer = _safe_node_id(peer_node_id)
        dimensions = _safe_dimensions(
            chain_id=chain_id, generation=generation, phase=phase,
            from_segment=from_segment, to_segment=to_segment,
            size_bytes=size_bytes, sha256=sha256, peer_epoch=peer_epoch,
        )
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket TTL is invalid",
            ) from exc
        if not 1 <= ttl <= MAX_TRANSFER_TICKET_SECONDS:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket TTL is outside limits",
            )
        nonce_value = str(nonce or uuid.uuid4().hex)
        if _NONCE.fullmatch(nonce_value) is None:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket nonce is invalid",
            )
        payload = {
            "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
            "transfer_id": str(transfer_id),
            "peer_node_id": peer,
            **dimensions,
            "expires_at": int(self._clock() + ttl),
            "nonce": nonce_value,
        }
        encoded = self._encode(_canonical_bytes(payload))
        signature = self._encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest(),
        )
        return f"{encoded}.{signature}"

    def verify(
        self, ticket: str, *, authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
    ) -> dict[str, Any]:
        if not isinstance(ticket, str) or not ticket or len(ticket) > 4096:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket is malformed",
            )
        try:
            encoded, signature = ticket.split(".")
        except ValueError as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket is malformed",
            ) from exc
        if _B64URL.fullmatch(encoded) is None or _B64URL.fullmatch(signature) is None:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket is malformed",
            )
        expected = self._encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest(),
        )
        if not hmac.compare_digest(expected, signature):
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_signature", "transfer ticket signature is invalid",
            )
        try:
            payload = json.loads(self._decode(encoded))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket payload is invalid",
            ) from exc
        required = {
            "schema_version", "transfer_id", "peer_node_id", "peer_epoch", "chain_id",
            "generation", "phase", "from_segment", "to_segment",
            "size_bytes", "sha256", "expires_at", "nonce",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket fields are invalid",
            )
        if payload.get("schema_version") != QWEN3_TRANSFER_SCHEMA_VERSION:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket schema is unsupported",
            )
        if _TRANSFER_ID.fullmatch(str(payload.get("transfer_id", ""))) is None:
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket identity is invalid",
            )
        peer = _safe_node_id(payload.get("peer_node_id"))
        authenticated = _safe_node_id(authenticated_peer_id)
        if not hmac.compare_digest(peer, authenticated):
            raise Qwen3TransferError(
                "qwen3_transfer_peer_mismatch", "transfer ticket belongs to another peer",
            )
        dimensions = _safe_dimensions(
            chain_id=payload.get("chain_id"), generation=payload.get("generation"),
            phase=payload.get("phase"), from_segment=payload.get("from_segment"),
            to_segment=payload.get("to_segment"), size_bytes=payload.get("size_bytes"),
            sha256=payload.get("sha256"), peer_epoch=payload.get("peer_epoch"),
        )
        expires_at = payload.get("expires_at")
        nonce = str(payload.get("nonce", ""))
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, int)
            or _NONCE.fullmatch(nonce) is None
        ):
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_invalid", "transfer ticket lifetime is invalid",
            )
        if expires_at <= int(self._clock()):
            raise Qwen3TransferError(
                "qwen3_transfer_ticket_expired", "transfer ticket has expired",
            )
        if int(payload["peer_epoch"]) != int(authenticated_peer_epoch):
            raise Qwen3TransferError(
                "qwen3_transfer_peer_epoch_mismatch",
                "transfer ticket registration epoch is stale",
            )
        result = {
            "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
            "transfer_id": str(payload["transfer_id"]),
            "peer_node_id": peer,
            **{key: value for key, value in dimensions.items() if key != "peer_epoch"},
            "expires_at": expires_at,
            "nonce": nonce,
        }
        if int(payload["peer_epoch"]):
            result["peer_epoch"] = int(payload["peer_epoch"])
        return result


@dataclass
class _ReceiveSession:
    transfer_id: str
    peer_node_id: str
    peer_epoch: int
    chain_id: str
    generation: int
    phase: str
    from_segment: int
    to_segment: int
    size_bytes: int
    sha256: str
    expires_at: int
    nonce: str
    temp_path: Path
    final_path: Path
    received_bytes: int = 0
    status: str = "receiving"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
            "transfer_id": self.transfer_id,
            "peer_node_id": self.peer_node_id,
            "peer_epoch": self.peer_epoch,
            "chain_id": self.chain_id,
            "generation": self.generation,
            "phase": self.phase,
            "from_segment": self.from_segment,
            "to_segment": self.to_segment,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "expires_at": self.expires_at,
            "received_bytes": self.received_bytes,
            "status": self.status,
            "full_model_materialized": False,
        }


class Qwen3ArtifactReceiver:
    """Receive authenticated Qwen3 artifacts without buffering the whole file."""

    def __init__(
        self,
        root: str | Path,
        *,
        signer: Qwen3TransferTicketSigner,
        clock: Callable[[], float] = time.time,
    ) -> None:
        path = Path(root).expanduser().absolute().resolve(strict=False)
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError("Qwen3 transfer root is unavailable")
        self.root = path
        self.signer = signer
        self._clock = clock
        self._lock = threading.RLock()
        self._sessions: dict[str, _ReceiveSession] = {}

    def begin(
        self,
        *,
        peer_node_id: str,
        chain_id: str,
        generation: int,
        phase: str,
        from_segment: int,
        to_segment: int,
        size_bytes: int,
        sha256: str,
        ttl_seconds: float = 60,
        peer_epoch: int = 0,
    ) -> dict[str, Any]:
        peer = _safe_node_id(peer_node_id)
        dimensions = _safe_dimensions(
            chain_id=chain_id, generation=generation, phase=phase,
            from_segment=from_segment, to_segment=to_segment,
            size_bytes=size_bytes, sha256=sha256, peer_epoch=peer_epoch,
        )
        transfer_id = f"qtx_{uuid.uuid4().hex}"
        nonce = uuid.uuid4().hex
        ticket = self.signer.issue(
            transfer_id=transfer_id, peer_node_id=peer, **dimensions,
            ttl_seconds=ttl_seconds, nonce=nonce,
        )
        ticket_payload = self.signer.verify(
            ticket,
            authenticated_peer_id=peer,
            authenticated_peer_epoch=int(dimensions["peer_epoch"]),
        )
        temp_path = self.root / f".{transfer_id}.part"
        final_path = self.root / f"{transfer_id}.pt"
        try:
            with temp_path.open("xb"):
                pass
        except OSError as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_staging_failed", "transfer staging file could not be created",
            ) from exc
        session = _ReceiveSession(
            transfer_id=transfer_id,
            peer_node_id=peer,
            **dimensions,
            expires_at=int(ticket_payload["expires_at"]),
            nonce=nonce,
            temp_path=temp_path,
            final_path=final_path,
        )
        with self._lock:
            self._sessions[transfer_id] = session
        return {"ticket": ticket, "descriptor": session.snapshot()}

    def _cleanup_files_locked(self, session: _ReceiveSession) -> tuple[int, int]:
        removed = 0
        failures = 0
        for path in (session.temp_path, session.final_path):
            try:
                existed = path.exists()
                path.unlink(missing_ok=True)
                removed += int(existed)
            except OSError:
                failures += 1
        return removed, failures

    def cleanup_expired(self) -> dict[str, Any]:
        now = int(self._clock())
        expired = 0
        removed = 0
        failures = 0
        with self._lock:
            for session in self._sessions.values():
                if session.status == "receiving" and session.expires_at <= now:
                    session_removed, session_failures = self._cleanup_files_locked(session)
                    removed += session_removed
                    failures += session_failures
                    session.status = "expired"
                    expired += 1
        return {
            "expired_sessions": expired,
            "removed_artifacts": removed,
            "cleanup_failures": failures,
            "cleanup_complete": failures == 0,
        }

    def reconcile_orphans(self) -> dict[str, Any]:
        """Remove files left by a prior process before accepting sessions."""
        removed = 0
        failures = 0
        with self._lock:
            if self._sessions:
                raise Qwen3TransferError(
                    "qwen3_transfer_reconcile_active",
                    "transfer orphan reconciliation requires an empty runtime",
                )
            for path in self.root.iterdir():
                if not path.is_file() or not (
                    _TEMP_FILE.fullmatch(path.name) or _FINAL_FILE.fullmatch(path.name)
                ):
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    failures += 1
        return {
            "removed_artifacts": removed,
            "cleanup_failures": failures,
            "cleanup_complete": failures == 0,
        }

    def _authorize_locked(
        self, transfer_id: str, ticket: str, authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
    ) -> tuple[_ReceiveSession, dict[str, Any]]:
        payload = self.signer.verify(
            ticket,
            authenticated_peer_id=authenticated_peer_id,
            authenticated_peer_epoch=authenticated_peer_epoch,
        )
        if not hmac.compare_digest(str(payload["transfer_id"]), str(transfer_id)):
            raise Qwen3TransferError(
                "qwen3_transfer_scope_mismatch", "transfer ticket targets another session",
            )
        session = self._sessions.get(str(transfer_id))
        if session is None:
            raise Qwen3TransferError(
                "qwen3_transfer_missing", "transfer session is unavailable",
            )
        expected = {
            "peer_node_id": session.peer_node_id,
            "peer_epoch": session.peer_epoch,
            "chain_id": session.chain_id,
            "generation": session.generation,
            "phase": session.phase,
            "from_segment": session.from_segment,
            "to_segment": session.to_segment,
            "size_bytes": session.size_bytes,
            "sha256": session.sha256,
            "expires_at": session.expires_at,
            "nonce": session.nonce,
        }
        if any(payload.get(key, 0 if key == "peer_epoch" else None) != value for key, value in expected.items()):
            raise Qwen3TransferError(
                "qwen3_transfer_scope_mismatch", "transfer ticket contract does not match session",
            )
        return session, payload

    def status(
        self, transfer_id: str, *, ticket: str, authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            session, _ = self._authorize_locked(
                transfer_id, ticket, authenticated_peer_id,
                authenticated_peer_epoch,
            )
            return session.snapshot()

    def _fail_locked(self, session: _ReceiveSession) -> None:
        self._cleanup_files_locked(session)
        session.status = "failed"

    def write(
        self,
        transfer_id: str,
        *,
        ticket: str,
        authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
        offset: int,
        data: bytes,
    ) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise Qwen3TransferError(
                "qwen3_transfer_offset_invalid", "transfer offset is invalid",
            )
        chunk = bytes(data)
        if not chunk:
            raise Qwen3TransferError(
                "qwen3_transfer_chunk_empty", "transfer chunk is empty",
            )
        if len(chunk) > MAX_TRANSFER_CHUNK_BYTES:
            raise Qwen3TransferError(
                "qwen3_transfer_chunk_oversize", "transfer chunk exceeds its byte limit",
            )
        with self._lock:
            session, _ = self._authorize_locked(
                transfer_id, ticket, authenticated_peer_id,
                authenticated_peer_epoch,
            )
            if session.status != "receiving":
                raise Qwen3TransferError(
                    "qwen3_transfer_not_active", "transfer session is not receiving",
                )
            if offset > session.received_bytes:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_offset_mismatch", "transfer chunk arrived out of order",
                )
            if offset < session.received_bytes:
                if offset + len(chunk) > session.received_bytes:
                    self._fail_locked(session)
                    raise Qwen3TransferError(
                        "qwen3_transfer_replay_mismatch", "transfer replay overlaps new bytes",
                    )
                try:
                    with session.temp_path.open("rb") as handle:
                        handle.seek(offset)
                        previous = handle.read(len(chunk))
                except OSError as exc:
                    self._fail_locked(session)
                    raise Qwen3TransferError(
                        "qwen3_transfer_staging_failed", "transfer staging file is unavailable",
                    ) from exc
                if not hmac.compare_digest(previous, chunk):
                    self._fail_locked(session)
                    raise Qwen3TransferError(
                        "qwen3_transfer_replay_mismatch", "transfer replay bytes changed",
                    )
                return session.snapshot()
            if session.received_bytes + len(chunk) > session.size_bytes:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_size_mismatch", "transfer exceeds declared artifact size",
                )
            try:
                with session.temp_path.open("ab") as handle:
                    handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_staging_failed", "transfer chunk could not be persisted",
                ) from exc
            session.received_bytes += len(chunk)
            return session.snapshot()

    def _receipt_locked(self, session: _ReceiveSession) -> dict[str, Any]:
        receipt = session.snapshot()
        receipt["status"] = "committed"
        receipt["received_bytes"] = session.size_bytes
        return receipt

    def commit(
        self, transfer_id: str, *, ticket: str, authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            session, _ = self._authorize_locked(
                transfer_id, ticket, authenticated_peer_id, authenticated_peer_epoch,
            )
            if session.status == "committed":
                return self._receipt_locked(session)
            if session.status != "receiving":
                raise Qwen3TransferError(
                    "qwen3_transfer_not_active", "transfer session cannot be committed",
                )
            if session.received_bytes != session.size_bytes:
                raise Qwen3TransferError(
                    "qwen3_transfer_incomplete", "transfer has not received all bytes",
                )
            try:
                actual_size, actual_sha256 = _file_evidence(session.temp_path)
            except OSError as exc:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_staging_failed", "transfer staging file is unavailable",
                ) from exc
            if actual_size != session.size_bytes or actual_sha256 != session.sha256:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_digest_mismatch", "transfer artifact digest does not match",
                )
            try:
                os.replace(session.temp_path, session.final_path)
            except OSError as exc:
                self._fail_locked(session)
                raise Qwen3TransferError(
                    "qwen3_transfer_commit_failed", "transfer artifact could not be committed",
                ) from exc
            session.status = "committed"
            return self._receipt_locked(session)

    def cancel(
        self, transfer_id: str, *, ticket: str, authenticated_peer_id: str,
        authenticated_peer_epoch: int = 0,
    ) -> dict[str, Any]:
        with self._lock:
            session, _ = self._authorize_locked(
                transfer_id, ticket, authenticated_peer_id, authenticated_peer_epoch,
            )
            removed, failures = self._cleanup_files_locked(session)
            session.status = "cancelled"
            result = session.snapshot()
            result.update({
                "cleanup_complete": failures == 0,
                "cleanup_failures": failures,
                "removed_artifacts": removed,
            })
            return result

    def discard(self, transfer_id: str) -> dict[str, Any]:
        """Internally revoke one session without relying on an expiring ticket."""
        with self._lock:
            session = self._sessions.get(str(transfer_id))
            if session is None:
                return {
                    "transfer_id": str(transfer_id),
                    "status": "missing",
                    "cleanup_complete": True,
                    "cleanup_failures": 0,
                    "removed_artifacts": 0,
                }
            removed, failures = self._cleanup_files_locked(session)
            session.status = "cancelled"
            return {
                "transfer_id": session.transfer_id,
                "status": session.status,
                "cleanup_complete": failures == 0,
                "cleanup_failures": failures,
                "removed_artifacts": removed,
            }

    def artifact_path(self, transfer_id: str) -> Path:
        """Return a committed path only to the future local sidecar integrator."""
        with self._lock:
            session = self._sessions.get(str(transfer_id))
            if session is None or session.status != "committed" or not session.final_path.is_file():
                raise Qwen3TransferError(
                    "qwen3_transfer_missing", "committed transfer artifact is unavailable",
                )
            return session.final_path

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for session in self._sessions.values():
                counts[session.status] = counts.get(session.status, 0) + 1
            return {
                "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
                "enabled": True,
                "max_chunk_bytes": MAX_TRANSFER_CHUNK_BYTES,
                "max_artifact_bytes": MAX_TRANSFER_ARTIFACT_BYTES,
                "sessions": counts,
            }

    def session_status(self, transfer_id: str) -> str | None:
        """Return bounded lifecycle state for coordinator reconciliation."""
        with self._lock:
            session = self._sessions.get(str(transfer_id))
            return None if session is None else str(session.status)


@dataclass(frozen=True)
class TransferResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes


TransferRequester = Callable[[str, str, Mapping[str, str], bytes | None], TransferResponse]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def default_transfer_request(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None,
) -> TransferResponse:
    request = Request(url, data=body, headers=dict(headers), method=method)
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=30.0) as response:
            response_limit = (
                MAX_TRANSFER_CHUNK_BYTES + 1
                if str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip()
                == "application/octet-stream"
                else MAX_CONTROL_RESPONSE_BYTES + 1
            )
            content = response.read(response_limit)
            if len(content) > response_limit - 1:
                raise Qwen3TransferError(
                    "qwen3_transfer_response_oversize", "transfer response exceeds its limit",
                )
            return TransferResponse(
                status_code=int(response.status),
                headers=dict(response.headers.items()),
                content=content,
            )
    except HTTPError as exc:
        error_content_type = "" if exc.headers is None else str(exc.headers.get("Content-Type", ""))
        response_limit = (
            MAX_TRANSFER_CHUNK_BYTES + 1
            if error_content_type.split(";", 1)[0].strip()
            == "application/octet-stream"
            else MAX_CONTROL_RESPONSE_BYTES + 1
        )
        content = exc.read(response_limit)
        if len(content) > response_limit - 1:
            raise Qwen3TransferError(
                "qwen3_transfer_response_oversize", "transfer response exceeds its limit",
            ) from exc
        return TransferResponse(
            status_code=int(exc.code),
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            content=content,
        )
    except Qwen3TransferError:
        raise
    except (URLError, OSError) as exc:
        raise Qwen3TransferError(
            "qwen3_transfer_connection_failed", "Qwen3 transfer endpoint is unreachable",
        ) from exc


def _header(headers: Mapping[str, str], name: str) -> str:
    needle = name.lower()
    for key, value in headers.items():
        if str(key).lower() == needle:
            return str(value)
    return ""


class Qwen3ArtifactTransferClient:
    """Upload one scoped artifact with sequential bounded request/ack backpressure."""

    def __init__(
        self,
        source_root: str | Path,
        requester: TransferRequester | None = None,
        *,
        chunk_bytes: int = MAX_TRANSFER_CHUNK_BYTES,
        peer_proof_headers: Callable[[str, str, str], Mapping[str, str]] | None = None,
    ) -> None:
        root = Path(source_root).expanduser().absolute().resolve(strict=False)
        if not root.is_dir():
            raise ValueError("Qwen3 transfer source root is unavailable")
        if not 1 <= int(chunk_bytes) <= MAX_TRANSFER_CHUNK_BYTES:
            raise ValueError("Qwen3 transfer chunk size is outside limits")
        self.source_root = root
        self._requester = requester or default_transfer_request
        self.chunk_bytes = int(chunk_bytes)
        self._peer_proof_headers = peer_proof_headers

    def _headers(
        self,
        method: str,
        url: str,
        ticket: str,
        base: Mapping[str, str],
    ) -> dict[str, str]:
        result = {str(key): str(value) for key, value in base.items()}
        if self._peer_proof_headers is None:
            return result
        try:
            extra = dict(self._peer_proof_headers(method, url, ticket))
        except Qwen3TransferError:
            raise
        except Exception as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_peer_proof_failed", "Qwen3 peer proof could not be created",
            ) from exc
        protected = {key.lower() for key in result}
        if any(str(key).lower() in protected for key in extra):
            raise Qwen3TransferError(
                "qwen3_transfer_peer_proof_failed", "peer proof attempted to replace request headers",
            )
        result.update({str(key): str(value) for key, value in extra.items()})
        return result

    @staticmethod
    def _plan(value: Mapping[str, Any]) -> dict[str, Any]:
        plan = dict(value)
        if set(plan) != {
            "schema_version", "base_url", "transfer_id", "ticket", "descriptor",
        } or plan.get("schema_version") != QWEN3_TRANSFER_SCHEMA_VERSION:
            raise Qwen3TransferError(
                "qwen3_transfer_plan_invalid", "transfer plan fields are invalid",
            )
        try:
            base_url = validate_loopback_base_url(str(plan.get("base_url", "")))
        except Qwen3LoopbackError as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_plan_invalid", "transfer plan URL is invalid",
            ) from exc
        transfer_id = str(plan.get("transfer_id", ""))
        ticket = str(plan.get("ticket", ""))
        descriptor = plan.get("descriptor")
        if (
            _TRANSFER_ID.fullmatch(transfer_id) is None
            or not ticket
            or not isinstance(descriptor, Mapping)
        ):
            raise Qwen3TransferError(
                "qwen3_transfer_plan_invalid", "transfer plan identity is invalid",
            )
        descriptor = dict(descriptor)
        expected_fields = {
            "schema_version", "transfer_id", "peer_node_id", "peer_epoch", "chain_id",
            "generation", "phase", "from_segment", "to_segment", "size_bytes",
            "sha256", "expires_at", "received_bytes", "status",
            "full_model_materialized",
        }
        if set(descriptor) != expected_fields or descriptor.get("transfer_id") != transfer_id:
            raise Qwen3TransferError(
                "qwen3_transfer_plan_invalid", "transfer descriptor fields are invalid",
            )
        _safe_node_id(descriptor.get("peer_node_id"))
        _safe_dimensions(
            chain_id=descriptor.get("chain_id"), generation=descriptor.get("generation"),
            phase=descriptor.get("phase"), from_segment=descriptor.get("from_segment"),
            to_segment=descriptor.get("to_segment"), size_bytes=descriptor.get("size_bytes"),
            sha256=descriptor.get("sha256"), peer_epoch=descriptor.get("peer_epoch"),
        )
        if (
            descriptor.get("schema_version") != QWEN3_TRANSFER_SCHEMA_VERSION
            or isinstance(descriptor.get("expires_at"), bool)
            or not isinstance(descriptor.get("expires_at"), int)
            or descriptor.get("expires_at") <= 0
            or descriptor.get("received_bytes") != 0
            or descriptor.get("status") != "receiving"
            or descriptor.get("full_model_materialized") is not False
        ):
            raise Qwen3TransferError(
                "qwen3_transfer_plan_invalid", "transfer descriptor state is invalid",
            )
        return {
            "schema_version": QWEN3_TRANSFER_SCHEMA_VERSION,
            "base_url": base_url,
            "transfer_id": transfer_id,
            "ticket": ticket,
            "descriptor": descriptor,
        }

    def _source(self, value: str | Path) -> Path:
        path = Path(value).expanduser().absolute().resolve(strict=False)
        try:
            path.relative_to(self.source_root)
        except ValueError as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_source_scope", "transfer source escapes its artifact root",
            ) from exc
        if not path.is_file():
            raise Qwen3TransferError(
                "qwen3_transfer_source_missing", "transfer source artifact is unavailable",
            )
        return path

    @staticmethod
    def _json(response: TransferResponse) -> dict[str, Any]:
        if len(response.content) > MAX_CONTROL_RESPONSE_BYTES:
            raise Qwen3TransferError(
                "qwen3_transfer_response_oversize", "transfer response exceeds its limit",
            )
        try:
            value = json.loads(response.content)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_response_invalid", "transfer response is not valid JSON",
            ) from exc
        if not isinstance(value, dict):
            raise Qwen3TransferError(
                "qwen3_transfer_response_invalid", "transfer response is not an object",
            )
        return value

    def _request(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None,
    ) -> TransferResponse:
        try:
            return self._requester(method, url, headers, body)
        except Qwen3TransferError:
            raise
        except Exception as exc:
            raise Qwen3TransferError(
                "qwen3_transfer_connection_failed", "Qwen3 transfer endpoint is unreachable",
            ) from exc

    def upload_chunks(
        self,
        *,
        plan: Mapping[str, Any],
        total_bytes: int,
        sha256: str,
        chunk_provider: Callable[[int, int], bytes | Mapping[str, Any]],
        progress_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        """Upload a path-free artifact through an offset-aware chunk provider.

        The provider is called only after the receiver status is read, so a
        retry resumes from the receiver's acknowledged offset.  It may return
        raw bytes or a mapping containing ``data``, ``offset``, ``total_bytes``
        and ``sha256``; the latter binds every read to the registered output
        reference without exposing a filesystem path.
        """
        safe_plan = self._plan(plan)
        descriptor = safe_plan["descriptor"]
        if (
            isinstance(total_bytes, bool)
            or not isinstance(total_bytes, int)
            or total_bytes != descriptor["size_bytes"]
            or not isinstance(sha256, str)
            or sha256.lower() != descriptor["sha256"]
        ):
            raise Qwen3TransferError(
                "qwen3_transfer_source_mismatch", "chunk source does not match transfer plan",
            )
        if not callable(chunk_provider):
            raise Qwen3TransferError(
                "qwen3_transfer_source_mismatch", "chunk source provider is unavailable",
            )
        url = (
            f"{safe_plan['base_url']}{QWEN3_TRANSFER_PREFIX}/"
            f"{safe_plan['transfer_id']}"
        )
        headers = {
            "Authorization": f"Bearer {safe_plan['ticket']}",
            "Accept": "application/json",
        }
        status_response = self._request(
            "GET", url, self._headers("GET", url, safe_plan["ticket"], headers), None,
        )
        if status_response.status_code != 200:
            raise Qwen3TransferError(
                "qwen3_transfer_status_rejected", "transfer status request was rejected",
            )
        status = self._json(status_response)
        offset = status.get("received_bytes")
        immutable_status = {
            key: descriptor[key]
            for key in (
                "schema_version", "transfer_id", "peer_node_id", "peer_epoch", "chain_id",
                "generation", "phase", "from_segment", "to_segment",
                "size_bytes", "sha256", "expires_at", "full_model_materialized",
            )
        }
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= total_bytes
            or _header(status_response.headers, "Upload-Offset") != str(offset)
            or any(status.get(key) != value for key, value in immutable_status.items())
            or status.get("status") not in {"receiving", "committed"}
        ):
            raise Qwen3TransferError(
                "qwen3_transfer_response_invalid", "transfer status offset is invalid",
            )
        if progress_callback is not None:
            progress_callback(offset)
        while offset < total_bytes:
            requested = min(self.chunk_bytes, total_bytes - offset)
            try:
                provided = chunk_provider(offset, requested)
            except Qwen3TransferError:
                raise
            except Exception as exc:
                raise Qwen3TransferError(
                    "qwen3_transfer_connection_failed", "chunk source is temporarily unavailable",
                ) from exc
            metadata = provided if isinstance(provided, Mapping) else {}
            chunk = metadata.get("data", provided) if isinstance(provided, Mapping) else provided
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise Qwen3TransferError(
                    "qwen3_transfer_source_mismatch", "chunk source returned invalid data",
                )
            chunk = bytes(chunk)
            if not chunk or len(chunk) > requested:
                raise Qwen3TransferError(
                    "qwen3_transfer_source_mismatch", "chunk source returned an invalid size",
                )
            if isinstance(provided, Mapping):
                if (
                    provided.get("offset") != offset
                    or provided.get("total_bytes") != total_bytes
                    or str(provided.get("sha256", "")).lower() != sha256.lower()
                ):
                    raise Qwen3TransferError(
                        "qwen3_transfer_source_mismatch", "chunk source metadata changed",
                    )
            response = self._request(
                "PATCH",
                url,
                self._headers("PATCH", url, safe_plan["ticket"], {
                    **headers,
                    "Content-Type": "application/octet-stream",
                    "Upload-Offset": str(offset),
                }),
                chunk,
            )
            if response.status_code != 200:
                raise Qwen3TransferError(
                    "qwen3_transfer_chunk_rejected", "transfer chunk was rejected",
                )
            result = self._json(response)
            expected_offset = offset + len(chunk)
            if (
                result.get("received_bytes") != expected_offset
                or _header(response.headers, "Upload-Offset") != str(expected_offset)
                or any(
                    result.get(key) != value
                    for key, value in immutable_status.items()
                )
                or result.get("status") != "receiving"
            ):
                raise Qwen3TransferError(
                    "qwen3_transfer_response_invalid", "transfer acknowledgement offset changed",
                )
            offset = expected_offset
            if progress_callback is not None:
                progress_callback(offset)
        commit_url = f"{url}/commit"
        committed = self._request(
            "POST",
            commit_url,
            self._headers("POST", commit_url, safe_plan["ticket"], headers),
            None,
        )
        if committed.status_code != 200:
            raise Qwen3TransferError(
                "qwen3_transfer_commit_rejected", "transfer commit was rejected",
            )
        receipt = self._json(committed)
        expected_receipt = {
            "transfer_id": safe_plan["transfer_id"],
            "chain_id": descriptor["chain_id"],
            "generation": descriptor["generation"],
            "phase": descriptor["phase"],
            "from_segment": descriptor["from_segment"],
            "to_segment": descriptor["to_segment"],
            "size_bytes": total_bytes,
            "sha256": sha256.lower(),
            "peer_epoch": descriptor["peer_epoch"],
            "received_bytes": total_bytes,
            "status": "committed",
            "full_model_materialized": False,
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise Qwen3TransferError(
                "qwen3_transfer_receipt_mismatch", "transfer receipt does not match source",
            )
        return {
            key: receipt[key]
            for key in expected_receipt
        }

    def upload(self, *, source: str | Path, plan: Mapping[str, Any]) -> dict[str, Any]:
        safe_plan = self._plan(plan)
        path = self._source(source)
        descriptor = safe_plan["descriptor"]
        actual_size, actual_sha256 = _file_evidence(path)
        if actual_size != descriptor["size_bytes"] or actual_sha256 != descriptor["sha256"]:
            raise Qwen3TransferError(
                "qwen3_transfer_source_mismatch", "source artifact does not match transfer plan",
            )

        def provider(offset: int, limit: int) -> bytes:
            with path.open("rb") as handle:
                handle.seek(offset)
                return handle.read(limit)

        return self.upload_chunks(
            plan=safe_plan,
            total_bytes=actual_size,
            sha256=actual_sha256,
            chunk_provider=provider,
        )


__all__ = [
    "MAX_CONTROL_RESPONSE_BYTES",
    "MAX_TRANSFER_ARTIFACT_BYTES",
    "MAX_TRANSFER_CHUNK_BYTES",
    "MAX_TRANSFER_TICKET_SECONDS",
    "QWEN3_TRANSFER_PREFIX",
    "QWEN3_TRANSFER_SCHEMA_VERSION",
    "Qwen3ArtifactReceiver",
    "Qwen3ArtifactTransferClient",
    "Qwen3TransferError",
    "Qwen3TransferTicketSigner",
    "TransferRequester",
    "TransferResponse",
    "default_transfer_request",
]
