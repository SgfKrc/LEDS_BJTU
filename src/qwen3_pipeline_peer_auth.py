"""Per-request peer proof for the Qwen3 artifact HTTP data plane."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from starlette.responses import JSONResponse

from qwen3_pipeline_transfer import QWEN3_TRANSFER_PREFIX


QWEN3_PEER_PROOF_HEADER = "X-QLH-Qwen3-Peer-Proof"
QWEN3_PEER_PROOF_SCHEMA_VERSION = 2
QWEN3_NETWORK_CONTROL_PREFIX = "/internal/v1/qwen3/network-control"
MAX_PEER_PROOF_SECONDS = 30
MAX_SEEN_NONCES = 4096

_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NONCE = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Qwen3PeerAuthError(RuntimeError):
    """A request did not prove an authenticated Qwen3 peer identity."""

    def __init__(self, reason_code: str, reason: str) -> None:
        self.reason_code = str(reason_code)[:128]
        self.reason = str(reason)[:1024]
        super().__init__(self.reason)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_invalid", "peer proof is not strict JSON",
        ) from exc
    if len(encoded) > 4096:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_invalid", "peer proof exceeds its byte limit",
        )
    return encoded


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if _B64URL.fullmatch(value) is None:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_invalid", "peer proof is malformed",
        )
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_invalid", "peer proof is malformed",
        ) from exc


def _request_path(value: str) -> str:
    parsed = urlsplit(str(value or ""))
    path = parsed.path if parsed.scheme else str(value or "").split("?", 1)[0]
    if (
        not any(
            path.startswith(f"{prefix}/")
            for prefix in (QWEN3_TRANSFER_PREFIX, QWEN3_NETWORK_CONTROL_PREFIX)
        )
        or len(path) > 512
    ):
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_scope", "peer proof path is outside the Qwen3 data plane",
        )
    return path


def _ticket_digest(ticket: str) -> str:
    if not isinstance(ticket, str) or not ticket or len(ticket) > 4096:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_ticket", "peer proof has no bounded transfer ticket",
        )
    try:
        encoded = ticket.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise Qwen3PeerAuthError(
            "qwen3_peer_proof_ticket", "peer proof transfer ticket is invalid",
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


class Qwen3PeerRequestSigner:
    """Sign one HTTP request as a named cluster peer."""

    def __init__(
        self,
        secret: str | bytes,
        *,
        peer_node_id: str,
        peer_epoch: int = 0,
        peer_epoch_provider: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(key) < 32:
            raise ValueError("Qwen3 peer proof secret must contain at least 32 bytes")
        peer = str(peer_node_id or "")
        if _NODE_ID.fullmatch(peer) is None:
            raise ValueError("Qwen3 peer proof node identity is invalid")
        self._key = key
        self.peer_node_id = peer
        if isinstance(peer_epoch, bool) or int(peer_epoch) < 0:
            raise ValueError("Qwen3 peer proof epoch is invalid")
        self._peer_epoch = int(peer_epoch)
        self._peer_epoch_provider = peer_epoch_provider
        self._clock = clock

    @property
    def peer_epoch(self) -> int:
        provider = self._peer_epoch_provider
        if provider is None:
            return self._peer_epoch
        try:
            value = provider()
        except Exception as exc:
            raise Qwen3PeerAuthError(
                "qwen3_peer_registry_unavailable", "local TCP peer epoch is unavailable",
            ) from exc
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "local TCP peer epoch is invalid",
            )
        return int(value)

    def proof(
        self,
        method: str,
        url_or_path: str,
        transfer_ticket: str,
        *,
        nonce: str | None = None,
    ) -> str:
        method_value = str(method or "").upper()
        if method_value not in {"GET", "PATCH", "POST", "DELETE"}:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_scope", "peer proof method is invalid",
            )
        nonce_value = str(nonce or uuid4().hex)
        if _NONCE.fullmatch(nonce_value) is None:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof nonce is invalid",
            )
        payload = {
            "schema_version": QWEN3_PEER_PROOF_SCHEMA_VERSION,
            "peer_node_id": self.peer_node_id,
            "peer_epoch": self.peer_epoch,
            "method": method_value,
            "path": _request_path(url_or_path),
            "ticket_sha256": _ticket_digest(transfer_ticket),
            "timestamp": int(self._clock()),
            "nonce": nonce_value,
        }
        encoded = _encode(_canonical_bytes(payload))
        signature = _encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest(),
        )
        return f"{encoded}.{signature}"

    def headers(
        self, method: str, url_or_path: str, transfer_ticket: str,
    ) -> dict[str, str]:
        return {
            QWEN3_PEER_PROOF_HEADER: self.proof(
                method, url_or_path, transfer_ticket,
            ),
        }


class Qwen3PeerRequestVerifier:
    """Verify proof and fence it against the live TCP peer registry."""

    def __init__(
        self,
        secret: str | bytes,
        *,
        is_authenticated_peer: Callable[[str], bool],
        is_authenticated_peer_epoch: Callable[[str, int], bool] | None = None,
        require_peer_epoch: bool = False,
        clock: Callable[[], float] = time.time,
        window_seconds: int = MAX_PEER_PROOF_SECONDS,
    ) -> None:
        key = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if len(key) < 32:
            raise ValueError("Qwen3 peer proof secret must contain at least 32 bytes")
        if not 1 <= int(window_seconds) <= MAX_PEER_PROOF_SECONDS:
            raise ValueError("Qwen3 peer proof window is outside limits")
        self._key = key
        self._is_authenticated_peer = is_authenticated_peer
        self._is_authenticated_peer_epoch = is_authenticated_peer_epoch
        self.require_peer_epoch = bool(require_peer_epoch)
        self._clock = clock
        self.window_seconds = int(window_seconds)
        self._lock = threading.RLock()
        self._seen: OrderedDict[tuple[str, str], int] = OrderedDict()

    def _remember(self, peer: str, nonce: str, timestamp: int) -> None:
        current = int(self._clock())
        with self._lock:
            for key, seen_at in list(self._seen.items()):
                if current - seen_at > self.window_seconds:
                    self._seen.pop(key, None)
            replay_key = (peer, nonce)
            if replay_key in self._seen:
                raise Qwen3PeerAuthError(
                    "qwen3_peer_proof_replay", "peer proof nonce was already used",
                )
            self._seen[replay_key] = timestamp
            while len(self._seen) > MAX_SEEN_NONCES:
                self._seen.popitem(last=False)

    def verify_identity(
        self,
        proof: str,
        *,
        method: str,
        path: str,
        transfer_ticket: str,
    ) -> str:
        if not isinstance(proof, str) or not proof or len(proof) > 8192:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_missing", "peer proof is missing",
            )
        try:
            encoded, signature = proof.split(".")
        except ValueError as exc:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof is malformed",
            ) from exc
        if _B64URL.fullmatch(encoded) is None or _B64URL.fullmatch(signature) is None:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof is malformed",
            )
        expected = _encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest(),
        )
        if not hmac.compare_digest(expected, signature):
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_signature", "peer proof signature is invalid",
            )
        try:
            payload = json.loads(_decode(encoded))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof payload is invalid",
            ) from exc
        required = {
            "schema_version", "peer_node_id", "method", "path",
            "ticket_sha256", "timestamp", "nonce", "peer_epoch",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof fields are invalid",
            )
        peer = str(payload.get("peer_node_id", ""))
        nonce = str(payload.get("nonce", ""))
        digest = str(payload.get("ticket_sha256", ""))
        timestamp = payload.get("timestamp")
        peer_epoch = payload.get("peer_epoch")
        if (
            payload.get("schema_version") != QWEN3_PEER_PROOF_SCHEMA_VERSION
            or _NODE_ID.fullmatch(peer) is None
            or _NONCE.fullmatch(nonce) is None
            or _SHA256.fullmatch(digest) is None
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or isinstance(peer_epoch, bool)
            or not isinstance(peer_epoch, int)
            or peer_epoch < 0
        ):
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_invalid", "peer proof fields are invalid",
            )
        if (
            payload.get("method") != str(method or "").upper()
            or payload.get("path") != _request_path(path)
            or digest != _ticket_digest(transfer_ticket)
        ):
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_scope", "peer proof request binding does not match",
            )
        if abs(int(self._clock()) - timestamp) > self.window_seconds:
            raise Qwen3PeerAuthError(
                "qwen3_peer_proof_expired", "peer proof is outside its time window",
            )
        try:
            authenticated = bool(self._is_authenticated_peer(peer))
        except Exception as exc:
            raise Qwen3PeerAuthError(
                "qwen3_peer_registry_unavailable", "TCP peer registry is unavailable",
            ) from exc
        if not authenticated:
            raise Qwen3PeerAuthError(
                "qwen3_peer_not_authenticated", "peer is not authenticated on TCP",
            )
        if self._is_authenticated_peer_epoch is not None:
            try:
                epoch_authenticated = bool(
                    self._is_authenticated_peer_epoch(peer, int(peer_epoch))
                )
            except Exception as exc:
                raise Qwen3PeerAuthError(
                    "qwen3_peer_registry_unavailable", "TCP peer registry is unavailable",
                ) from exc
            if not epoch_authenticated:
                raise Qwen3PeerAuthError(
                    "qwen3_peer_epoch_mismatch", "peer registration epoch is stale",
                )
        elif self.require_peer_epoch:
            raise Qwen3PeerAuthError(
                "qwen3_peer_registry_unavailable", "TCP peer epoch registry is unavailable",
            )
        self._remember(peer, nonce, timestamp)
        return peer, int(peer_epoch)

    def verify(
        self,
        proof: str,
        *,
        method: str,
        path: str,
        transfer_ticket: str,
    ) -> str:
        """Backward-compatible identity-only verification."""
        peer, _epoch = self.verify_identity(
            proof, method=method, path=path, transfer_ticket=transfer_ticket,
        )
        return peer


class Qwen3PeerAuthMiddleware:
    """Inject trusted peer identity for Qwen3 transfer routes only."""

    def __init__(self, app, *, verifier: Qwen3PeerRequestVerifier) -> None:
        self.app = app
        self.verifier = verifier

    async def __call__(self, scope, receive, send):
        path = str(scope.get("path", ""))
        protected = any(
            path.startswith(f"{prefix}/")
            for prefix in (QWEN3_TRANSFER_PREFIX, QWEN3_NETWORK_CONTROL_PREFIX)
        )
        if (
            scope.get("type") != "http"
            or not protected
            or path == f"{QWEN3_TRANSFER_PREFIX}/status"
        ):
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        authorization = headers.get("authorization", "")
        scheme, separator, ticket = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer":
            ticket = ""
        try:
            peer, peer_epoch = self.verifier.verify_identity(
                headers.get(QWEN3_PEER_PROOF_HEADER.lower(), ""),
                method=str(scope.get("method", "")),
                path=path,
                transfer_ticket=ticket,
            )
        except Qwen3PeerAuthError as exc:
            response = JSONResponse(
                status_code=401,
                content={"detail": {"code": exc.reason_code, "message": exc.reason}},
                headers={"WWW-Authenticate": "QLH-Qwen3-Peer"},
            )
            await response(scope, receive, send)
            return
        scope["qlh_authenticated_peer_id"] = peer
        scope["qlh_authenticated_peer_epoch"] = peer_epoch
        await self.app(scope, receive, send)


PeerProofHeaders = Callable[[str, str, str], Mapping[str, str]]


__all__ = [
    "MAX_PEER_PROOF_SECONDS",
    "QWEN3_NETWORK_CONTROL_PREFIX",
    "QWEN3_PEER_PROOF_HEADER",
    "PeerProofHeaders",
    "Qwen3PeerAuthError",
    "Qwen3PeerAuthMiddleware",
    "Qwen3PeerRequestSigner",
    "Qwen3PeerRequestVerifier",
]
