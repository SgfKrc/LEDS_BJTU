"""Local-only WSS transport for the NW3.1 development gate.

This module is deliberately separate from the production TCP route.  It
creates a short-lived, user-owned CA/server certificate, binds the listener to
loopback, authenticates a node with a TLS-pinned HMAC challenge, and carries
only Transport v2 envelopes.  It is suitable for protocol tests and local
diagnostics; it is not a public certificate or a production gateway.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import ssl
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cluster_transport import TransportContractError, TransportEnvelope


WSS_LOOPBACK_SCHEMA = "qlh.wss_loopback.v1"
_MAX_MESSAGE_BYTES = 1 << 20
_VALID_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class WssLoopbackError(ValueError):
    """Stable NW3.1 local WSS error."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(message)


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(bytes(value)).decode("ascii")


def _b64_decode(value: object, *, code: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > _MAX_MESSAGE_BYTES * 2:
        raise WssLoopbackError(code, "encoded WSS value is invalid")
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise WssLoopbackError(code, "encoded WSS value is invalid") from exc


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class LocalWssMaterials:
    """Ephemeral or user-owned certificate material for a loopback endpoint."""

    ca_certificate_pem: bytes
    server_certificate_pem: bytes
    server_private_key_pem: bytes
    server_fingerprint_sha256: str

    @classmethod
    def generate(cls, *, valid_days: int = 30) -> "LocalWssMaterials":
        if isinstance(valid_days, bool) or not 1 <= int(valid_days) <= 365:
            raise WssLoopbackError("certificate_validity_invalid", "certificate validity is outside the local limit")
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.x509.oid import NameOID
        except ImportError as exc:  # pragma: no cover - dependency is in runtime requirements
            raise WssLoopbackError("certificate_dependency_missing", "cryptography is required for local WSS") from exc

        now = datetime.now(timezone.utc)
        ca_key = Ed25519PrivateKey.generate()
        server_key = Ed25519PrivateKey.generate()
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QLH Local WSS CA")])
        server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "QLH Local WSS")])
        ca_certificate = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=int(valid_days)))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(ca_key, algorithm=None)
        )
        server_certificate = (
            x509.CertificateBuilder()
            .subject_name(server_name)
            .issuer_name(ca_certificate.subject)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=int(valid_days)))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address("::1")),
                ]),
                critical=False,
            )
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(ca_key, algorithm=None)
        )
        return cls(
            ca_certificate.public_bytes(serialization.Encoding.PEM),
            server_certificate.public_bytes(serialization.Encoding.PEM),
            server_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            server_certificate.fingerprint(hashes.SHA256()).hex(),
        )

    def write(self, directory: str | os.PathLike[str]) -> Path:
        """Write local materials with a private-key file restricted to the user."""
        target = Path(directory).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        files = {
            "ca.pem": self.ca_certificate_pem,
            "server-cert.pem": self.server_certificate_pem,
            "server-key.pem": self.server_private_key_pem,
        }
        for name, content in files.items():
            path = target / name
            path.write_bytes(content)
            if name.endswith("key.pem"):
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
        return target


def _server_context(materials: LocalWssMaterials, directory: Path) -> ssl.SSLContext:
    materials.write(directory)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(directory / "server-cert.pem"), str(directory / "server-key.pem"))
    return context


def _client_context(materials: LocalWssMaterials) -> ssl.SSLContext:
    try:
        context = ssl.create_default_context(cadata=materials.ca_certificate_pem.decode("ascii"))
    except (UnicodeDecodeError, ssl.SSLError) as exc:
        raise WssLoopbackError("certificate_invalid", "local CA certificate cannot be loaded") from exc
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    return context


def _load_websockets() -> tuple[Any, Any, Any]:
    try:
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
        from websockets.exceptions import ConnectionClosed
    except ImportError as exc:  # pragma: no cover - exercised by packaging smoke
        raise WssLoopbackError("wss_dependency_missing", "websockets>=16 is required for local WSS") from exc
    return connect, serve, ConnectionClosed


def _validate_host(host: str) -> str:
    if not isinstance(host, str) or host not in _VALID_HOSTS:
        raise WssLoopbackError("loopback_host_required", "NW3.1 WSS must bind to loopback")
    return host


def _auth_digest(secret: bytes, node_id: str, nonce: bytes, fingerprint: str, generation: int, attempt_id: str) -> str:
    body = "\n".join((node_id, _b64_encode(nonce), fingerprint, str(generation), attempt_id)).encode("utf-8")
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def _decode_object(raw: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_MESSAGE_BYTES:
        raise WssLoopbackError(code, "WSS message exceeds the local limit")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WssLoopbackError(code, "WSS message is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WssLoopbackError(code, "WSS message must be an object")
    return value


@dataclass
class _ServerSession:
    node_id: str
    generation: int
    attempt_id: str
    websocket: Any
    last_sequences: dict[str, int]


class WssLoopbackServer:
    """Authenticated loopback WSS server for NW3.1 protocol tests."""

    def __init__(
        self,
        secret: bytes | str,
        *,
        materials: LocalWssMaterials | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
    ) -> None:
        self.host = _validate_host(host)
        if isinstance(port, bool) or not 0 <= int(port) <= 65_535:
            raise WssLoopbackError("port_invalid", "WSS port is invalid")
        if isinstance(max_message_bytes, bool) or not 4_096 <= int(max_message_bytes) <= 8 * 1024 * 1024:
            raise WssLoopbackError("message_limit_invalid", "WSS message limit is invalid")
        self.secret = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if not self.secret:
            raise WssLoopbackError("secret_invalid", "WSS HMAC secret is required")
        self.materials = materials or LocalWssMaterials.generate()
        self.port = int(port)
        self.max_message_bytes = int(max_message_bytes)
        self._server: Any = None
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._sessions: dict[str, _ServerSession] = {}
        self._connect, self._serve, self._connection_closed = _load_websockets()

    @property
    def uri(self) -> str:
        if not self.port:
            raise WssLoopbackError("server_not_started", "WSS server is not started")
        host = "[::1]" if self.host == "::1" else self.host
        return f"wss://{host}:{self.port}"

    async def start(self) -> "WssLoopbackServer":
        if self._server is not None:
            return self
        self._tempdir = tempfile.TemporaryDirectory(prefix="qlh-wss-loopback-")
        context = _server_context(self.materials, Path(self._tempdir.name))
        try:
            self._server = await self._serve(
                self._handler,
                self.host,
                self.port,
                ssl=context,
                max_size=self.max_message_bytes,
                ping_interval=20,
                ping_timeout=20,
            )
        except Exception as exc:
            self._tempdir.cleanup()
            self._tempdir = None
            raise WssLoopbackError("server_start_failed", "WSS loopback server could not start") from exc
        sockets = list(self._server.sockets or ())
        if not sockets:
            await self.stop()
            raise WssLoopbackError("server_start_failed", "WSS loopback server has no listening socket")
        self.port = int(sockets[0].getsockname()[1])
        return self

    async def stop(self) -> None:
        for session in list(self._sessions.values()):
            try:
                await session.websocket.close(code=1001, reason="server_shutdown")
            except Exception:
                pass
        self._sessions.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    async def __aenter__(self) -> "WssLoopbackServer":
        return await self.start()

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def _send_error(self, websocket: Any, code: str) -> None:
        try:
            await websocket.send(_canonical_json({"schema": WSS_LOOPBACK_SCHEMA, "type": "error", "code": code}).decode("utf-8"))
        except Exception:
            pass

    async def _handler(self, websocket: Any) -> None:
        session: _ServerSession | None = None
        try:
            nonce = secrets.token_bytes(24)
            await websocket.send(_canonical_json({
                "schema": WSS_LOOPBACK_SCHEMA,
                "type": "challenge",
                "nonce": _b64_encode(nonce),
                "server_fingerprint_sha256": self.materials.server_fingerprint_sha256,
            }).decode("utf-8"))
            auth = _decode_object(await websocket.recv(), code="auth_invalid")
            expected_fields = {"schema", "type", "node_id", "generation", "attempt_id", "mac"}
            if set(auth) != expected_fields or auth.get("schema") != WSS_LOOPBACK_SCHEMA or auth.get("type") != "auth":
                raise WssLoopbackError("auth_invalid", "WSS authentication fields are invalid")
            node_id = auth.get("node_id")
            attempt_id = auth.get("attempt_id")
            generation = auth.get("generation")
            if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > 128:
                raise WssLoopbackError("auth_invalid", "node identity is invalid")
            if not isinstance(attempt_id, str) or not attempt_id.strip() or len(attempt_id) > 128:
                raise WssLoopbackError("auth_invalid", "attempt identity is invalid")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
                raise WssLoopbackError("auth_invalid", "connection generation is invalid")
            expected_mac = _auth_digest(self.secret, node_id, nonce, self.materials.server_fingerprint_sha256, generation, attempt_id)
            if not isinstance(auth.get("mac"), str) or not hmac.compare_digest(auth["mac"], expected_mac):
                raise WssLoopbackError("auth_failed", "WSS authentication failed")
            previous = self._sessions.get(node_id)
            if previous is not None:
                if generation <= previous.generation:
                    raise WssLoopbackError("generation_stale", "an equal or newer node generation is active")
                try:
                    await previous.websocket.close(code=4001, reason="generation_fenced")
                except Exception:
                    pass
            session = _ServerSession(node_id, generation, attempt_id, websocket, {})
            self._sessions[node_id] = session
            await websocket.send(_canonical_json({
                "schema": WSS_LOOPBACK_SCHEMA,
                "type": "auth_ok",
                "server_fingerprint_sha256": self.materials.server_fingerprint_sha256,
                "generation": generation,
            }).decode("utf-8"))
            while True:
                message = _decode_object(await websocket.recv(), code="frame_invalid")
                if message.get("schema") != WSS_LOOPBACK_SCHEMA or message.get("type") != "frame":
                    raise WssLoopbackError("frame_invalid", "WSS frame envelope is invalid")
                envelope_value = message.get("envelope")
                payload = _b64_decode(message.get("payload"), code="payload_invalid")
                envelope = TransportEnvelope.decode(_canonical_json(envelope_value) if isinstance(envelope_value, dict) else b"{}")
                if envelope.connection_generation != session.generation or envelope.attempt_id != session.attempt_id:
                    raise WssLoopbackError("attempt_fenced", "WSS frame belongs to an old attempt")
                if envelope.is_expired():
                    raise WssLoopbackError("deadline_exceeded", "WSS frame deadline has expired")
                if envelope.payload_size != len(payload) or envelope.payload_digest != hashlib.sha256(payload).hexdigest():
                    raise WssLoopbackError("payload_mismatch", "WSS frame payload does not match its envelope")
                previous_sequence = session.last_sequences.get(envelope.channel, -1)
                if envelope.sequence <= previous_sequence:
                    raise WssLoopbackError("sequence_duplicate", "WSS frame sequence was already delivered")
                if envelope.sequence != previous_sequence + 1:
                    raise WssLoopbackError("sequence_out_of_order", "WSS frame sequence is not contiguous")
                session.last_sequences[envelope.channel] = envelope.sequence
                await websocket.send(_canonical_json({
                    "schema": WSS_LOOPBACK_SCHEMA,
                    "type": "frame",
                    "envelope": envelope.to_dict(),
                    "payload": _b64_encode(payload),
                }).decode("utf-8"))
        except self._connection_closed:
            pass
        except (WssLoopbackError, TransportContractError) as exc:
            await self._send_error(websocket, getattr(exc, "code", "wss_protocol_error"))
            try:
                await websocket.close(code=1008, reason="wss_protocol_error")
            except Exception:
                pass
        except Exception:
            await self._send_error(websocket, "wss_protocol_error")
        finally:
            if session is not None and self._sessions.get(session.node_id) is session:
                self._sessions.pop(session.node_id, None)


class WssLoopbackClient:
    """Pinned and authenticated client for the local NW3.1 endpoint."""

    def __init__(
        self,
        secret: bytes | str,
        materials: LocalWssMaterials,
        *,
        node_id: str,
        generation: int,
        attempt_id: str,
        max_message_bytes: int = _MAX_MESSAGE_BYTES,
    ) -> None:
        self.secret = secret.encode("utf-8") if isinstance(secret, str) else bytes(secret)
        if not self.secret:
            raise WssLoopbackError("secret_invalid", "WSS HMAC secret is required")
        self.materials = materials
        if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > 128:
            raise WssLoopbackError("node_id_invalid", "WSS node identity is invalid")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise WssLoopbackError("generation_invalid", "WSS generation is invalid")
        if not isinstance(attempt_id, str) or not attempt_id.strip() or len(attempt_id) > 128:
            raise WssLoopbackError("attempt_id_invalid", "WSS attempt identity is invalid")
        if isinstance(max_message_bytes, bool) or not 4_096 <= int(max_message_bytes) <= 8 * 1024 * 1024:
            raise WssLoopbackError("message_limit_invalid", "WSS message limit is invalid")
        self.node_id = node_id
        self.generation = generation
        self.attempt_id = attempt_id
        self.max_message_bytes = int(max_message_bytes)
        self._websocket: Any = None
        self._connect, _serve, self._connection_closed = _load_websockets()

    async def connect(self, uri: str) -> "WssLoopbackClient":
        if self._websocket is not None:
            return self
        parsed = urlsplit(uri) if isinstance(uri, str) else None
        if parsed is None or parsed.scheme != "wss" or parsed.hostname not in _VALID_HOSTS or parsed.username or parsed.password:
            raise WssLoopbackError("uri_invalid", "NW3.1 client requires a wss:// URI")
        context = _client_context(self.materials)
        try:
            websocket = await self._connect(uri, ssl=context, max_size=self.max_message_bytes, server_hostname="localhost")
            challenge = _decode_object(await websocket.recv(), code="challenge_invalid")
            if (
                set(challenge) != {"schema", "type", "nonce", "server_fingerprint_sha256"}
                or challenge.get("schema") != WSS_LOOPBACK_SCHEMA
                or challenge.get("type") != "challenge"
            ):
                raise WssLoopbackError("challenge_invalid", "WSS challenge is invalid")
            fingerprint = challenge.get("server_fingerprint_sha256")
            if fingerprint != self.materials.server_fingerprint_sha256:
                raise WssLoopbackError("tls_fingerprint_mismatch", "WSS server certificate fingerprint does not match")
            nonce = _b64_decode(challenge.get("nonce"), code="challenge_invalid")
            await websocket.send(_canonical_json({
                "schema": WSS_LOOPBACK_SCHEMA,
                "type": "auth",
                "node_id": self.node_id,
                "generation": self.generation,
                "attempt_id": self.attempt_id,
                "mac": _auth_digest(self.secret, self.node_id, nonce, fingerprint, self.generation, self.attempt_id),
            }).decode("utf-8"))
            response = _decode_object(await websocket.recv(), code="auth_invalid")
            if (
                set(response) != {"schema", "type", "server_fingerprint_sha256", "generation"}
                or response.get("schema") != WSS_LOOPBACK_SCHEMA
                or response.get("type") != "auth_ok"
                or response.get("server_fingerprint_sha256") != fingerprint
                or response.get("generation") != self.generation
            ):
                raise WssLoopbackError("auth_failed", "WSS authentication was not accepted")
            self._verify_peer_fingerprint(websocket, fingerprint)
            self._websocket = websocket
            return self
        except (ssl.SSLError, OSError) as exc:
            try:
                await websocket.close()
            except Exception:
                pass
            raise WssLoopbackError("tls_auth_failed", "WSS TLS authentication failed") from exc
        except Exception:
            try:
                await websocket.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _verify_peer_fingerprint(websocket: Any, expected_fingerprint: str) -> None:
        transport = getattr(websocket, "transport", None)
        ssl_object = transport.get_extra_info("ssl_object") if transport is not None else None
        certificate = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
        if not certificate:
            raise WssLoopbackError("tls_peer_missing", "WSS peer certificate is unavailable")
        actual_fingerprint = hashlib.sha256(certificate).hexdigest()
        if not hmac.compare_digest(actual_fingerprint, expected_fingerprint):
            raise WssLoopbackError("tls_fingerprint_mismatch", "WSS peer certificate fingerprint does not match")

    async def exchange(self, envelope: TransportEnvelope, payload: bytes) -> tuple[TransportEnvelope, bytes]:
        if self._websocket is None:
            raise WssLoopbackError("client_not_connected", "WSS client is not connected")
        body = bytes(payload)
        if envelope.connection_generation != self.generation or envelope.attempt_id != self.attempt_id:
            raise WssLoopbackError("attempt_fenced", "WSS client envelope belongs to another attempt")
        if envelope.is_expired():
            raise WssLoopbackError("deadline_exceeded", "WSS client envelope deadline has expired")
        if envelope.payload_size != len(body) or envelope.payload_digest != hashlib.sha256(body).hexdigest():
            raise WssLoopbackError("payload_mismatch", "WSS client payload does not match its envelope")
        try:
            await self._websocket.send(_canonical_json({
                "schema": WSS_LOOPBACK_SCHEMA,
                "type": "frame",
                "envelope": envelope.to_dict(),
                "payload": _b64_encode(body),
            }).decode("utf-8"))
            response = _decode_object(await self._websocket.recv(), code="frame_invalid")
        except self._connection_closed as exc:
            raise WssLoopbackError("connection_closed", "WSS connection was closed") from exc
        if response.get("type") == "error":
            if set(response) != {"schema", "type", "code"} or response.get("schema") != WSS_LOOPBACK_SCHEMA:
                raise WssLoopbackError("frame_invalid", "WSS error frame is invalid")
            raise WssLoopbackError(str(response.get("code", "wss_protocol_error")), "WSS server rejected frame")
        if (
            set(response) != {"schema", "type", "envelope", "payload"}
            or response.get("schema") != WSS_LOOPBACK_SCHEMA
            or response.get("type") != "frame"
        ):
            raise WssLoopbackError("frame_invalid", "WSS response frame is invalid")
        returned = TransportEnvelope.decode(_canonical_json(response.get("envelope")) if isinstance(response.get("envelope"), dict) else b"{}")
        returned_payload = _b64_decode(response.get("payload"), code="payload_invalid")
        if returned != envelope or returned_payload != body:
            raise WssLoopbackError("echo_mismatch", "WSS response does not match the submitted frame")
        return returned, returned_payload

    async def close(self) -> None:
        if self._websocket is not None:
            try:
                await self._websocket.close()
            finally:
                self._websocket = None

    async def __aenter__(self) -> "WssLoopbackClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()
