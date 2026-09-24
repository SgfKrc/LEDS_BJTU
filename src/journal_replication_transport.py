"""Authenticated framed TCP transport for durable journal checkpoints.

The transport is intentionally small and synchronous. Deployments should run
it over an authenticated private network or TLS tunnel; the HMAC binds the
declared source identity and checkpoint to a shared secret, but does not
provide confidentiality or quorum commitment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import threading
from typing import Any, Mapping

from journal_replication import (
    DurableJournalReplica,
    JournalCheckpoint,
    JournalReplicationError,
    REPLICATION_SCHEMA_VERSION,
)


TRANSPORT_SCHEMA_VERSION = "qlh.journal.replication.transport.v1"
DEFAULT_MAX_FRAME_BYTES = 32 * 1024 * 1024


class JournalReplicationTransportError(ConnectionError):
    """Transport, authentication, or remote apply failure."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JournalReplicationTransportError("transport payload is not canonical JSON") from exc


def _mac(secret: bytes, value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "mac"}
    return hmac.new(secret, _canonical(unsigned), hashlib.sha256).hexdigest()


def _recv_exact(connection: socket.socket, length: int, max_frame_bytes: int) -> bytes:
    if length < 0 or length > max_frame_bytes:
        raise JournalReplicationTransportError("invalid frame length")
    output = bytearray()
    while len(output) < length:
        chunk = connection.recv(length - len(output))
        if not chunk:
            raise JournalReplicationTransportError("peer closed before frame completed")
        output.extend(chunk)
    return bytes(output)


def _read_frame(connection: socket.socket, max_frame_bytes: int) -> dict[str, Any]:
    (length,) = struct.unpack("!I", _recv_exact(connection, 4, max_frame_bytes))
    if length == 0:
        raise JournalReplicationTransportError("empty frame")
    try:
        value = json.loads(_recv_exact(connection, length, max_frame_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JournalReplicationTransportError("invalid JSON frame") from exc
    if not isinstance(value, dict):
        raise JournalReplicationTransportError("frame must contain an object")
    return value


def _write_frame(connection: socket.socket, value: Mapping[str, Any], max_frame_bytes: int) -> None:
    body = _canonical(value)
    if not body or len(body) > max_frame_bytes:
        raise JournalReplicationTransportError("frame exceeds configured limit")
    connection.sendall(struct.pack("!I", len(body)) + body)


class JournalReplicationTcpServer:
    """Single-checkpoint TCP receiver backed by ``DurableJournalReplica``."""

    def __init__(
        self,
        replica: DurableJournalReplica,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        secret: bytes,
        timeout: float = 5.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if len(secret) < 32:
            raise JournalReplicationTransportError("replication secret must be at least 32 bytes")
        if max_frame_bytes < 1024:
            raise JournalReplicationTransportError("max_frame_bytes must be at least 1024")
        self.replica = replica
        self.bind_host = bind_host
        self.port = int(port)
        self.secret = bytes(secret)
        self.timeout = max(0.1, float(timeout))
        self.max_frame_bytes = int(max_frame_bytes)
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self._ready = threading.Event()
        self.address: tuple[str, int] | None = None

    def start(self) -> tuple[str, int]:
        if self._thread is not None:
            raise JournalReplicationTransportError("server has already been started")
        self._thread = threading.Thread(
            target=self._serve,
            name="journal-replication-tcp",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(self.timeout):
            self.close()
            raise JournalReplicationTransportError("replication receiver startup timed out")
        if self._startup_error is not None:
            raise JournalReplicationTransportError("replication receiver failed to start") from self._startup_error
        assert self.address is not None
        return self.address

    def _serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.bind_host, self.port))
            listener.listen(8)
            listener.settimeout(0.2)
            self._listener = listener
            bound = listener.getsockname()
            self.address = (str(bound[0]), int(bound[1]))
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connection, _peer = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(self.timeout)
                    self._handle(connection)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        finally:
            try:
                listener.close()
            except OSError:
                pass

    def _handle(self, connection: socket.socket) -> None:
        try:
            request = _read_frame(connection, self.max_frame_bytes)
            supplied_mac = request.get("mac")
            if not isinstance(supplied_mac, str) or not hmac.compare_digest(
                supplied_mac, _mac(self.secret, request)
            ):
                raise JournalReplicationTransportError("authentication_failed")
            if request.get("transport_schema") != TRANSPORT_SCHEMA_VERSION:
                raise JournalReplicationTransportError("unsupported_transport_schema")
            if request.get("operation") != "apply_checkpoint":
                raise JournalReplicationTransportError("unsupported_operation")
            raw_checkpoint = request.get("checkpoint")
            if not isinstance(raw_checkpoint, Mapping):
                raise JournalReplicationTransportError("checkpoint_required")
            checkpoint = JournalCheckpoint.from_dict(raw_checkpoint)
            if request.get("source_node_id") != checkpoint.source_node_id:
                raise JournalReplicationTransportError("source_identity_mismatch")
            result = self.replica.apply_checkpoint(checkpoint)
            response: dict[str, Any] = {
                "transport_schema": TRANSPORT_SCHEMA_VERSION,
                "ok": True,
                "result": result,
            }
        except (JournalReplicationError, JournalReplicationTransportError, ValueError):
            response = {
                "transport_schema": TRANSPORT_SCHEMA_VERSION,
                "ok": False,
                "error": "replication_rejected",
            }
        except (OSError, socket.timeout):
            return
        try:
            response["mac"] = _mac(self.secret, response)
            _write_frame(connection, response, self.max_frame_bytes)
        except (OSError, JournalReplicationTransportError):
            return

    def close(self) -> None:
        self._stop.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout + 1.0)
            if thread.is_alive():
                raise JournalReplicationTransportError("replication receiver did not stop")

    def __enter__(self) -> "JournalReplicationTcpServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def send_checkpoint(
    host: str,
    port: int,
    checkpoint: JournalCheckpoint,
    *,
    secret: bytes,
    timeout: float = 5.0,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Send one authenticated checkpoint and verify its durable ACK."""

    if len(secret) < 32:
        raise JournalReplicationTransportError("replication secret must be at least 32 bytes")
    request: dict[str, Any] = {
        "transport_schema": TRANSPORT_SCHEMA_VERSION,
        "operation": "apply_checkpoint",
        "source_node_id": checkpoint.source_node_id,
        "checkpoint": checkpoint.to_dict(),
    }
    request["mac"] = _mac(secret, request)
    try:
        with socket.create_connection((host, int(port)), timeout=max(0.1, timeout)) as connection:
            connection.settimeout(max(0.1, timeout))
            _write_frame(connection, request, max_frame_bytes)
            response = _read_frame(connection, max_frame_bytes)
    except (OSError, JournalReplicationTransportError) as exc:
        raise JournalReplicationTransportError("journal replication exchange failed") from exc
    supplied_mac = response.get("mac")
    if not isinstance(supplied_mac, str) or not hmac.compare_digest(
        supplied_mac, _mac(secret, response)
    ):
        raise JournalReplicationTransportError("replication acknowledgement authentication failed")
    if response.get("transport_schema") != TRANSPORT_SCHEMA_VERSION:
        raise JournalReplicationTransportError("unsupported acknowledgement schema")
    if response.get("ok") is not True or not isinstance(response.get("result"), dict):
        raise JournalReplicationTransportError("remote replica rejected checkpoint")
    result = response["result"]
    if (
        result.get("source_node_id") != checkpoint.source_node_id
        or result.get("stream_id") != checkpoint.stream_id
        or result.get("workflow_id") != checkpoint.workflow_id
        or int(result.get("durable_sequence", 0)) != checkpoint.last_sequence
        or result.get("checkpoint_digest") != checkpoint.checkpoint_digest
    ):
        raise JournalReplicationTransportError("replica acknowledgement watermark mismatch")
    return result


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "JournalReplicationTcpServer",
    "JournalReplicationTransportError",
    "TRANSPORT_SCHEMA_VERSION",
    "send_checkpoint",
]
