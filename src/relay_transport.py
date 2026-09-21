"""Framed loopback transport for experimental Relay hidden-state handoff.

The transport is deliberately independent from both inference engines. It carries raw
little-endian hidden tensors to the existing stdio runner while enforcing framing, bounds,
ordering, and loopback-only exposure. Passing this transport does not open the Relay
production admission gate.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import struct
import subprocess
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import BinaryIO, Sequence

RELAY_WIRE_MAGIC = b"QLHR"
RELAY_WIRE_VERSION = 1
RELAY_DTYPE = "float32_le"
RELAY_DTYPE_BYTES = 4
RELAY_DEFAULT_MAX_TOKENS = 4096
RELAY_DEFAULT_MAX_PAYLOAD = 256 * 1024 * 1024
#: ★ P3：`HIDDEN_SEQ` 帧的元数据（seq/pos）上限 —— 帧校验按 hidden 字节 + 该上限放宽。
RELAY_SEQ_META_LIMIT = 64 * 1024

_HEADER = struct.Struct("!4sBBHIIQ")
_TOKEN = struct.Struct("<i")
_COUNT = struct.Struct("<i")
#: ★ P3：`HIDDEN_SEQ` payload 头 = `n_tokens` + 元数据字节数（其后是 meta JSON + f32 数据）。
_SEQ_HEADER = struct.Struct("<II")
_RELAY_ERROR_PAYLOAD_LIMIT = 64

logger = logging.getLogger(__name__)

# These are the only values allowed to cross the Relay trust boundary.  The
# existing protocol codes remain stable; implementation failures are grouped
# so exception class names and messages never become wire data.
RELAY_REMOTE_ERROR = "remote_error"
RELAY_PROTOCOL_ERROR = "relay_protocol_error"
RELAY_TRANSPORT_ERROR = "relay_transport_error"
RELAY_INTERNAL_ERROR = "relay_internal_error"
RELAY_RUNNER_ERROR = "runner_failed"
_RELAY_ERROR_CODES = frozenset({
    "client_closed",
    "connection_closed_mid_frame",
    "hidden_frame_required",
    "hidden_payload_size_mismatch",
    "hidden_seq_frame_required",
    "hidden_seq_meta_invalid",
    "hidden_seq_meta_too_large",
    "hidden_seq_meta_truncated",
    "hidden_seq_meta_unknown",
    "hidden_seq_payload_too_small",
    "hidden_seq_unsupported",
    "invalid_close_ack",
    "invalid_close_frame",
    "invalid_frame_header",
    "invalid_hidden_shape",
    "invalid_token_response",
    "invalid_transport_limits",
    "non_loopback_bind_rejected",
    "non_loopback_endpoint_rejected",
    "payload_too_large",
    "request_sequence_mismatch",
    "response_sequence_mismatch",
    "runner_closed_mid_response",
    "runner_failed",
    "runner_not_available",
    "runner_pipe_missing",
    "token_count_exceeds_limit",
    "unknown_frame_kind",
    "unsupported_flags",
    "unsupported_version",
    RELAY_INTERNAL_ERROR,
    RELAY_PROTOCOL_ERROR,
    RELAY_REMOTE_ERROR,
    RELAY_TRANSPORT_ERROR,
})


class RelayFrameKind(IntEnum):
    HIDDEN = 1
    TOKEN = 2
    CLOSE = 3
    ERROR = 4
    #: ★ P3：带 `seq_ids` / `positions` 的 hidden 请求（响应仍是纯 `HIDDEN`）——
    #: 多序列数据流跨机时必须逐 token 显式绑定，不能靠远端隐式位置递增。
    HIDDEN_SEQ = 5


class RelayProtocolError(RuntimeError):
    """The peer or runner violated the experimental Relay wire contract."""


@dataclass(frozen=True)
class RelayFrame:
    kind: RelayFrameKind
    sequence: int
    n_tokens: int = 0
    payload: bytes = b""


@dataclass(frozen=True)
class RelayBridgeResult:
    frames: int
    tokens: int
    payload_bytes: int
    closed_cleanly: bool
    error: str = ""

    def to_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


def is_loopback_host(host: str) -> bool:
    candidate = str(host).strip().lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def expected_hidden_bytes(n_tokens: int, n_embd: int) -> int:
    tokens = int(n_tokens)
    width = int(n_embd)
    if tokens < 1 or width < 1:
        raise RelayProtocolError("invalid_hidden_shape")
    return tokens * width * RELAY_DTYPE_BYTES


def encode_hidden_seq(hidden: bytes, *, n_tokens: int, meta: dict[str, object]) -> bytes:
    """★ P3：把 `hidden` + 元数据打包成 `HIDDEN_SEQ` 的 payload。

    布局：`<n_tokens:u32><n_meta_bytes:u32><meta JSON><f32 hidden>`。
    元数据只允许 `n_seq_id` / `seq_ids` / `positions`（其余键一律拒绝，避免协议被当通用通道）。
    """
    allowed = {"n_seq_id", "seq_ids", "positions"}
    unknown = set(meta) - allowed
    if unknown:
        raise RelayProtocolError(f"hidden_seq_meta_unknown:{sorted(unknown)[0]}")
    meta_bytes = json.dumps(meta, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    if len(meta_bytes) > RELAY_SEQ_META_LIMIT:
        raise RelayProtocolError("hidden_seq_meta_too_large")
    return _SEQ_HEADER.pack(int(n_tokens), len(meta_bytes)) + meta_bytes + bytes(hidden)


def decode_hidden_seq(payload: bytes, *, n_embd: int) -> tuple[bytes, int, dict[str, object]]:
    """★ P3：拆 `HIDDEN_SEQ` payload → `(hidden, n_tokens, meta)`，形状不符即拒。"""
    if len(payload) < _SEQ_HEADER.size:
        raise RelayProtocolError("hidden_seq_payload_too_small")
    n_tokens, meta_len = _SEQ_HEADER.unpack_from(payload, 0)
    if meta_len > RELAY_SEQ_META_LIMIT:
        raise RelayProtocolError("hidden_seq_meta_too_large")
    start = _SEQ_HEADER.size
    end = start + int(meta_len)
    if end > len(payload):
        raise RelayProtocolError("hidden_seq_meta_truncated")
    try:
        meta = json.loads(payload[start:end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RelayProtocolError("hidden_seq_meta_invalid") from exc
    if not isinstance(meta, dict):
        raise RelayProtocolError("hidden_seq_meta_invalid")
    hidden = payload[end:]
    if len(hidden) != expected_hidden_bytes(int(n_tokens), int(n_embd)):
        raise RelayProtocolError("hidden_payload_size_mismatch")
    return hidden, int(n_tokens), meta


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RelayProtocolError("connection_closed_mid_frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, frame: RelayFrame) -> None:
    payload = bytes(frame.payload)
    try:
        header = _HEADER.pack(
            RELAY_WIRE_MAGIC,
            RELAY_WIRE_VERSION,
            int(frame.kind),
            0,
            int(frame.sequence),
            int(frame.n_tokens),
            len(payload),
        )
    except (OverflowError, struct.error, ValueError) as exc:
        raise RelayProtocolError("invalid_frame_header") from exc
    sock.sendall(header + payload)


def recv_frame(
    sock: socket.socket, *, max_payload_bytes: int = RELAY_DEFAULT_MAX_PAYLOAD
) -> RelayFrame:
    raw = _recv_exact(sock, _HEADER.size)
    magic, version, kind_value, flags, sequence, n_tokens, payload_size = _HEADER.unpack(raw)
    if magic != RELAY_WIRE_MAGIC:
        raise RelayProtocolError("invalid_magic")
    if version != RELAY_WIRE_VERSION:
        raise RelayProtocolError("unsupported_version")
    if flags != 0:
        raise RelayProtocolError("unsupported_flags")
    try:
        kind = RelayFrameKind(kind_value)
    except ValueError as exc:
        raise RelayProtocolError("unknown_frame_kind") from exc
    if payload_size > int(max_payload_bytes):
        raise RelayProtocolError("payload_too_large")
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    return RelayFrame(kind=kind, sequence=sequence, n_tokens=n_tokens, payload=payload)


def _decode_token(frame: RelayFrame, expected_sequence: int) -> int:
    if frame.kind == RelayFrameKind.ERROR:
        if frame.sequence != expected_sequence:
            raise RelayProtocolError("response_sequence_mismatch")
        if len(frame.payload) > _RELAY_ERROR_PAYLOAD_LIMIT:
            raise RelayProtocolError(RELAY_REMOTE_ERROR)
        try:
            code = frame.payload.decode("ascii")
        except UnicodeDecodeError:
            code = RELAY_REMOTE_ERROR
        if code not in _RELAY_ERROR_CODES:
            code = RELAY_REMOTE_ERROR
        raise RelayProtocolError(code)
    if frame.sequence != expected_sequence:
        raise RelayProtocolError("response_sequence_mismatch")
    if frame.kind != RelayFrameKind.TOKEN or frame.n_tokens != 1 or len(frame.payload) != 4:
        raise RelayProtocolError("invalid_token_response")
    return _TOKEN.unpack(frame.payload)[0]


class RelayTcpClient:
    """One ordered Relay session over loopback or a local SSH tunnel endpoint."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        n_embd: int,
        timeout: float = 60.0,
        max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
    ) -> None:
        if not is_loopback_host(host):
            raise RelayProtocolError("non_loopback_endpoint_rejected")
        if int(n_embd) < 1 or int(max_tokens) < 1:
            raise RelayProtocolError("invalid_transport_limits")
        self.host = "127.0.0.1" if str(host).lower() == "localhost" else str(host)
        self.port = int(port)
        self.n_embd = int(n_embd)
        self.max_tokens = int(max_tokens)
        self.max_payload_bytes = (expected_hidden_bytes(self.max_tokens, self.n_embd)
                                 + RELAY_SEQ_META_LIMIT + _SEQ_HEADER.size)
        self._sock = socket.create_connection((self.host, self.port), timeout=float(timeout))
        self._sock.settimeout(float(timeout))
        self._sequence = 0
        self._closed = False

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        sequence = self._sequence
        send_frame(
            self._sock,
            RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=count, payload=hidden),
        )
        response = recv_frame(self._sock, max_payload_bytes=self.max_payload_bytes)
        token = _decode_token(response, sequence)
        if token < 0:
            raise RelayProtocolError("runner_failed")
        self._sequence += 1
        return token

    def request_hidden(self, hidden: bytes, *, n_tokens: int) -> bytes:
        """★ 中间段往返：发 HIDDEN，收 HIDDEN（远端段交出它自己的 hidden）。

        与 `request_token`（末段，收 token）配对 —— 这正是「1 个 torch 上游 + n 个
        llama 下游」链式拼接所需的**两种远端角色**：中间段吐 hidden、末段吐 token。

        远端实现的语义必须与本地 keep-head 一致（末层输出，`output_norm` 之前）；
        Android 侧由 `nativeLayerForwardHiddenKeepHead` 提供同一语义。
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        sequence = self._sequence
        send_frame(
            self._sock,
            RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=count, payload=hidden),
        )
        response = recv_frame(self._sock, max_payload_bytes=self.max_payload_bytes)
        if response.sequence != sequence:
            raise RelayProtocolError("response_sequence_mismatch")
        if response.kind != RelayFrameKind.HIDDEN:
            raise RelayProtocolError("hidden_response_required")
        if response.n_tokens != count:
            raise RelayProtocolError("hidden_token_count_mismatch")
        if len(response.payload) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        self._sequence += 1
        return bytes(response.payload)

    def request_hidden_seq(self, hidden: bytes, *, n_tokens: int,
                           meta: dict[str, object]) -> bytes:
        """★ P3：**多序列**中间段往返 —— 请求帧带 `seq_ids` / `positions`（`HIDDEN_SEQ`）。

        响应仍是纯 `HIDDEN`（远端已按显式 seq/pos 算完）。元数据只允许
        `n_seq_id` / `seq_ids` / `positions`（见 `encode_hidden_seq`）。
        """
        if self._closed:
            raise RelayProtocolError("client_closed")
        count = int(n_tokens)
        if count > self.max_tokens:
            raise RelayProtocolError("token_count_exceeds_limit")
        if len(hidden) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        payload = encode_hidden_seq(hidden, n_tokens=count, meta=meta)
        sequence = self._sequence
        send_frame(
            self._sock,
            RelayFrame(RelayFrameKind.HIDDEN_SEQ, sequence, n_tokens=count, payload=payload),
        )
        response = recv_frame(self._sock, max_payload_bytes=self.max_payload_bytes)
        if response.sequence != sequence:
            raise RelayProtocolError("response_sequence_mismatch")
        if response.kind != RelayFrameKind.HIDDEN:
            raise RelayProtocolError("hidden_response_required")
        if response.n_tokens != count:
            raise RelayProtocolError("hidden_token_count_mismatch")
        if len(response.payload) != expected_hidden_bytes(count, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        self._sequence += 1
        return bytes(response.payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            send_frame(self._sock, RelayFrame(RelayFrameKind.CLOSE, self._sequence))
            response = recv_frame(self._sock, max_payload_bytes=1024)
            token = _decode_token(response, self._sequence)
            if token != -1:
                raise RelayProtocolError("invalid_close_ack")
        finally:
            self._sock.close()

    def __enter__(self) -> "RelayTcpClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._closed = True
            self._sock.close()


class StdioRelayRunner:
    """Adapter for the current ``int32 count + float32 hidden -> int32 token`` runner."""

    def __init__(self, command: Sequence[str], *, n_embd: int, timeout: float = 60.0) -> None:
        self.n_embd = int(n_embd)
        self.timeout = float(timeout)
        if self.n_embd < 1 or not command:
            raise RelayProtocolError("invalid_runner_configuration")
        self._process = subprocess.Popen(
            [str(part) for part in command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._closed = False

    @staticmethod
    def _read_pipe_exact(pipe: BinaryIO, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = pipe.read(remaining)
            if not chunk:
                raise RelayProtocolError("runner_closed_mid_response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def request_token(self, hidden: bytes, *, n_tokens: int) -> int:
        if self._closed or self._process.poll() is not None:
            raise RelayProtocolError("runner_not_available")
        if len(hidden) != expected_hidden_bytes(n_tokens, self.n_embd):
            raise RelayProtocolError("hidden_payload_size_mismatch")
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RelayProtocolError("runner_pipe_missing")
        stdin.write(_COUNT.pack(int(n_tokens)))
        stdin.write(hidden)
        stdin.flush()
        return _TOKEN.unpack(self._read_pipe_exact(stdout, _TOKEN.size))[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stdin = self._process.stdin
        stdout = self._process.stdout
        if self._process.poll() is None and stdin is not None and stdout is not None:
            try:
                stdin.write(_COUNT.pack(0))
                stdin.flush()
                self._read_pipe_exact(stdout, _TOKEN.size)
            except (BrokenPipeError, RelayProtocolError):
                pass
        try:
            self._process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)


def _safe_error_code(value: BaseException | str, fallback: str) -> str:
    candidate = getattr(value, "code", None) or str(value)
    return candidate if candidate in _RELAY_ERROR_CODES else fallback


def _send_error(sock: socket.socket, sequence: int, code: str) -> None:
    payload = _safe_error_code(code, RELAY_INTERNAL_ERROR).encode("ascii")
    if len(payload) > _RELAY_ERROR_PAYLOAD_LIMIT:
        payload = RELAY_INTERNAL_ERROR.encode("ascii")
    send_frame(sock, RelayFrame(RelayFrameKind.ERROR, sequence, payload=payload))


def serve_relay_connection(
    sock: socket.socket,
    runner,
    *,
    n_embd: int,
    max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
) -> RelayBridgeResult:
    """Forward one ordered client session to a runner and return bounded evidence stats."""

    width = int(n_embd)
    limit = int(max_tokens)
    max_payload = expected_hidden_bytes(limit, width)
    sequence = 0
    frames = 0
    tokens = 0
    payload_bytes = 0
    try:
        while True:
            frame = recv_frame(sock, max_payload_bytes=max_payload)
            if frame.sequence != sequence:
                raise RelayProtocolError("request_sequence_mismatch")
            if frame.kind == RelayFrameKind.CLOSE:
                if frame.n_tokens != 0 or frame.payload:
                    raise RelayProtocolError("invalid_close_frame")
                try:
                    runner.close()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay runner close failed: code=%s", RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1, payload=_TOKEN.pack(-1)),
                )
                return RelayBridgeResult(frames, tokens, payload_bytes, True)
            if frame.kind != RelayFrameKind.HIDDEN:
                raise RelayProtocolError("hidden_frame_required")
            if frame.n_tokens < 1 or frame.n_tokens > limit:
                raise RelayProtocolError("token_count_exceeds_limit")
            if len(frame.payload) != expected_hidden_bytes(frame.n_tokens, width):
                raise RelayProtocolError("hidden_payload_size_mismatch")

            try:
                token = int(runner.request_token(frame.payload, n_tokens=frame.n_tokens))
            except RelayProtocolError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Relay runner request failed: code=%s", RELAY_RUNNER_ERROR)
                raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
            if token < 0:
                raise RelayProtocolError("runner_failed")
            send_frame(
                sock,
                RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1, payload=_TOKEN.pack(token)),
            )
            frames += 1
            tokens += frame.n_tokens
            payload_bytes += len(frame.payload)
            sequence += 1
    except RelayProtocolError as exc:
        code = _safe_error_code(exc, RELAY_PROTOCOL_ERROR)
        logger.warning("Relay protocol failure: code=%s detail=%s", code, str(exc))
        try:
            _send_error(sock, sequence, code)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, code)
    except OSError as exc:
        logger.warning("Relay transport failure: code=%s detail=%s", RELAY_TRANSPORT_ERROR, exc)
        try:
            _send_error(sock, sequence, RELAY_TRANSPORT_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_TRANSPORT_ERROR)
    except Exception as exc:  # noqa: BLE001
        # Never put exception class names or messages on the Relay wire.
        logger.exception("Relay internal failure: code=%s", RELAY_INTERNAL_ERROR)
        try:
            _send_error(sock, sequence, RELAY_INTERNAL_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_INTERNAL_ERROR)


def open_loopback_listener(host: str, port: int, *, backlog: int = 1) -> socket.socket:
    """Create a listener that cannot be exposed directly to a LAN or tailnet."""

    if not is_loopback_host(host):
        raise RelayProtocolError("non_loopback_bind_rejected")
    bind_host = "127.0.0.1" if str(host).lower() == "localhost" else str(host)
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_host, int(port)))
    listener.listen(max(1, int(backlog)))
    return listener


def serve_relay_middle_connection(
    sock: socket.socket,
    runner,
    *,
    n_embd: int,
    max_tokens: int = RELAY_DEFAULT_MAX_TOKENS,
) -> RelayBridgeResult:
    """★ 中间段服务：HIDDEN → `runner.request_hidden()` → HIDDEN（末位 argmax 不传）。

    与 `serve_relay_connection`（末段，回 TOKEN）配对。`runner` 必须提供
    `request_hidden(hidden_bytes, n_tokens=...) -> bytes` 与 `close()`；
    主仓的 `llama_keep_head.KeepHeadUpstream`（经 `forward_hidden_to_hidden`）与
    Android 的 `nativeLayerForwardHiddenKeepHead` 语义一致。
    """

    width = int(n_embd)
    limit = int(max_tokens)
    max_payload = expected_hidden_bytes(limit, width)
    sequence = 0
    frames = 0
    tokens = 0
    payload_bytes = 0
    try:
        while True:
            frame = recv_frame(sock, max_payload_bytes=max_payload)
            if frame.sequence != sequence:
                raise RelayProtocolError("request_sequence_mismatch")
            if frame.kind == RelayFrameKind.CLOSE:
                if frame.n_tokens != 0 or frame.payload:
                    raise RelayProtocolError("invalid_close_frame")
                try:
                    runner.close()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay middle runner close failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.TOKEN, sequence, n_tokens=1,
                               payload=_TOKEN.pack(-1)),
                )
                return RelayBridgeResult(frames, tokens, payload_bytes, True)
            if frame.kind == RelayFrameKind.HIDDEN_SEQ:
                # ★ P3 多序列：payload 自带 seq/pos；远端 runner 必须支持显式绑定，
                # 否则 fail-loud（绝不退回"远端按隐式位置猜"——那会静默算错）。
                hidden, n_tokens, meta = decode_hidden_seq(frame.payload, n_embd=width)
                if n_tokens < 1 or n_tokens > limit:
                    raise RelayProtocolError("token_count_exceeds_limit")
                if not hasattr(runner, "request_hidden_seq"):
                    raise RelayProtocolError("hidden_seq_unsupported")
                try:
                    produced = bytes(runner.request_hidden_seq(hidden, n_tokens=n_tokens,
                                                               meta=meta))
                except RelayProtocolError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Relay middle runner (seq) failed: code=%s",
                                     RELAY_RUNNER_ERROR)
                    raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
                if len(produced) != expected_hidden_bytes(n_tokens, width):
                    raise RelayProtocolError("hidden_payload_size_mismatch")
                send_frame(
                    sock,
                    RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=n_tokens,
                               payload=produced),
                )
                frames += 1
                tokens += n_tokens
                payload_bytes += len(hidden)
                sequence += 1
                continue
            if frame.kind != RelayFrameKind.HIDDEN:
                raise RelayProtocolError("hidden_frame_required")
            if frame.n_tokens < 1 or frame.n_tokens > limit:
                raise RelayProtocolError("token_count_exceeds_limit")
            if len(frame.payload) != expected_hidden_bytes(frame.n_tokens, width):
                raise RelayProtocolError("hidden_payload_size_mismatch")

            try:
                produced = bytes(runner.request_hidden(frame.payload, n_tokens=frame.n_tokens))
            except RelayProtocolError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Relay middle runner failed: code=%s", RELAY_RUNNER_ERROR)
                raise RelayProtocolError(RELAY_RUNNER_ERROR) from exc
            if len(produced) != expected_hidden_bytes(frame.n_tokens, width):
                raise RelayProtocolError("hidden_payload_size_mismatch")
            send_frame(
                sock,
                RelayFrame(RelayFrameKind.HIDDEN, sequence, n_tokens=frame.n_tokens,
                           payload=produced),
            )
            frames += 1
            tokens += frame.n_tokens
            payload_bytes += len(frame.payload)
            sequence += 1
    except RelayProtocolError as exc:
        code = _safe_error_code(exc, RELAY_PROTOCOL_ERROR)
        logger.warning("Relay middle protocol failure: code=%s detail=%s", code, str(exc))
        try:
            _send_error(sock, sequence, code)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, code)
    except OSError as exc:
        logger.warning("Relay middle transport failure: code=%s detail=%s",
                       RELAY_TRANSPORT_ERROR, exc)
        try:
            _send_error(sock, sequence, RELAY_TRANSPORT_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_TRANSPORT_ERROR)
    except Exception as exc:  # noqa: BLE001
        # Never put exception class names or messages on the Relay wire.
        logger.exception("Relay middle internal failure: code=%s", RELAY_INTERNAL_ERROR)
        try:
            _send_error(sock, sequence, RELAY_INTERNAL_ERROR)
        except OSError:
            pass
        return RelayBridgeResult(frames, tokens, payload_bytes, False, RELAY_INTERNAL_ERROR)
