"""Local MCP stdio and SSE transports with no third-party dependencies."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import TextIOBase
from typing import Any, TextIO
from urllib.parse import parse_qs, urlsplit

from .contracts import MAX_JSON_BYTES
from .server import MCPServer


class StdioMCPTransport:
    """Serve newline-delimited JSON-RPC over stdin/stdout."""

    def __init__(
        self,
        server: MCPServer,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        max_line_bytes: int = MAX_JSON_BYTES,
    ) -> None:
        if not isinstance(server, MCPServer):
            raise TypeError("server must be an MCPServer")
        if isinstance(max_line_bytes, bool) or not isinstance(max_line_bytes, int) or not 1_024 <= max_line_bytes <= MAX_JSON_BYTES:
            raise ValueError("max_line_bytes is outside the MCP policy")
        self.server = server
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.max_line_bytes = max_line_bytes

    def handle_line(self, line: str | bytes) -> dict[str, Any] | None:
        if isinstance(line, bytes):
            try:
                encoded = line.rstrip(b"\r\n")
                if len(encoded) > self.max_line_bytes:
                    return self.server._error(None, -32600, "MCP message exceeds the size limit")
                line = encoded.decode("utf-8")
            except UnicodeDecodeError:
                return self.server._error(None, -32700, "invalid UTF-8")
        elif isinstance(line, str):
            if len(line.encode("utf-8")) > self.max_line_bytes:
                return self.server._error(None, -32600, "MCP message exceeds the size limit")
            line = line.rstrip("\r\n")
        else:
            return self.server._error(None, -32600, "MCP message must be text")
        if not line.strip():
            return None
        return self.server.handle_json(line)

    def run(self, *, input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> None:
        reader = input_stream or self.input_stream
        writer = output_stream or self.output_stream
        if reader is None or writer is None:
            import sys

            reader = reader or sys.stdin
            writer = writer or sys.stdout
        for line in reader:
            response = self.handle_line(line)
            if response is None:
                continue
            writer.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.flush()

    serve_forever = run


class _SSEClient:
    def __init__(self, writer: Any) -> None:
        self.writer = writer
        self.lock = threading.Lock()
        self.closed = False

    def send(self, event: str, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self.lock:
            if self.closed:
                return
            self.writer.write(f"event: {event}\ndata: {encoded}\n\n".encode("utf-8"))
            self.writer.flush()

    def close(self) -> None:
        with self.lock:
            self.closed = True


class SSE_MCPTransport:
    """Serve the classic MCP SSE handshake on a loopback HTTP listener.

    The listener is intentionally local-only.  It is useful for an offline
    client demo and deterministic tests; no authentication or public bind is
    implied by this transport.
    """

    def __init__(
        self,
        server: MCPServer,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        sse_path: str = "/sse",
        max_body_bytes: int = MAX_JSON_BYTES,
    ) -> None:
        if not isinstance(server, MCPServer):
            raise TypeError("server must be an MCPServer")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("MCP SSE transport is loopback-only")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("MCP SSE port is invalid")
        if not isinstance(sse_path, str) or not sse_path.startswith("/") or "?" in sse_path or "#" in sse_path or "//" in sse_path:
            raise ValueError("MCP SSE path is invalid")
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int) or not 1_024 <= max_body_bytes <= MAX_JSON_BYTES:
            raise ValueError("max_body_bytes is outside the MCP policy")
        self.server = server
        self.host = host
        self.port = port
        self.sse_path = sse_path
        self.max_body_bytes = max_body_bytes
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, _SSEClient] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()

    @property
    def address(self) -> str | None:
        if self._httpd is None:
            return None
        host = "[::1]" if self.host == "::1" else self.host
        return f"http://{host}:{self._httpd.server_address[1]}{self.sse_path}"

    def start(self) -> str:
        if self._httpd is not None:
            return self.address or ""
        transport = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "QLH-MCP/1"
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:
                return None

            @property
            def transport(self) -> "SSE_MCPTransport":
                return self.server.mcp_transport  # type: ignore[attr-defined]

            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                if parsed.path != self.transport.sse_path:
                    self._send_json(404, {"error": "not_found"})
                    return
                session_id = "mcp_" + uuid.uuid4().hex
                client = _SSEClient(self.wfile)
                with self.transport._lock:
                    self.transport._clients[session_id] = client
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    client.send("endpoint", f"/messages?sessionId={session_id}")
                    while not self.transport._closed.wait(0.5):
                        if client.closed:
                            break
                        try:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                finally:
                    client.close()
                    with self.transport._lock:
                        self.transport._clients.pop(session_id, None)

            def do_POST(self) -> None:
                parsed = urlsplit(self.path)
                if parsed.path != "/messages":
                    self._send_json(404, {"error": "not_found"})
                    return
                query = parse_qs(parsed.query)
                session_id = query.get("sessionId", [""])[0]
                with self.transport._lock:
                    client = self.transport._clients.get(session_id)
                if client is None:
                    self._send_json(404, {"error": "session_not_found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > self.transport.max_body_bytes:
                    self._send_json(413, {"error": "request_too_large"})
                    return
                payload = self.rfile.read(length)
                response = self.transport.server.handle_json(payload)
                if response is not None:
                    try:
                        client.send("message", response)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        client.close()
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._closed.clear()
        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self._httpd.mcp_transport = transport  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="qlh-mcp-sse", daemon=True)
        self._thread.start()
        return self.address or ""

    def serve_forever(self) -> None:
        if self._httpd is None:
            self.start()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        else:
            assert self._httpd is not None
            self._httpd.serve_forever()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "SSE_MCPTransport":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


SSEMCPTransport = SSE_MCPTransport


__all__ = ["SSEMCPTransport", "SSE_MCPTransport", "StdioMCPTransport"]
