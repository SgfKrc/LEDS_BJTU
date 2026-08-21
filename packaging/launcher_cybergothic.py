# -*- mode: python ; coding: utf-8 -*-
"""cybergothic 控制台桌面窗口（复用主应用同栈 pywebview / Edge WebView2）。

形态：本地 HTTP 服务 serve ``frontend_cybergothic/dist``（React 构建产物），
并对 ``/api/*`` 反向代理到 QLH 后端（默认 ``http://127.0.0.1:8000``，即 cybergothic
前端与主前端共用的同一个 FastAPI 后端；生产构建下 ``BASE='/api'`` 是相对同源，
必须由本壳代理才能访问真实 API）。随后用 pywebview 开原生窗口加载
``http://127.0.0.1:<port>/``；Windows 用 Edge WebView2，pywebview 不可用或
非 Windows 时回退到系统浏览器。

不 import 主应用 ``launcher.py``（避免拉起后端/推理重依赖）；纯 stdlib，唯一
可选第三方是运行时弹窗用的 ``pywebview``（与主应用同一份依赖，requirements 已含
``pywebview>=6.0``）。

CLI：
    python packaging/launcher_cybergothic.py [--port 9851] [--api-target http://127.0.0.1:8000] \
        [--dist <…/frontend_cybergothic/dist>] [--no-window]
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import socket
import sys
import threading
import urllib.parse
from http.client import (HTTPConnection, HTTPSConnection,
                         HTTPException as _HttpClientError,
                         responses as HTTP_STATUS_TEXT)  # noqa: F401
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger("qlh.cg_desktop")

APP_TITLE = "QLH · Cybergothic Console"
DEFAULT_PORT = 9851                      # 避开 5174(vite dev)/8000(api)/9090(launcher 更新)
DEFAULT_API_TARGET = "http://127.0.0.1:8000"
ENV_DIST = "QLH_CG_DIST"
ENV_API_TARGET = "QLH_API_TARGET"
WINDOW_W, WINDOW_H = 1440, 900

# 反向代理时需要剥掉的逐跳头（由本壳重算 Host/Content-Length）+ 可信客户端伪造头
_HOP_BY_HOP = {"host", "connection", "content-length", "keep-alive",
               "transfer-encoding", "te", "trailer", "upgrade", "proxy-connection",
               "proxy-authorization", "accept-encoding",
               "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
               "x-forwarded-port", "x-remote-addr", "x-real-ip"}


def _dist_ok(candidate: Path) -> bool:
    return candidate.is_dir() and (candidate / "index.html").is_file()


def resolve_dist(explicit: str | None) -> Path:
    """定位 cybergothic 构建产物。优先级：--dist > QLH_CG_DIST > 仓库内默认路径。

    显式给定 ``--dist``/``QLH_CG_DIST`` 但产物缺失时**直接报错**（不静默回退
    到默认路径，避免用户以为用的是他指定的目录）。
    """
    if explicit is not None:
        candidate = Path(explicit)
        if _dist_ok(candidate):
            return candidate
        raise FileNotFoundError(f"cybergothic dist 不存在: {candidate}（请先 cd frontend_cybergothic && npm run build）")
    env_dist = os.environ.get(ENV_DIST)
    if env_dist:
        candidate = Path(env_dist)
        if _dist_ok(candidate):
            return candidate
        raise FileNotFoundError(f"QLH_CG_DIST 无效: {candidate}")
    for candidate in (
        # 1) 开发形态：packaging/../frontend_cybergothic/dist（仓库根）
        Path(__file__).resolve().parent.parent / "frontend_cybergothic" / "dist",
        # 2) 打包形态：_internal/frontend_cybergothic/dist（datas 收集目标）
        Path(__file__).resolve().parent / "frontend_cybergothic" / "dist",
    ):
        if _dist_ok(candidate):
            return candidate
    raise FileNotFoundError(
        "未找到 cybergothic dist；请先构建前端（cd frontend_cybergothic && npm run build）"
    )


def pick_port(preferred: int) -> int:
    """优先使用 preferred；被占用则向后探测，探测不到再取系统空闲端口。"""
    start = preferred or DEFAULT_PORT
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_api_target(raw: str) -> tuple[str, int, str]:
    """把 ``http://host:port`` 解析为 (host, port, scheme)。"""
    url = urllib.parse.urlsplit(raw if "://" in raw else f"http://{raw}")
    scheme = url.scheme.lower() or "http"
    if scheme not in {"http", "https"}:
        raise ValueError(f"api-target 仅支持 http/https: {raw}")
    host = url.hostname or "127.0.0.1"
    port = url.port or (443 if scheme == "https" else 80)
    return host, port, scheme


def _has_webview() -> bool:
    """pywebview 是否可导入（仅 Windows 原生窗口；Linux 走系统浏览器）。"""
    if os.name != "nt":
        return False
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False


class CgRequestHandler(BaseHTTPRequestHandler):
    """静态 SPA 服务 + ``/api/*`` 反向代理。只允许 localhost 回环访问。"""

    server_version = "QLHCgDesktop/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def _dist(self) -> Path | None:  # type: ignore[override]
        return getattr(self.server, "dist", None)  # type: ignore[attr-defined]

    @property
    def _api_target(self) -> tuple[str, int, str] | None:  # type: ignore[override]
        return getattr(self.server, "api_target", None)  # type: ignore[attr-defined]

    # ---- 访问边界：仅 127.0.0.1 ----
    def check_loopback(self) -> bool:
        addr = self.client_address[0] if self.client_address else ""
        return addr in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}

    def _deny(self, status: int = 403) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ---- 核心分发 ----
    def do_GET(self) -> None:  # noqa: N802
        if not self.check_loopback():
            self._deny(403)
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path.startswith("/api/") or parsed.path == "/api":
            self._proxy_api(parsed)
            return
        self._serve_static(parsed.path)

    def do_HEAD(self) -> None:  # noqa: N802
        if not self.check_loopback():
            self._deny(403)
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path.startswith("/api/"):
            self._proxy_api(parsed, head=True)
            return
        # HEAD 与 GET 同源，仅不发 body
        original = self.command
        self.command = "GET"
        try:
            self._serve_static(parsed.path, head_only=True)
        finally:
            self.command = original

    # ---- 静态 SPA ----
    def _serve_static(self, path: str, head_only: bool = False) -> None:
        dist = self._dist
        if dist is None:
            self._deny(500)
            return
        dist_root = dist.resolve()
        rel = urllib.parse.unquote(path).lstrip("/")
        target = (dist_root / rel).resolve() if rel else dist_root
        try:
            target.relative_to(dist_root)
        except ValueError:
            self._deny(404)
            return
        if target.is_dir():
            target = target / "index.html"
        # SPA history 回退：无扩展名的导航路由 → index.html；带扩展名的缺失资源 → 404
        if not target.is_file():
            if rel and Path(rel).suffix:
                self._deny(404)
                return
            target = dist_root / "index.html"
        if not target.is_file():
            self._deny(404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if str(target).lower().endswith(".html"):
            ctype = "text/html; charset=utf-8"
        if ctype.startswith("text/") or ctype in {"application/json",
                                                  "application/javascript",
                                                  "application/xml"}:
            ctype = f"{ctype}; charset=utf-8" if "; charset" not in ctype else ctype
        try:
            size = target.stat().st_size
            with target.open("rb") as fh:
                payload = fh.read() if not head_only else b""
        except OSError:
            self._deny(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    # ---- /api 反向代理 ----
    def _check_origin(self) -> bool:
        """防 DNS rebinding/CSRF：非本地来源的 Origin 一律拒绝（无 Origin 放行）。"""
        origin = self.headers.get("Origin", "")
        if not origin:
            return True
        port = self.server.server_address[1] if self.server else None  # type: ignore[attr-defined]
        allowed = {
            f"http://127.0.0.1:{port}", f"http://localhost:{port}",
            "http://127.0.0.1", "http://localhost",
        }
        if origin not in allowed:
            logger.warning("拒绝非本地 Origin: %s", origin)
            return False
        return True

    def _proxy_api(self, parsed: urllib.parse.SplitResult, head: bool = False) -> None:
        if not self._check_origin():
            self._deny(403)
            return
        target = self._api_target
        if target is None:
            self._deny(500)
            return
        host, port, scheme = target
        assert scheme in {"http", "https"}
        qs = f"?{parsed.query}" if parsed.query else ""
        path = f"{parsed.path}{qs}"
        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() not in _HOP_BY_HOP:
                headers[key] = value
        headers["Host"] = f"{host}:{port}" if port not in {80, 443} else host
        body = b""
        if self.headers.get("Content-Length"):
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
            except (ValueError, OSError):
                self._deny(400)
                return
        conn_cls: type[Any] = HTTPSConnection if scheme == "https" else HTTPConnection
        conn = conn_cls(host, port, timeout=30)
        method = "HEAD" if head else "GET"
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, _HttpClientError) as exc:
            conn.close()
            logger.warning("api 反向代理失败 %s: %s", path, exc)
            self._deny(502)
            return
        # 流式判定：上游无固定长度（chunked / SSE）→ close-delimited 边读边写，
        # 避免把 SSE/大响应全量缓冲
        streaming = (resp.getheader("Transfer-Encoding", "").lower() == "chunked"
                     or resp.getheader("Content-Type", "").startswith("text/event-stream"))
        self.send_response(resp.status)
        for key, value in resp.getheaders():
            lk = key.lower()
            if lk in _HOP_BY_HOP and lk != "content-length":
                continue
            if head and lk == "content-length":
                continue
            if streaming and lk == "content-length":
                continue  # 流式不报长度，用 Connection: close 定界
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            if head or not streaming:
                payload = b"" if head else resp.read()
                if payload:
                    self.wfile.write(payload)
            else:
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, OSError):
            pass  # 客户端已断开
        finally:
            conn.close()

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("  " + fmt, *args)


class CgServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], dist: Path, api_target: tuple[str, int, str]):
        self.dist = dist
        self.api_target = api_target
        super().__init__(addr, CgRequestHandler)


def run_window(url: str) -> str:
    """开 pywebview 原生窗口（Windows/可导入）或系统浏览器；返回实际窗口形态描述。"""
    if _has_webview():
        import webview  # type: ignore
        webview.create_window(
            APP_TITLE, url,
            width=WINDOW_W, height=WINDOW_H,
            background_color="#0c0b0b",
            min_size=(1024, 700),
        )
        webview.start()
        return "pywebview"
    import webbrowser
    webbrowser.open(url)
    return "browser"


def _wait_for_interrupt(server_thread: threading.Thread) -> None:
    """browser fallback 下保持本地服务存活，直到用户 Ctrl+C。"""
    try:
        server_thread.join()
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qlh-cg-desktop", description=APP_TITLE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"本地服务端口（默认 {DEFAULT_PORT}）")
    parser.add_argument("--api-target", default=os.environ.get(ENV_API_TARGET, DEFAULT_API_TARGET),
                        help=f"后端 API 地址（默认 {DEFAULT_API_TARGET}，/api/* 反向代理到此处）")
    parser.add_argument("--dist", default=None, help="cybergothic dist 目录（默认仓库内 frontend_cybergothic/dist）")
    parser.add_argument("--no-window", action="store_true", help="不开窗口，仅打印访问地址（供调试/无人环境）")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="[%(levelname)s] %(message)s")
    try:
        dist = resolve_dist(args.dist)
        api_target = parse_api_target(args.api_target)
        port = pick_port(args.port)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    server = CgServer(("127.0.0.1", port), dist, api_target)
    thread = threading.Thread(target=server.serve_forever, name="cg-static", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/"
    logger.info("cybergothic console: %s（dist=%s，/api → %s:%s）",
                url, dist, api_target[0], api_target[1])
    try:
        if args.no_window:
            logger.info("窗口已禁用（--no-window），请手动访问 %s", url)
            mode = "none"
        else:
            mode = run_window(url)
            logger.info("窗口形态: %s", mode)
            if mode == "browser":
                # webbrowser.open 立即返回；保持本地服务存活直到 Ctrl+C
                logger.info("%s 已交由系统浏览器打开；本服务保持运行，Ctrl+C 结束", url)
                _wait_for_interrupt(thread)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
