"""cybergothic 桌面壳（packaging/launcher_cybergothic.py）测试。

避免真开 pywebview 窗口 / 真开系统浏览器；用 monkeypatch 与真实回环 socket 验证
静态 SPA、/api 反向代理、端口/路径解析和窗口分支。对外只连接回环 fake 后端。
"""

from __future__ import annotations

import logging
import socket
import sys
import tempfile
import threading
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "packaging"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launcher_cybergothic as cg  # noqa: E402


class _FakeApi(BaseHTTPRequestHandler):
    """回环 fake 后端：/api/ping 返回 JSON；/api/boom 返回 500。"""

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/ping"):
            body = b'{"ok":true,"v":1}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/boom"):
            self.send_response(500)
            body = b"boom"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):  # noqa: D102
        pass


@pytest.fixture()
def fake_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture()
def fake_webview(monkeypatch):
    calls: list[dict] = []

    class FakeWebviewModule:
        def create_window(self, title, url, **kwargs):
            calls.append({"title": title, "url": url, **kwargs})

        def start(self):
            calls.append({"started": True})

    fake = FakeWebviewModule()
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(cg.os, "name", "nt")
    return calls


# ---------- 路径/端口/解析 ----------

def test_resolve_dist_precedence(tmp_path):
    explicit = tmp_path / "e"
    explicit.mkdir(parents=True)
    (explicit / "index.html").write_text("x")
    assert cg.resolve_dist(str(explicit)) == explicit.resolve()


def test_resolve_dist_env_fallback(tmp_path, monkeypatch):
    env_path = tmp_path / "env"
    env_path.mkdir(parents=True)
    (env_path / "index.html").write_text("x")
    monkeypatch.setenv(cg.ENV_DIST, str(env_path))
    # 无显式参数时优先 env
    assert cg.resolve_dist(None) == env_path.resolve()
    # 显式参数优先于 env
    other = tmp_path / "other"
    other.mkdir(parents=True)
    (other / "index.html").write_text("x")
    assert cg.resolve_dist(str(other)) == other.resolve()


def test_resolve_dist_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cg.resolve_dist(str(tmp_path / "missing"))


def test_pick_port_int_and_busy():
    port = cg.pick_port(9841)
    assert isinstance(port, int) and 0 < port < 65536
    # 占用 preferred → 向前探测得到下一个可用
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied = blocker.getsockname()[1]
    got = cg.pick_port(occupied)
    assert got > occupied and got != occupied
    blocker.close()


def test_parse_api_target_defaults():
    assert cg.parse_api_target("http://x:99") == ("x", 99, "http")
    assert cg.parse_api_target("http://x") == ("x", 80, "http")
    assert cg.parse_api_target("https://x") == ("x", 443, "https")
    assert cg.parse_api_target("127.0.0.1:8000") == ("127.0.0.1", 8000, "http")
    with pytest.raises(ValueError):
        cg.parse_api_target("ftp://x")


# ---------- 真实回环服务：静态 / SPA / API 代理 ----------

@pytest.fixture()
def cg_server(tmp_path, fake_api):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>idx</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    server = cg.CgServer(("127.0.0.1", 0), dist, ("127.0.0.1", fake_api.server_address[1], "http"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def _get(server, path):
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    ctype = resp.getheader("Content-Type", "")
    status = resp.status
    conn.close()
    return status, ctype, body


def test_static_root_serves_index(cg_server):
    status, ctype, body = _get(cg_server, "/")
    assert status == 200
    assert "text/html" in ctype
    assert body == b"<html>idx</html>"


def test_spa_history_fallback(cg_server):
    status, _, body = _get(cg_server, "/routes/accounts/42")
    assert status == 200 and body == b"<html>idx</html>"


def test_asset_missing_returns_404(cg_server):
    # 带扩展名的缺失静态资源不返回 index.html（避免 SPA 吞 404）
    status, _, _ = _get(cg_server, "/assets/missing.png")
    assert status == 404


def test_asset_served_with_mime(cg_server):
    status, ctype, body = _get(cg_server, "/assets/app.js")
    assert status == 200 and "javascript" in ctype
    assert body == b"console.log(1)"


def test_path_traversal_rejected(cg_server):
    # 手工 raw socket 发未规范化路径（http.client 会自行规范化，绕过它）
    import socket as _socket
    sock = _socket.create_connection(("127.0.0.1", cg_server.server_address[1]), timeout=5)
    sock.sendall(b"GET /../qlh_launcher.py HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    data = b""
    while True:
        part = sock.recv(4096)
        if not part:
            break
        data += part
    sock.close()
    # http.server 对畸形路径也可能直接 400；两者均为拒绝，且不得泄漏静态内容
    assert data.startswith((b"HTTP/1.1 404", b"HTTP/1.1 400"))
    assert b"idx" not in data


def test_api_proxy_rejects_foreign_origin(cg_server):
    conn = HTTPConnection("127.0.0.1", cg_server.server_address[1], timeout=5)
    conn.request("GET", "/api/ping", headers={"Origin": "http://evil.example"})
    resp = conn.getresponse()
    assert resp.status == 403
    conn.close()
    # 本地 Origin 放行
    conn = HTTPConnection("127.0.0.1", cg_server.server_address[1], timeout=5)
    conn.request("GET", "/api/ping",
                 headers={"Origin": f"http://127.0.0.1:{cg_server.server_address[1]}"})
    resp = conn.getresponse()
    assert resp.status == 200
    conn.close()


def test_api_proxy_survives_broken_upstream(tmp_path, caplog):
    """上游回垃圾响应（BadStatusLine）→ 502 而非悬挂/500。"""
    class _Bad(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.wfile.write(b"HTTP/1.1 20")
            self.wfile.flush()
            self.connection.close()

        def log_message(self, *a):
            pass

    bad = ThreadingHTTPServer(("127.0.0.1", 0), _Bad)
    threading.Thread(target=bad.serve_forever, daemon=True).start()
    try:
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("x")
        server = cg.CgServer(("127.0.0.1", 0), dist, ("127.0.0.1", bad.server_address[1], "http"))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            conn.request("GET", "/api/ping")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            assert resp.status == 502
            assert body == b""
        finally:
            server.shutdown()
            server.server_close()
    finally:
        bad.shutdown()
        bad.server_close()


def test_api_proxied_to_backend(cg_server):
    status, ctype, body = _get(cg_server, "/api/ping")
    assert status == 200
    assert "application/json" in ctype
    assert body == b'{"ok":true,"v":1}'


def test_api_proxy_preserves_upstream_error(cg_server):
    status, _, body = _get(cg_server, "/api/boom")
    assert status == 500 and body == b"boom"


def test_non_api_path_not_proxied(cg_server):
    status, _, _ = _get(cg_server, "/api")
    # /api 精确路径不属于 /api/* 代理集，走静态 → 兜底 404（无 index 之外文件）
    assert status in (404,)


# ---------- 窗口分支（monkeypatch，不开真窗口） ----------

def test_run_window_prefers_pywebview_on_windows(monkeypatch, fake_webview):
    monkeypatch.setattr(cg.os, "name", "nt")
    mode = cg.run_window("http://127.0.0.1:9/")
    assert mode == "pywebview"
    assert fake_webview[0]["title"] == cg.APP_TITLE
    assert fake_webview[0]["url"].startswith("http://127.0.0.1:9")
    assert fake_webview[1]["started"] is True


def test_run_window_falls_back_to_browser(monkeypatch):
    monkeypatch.setattr(cg.os, "name", "posix")
    opened: list[str] = []

    class _FakeBrowser:
        def open(self, url):
            opened.append(url)
            return True

    monkeypatch.setitem(sys.modules, "webbrowser", _FakeBrowser())
    mode = cg.run_window("http://127.0.0.1:9/")
    assert mode == "browser"
    assert opened == ["http://127.0.0.1:9/"]


def test_main_browser_mode_keeps_service_alive(tmp_path, fake_api, monkeypatch, caplog):
    """browser fallback 分支必须进入 _wait_for_interrupt（服务保持存活），不得立即关。"""
    waited: list[bool] = []
    monkeypatch.setattr(cg, "_wait_for_interrupt", lambda t: waited.append(True))
    monkeypatch.setattr(cg, "_has_webview", lambda: False)  # 强制 browser fallback（不 mock os.name，避免 Path 类崩）

    class _FakeBrowser:
        def open(self, url):
            return True

    monkeypatch.setitem(sys.modules, "webbrowser", _FakeBrowser())
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>idx</html>", encoding="utf-8")
    code = cg.main([
        "--dist", str(dist),
        "--api-target", f"http://127.0.0.1:{fake_api.server_address[1]}",
    ])
    assert code == 0
    assert waited == [True]


def test_main_no_window_serves_and_quits(tmp_path, fake_api, caplog):
    caplog.set_level(logging.INFO, logger="qlh.cg_desktop")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>idx</html>", encoding="utf-8")
    code = cg.main([
        "--no-window", "--dist", str(dist),
        "--api-target", f"http://127.0.0.1:{fake_api.server_address[1]}",
    ])
    assert code == 0
    assert any("127.0.0.1" in record.message and "cybergothic" in record.message
               for record in caplog.records)


def test_main_errors_on_missing_dist(tmp_path, caplog):
    code = cg.main(["--no-window", "--dist", str(tmp_path / "nope")])
    assert code == 2
    assert any("dist" in record.message for record in caplog.records)
