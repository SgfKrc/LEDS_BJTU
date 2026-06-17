"""
极简 HTTP 静态文件服务器 — 分发安装包
=====================================
用法: python serve.py [port]

默认监听 0.0.0.0:9090，生成 Tailscale 可达的下载链接。
其他设备浏览器直接访问 http://<tailscale-ip>:9090/ 即可下载。

Ctrl+C 停止。
"""

import http.server
import os
import socket
import sys
from functools import partial

HOST = "0.0.0.0"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
ROOT = os.path.dirname(os.path.abspath(__file__))  # packaging/

os.chdir(os.path.join(ROOT, "dist"))


def _detect_tailscale_ip() -> str:
    """检测本机 Tailscale IP，方便拼接下载 URL。"""
    try:
        import psutil
        addrs = psutil.net_if_addrs()
        for iface, addr_list in addrs.items():
            if "tailscale" in iface.lower():
                for addr in addr_list:
                    if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                        return addr.address
    except Exception:
        pass
    return "?"


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """仅记录下载，不刷屏。"""

    def log_message(self, fmt, *args):
        if "200" in str(args[0]) or "206" in str(args[0]):
            print(f"  ✓ {args[0]}  {args[1]}")
        elif "404" in str(args[0]):
            print(f"  ✗ 404 {args[1]}")
        # 304 不打印


ts_ip = _detect_tailscale_ip()

print()
print("=" * 55)
print("  📦 QLH 安装包分发服务")
print("=" * 55)
print()
print(f"  本机 Tailscale IP: {ts_ip}")
print(f"  监听: http://{HOST}:{PORT}")
print()
print("  其他设备浏览器访问:")
if ts_ip and ts_ip != "?":
    print(f"    http://{ts_ip}:{PORT}/")
else:
    print(f"    http://<本机IP>:{PORT}/")
print()
print("  按 Ctrl+C 停止服务")
print("─" * 55)
print()

server = http.server.HTTPServer(
    (HOST, PORT),
    partial(QuietHTTPRequestHandler, directory=os.getcwd()),
)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print()
    print("  服务已停止。")
    server.server_close()
