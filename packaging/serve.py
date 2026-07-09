"""
极简 HTTP 静态文件服务器 — 分发安装包
=====================================
用法: python serve.py [port]

默认监听 0.0.0.0:9090，生成 Tailscale 可达的下载链接。
其他设备浏览器直接访问 http://<tailscale-ip>:9090/ 即可下载。

支持分发:
- PC 安装包: packaging/dist/*.exe
- Android 安装包: android/app/build/outputs/**/*.apk / *.aab
- 模型压缩包: models.7z

Ctrl+C 停止。
"""

import http.server
import html
import io
import os
import socket
import sys
from functools import partial
from urllib.parse import quote, unquote, urlparse

HOST = "0.0.0.0"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
ROOT = os.path.dirname(os.path.abspath(__file__))  # packaging/
DIST_DIR = os.path.join(ROOT, "dist")
PROJECT_ROOT = os.path.dirname(ROOT)
MODEL_ARCHIVE = os.path.join(PROJECT_ROOT, "models.7z")
ANDROID_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "android", "app", "build", "outputs")

ANDROID_EXTS = (".apk", ".aab")


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


def _format_size(path: str) -> str:
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


def _scan_android_packages() -> list[tuple[str, str]]:
    """
    扫描 Android Gradle 输出目录，返回 [(display_name, absolute_path)]。

    display_name 使用相对 outputs/ 的路径，避免 debug/release 同名文件冲突，
    例如: apk/debug/app-debug.apk。
    """
    packages: list[tuple[str, str]] = []
    if not os.path.isdir(ANDROID_OUTPUT_DIR):
        return packages

    for root, _dirs, files in os.walk(ANDROID_OUTPUT_DIR):
        for name in files:
            if not name.lower().endswith(ANDROID_EXTS):
                continue
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, ANDROID_OUTPUT_DIR).replace(os.sep, "/")
            packages.append((rel_path, abs_path))

    # 常用 debug/release apk 优先，其他按名称排序
    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        rel, _path = item
        lower = rel.lower()
        if lower.endswith("app-debug.apk"):
            rank = 0
        elif lower.endswith("app-release.apk"):
            rank = 1
        elif lower.endswith("app-release.aab"):
            rank = 2
        else:
            rank = 9
        return (rank, lower)

    return sorted(packages, key=sort_key)


def _android_url(rel_path: str) -> str:
    """Android 包下载 URL。"""
    return "/android/" + quote(rel_path, safe="/")


def _resolve_android_path(request_path: str) -> str | None:
    """将 /android/<rel> 映射到 Gradle 输出目录内的 apk/aab 文件。"""
    prefix = "/android/"
    if not request_path.startswith(prefix):
        return None

    rel = unquote(request_path[len(prefix):]).replace("\\", "/").lstrip("/")
    if not rel or rel.endswith("/"):
        return None
    if not rel.lower().endswith(ANDROID_EXTS):
        return None

    # 防路径穿越: 归一化后必须仍在 ANDROID_OUTPUT_DIR 内
    candidate = os.path.abspath(os.path.join(ANDROID_OUTPUT_DIR, rel))
    output_root = os.path.abspath(ANDROID_OUTPUT_DIR)
    if candidate != output_root and not candidate.startswith(output_root + os.sep):
        return None
    if not os.path.isfile(candidate):
        return None
    return candidate


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """仅记录下载，不刷屏。"""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".apk": "application/vnd.android.package-archive",
        ".aab": "application/octet-stream",
        ".7z": "application/x-7z-compressed",
    }

    def translate_path(self, path):
        request_path = unquote(urlparse(path).path)
        if request_path == "/models.7z":
            return MODEL_ARCHIVE

        android_path = _resolve_android_path(request_path)
        if android_path:
            return android_path

        return super().translate_path(path)

    def list_directory(self, path):
        if os.path.abspath(path) != os.path.abspath(DIST_DIR):
            return super().list_directory(path)

        pc_entries = []
        if os.path.isdir(DIST_DIR):
            for name in sorted(os.listdir(DIST_DIR), key=str.lower):
                item_path = os.path.join(DIST_DIR, name)
                display_name = name + "/" if os.path.isdir(item_path) else name
                href = quote(display_name)
                pc_entries.append(
                    (display_name, href, _format_size(item_path) if os.path.isfile(item_path) else "目录")
                )

        android_entries = []
        for rel_path, abs_path in _scan_android_packages():
            android_entries.append((rel_path, _android_url(rel_path), _format_size(abs_path)))

        model_entries = []
        if os.path.isfile(MODEL_ARCHIVE):
            model_entries.append(("models.7z", "models.7z", _format_size(MODEL_ARCHIVE)))

        def render_rows(entries: list[tuple[str, str, str]], empty_text: str) -> str:
            if not entries:
                return f"<li>{html.escape(empty_text)}</li>"
            return "\n".join(
                f'<li><a href="{href}">{html.escape(name)}</a> <span>{html.escape(size)}</span></li>'
                for name, href, size in entries
            )

        pc_rows = render_rows(pc_entries, "暂无 PC 安装包（请先运行 build-installer.bat）")
        android_rows = render_rows(android_entries, "暂无 Android 安装包（请先运行 android/gradlew.bat assembleDebug）")
        model_rows = render_rows(model_entries, "暂无模型压缩包 models.7z")

        body = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QLH 文件分发</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.55; }}
    h1 {{ font-size: 24px; margin-bottom: 24px; }}
    h2 {{ font-size: 18px; margin: 22px 0 8px; }}
    ul {{ line-height: 1.9; padding-left: 20px; margin-top: 6px; }}
    span {{ color: #666; margin-left: 12px; }}
    .hint {{ color: #666; font-size: 14px; margin-top: 24px; }}
    code {{ background: #f3f3f3; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>QLH 文件分发</h1>

  <h2>Windows PC 安装包</h2>
  <ul>
    {pc_rows}
  </ul>

  <h2>Android 安装包</h2>
  <ul>
    {android_rows}
  </ul>

  <h2>模型压缩包</h2>
  <ul>
    {model_rows}
  </ul>

  <p class="hint">
    Android Debug APK 默认路径: <code>android/app/build/outputs/apk/debug/app-debug.apk</code><br>
    PC 安装包默认路径: <code>packaging/dist/QLH-Edge-Inference-Setup-v*.exe</code>
  </p>
</body>
</html>
"""
        encoded = body.encode("utf-8", "surrogateescape")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return io.BytesIO(encoded)

    def log_message(self, fmt, *args):
        if "200" in str(args[0]) or "206" in str(args[0]):
            print(f"  ✓ {args[0]}  {args[1]}")
        elif "404" in str(args[0]):
            print(f"  ✗ 404 {args[1]}")
        # 304 不打印


ts_ip = _detect_tailscale_ip()
android_packages = _scan_android_packages()

print()
print("=" * 55)
print("  📦 QLH 文件分发服务")
print("=" * 55)
print()
print(f"  本机 Tailscale IP: {ts_ip}")
print(f"  监听: http://{HOST}:{PORT}")
print(f"  PC 安装包目录: {DIST_DIR}")
print(f"  Android 输出目录: {ANDROID_OUTPUT_DIR}")
if android_packages:
    print("  Android 安装包:")
    for rel_path, abs_path in android_packages:
        print(f"    /android/{rel_path} ({_format_size(abs_path)})")
else:
    print("  Android 安装包: 未找到（请先运行 android/gradlew.bat assembleDebug）")
if os.path.isfile(MODEL_ARCHIVE):
    print(f"  模型压缩包: {MODEL_ARCHIVE} ({_format_size(MODEL_ARCHIVE)})")
else:
    print(f"  模型压缩包: 未找到 {MODEL_ARCHIVE}")
print()
print("  其他设备浏览器访问:")
if ts_ip and ts_ip != "?":
    print(f"    http://{ts_ip}:{PORT}/")
    for rel_path, _abs_path in android_packages:
        print(f"    http://{ts_ip}:{PORT}{_android_url(rel_path)}")
    if os.path.isfile(MODEL_ARCHIVE):
        print(f"    http://{ts_ip}:{PORT}/models.7z")
else:
    print(f"    http://<本机IP>:{PORT}/")
    for rel_path, _abs_path in android_packages:
        print(f"    http://<本机IP>:{PORT}{_android_url(rel_path)}")
    if os.path.isfile(MODEL_ARCHIVE):
        print(f"    http://<本机IP>:{PORT}/models.7z")
print()
print("  按 Ctrl+C 停止服务")
print("─" * 55)
print()

server = http.server.HTTPServer(
    (HOST, PORT),
    partial(QuietHTTPRequestHandler, directory=DIST_DIR),
)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print()
    print("  服务已停止。")
    server.server_close()
