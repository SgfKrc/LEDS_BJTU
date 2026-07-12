"""
极简 HTTP 静态文件服务器 — 分发安装包
=====================================
用法: python serve.py [port]

默认监听 0.0.0.0:9090，生成 Tailscale 可达的下载链接。
其他设备浏览器直接访问 http://<tailscale-ip>:9090/ 即可下载。

支持分发:
- PC 安装包: packaging/dist/*.exe
- Android 安装包: packaging/dist/*.apk / *.aab，或 android/app/build/outputs/**/*.apk / *.aab
- PC 模型压缩包: models_pc.7z 或 models_pc/*.7z
- Android 模型压缩包: models_android.7z 或 models_android/*.7z

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
DEFAULT_PORT = 9090
ROOT = os.path.dirname(os.path.abspath(__file__))  # packaging/
DIST_DIR = os.path.join(ROOT, "dist")
PROJECT_ROOT = os.path.dirname(ROOT)
MODEL_ARCHIVES = {
    "pc": {
        "title": "PC 模型压缩包",
        "root_file": "models_pc.7z",
        "dir": "models_pc",
        "url_prefix": "/models-pc/",
    },
    "android": {
        "title": "Android 模型压缩包",
        "root_file": "models_android.7z",
        "dir": "models_android",
        "url_prefix": "/models-android/",
    },
}
ANDROID_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "android", "app", "build", "outputs")

PC_INSTALLER_EXTS = (".exe",)
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
    扫描 Android Gradle release 输出目录，返回 [(display_name, absolute_path)]。

    display_name 使用相对 outputs/ 的路径，避免 debug/release 同名文件冲突，
    例如: apk/full/release/app-full-release.apk。
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
            rel_parts = rel_path.lower().split("/")
            if "androidtest" in rel_parts or "release" not in rel_parts:
                continue
            packages.append((rel_path, abs_path))

    # 常用 release apk 优先，其他按名称排序
    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        rel, _path = item
        lower = rel.lower()
        if lower.endswith("app-release.apk"):
            rank = 0
        elif lower.endswith("app-release.aab"):
            rank = 1
        else:
            rank = 9
        return (rank, lower)

    return sorted(packages, key=sort_key)


def _android_url(rel_path: str) -> str:
    """Android 包下载 URL。"""
    return "/android/" + quote(rel_path, safe="/")


def _scan_pc_installers() -> list[tuple[str, str, str]]:
    """扫描 packaging/dist 内可分发的 PC 安装包。"""
    installers: list[tuple[str, str, str]] = []
    if not os.path.isdir(DIST_DIR):
        return installers

    for name in sorted(os.listdir(DIST_DIR), key=str.lower):
        item_path = os.path.join(DIST_DIR, name)
        if not os.path.isfile(item_path) or not name.lower().endswith(PC_INSTALLER_EXTS):
            continue
        installers.append((name, "/" + quote(name), item_path))
    return installers


def _scan_dist_android_packages() -> list[tuple[str, str, str]]:
    """扫描 packaging/dist 内可直接分发的 Android 安装包。"""
    packages: list[tuple[str, str, str]] = []
    if not os.path.isdir(DIST_DIR):
        return packages

    for name in sorted(os.listdir(DIST_DIR), key=str.lower):
        item_path = os.path.join(DIST_DIR, name)
        if not os.path.isfile(item_path) or not name.lower().endswith(ANDROID_EXTS):
            continue
        packages.append((name, "/" + quote(name), item_path))
    return packages


def _scan_android_downloads() -> list[tuple[str, str, str]]:
    """
    扫描所有 Android 安装包下载项，返回 [(display_name, href, absolute_path)]。

    packaging/dist 是对外分发目录，优先显示；Gradle 输出目录保留为构建产物备用入口。
    """
    entries = _scan_dist_android_packages()
    seen_paths = {os.path.abspath(abs_path) for _display, _href, abs_path in entries}

    for rel_path, abs_path in _scan_android_packages():
        normalized = os.path.abspath(abs_path)
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        entries.append((f"android/app/build/outputs/{rel_path}", _android_url(rel_path), abs_path))

    return entries


def _scan_model_archives(kind: str | None = None) -> list[tuple[str, str, str, str]]:
    """
    扫描模型压缩包，返回 [(kind, display_name, href, absolute_path)]。

    支持两种约定:
    - 根目录固定文件: models_pc.7z / models_android.7z
    - 分类目录文件: models_pc/*.7z / models_android/*.7z
    """
    entries: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    items = MODEL_ARCHIVES.items() if kind is None else [(kind, MODEL_ARCHIVES[kind])]

    for archive_kind, config in items:
        root_file = os.path.join(PROJECT_ROOT, config["root_file"])
        if os.path.isfile(root_file):
            abs_path = os.path.abspath(root_file)
            seen.add(abs_path)
            entries.append((archive_kind, config["root_file"], "/" + quote(config["root_file"]), root_file))

        archive_dir = os.path.join(PROJECT_ROOT, config["dir"])
        if not os.path.isdir(archive_dir):
            continue

        for root, _dirs, files in os.walk(archive_dir):
            for name in files:
                if not name.lower().endswith(".7z"):
                    continue
                abs_path = os.path.abspath(os.path.join(root, name))
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                rel_path = os.path.relpath(abs_path, archive_dir).replace(os.sep, "/")
                display_name = f'{config["dir"]}/{rel_path}'
                href = config["url_prefix"] + quote(rel_path, safe="/")
                entries.append((archive_kind, display_name, href, abs_path))

    return sorted(entries, key=lambda item: (item[0], item[1].lower()))


def _resolve_model_archive_path(request_path: str) -> str | None:
    """将模型压缩包下载 URL 映射到项目内 .7z 文件。"""
    for archive_kind, config in MODEL_ARCHIVES.items():
        root_url = "/" + config["root_file"]
        if request_path == root_url:
            candidate = os.path.abspath(os.path.join(PROJECT_ROOT, config["root_file"]))
            return candidate if os.path.isfile(candidate) else None

        prefix = config["url_prefix"]
        if not request_path.startswith(prefix):
            continue

        rel = unquote(request_path[len(prefix):]).replace("\\", "/").lstrip("/")
        if not rel or rel.endswith("/") or not rel.lower().endswith(".7z"):
            return None

        archive_dir = os.path.abspath(os.path.join(PROJECT_ROOT, config["dir"]))
        candidate = os.path.abspath(os.path.join(archive_dir, rel))
        if candidate != archive_dir and not candidate.startswith(archive_dir + os.sep):
            return None
        return candidate if os.path.isfile(candidate) else None

    return None


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
        model_archive_path = _resolve_model_archive_path(request_path)
        if model_archive_path:
            return model_archive_path

        android_path = _resolve_android_path(request_path)
        if android_path:
            return android_path

        return super().translate_path(path)

    def list_directory(self, path):
        if os.path.abspath(path) != os.path.abspath(DIST_DIR):
            return super().list_directory(path)

        pc_entries = [
            (display, href, _format_size(abs_path))
            for display, href, abs_path in _scan_pc_installers()
        ]

        android_entries = [
            (display, href, _format_size(abs_path))
            for display, href, abs_path in _scan_android_downloads()
        ]

        pc_model_entries = [
            (display, href, _format_size(abs_path))
            for kind, display, href, abs_path in _scan_model_archives("pc")
        ]
        android_model_entries = [
            (display, href, _format_size(abs_path))
            for kind, display, href, abs_path in _scan_model_archives("android")
        ]

        def render_rows(entries: list[tuple[str, str, str]], empty_text: str) -> str:
            if not entries:
                return f"<li>{html.escape(empty_text)}</li>"
            return "\n".join(
                f'<li><a href="{href}">{html.escape(name)}</a> <span>{html.escape(size)}</span></li>'
                for name, href, size in entries
            )

        pc_rows = render_rows(pc_entries, "暂无 PC 安装包（请先运行 build-installer.bat）")
        android_rows = render_rows(android_entries, "暂无 Android 安装包（请先运行 android/gradlew.bat assembleRelease）")
        pc_model_rows = render_rows(pc_model_entries, "暂无 PC 模型压缩包 models_pc.7z / models_pc/*.7z")
        android_model_rows = render_rows(
            android_model_entries,
            "暂无 Android 模型压缩包 models_android.7z / models_android/*.7z",
        )

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

  <h2>PC 模型压缩包</h2>
  <ul>
    {pc_model_rows}
  </ul>

  <h2>Android 模型压缩包</h2>
  <ul>
    {android_model_rows}
  </ul>

  <p class="hint">
    Android Release APK 默认路径: <code>android/app/build/outputs/apk/*/release/*.apk</code><br>
    PC 安装包默认路径: <code>packaging/dist/QLH-Edge-Inference-Setup-v*.exe</code><br>
    Android 模型包仅需包含 GGUF 模型；PC 模型包可包含 PC 端需要的完整模型目录。
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


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else DEFAULT_PORT

    ts_ip = _detect_tailscale_ip()
    android_packages = _scan_android_downloads()
    model_archives = _scan_model_archives()

    print()
    print("=" * 55)
    print("  📦 QLH 文件分发服务")
    print("=" * 55)
    print()
    print(f"  本机 Tailscale IP: {ts_ip}")
    print(f"  监听: http://{HOST}:{port}")
    print(f"  PC 安装包目录: {DIST_DIR}")
    print(f"  Android 输出目录: {ANDROID_OUTPUT_DIR}")
    if android_packages:
        print("  Android 安装包:")
        for display, href, abs_path in android_packages:
            print(f"    {display} -> {href} ({_format_size(abs_path)})")
    else:
        print("  Android 安装包: 未找到（请先运行 android/gradlew.bat assembleRelease）")
    if model_archives:
        print("  模型压缩包:")
        for kind, display, href, abs_path in model_archives:
            title = MODEL_ARCHIVES[kind]["title"]
            print(f"    {title}: {display} -> {href} ({_format_size(abs_path)})")
    else:
        print("  模型压缩包: 未找到 models_pc.7z / models_android.7z 或分类目录 .7z")
    print()
    print("  其他设备浏览器访问:")
    if ts_ip and ts_ip != "?":
        print(f"    http://{ts_ip}:{port}/")
        for _display, href, _abs_path in android_packages:
            print(f"    http://{ts_ip}:{port}{href}")
        for _kind, _display, href, _abs_path in model_archives:
            print(f"    http://{ts_ip}:{port}{href}")
    else:
        print(f"    http://<本机IP>:{port}/")
        for _display, href, _abs_path in android_packages:
            print(f"    http://<本机IP>:{port}{href}")
        for _kind, _display, href, _abs_path in model_archives:
            print(f"    http://<本机IP>:{port}{href}")
    print()
    print("  按 Ctrl+C 停止服务")
    print("─" * 55)
    print()

    server = http.server.HTTPServer(
        (HOST, port),
        partial(QuietHTTPRequestHandler, directory=DIST_DIR),
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("  服务已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
