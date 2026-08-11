"""
极简 HTTP 静态文件服务器 — 分发安装包
=====================================
用法: python serve.py [port]

默认监听 [::]:9090（同时接受 IPv4/IPv6），生成 Tailscale 可达的下载链接。
其他设备浏览器直接访问 http://<tailscale-ip>:9090/ 即可下载。

支持分发:
- PC 安装包: packaging/dist/*.exe（主应用，不含启动器）
- QLH 启动器: packaging/dist/QLH-Launcher-Setup-v*.exe（安装包）+ QLH-Launcher-v*.zip（自更新资产）
- Android 安装包: packaging/dist/*.apk / *.aab，或 android/app/build/outputs/**/*.apk / *.aab
- PC 模型压缩包: models_pc.7z 或 models_pc/*.7z
- Android 模型压缩包: models_android.7z 或 models_android/*.7z

Ctrl+C 停止。
"""

import http.server
import hashlib
import html
import io
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlparse

HOST = os.environ.get("QLH_DISTRIBUTION_HOST", "::")
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
UPDATE_MANIFEST_PATH = "/latest.json"
_SIGNER = None  # set by main() when QLH_SIGNING_KEY is configured
_VERSION_RE = re.compile(r"(?<!\d)v?(\d+\.\d+\.\d+(?:\.\d+)?)(?!\d)", re.IGNORECASE)
_REPAIR_INDEX_RE = re.compile(
    r"^qlh-edge-inference-repair-v\d+\.\d+\.\d+(?:\.\d+)?-"
    r"(?P<platform>windows|linux)-(?P<variant>cpu|cuda)\.json$",
    re.IGNORECASE,
)
_SHA256_CACHE: dict[str, tuple[int, int, str]] = {}


class Signer:
    """Signs update manifests with a release private key (UP-N2).

    The private key is never bundled; it lives only on the publisher's
    machine (or a signing host) and is selected via QLH_SIGNING_KEY.
    """

    def __init__(self, private_key_path: str):
        from signing import sign_manifest

        self._sign_manifest = sign_manifest
        self._private_key_path = private_key_path
        self._key_id = Path(private_key_path).stem

    def sign(self, manifest: dict) -> dict:
        return self._sign_manifest(
            manifest,
            private_key_path=self._private_key_path,
            key_id=self._key_id,
        )


def _project_version() -> str:
    override = os.environ.get("QLH_RELEASE_TAG", "").strip().lstrip("vV")
    if override:
        return override
    init_path = os.path.join(PROJECT_ROOT, "src", "__init__.py")
    try:
        with open(init_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return "0.0.0"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def _sha256_cached(path: str) -> str:
    stat = os.stat(path)
    key = os.path.abspath(path)
    cached = _SHA256_CACHE.get(key)
    identity = (stat.st_mtime_ns, stat.st_size)
    if cached and cached[:2] == identity:
        return cached[2]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    _SHA256_CACHE[key] = (identity[0], identity[1], result)
    return result


def _classify_update_asset(name: str) -> tuple[str, str, str, str] | None:
    lower = name.lower()
    repair_index = _REPAIR_INDEX_RE.fullmatch(lower)
    if repair_index:
        return (
            repair_index.group("platform"), repair_index.group("variant"),
            "x86_64", "repair-index",
        )
    if lower.endswith(".zip") and "qlh-launcher" in lower:
        return "windows", "any", "x86_64", "launcher"
    if lower.endswith(".zip") and "qlh-sd15-assets-" in lower:
        return "any", "any", "any", "sd15-asset"
    if lower.endswith(".exe") and "qlh-launcher-setup" in lower:
        return "windows", "any", "x86_64", "launcher-setup"
    if lower.endswith(".exe") and "qlh-edge-inference-setup" in lower:
        return "windows", "cuda" if "cuda" in lower else "cpu", "x86_64", "installer"
    if lower.endswith((".apk", ".aab")) and "qlh-inference" in lower:
        if "lite" in lower:
            return "android", "lite", "any", "installer"
        if "full" in lower:
            return "android", "full", "any", "installer"
    if lower.endswith(".deb") and "qlh-edge-inference" in lower:
        return "linux", "cuda" if "cuda" in lower else "cpu", "x86_64", "installer"
    return None


def build_update_manifest(
    dist_dir: str = DIST_DIR, *, signer: "Signer | None" = None,
) -> dict:
    """Build a deterministic v1 update manifest from current release assets.

    When ``signer`` is provided the manifest is signed in place with
    key_id/signed_at/signature (UP-N2 trusted publishing).
    """
    tag = _project_version()
    assets: list[dict] = []
    if os.path.isdir(dist_dir):
        for name in sorted(os.listdir(dist_dir), key=str.lower):
            path = os.path.join(dist_dir, name)
            classification = _classify_update_asset(name)
            if not os.path.isfile(path) or classification is None:
                continue
            version_match = _VERSION_RE.search(name)
            if version_match and version_match.group(1) != tag:
                continue
            target_platform, variant, arch, kind = classification
            assets.append({
                "name": name,
                "url": "/" + quote(name),
                "size": os.path.getsize(path),
                "sha256": _sha256_cached(path),
                "platform": target_platform,
                "variant": variant,
                "arch": arch,
                "kind": kind,
            })
    manifest = {
        "schema_version": 1,
        "tag": tag,
        "channel": os.environ.get("QLH_RELEASE_CHANNEL", "stable"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    if signer is not None:
        manifest = signer.sign(manifest)
    return manifest


def _detect_tailscale_ip() -> str:
    """检测本机 Tailscale IP，方便拼接下载 URL。"""
    try:
        import psutil
        addrs = psutil.net_if_addrs()
        candidates = []
        for iface, addr_list in addrs.items():
            if "tailscale" in iface.lower():
                for addr in addr_list:
                    if addr.family not in (socket.AF_INET, socket.AF_INET6):
                        continue
                    address = str(addr.address or "").split("%", 1)[0]
                    if not address or address.startswith("127.") or address == "::1":
                        continue
                    candidates.append((0 if addr.family == socket.AF_INET else 1, address))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
    except Exception:
        pass
    return "?"


def _url_host(host: str) -> str:
    value = str(host or "").strip()
    if value.startswith("[") and value.endswith("]"):
        return value
    return f"[{value.replace('%', '%25')}]" if ":" in value else value


class DualStackHTTPServer(http.server.ThreadingHTTPServer):
    """One IPv6 socket accepting native IPv6 and IPv4-mapped clients."""

    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def create_distribution_server(host: str, port: int, handler):
    value = str(host or "").strip()
    if value in {"", "::", "0.0.0.0"}:
        try:
            return DualStackHTTPServer(("::", port), handler)
        except OSError:
            return http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    if ":" in value:
        class IPv6HTTPServer(http.server.ThreadingHTTPServer):
            address_family = socket.AF_INET6

        return IPv6HTTPServer((value.strip("[]"), port), handler)
    return http.server.ThreadingHTTPServer((value, port), handler)


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
    """扫描 packaging/dist 内可分发的 PC 主应用安装包（启动器在独立分区列出）。"""
    installers: list[tuple[str, str, str]] = []
    if not os.path.isdir(DIST_DIR):
        return installers

    for name in sorted(os.listdir(DIST_DIR), key=str.lower):
        item_path = os.path.join(DIST_DIR, name)
        if not os.path.isfile(item_path) or not name.lower().endswith(PC_INSTALLER_EXTS):
            continue
        if "qlh-launcher" in name.lower():
            continue
        installers.append((name, "/" + quote(name), item_path))
    return installers


def _scan_launcher_assets() -> list[tuple[str, str, str]]:
    """扫描 packaging/dist 内的 QLH 启动器资产（Setup 安装包 + 自更新 ZIP）。"""
    assets: list[tuple[str, str, str]] = []
    if not os.path.isdir(DIST_DIR):
        return assets

    for name in sorted(os.listdir(DIST_DIR), key=str.lower):
        lower = name.lower()
        if "qlh-launcher" not in lower:
            continue
        is_setup_exe = lower.endswith(".exe") and "setup" in lower
        is_bundle_zip = lower.endswith(".zip")
        if not (is_setup_exe or is_bundle_zip):
            continue
        item_path = os.path.join(DIST_DIR, name)
        if not os.path.isfile(item_path):
            continue
        assets.append((name, "/" + quote(name), item_path))
    return assets


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

    def do_GET(self):
        if unquote(urlparse(self.path).path) == UPDATE_MANIFEST_PATH:
            encoded = json.dumps(
                build_update_manifest(signer=_SIGNER), ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(encoded)
            return
        super().do_GET()

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

        launcher_entries = [
            (display, href, _format_size(abs_path))
            for display, href, abs_path in _scan_launcher_assets()
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
        launcher_rows = render_rows(launcher_entries, "暂无 QLH 启动器（请先运行 build-launcher.bat）")
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

  <h2>QLH 启动器</h2>
  <ul>
    {launcher_rows}
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
    启动器: <code>packaging/dist/QLH-Launcher-Setup-v*.exe</code>（安装）+ <code>QLH-Launcher-v*.zip</code>（自更新资产，供 <code>qlh_launcher.py launcher-install</code> 使用）<br>
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


def _load_signer() -> "Signer | None":
    """Load the publisher signing key from QLH_SIGNING_KEY, fail-closed.

    When the variable is set the private key must exist and be loadable;
    silently serving an unsigned manifest would bypass the trusted gate.
    """
    env = os.environ.get("QLH_SIGNING_KEY", "").strip()
    if not env:
        return None
    if not os.path.isfile(env):
        raise RuntimeError(f"QLH_SIGNING_KEY 指向的私钥不存在: {env}")
    return Signer(env)


def _reconfigure_utf8(streams: Iterable[Any] | None = None) -> None:
    """Reconfigure console streams to UTF-8 (Windows GBK cannot emit emoji)."""
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> None:
    global _SIGNER
    # Windows 控制台默认 GBK 无法输出 emoji/部分中文，重配为 UTF-8。
    _reconfigure_utf8()
    argv = sys.argv[1:] if argv is None else argv
    port = int(argv[0]) if argv else DEFAULT_PORT
    try:
        _SIGNER = _load_signer()
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if _SIGNER is not None:
        print(f"  发布签名: 已启用（key_id={_SIGNER._key_id}）")

    ts_ip = _detect_tailscale_ip()
    android_packages = _scan_android_downloads()
    model_archives = _scan_model_archives()
    launcher_assets = _scan_launcher_assets()

    print()
    print("=" * 55)
    print("  📦 QLH 文件分发服务")
    print("=" * 55)
    print()
    print(f"  本机 Tailscale IP: {ts_ip}")
    print(f"  监听: http://{_url_host(HOST)}:{port}")
    print(f"  PC 安装包目录: {DIST_DIR}")
    print(f"  Android 输出目录: {ANDROID_OUTPUT_DIR}")
    if launcher_assets:
        print("  QLH 启动器:")
        for display, href, abs_path in launcher_assets:
            print(f"    {display} -> {href} ({_format_size(abs_path)})")
    else:
        print("  QLH 启动器: 未找到（请先运行 packaging/build-launcher.bat）")
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
        print(f"    http://{_url_host(ts_ip)}:{port}/")
        for _display, href, _abs_path in launcher_assets:
            print(f"    http://{_url_host(ts_ip)}:{port}{href}")
        for _display, href, _abs_path in android_packages:
            print(f"    http://{_url_host(ts_ip)}:{port}{href}")
        for _kind, _display, href, _abs_path in model_archives:
            print(f"    http://{_url_host(ts_ip)}:{port}{href}")
    else:
        print(f"    http://<本机IP>:{port}/")
        for _display, href, _abs_path in launcher_assets:
            print(f"    http://<本机IP>:{port}{href}")
        for _display, href, _abs_path in android_packages:
            print(f"    http://<本机IP>:{port}{href}")
        for _kind, _display, href, _abs_path in model_archives:
            print(f"    http://<本机IP>:{port}{href}")
    print()
    print("  按 Ctrl+C 停止服务")
    print("─" * 55)
    print()

    server = create_distribution_server(
        HOST,
        port,
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
