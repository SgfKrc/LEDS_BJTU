"""QLH update CLI.

This module is intentionally independent from the inference runtime.  It can
run before torch, transformers, Tailscale, or the application process starts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform as platform_module
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from update_core import (
    UpdateError,
    UpdateManifest,
    default_state_dir,
    download_asset,
    fetch_latest,
    load_json_state,
    select_asset,
    save_json_state,
    version_key,
)


LAUNCHER_VERSION = "0.1.8.1"
DEFAULT_UPDATE_SOURCES = (
    "http://100.90.76.108:9090/latest.json",
    "https://github.com/SgfKrc/LEDS_BJTU/releases/latest/download/latest.json",
)


def detect_current_version(
    search_root: Path | None = None, *, fallback: str = LAUNCHER_VERSION,
) -> str:
    override = os.environ.get("QLH_CURRENT_VERSION", "").strip()
    if override:
        version_key(override)
        return override
    if search_root is None:
        search_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    candidates = [search_root / "version.txt", search_root.parent / "version.txt"]
    for candidate in candidates:
        try:
            value = candidate.read_text(encoding="utf-8").strip()
            version_key(value)
            return value
        except (OSError, UpdateError):
            continue
    init_path = search_root / "src" / "__init__.py"
    try:
        content = init_path.read_text(encoding="utf-8")
    except OSError:
        pass
    else:
        match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            value = match.group(1)
            try:
                version_key(value)
            except UpdateError:
                pass
            else:
                return value
    return fallback


def detect_profile(*, variant_override: str | None = None) -> dict[str, str]:
    system = platform_module.system().lower()
    if system == "windows":
        platform = "windows"
    elif system == "linux":
        platform = "linux"
    elif system == "darwin":
        platform = "darwin"
    else:
        platform = system or "other"
    variant = (variant_override or os.environ.get("QLH_UPDATE_VARIANT", "")).lower()
    if not variant:
        variant = "cuda" if shutil.which("nvidia-smi") else "cpu"
    return {
        "platform": platform,
        "arch": platform_module.machine().lower() or "unknown",
        "variant": variant,
    }


def configured_sources(explicit: list[str] | None = None) -> list[str]:
    if explicit:
        return explicit
    raw = os.environ.get("QLH_UPDATE_SOURCE", "")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    config = load_json_state(default_state_dir() / "launcher.json")
    source = config.get("update_source")
    if isinstance(source, list):
        return [str(item).strip() for item in source if str(item).strip()]
    return [str(source)] if source else list(DEFAULT_UPDATE_SOURCES)


def check_updates(
    sources: list[str],
    *,
    profile: dict[str, str] | None = None,
    current_version: str | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    profile = profile or detect_profile()
    current_version = current_version or detect_current_version()
    manifest, failures = fetch_latest(sources, timeout=timeout)
    return _result_for_manifest(manifest, failures, profile, current_version)


def _result_for_manifest(
    manifest: UpdateManifest,
    failures: tuple[str, ...],
    profile: dict[str, str],
    current_version: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "current_version": current_version,
        "launcher_version": LAUNCHER_VERSION,
        "latest_version": manifest.tag,
        "channel": manifest.channel,
        "source": manifest.source_url,
        "signature_present": manifest.signature_present,
        "signature_verified": manifest.signature_verified,
        "profile": profile,
        "update_available": version_key(manifest.tag) > version_key(current_version),
        "source_failures": list(failures),
    }
    try:
        asset = select_asset(
            manifest,
            platform=profile["platform"],
            variant=profile["variant"],
            arch=profile["arch"],
        )
    except UpdateError as exc:
        result["asset_error"] = str(exc)
    else:
        result["asset"] = {
            "name": asset.name,
            "size": asset.size,
            "sha256": asset.sha256,
            "url": asset.url,
        }
    return result


def _manifest_for_request(
    sources: list[str], timeout: float,
) -> tuple[UpdateManifest, tuple[str, ...]]:
    manifest, failures = fetch_latest(sources, timeout=timeout)
    return manifest, failures


def _print(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    elif isinstance(value, dict):
        if value.get("error"):
            print(f"更新失败: {value['error']}")
            return
        if value.get("downloaded"):
            print(f"已下载并校验: {value['downloaded']}")
            return
        print(f"当前版本: {value.get('current_version', LAUNCHER_VERSION)}")
        print(f"最新版本: {value.get('latest_version', '-')}")
        print("有可用更新" if value.get("update_available") else "当前已是最新版本")
        if value.get("asset_error"):
            print(f"资产匹配失败: {value['asset_error']}")
        elif value.get("asset"):
            print(f"匹配资产: {value['asset']['name']}")
        if value.get("source_failures"):
            print("失败源: " + "; ".join(value["source_failures"]))
    else:
        print(value)


def _launch_installer(path: Path) -> int:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return 0
    if path.suffix.lower() == ".deb":
        print(f"请执行: sudo dpkg -i {path}")
        return 0
    print(f"已下载安装包: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlh-updater")
    parser.add_argument("command", nargs="?", default="check", choices=("check", "download", "install"))
    parser.add_argument("--source", action="append", help="manifest URL; may be repeated")
    parser.add_argument("--variant", choices=("cpu", "cuda", "full", "lite"))
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--yes", action="store_true", help="do not ask before launching an installer")
    parser.add_argument(
        "--allow-unsigned", action="store_true",
        help="explicitly allow a manifest without a signature; never used by auto-check",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = configured_sources(args.source)
    if not sources:
        error = "未配置更新源，请使用 --source 或 QLH_UPDATE_SOURCE"
        _print({"error": error}, as_json=args.as_json)
        return 2
    profile = detect_profile(variant_override=args.variant)
    try:
        if args.command == "check":
            result = check_updates(sources, profile=profile, timeout=args.timeout)
            _print(result, as_json=args.as_json)
            return 3 if result.get("update_available") else 0
        manifest, failures = _manifest_for_request(sources, args.timeout)
        result = _result_for_manifest(
            manifest, failures, profile, detect_current_version(),
        )
        if not args.as_json:
            _print(result, as_json=False)
        asset = select_asset(
            manifest,
            platform=profile["platform"],
            variant=profile["variant"],
            arch=profile["arch"],
        )
        destination = args.download_dir or default_state_dir() / "downloads"
        path = download_asset(asset, destination, timeout=max(30.0, args.timeout))
        if args.command == "download":
            _print({"downloaded": str(path), "source_failures": list(failures)}, as_json=args.as_json)
            return 0
        if not manifest.signature_verified and not args.allow_unsigned:
            detail = "清单签名尚未验证" if manifest.signature_present else "清单没有签名"
            raise UpdateError(f"{detail}；只能下载，必须显式 --allow-unsigned 才能启动安装")
        if not args.yes and not args.as_json:
            answer = input(f"将启动安装包 {path.name}，继续？ [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                return 1
        state_path = default_state_dir() / "launcher.json"
        state = load_json_state(state_path)
        state["pending_install"] = {
            "tag": manifest.tag,
            "asset": asset.name,
            "path": str(path),
        }
        save_json_state(state_path, state)
        code = _launch_installer(path)
        if args.as_json:
            _print(
                {"installer_started": code == 0, "path": str(path), "tag": manifest.tag},
                as_json=True,
            )
        return code
    except UpdateError as exc:
        _print({"error": str(exc)}, as_json=args.as_json)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
