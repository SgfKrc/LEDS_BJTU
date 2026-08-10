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
import subprocess
import sys
from datetime import datetime, timezone
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
from signing import default_trusted_keys_dir
from launcher_slots import LauncherSlotError, LauncherSlotStore
from version_store import VersionStore, VersionStoreError


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
    trusted_keys_dir: str | None = None,
) -> dict[str, Any]:
    profile = profile or detect_profile()
    current_version = current_version or detect_current_version()
    manifest, failures = fetch_latest(
        sources, timeout=timeout,
        fetcher=lambda url, timeout: fetch_manifest_with_keys(
            url, timeout=timeout, trusted_keys_dir=trusted_keys_dir,
        ),
    )
    return _result_for_manifest(manifest, failures, profile, current_version)


def _result_for_manifest(
    manifest: UpdateManifest,
    failures: tuple[str, ...],
    profile: dict[str, str],
    current_version: str,
    *,
    kind: str = "installer",
    asset_variant: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "current_version": current_version,
        "launcher_version": LAUNCHER_VERSION,
        "latest_version": manifest.tag,
        "channel": manifest.channel,
        "source": manifest.source_url,
        "signature_present": manifest.signature_present,
        "signature_verified": manifest.signature_verified,
        "signature_key_id": manifest.signature_key_id,
        "signature_error": manifest.signature_error,
        "profile": profile,
        "asset_kind": kind,
        "update_available": version_key(manifest.tag) > version_key(current_version),
        "source_failures": list(failures),
    }
    try:
        asset = select_asset(
            manifest,
            platform=profile["platform"],
            variant=asset_variant or profile["variant"],
            arch=profile["arch"],
            kind=kind,
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


def check_launcher_updates(
    sources: list[str],
    *,
    profile: dict[str, str] | None = None,
    current_version: str = LAUNCHER_VERSION,
    timeout: float = 8.0,
    trusted_keys_dir: str | None = None,
) -> dict[str, Any]:
    """Check the shared Launcher bundle without selecting an app variant."""
    profile = profile or detect_profile()
    manifest, failures = _manifest_for_request(
        sources, timeout, trusted_keys_dir=trusted_keys_dir,
    )
    return _result_for_manifest(
        manifest, failures, profile, current_version,
        kind="launcher", asset_variant="any",
    )


def fetch_manifest_with_keys(
    url: str, *, timeout: float, trusted_keys_dir: str | None = None,
) -> UpdateManifest:
    """fetch_manifest with the default (or explicit) trusted key set."""
    from update_core import fetch_manifest

    return fetch_manifest(
        url, timeout=timeout,
        trusted_keys_dir=trusted_keys_dir or default_trusted_keys_dir(),
    )


def _manifest_for_request(
    sources: list[str], timeout: float, trusted_keys_dir: str | None = None,
) -> tuple[UpdateManifest, tuple[str, ...]]:
    manifest, failures = fetch_latest(
        sources, timeout=timeout,
        fetcher=lambda url, timeout: fetch_manifest_with_keys(
            url, timeout=timeout, trusted_keys_dir=trusted_keys_dir,
        ),
    )
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


def _version_store(args: argparse.Namespace) -> VersionStore:
    return VersionStore(args.version_store)


def _version_health(path: Path) -> bool:
    # Atomic activation requires an explicit publisher-created health marker.
    return (path / "health.ok").is_file()


def _version_command(args: argparse.Namespace) -> int:
    store = _version_store(args)
    try:
        if args.command == "version-status":
            _print(store.status(), as_json=args.as_json)
            return 0
        if args.command == "version-stage":
            if not args.bundle or not args.version or not args.variant:
                raise VersionStoreError("version-stage requires --bundle --version --variant")
            bundle = Path(args.bundle).expanduser()
            target = (
                store.stage_directory(bundle, args.version, args.variant)
                if bundle.is_dir()
                else store.stage_archive(bundle, args.version, args.variant)
            )
            _print({"staged": str(target)}, as_json=args.as_json)
            return 0
        if args.command == "version-activate":
            if not args.version or not args.variant:
                raise VersionStoreError("version-activate requires --version --variant")
            pointer = store.activate(
                args.version, args.variant, health_check=_version_health,
            )
            _print({"activated": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "version-rollback":
            pointer = store.rollback(health_check=_version_health)
            _print({"rolled_back": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "version-recover":
            pointer = store.recover()
            if pointer is None:
                raise VersionStoreError("no recoverable version is available")
            _print({"recovered": pointer.as_dict()}, as_json=args.as_json)
            return 0
    except VersionStoreError as exc:
        _print({"error": str(exc)}, as_json=args.as_json)
        return 2
    raise VersionStoreError(f"unsupported version command: {args.command}")


def _launcher_store(args: argparse.Namespace) -> LauncherSlotStore:
    return LauncherSlotStore(args.launcher_store)


def _launcher_health(path: Path) -> bool:
    """Run the staged Launcher in an isolated, non-interactive health mode."""
    marker = path / "health.ok"
    if not marker.is_file() or not marker.read_text(encoding="utf-8", errors="replace").strip():
        return False
    candidates = (
        path / "QLH-Launcher.exe",
        path / "qlh-launcher.exe",
        path / "qlh-launcher",
        path / "qlh_launcher.py",
    )
    entrypoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    if entrypoint is None:
        return False
    command = (
        [sys.executable, str(entrypoint), "--health-check"]
        if entrypoint.suffix.lower() == ".py"
        else [str(entrypoint), "--health-check"]
    )
    environment = os.environ.copy()
    environment["QLH_LAUNCHER_ACTIVE_SLOT"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=str(path),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _launcher_current_version(store: LauncherSlotStore) -> str:
    pointer = store.current()
    return pointer.version if pointer is not None else LAUNCHER_VERSION


def _require_verified_signature(manifest: UpdateManifest, args: argparse.Namespace) -> None:
    if manifest.signature_verified or args.allow_unsigned:
        return
    if manifest.signature_error:
        detail = f"发布签名验证失败：{manifest.signature_error}"
    elif manifest.signature_present:
        detail = "清单签名尚未验证"
    else:
        detail = "清单没有签名"
    raise UpdateError(f"{detail}；只能下载，必须显式 --allow-unsigned 才能安装")


def _diagnostic_output(args: argparse.Namespace) -> Path:
    if args.diagnostics_output:
        return Path(args.diagnostics_output).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return default_state_dir() / "diagnostics" / f"qlh-launcher-{stamp}.zip"


def _diagnose_report_path(args: argparse.Namespace) -> Path | None:
    if not args.diagnose_report:
        return None
    candidate = Path(args.diagnose_report).expanduser().resolve()
    allowed = (default_state_dir() / "diagnostics").resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise LauncherSlotError("diagnose report must stay in the Launcher diagnostics directory") from exc
    if candidate.suffix.lower() != ".json" or not candidate.is_file():
        raise LauncherSlotError("diagnose report must be an existing JSON file")
    try:
        if candidate.stat().st_size > 1024 * 1024:
            raise LauncherSlotError("diagnose report exceeds the 1 MiB limit")
    except OSError as exc:
        raise LauncherSlotError("cannot read diagnose report") from exc
    return candidate


def _launcher_command(
    args: argparse.Namespace,
    *,
    sources: list[str] | None = None,
    trusted_keys_dir: str | None = None,
) -> int:
    store = _launcher_store(args)
    try:
        if args.command == "launcher-status":
            _print(store.status(), as_json=args.as_json)
            return 0
        if args.command == "launcher-stage":
            if not args.bundle or not args.version:
                raise LauncherSlotError("launcher-stage requires --bundle --version")
            bundle = Path(args.bundle).expanduser()
            pointer = (
                store.stage_directory(bundle, args.version)
                if bundle.is_dir()
                else store.stage_archive(bundle, args.version)
            )
            _print({"staged": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "launcher-activate":
            if not args.version:
                raise LauncherSlotError("launcher-activate requires --version")
            pointer = store.activate(args.version, health_check=_launcher_health)
            _print({"activated": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "launcher-rollback":
            pointer = store.rollback(health_check=_launcher_health)
            _print({"rolled_back": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "launcher-recover":
            pointer = store.recover()
            if pointer is None:
                raise LauncherSlotError("no recoverable Launcher slot is available")
            _print({"recovered": pointer.as_dict()}, as_json=args.as_json)
            return 0
        if args.command == "diagnostics":
            report = _diagnose_report_path(args)
            destination = store.diagnostics(
                _diagnostic_output(args), extra_paths=[report] if report else None,
            )
            _print({"diagnostics": str(destination)}, as_json=args.as_json)
            return 0
        if args.command not in {"launcher-check", "launcher-download", "launcher-install"}:
            raise LauncherSlotError(f"unsupported Launcher command: {args.command}")
        if not sources:
            raise LauncherSlotError("no update source configured")
        profile = detect_profile(variant_override=args.variant)
        manifest, failures = _manifest_for_request(
            sources, args.timeout, trusted_keys_dir=trusted_keys_dir,
        )
        result = _result_for_manifest(
            manifest, failures, profile, _launcher_current_version(store),
            kind="launcher", asset_variant="any",
        )
        if args.command == "launcher-check":
            _print(result, as_json=args.as_json)
            return 3 if result.get("update_available") else 0
        if not args.as_json:
            _print(result, as_json=False)
        asset = select_asset(
            manifest, platform=profile["platform"], variant="any",
            arch=profile["arch"], kind="launcher",
        )
        destination = args.download_dir or default_state_dir() / "downloads"
        path = download_asset(asset, destination, timeout=max(30.0, args.timeout))
        if args.command == "launcher-download":
            _print({"downloaded": str(path), "source_failures": list(failures)}, as_json=args.as_json)
            return 0
        _require_verified_signature(manifest, args)
        if not args.yes and not args.as_json:
            answer = input(f"将替换活动 Launcher 为 {path.name}，继续？ [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                return 1
        staged = store.stage_archive(
            path, manifest.tag, require_install_manifest=True,
        )
        pointer = store.activate(staged.version, health_check=_launcher_health)
        _print(
            {"launcher_updated": pointer.as_dict(), "path": str(path)},
            as_json=args.as_json,
        )
        return 0
    except (LauncherSlotError, UpdateError) as exc:
        _print({"error": str(exc)}, as_json=args.as_json)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qlh-updater")
    parser.add_argument(
        "command", nargs="?", default="check",
        choices=(
            "check", "download", "install",
            "version-status", "version-stage", "version-activate",
            "version-rollback", "version-recover",
            "launcher-status", "launcher-check", "launcher-download",
            "launcher-install", "launcher-stage", "launcher-activate",
            "launcher-rollback", "launcher-recover", "diagnostics",
        ),
    )
    parser.add_argument("--source", action="append", help="manifest URL; may be repeated")
    parser.add_argument("--variant", choices=("cpu", "cuda", "full", "lite"))
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--trusted-keys-dir", help="pubkeys directory for manifest verification")
    parser.add_argument("--version-store", help="UP-N3 version store directory")
    parser.add_argument("--launcher-store", help="UP-N4 Launcher A/B store directory")
    parser.add_argument("--bundle", help="directory or zip/tar bundle for version-stage")
    parser.add_argument("--version", help="application version for UP-N3 operations")
    parser.add_argument("--diagnostics-output", help="ZIP path for the redacted diagnostic bundle")
    parser.add_argument("--diagnose-report", help="bundle-safe diagnosis JSON inside the Launcher state directory")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--yes", action="store_true", help="do not ask before launching an installer")
    parser.add_argument(
        "--allow-unsigned", action="store_true",
        help="explicitly allow a manifest without a signature; never used by auto-check",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command.startswith("version-"):
        return _version_command(args)
    local_launcher_commands = {
        "launcher-status", "launcher-stage", "launcher-activate",
        "launcher-rollback", "launcher-recover", "diagnostics",
    }
    if args.command in local_launcher_commands:
        return _launcher_command(args)
    trusted_keys_dir = args.trusted_keys_dir or default_trusted_keys_dir()
    sources = configured_sources(args.source)
    if not sources:
        error = "未配置更新源，请使用 --source 或 QLH_UPDATE_SOURCE"
        _print({"error": error}, as_json=args.as_json)
        return 2
    profile = detect_profile(variant_override=args.variant)
    try:
        if args.command in {"launcher-check", "launcher-download", "launcher-install"}:
            return _launcher_command(
                args, sources=sources, trusted_keys_dir=trusted_keys_dir,
            )
        if args.command == "check":
            result = check_updates(
                sources, profile=profile, timeout=args.timeout,
                trusted_keys_dir=trusted_keys_dir,
            )
            _print(result, as_json=args.as_json)
            return 3 if result.get("update_available") else 0
        manifest, failures = _manifest_for_request(
            sources, args.timeout, trusted_keys_dir=trusted_keys_dir,
        )
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
            if manifest.signature_error:
                detail = f"发布签名验证失败：{manifest.signature_error}"
            elif manifest.signature_present:
                detail = "清单签名尚未验证"
            else:
                detail = "清单没有签名"
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
