"""Run the Android validation layers available on a non-Android development host.

This command deliberately separates JVM/build evidence from device evidence. A
successful Gradle build never implies that the arm64 JNI RPC worker ran.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANDROID_ROOT = ROOT / "android"


def _tool(name: str) -> str | None:
    return shutil.which(name)


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float = 300.0,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": -1,
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _android_sdk(adb: str | None) -> Path | None:
    candidates = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
    ]
    if adb:
        candidates.append(str(Path(adb).resolve().parent.parent))
    if os.name == "nt":
        candidates.append(str(Path.home() / "AppData" / "Local" / "Android" / "Sdk"))
        candidates.append(str(Path.home() / "Android" / "Sdk"))
    for value in candidates:
        if value:
            path = Path(value).expanduser()
            if (path / "platform-tools").is_dir() or (path / "platforms").is_dir():
                return path.resolve()
    return None


def _adb_devices(adb: str) -> list[dict[str, str]]:
    result = _run([adb, "devices", "-l"], cwd=ROOT, timeout=30.0)
    devices: list[dict[str, str]] = []
    for line in result["stdout"].splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2 or fields[1] != "device":
            continue
        item = {"serial": fields[0], "state": fields[1]}
        for field in fields[2:]:
            if ":" in field:
                key, value = field.split(":", 1)
                item[key] = value
        devices.append(item)
    return devices


def _device_props(adb: str, serial: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for name in (
        "ro.product.cpu.abi",
        "ro.product.cpu.abilist",
        "ro.build.version.sdk",
        "ro.product.model",
        "ro.kernel.qemu",
    ):
        result = _run([adb, "-s", serial, "shell", "getprop", name], cwd=ROOT, timeout=15.0)
        if result["ok"]:
            props[name] = result["stdout"].strip()
    return props


def _apk_has_arm64_worker(apk: Path) -> bool:
    if not apk.is_file():
        return False
    try:
        with zipfile.ZipFile(apk) as archive:
            return any(
                name.startswith("lib/arm64-v8a/") and "qlh_llama_jni" in name
                for name in archive.namelist()
            )
    except (OSError, zipfile.BadZipFile):
        return False


def validate(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="分层执行 QLH Android 构建、单元测试和设备证据检查。",
    )
    parser.add_argument("--skip-unit", action="store_true", help="跳过 JVM 单元测试")
    parser.add_argument("--assemble", action="store_true", help="构建 fullDebug APK")
    parser.add_argument("--serial", help="指定 adb serial；默认选择第一个在线设备")
    parser.add_argument("--package", default="com.qlh.inference.debug")
    parser.add_argument("--install", action="store_true", help="安装已构建的 fullDebug APK")
    parser.add_argument("--launch", action="store_true", help="启动已安装的 Android Activity")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "host": {
            "os": os.name,
            "java": bool(_tool("java")),
            "adb": bool(_tool("adb")),
            "emulator": bool(_tool("emulator")),
        },
        "checks": {},
        "gradle": [],
        "devices": [],
        "evidence": {
            "jvm_and_build": False,
            "apk_device_control_plane": False,
            "arm64_native_worker": False,
        },
    }

    wrapper = ANDROID_ROOT / ("gradlew.bat" if os.name == "nt" else "gradlew")
    result["checks"]["android_project"] = ANDROID_ROOT.is_dir()
    result["checks"]["gradle_wrapper"] = wrapper.is_file()
    result["checks"]["java"] = bool(_tool("java"))
    adb = _tool("adb")
    sdk = _android_sdk(adb)
    result["host"]["android_sdk"] = str(sdk) if sdk else None
    result["checks"]["android_sdk"] = sdk is not None
    gradle_environment = os.environ.copy()
    if sdk is not None:
        gradle_environment.setdefault("ANDROID_HOME", str(sdk))
        gradle_environment.setdefault("ANDROID_SDK_ROOT", str(sdk))

    if not args.skip_unit and wrapper.is_file() and result["checks"]["java"]:
        unit = _run(
            [str(wrapper), ":app:testFullDebugUnitTest"],
            cwd=ANDROID_ROOT,
            environment=gradle_environment,
        )
        result["gradle"].append(unit)
        result["checks"]["jvm_unit_tests"] = bool(unit["ok"])
        result["evidence"]["jvm_and_build"] = bool(unit["ok"])
    elif args.skip_unit:
        result["checks"]["jvm_unit_tests"] = "skipped"
    else:
        result["checks"]["jvm_unit_tests"] = False

    apk = ANDROID_ROOT / "app" / "build" / "outputs" / "apk" / "full" / "debug" / "app-full-debug.apk"
    if args.assemble and wrapper.is_file() and result["checks"]["java"]:
        build = _run(
            [str(wrapper), ":app:assembleFullDebug"],
            cwd=ANDROID_ROOT,
            environment=gradle_environment,
        )
        result["gradle"].append(build)
        result["checks"]["apk_build"] = bool(build["ok"] and apk.is_file())
    elif args.assemble:
        result["checks"]["apk_build"] = False
    else:
        result["checks"]["apk_build"] = "not_requested"
    result["checks"]["apk_arm64_jni"] = (
        _apk_has_arm64_worker(apk) if apk.is_file() else "not_observed"
    )

    if adb:
        result["devices"] = _adb_devices(adb)
    result["checks"]["adb_available"] = bool(adb)
    result["checks"]["online_device"] = bool(result["devices"])

    serial = args.serial or (result["devices"][0]["serial"] if result["devices"] else None)
    if serial and adb:
        props = _device_props(adb, serial)
        result["device"] = {"serial": serial, "props": props}
        abi_list = props.get("ro.product.cpu.abilist", "")
        primary_abi = props.get("ro.product.cpu.abi", "")
        arm64 = "arm64-v8a" in {item.strip() for item in (abi_list + "," + primary_abi).split(",")}
        result["checks"]["arm64_abi"] = arm64
        if args.install:
            if not apk.is_file():
                result["checks"]["apk_install"] = False
                result["install_error"] = "fullDebug APK 不存在，请先使用 --assemble"
            else:
                install = _run([adb, "-s", serial, "install", "-r", str(apk)], cwd=ROOT, timeout=180.0)
                result["install"] = install
                result["checks"]["apk_install"] = bool(install["ok"])
        if args.launch:
            launch = _run(
                [adb, "-s", serial, "shell", "monkey", "-p", args.package, "1"],
                cwd=ROOT,
                timeout=30.0,
            )
            result["launch"] = launch
            result["checks"]["apk_launch"] = bool(launch["ok"])
        result["evidence"]["apk_device_control_plane"] = bool(
            result["checks"].get("apk_install") and result["checks"].get("apk_launch")
        )
        result["evidence"]["arm64_native_worker"] = bool(
            result["evidence"]["apk_device_control_plane"] and arm64
        )
    else:
        result["checks"]["arm64_abi"] = "not_observed"
        result["checks"]["apk_install"] = "not_requested_or_no_device"
        result["checks"]["apk_launch"] = "not_requested_or_no_device"

    result["ok"] = bool(result["evidence"]["jvm_and_build"])
    result["status"] = (
        "jvm/build-pass-device-pending"
        if result["ok"] and not result["evidence"]["apk_device_control_plane"]
        else "device-control-pass-native-worker-pending"
        if result["evidence"]["apk_device_control_plane"] and not result["evidence"]["arm64_native_worker"]
        else "native-worker-candidate"
        if result["evidence"]["arm64_native_worker"]
        else "failed"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    result = validate(argv)
    if next((item for item in (argv or sys.argv[1:]) if item == "--json"), None):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Android validation: %s" % result["status"])
        for name, passed in result["checks"].items():
            print("%-24s %s" % (name, passed if isinstance(passed, str) else ("PASS" if passed else "FAIL")))
        print("evidence: %s" % json.dumps(result["evidence"], ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
