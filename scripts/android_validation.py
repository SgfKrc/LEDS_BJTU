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
            # ★ 2026-09-30：`adb logcat` 里含非 GBK 字节（UTF-8 中文/符号）时，
            #   默认编码解码会抛 `UnicodeDecodeError` 并**打断整个校验**（实测踩到）。
            #   adb 输出是 UTF-8 ⇒ 显式指定并容错。
            encoding="utf-8",
            errors="replace",
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


def _device_logcat(adb: str, serial: str, lines: int = 4000) -> str:
    """抓设备当前 logcat ring buffer（一次 dump，不阻塞跟随）。"""
    result = _run([adb, "-s", serial, "logcat", "-d", "-v", "brief"],
                  cwd=ROOT, timeout=90.0)
    if not result["ok"]:
        return ""
    return "\n".join(result["stdout"].splitlines()[-lines:])


def _device_process_alive(adb: str, serial: str, package: str) -> bool:
    result = _run([adb, "-s", serial, "shell", "pidof", package], cwd=ROOT, timeout=20.0)
    return bool(result["ok"] and result["stdout"].strip())


def _logcat_native_evidence(logcat: str, package: str) -> dict[str, Any]:
    """从 logcat 里提取**真机**证据，替代"安装+启动+ABI"式的推导。

    ★ 2026-09-30（审计 P1 Android 条）：`arm64_native_worker` 此前只由
    「APK 装了 + Activity 起来了 + 设备是 arm64」推导，**不能证明** JNI 层实际装载/运行。
    这里改为要求两条**可观测**证据：

    - `jni_library_loaded`：`nativeloader` 真的把包内的 `libqlh_llama_jni.so` 映射进来
      （实测日志：`Load …/base.apk!/lib/arm64-v8a/libqlh_llama_jni.so using ns …`）；
    - `foreground_service_started`：`ActivityManager` 放行前台服务
      （实测：`Background started FGS: Allowed [callingPackage: <pkg>…`）。

    `worker_service_referenced` 是**弱**证据（只说明系统见过 `TaskWorkerService`），
    单列出来供人工判断，**不**计入硬判据。
    """
    lower = logcat.lower()
    pkg_lower = package.lower()
    jni_loaded = any(
        "nativeloader" in line.lower() and "libqlh_llama_jni.so" in line.lower()
        for line in logcat.splitlines()
    )
    fgs_started = any(
        "background started fgs" in line.lower() and pkg_lower in line.lower()
        for line in logcat.splitlines()
    )
    worker_referenced = "taskworkerservice" in lower
    return {
        "jni_library_loaded": jni_loaded,
        "foreground_service_started": fgs_started,
        "worker_service_referenced": worker_referenced,
    }


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
    parser.add_argument("--settle-seconds", type=float, default=12.0,
                        help="启动后等待多少秒再抓 logcat（等 Application/JNI 库就绪）")
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
            [str(wrapper), ":app:testDebugUnitTest"],
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

    apk = ANDROID_ROOT / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
    if args.assemble and wrapper.is_file() and result["checks"]["java"]:
        build = _run(
            [str(wrapper), ":app:assembleDebug"],
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
            # ★ 2026-09-30（真机 Y700 实测）：**不要用 `monkey`** —— 联想 ZUI 的
            #   `AutoRunService` 会把 `monkey`（经 adb shell）拉起的进程判为
            #   「system relative auto run」并**主动 kill**（实测 logcat：
            #   `ProcessStateHandler … try to kill system relative auto run. package …`），
            #   表现为"闪退"但其实不是崩溃。`am start` 走正常 Activity 启动路径，不被杀。
            launch = _run(
                [adb, "-s", serial, "shell", "am", "start", "-W",
                 "-n", f"{args.package}/com.qlh.inference.MainActivity"],
                cwd=ROOT,
                timeout=60.0,
            )
            result["launch"] = launch
            result["checks"]["apk_launch"] = bool(launch["ok"])
            if launch["ok"]:
                # 让 App 完成冷启动（进程 fork + Application/Activity 起来 + 加载 JNI 库）
                # 再抓 logcat；否则 `nativeloader`/FGS 行可能还没出现。
                time.sleep(float(args.settle_seconds))
        result["evidence"]["apk_device_control_plane"] = bool(
            result["checks"].get("apk_install") and result["checks"].get("apk_launch")
        )
        # ★ 2026-09-30（审计 P1 Android 条）：`arm64_native_worker` 不再只由
        #   「安装 + 启动 + ABI」推导 —— 必须叠加**真机可观测**证据：
        #   进程存活 **且** JNI 库被 `nativeloader` 实际装载。任一条缺失即 False。
        logcat = _device_logcat(adb, serial)
        native_evidence = _logcat_native_evidence(logcat, args.package)
        result["logcat_evidence"] = native_evidence
        result["checks"]["device_process_alive"] = _device_process_alive(
            adb, serial, args.package)
        result["checks"]["device_jni_library_loaded"] = native_evidence["jni_library_loaded"]
        result["checks"]["device_foreground_service_started"] = (
            native_evidence["foreground_service_started"])
        result["checks"]["device_worker_service_referenced"] = (
            native_evidence["worker_service_referenced"])
        result["evidence"]["arm64_native_worker"] = bool(
            result["evidence"]["apk_device_control_plane"]
            and arm64
            and result["checks"]["device_process_alive"]
            and native_evidence["jni_library_loaded"]
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
