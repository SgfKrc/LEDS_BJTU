#!/usr/bin/env python3
"""Standalone QLH bootstrap launcher with GUI and TUI frontends.

It intentionally knows how to launch the application but never imports the
inference runtime.  This keeps startup, update checks, and repair available
when torch/model/database dependencies are broken.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

_PACKAGING_DIR = Path(__file__).resolve().parent
if str(_PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGING_DIR))

from update_core import UpdateError, default_state_dir, load_json_state
from diagnose import diagnose_install, format_diagnosis, write_diagnosis_report
from install_manifest import main as install_manifest_main
from install_manifest import verify_install_tree
from launcher_slots import LauncherSlotStore, should_delegate
from updater import (
    check_launcher_updates, check_updates, configured_sources, detect_current_version, detect_profile,
    main as updater_main,
)
from version_store import VersionStore


LAUNCHER_VERSION = "0.1.8.1"


def install_root() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        if root.name.lower() == "bin":
            return root.parent
        return root
    return _PACKAGING_DIR.parent


def _candidate_app_roots(preferred_variant: str | None = None) -> list[Path]:
    state = load_json_state(default_state_dir() / "launcher.json")
    configured = [os.environ.get("QLH_APP_HOME", ""), state.get("app_path", "")]
    roots = [Path(value).expanduser() for value in configured if isinstance(value, str) and value]
    try:
        active = VersionStore().active_path()
    except Exception:
        active = None
    if active is not None:
        roots.append(active)
    roots.append(install_root())
    if os.name == "nt":
        names = (
            ("QLH-Edge-Inference-CUDA", "QLH-Edge-Inference")
            if preferred_variant == "cuda"
            else ("QLH-Edge-Inference", "QLH-Edge-Inference-CUDA")
        )
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(variable)
            if base:
                roots.extend(Path(base) / name for name in names)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.extend(Path(local) / "Programs" / name for name in names)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def installed_app_root(preferred_variant: str | None = None) -> Path | None:
    for root in _candidate_app_roots(preferred_variant):
        if os.name == "nt" and (root / "QLH-Edge-Inference.exe").is_file():
            return root
        if os.name != "nt" and (root / "bin" / "qlh-app").is_file():
            return root
        if not getattr(sys, "frozen", False) and (root / "src" / "api_server.py").is_file():
            return root
    return None


def quick_verify_after_launch_failure(
    preferred_variant: str | None = None,
) -> tuple[Path, dict] | None:
    """Run the bounded UP-N6.1 check only after a startup call fails."""
    candidates: list[Path] = []
    detected = installed_app_root(preferred_variant)
    if detected is not None:
        candidates.append(detected)
    candidates.extend(_candidate_app_roots(preferred_variant))
    seen: set[str] = set()
    for root in candidates:
        try:
            candidate = root.expanduser().absolute()
        except OSError:
            continue
        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)
        if not (candidate / "manifest" / "install-manifest.json").is_file():
            continue
        return candidate, verify_install_tree(candidate, level="quick")
    return None


def diagnosis_app_root(preferred_variant: str | None = None) -> Path | None:
    """Find an application root for read-only diagnosis without following links."""
    detected = installed_app_root(preferred_variant)
    if detected is not None:
        return detected.expanduser().absolute()
    for root in _candidate_app_roots(preferred_variant):
        candidate = root.expanduser().absolute()
        if (candidate / "manifest" / "install-manifest.json").is_file():
            return candidate
    return None


def app_command(mode: str, variant_override: str | None = None) -> list[str]:
    root = installed_app_root(variant_override)
    if root is None:
        raise FileNotFoundError("未检测到 QLH 主程序；请先检查更新并安装应用包")
    if os.name == "nt":
        app = root / "QLH-Edge-Inference.exe"
        if app.is_file():
            return [str(app), f"--{mode}"]
    else:
        installed_app = root / "bin" / "qlh-app"
        if installed_app.is_file():
            # Linux 包内 venv 是主应用的唯一依赖边界；仅旧包缺失时才回退 shebang。
            venv_python = root / "venv" / "bin" / "python"
            if venv_python.is_file():
                return [str(venv_python), str(installed_app), f"--{mode}"]
            return [str(installed_app), f"--{mode}"]
    legacy = root / "packaging" / "launcher.py"
    if legacy.is_file():
        return [sys.executable, str(legacy), f"--{mode}"]
    raise FileNotFoundError("QLH 安装目录存在，但主程序入口缺失；请执行覆盖安装")


def launch_app(mode: str, variant_override: str | None = None) -> subprocess.Popen:
    command = app_command(mode, variant_override)
    root = installed_app_root(variant_override) or install_root()
    return subprocess.Popen(command, cwd=str(root))


def _delegate_to_active_launcher(argv: list[str]) -> int | None:
    """Run a normal command from the active A/B slot, if one is healthy.

    The installed Launcher intentionally stays outside the slots.  Maintenance
    commands are handled by that stable copy, so a bad update cannot remove
    the only repair path.
    """
    if os.environ.get("QLH_LAUNCHER_ACTIVE_SLOT") == "1":
        return None
    try:
        store = LauncherSlotStore()
        if store.recover() is None:
            return None
        active = store.active_path()
    except Exception:
        return None
    if active is None:
        return None
    current = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()
    if active.resolve() == current:
        return None
    if not _probe_launcher_path(active.parent, active):
        # A slot can be present yet fail to import.  Roll back only after the
        # probe fails; if rollback is unavailable, keep using this stable copy.
        try:
            store.rollback(health_check=lambda path: _probe_launcher_path(
                path,
                store._entrypoint_path(store.previous()),  # type: ignore[arg-type]
            ))
            active = store.active_path()
        except Exception:
            return None
        if active is None or not _probe_launcher_path(active.parent, active):
            return None
    command = [str(active), *argv]
    if active.suffix.lower() == ".py":
        command.insert(0, sys.executable)
    environment = os.environ.copy()
    environment["QLH_LAUNCHER_ACTIVE_SLOT"] = "1"
    try:
        return subprocess.run(
            command, cwd=str(active.parent), env=environment, check=False,
        ).returncode
    except OSError:
        return None


def _probe_launcher_path(root: Path, entrypoint: Path | None) -> bool:
    if entrypoint is None or not entrypoint.is_file():
        return False
    marker = root / "health.ok"
    if not marker.is_file():
        return False
    command = (
        [sys.executable, str(entrypoint), "--health-check"]
        if entrypoint.suffix.lower() == ".py"
        else [str(entrypoint), "--health-check"]
    )
    environment = os.environ.copy()
    environment["QLH_LAUNCHER_ACTIVE_SLOT"] = "1"
    try:
        return subprocess.run(
            command,
            cwd=str(root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _self_health_check() -> int:
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    marker = root / "health.ok"
    # Source-tree execution is useful for tests and development.  A staged
    # release must carry the explicit marker generated by build-launcher.bat.
    if not getattr(sys, "frozen", False) and not marker.exists():
        return 0
    try:
        return 0 if marker.read_text(encoding="utf-8", errors="replace").strip() else 2
    except OSError:
        return 2


def _ensure_windows_console() -> bool:
    if os.name != "nt":
        return True
    if sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        return True
    try:
        import ctypes

        if not ctypes.windll.kernel32.AllocConsole():
            return False
        sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        return True
    except Exception:
        return False


def _render_update_result(result: dict) -> str:
    if result.get("error"):
        return f"更新检查失败: {result['error']}"
    if result.get("asset_error"):
        return f"发现 {result.get('latest_version')}，但没有匹配当前设备的安装包。"
    if result.get("update_available"):
        signed = "已验签" if result.get("signature_verified") else "未验签"
        return f"发现新版本 {result.get('latest_version')}（{signed}）：{result['asset']['name']}"
    return f"当前已是最新版本 {result.get('current_version')}。"


class LauncherController:
    """Shared actions used by the GUI and TUI without sharing event loops."""

    def __init__(
        self, sources: list[str] | None = None, variant_override: str | None = None,
    ):
        self.sources = sources
        self.variant_override = variant_override
        self.last_error = ""
        self.last_diagnosis: dict | None = None

    def set_variant(self, variant: str) -> None:
        if variant not in {"cpu", "cuda"}:
            raise ValueError(f"不支持的 PC 安装包变体: {variant}")
        self.variant_override = variant

    def check_update(self) -> dict:
        sources = self.sources or configured_sources()
        if not sources:
            return {"error": "未配置更新源，请设置 QLH_UPDATE_SOURCE"}
        app_root = installed_app_root(self.variant_override)
        try:
            return check_updates(
                sources,
                profile=detect_profile(variant_override=self.variant_override),
                current_version=(
                    detect_current_version(app_root, fallback="0.0.0")
                    if app_root else "0.0.0"
                ),
            )
        except UpdateError as exc:
            return {"error": str(exc)}

    def check_launcher_update(self) -> dict:
        sources = self.sources or configured_sources()
        if not sources:
            return {"error": "未配置更新源，请设置 QLH_UPDATE_SOURCE"}
        try:
            pointer = LauncherSlotStore().current()
            return check_launcher_updates(
                sources,
                profile=detect_profile(variant_override=self.variant_override),
                current_version=pointer.version if pointer else LAUNCHER_VERSION,
            )
        except UpdateError as exc:
            return {"error": str(exc)}

    def start_gui(self) -> int:
        try:
            launch_app("ui", self.variant_override)
            return 0
        except (OSError, FileNotFoundError) as exc:
            return self._launch_failed(exc)

    def start_tui(self) -> int:
        try:
            launch_app("tui", self.variant_override)
            return 0
        except (OSError, FileNotFoundError) as exc:
            return self._launch_failed(exc)

    def _launch_failed(self, exc: Exception) -> int:
        self.last_error = str(exc)
        verification = quick_verify_after_launch_failure(self.variant_override)
        if verification is not None:
            root, report = verification
            summary = report["summary"]
            if report["ok"]:
                detail = "签名安装清单 quick 校验通过，未发现关键程序文件损坏。"
            else:
                first = report["failed"][0]
                detail = (
                    "签名安装清单 quick 校验失败："
                    f"{first['path']}（{first['category']}）；"
                    f"检查 {summary['checked']} 项，失败 {summary['failed']} 项。"
                )
            self.last_error = f"{self.last_error}\n{detail}\n安装目录：{root}"
            diagnosis = self.diagnose_app(error=str(exc), integrity_report=report)
            if diagnosis and diagnosis["issues"]:
                first_issue = diagnosis["issues"][0]
                self.last_error = (
                    f"{self.last_error}\n诊断建议：{first_issue['title']}；"
                    f"{first_issue['manual_steps'][0]}"
                )
        return 2

    def diagnose_app(
        self, *, error: str | None = None, integrity_report: dict | None = None,
    ) -> dict | None:
        root = diagnosis_app_root(self.variant_override)
        if root is None:
            self.last_diagnosis = None
            return None
        self.last_diagnosis = diagnose_install(
            root,
            error=error,
            integrity_report=integrity_report,
        )
        return self.last_diagnosis

    def install_update(self) -> int:
        # UP-N2: install only proceeds when the manifest signature verifies.
        # No --allow-unsigned here: unsigned manifests must fail closed, and
        # the GUI/TUI flows never get a chance to bypass the gate.
        forwarded = ["install", "--yes"]
        if self.variant_override:
            forwarded.extend(("--variant", self.variant_override))
        for source in self.sources or configured_sources():
            forwarded.extend(("--source", source))
        return updater_main(forwarded)

    def install_launcher_update(self) -> int:
        forwarded = ["launcher-install", "--yes"]
        for source in self.sources or configured_sources():
            forwarded.extend(("--source", source))
        return updater_main(forwarded)

    def recover_launcher(self) -> int:
        return updater_main(["launcher-recover"])

    def create_diagnostics(self) -> int:
        forwarded = ["diagnostics"]
        report = self.diagnose_app()
        if report is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = default_state_dir() / "diagnostics" / f"qlh-diagnose-{stamp}.json"
            write_diagnosis_report(report, path, bundle_safe=True)
            forwarded.extend(("--diagnose-report", str(path)))
        return updater_main(forwarded)


def run_tui(controller: LauncherController | None = None) -> int:
    controller = controller or LauncherController()
    if not _ensure_windows_console():
        print("当前环境无法创建交互终端，请使用 --gui。", file=sys.stderr)
        return 2
    while True:
        print("\n" + "=" * 56)
        print("  QLH · BJTU Launcher")
        print(f"  Launcher {LAUNCHER_VERSION} | {platform.system()} {platform.machine()}")
        app_root = installed_app_root(controller.variant_override)
        app_version = detect_current_version(
            app_root, fallback="unknown",
        ) if app_root else "not installed"
        print(f"  Application {app_version}")
        print("-" * 56)
        print("  [1] 启动普通界面")
        print("  [2] 启动管理 TUI")
        print("  [3] 检查更新")
        print("  [4] 下载并启动更新安装包")
        selected_variant = controller.variant_override or detect_profile()["variant"]
        print(f"  [5] 切换安装包类型（当前: {selected_variant}）")
        print("  [6] 检查 Launcher 自更新")
        print("  [7] 更新 Launcher")
        print("  [8] 恢复上一个 Launcher 槽")
        print("  [9] 运行完整性诊断")
        print("  [10] 导出脱敏诊断包")
        print("  [Q] 退出")
        choice = input("请选择: ").strip().lower()
        if choice in {"1", "ui", "gui"}:
            code = controller.start_gui()
            if code:
                print(f"启动失败: {controller.last_error}")
                continue
            return 0
        if choice in {"2", "tui"}:
            code = controller.start_tui()
            if code:
                print(f"启动失败: {controller.last_error}")
                continue
            return 0
        if choice in {"3", "update", "check"}:
            print(_render_update_result(controller.check_update()))
            continue
        if choice in {"4", "install"}:
            result = controller.check_update()
            if result.get("signature_verified"):
                prompt = "更新包会校验大小、SHA-256 与发布签名。继续？ [y/N] "
            else:
                prompt = (
                    "更新包会校验大小和 SHA-256，但发布签名未验证"
                    "（或清单未签名），安装会被拒绝。继续？ [y/N] "
                )
            answer = input(prompt).strip().lower()
            if answer in {"y", "yes"}:
                return controller.install_update()
            continue
        if choice in {"5", "variant"}:
            value = input("请选择 cpu（集显/通用）或 cuda（NVIDIA 独显）: ").strip().lower()
            if value in {"cpu", "cuda"}:
                controller.set_variant(value)
                print(f"已选择 {value} 安装包。")
            else:
                print("无效选择。")
            continue
        if choice in {"6", "launcher-check"}:
            print(_render_update_result(controller.check_launcher_update()))
            continue
        if choice in {"7", "launcher-install"}:
            result = controller.check_launcher_update()
            if result.get("update_available") and result.get("asset"):
                answer = input("确认下载并更新 Launcher？ [y/N] ").strip().lower()
                if answer in {"y", "yes"}:
                    code = controller.install_launcher_update()
                    print("Launcher 更新完成。" if code == 0 else f"Launcher 更新失败（退出码 {code}）。")
            else:
                print(_render_update_result(result))
            continue
        if choice in {"8", "launcher-recover"}:
            code = controller.recover_launcher()
            print("Launcher 已恢复。" if code == 0 else f"Launcher 恢复失败（退出码 {code}）。")
            continue
        if choice in {"9", "diagnose"}:
            report = controller.diagnose_app()
            print(format_diagnosis(report) if report else "未检测到可诊断的 QLH 主程序安装目录。")
            continue
        if choice in {"10", "diagnostics"}:
            code = controller.create_diagnostics()
            print("诊断包已生成。" if code == 0 else f"诊断包生成失败（退出码 {code}）。")
            continue
        if choice in {"q", "quit", "exit"}:
            return 0


def run_gui(controller: LauncherController | None = None) -> int:
    controller = controller or LauncherController()
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        print(f"GUI 不可用: {exc}; 回退到 TUI。", file=sys.stderr)
        return run_tui(controller) if sys.stdin and sys.stdin.isatty() else controller.start_gui()

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"GUI 不可用: {exc}; 启动普通应用界面。", file=sys.stderr)
        return controller.start_gui()
    root.title("QLH · BJTU Launcher")
    root.geometry("660x540")
    root.minsize(600, 500)
    root.configure(bg="#0c0b0b")
    for icon_path in (_PACKAGING_DIR / "leds.ico", install_root() / "leds.ico"):
        if icon_path.is_file():
            try:
                root.iconbitmap(str(icon_path))
                break
            except tk.TclError:
                pass
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Launcher.TButton", padding=(14, 9), font=("Segoe UI", 11))
    style.configure("Launcher.TFrame", background="#0c0b0b")
    style.configure("Launcher.TLabel", background="#0c0b0b", foreground="#f5f5f5")
    style.configure("Launcher.Title.TLabel", background="#0c0b0b", foreground="#ffffff", font=("Segoe UI", 21, "bold"))

    frame = tk.Frame(root, bg="#0c0b0b", padx=34, pady=28)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="QLH 边缘推理系统", style="Launcher.Title.TLabel").pack(anchor="w")
    ttk.Label(
        frame, text=f"BJTU Launcher {LAUNCHER_VERSION} · GUI / TUI / 更新与修复",
        style="Launcher.TLabel",
    ).pack(anchor="w", pady=(5, 24))
    initial_variant = controller.variant_override or detect_profile()["variant"]
    app_root = installed_app_root(initial_variant)
    app_version = (
        detect_current_version(app_root, fallback="unknown") if app_root else "未安装"
    )
    status = tk.StringVar(value=f"当前应用：{app_version}。选择启动方式，或检查更新。")
    ttk.Label(frame, textvariable=status, style="Launcher.TLabel", wraplength=540).pack(anchor="w", pady=(0, 18))
    profile_row = ttk.Frame(frame, style="Launcher.TFrame")
    profile_row.pack(fill="x", pady=(0, 14))
    ttk.Label(profile_row, text="更新包类型", style="Launcher.TLabel").pack(side="left")
    variant = tk.StringVar(value=initial_variant)
    variant_box = ttk.Combobox(
        profile_row, textvariable=variant, values=("cpu", "cuda"),
        width=9, state="readonly",
    )
    variant_box.pack(side="left", padx=(10, 0))
    ttk.Label(
        profile_row, text="cpu = 集显/通用，cuda = NVIDIA 独显",
        style="Launcher.TLabel",
    ).pack(side="left", padx=(12, 0))

    def variant_changed(_event=None) -> None:
        controller.set_variant(variant.get())
        install_button.state(["disabled"])
        status.set("安装包类型已变更，请重新检查更新。")

    variant_box.bind("<<ComboboxSelected>>", variant_changed)
    controller.set_variant(initial_variant)
    actions = ttk.Frame(frame, style="Launcher.TFrame")
    actions.pack(fill="x")
    actions.columnconfigure(0, weight=1)
    actions.columnconfigure(1, weight=1)
    def start(mode: str) -> None:
        code = controller.start_gui() if mode == "ui" else controller.start_tui()
        if code:
            status.set(f"启动失败: {controller.last_error}")
            messagebox.showerror("启动失败", controller.last_error, parent=root)

    ttk.Button(
        actions, text="启动普通界面", style="Launcher.TButton",
        command=lambda: start("ui"),
    ).grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 8))
    ttk.Button(
        actions, text="启动管理 TUI", style="Launcher.TButton",
        command=lambda: start("tui"),
    ).grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 8))

    install_button = ttk.Button(actions, text="下载并安装", style="Launcher.TButton")
    install_button.grid(row=1, column=1, sticky="ew", padx=(6, 0))
    install_button.state(["disabled"])

    def check() -> None:
        status.set("正在检查更新...")
        install_button.state(["disabled"])

        def worker() -> None:
            result = controller.check_update()
            def apply_result() -> None:
                status.set(_render_update_result(result))
                if result.get("update_available") and result.get("asset"):
                    install_button.state(["!disabled"])
            root.after(0, apply_result)

        threading.Thread(target=worker, daemon=True, name="launcher-update-check").start()

    ttk.Button(
        actions, text="检查更新", style="Launcher.TButton", command=check,
    ).grid(row=1, column=0, sticky="ew", padx=(0, 6))

    launcher_install_button = ttk.Button(
        actions, text="更新 Launcher", style="Launcher.TButton",
    )
    launcher_install_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=(8, 0))
    launcher_install_button.state(["disabled"])

    def check_launcher() -> None:
        status.set("正在检查 Launcher 更新...")
        launcher_install_button.state(["disabled"])

        def worker() -> None:
            result = controller.check_launcher_update()

            def apply_result() -> None:
                status.set(_render_update_result(result))
                if result.get("update_available") and result.get("asset"):
                    launcher_install_button.state(["!disabled"])

            root.after(0, apply_result)

        threading.Thread(target=worker, daemon=True, name="launcher-self-update-check").start()

    ttk.Button(
        actions, text="检查 Launcher", style="Launcher.TButton", command=check_launcher,
    ).grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=(8, 0))

    def install_launcher() -> None:
        if not messagebox.askyesno(
            "确认更新 Launcher",
            "将下载并校验新的 Launcher，验证通过后切换活动槽。继续吗？",
            parent=root,
        ):
            return
        launcher_install_button.state(["disabled"])
        status.set("正在下载、校验并切换 Launcher...")

        def worker() -> None:
            code = controller.install_launcher_update()
            root.after(0, lambda: status.set(
                "Launcher 更新完成，下次启动生效。" if code == 0
                else f"Launcher 更新失败（退出码 {code}）。"
            ))

        threading.Thread(target=worker, daemon=True, name="launcher-self-update-install").start()

    launcher_install_button.configure(command=install_launcher)

    def repair_launcher() -> None:
        code = controller.recover_launcher()
        status.set("Launcher 已恢复上一个可用槽。" if code == 0 else f"恢复失败（退出码 {code}）。")

    def export_diagnostics() -> None:
        code = controller.create_diagnostics()
        status.set("诊断包已生成，请查看状态目录。" if code == 0 else f"诊断包生成失败（退出码 {code}）。")

    def diagnose() -> None:
        report = controller.diagnose_app()
        if report is None:
            status.set("未检测到可诊断的 QLH 主程序安装目录。")
            return
        text = format_diagnosis(report)
        status.set(text)
        messagebox.showinfo("完整性诊断", text, parent=root)

    tools_row = ttk.Frame(frame, style="Launcher.TFrame")
    tools_row.pack(fill="x", pady=(12, 0))
    ttk.Button(
        tools_row, text="恢复 Launcher", style="Launcher.TButton", command=repair_launcher,
    ).pack(side="left")
    ttk.Button(
        tools_row, text="运行诊断", style="Launcher.TButton", command=diagnose,
    ).pack(side="left", padx=(10, 0))
    ttk.Button(
        tools_row, text="导出诊断包", style="Launcher.TButton", command=export_diagnostics,
    ).pack(side="left", padx=(10, 0))

    def install() -> None:
        if not messagebox.askyesno(
            "确认更新",
            "更新包会校验大小、SHA-256 与发布签名。\n"
            "确认下载并启动系统安装器吗？",
            parent=root,
        ):
            return
        install_button.state(["disabled"])
        status.set("正在下载并校验更新包...")

        def worker() -> None:
            code = controller.install_update()
            root.after(0, lambda: status.set(
                "安装器已启动。" if code == 0 else f"更新失败（退出码 {code}）。"
            ))

        threading.Thread(target=worker, daemon=True, name="launcher-update-install").start()

    install_button.configure(command=install)
    ttk.Button(frame, text="退出", command=root.destroy).pack(anchor="e", pady=(24, 0))
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlh-launcher",
        description="Independent QLH GUI/TUI bootstrap and package updater.",
    )
    parser.add_argument(
        "command", nargs="?",
        choices=(
            "gui", "tui", "app-ui", "app-tui", "check", "download", "install",
            "version-status", "version-stage", "version-activate",
            "version-rollback", "version-recover",
            "launcher-status", "launcher-check", "launcher-download",
            "launcher-install", "launcher-stage", "launcher-activate",
            "launcher-rollback", "launcher-recover", "diagnostics", "verify", "diagnose",
        ),
    )
    parser.add_argument("--gui", action="store_true", help="open the Launcher GUI")
    parser.add_argument("--tui", action="store_true", help="open the Launcher TUI")
    parser.add_argument("--check-update", action="store_true", help="alias for check")
    parser.add_argument("--source", action="append", help="manifest URL; may be repeated")
    parser.add_argument("--variant", choices=("cpu", "cuda"), help="application package variant")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--download-dir")
    parser.add_argument("--version-store")
    parser.add_argument("--launcher-store")
    parser.add_argument("--bundle")
    parser.add_argument("--version")
    parser.add_argument("--diagnostics-output")
    parser.add_argument("--diagnose-report", help="bundle-safe diagnosis JSON for diagnostics")
    parser.add_argument("--trusted-keys-dir")
    parser.add_argument("--root", help="install root for the verify command")
    parser.add_argument("--level", choices=("quick", "full", "deep"), default="quick")
    parser.add_argument("--error", help="optional local startup error text for diagnose")
    parser.add_argument("--diagnosis-output", help="local JSON output path for diagnose")
    parser.add_argument("--no-gpu-probe", action="store_true", help="skip the bounded CUDA driver probe")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-unsigned", action="store_true")
    parser.add_argument("--headless", action="store_true", help="run the application backend")
    parser.add_argument("--app-ui", action="store_true", help="start the application UI directly")
    parser.add_argument("--app-tui", action="store_true", help="start the management TUI directly")
    parser.add_argument("--health-check", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.health_check:
        return _self_health_check()
    if args.command not in {
        "check", "download", "install", "version-status", "version-stage",
        "version-activate", "version-rollback", "version-recover",
        "launcher-status", "launcher-check", "launcher-download", "launcher-install",
        "launcher-stage", "launcher-activate", "launcher-rollback", "launcher-recover",
        "diagnostics", "verify", "diagnose",
    } and not args.check_update:
        delegated = _delegate_to_active_launcher(list(sys.argv[1:] if argv is None else argv))
        if delegated is not None:
            return delegated
    if args.command == "verify":
        root = Path(args.root).expanduser() if args.root else installed_app_root(args.variant)
        if root is None:
            print("未检测到带签名安装清单的 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        forwarded = ["verify", "--root", str(root), "--level", args.level]
        if args.trusted_keys_dir:
            forwarded.extend(("--trusted-keys-dir", args.trusted_keys_dir))
        if args.as_json:
            forwarded.append("--json")
        return install_manifest_main(forwarded)
    if args.command == "diagnose":
        root = Path(args.root).expanduser() if args.root else diagnosis_app_root(args.variant)
        if root is None:
            print("未检测到 QLH 主程序；请使用 --root 指定安装目录。", file=sys.stderr)
            return 2
        report = diagnose_install(
            root,
            error=args.error,
            trusted_keys_dir=args.trusted_keys_dir,
            probe_gpu=not args.no_gpu_probe,
        )
        if args.diagnosis_output:
            write_diagnosis_report(report, args.diagnosis_output)
        if args.as_json:
            import json

            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(format_diagnosis(report))
        return 0 if report["ok"] else 3
    if args.command and args.command.startswith("version-"):
        forwarded = [args.command]
        for name in ("version_store", "bundle", "version", "variant"):
            value = getattr(args, name, None)
            if value:
                forwarded.extend((f"--{name.replace('_', '-')}", value))
        if args.as_json:
            forwarded.append("--json")
        return updater_main(forwarded)
    if args.command and (args.command.startswith("launcher-") or args.command == "diagnostics"):
        forwarded = [args.command]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        for name in ("launcher_store", "bundle", "version", "download_dir", "variant", "diagnostics_output", "diagnose_report", "trusted_keys_dir"):
            value = getattr(args, name, None)
            if value:
                forwarded.extend((f"--{name.replace('_', '-')}", str(value)))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        for flag in ("as_json", "yes", "allow_unsigned"):
            if getattr(args, flag):
                forwarded.append("--" + ("json" if flag == "as_json" else flag.replace("_", "-")))
        return updater_main(forwarded)
    if args.check_update or args.command == "check":
        forwarded = ["check"]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        if args.as_json:
            forwarded.append("--json")
        if args.variant:
            forwarded.extend(("--variant", args.variant))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        return updater_main(forwarded)
    if args.command in {"download", "install"}:
        forwarded = [args.command]
        for source in args.source or []:
            forwarded.extend(("--source", source))
        if args.variant:
            forwarded.extend(("--variant", args.variant))
        if args.download_dir:
            forwarded.extend(("--download-dir", args.download_dir))
        if args.timeout != 8.0:
            forwarded.extend(("--timeout", str(args.timeout)))
        for flag in ("as_json", "yes", "allow_unsigned"):
            if getattr(args, flag):
                forwarded.append("--" + ("json" if flag == "as_json" else flag.replace("_", "-")))
        return updater_main(forwarded)
    controller = LauncherController(args.source, args.variant)
    if args.app_ui or args.command == "app-ui":
        return controller.start_gui()
    if args.app_tui or args.command == "app-tui":
        return controller.start_tui()
    if args.headless:
        process = launch_app("headless", args.variant)
        return process.wait() if os.name != "nt" else 0
    if args.tui or args.command == "tui":
        return run_tui(controller)
    if args.gui or args.command == "gui":
        return run_gui(controller)
    return run_gui(controller) if os.name == "nt" else run_tui(controller)


if __name__ == "__main__":
    raise SystemExit(main())
