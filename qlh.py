"""Cross-platform command entry for the unified QLH TUI.

2026-09-17 起：交互入口默认使用 **Textual 外壳**（``src/tui_textual.py``），
因为它自己承担终端适配；自绘 ANSI（``src/tui_admin.py`` + ``tui_splash.py``）在真实
conhost 下实测不可见（见该文件头注释与 ``docs/TUI重写方案``）。

* ``qlh`` / ``qlh chat`` / ``qlh --port N``  -> Textual 外壳（含本机后端冷启动反馈）
* ``qlh --tui-engine builtin``               -> 标准库 TUI（过渡期回退入口）
* ``qlh status`` / ``qlh models`` / ``qlh admin`` -> 仍走标准库薄层（无需 UI 依赖）
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _usage() -> str:
    return (
        "QLH core TUI\n"
        "  qlh                         进入统一 TUI（Textual 外壳；本机后端自动启动）\n"
        "  qlh [options]               同上；选项直通（--port/--route/--host/--thinking…）\n"
        "  qlh chat [--host URL] [--route auto|local_only|distributed_preferred|distributed_required]\n"
        "  qlh chat --fixture PATH     离线回放聊天屏 fixture\n"
        "  qlh --tui-engine builtin    过渡期：回退标准库 TUI\n"
        "  qlh admin [tui_admin.py options]\n"
        "  qlh status / qlh models     单命令模式（不起后端、无 UI 依赖）\n"
    )


def _python() -> str:
    return sys.executable or os.environ.get("PYTHON", "python")


def _ensure_src_on_path() -> str:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def _load_tui_admin():
    _ensure_src_on_path()
    import tui_admin

    return tui_admin


def _textual_available() -> bool:
    try:
        return importlib.util.find_spec("textual") is not None
    except Exception:  # noqa: BLE001 - 探测失败按不可用处理
        return False


def _option_value(args: list[str], name: str, default: str) -> str:
    if name in args:
        index = args.index(name) + 1
        if index < len(args):
            return args[index]
    return default


def _run(module_script: str, args: list[str]) -> int:
    """保留旧单命令/fixture 子进程语义，统一交互入口不经过这里。"""
    completed = subprocess.run(
        [_python(), str(ROOT / module_script), *args],
        cwd=str(ROOT),
    )
    return int(completed.returncode)


def _chat_args(args: list[str]) -> list[str]:
    """把 URL 形状的 chat CLI 翻译成 ``--host/--port`` 形式，并补默认值。"""
    translated = ["--auto-start", "--screen", "chat"]
    i = 0
    while i < len(args):
        value = args[i]
        if value == "--fixture":
            # Fixture mode is intentionally independent of the backend.
            return ["--fixture", args[i + 1]]
        if value in {"--host", "--route", "--timeout", "--port", "--interval", "--tui-engine"}:
            if i + 1 >= len(args):
                raise ValueError("%s 缺少参数" % value)
            item = args[i + 1]
            if value == "--host":
                parsed = urllib.parse.urlsplit(item if "://" in item else "http://" + item)
                if parsed.hostname:
                    translated.extend(["--host", parsed.hostname])
                if parsed.port:
                    translated.extend(["--port", str(parsed.port)])
            elif value == "--route":
                translated.extend(["--route", item])
            else:
                translated.extend([value, item])
            i += 2
            continue
        if value in {"--thinking", "--plain", "--no-splash", "--no-color", "--auto-start"}:
            translated.append(value)
            i += 1
            continue
        if value == "--screen":
            # 内部调用（_run_unified(["--auto-start", "--screen", "chat"])）也会经过这里
            if i + 1 >= len(args):
                raise ValueError("--screen 缺少参数")
            translated.extend(["--screen", args[i + 1]])
            i += 2
            continue
        if value in {"--thinking", "--plain", "--no-splash", "--no-color"}:
            translated.append(value)
            i += 1
            continue
        raise ValueError("chat 不支持参数: %s" % value)
    if "--host" in translated:
        host = translated[translated.index("--host") + 1].lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            translated.remove("--auto-start")
    return translated


def _shell_options(translated: list[str]) -> dict:
    def value_of(name: str, default):
        return translated[translated.index(name) + 1] if name in translated else default

    return {
        "host": value_of("--host", "127.0.0.1"),
        "port": int(value_of("--port", 8000)),
        "route": value_of("--route", "auto"),
        "interval": float(value_of("--interval", 5.0)),
        "thinking": "--thinking" in translated,
        "auto_start": "--auto-start" in translated,
    }


def _ensure_backend_visible(supervisor) -> None:
    """后端冷启动的**跨终端可靠**反馈：单行 CR 进度。

    刻意不用自绘 ANSI 动画/alt-screen：实测在传统 conhost 下不可见（见 tui_textual 头注释）。
    ``\\r`` 回列首在所有终端都有效，因此阶段状态一定可见。
    """
    box: dict = {}

    def _work() -> None:
        try:
            box["started"] = supervisor.ensure_ready()
        except BaseException as exc:  # noqa: BLE001 - 交回主线程
            box["error"] = exc

    thread = threading.Thread(target=_work, name="qlh-backend-start", daemon=True)
    thread.start()
    frames = "|/-\\"
    index = 0
    while thread.is_alive():
        sys.stdout.write(f"\r{frames[index % 4]} Koakuma: {supervisor.status_message} …  ")
        sys.stdout.flush()
        index += 1
        time.sleep(0.2)
    sys.stdout.write("\r" + " " * 72 + "\r")
    sys.stdout.flush()
    if box.get("error") is not None:
        raise box["error"]


def _run_textual_shell(args: list[str]) -> int:
    """统一入口：本机后端就绪（带可见进度）后进入 Textual 外壳。"""
    _ensure_src_on_path()
    translated = _chat_args(args)
    options = _shell_options(translated)

    from tui_backend import BackendStartupError, BackendSupervisor

    supervisor = BackendSupervisor(host=options["host"], port=options["port"])
    if options["auto_start"]:
        try:
            _ensure_backend_visible(supervisor)
        except BackendStartupError as exc:
            print("[错误] %s" % exc)
            return 1

    from tui_textual import run as run_textual

    return int(run_textual(
        options["host"],
        options["port"],
        interval=options["interval"],
        routing_preference=options["route"],
        show_thinking=options["thinking"],
    ))


def _run_unified(args: list[str]) -> int:
    """统一交互入口：默认 Textual 外壳；显式 builtin / 缺依赖时回退标准库 TUI。"""
    if _option_value(args, "--tui-engine", "auto") == "builtin":
        return int(_load_tui_admin().main(args))
    if not _textual_available():
        print("[提示] 未安装 textual（python -m pip install -r requirements-tui.txt）；"
              "本次回退标准库 TUI。")
        return int(_load_tui_admin().main(args))
    return _run_textual_shell(args)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _run_unified(["--auto-start", "--screen", "chat"])
    if args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0].startswith("-"):
        # 裸选项（`qlh --no-splash` / `qlh --port 9000`）等价 `qlh chat <options>`：
        # 让启动脚本可直接透传参数（start_tui.* 走的就是这条路）。
        # 未识别选项由 _chat_args 抛 ValueError -> 友好报错，不会误启动后端。
        try:
            return _run_unified(_chat_args(args))
        except ValueError as exc:
            print("[错误] %s" % exc)
            return 2

    command = args.pop(0).lower()
    if command == "chat":
        if args and args[0] in {"-h", "--help"}:
            print(
                "qlh chat [--host URL] [--port PORT] [--interval SECONDS] "
                "[--route auto|local_only|distributed_preferred|distributed_required] "
                "[--thinking]\n"
                "qlh chat --fixture PATH"
            )
            return 0
        try:
            if args and args[0] == "--fixture":
                if len(args) != 2:
                    raise ValueError("--fixture 需要一个路径")
                return _load_fixture_smoke(args[1])
            return _run_unified(_chat_args(args))
        except ValueError as exc:
            print("[错误] %s" % exc)
            return 2
    if command == "admin":
        return _run("src/tui_admin.py", args)
    if command in {"status", "models"}:
        return _run("src/tui_admin.py", [command, *args])
    if command in {"tui", "ui"}:
        return _run_unified(["--auto-start", "--screen", "chat", *args])
    print("unknown qlh command: %s\n\n%s" % (command, _usage()))
    return 2


def _load_fixture_smoke(path: str) -> int:
    _ensure_src_on_path()
    from tui_chat_screen import smoke_from_fixture

    return int(smoke_from_fixture(path))


if __name__ == "__main__":
    raise SystemExit(main())
