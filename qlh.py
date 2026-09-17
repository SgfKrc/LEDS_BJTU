"""Cross-platform command entry for the unified core TUI."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _usage() -> str:
    return (
        "QLH core TUI\n"
        "  qlh                         进入统一 TUI（本机后端自动启动）\n"
        "  qlh [options]               同上；选项直通统一入口（--port/--no-splash/--plain/--route…）\n"
        "  qlh chat [--host URL] [--route auto|local_only|distributed_preferred|distributed_required]\n"
        "  qlh chat --fixture PATH     离线回放零依赖聊天屏 fixture\n"
        "  qlh admin [tui_admin.py options]\n"
        "  qlh status\n"
        "  qlh models\n"
    )


def _python() -> str:
    return sys.executable or os.environ.get("PYTHON", "python")


def _load_tui_admin():
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import tui_admin

    return tui_admin


def _run_unified(args: list[str]) -> int:
    return int(_load_tui_admin().main(args))


def _run(module_script: str, args: list[str]) -> int:
    """保留旧单命令/fixture 子进程语义，统一交互入口不经过这里。"""
    completed = subprocess.run(
        [_python(), str(ROOT / module_script), *args],
        cwd=str(ROOT),
    )
    return int(completed.returncode)


def _chat_args(args: list[str]) -> list[str]:
    """Translate the URL-shaped chat CLI into the admin TUI's host/port form."""
    translated = ["--auto-start", "--screen", "chat"]
    i = 0
    while i < len(args):
        value = args[i]
        if value == "--fixture":
            # Fixture mode is intentionally independent of the backend.
            return ["--fixture", args[i + 1]]
        if value in {"--host", "--route", "--timeout", "--port"}:
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
                "qlh chat [--host URL] [--port PORT] "
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
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from tui_chat_screen import smoke_from_fixture

    return int(smoke_from_fixture(path))


if __name__ == "__main__":
    raise SystemExit(main())
