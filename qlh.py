"""Cross-platform command entry for the core TUI.

The core keeps the chat shell and the zero-dependency admin TUI as separate
programs. This dispatcher gives both a stable ``qlh`` command without
pulling product-shell dependencies into the main repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _usage() -> str:
    return (
        "QLH core TUI\n"
        "  qlh chat [--host URL|--fixture PATH] [--route auto|local_only|cluster]\n"
        "  qlh admin [tui_admin.py options]\n"
        "  qlh status\n"
        "  qlh models\n"
    )


def _python() -> str:
    return sys.executable or os.environ.get("PYTHON", "python")


def _run(module_script: str, args: list[str]) -> int:
    completed = subprocess.run(
        [_python(), str(ROOT / module_script), *args],
        cwd=str(ROOT),
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0

    command = args.pop(0).lower()
    if command == "chat":
        return _run("src/tui_chat.py", args)
    if command == "admin":
        return _run("src/tui_admin.py", args)
    if command in {"status", "models"}:
        return _run("src/tui_admin.py", [command, *args])
    print("unknown qlh command: %s\n\n%s" % (command, _usage()))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
