#!/usr/bin/env bash
# QLH TUI 管理菜单启动脚本 (Linux / macOS)
cd "$(dirname "$0")" || exit 1

if command -v python3 >/dev/null 2>&1; then
    exec python3 src/tui_admin.py "$@"
else
    exec python src/tui_admin.py "$@"
fi
