#!/usr/bin/env bash
# ============================================================
#  QLH 全局 bjtu 命令 (Linux / macOS)
#
#  用法: bjtu [launcher|ui|tui|chat|tui_admin.py 参数...]
#
#  一键启动: 自动启动后端(若未运行) -> 等待就绪 -> 进入 TUI。
#  退出 TUI 后后端继续运行; 停止后端:
#      kill "$(cat logs/backend_tui.pid)"
#
#  安装(任意一种):
#     ln -s <项目根>/bjtu.sh /usr/local/bin/bjtu
#     或把 <项目根> 加入 PATH
#  本脚本支持被符号链接调用, 会解析回项目根目录。
# ============================================================

# 解析脚本真实路径(兼容符号链接; macOS 无 readlink -f 时退回 $0)
SELF="$0"
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
    SELF="$(readlink -f "$0")"
fi
PROJECT_ROOT="$(cd "$(dirname "$SELF")" && pwd)" || exit 1
cd "$PROJECT_ROOT" || exit 1

if [ ! -f "src/api_server.py" ]; then
    echo "[错误] 未找到 src/api_server.py —— bjtu 脚本必须位于 QLH 项目根目录。"
    echo "       当前目录: $PROJECT_ROOT"
    exit 1
fi

# ---- help: 仅打印命令集与参数帮助，不启动后端 ----
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "QLH BJTU 统一入口:"
    echo "  bjtu launcher   启动选择页（普通界面 / TUI）"
    echo "  bjtu ui         直接启动普通 Web/原生界面"
    echo "  bjtu tui        启动后端并进入 TUI 管理界面"
    echo "  bjtu chat       进入终端对话页"
    echo "  bjtu status     执行 TUI 单命令（不自动启动后端）"
    echo
    PY=""
    if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
        PY=python3
    elif command -v python >/dev/null 2>&1; then
        PY=python
    else
        echo "[错误] 未找到可用的 python3 / python。"
        exit 1
    fi
    exec "$PY" "$PROJECT_ROOT/src/tui_admin.py" --help
fi

# ---- unified launcher modes ----
if [ "$1" = "launcher" ] || [ "$1" = "ui" ] || [ "$1" = "tui" ]; then
    MODE="$1"
    shift
    PY=""
    if [ -x "$PROJECT_ROOT/.venv-packaging/bin/python" ]; then
        PY="$PROJECT_ROOT/.venv-packaging/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PY=python3
    else
        PY=python
    fi
    export PYTHONIOENCODING=utf-8
    if [ "$MODE" = "launcher" ]; then
        exec "$PY" "$PROJECT_ROOT/packaging/launcher.py" --launcher "$@"
    elif [ "$MODE" = "ui" ]; then
        exec "$PY" "$PROJECT_ROOT/packaging/launcher.py" --ui "$@"
    else
        exec "$PY" "$PROJECT_ROOT/packaging/launcher.py" --tui "$@"
    fi
fi

# ---- chat: T9 简化聊天页（可选依赖 Textual/httpx）----
if [ "$1" = "chat" ]; then
    PY=""
    if [ -x "$PROJECT_ROOT/.venv-tui/bin/python" ]; then
        PY="$PROJECT_ROOT/.venv-tui/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PY=python3
    else
        PY=python
    fi
    if ! "$PY" -c "import textual, httpx" >/dev/null 2>&1; then
        echo "[T9] 聊天页缺少可选依赖 Textual/httpx。"
        echo "     安装: python3 scripts/setup_tui_env.py"
        echo "     或:   pip install -r packaging/requirements-tui.txt"
        echo "     管理 TUI（bjtu / start_tui.sh）不受影响。"
        exit 2
    fi
    shift
    export PYTHONIOENCODING=utf-8
    exec "$PY" "$PROJECT_ROOT/src/tui_chat.py" "$@"
fi

exec bash start_tui.sh "$@"
