#!/usr/bin/env bash
# ============================================================
#  QLH 全局 bjtu 命令 (Linux / macOS)
#
#  用法: bjtu [tui_admin.py 参数...]
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

exec bash start_tui.sh "$@"
