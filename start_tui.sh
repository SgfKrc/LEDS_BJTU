#!/usr/bin/env bash
# ============================================================
#  QLH TUI 一键启动（Linux / macOS / Git Bash）
#
#  交互模式统一走 `qlh` 新入口：
#    1. 后端在**当前进程内**启动（BackendSupervisor），不再另开后端进程/窗口；
#    2. 冷启动由 Koakuma splash 承载（动画 + 实时状态行），
#       不再出现「静默轮询 /api/health 最长 120s」的黑屏；
#    3. 后端随 TUI 退出而停止（需要常驻请直接运行 src/api_server.py）。
#
#  单命令模式：`start_tui.sh status` 执行一条 TUI 命令后退出（不启动后端）。
# ============================================================
set -eu
cd "$(dirname "$0")" || exit 1

# 注意：Windows Git Bash 下 command -v python3 可能命中 Microsoft Store
# 假别名（WindowsApps/python3），必须实际运行验证；不可用时回退 python。
PY=""
if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[错误] 未找到可用的 python3 / python，请先安装 Python 3.10+。"
    exit 1
fi

echo "============================================"
echo "  QLH 分布式边缘推理 — TUI 管理菜单"
echo "============================================"
echo

# ---- 单命令模式：直接执行一条命令后退出（不启动后端）----
#      命令名/别名清单与 src/tui_admin.py 的 COMMANDS 注册表保持一致。
#      注意：命令必须是第一个参数（start_tui.sh status --port 9000）；
#      选项在前（start_tui.sh --port 9000 status）会回退为交互模式。
FIRST_ARG="${1:-}"
case "$FIRST_ARG" in
    /*|help|h|quit|q|exit|shutdown|halt|status|st|screen|goto|refresh|r|model|models|switch|load|quant|engine|presets|gpu|device|nodes|connect|join|dist|queue|logs|log|host|interval|timeout|token|chat|new|sessions|resume|rename|delete-session|route|thinking|cancel)
        exec "$PY" src/tui_admin.py "$@"
        ;;
esac

# ---- 交互模式：统一入口（进程内后端 + splash，无静默等待）----
PORT_ARGS=""
if [ -n "${QLH_BACKEND_PORT:-}" ]; then
    PORT_ARGS="--port $QLH_BACKEND_PORT"
fi
# shellcheck disable=SC2086  # PORT_ARGS 需按空格拆分（可能为空）
exec "$PY" qlh.py $PORT_ARGS "$@"
