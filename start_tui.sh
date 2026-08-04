#!/usr/bin/env bash
# ============================================================
#  QLH TUI 管理菜单 — 一键启动 (Linux / macOS)
#
#  自动完成:
#    1. 检查 8000 端口后端是否已运行（QLH_BACKEND_PORT 可覆盖）
#    2. 未运行则后台启动 API 服务器 (src/api_server.py)
#       → 日志 logs/backend_tui.log，PID 存 logs/backend_tui.pid
#    3. 等待 /api/health 就绪后进入 TUI 管理菜单
#
#  退出 TUI 后后端继续运行；停止后端:
#      kill "$(cat logs/backend_tui.pid)"
# ============================================================
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
BACKEND_PORT="${QLH_BACKEND_PORT:-8000}"

echo "============================================"
echo "  QLH 分布式边缘推理 — TUI 管理菜单"
echo "============================================"
echo

# ---- [1/3] 检查后端是否已运行 ----
if "$PY" - "$BACKEND_PORT" <<'EOF'
import socket, sys
s = socket.socket()
try:
    s.settimeout(2)
    s.connect(("127.0.0.1", int(sys.argv[1])))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
then
    echo "[1/3] 检测到后端已在端口 $BACKEND_PORT 运行，跳过启动。"
else
    echo "[1/3] 后端未运行，正在后台启动 API 服务器 (port $BACKEND_PORT) ..."
    mkdir -p logs
    # 用 python src/api_server.py 启动（而非 -m uvicorn），使
    # POST /api/system/shutdown 能触发跨平台优雅退出（资源清理）。
    QLH_BACKEND_PORT="$BACKEND_PORT" nohup "$PY" src/api_server.py \
        >> logs/backend_tui.log 2>&1 &
    echo $! > logs/backend_tui.pid
fi

# ---- [2/3] 等待后端就绪（最多 120 秒）----
echo "[2/3] 等待后端就绪（/api/health）..."
TRIES=0
until "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$BACKEND_PORT/api/health', timeout=2); sys.exit(0)" >/dev/null 2>&1; do
    TRIES=$((TRIES + 1))
    if [ "$TRIES" -ge 60 ]; then
        echo
        echo "[提示] 后端在 120 秒内未就绪。请检查 logs/backend_tui.log："
        echo "      端口占用、Python 环境、依赖缺失等均会导致启动失败。"
        exit 1
    fi
    sleep 2
done
echo "      后端已就绪。"

# ---- [3/3] 启动 TUI ----
echo "[3/3] 启动 TUI 管理菜单 ..."
echo "      退出 TUI 后后端继续运行（停止后端: kill \"\$(cat logs/backend_tui.pid)\"）。"
echo
exec "$PY" src/tui_admin.py "$@"
