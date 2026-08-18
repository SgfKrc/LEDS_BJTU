#!/usr/bin/env bash
# ============================================================
#  QLH 一键环境配置 (Linux / macOS)
#
#  创建/安装 主运行时 + 全部虚拟环境（.venv-*）依赖，
#  以及 frontend / gateway / control 三个 Node 子项目。
#  等价于: python3 scripts/setup_envs.py --all
#
#  常用变体：
#    ./setup_all_envs.sh --no-node                仅 Python 环境
#    ./setup_all_envs.sh --only test,tui          只配指定环境
#    ./setup_all_envs.sh --check                  只校验不安装
#    ./setup_all_envs.sh --torch-index-url https://download.pytorch.org/whl/cu126
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8

# 注意：部分系统 python3 可能是 MSYS2/Store 假别名（无 pip/无法建 venv），需实际验证
PY=""
if command -v python3 >/dev/null 2>&1 \
    && python3 -c "import ensurepip" >/dev/null 2>&1 \
    && python3 -m pip --version >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1 \
    && python -c "import ensurepip" >/dev/null 2>&1 \
    && python -m pip --version >/dev/null 2>&1; then
    PY=python
else
    echo "[错误] 未找到带 pip 的 python3 / python（可创建 venv），请安装 Python 3.10+ 并加入 PATH。"
    exit 1
fi

echo "使用解释器: $PY"
exec "$PY" scripts/setup_envs.py --all "$@"
