#!/usr/bin/env bash
# Kllama —— 项目当前名称；入口脚本仍是 `qlh.py`。
# 本文件只是别名薄壳：行为与 qlh.sh 完全一致，参数原样透传。
set -eu

SELF="$0"
if command -v readlink >/dev/null 2>&1 && readlink -f "$0" >/dev/null 2>&1; then
    SELF="$(readlink -f "$0")"
fi
ROOT="$(cd "$(dirname "$SELF")" && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "kllama: 未找到 python3 / python" >&2
    exit 127
fi

exec "$PY" qlh.py "$@"
