#!/usr/bin/env bash
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
    echo "[error] Python 3 is required" >&2
    exit 1
fi
exec "$PY" "$ROOT/qlh.py" "$@"
