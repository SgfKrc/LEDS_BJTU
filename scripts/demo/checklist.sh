#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT"
if [ -x ".venv-test/bin/python" ]; then
  PYTHON_CMD=".venv-test/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_CMD=".venv/bin/python"
else
  PYTHON_CMD="python3"
fi
exec "$PYTHON_CMD" scripts/demo/defense_preflight.py "$@"
