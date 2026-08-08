#!/usr/bin/env python3
"""QLH 全自动优化实验调度器入口（EX-N1）。

用法：
    python scripts/run_experiments.py --plan fixtures/experiment-plan.example.json
    python scripts/run_experiments.py --plan plan.json --parallel 2 --resume
    python scripts/run_experiments.py --plan plan.json --check
"""

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from experiment_core.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
