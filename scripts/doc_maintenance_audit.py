#!/usr/bin/env python3
"""设计文档约定的文档维护工具入口。"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = REPO_ROOT / "docs" / "agent_tool"
sys.path.insert(0, str(TOOL_DIR))

from doc_maintenance_audit import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
