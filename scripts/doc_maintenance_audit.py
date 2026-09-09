#!/usr/bin/env python3
"""Stable main-project entrypoint for the standalone docagent core."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCAGENT_DIR = REPO_ROOT / "tools" / "docagent"
sys.path.insert(0, str(DOCAGENT_DIR))

from docagent.compat import run_compat  # noqa: E402


_EXTENSION_FLAGS = frozenset({
    "--llm", "--provider", "--cost", "--apply", "--index", "--rebuild",
    "--index-chunks", "--search", "--search-limit", "--embed",
    "--semantic-search", "--embedding-model",
})


def main(argv: list[str] | None = None) -> int:
    """Delegate M1 flags to docagent; preserve M2/M3 flags during migration."""
    effective = list(sys.argv[1:] if argv is None else argv)
    if any(argument.split("=", 1)[0] in _EXTENSION_FLAGS for argument in effective):
        legacy_dir = REPO_ROOT / "docs" / "agent_tool"
        sys.path.insert(0, str(legacy_dir))
        from doc_maintenance_audit import main as legacy_main

        return int(legacy_main(effective))
    return int(run_compat(effective, REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
