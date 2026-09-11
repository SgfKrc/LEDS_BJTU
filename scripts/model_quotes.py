"""Zero-dependency entry point for the FUN-CLI-01 ``model_quotes`` demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness_workbench.tools.fun_cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(["quotes", *sys.argv[1:]]))
