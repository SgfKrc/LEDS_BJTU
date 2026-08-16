#!/usr/bin/env python3
"""QLH model-tools CLI entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_utf8_stdio() -> None:
    """Keep CLI pipes decodable on Windows regardless of the console code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="strict")
        except (OSError, ValueError):
            # Embedded callers and test capture streams may not support reconfigure.
            continue


if __name__ == "__main__":
    _configure_utf8_stdio()

from scripts.model_tools.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
