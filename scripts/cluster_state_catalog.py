#!/usr/bin/env python
"""Emit the reviewed P4.5 state catalog without reading user state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cluster_state_catalog import build_state_catalog, validate_state_catalog  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the P4.5 cluster state catalog")
    parser.add_argument("--out", default="-", help="JSON output path, or '-' for stdout")
    args = parser.parse_args(argv)
    catalog = build_state_catalog()
    payload = validate_state_catalog(catalog).to_dict()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(rendered)
    else:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
