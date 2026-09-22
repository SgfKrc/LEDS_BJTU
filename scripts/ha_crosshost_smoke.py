"""Run the HA-CROSSHOST-01 local dual-process gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster_crosshost import CrossHostProcessHarness, write_crosshost_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local QLH HA cross-process smoke gate")
    parser.add_argument("--evidence", type=Path, help="optional metadata-only JSON output path")
    args = parser.parse_args()
    evidence = CrossHostProcessHarness().run()
    if args.evidence:
        write_crosshost_evidence(args.evidence, evidence)
    print(json.dumps(evidence.to_dict(), ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if evidence.safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
