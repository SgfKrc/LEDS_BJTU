#!/usr/bin/env python3
"""Deterministic CACHE-04 unit-boundary and prefix-reuse experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cache_unit_layout import (  # noqa: E402
    CacheUnitLayout,
    build_cache_units,
    match_cached_prefix,
    raw_common_prefix_tokens,
)


def build_report() -> dict[str, object]:
    page_size = 8
    unit_size = 4
    layout = CacheUnitLayout(page_size=page_size, unit_size=unit_size)
    cached_tokens = [101, 102, 103, 104, 105, 106, 107, 108]
    cases = {
        "aligned_extension": cached_tokens + [201, 202, 203, 204],
        "aligned_change": [101, 102, 103, 104, 999, 106, 107, 108],
        "unaligned_change": [101, 102, 999, 104, 105, 106, 107, 108],
    }
    cached_units = build_cache_units(cached_tokens, unit_size)
    measurements = []
    for name, request_tokens in cases.items():
        match = match_cached_prefix(cached_units, request_tokens, unit_size)
        measurements.append(
            {
                "name": name,
                "request_tokens": len(request_tokens),
                "raw_common_tokens": raw_common_prefix_tokens(cached_tokens, request_tokens),
                "matched_tokens": match.matched_tokens,
                "matched_units": match.matched_units,
                "request_unit_count": match.request_unit_count,
            }
        )
    return {
        "schema": "qlh.cache_unit_experiment.v1",
        "layout": {
            "page_size": page_size,
            "unit_size": unit_size,
            "units_per_page": layout.units_per_page,
            "unit_boundaries_12": list(layout.unit_boundaries(12)),
            "page_boundaries_12": list(layout.page_boundaries(12)),
        },
        "cached_prefix": {
            "token_count": len(cached_tokens),
            "complete_unit_count": len(cached_units),
        },
        "measurements": measurements,
        "interpretation": {
            "aligned_extension_hits": 8,
            "aligned_change_hits": 4,
            "unaligned_change_hits": 0,
        },
    }


def main() -> int:
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
