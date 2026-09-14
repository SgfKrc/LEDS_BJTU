"""Deterministic fixed-token cache units and prefix matching helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CacheUnitLayout:
    """Fixed token units whose boundaries are compatible with KV pages."""

    page_size: int
    unit_size: int

    def __post_init__(self) -> None:
        _positive_int(self.page_size, "page_size")
        _positive_int(self.unit_size, "unit_size")
        if self.page_size % self.unit_size:
            raise ValueError("unit_size must divide page_size for aligned cache units")

    @property
    def units_per_page(self) -> int:
        return self.page_size // self.unit_size

    def complete_token_count(self, total_tokens: int) -> int:
        total_tokens = _nonnegative_int(total_tokens, "total_tokens")
        return total_tokens - (total_tokens % self.unit_size)

    def unit_ranges(self, total_tokens: int) -> tuple[tuple[int, int], ...]:
        complete = self.complete_token_count(total_tokens)
        return tuple(
            (start, start + self.unit_size)
            for start in range(0, complete, self.unit_size)
        )

    def unit_boundaries(self, total_tokens: int) -> tuple[int, ...]:
        complete = self.complete_token_count(total_tokens)
        return tuple(range(0, complete + 1, self.unit_size))

    def page_boundaries(self, total_tokens: int) -> tuple[int, ...]:
        """Return complete physical page boundaries in the aligned prefix."""
        complete = self.complete_token_count(total_tokens)
        return tuple(range(0, complete + 1, self.page_size))


@dataclass(frozen=True)
class CacheUnit:
    index: int
    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class CachePrefixMatch:
    matched_tokens: int
    matched_units: int
    request_unit_count: int


def _token_digest(tokens: Sequence[int]) -> str:
    canonical = json.dumps(
        [int(token) for token in tokens], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_cache_units(tokens: Sequence[int], unit_size: int) -> tuple[CacheUnit, ...]:
    """Build complete, content-addressed cache units; ignore an incomplete tail."""
    unit_size = _positive_int(unit_size, "unit_size")
    units = []
    complete = len(tokens) - (len(tokens) % unit_size)
    for index, start in enumerate(range(0, complete, unit_size)):
        end = start + unit_size
        units.append(
            CacheUnit(
                index=index,
                start=start,
                end=end,
                sha256=_token_digest(tokens[start:end]),
            )
        )
    return tuple(units)


def raw_common_prefix_tokens(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def aligned_common_prefix_tokens(
    left: Sequence[int], right: Sequence[int], unit_size: int
) -> int:
    """Return the common prefix that ends on a complete cache-unit boundary."""
    unit_size = _positive_int(unit_size, "unit_size")
    raw = raw_common_prefix_tokens(left, right)
    return raw - (raw % unit_size)


def match_cached_prefix(
    cached_units: Sequence[CacheUnit],
    request_tokens: Sequence[int],
    unit_size: int,
) -> CachePrefixMatch:
    """Match only complete, unchanged units from a cached request prefix."""
    unit_size = _positive_int(unit_size, "unit_size")
    request_units = build_cache_units(request_tokens, unit_size)
    matched_units = 0
    for cached, requested in zip(cached_units, request_units):
        if (
            cached.index != requested.index
            or cached.start != requested.start
            or cached.end != requested.end
            or cached.sha256 != requested.sha256
        ):
            break
        matched_units += 1
    return CachePrefixMatch(
        matched_tokens=matched_units * unit_size,
        matched_units=matched_units,
        request_unit_count=len(request_units),
    )
