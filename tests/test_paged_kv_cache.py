"""PagedKVCache tail rollback and page reuse tests."""

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paged_kv_cache import PagedKVCache


def _cache(page_size=4, max_pages=4):
    return PagedKVCache(
        page_size=page_size,
        max_pages=max_pages,
        device="cpu",
        dtype=torch.float32,
    )


def _kv(start, count, heads=2, dim=3):
    values = torch.arange(
        start,
        start + heads * count * dim,
        dtype=torch.float32,
    ).reshape(heads, count, dim)
    return values, values + 10000


def test_truncate_within_tail_page_preserves_prefix():
    cache = _cache()
    keys, values = _kv(0, 10)
    cache.append_kv(keys, values)

    assert cache.truncate(1) == 9
    actual_k, actual_v = cache.get_all_kv()

    assert torch.equal(actual_k, keys[:, :9, :])
    assert torch.equal(actual_v, values[:, :9, :])
    assert cache.allocated_page_count == 3
    assert cache.free_page_count == 0
    assert cache._current_page.used == 1
    assert len(cache.page_table) == cache.total_tokens


def test_truncate_releases_pages_and_append_reuses_them():
    cache = _cache()
    original_k, original_v = _kv(0, 10)
    cache.append_kv(original_k, original_v)

    assert cache.truncate(3) == 7
    assert cache.allocated_page_count == 2
    assert cache.free_page_count == 1
    assert cache._current_page.used == 3

    extra_k, extra_v = _kv(1000, 5)
    assert cache.append_kv(extra_k, extra_v) == 12
    actual_k, actual_v = cache.get_all_kv()

    assert torch.equal(actual_k, torch.cat((original_k[:, :7, :], extra_k), dim=1))
    assert torch.equal(actual_v, torch.cat((original_v[:, :7, :], extra_v), dim=1))
    assert cache.allocated_page_count == 3
    assert cache.free_page_count == 0
    assert len(cache.page_table) == cache.total_tokens


def test_truncate_exact_page_boundary_reuses_released_page():
    cache = _cache()
    keys, values = _kv(0, 12)
    cache.append_kv(keys, values)

    assert cache.truncate(4) == 8
    assert cache.allocated_page_count == 2
    assert cache.free_page_count == 1
    assert cache._current_page.used == cache.page_size

    extra_k, extra_v = _kv(500, 1)
    cache.append_kv_single(extra_k, extra_v)
    actual_k, actual_v = cache.get_all_kv()

    assert torch.equal(actual_k, torch.cat((keys[:, :8, :], extra_k), dim=1))
    assert torch.equal(actual_v, torch.cat((values[:, :8, :], extra_v), dim=1))
    assert cache.allocated_page_count == 3
    assert cache.free_page_count == 0


def test_truncate_all_keeps_typed_empty_shape_and_reusable_pages():
    cache = _cache()
    keys, values = _kv(0, 9)
    cache.append_kv(keys, values)

    assert cache.truncate(9) == 0
    empty_k, empty_v = cache.get_all_kv()

    assert empty_k.shape == (2, 0, 3)
    assert empty_v.shape == (2, 0, 3)
    assert cache.allocated_page_count == 0
    assert cache.free_page_count == 3
    assert cache._current_page is None
    assert cache.page_table == []


def test_truncate_validates_count_without_mutation():
    cache = _cache()
    keys, values = _kv(0, 3)
    cache.append_kv(keys, values)

    assert cache.truncate(0) == 3
    for invalid, error in ((-1, ValueError), (4, ValueError), (1.5, TypeError), (True, TypeError)):
        with pytest.raises(error):
            cache.truncate(invalid)

    actual_k, actual_v = cache.get_all_kv()
    assert torch.equal(actual_k, keys)
    assert torch.equal(actual_v, values)
    assert cache.total_tokens == 3


def test_repeated_truncate_and_append_does_not_leak_pages():
    cache = _cache(page_size=4, max_pages=3)
    keys, values = _kv(0, 12)

    for _ in range(20):
        cache.append_kv(keys, values)
        assert cache.allocated_page_count == 3
        assert cache.free_page_count == 0
        cache.truncate(12)
        assert cache.allocated_page_count == 0
        assert cache.free_page_count == 3
        assert cache.allocated_page_count + cache.free_page_count == 3

    stats = cache.get_stats()
    assert stats["truncate_call_count"] == 20
    assert stats["total_truncated_tokens"] == 240


def test_bulk_append_over_capacity_is_atomic():
    cache = _cache(page_size=4, max_pages=2)
    initial_k, initial_v = _kv(0, 3)
    cache.append_kv(initial_k, initial_v)
    oversized_k, oversized_v = _kv(100, 6)

    with pytest.raises(RuntimeError, match="超过缓存容量"):
        cache.append_kv(oversized_k, oversized_v)

    actual_k, actual_v = cache.get_all_kv()
    assert torch.equal(actual_k, initial_k)
    assert torch.equal(actual_v, initial_v)
    assert cache.total_tokens == 3
    assert len(cache.page_table) == 3
    assert cache.allocated_page_count == 1


def test_randomized_append_truncate_matches_reference_sequence():
    cache = _cache(page_size=4, max_pages=8)
    rng = random.Random(20260730)
    reference_k = torch.empty((2, 0, 3), dtype=torch.float32)
    reference_v = torch.empty((2, 0, 3), dtype=torch.float32)
    next_value = 0

    for _ in range(200):
        available = cache.page_size * cache.max_pages - cache.total_tokens
        should_truncate = cache.total_tokens and (
            available == 0 or rng.random() < 0.45
        )
        if should_truncate:
            count = rng.randint(0, cache.total_tokens)
            cache.truncate(count)
            if count:
                reference_k = reference_k[:, :-count, :]
                reference_v = reference_v[:, :-count, :]
        elif available:
            count = rng.randint(1, min(6, available))
            new_k, new_v = _kv(next_value, count)
            next_value += new_k.numel()
            cache.append_kv(new_k, new_v)
            reference_k = torch.cat((reference_k, new_k), dim=1)
            reference_v = torch.cat((reference_v, new_v), dim=1)

        actual_k, actual_v = cache.get_all_kv()
        assert torch.equal(actual_k, reference_k)
        assert torch.equal(actual_v, reference_v)
        assert cache.total_tokens == reference_k.shape[1]
        assert len(cache.page_table) == cache.total_tokens
        assert sum(page.used for page in cache.allocated_pages) == cache.total_tokens
        assert cache.allocated_page_count + cache.free_page_count <= cache.max_pages
        assert all(not page.is_free for page in cache.allocated_pages)
        assert all(page.is_free and page.used == 0 for page in cache.free_pages)
        assert cache._current_page == (
            cache.allocated_pages[-1] if cache.allocated_pages else None
        )
