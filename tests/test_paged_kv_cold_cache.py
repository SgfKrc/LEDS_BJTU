import concurrent.futures
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paged_kv_cache import PagedKVCache
from inference_service.kv_host import KVHost


def _cache(tmp_path, *, max_pages=1, cold_max_pages=2):
    return PagedKVCache(
        page_size=4,
        max_pages=max_pages,
        device="cpu",
        dtype=torch.float32,
        cold_cache_dir=tmp_path,
        cold_max_pages=cold_max_pages,
    )


def _kv(start, count, heads=2, dim=3):
    values = torch.arange(start, start + heads * count * dim, dtype=torch.float32).reshape(heads, count, dim)
    return values, values + 10000


def test_cold_tier_spills_sealed_pages_and_reads_logical_order(tmp_path):
    cache = _cache(tmp_path)
    keys, values = _kv(0, 8)

    assert cache.append_kv(keys, values) == 8
    stats = cache.get_stats()
    assert stats["cold_enabled"] is True
    assert stats["allocated_pages"] == 1
    assert stats["cold_pages"] == 1
    assert stats["cold_eviction_count"] == 1
    assert stats["resident_peak_pages"] == 1
    assert stats["max_tokens"] == 12

    actual_k, actual_v = cache.get_all_kv()
    assert torch.equal(actual_k, keys)
    assert torch.equal(actual_v, values)
    assert cache.get_stats()["cold_hit_count"] == 1
    assert list(cache.cold_cache_dir.glob("page-*.pt"))


def test_cold_page_hash_mismatch_and_missing_file_fail_closed(tmp_path):
    cache = _cache(tmp_path)
    keys, values = _kv(0, 8)
    cache.append_kv(keys, values)
    cold_path = next(cache.cold_cache_dir.glob("page-*.pt"))

    cold_path.write_bytes(cold_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        cache.get_all_kv()
    assert cache.get_stats()["cold_miss_count"] == 1

    cold_path.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        cache.get_all_kv()
    assert cache.get_stats()["cold_miss_count"] == 2


def test_truncate_invalidates_cold_pages_and_hydrates_new_tail(tmp_path):
    cache = _cache(tmp_path)
    keys, values = _kv(0, 8)
    cache.append_kv(keys, values)

    assert cache.truncate(4) == 4
    stats = cache.get_stats()
    assert stats["cold_pages"] == 0
    assert stats["allocated_pages"] == 1
    assert stats["cold_hit_count"] == 1
    assert 0 < stats["cold_read_last_ms"] <= stats["cold_read_max_ms"]
    assert math.isfinite(stats["cold_read_max_ms"])
    assert stats["page_utilization"] == 1.0
    assert torch.equal(cache.get_all_kv()[0], keys[:, :4, :])

    extra_k, extra_v = _kv(500, 1)
    cache.append_kv_single(extra_k, extra_v)
    actual_k, actual_v = cache.get_all_kv()
    assert torch.equal(actual_k, torch.cat((keys[:, :4, :], extra_k), dim=1))
    assert torch.equal(actual_v, torch.cat((values[:, :4, :], extra_v), dim=1))


def test_cold_reads_are_serialized_and_do_not_corrupt_page_table(tmp_path):
    cache = _cache(tmp_path, max_pages=2, cold_max_pages=2)
    keys, values = _kv(0, 12)
    cache.append_kv(keys, values)

    def read_cache(_):
        return cache.get_all_kv()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(read_cache, range(8)))
    assert all(torch.equal(result[0], keys) and torch.equal(result[1], values) for result in results)
    assert cache.total_tokens == 12
    assert len(cache.page_table) == 12
    assert cache.get_stats()["cold_hit_count"] >= 8


def test_kv_host_exposes_opt_in_cold_tier_and_status(tmp_path):
    host = KVHost()
    result = host.init(
        task_id="cold-task",
        device="cpu",
        page_size=4,
        max_pages=1,
        cold_cache_dir=tmp_path,
        cold_max_pages=1,
    )
    assert result["cold_enabled"] is True
    assert result["cold_max_pages"] == 1

    cache = host.get("cold-task")
    keys, values = _kv(0, 8)
    cache.append_kv(keys, values)
    status = host.status()
    task = status["tasks"][0]
    assert task["cold_enabled"] is True
    assert task["cold_pages"] == 1
    assert task["cold_hit_count"] == 0
    assert task["cache_unit_size"] == 4


def test_append_single_respects_logical_cold_capacity(tmp_path):
    cache = _cache(tmp_path, max_pages=1, cold_max_pages=1)
    key, value = _kv(0, 1)
    for _ in range(cache.page_size * 2):
        cache.append_kv_single(key, value)
    with pytest.raises(RuntimeError, match="缓存容量"):
        cache.append_kv_single(key, value)
