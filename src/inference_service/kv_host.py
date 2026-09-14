"""KV 缓存生命周期宿主（inference-svc 进程内）。

§4.1 决策 2：KV 缓存永不跨进程传输——跨进程只传任务级引用
（task_id，即 past_key_values_ref）。本类按任务持有 PagedKVCache，
提供 init / free / get / status。

1.2 把 api_server._init_kv_cache 的设备画像自适应逻辑复制进来
（from_profile），当前提供默认参数构造。
"""
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import KV_COLD_CACHE_ROOT, KV_COLD_MAX_BYTES
from paged_kv_cache import PagedKVCache


class KVCapacityError(RuntimeError):
    """The service-wide cold-tier byte quota cannot admit another page."""


class KVCleanupError(RuntimeError):
    """A task cache could not be released and remains available for retry."""

    def __init__(self, task_id: str, cause: Exception):
        self.task_id = task_id
        super().__init__(f"KV cleanup failed for task {task_id}; retry is available: {cause}")


class KVHost:
    """任务级 KV 缓存生命周期。"""

    def __init__(
        self,
        *,
        cold_cache_root: str | os.PathLike[str] | None = None,
        cold_max_bytes: int | None = None,
    ):
        self._caches: Dict[str, PagedKVCache] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self.cold_cache_root = Path(cold_cache_root or KV_COLD_CACHE_ROOT).expanduser().resolve(strict=False)
        self.cold_max_bytes = KV_COLD_MAX_BYTES if cold_max_bytes is None else cold_max_bytes
        if isinstance(self.cold_max_bytes, bool) or not isinstance(self.cold_max_bytes, int) or self.cold_max_bytes <= 0:
            raise ValueError("cold_max_bytes must be a positive integer")
        self._cold_bytes_used = self._scan_cold_bytes()
        self._cold_bytes_reserved = 0

    def _scan_cold_bytes(self) -> int:
        if not self.cold_cache_root.is_dir():
            return 0
        total = 0
        try:
            for path in self.cold_cache_root.rglob("page-*.pt"):
                if path.is_file():
                    total += path.stat().st_size
        except OSError as exc:
            raise RuntimeError("cannot inspect the server-managed KV cold-tier root") from exc
        return total

    def _reserve_cold_bytes(self, amount: int) -> None:
        with self._lock:
            if amount < 0 or self._cold_bytes_used + self._cold_bytes_reserved + amount > self.cold_max_bytes:
                raise KVCapacityError(
                    f"KV cold-tier byte quota exceeded ({self.cold_max_bytes} bytes)"
                )
            self._cold_bytes_reserved += amount

    def _commit_cold_bytes(self, amount: int) -> None:
        with self._lock:
            self._cold_bytes_reserved = max(0, self._cold_bytes_reserved - amount)
            self._cold_bytes_used += amount

    def _release_cold_bytes(self, amount: int) -> None:
        with self._lock:
            self._cold_bytes_used = max(0, self._cold_bytes_used - amount)

    def _release_cold_reservation(self, amount: int) -> None:
        with self._lock:
            self._cold_bytes_reserved = max(0, self._cold_bytes_reserved - amount)

    def _cold_dir_for_request(
        self,
        cold_cache_dir: Optional[str],
        cold_max_pages: Optional[int],
    ) -> Optional[Path]:
        if cold_cache_dir is not None:
            raise ValueError("cold_cache_dir is server-managed and cannot be set by the caller")
        return self.cold_cache_root if cold_max_pages not in (None, 0) else None

    def init(
        self,
        task_id: Optional[str] = None,
        device: Optional[str] = None,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
        cold_cache_dir: Optional[str] = None,
        cold_max_pages: Optional[int] = None,
        cache_unit_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """为任务创建 KV 缓存页；已存在则复用（幂等）。"""
        with self._lock:
            self._counter += 1
            tid = task_id or f"task_{self._counter}"
            existing = self._caches.get(tid)
            if existing is not None:
                stats = existing.get_stats()
                return {
                    "task_id": tid,
                    "reused": True,
                    "total_pages": existing.allocated_page_count,
                    "max_pages": existing.max_pages,
                    "cold_enabled": stats["cold_enabled"],
                    "cold_max_pages": stats["cold_max_pages"],
                    "cache_unit_size": stats["cache_unit_size"],
                }
            cache = PagedKVCache(
                page_size=page_size,
                max_pages=max_pages,
                device=device,
                cold_cache_dir=self._cold_dir_for_request(cold_cache_dir, cold_max_pages),
                cold_max_pages=cold_max_pages,
                cache_unit_size=cache_unit_size,
                cold_bytes_reserve=self._reserve_cold_bytes,
                cold_bytes_commit=self._commit_cold_bytes,
                cold_bytes_release=self._release_cold_bytes,
                cold_bytes_release_reservation=self._release_cold_reservation,
            )
            self._caches[tid] = cache
            stats = cache.get_stats()
            return {
                "task_id": tid,
                "reused": False,
                "total_pages": cache.allocated_page_count,
                "max_pages": cache.max_pages,
                "cold_enabled": stats["cold_enabled"],
                "cold_max_pages": stats["cold_max_pages"],
                "cache_unit_size": stats["cache_unit_size"],
            }

    def free(self, task_id: str) -> Dict[str, Any]:
        """任务结束/取消时释放 KV 页。"""
        with self._lock:
            cache = self._caches.get(task_id)
            if cache is None:
                raise KeyError(task_id)
            try:
                cache.close()
            except Exception as exc:
                raise KVCleanupError(task_id, exc) from exc
            self._caches.pop(task_id, None)
            return {"task_id": task_id, "freed": True}

    def from_profile(
        self,
        task_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Any = None,
        num_heads: int = 16,
        head_dim: int = 64,
        cold_cache_dir: Optional[str] = None,
        cold_max_pages: Optional[int] = None,
        cache_unit_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """按设备画像自适应初始化 KV 缓存（1.2a 复制自 api_server._init_kv_cache
        api_server.py:1575-1618 的自适应逻辑；源文件保持不动）。

        与 init() 的差异：page_size/max_pages 由 PagedKVCache.from_profile
        按设备档位决定，而非调用方显式指定。num_heads/head_dim 由宿主
        （EngineHost）从已加载模型的 config 读取后传入。
        """
        import torch as _torch

        if dtype is None:
            dtype = _torch.float16
        with self._lock:
            self._counter += 1
            tid = task_id or f"task_{self._counter}"
            existing = self._caches.get(tid)
            if existing is not None:
                stats = existing.get_stats()
                return {
                    "task_id": tid,
                    "reused": True,
                    "total_pages": existing.allocated_page_count,
                    "max_pages": existing.max_pages,
                    "cold_enabled": stats["cold_enabled"],
                    "cold_max_pages": stats["cold_max_pages"],
                    "cache_unit_size": stats["cache_unit_size"],
                }
            cache = PagedKVCache.from_profile(
                profile=profile or {},
                device=device,
                dtype=dtype,
                num_heads=num_heads,
                head_dim=head_dim,
                cold_cache_dir=self._cold_dir_for_request(cold_cache_dir, cold_max_pages),
                cold_max_pages=cold_max_pages,
                cache_unit_size=cache_unit_size,
                cold_bytes_reserve=self._reserve_cold_bytes,
                cold_bytes_commit=self._commit_cold_bytes,
                cold_bytes_release=self._release_cold_bytes,
                cold_bytes_release_reservation=self._release_cold_reservation,
            )
            self._caches[tid] = cache
            stats = cache.get_stats()
            return {
                "task_id": tid,
                "reused": False,
                "total_pages": cache.allocated_page_count,
                "max_pages": cache.max_pages,
                "cold_enabled": stats["cold_enabled"],
                "cold_max_pages": stats["cold_max_pages"],
                "cache_unit_size": stats["cache_unit_size"],
            }

    def get(self, task_id: str) -> Optional[PagedKVCache]:
        with self._lock:
            return self._caches.get(task_id)

    def status(self) -> Dict[str, Any]:
        """当前 KV 缓存快照（/v1/status.kv_cache 用）。"""
        with self._lock:
            tasks = []
            for tid, cache in sorted(self._caches.items()):
                stats = cache.get_stats()
                tasks.append(
                    {
                        "task_id": tid,
                        "total_tokens": cache.total_tokens,
                        "pages": cache.allocated_page_count,
                        "max_pages": cache.max_pages,
                        "cold_enabled": stats["cold_enabled"],
                        "cold_pages": stats["cold_pages"],
                        "cold_hit_count": stats["cold_hit_count"],
                        "cache_unit_size": stats["cache_unit_size"],
                        "cache_unit_count": stats["cache_unit_count"],
                    }
                )
            return {
                "task_count": len(self._caches),
                "cold_max_bytes": self.cold_max_bytes,
                "cold_used_bytes": self._cold_bytes_used,
                "cold_reserved_bytes": self._cold_bytes_reserved,
                "tasks": tasks,
            }
