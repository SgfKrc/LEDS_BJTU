"""KV 缓存生命周期宿主（inference-svc 进程内）。

§4.1 决策 2：KV 缓存永不跨进程传输——跨进程只传任务级引用
（task_id，即 past_key_values_ref）。本类按任务持有 PagedKVCache，
提供 init / free / get / status。

1.2 把 api_server._init_kv_cache 的设备画像自适应逻辑复制进来
（from_profile），当前提供默认参数构造。
"""
import threading
from typing import Any, Dict, Optional

from paged_kv_cache import PagedKVCache


class KVHost:
    """任务级 KV 缓存生命周期。"""

    def __init__(self):
        self._caches: Dict[str, PagedKVCache] = {}
        self._lock = threading.RLock()
        self._counter = 0

    def init(
        self,
        task_id: Optional[str] = None,
        device: Optional[str] = None,
        page_size: Optional[int] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """为任务创建 KV 缓存页；已存在则复用（幂等）。"""
        with self._lock:
            self._counter += 1
            tid = task_id or f"task_{self._counter}"
            existing = self._caches.get(tid)
            if existing is not None:
                return {
                    "task_id": tid,
                    "reused": True,
                    "total_pages": existing.allocated_page_count,
                }
            cache = PagedKVCache(
                page_size=page_size, max_pages=max_pages, device=device
            )
            self._caches[tid] = cache
            return {
                "task_id": tid,
                "reused": False,
                "total_pages": cache.allocated_page_count,
                "max_pages": cache.max_pages,
            }

    def free(self, task_id: str) -> Dict[str, Any]:
        """任务结束/取消时释放 KV 页。"""
        with self._lock:
            cache = self._caches.pop(task_id, None)
            if cache is None:
                raise KeyError(task_id)
            try:
                cache.clear()
            except Exception:
                pass
            return {"task_id": task_id, "freed": True}

    def from_profile(
        self,
        task_id: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Any = None,
        num_heads: int = 16,
        head_dim: int = 64,
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
                return {
                    "task_id": tid,
                    "reused": True,
                    "total_pages": existing.allocated_page_count,
                }
            cache = PagedKVCache.from_profile(
                profile=profile or {},
                device=device,
                dtype=dtype,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            self._caches[tid] = cache
            return {
                "task_id": tid,
                "reused": False,
                "total_pages": cache.allocated_page_count,
                "max_pages": cache.max_pages,
            }

    def get(self, task_id: str) -> Optional[PagedKVCache]:
        with self._lock:
            return self._caches.get(task_id)

    def status(self) -> Dict[str, Any]:
        """当前 KV 缓存快照（/v1/status.kv_cache 用）。"""
        with self._lock:
            return {
                "task_count": len(self._caches),
                "tasks": [
                    {
                        "task_id": tid,
                        "total_tokens": c.total_tokens,
                        "pages": c.allocated_page_count,
                        "max_pages": c.max_pages,
                    }
                    for tid, c in sorted(self._caches.items())
                ],
            }
